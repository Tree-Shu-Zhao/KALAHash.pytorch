import os
import os.path as osp

import torch
import torch.nn as nn
from loguru import logger

from clip import clip

from .image_encoder import ImageEncoder
from .clora import apply_clora


class KalaHash(nn.Module):
    def __init__(self, cfg, classnames):
        super().__init__()
        self.cfg = cfg

        # Load CLIP model
        clip_model = self._load_clip_to_cpu(cfg.CLIP_BACKBONE)
        clip_model.eval()
        self.dtype = clip_model.dtype

        # Prepare Knowledge K
        prompt_template = "a photo of a {}."
        static_prompts = [prompt_template.format(name) for name in classnames]
        static_prompt_embeddings = torch.cat([clip.tokenize(p) for p in static_prompts])
        with torch.no_grad():
            textual_knowledge = clip_model.encode_text(static_prompt_embeddings)
            self.register_buffer("textual_knowledge", textual_knowledge)

        # Apply CLoRA
        self.enable_lora = cfg.lora.ENABLE
        if self.enable_lora:
            lora_type = cfg.lora.TYPE.lower()
            if lora_type == "clora":
                apply_clora(cfg.lora, clip_model)
            else:
                raise ValueError(f"Unknown lora type: {lora_type}")
            self.module_F = nn.Linear(cfg.HIDDEN_DIM, cfg.lora.HIDDEN_DIM).half()
            self.image_encoder = ImageEncoder(clip_model.visual)
        else:
            self.image_encoder = clip_model.visual

        # Build Hash Layer and Module G
        criterion_name = cfg.CRITERION_NAME.lower()
        if criterion_name in ("greedy", "csq", "ortho", "mdsh"):
            # For these criterions, they cannot be optimized if we do not use BN
            self.img_hash_head = nn.Sequential(
                nn.Linear(cfg.HIDDEN_DIM, cfg.hash_head.BITS),
                nn.BatchNorm1d(cfg.hash_head.BITS, momentum=cfg.BN_MOMENTUM),
                nn.Tanh(),
            ).half()
            self.module_G = nn.Sequential(
                nn.Linear(cfg.HIDDEN_DIM, cfg.hash_head.BITS),
                nn.BatchNorm1d(cfg.hash_head.BITS, momentum=cfg.BN_MOMENTUM),
            ).half()
        else:
            self.img_hash_head = nn.Sequential(
                nn.Linear(cfg.HIDDEN_DIM, cfg.hash_head.BITS),
                nn.Tanh(),
            ).half()
            self.module_G = nn.Sequential(
                nn.Linear(cfg.HIDDEN_DIM, cfg.hash_head.BITS),
            ).half()
        nn.init.normal_(self.img_hash_head[0].weight, std=0.01)
        nn.init.normal_(self.module_G[0].weight, std=0.01)

        self._models = {}
        self.register_learnable_module("img_hash_head", self.img_hash_head)
        self.register_learnable_module("module_G", self.module_G)

        # Do not register lora, we have not implement the merge function now
        # We manually set its learning mode as trainable in self.get_learnable_params
        # self.register_learnable_module("module_F", self.module_F)

    def forward(self, batch_data):
        images = batch_data["images"]

        textual_knowledge = self.module_G(self.textual_knowledge)

        if self.enable_lora:
            knowlegde_pool = self.module_F(self.textual_knowledge)
            image_features = self.image_encoder(images.type(self.dtype), knowlegde_pool)
        else:
            image_features = self.image_encoder(images.type(self.dtype))
        image_hash_features = self.img_hash_head(image_features)

        return {
            "image_features": image_features,
            "image_hash_features": image_hash_features,
            "textual_knowledge": textual_knowledge,
        }

    def get_learnable_params(self):
        learnable_param_names = self._models.keys()

        def check_learnable_params(name):
            for n in learnable_param_names:
                if n in name:
                    # Registered learnable modules
                    return True
            if 'lora_' in name:
                return True
            if "module_F" in name:
                return True
            if "text_encoder" in name:
                return False
            return False

        learnable_param_name_list = []
        learnable_params = []
        for name, param in self.named_parameters():
            if check_learnable_params(name):
                param.requires_grad_(True)
                learnable_param_name_list.append(name)
                learnable_params.append(param)
            else:
                param.requires_grad_(False)
        logger.info(f"learnable params: {learnable_param_name_list}")

        return list(set(learnable_params))

    def set_model_mode(self, mode):
        for name in self._models.keys():
            if mode == "train":
                self._models[name].train()
            elif mode == "eval":
                self._models[name].eval()
            else:
                raise KeyError

    def register_learnable_module(self, name, model):
        self._models[name] = model

    def _load_clip_to_cpu(self, backbone_name):
        from hydra.utils import to_absolute_path

        url = clip._MODELS[backbone_name]
        cache_dir = osp.join(to_absolute_path("data"), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = clip._download(url, cache_dir)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())

        return model
