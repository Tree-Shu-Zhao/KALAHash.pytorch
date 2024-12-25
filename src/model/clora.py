import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loralib import LoRALayer, PlainMultiheadAttentionLoRA


def apply_clora(cfg, clip_model):
    list_lora_layers = []
    vision_encoder = clip_model.visual.transformer
    for i, block in enumerate(vision_encoder.resblocks):
        if i >= cfg.LORA_START_LAYERS:
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    new_multi_head_lora = MultiheadAttentionCLoRA(
                        submodule, 
                        enable_lora=cfg.LORA_POSITION, 
                        r=cfg.RANK, 
                        lora_alpha=cfg.LORA_STRENGTH, 
                        dropout_rate=0.0, # We disable dropout
                    ).half()
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


class MultiheadAttentionCLoRA(PlainMultiheadAttentionLoRA):
    def __init__(self, submodule, enable_lora, r, lora_alpha, dropout_rate):
        super().__init__(submodule, enable_lora, r, lora_alpha, dropout_rate)
        self.enable_lora = enable_lora

        # Init qkv as a new lora linear layer 
        for item in enable_lora:
            if item == 'q':
                self.q_proj = LinearCLoRA(
                    self.q_proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate
                )
            elif item == 'k':
                self.k_proj = LinearCLoRA(
                    self.k_proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate,
                )
            elif item == 'v':
                self.v_proj = LinearCLoRA(
                    self.v_proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate,
                )
    
    def forward_module(
            self,
            query,
            key,
            value,
            knowledge_pool,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        """
        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        """
        
        if "q" in self.enable_lora:
            q = self.q_proj(query, knowledge_pool=knowledge_pool)
        else:
            q = self.q_proj(query)
        if "k" in self.enable_lora:
            k = self.k_proj(key, knowledge_pool=knowledge_pool)
        else:
            k = self.k_proj(key)
        if "v" in self.enable_lora:
            v = self.v_proj(value, knowledge_pool=knowledge_pool)
        else:
            v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        if "o" in self.enable_lora:
            attn_output = self.proj(attn_output, knowledge_pool=knowledge_pool)
        else:
            attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None  

    def train(self, mode: bool = True):
        super().train(mode)
        #self.lora_train(mode)  

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            knowledge_pool,
            **kwargs):
        

        return self.forward_module(query, key, value, knowledge_pool, **kwargs) 

class LinearCLoRA(nn.Linear, LoRALayer):
     # LoRA implemented in a Linear layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 0, 
        lora_alpha: int = 1, 
        fan_in_fan_out: bool = False,
        dropout_rate = 0.,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        
        self.load_state_dict(existing_linear.state_dict(), strict=False)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                )
            eval(f'self.{param_name}').requires_grad = False
    
    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))

    def train(self, mode: bool = True):
        super().train(mode)     
        self.lora_train(mode)

        
    def forward(self, x: torch.Tensor, **kwargs):
        
        # Compute the original linear transformation
        original_output = nn.Linear.forward(self, x)

        # Disable Dropout
        # if self.training and self.dropout.p > 0:
            # x = self.dropout(x)
        
        if self.r > 0 and not self.merged:
            lora_weights = self.merge_BA(x, kwargs["knowledge_pool"])
            lora_adjustment = torch.matmul(x.transpose(0, 1), lora_weights.to(x.dtype)).transpose(0, 1) * self.scaling 
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result
    
    def merge_BA(self, query, knowledge_pool):
        # Normalize query and language features
        query = F.normalize(query.detach().clone()[1:].mean(dim=0), p=2, dim=1)
        lora_pool = F.normalize(knowledge_pool, p=2, dim=1)

        # Calculate similarity and get top-k indices
        similarity = torch.matmul(query, lora_pool.t())
        _, top_k_indices = torch.topk(similarity, self.r, dim=1)

        # Use advanced indexing for faster selection
        w_lora_B = knowledge_pool[top_k_indices.view(-1)].view(query.size(0), self.r, -1)

        # Prepare w_lora_A for batch multiplication
        w_lora_A = self.w_lora_A.view(self.r, -1, 1).expand(-1, -1, query.size(0)).permute(2, 1, 0)

        # Perform batch matrix multiplication
        lora_weights = torch.bmm(w_lora_A, w_lora_B)

        return self.transpose(lora_weights)
