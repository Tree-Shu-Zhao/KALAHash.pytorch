import torch
import torch.nn as nn

from .clora import MultiheadAttentionCLoRA


class ImageEncoder(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()

        self.conv1 = clip_visual.conv1
        self.class_embedding = clip_visual.class_embedding
        self.positional_embedding = clip_visual.positional_embedding
        self.ln_pre = clip_visual.ln_pre
        self.transformer = clip_visual.transformer
        self.ln_post = clip_visual.ln_post
        self.proj = clip_visual.proj
    
    def forward(self, x: torch.Tensor, knowledge_pool: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        # Inject knowledge
        for layer_id in range(self.transformer.layers):
            resblock = self.transformer.resblocks[layer_id]
            if isinstance(resblock.attn, MultiheadAttentionCLoRA):
                resblock.attn_mask = resblock.attn_mask.to(
                    dtype=x.dtype,
                    device=x.device) if resblock.attn_mask is not None else None

                res = x
                x = resblock.ln_1(x)
                x = resblock.attn(x, x, x, knowledge_pool, need_weights=False,
                                                              attn_mask=resblock.attn_mask)[0]
                x = res + x
            else:
                x = x + resblock.attention(resblock.ln_1(x))
            x = x + resblock.mlp(resblock.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
