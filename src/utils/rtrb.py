from __future__ import annotations

import torch
from torch import nn


class RelationalTransformerBlock(nn.Module):
    """Dual-head self/cross attention over ROI features.

    Parameters

    in_channels : channels of the input feature maps f_m and f_w.
    embed_dim : per-head embedding width; each head outputs embed_dim channels,
        so the block outputs 2 * embed_dim.
    num_heads : attention heads; embed_dim must be divisible by it.

    forward(f_m, f_w) where both are (B, in_channels, H, W); f_w may have a
    different spatial size than f_m (surrounding region), which cross-attention
    handles naturally since queries and keys need not share token counts.
    Returns (B, 2 * embed_dim, H, W).
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, num_heads: int = 8):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Q/K/V generators: 3x3 convs (paper). Self head reads f_m for all
        # three; cross head reads f_m for query, f_w for key and value
        def qkv_conv():
            return nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.q_sa, self.k_sa, self.v_sa = qkv_conv(), qkv_conv(), qkv_conv()
        self.q_ca, self.k_ca, self.v_ca = qkv_conv(), qkv_conv(), qkv_conv()

        # w_k residual embeddings (Eq. 19): 1x1 convs, one per attention branch
        # (self/cross). They map the attended embed_dim features to in_channels
        # so the residual add to f_m is shape-compatible.

        self.w_sa = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.w_ca = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # After residual, each head is back at in_channels; project each to
        # embed_dim so the channel-concat output is a clean 2 * embed_dim.
        self.out_sa = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.out_ca = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def _to_tokens(self, feat: torch.Tensor) -> torch.Tensor:

        # (B, embed, H, W) -> (B, heads, H*W, head_dim)
        b, _, h, w = feat.shape
        return feat.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q,k,v: (B, heads, N, head_dim). Standard scaled dot-product

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, v)

    def _merge(self, attended: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # (B, heads, N, head_dim) -> (B, embed, H, W)
        b = attended.shape[0]
        x = attended.permute(0, 1, 3, 2).reshape(b, self.embed_dim, h, w)
        return x

    def forward(self, f_m: torch.Tensor, f_w: torch.Tensor) -> torch.Tensor:
        _b, _, h, w = f_m.shape

        # Self-attention head over f_m
        q_s = self._to_tokens(self.q_sa(f_m))
        k_s = self._to_tokens(self.k_sa(f_m))
        v_s = self._to_tokens(self.v_sa(f_m))
        g_sa = self._merge(self._attention(q_s, k_s, v_s), h, w)

        # Cross-attention head: f_m queries, f_w keys/values
        q_c = self._to_tokens(self.q_ca(f_m))
        k_c = self._to_tokens(self.k_ca(f_w))
        v_c = self._to_tokens(self.v_ca(f_w))
        g_ca = self._merge(self._attention(q_c, k_c, v_c), h, w)

        # residual learning: w_k is 1x1 conv, residual add to f_m
        f_sa = self.w_sa(g_sa) + f_m
        f_ca = self.w_ca(g_ca) + f_m

        out = torch.cat([self.out_sa(f_sa), self.out_ca(f_ca)], dim=1)
        return out
