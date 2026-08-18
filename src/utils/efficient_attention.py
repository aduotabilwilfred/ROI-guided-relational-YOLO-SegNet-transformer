from __future__ import annotations

import torch
from torch import nn


class EfficientSelffAtteniton(nn.Module):
    """
    Reduces the number of key/value token by a factor r before attention

    Input and Output are expected to be (B, C, H, W)
    """

    def __init__(self, dim: int, num_heads: int = 4, reduction: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.reduction = reduction

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=reduction, stride=reduction)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        n = tokens.shape[1]

        q = (
            self.q(tokens)
            .reshape(b, n, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.reduction > 1:
            reduced = self.sr(x).flatten(2).transpose(1, 2)  # (B, N', C)
            reduced = self.norm(reduced)
        else:
            reduced = tokens
        kv = (
            self.kv(reduced)
            .reshape(b, -1, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out.transpose(1, 2).reshape(b, c, h, w)
