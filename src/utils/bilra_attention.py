from __future__ import annotations

import torch
from torch import nn


class BiLevelRoutingAttention(nn.Module):
    """
    Paramters:
    dim: token channel dimension
    num_heads: attention heads
    n_regions: regions per side (s); the map is divided into n_regions**2
    top_k: how many regions each region routes to

    input and output are (B, C, H, W). H and W are divisible by n_regions.
    """

    def __init__(
        self, dim: int, num_heads: int = 4, n_regions: int = 4, top_k: int = 4
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dimension {dim} not divisible by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.n_regions = n_regions
        self.top_k = top_k
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.lce = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.n_regions

        if h % s != 0 or w % s != 0:
            raise ValueError(
                f"H={h}, W={w}, and n_regions={s} are not compatible. H and W must be divisible by n_regions."
            )

        topk = min(self.top_k, s * s)

        rh, rw = h // s, w // s  # region height, width in tokens
        n_reg = s * s
        tokens_per_reg = rh * rw

        t = x.view(b, c, s, rh, s, rw)
        t = t.permute(0, 2, 4, 1, 3, 5).reshape(b, n_reg, tokens_per_reg, c)

        q = self.q(t)
        k = self.k(t)
        v = self.v(t)

        q_reg = q.mean(dim=2)  # (B, n_reg, c)
        k_reg = k.mean(dim=2)  # (B, n_reg, c)
        affinity = torch.matmul(q_reg, k_reg.transpose(-2, -1))  # (B, n_reg, n_reg)
        routing = affinity.topk(topk, dim=-1).indices  # (B, n_reg, topk)

        def gather_routed(src: torch.Tensor) -> torch.Tensor:
            """
            Gather the tokens of the routed regions for keys and values.
            src: (B, n_reg, tpr, C)
            routing: (B, n_reg, topk)
            """
            idx = routing.view(b, n_reg, topk, 1, 1).expand(
                b, n_reg, topk, tokens_per_reg, c
            )
            src_exp = src.unsqueeze(1).expand(b, n_reg, n_reg, tokens_per_reg, c)
            gathered = torch.gather(src_exp, dim=2, index=idx)

            return gathered.reshape(b, n_reg, topk * tokens_per_reg, c)

        k_routed = gather_routed(k)
        v_routed = gather_routed(v)

        # multihead attention: each region's tokens attend to routed tokens

        def split_heads(z: torch.Tensor) -> torch.Tensor:
            """
            (B, n_reg, T, C) -> (B, n_reg, heads, T, head_dim)
            """
            bb, nr, tt, _cc = z.shape
            return z.view(bb, nr, tt, self.num_heads, self.head_dim).permute(
                0, 1, 3, 2, 4
            )

        qh = split_heads(q)  # (B, n_reg, heads, tpr, hd)
        kh = split_heads(k_routed)  # (B, n_reg, heads, tpr*k, hd)
        vh = split_heads(v_routed)

        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, vh)  # (B, n_reg, heads, tpr, hd)

        # merge heads and un-partition back to (b, c, h, w)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, n_reg, tokens_per_reg, c)
        out = self.proj(out)
        # Regions were flattened in (region_row, region_col, token_row,
        # token_col) order above.  Restore those axes, interleave each region
        # coordinate with its within-region coordinate, and finally collapse
        # them back to the original spatial map.
        out = (
            out.view(b, s, s, rh, rw, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, h, w)
        )

        v_spatial = (
            v.view(b, s, s, rh, rw, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        )
        return out + self.lce(v_spatial)
