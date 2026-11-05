"""BiFormer bi-level routing attention (BiLRA), adopted by the paper.

Region-level top-k routing followed by fine attention within routed regions.
Matches the paper's Eqs. 4-6. Faithful to Zhu et al. (BiFormer)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BiLevelRoutingAttention(nn.Module):
    """Bi-level routing attention from BiFormer (adopted by the paper).

    Standard full attention is quadratic in the number of tokens. BiLRA avoids
    that by working in two levels. The feature map is partitioned into s x s
    regions. A coarse routing step computes region-to-region affinity from
    region-averaged queries and keys, and keeps only the top-k most relevant
    regions for each region. Fine attention is then applied from each token to
    the tokens of its region's top-k routed regions only, rather than to every
    token in the image. This keeps long-range reach while bounding cost.

    This follows the mechanism in Zhu et al. (BiFormer) and matches the paper's
    Eqs. 4-6 (Eq. 5's topIndex is the routing, Eq. 6 the gathered attention).

    Parameters
    ----------
    dim : token channel dimension.
    num_heads : attention heads.
    n_regions : regions per side (s); the map is divided into n_regions**2.
    topk : how many regions each region routes to.

    Input and output are (B, C, H, W). H and W must be divisible by n_regions.
    """

    def __init__(self, dim: int, num_heads: int = 4, n_regions: int = 4,
                 topk: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.n_regions = n_regions
        self.topk = topk

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        # Local context enhancement (Eq. 6, lce): a depth-wise convolution on
        # the value, added to the attention output. Depth-wise = groups equal
        # to channels, so each channel is convolved independently.
        self.lce = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.n_regions
        if h % s != 0 or w % s != 0:
            raise ValueError(f"H={h}, W={w} must be divisible by n_regions={s}")
        topk = min(self.topk, s * s)

        rh, rw = h // s, w // s          # region height, width in tokens
        n_reg = s * s
        tokens_per_reg = rh * rw

        # Tokens grouped by region: (B, n_reg, tokens_per_reg, C)
        t = x.view(b, c, s, rh, s, rw)
        t = t.permute(0, 2, 4, 3, 5, 1).reshape(b, n_reg, tokens_per_reg, c)

        q = self.q(t)                    # (B, n_reg, tpr, C)
        k = self.k(t)
        v = self.v(t)

        # Region-level routing: affinity between region-mean queries and keys.
        q_reg = q.mean(dim=2)            # (B, n_reg, C)
        k_reg = k.mean(dim=2)
        affinity = torch.matmul(q_reg, k_reg.transpose(-2, -1))  # (B, n_reg, n_reg)
        routing = affinity.topk(topk, dim=-1).indices            # (B, n_reg, topk)

        # Gather the tokens of the routed regions for keys and values.
        # k, v: (B, n_reg, tpr, C) -> gather along region axis per query region.
        def gather_routed(src: torch.Tensor) -> torch.Tensor:
            # src (B, n_reg, tpr, C); routing (B, n_reg, topk)
            idx = routing.view(b, n_reg, topk, 1, 1).expand(
                b, n_reg, topk, tokens_per_reg, c)
            src_exp = src.unsqueeze(1).expand(b, n_reg, n_reg, tokens_per_reg, c)
            gathered = torch.gather(src_exp, 2, idx)     # (B,n_reg,topk,tpr,C)
            return gathered.reshape(b, n_reg, topk * tokens_per_reg, c)

        k_routed = gather_routed(k)      # (B, n_reg, topk*tpr, C)
        v_routed = gather_routed(v)

        # Multi-head attention: each region's tokens attend to routed tokens.
        def split_heads(z: torch.Tensor) -> torch.Tensor:
            # (B, n_reg, T, C) -> (B, n_reg, heads, T, head_dim)
            bb, nr, tt, cc = z.shape
            return z.view(bb, nr, tt, self.num_heads, self.head_dim
                          ).permute(0, 1, 3, 2, 4)

        qh = split_heads(q)              # (B,n_reg,heads,tpr,hd)
        kh = split_heads(k_routed)       # (B,n_reg,heads,topk*tpr,hd)
        vh = split_heads(v_routed)

        attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, vh)     # (B,n_reg,heads,tpr,hd)

        # Merge heads and un-partition attention output back to (B, C, H, W).
        out = out.permute(0, 1, 3, 2, 4).reshape(b, n_reg, tokens_per_reg, c)
        out = self.proj(out)
        out = out.view(b, s, s, rh, rw, c).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(b, c, h, w)

        # Eq. 6: add local context enhancement, a depth-wise conv on the value.
        # v is region-partitioned (B, n_reg, tpr, C); restore it to spatial form
        # to match the original layout before the depth-wise conv.
        v_spatial = v.view(b, s, s, rh, rw, c).permute(0, 5, 1, 3, 2, 4
                                                       ).reshape(b, c, h, w)
        return out + self.lce(v_spatial)


# ---------------------------------------------------------------------------
# SegFormer efficient self-attention
# ---------------------------------------------------------------------------
