"""Relational Transformer Block (RTrB), Eqs. 15-20 of the paper.

A dual-head attention block that operates on ROI-extracted feature tokens. One
head is self-attention over tumour-region features f_m (internal tumour
patterns); the other is cross-attention where tumour features f_m query
surrounding-bone features f_w (global bone context). The two head outputs are
concatenated along the channel dimension.

This block receives f_m and f_w as already-extracted feature maps. Producing
them (cropping the ROI from the backbone features and defining the surrounding
region) is a separate concern handled upstream, so this module can be built and
tested against the equations without depending on the ROI-cropping decision.

Equation mapping, using the paper's own attention convention (Eq. 7,
score = a . b^T, scaled by 1/sqrt(D_head)):

  Eq. 15  F_SA(f_m)      = q_SA(f_m) . k_SA(f_m)^T      (self scores)
  Eq. 16  F_CA(f_m,f_w)  = q_CA(f_m) . k_CA(f_w)^T      (cross scores)
  Eq. 17  G_SA           = softmax(F_SA) . v_SA(f_m)
  Eq. 18  G_CA           = softmax(F_CA) . v_CA(f_w)
  Eq. 19  f_k            = w_k(G_k) (+) f_m   for k in {SA, CA}, w_k a 1x1 conv
  Eq. 20  f_out          = [ f_SA ; f_CA ]    concatenated over channels

Notes on choices, each documented rather than transcribed:
  - Q/K/V are 3x3 convolutions (paper's text). padding=1, groups=1 to preserve
    spatial resolution; the paper does not give padding/groups.
  - head count is a constructor argument (paper does not state it); embed_dim
    must be divisible by it. Both are among the values FHEO is meant to tune.
  - the printed Eqs. 15-16 write the score as K^T q; in the paper's row-token
    convention that is the standard q . k^T score, which is what is implemented.

Output has 2 * embed_dim channels (Eq. 20 concatenates two embed_dim heads).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalTransformerBlock(nn.Module):
    """Dual-head self/cross attention over ROI features (Eqs. 15-20).

    Parameters
    ----------
    in_channels : channels of the input feature maps f_m and f_w.
    embed_dim : per-head embedding width; each head outputs embed_dim channels,
        so the block outputs 2 * embed_dim.
    num_heads : attention heads; embed_dim must be divisible by it.

    forward(feat, m_mask, w_mask): feat is (B, in_channels, H, W); m_mask and
    w_mask are (B, 1, H, W) binary maps selecting tumour and ring tokens.
    Attention runs only over the selected ROI tokens per sample, so cost scales
    with tumour size, not image size. Returns (B, 2 * embed_dim, H, W).
    """

    def __init__(self, in_channels: int, embed_dim: int = 128,
                 num_heads: int = 8, max_roi_tokens: int = 1024):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        # Bounds attention cost regardless of how large the caller's ROI mask
        # is. Self-attention here is O(tokens^2); nothing upstream guarantees
        # a small mask - a mask built from an untrained or still-training box
        # head can cover most of the feature map (checked: ~30-50% of a
        # 512x512 image from random init alone, before the ring's dilation is
        # even applied), and that produced an out-of-memory crash with no
        # cap. Above this budget, tokens are randomly subsampled down to it;
        # below it, gather is unaffected (verified bit-identical output on a
        # mask under the cap, same seed, before vs after this was added).
        self.max_roi_tokens = max_roi_tokens

        # Q/K/V generators: 3x3 convs (paper). Self head reads f_m for all
        # three; cross head reads f_m for query, f_w for key and value.
        def qkv_conv():
            return nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.q_sa, self.k_sa, self.v_sa = qkv_conv(), qkv_conv(), qkv_conv()
        self.q_ca, self.k_ca, self.v_ca = qkv_conv(), qkv_conv(), qkv_conv()

        # w_k residual embeddings (Eq. 19): 1x1 convs, one per head. They map
        # the attended embed_dim features to in_channels so the residual add to
        # f_m is shape-compatible.
        self.w_sa = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.w_ca = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # After residual, each head is back at in_channels; project each to
        # embed_dim so the channel-concat output is a clean 2 * embed_dim.
        self.out_sa = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.out_ca = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def _project_qkv(self, feat, conv):
        # feat (B, C, H, W) -> tokens (B, H*W, embed_dim)
        x = conv(feat)
        b, e, h, w = x.shape
        return x.reshape(b, e, h * w).permute(0, 2, 1)

    def _heads(self, tokens):
        # (n, embed_dim) -> (heads, n, head_dim)
        n = tokens.shape[0]
        return tokens.view(n, self.num_heads, self.head_dim).permute(1, 0, 2)

    def _attend_tokens(self, q, k, v):
        # q,k,v (heads, n, head_dim) -> (heads, nq, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, v)

    def _cap(self, idx: torch.Tensor) -> torch.Tensor:
        """Randomly subsample idx down to max_roi_tokens if it is larger."""
        if idx.numel() <= self.max_roi_tokens:
            return idx
        perm = torch.randperm(idx.numel(), device=idx.device)[:self.max_roi_tokens]
        return idx[perm]

    def forward(self, feat: torch.Tensor, m_mask: torch.Tensor,
                w_mask: torch.Tensor) -> torch.Tensor:
        """ROI-restricted dual-head attention.

        feat   : (B, C, H, W) full feature map.
        m_mask : (B, 1, H, W) binary, tumour region (f_m tokens).
        w_mask : (B, 1, H, W) binary, surrounding ring (f_w tokens).

        Attention runs only over the ROI tokens of each sample, so cost scales
        with tumour size, not image size (the paper's efficiency point). The
        block output is 2*embed_dim channels (Eq. 20). Samples with no ROI
        return zeros for their attended features and fall back to the residual.
        """
        b, _, h, w = feat.shape
        n = h * w
        dev = feat.device

        # Project once over the whole map; gather per-sample below.
        q_sa = self._project_qkv(feat, self.q_sa)      # (B, N, embed)
        k_sa = self._project_qkv(feat, self.k_sa)
        v_sa = self._project_qkv(feat, self.v_sa)
        q_ca = self._project_qkv(feat, self.q_ca)
        k_ca = self._project_qkv(feat, self.k_ca)
        v_ca = self._project_qkv(feat, self.v_ca)

        m_flat = m_mask.reshape(b, n) > 0.5
        w_flat = w_mask.reshape(b, n) > 0.5

        # Match feat's dtype (not just device): under mixed-precision (AMP)
        # feat is float16, and default float32 buffers would break the scatter
        # or silently upcast, defeating AMP.
        g_sa = torch.zeros(b, n, self.embed_dim, dtype=feat.dtype, device=dev)
        g_ca = torch.zeros(b, n, self.embed_dim, dtype=feat.dtype, device=dev)

        for i in range(b):
            m_idx = self._cap(m_flat[i].nonzero(as_tuple=True)[0])   # tumour tokens
            if m_idx.numel() == 0:
                continue                                     # background: skip
            w_idx = self._cap(w_flat[i].nonzero(as_tuple=True)[0])   # ring tokens

            # Self-attention over tumour tokens only.
            qs = self._heads(q_sa[i, m_idx])
            ks = self._heads(k_sa[i, m_idx])
            vs = self._heads(v_sa[i, m_idx])
            att_sa = self._attend_tokens(qs, ks, vs)         # (heads, m, hd)
            g_sa[i, m_idx] = att_sa.permute(1, 0, 2).reshape(-1, self.embed_dim)

            # Cross-attention: tumour queries, ring keys/values. If no ring
            # tokens, cross-attention has nothing to attend to; leave zeros.
            if w_idx.numel() > 0:
                qc = self._heads(q_ca[i, m_idx])
                kc = self._heads(k_ca[i, w_idx])
                vc = self._heads(v_ca[i, w_idx])
                att_ca = self._attend_tokens(qc, kc, vc)     # (heads, m, hd)
                g_ca[i, m_idx] = att_ca.permute(1, 0, 2).reshape(-1, self.embed_dim)

        # Scatter attended tokens back to spatial maps.
        def to_map(tokens):
            return tokens.permute(0, 2, 1).reshape(b, self.embed_dim, h, w)

        g_sa_map = to_map(g_sa)
        g_ca_map = to_map(g_ca)

        # Residual learning (Eq. 19): w_k is a 1x1 conv, residual add to feat.
        f_sa = self.w_sa(g_sa_map) + feat
        f_ca = self.w_ca(g_ca_map) + feat

        # Project each head to embed_dim and concatenate over channels (Eq. 20).
        out = torch.cat([self.out_sa(f_sa), self.out_ca(f_ca)], dim=1)
        return out                                # (B, 2 * embed_dim, H, W)
