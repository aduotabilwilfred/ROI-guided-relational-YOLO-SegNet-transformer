from __future__ import annotations

import torch
from torch import nn


class RelationalTransformerBlock(nn.Module):
    """Dual-head self/cross attention over ROI features.

    Parameters

    in_channels : channels of the input feature map.
    embed_dim   : per-head embedding width; each head outputs embed_dim
        channels, so the block outputs 2 * embed_dim.
    num_heads   : attention heads; embed_dim must be divisible by it.

    forward(feat, m_mask, w_mask) where feat is (B, in_channels, H, W),
    m_mask is the tumour binary mask (B, 1, H, W), and w_mask is the
    surrounding ring mask (B, 1, H, W). Attention is computed only over
    the masked ROI tokens of each sample.
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

        # w_k residual embeddings: 1x1 convs, one per attention branch
        # (self/cross). They map the attended embed_dim features to in_channels
        # so the residual add to feat is shape-compatible.
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
        return tokens.reshape(n, self.num_heads, self.head_dim).permute(1, 0, 2)

    def _attend_tokens(self, q, k, v):
        # q,k,v (heads, n, head_dim) -> (heads, nq, head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = scores.softmax(dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self, feat: torch.Tensor, m_mask: torch.Tensor, w_mask: torch.Tensor
    ) -> torch.Tensor:
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
        q_sa = self._project_qkv(feat, self.q_sa)  # (B, N, embed)
        k_sa = self._project_qkv(feat, self.k_sa)
        v_sa = self._project_qkv(feat, self.v_sa)
        q_ca = self._project_qkv(feat, self.q_ca)
        k_ca = self._project_qkv(feat, self.k_ca)
        v_ca = self._project_qkv(feat, self.v_ca)

        m_flat = m_mask.reshape(b, n) > 0.5
        w_flat = w_mask.reshape(b, n) > 0.5

        g_sa = torch.zeros(b, n, self.embed_dim, dtype=feat.dtype, device=dev)
        g_ca = torch.zeros(b, n, self.embed_dim, dtype=feat.dtype, device=dev)

        for i in range(b):
            m_idx = m_flat[i].nonzero(as_tuple=True)[0]  # tumour tokens
            if m_idx.numel() == 0:
                continue  # background: skip
            w_idx = w_flat[i].nonzero(as_tuple=True)[0]  # ring tokens

            # Self-attention over tumour tokens only.
            qs = self._heads(q_sa[i, m_idx])
            ks = self._heads(k_sa[i, m_idx])
            vs = self._heads(v_sa[i, m_idx])
            att_sa = self._attend_tokens(qs, ks, vs)  # (heads, m, hd)
            g_sa[i, m_idx] = att_sa.permute(1, 0, 2).reshape(-1, self.embed_dim)

            # Cross-attention: tumour queries, ring keys/values. If no ring
            # tokens, cross-attention has nothing to attend to; leave zeros.
            if w_idx.numel() > 0:
                qc = self._heads(q_ca[i, m_idx])
                kc = self._heads(k_ca[i, w_idx])
                vc = self._heads(v_ca[i, w_idx])
                att_ca = self._attend_tokens(qc, kc, vc)  # (heads, m, hd)
                g_ca[i, m_idx] = att_ca.permute(1, 0, 2).reshape(-1, self.embed_dim)

        # Scatter attended tokens back to spatial maps.
        def to_map(tokens):
            return tokens.permute(0, 2, 1).reshape(b, self.embed_dim, h, w)

        g_sa_map = to_map(g_sa)
        g_ca_map = to_map(g_ca)

        # Residual learning: w_k is a 1x1 conv, residual add to feat.
        f_sa = self.w_sa(g_sa_map) + feat
        f_ca = self.w_ca(g_ca_map) + feat

        # Project each head to embed_dim and concatenate over channels
        out = torch.cat([self.out_sa(f_sa), self.out_ca(f_ca)], dim=1)
        return out
