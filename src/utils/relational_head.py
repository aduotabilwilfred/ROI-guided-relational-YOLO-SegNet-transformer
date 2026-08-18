from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from utils.allmlp_decoder import AllMLPDecoder
from utils.bilra_attention import BiLevelRoutingAttention
from utils.roi_partition import apply_mask, build_roi_masks
from utils.rtrb import RelationalTransformerBlock


@dataclass
class HeadConfig:
    image_size: int = 512
    backbone_channels: tuple = (64, 128, 256)  # P3, P4, P5 (nano YOLOv8)
    embed_dim: int = 128
    num_heads: int = 8
    reduction: int = 4  # segformer efficient-attention (tunable)
    dilation: float = 1.5  # f_w ring factor
    n_regions: int = 4  # BiLRA region grid
    top_k: int = 4  # BiLRA routing top_k
    num_classes: int = 1  # segmentation classes
    use_bilra_in_head: bool = True


class RelationalStage(nn.Module):
    """
    One head stage: optional BiLRA, then the RTrB dual-head block.

    The RTrB consumes f_m and f_w feature maps (already masked). BiLRA, when
    enabled, refines the fused feature first, giving the efficient routing the
    soft-mask design leaves room for.
    """

    def __init__(self, cfg: HeadConfig, in_channels: int):
        super().__init__()
        self.use_bilra = cfg.use_bilra_in_head
        if self.use_bilra:
            self.bilra = BiLevelRoutingAttention(
                in_channels,
                num_heads=cfg.num_heads,
                n_regions=cfg.n_regions,
                top_k=cfg.top_k,
            )
        self.rtrb = RelationalTransformerBlock(
            in_channels, embed_dim=cfg.embed_dim, num_heads=cfg.num_heads
        )
        self.project = nn.Conv2d(2 * cfg.embed_dim, cfg.embed_dim, kernel_size=1)

    def forward(
        self, feat: torch.Tensor, m_mask: torch.Tensor, w_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.use_bilra:
            feat = self.bilra(feat)
        f_m = apply_mask(feat, m_mask)
        f_w = apply_mask(feat, w_mask)
        out = self.rtrb(f_m, f_w)  # (B, 2*embed_dim, H, W)
        return self.project(out)  # (B, embed_dim, H, W)


class RelationalSegHead(nn.Module):
    """Full segmentation head composing all six tested components"""

    def __init__(self, cfg: HeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or HeadConfig()
        c = self.cfg

        # Project each backbone to embed_dim
        self.scale_proj = nn.ModuleList(
            [nn.Conv2d(ch, c.embed_dim, kernel_size=1) for ch in c.backbone_channels]
        )

        # one relational stage per backbone scale
        self.stages = nn.ModuleList(
            [RelationalStage(c, c.embed_dim) for _ in c.backbone_channels]
        )

        # decoder fuses stage outputs to the mask
        self.decoder = AllMLPDecoder(
            in_channels=[c.embed_dim] * len(c.backbone_channels),
            embed_dim=c.embed_dim,
            num_classes=c.num_classes,
            out_size=c.image_size,
        )

    def forward(
        self, feats: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> torch.Tensor:
        c = self.cfg
        stage_outputs = []
        for proj, stage, feat in zip(self.scale_proj, self.stages, feats):
            x = proj(feat)
            h, w = x.shape[-2:]
            masks = build_roi_masks(boxes, h, w, dilation=c.dilation, device=x.device)

            stage_outputs.append(stage(x, masks.m_mask, masks.w_mask))
        return self.decoder(stage_outputs)


def buildHead(cfg: HeadConfig | None = None) -> RelationalSegHead:
    return RelationalSegHead(cfg)
