from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .allmlp_decoder import AllMLPDecoder
from .bilra_attention import BiLevelRoutingAttention
from .roi_partition import build_roi_masks
from .rtrb import RelationalTransformerBlock


@dataclass
class HeadConfig:
    image_size: int = 512
    # Backwards-compatible standalone default. The training handoff replaces
    # this tuple with the selected frozen detector's probed P3/P4/P5 channels.
    backbone_channels: tuple[int, ...] = (64, 128, 256)
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

    BiLRA (when enabled) refines the full feature map first. The RTrB then
    receives the full feature map together with m_mask and w_mask; it gathers
    ROI tokens internally, so no explicit masking is needed here.
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
        out = self.rtrb(feat, m_mask, w_mask)  # (B, 2*embed_dim, H, W)
        return self.project(out)  # (B, embed_dim, H, W)


class RelationalSegHead(nn.Module):
    """Relational segmentation head with binary tumour-presence output."""

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
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c.embed_dim, 1),
        )
    def forward(
        self, feats: list[torch.Tensor], boxes: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        if len(feats) != len(self.scale_proj):
            raise ValueError(
                f"expected {len(self.scale_proj)} feature maps, got {len(feats)}"
            )
        if not feats:
            raise ValueError("at least one feature map is required")
        if len(boxes) != feats[0].shape[0]:
            raise ValueError(
                f"expected boxes for {feats[0].shape[0]} images, got {len(boxes)}"
            )

        stage_outputs = []
        for scale, (proj, stage, feat) in enumerate(
            zip(self.scale_proj, self.stages, feats)
        ):
            if feat.shape[1] != proj.in_channels:
                raise ValueError(
                    f"feature scale {scale} has {feat.shape[1]} channels; "
                    f"head expects {proj.in_channels}"
                )
            x = proj(feat)
            h, w = x.shape[-2:]
            masks = build_roi_masks(boxes, h, w, dilation=c.dilation, device=x.device)

            stage_outputs.append(stage(x, masks.m_mask, masks.w_mask))

        segmentation_logits = self.decoder(stage_outputs)
        # The deepest relational representation carries the widest context and
        # is already available, so classification adds only pooling + one logit.
        classification_logits = self.classifier(stage_outputs[-1])
        return segmentation_logits, classification_logits


def buildHead(cfg: HeadConfig | None = None) -> RelationalSegHead:
    return RelationalSegHead(cfg)
