from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AllMLPDecoder(nn.Module):
    """
    Projects each input scale to a common dimension, upsamples all to the
    finest scale, concatenates, fuses with an MLP, and classifies the output

    """

    def __init__(
        self, in_channels: list[int], embed_dim: int, num_classes: int, out_size: int
    ):
        super().__init__()
        self.out_size = out_size
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        target = max(f.shape[-1] for f in feats)  # finest resolution
        fused = []
        for proj, f in zip(self.projections, feats):
            x = proj(f)
            if x.shape[-1] != target:
                x = F.interpolate(
                    x, size=(target, target), mode="bilinear", align_corners=False
                )
            fused.append(x)
        x = self.fuse(torch.cat(fused, dim=1))
        logits = self.classifier(x)
        return F.interpolate(
            logits,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )
