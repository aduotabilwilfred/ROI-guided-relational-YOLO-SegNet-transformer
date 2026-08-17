from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Soft dice loss for binary segmentation.
    logits: (B, 1, H, W) raw scores
    target: (B, 1, H, W) or (B, H, W) class indices 0/1 or binary masks
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5
) -> torch.Tensor:
    """
    Combined BCE + Dice loss for binary segmentation.
    """
    if target.dim() == 3:
        target = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dsc = dice_loss(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dsc


@torch.no_grad()
def dice_coefficient(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Hard dice overlap after threshold, for evaluation"""
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred = torch.sigmoid(logits) > threshold.float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() + eps

    return float((2 * inter + eps) / (union + eps))
