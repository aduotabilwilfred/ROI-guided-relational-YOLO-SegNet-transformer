from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ROIMasks:
    """
    Per-sample f_m and f_w masks at feature resolution.

    m_mask, w_mask: (B, 1, H, W) binary maps. has_roi: (B,) bool, False for
    samples with no detected box (all-zero masks).
    """

    m_mask: torch.Tensor
    w_mask: torch.Tensor
    has_roi: torch.Tensor


def _boxes_to_mask(
    boxes: torch.Tensor, h: int, w: int, device: torch.device
) -> torch.Tensor:
    """
    Union of axis-aligned boxes into a binary (H, W) mask.

    boxes: (N, 4) in normalised xyxy (x0, y0, x1, y1), values in [0, 1].
    Empty N gives an all-zero mask.
    """
    mask = torch.zeros(h, w, device=device)
    if boxes.numel() == 0:
        return mask
    for x0, y0, x1, y1 in boxes:
        c0 = int(torch.clamp(x0 * w, 0, w).item())
        c1 = int(torch.clamp(x1 * w, 0, w).item())
        r0 = int(torch.clamp(y0 * h, 0, h).item())
        r1 = int(torch.clamp(y1 * h, 0, h).item())
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = 1.0
    return mask


def _dilate_boxes(boxes: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale each box about its centre by `factor`, clamped to [0, 1]."""
    if boxes.numel() == 0:
        return boxes
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = (x1 - x0) / 2 * factor
    half_h = (y1 - y0) / 2 * factor
    dilated = torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)
    return dilated.clamp(0.0, 1.0)


def build_roi_masks(
    boxes_per_image: list[torch.Tensor],
    feat_h: int,
    feat_w: int,
    dilation: float = 1.5,
    device: torch.device | None = None,
) -> ROIMasks:
    """Build f_m and f_w masks for a batch.

    Parameters

    boxes_per_image : list of length B; each entry (N_i, 4) normalised xyxy
        boxes for that image (N_i may be 0).
    feat_h, feat_w : feature-map resolution the masks are built at.
    dilation : context-box scale factor for the f_w ring.

    Returns ROIMasks with m_mask, w_mask (B, 1, H, W) and has_roi (B,).
    """
    device = device or (
        boxes_per_image[0].device if boxes_per_image else torch.device("cpu")
    )
    b = len(boxes_per_image)
    m_masks = torch.zeros(b, 1, feat_h, feat_w, device=device)
    w_masks = torch.zeros(b, 1, feat_h, feat_w, device=device)
    has_roi = torch.zeros(b, device=device, dtype=torch.bool)

    for i, boxes in enumerate(boxes_per_image):
        boxes = boxes.to(device)
        m = _boxes_to_mask(boxes, feat_h, feat_w, device)
        if m.sum() == 0:
            continue  # no tumor: we leave masks all zero
        has_roi[i] = True
        dilated = _dilate_boxes(boxes, dilation)
        context = _boxes_to_mask(dilated, feat_h, feat_w, device)
        ring = torch.clamp(context - m, min=0.0)  # dilated minus box = ring
        m_masks[i, 0] = m
        w_masks[i, 0] = ring
    return ROIMasks(m_mask=m_masks, w_mask=w_masks, has_roi=has_roi)


def apply_mask(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Zero out feature-map positions where mask is 0.

    feat: (B, C, H, W); mask: (B, 1, H, W). Used to produce the f_m and f_w
    feature maps the RTrB consumes: apply_mask(feat, m_mask) -> f_m, etc.
    """
    return feat * mask
