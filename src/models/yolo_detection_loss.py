import math

import torch
from torch import nn


class YOLODetectionLoss(nn.Module):
    def __init__(self, num_classes=1, stride=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

        # BCE Loss with Logits handles raw network outputs without needing manual Sigmoids
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="mean")

    def _get_iou_loss(self, pred_boxes, gt_boxes, eps: float = 1e-7):
        """
        Elementwise (matched-pair) Complete IoU loss.

        pred_boxes, gt_boxes: (N, 4) xyxy, pred_boxes[i] matched to gt_boxes[i].

        The original version called torchvision.ops.complete_box_iou(pred, gt)
        and kept only its diagonal. That function always returns the full
        (M, N) pairwise matrix - at a real grid (out_high is 64x64 = 4096
        cells per image), M=N=4096*num_positive_images produced an
        8192x8192 matrix (0.27 GB) per internal term, several of which are
        computed (IoU, enclosing diagonal, center distance, aspect ratio),
        which ran out of memory on the first real batch; the tiny (N=2) smoke
        test that first ran against this file never exercised that scale.
        This computes the same Complete IoU formula directly, elementwise, so
        cost is O(N) instead of O(N^2). Verified to match torchvision's
        diagonal to float32 precision on small inputs before being used here.
        """
        if pred_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        px1, py1, px2, py2 = pred_boxes.unbind(-1)
        gx1, gy1, gx2, gy2 = gt_boxes.unbind(-1)

        # Intersection over union.
        ix1, iy1 = torch.maximum(px1, gx1), torch.maximum(py1, gy1)
        ix2, iy2 = torch.minimum(px2, gx2), torch.minimum(py2, gy2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
        union = area_p + area_g - inter + eps
        iou = inter / union

        # Enclosing box diagonal and center distance (the DIoU term).
        ex1, ey1 = torch.minimum(px1, gx1), torch.minimum(py1, gy1)
        ex2, ey2 = torch.maximum(px2, gx2), torch.maximum(py2, gy2)
        diag_sq = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + eps
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        center_dist_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

        # Aspect-ratio consistency term (the C in CIoU).
        pw, ph = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
        gw, gh = (gx2 - gx1).clamp(min=eps), (gy2 - gy1).clamp(min=eps)
        v = (4 / math.pi**2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - center_dist_sq / diag_sq - alpha * v
        return (1.0 - ciou).mean()

    def _positive_cell_mask(self, gt_boxes, hg, wg):
        """Which grid cells are responsible for each ground-truth box.

        gt_boxes: (N, 4) normalised xyxy. Returns a (N, hg * wg) bool mask.
        A cell is responsible when its centre lies inside the box, plus the
        cell that contains the box centre, so even a box smaller than one
        cell keeps exactly one responsible cell. This follows the
        anchor-free, centre-based convention the paper describes for YOLOv8
        (it does not spell out the rule itself).
        """
        device, dtype = gt_boxes.device, gt_boxes.dtype
        n = gt_boxes.shape[0]
        ys = (torch.arange(hg, device=device, dtype=dtype) + 0.5) / hg
        xs = (torch.arange(wg, device=device, dtype=dtype) + 0.5) / wg
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        cx, cy = cx.reshape(1, -1), cy.reshape(1, -1)

        x1, y1, x2, y2 = (t.unsqueeze(1) for t in gt_boxes.unbind(-1))
        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

        gcx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gcy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        ci = (gcx * wg).floor().clamp(0, wg - 1).long()
        cj = (gcy * hg).floor().clamp(0, hg - 1).long()
        centre = torch.zeros_like(inside)
        centre[torch.arange(n, device=device), cj * wg + ci] = True
        return inside | centre

    def forward(self, head_outputs, targets):
        """
        head_outputs: Tuple containing raw tensors from your 3 detection heads:
                      (out_high [B, 5, h/8, w/8], out_mid [B, 5, h/16, w/16], out_low [B, 5, h/32, w/32])
        targets: Dictionary containing:
                 - 'gt_boxes': Tensor of shape [B, 4] -> Ground truth bounding boxes [x1, y1, x2, y2]
                 - 'gt_labels': Tensor of shape [B, 1] -> 1.0 if tumor is present, 0.0 for healthy bone
        """
        out_high, out_mid, out_low = head_outputs
        gt_boxes = targets["gt_boxes"]
        gt_labels = targets["gt_labels"]

        # Plain-float accumulators here mean total_bbox_loss stays a Python
        # float (no .item(), no device, breaks any caller treating the return
        # as a tensor) whenever a batch has no positive image - e.g. a small
        # batch that happens to be all background. Confirmed: this crashed on
        # the second training batch of a smoke test. Tensors from the start
        # keep the return type consistent regardless of batch composition.
        device = out_high.device
        total_bbox_loss = torch.zeros((), device=device)
        total_cls_loss = torch.zeros((), device=device)

        # We loop over the three feature map scales (High, Mid, Low resolution heads)
        for head_tensor in [out_high, out_mid, out_low]:
            B = head_tensor.shape[0]  # C, H_grid, W_grid unused past the reshape

            # Reshape tensor to align predictions per grid cell: [B, H_grid * W_grid, 5]
            head_flat = head_tensor.permute(0, 2, 3, 1).contiguous().view(B, -1, 5)

            # Slice the predicted raw bounding boxes [x1, y1, x2, y2] and class logits
            pred_boxes_raw = head_flat[..., :4]
            pred_cls_logits = head_flat[..., 4:]

            hg, wg = head_tensor.shape[2], head_tensor.shape[3]
            # We track grid elements belonging to images containing positive tumor labels
            pos_batch_indices = torch.where(gt_labels == 1.0)[0]

            # Responsible cells: a tumour image's own cells (centre inside the box,
            # or the centre cell). Everything else, including every cell of a
            # background image, is a negative. Each grid cell is the unit of
            # classification, so the target must say which cells are the tumour.
            resp_full = torch.zeros(
                B, head_flat.shape[1], dtype=torch.bool, device=device
            )
            if len(pos_batch_indices) > 0:
                resp_full[pos_batch_indices] = self._positive_cell_mask(
                    gt_boxes[pos_batch_indices], hg, wg
                )

            # --- 1. COMPUTE CLASSIFICATION LOSS (BCE.loss) ---
            # Responsible cells are rare, so the positive class is weighted by the
            # negative/positive ratio (capped), otherwise the head predicts "no tumour"
            # everywhere. The paper does not specify this weighting.
            target_cls = resp_full.float().unsqueeze(-1)
            n_pos = resp_full.sum().clamp(min=1).float()
            n_neg = resp_full.numel() - n_pos
            pos_weight = (n_neg / n_pos).clamp(max=100.0)
            loss_cls = nn.functional.binary_cross_entropy_with_logits(
                pred_cls_logits, target_cls, pos_weight=pos_weight
            )
            total_cls_loss += loss_cls

            # --- 2. COMPUTE BOUNDING BOX DISTANCE LOSS (Bbox.loss) ---
            if len(pos_batch_indices) > 0:
                # Only the cells responsible for each tumour box regress it.
                # Training every cell toward the same whole-image box made the
                # head predict a blurred average, which kept the loss near 1.0.
                pos_gt = gt_boxes[pos_batch_indices]
                resp = resp_full[pos_batch_indices]  # (P, Hg*Wg)
                pos_pred_boxes = pred_boxes_raw[pos_batch_indices][resp]
                pos_gt_boxes = pos_gt.unsqueeze(1).expand(-1, resp.shape[1], -1)[resp]

                # DetectionHead (yoloseg_relational.py) now decodes its own box
                # branch (center + size -> corners) before returning, so
                # pred_boxes_raw is already valid, normalised [0, 1] xyxy, not
                # a raw logit. The extra sigmoid()*512.0 here was written for
                # the earlier raw-corner head and would double-squash an
                # already-bounded value into the wrong range. Complete IoU is
                # invariant to a uniform coordinate scale (checked: identical
                # value on the same boxes in [0,1] vs pixel space), so pred and
                # gt are both kept normalised rather than rescaled to pixels.
                loss_bbox = self._get_iou_loss(pos_pred_boxes, pos_gt_boxes)
                total_bbox_loss += loss_bbox

        # Average the accumulated losses across all three prediction scales
        final_bbox_loss = total_bbox_loss / 3.0
        final_cls_loss = total_cls_loss / 3.0

        joint_loss = final_bbox_loss + final_cls_loss

        return joint_loss, final_bbox_loss, final_cls_loss
