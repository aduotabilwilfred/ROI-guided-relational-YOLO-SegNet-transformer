import math

import torch
import torch.nn as nn


class YOLODetectionLoss(nn.Module):
    def __init__(self, num_classes=1, stride=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

        # BCE Loss with Logits handles raw network outputs without needing manual Sigmoids
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')

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
        v = (4 / math.pi ** 2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - center_dist_sq / diag_sq - alpha * v
        return (1.0 - ciou).mean()

    def forward(self, head_outputs, targets):
        """
        head_outputs: Tuple containing raw tensors from your 3 detection heads:
                      (out_high [B, 5, h/8, w/8], out_mid [B, 5, h/16, w/16], out_low [B, 5, h/32, w/32])
        targets: Dictionary containing:
                 - 'gt_boxes': Tensor of shape [B, 4] -> Ground truth bounding boxes [x1, y1, x2, y2]
                 - 'gt_labels': Tensor of shape [B, 1] -> 1.0 if tumor is present, 0.0 for healthy bone
        """
        out_high, out_mid, out_low = head_outputs
        gt_boxes = targets['gt_boxes']
        gt_labels = targets['gt_labels']

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
            B = head_tensor.shape[0]   # C, H_grid, W_grid unused past the reshape

            # Reshape tensor to align predictions per grid cell: [B, H_grid * W_grid, 5]
            head_flat = head_tensor.permute(0, 2, 3, 1).contiguous().view(B, -1, 5)

            # Slice the predicted raw bounding boxes [x1, y1, x2, y2] and class logits
            pred_boxes_raw = head_flat[..., :4]
            pred_cls_logits = head_flat[..., 4:]

            # --- 1. COMPUTE CLASSIFICATION LOSS (BCE.loss) ---
            # Broaden target labels to match the spatial dimensions of the grid cells
            target_cls = gt_labels.unsqueeze(1).repeat(1, head_flat.shape[1], self.num_classes)
            loss_cls = self.bce_cls(pred_cls_logits, target_cls)
            total_cls_loss += loss_cls

            # --- 2. COMPUTE BOUNDING BOX DISTANCE LOSS (Bbox.loss) ---
            # We track grid elements belonging to images containing positive tumor labels
            pos_batch_indices = torch.where(gt_labels == 1.0)[0]

            if len(pos_batch_indices) > 0:
                # Isolate the boxes that correspond to valid tumor samples
                pos_pred_boxes = pred_boxes_raw[pos_batch_indices].view(-1, 4)

                # Expand ground truth boxes to calculate metrics against matching grid spaces
                pos_gt_boxes = gt_boxes[pos_batch_indices].unsqueeze(1).repeat(1, head_flat.shape[1], 1).view(-1, 4)

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
