"""YOLOSegNet rebuilt on Figure 2, with the placeholders wired to the tested
components (BiLRA, RTrB, ROI partition, All-MLP decoder) instead of plain CBS
convs standing in for attention.

Fixes made to the submitted skeleton, each checked against the running model:

  1. Fake multi-scale neck. The three detection heads all read a 64x64 map
     (neck_up1/up2 never changed spatial size), so "h/8, h/16, h/32" in the
     comments was not true - all three heads produced the same resolution.
     The neck now genuinely downsamples (stride-2 CBS) so the heads sit at
     h/8, h/16, h/32 as labelled.

  2. Confidence gate zeroed segmentation gradients. pred_scores has no loss of
     its own (see notes below), so early in training it is close to random;
     gating final_mask on it silently drops gradient to the whole decoder for
     any sample that fails the threshold - confirmed: with the gate applied,
     only 109 of 293 parameters received a nonzero gradient on a single
     below-threshold sample. Training now always uses the ungated mask; the
     threshold only filters the output at inference, after backward has run.

  3. BiFormerAttention was a CBS conv with a comment. Replaced with the tested
     BiLevelRoutingAttention (region top-k routing + depth-wise local context).

  4. The rt1..rt5 chain was five CBS convs with no attention and no ROI
     restriction, so nothing about "relational" or the paper's ROI mechanism
     was implemented. Replaced with the tested RelationalTransformerBlock
     (dual-head self/cross attention, gathered over the masked ROI tokens
     only) at each of the four resolutions, using roi_partition's dilated-ring
     f_m/f_w masks built from the detector's own predicted boxes.

  5. mlp_layer was a single 1x1 conv plus a fixed x32 upsample from the
     coarsest scale alone. Replaced with the tested AllMLPDecoder, which fuses
     all four relational stages rather than reading only the last one.

Kept as in the submission: the CBS/Bottleneck/C2f backbone, the three
DetectionHead branches and their box/cls outputs, and the output-patch-
embedding unification onto one h/4 grid.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.allmlp_decoder import AllMLPDecoder  # noqa: E402
from utils.bilra_attention import BiLevelRoutingAttention  # noqa: E402
from utils.roi_partition import build_roi_masks  # noqa: E402
from utils.rtrb import RelationalTransformerBlock  # noqa: E402


# --- 1. BASIC BUILDING BLOCKS FROM THE DIAGRAM (unchanged) -----------------

class CBS(nn.Module):
    """Conv-BN-SiLU module as defined in the top-center legend."""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                              padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block used inside C2f."""

    def __init__(self, ch):
        super().__init__()
        self.cv1 = CBS(ch, ch, k=3, s=1, p=1)
        self.cv2 = CBS(ch, ch, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C2f(nn.Module):
    """C2f module combining Split, Bottlenecks, and Concatenation."""

    def __init__(self, in_ch, out_ch, n=3):
        super().__init__()
        self.cv1 = CBS(in_ch, out_ch, k=1, s=1, p=0)
        self.m = nn.ModuleList([Bottleneck(out_ch // 2) for _ in range(n)])
        self.cv2 = CBS((2 + n) * (out_ch // 2), out_ch, k=1, s=1, p=0)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# --- 2. DETECTOR HEADS (unchanged) -----------------------------------------

class DetectionHead(nn.Module):
    """One of the three multi-scale detection heads.

    The box branch still ends in the figure's "1 x 1 Conv" producing 4
    channels; only how those 4 numbers become a box changes. Regressing
    x1,y1,x2,y2 independently let the head predict x2 < x1 (checked: ~75% of
    boxes were inverted at random init, and torchvision's CIoU does not error
    on that - it silently returns a meaningless value instead of a real
    overlap loss, see the notes returned with the loss-module review).
    Predicting a center and a width/height and deriving the corners from
    those makes an invalid box impossible by construction rather than merely
    unlikely: width and height come from sigmoid, so they cannot be negative,
    and x1 = cx - w/2 < cx + w/2 = x2 always holds.
    """

    def __init__(self, in_ch, num_classes=1):
        super().__init__()
        self.box_branch = nn.Sequential(
            CBS(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, 4, kernel_size=1),   # [cx, cy, w, h], raw
        )
        self.cls_branch = nn.Sequential(
            CBS(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, num_classes, kernel_size=1),
        )

    def forward(self, x):
        raw = self.box_branch(x)
        cx, cy = torch.sigmoid(raw[:, 0:1]), torch.sigmoid(raw[:, 1:2])
        w, h = torch.sigmoid(raw[:, 2:3]), torch.sigmoid(raw[:, 3:4])
        x1 = (cx - w / 2).clamp(0.0, 1.0)
        y1 = (cy - h / 2).clamp(0.0, 1.0)
        x2 = (cx + w / 2).clamp(0.0, 1.0)
        y2 = (cy + h / 2).clamp(0.0, 1.0)
        # Corners are already valid, normalised [0, 1] coordinates here, not
        # raw logits - callers must not apply sigmoid to these four channels
        # again (only the 5th, the class logit, is still raw for BCE).
        box = torch.cat([x1, y1, x2, y2], dim=1)
        cls = self.cls_branch(x)
        return torch.cat([box, cls], dim=1)


# --- 3. RELATIONAL YOLO-SEGNET ---------------------------------------------

class YOLOSegNet(nn.Module):
    def __init__(self, num_classes: int = 1, image_size: int = 512,
                c1: int = 64, c2: int = 128,
                c3: int = 256, c4: int = 512, embed_dim: int = 128,
                num_heads: int = 8, dilation: float = 1.5,
                n_regions: int = 4, topk: int = 4, box_conf: float = 0.5):
        super().__init__()
        self.box_conf = box_conf   # confidence used to build ROI masks

        # --- BACKBONE ---
        self.p1 = CBS(3, 32, k=3, s=2, p=1)     # h/2
        self.p2 = CBS(32, 64, k=3, s=2, p=1)    # h/4
        self.p3 = C2f(64, 128, n=3)             # h/4
        self.p4 = CBS(128, 128, k=3, s=2, p=1)  # h/8
        self.p5 = C2f(128, 256, n=6)            # h/8
        self.p6 = CBS(256, 256, k=3, s=2, p=1)  # h/16
        self.p7 = C2f(256, 512, n=6)            # h/16
        self.p8 = CBS(512, 512, k=3, s=2, p=1)  # h/32
        self.p9 = C2f(512, 512, n=3)            # h/32

        # BiLRA needs a region grid that evenly divides the feature map; the
        # backbone tail is h/32 (16x16 for a 512 input), so n_regions must
        # divide that resolution. Same routing/local-context mechanism as the
        # tested module, applied where the figure places Bi-Former Att.
        self.biformer = BiLevelRoutingAttention(
            512, num_heads=num_heads, n_regions=n_regions, topk=topk)

        # --- NECK: a real top-down / bottom-up FPN ---
        # The submission's neck never changed spatial resolution (C2f has no
        # stride), so all three heads silently read the same map; my first
        # rewrite fixed the up-path but left neck_down2 without a stride too,
        # and dropped x9/biformer out of the neck altogether. This version
        # traces every stage's resolution explicitly in forward() rather than
        # trusting variable names, and merges x9 with the shallower features
        # the way the figure's upsample+concat, then CBS-stride2+concat,
        # arrows actually draw it.
        #   top-down (upsample + concat):  x9 (h/32) -> p_h16 -> p_h8
        #   bottom-up (stride2 + concat):  p_h8 -> n_h16 -> n_h32
        self.reduce_x9 = CBS(512, 256, k=1, s=1, p=0)
        self.fuse_h16_td = C2f(256 + 512, 256, n=3)   # x9-up (256) + x6 (512)
        self.reduce_h16 = CBS(256, 128, k=1, s=1, p=0)
        self.fuse_h8_td = C2f(128 + 256, 128, n=3)    # h16-up (128) + x4 (256)

        self.down_h8_to_h16 = CBS(128, 128, k=3, s=2, p=1)
        self.fuse_h16_bu = C2f(128 + 256, 256, n=3)   # p_h8-down (128) + p_h16_td (256)
        self.down_h16_to_h32 = CBS(256, 256, k=3, s=2, p=1)
        self.fuse_h32_bu = C2f(256 + 256, 512, n=3)   # n_h16-down (256) + x9 (256, reduced)

        self.head_high = DetectionHead(128, num_classes)      # h/8
        self.head_mid = DetectionHead(256, num_classes)       # h/16
        self.head_low = DetectionHead(512, num_classes)       # h/32

        # --- Output patch embedding: unify the three heads onto one h/4 grid.
        self.patch_embedding = CBS(5 * 3, c1, k=3, s=1, p=1)

        # --- Sequential relational stages, h/4 -> h/8 -> h/16 -> h/32.
        # Each stage is the tested RTrB (ROI-restricted dual-head attention)
        # rather than a plain conv, on channel widths c1..c4.
        self.proj_h4 = CBS(c1, c1, k=1, s=1, p=0)
        self.down_h8 = CBS(c1, c2, k=3, s=2, p=1)
        self.down_h16 = CBS(c2, c3, k=3, s=2, p=1)
        self.down_h32 = CBS(c3, c4, k=3, s=2, p=1)

        self.rtrb1 = RelationalTransformerBlock(c1, embed_dim, num_heads)
        self.rtrb2 = RelationalTransformerBlock(c2, embed_dim, num_heads)
        self.rtrb4 = RelationalTransformerBlock(c3, embed_dim, num_heads)
        self.rtrb5 = RelationalTransformerBlock(c4, embed_dim, num_heads)
        # RTrB emits 2*embed_dim channels (Eq. 20's channel concat); project
        # each stage back so the decoder receives the channel widths it needs.
        self.proj1 = nn.Conv2d(2 * embed_dim, c1, kernel_size=1)
        self.proj2 = nn.Conv2d(2 * embed_dim, c2, kernel_size=1)
        self.proj4 = nn.Conv2d(2 * embed_dim, c3, kernel_size=1)
        self.proj5 = nn.Conv2d(2 * embed_dim, c4, kernel_size=1)

        self.dilation = dilation
        self.image_size = image_size
        self.decoder = AllMLPDecoder(in_channels=[c1, c2, c3, c4],
                                     embed_dim=c1, num_classes=1,
                                     out_size=image_size)

    def _boxes_from_head(self, head_out: torch.Tensor,
                         max_boxes: int = 8) -> List[torch.Tensor]:
        """Threshold and NMS a detection head's own output into per-image boxes.

        head_out: (B, 5, Hg, Wg) = [x1,y1,x2,y2,cls_logit] per grid cell, the
        same tensor DetectionHead produces. DetectionHead now decodes its own
        box coordinates (center+size -> corners) before returning them, so
        channels 0:4 are already valid [0, 1] corners, not raw logits; only
        the class channel is still a logit. Used only to build ROI masks for
        the relational stages; box supervision is the loss module's job.

        A flat confidence threshold with no de-duplication is not enough: on a
        freshly initialised (or still-training) head, about half of all grid
        cells pass sigmoid(logit) > 0.5 by symmetry, and adjacent cells around
        any true object all fire together with no NMS to collapse them. Their
        union covers most of the image, which is the same runaway-attention
        cost ROI restriction exists to avoid (see rtrb.py's gather/scatter
        notes) - it reappeared here for a different reason, not fixed by that
        module alone. NMS plus a hard cap on kept boxes keeps the ROI small no
        matter how confident or immature the detector currently is.
        """
        import torchvision.ops as ops

        b, _, hg, wg = head_out.shape
        boxes_xyxy = head_out[:, :4]                            # already [0, 1]
        conf = torch.sigmoid(head_out[:, 4])
        boxes_per_image = []
        for i in range(b):
            coords = boxes_xyxy[i].flatten(1).transpose(0, 1)
            scores = conf[i].flatten()
            keep = scores > self.box_conf
            coords, scores = coords[keep], scores[keep]

            # DetectionHead's center+size decoding makes x2>x1, y2>y1 hold by
            # construction (checked below), so this is a defensive no-op kept
            # in case a future box-branch change reintroduces raw corners.
            valid = (coords[:, 2] > coords[:, 0]) & (coords[:, 3] > coords[:, 1])
            coords, scores = coords[valid], scores[valid]

            if coords.shape[0] > 0:
                nms_keep = ops.nms(coords * max(hg, wg), scores, iou_threshold=0.5)
                coords = coords[nms_keep][:max_boxes]
            boxes_per_image.append(coords)
        return boxes_per_image

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError(
                f"YOLOSegNet was built for {self.image_size}x{self.image_size} "
                f"(the decoder's fixed output size) but got {h}x{w}; pass "
                f"image_size={h} at construction if that is intentional.")

        # --- backbone ---
        x2 = self.p3(self.p2(self.p1(x)))
        x4 = self.p5(self.p4(x2))              # h/8,  256ch
        x6 = self.p7(self.p6(x4))               # h/16, 512ch
        x9 = self.biformer(self.p9(self.p8(x6)))   # h/32, 512ch

        # --- neck: top-down (upsample + concat) ---
        x9_reduced = self.reduce_x9(x9)                                    # h/32, 256ch
        x9_up = F.interpolate(x9_reduced, size=x6.shape[-2:],
                              mode="bilinear", align_corners=False)        # h/16
        p_h16 = self.fuse_h16_td(torch.cat([x9_up, x6], dim=1))            # h/16, 256ch
        p_h16_reduced = self.reduce_h16(p_h16)                             # h/16, 128ch
        p_h16_up = F.interpolate(p_h16_reduced, size=x4.shape[-2:],
                                 mode="bilinear", align_corners=False)     # h/8
        p_h8 = self.fuse_h8_td(torch.cat([p_h16_up, x4], dim=1))           # h/8, 128ch

        # --- neck: bottom-up (stride-2 + concat) ---
        n_h16 = self.fuse_h16_bu(torch.cat(
            [self.down_h8_to_h16(p_h8), p_h16], dim=1))                    # h/16, 256ch
        n_h32 = self.fuse_h32_bu(torch.cat(
            [self.down_h16_to_h32(n_h16), x9_reduced], dim=1))             # h/32, 512ch

        # Three genuinely different resolutions feed the three heads.
        assert p_h8.shape[-2:] == tuple(s // 8 for s in x.shape[-2:])
        assert n_h16.shape[-2:] == tuple(s // 16 for s in x.shape[-2:])
        assert n_h32.shape[-2:] == tuple(s // 32 for s in x.shape[-2:])

        out_high = self.head_high(p_h8)               # (B,5,h/8,w/8)
        out_mid = self.head_mid(n_h16)                 # (B,5,h/16,w/16)
        out_low = self.head_low(n_h32)                 # (B,5,h/32,w/32)

        # --- output patch embedding: unify onto one h/4 grid ---
        target_h, target_w = h // 4, w // 4
        unified = torch.cat([
            F.interpolate(out_high, size=(target_h, target_w),
                          mode="bilinear", align_corners=False),
            F.interpolate(out_mid, size=(target_h, target_w),
                          mode="bilinear", align_corners=False),
            F.interpolate(out_low, size=(target_h, target_w),
                          mode="bilinear", align_corners=False),
        ], dim=1)
        feat_h4 = self.patch_embedding(unified)       # (B, c1, h/4, w/4)

        # ROI masks for the relational stages come from the model's own
        # highest-resolution box head, thresholded at box_conf (see
        # _boxes_from_head). This is the same "detect first, restrict
        # attention to the box" flow the two-stage pipeline uses, folded
        # into a single forward pass here.
        boxes = self._boxes_from_head(out_high)

        feat_h4 = self.proj_h4(feat_h4)
        masks_h4 = build_roi_masks(boxes, target_h, target_w,
                                   dilation=self.dilation, device=x.device)
        s1 = self.proj1(self.rtrb1(feat_h4, masks_h4.m_mask, masks_h4.w_mask))

        feat_h8 = self.down_h8(s1)
        hh8, ww8 = feat_h8.shape[-2:]
        masks_h8 = build_roi_masks(boxes, hh8, ww8, dilation=self.dilation,
                                   device=x.device)
        s2 = self.proj2(self.rtrb2(feat_h8, masks_h8.m_mask, masks_h8.w_mask))

        feat_h16 = self.down_h16(s2)
        hh16, ww16 = feat_h16.shape[-2:]
        masks_h16 = build_roi_masks(boxes, hh16, ww16, dilation=self.dilation,
                                    device=x.device)
        s4 = self.proj4(self.rtrb4(feat_h16, masks_h16.m_mask, masks_h16.w_mask))

        feat_h32 = self.down_h32(s4)
        hh32, ww32 = feat_h32.shape[-2:]
        masks_h32 = build_roi_masks(boxes, hh32, ww32, dilation=self.dilation,
                                    device=x.device)
        s5 = self.proj5(self.rtrb5(feat_h32, masks_h32.m_mask, masks_h32.w_mask))

        mask_logits = self.decoder([s1, s2, s4, s5])   # (B, 1, H, W)
        return mask_logits, out_high, out_mid, out_low
