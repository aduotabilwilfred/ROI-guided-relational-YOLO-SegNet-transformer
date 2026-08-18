from __future__ import annotations

import torch
from torch import nn


def _find_segment_source_layers(model: nn.Module) -> list[int]:
    """
    Return the layer indices that feed the Segment/Detect head.
    """
    for module in model.modules():
        if type(module).__name__ in ("Segment", "Detect"):
            f = module.f
            return list(f) if isinstance(f, (list, tuple)) else [f]
    raise RuntimeError("no Segment/Detect head found in model")


class FrozenYOLOFeatures(nn.Module):
    """Wrap a trained YOLOv8 to yield frozen neck features and detected boxes.

    Parameters

    weights : path to trained YOLOv8-seg weights (a fold's best.pt).
    conf    : detection confidence threshold for box extraction.
    imgsz   : inference size (must match how the head expects features, 512).
    """

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 512):
        super().__init__()
        from ultralytics import YOLO

        self.yolo = YOLO(weights)
        self.model = self.yolo.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.conf = conf
        self.imgsz = imgsz
        self.source_layers = _find_segment_source_layers(self.model)

        self._feats: dict = {}
        for idx in self.source_layers:
            self.model.model[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            self._feats[idx] = out

        return hook

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """
        Run the frozen backbone and return the neck features P3, P4, P5.

        images : (B, 3, H, W) preprocessed tensor. Returns the feature maps in
        source-layer order (finest first).
        """
        self._feats.clear()
        self.model(images)
        return [self._feats[idx] for idx in self.source_layers]

    @torch.no_grad()
    def detect_boxes(self, image_paths: list[str]) -> list[torch.Tensor]:
        """
        Return per-image normalised xyxy boxes from the frozen detector.

        Uses Ultralytics predict, which handles letterboxing and NMS.
        Each entry is (N_i, 4) in [0, 1]; empty (0, 4) when nothing is detected.
        """
        results = self.yolo.predict(
            image_paths, imgsz=self.imgsz, conf=self.conf, verbose=False
        )
        boxes = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                boxes.append(torch.zeros(0, 4))
            else:
                boxes.append(r.boxes.xyxyn.cpu())  # normalised xyxy
        return boxes

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, image_paths: list[str]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Return (neck features, per-image boxes) for a batch.

        images and image_paths must correspond: images is the tensor the head
        consumes, image_paths the same images on disk for the detector's own
        preprocessing. Both are needed because Ultralytics predict works from
        paths while the head works from the denoised tensor.
        """
        feats = self.extract_features(images)
        boxes = self.detect_boxes(image_paths)
        return feats, boxes
