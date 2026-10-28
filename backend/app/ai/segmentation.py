"""Segmentation (SAM3) adapter. Box-prompted from detections, run selectively (see ProfileSpec.sam_*).

The SAM3 loader is deliberately NOT implemented: weights/licence and the exact package API must be
confirmed by you. Implement `Segmenter` and register it as backend "sam3"; the mock backend exists
so the rest of the pipeline can be exercised."""
from __future__ import annotations

import numpy as np

from app.ai.base import MaskResult, ModelUnavailable, Segmenter, register


class MockSegmenter(Segmenter):
    def load(self) -> None:
        self.ready = True

    def segment(self, image, boxes):
        out = []
        h, w = image.shape[:2]
        for (x1, y1, x2, y2) in boxes:
            xi1, yi1, xi2, yi2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            m = np.zeros((h, w), bool)
            m[yi1:yi2, xi1:xi2] = True  # box-shaped fake mask
            out.append(MaskResult(int(m.sum()), (x1, y1, x2, y2), m))
        return out


class UnconfiguredSam(Segmenter):
    def load(self) -> None:
        raise ModelUnavailable(
            f"Segmentation model '{self.name}' backend '{self.spec.backend}' is not implemented. SAM3 weights and "
            "package API must be provided by you: implement app.ai.base.Segmenter (box prompts -> masks) and "
            "register('segmentation', 'sam3', factory). See README 'SAM3 setup'.")

    def segment(self, image, boxes):  # pragma: no cover
        raise ModelUnavailable("not loaded")


register("segmentation", "mock", lambda n, s, d: MockSegmenter(n, s, d))
for _b in ("sam3", "sam2", "sam"):
    register("segmentation", _b, lambda n, s, d: UnconfiguredSam(n, s, d))
