"""OCR adapter. Mock for dev; real engines plug in through `register("ocr", "<backend>", factory)`.
No concrete engine is bundled: configure one (e.g. EasyOCR, PaddleOCR, a plate-specific model) and register it."""
from __future__ import annotations

import numpy as np

from app.ai.base import ModelUnavailable, OcrEngine, OcrResult, register


class MockOcr(OcrEngine):
    def load(self) -> None:
        self.ready = True

    def read(self, crops: list[np.ndarray]) -> list[OcrResult]:
        # deterministic pseudo-text derived from crop mean intensity; clearly synthetic
        return [OcrResult(f"MOCK{int(c.mean()) % 1000:03d}", 0.5) for c in crops]


class UnconfiguredOcr(OcrEngine):
    def load(self) -> None:
        raise ModelUnavailable(
            f"OCR model '{self.name}' backend '{self.spec.backend}' is not implemented in this build. "
            "Implement app.ai.base.OcrEngine for your engine and call register('ocr', '<backend>', factory) "
            "(see README 'Adding a model adapter').")

    def read(self, crops):  # pragma: no cover
        raise ModelUnavailable("not loaded")


register("ocr", "mock", lambda n, s, d: MockOcr(n, s, d))
for _b in ("easyocr", "paddleocr", "plate"):
    register("ocr", _b, lambda n, s, d: UnconfiguredOcr(n, s, d))
