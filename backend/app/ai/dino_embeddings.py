"""DINO-family visual embeddings. Embeddings stay in-process / external store; events carry only
model version + dimension (never vectors)."""
from __future__ import annotations

import numpy as np

from app.ai.base import DinoModel, ModelUnavailable, register


class MockDino(DinoModel):
    dim = 64

    def load(self) -> None:
        self.ready = True

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), np.float32)
        for i, im in enumerate(images):
            g = im.reshape(-1, 3).mean(0)
            out[i, :3] = g
            out[i, 3:] = np.resize(im[:: max(1, im.shape[0] // 8), :: max(1, im.shape[1] // 8)].mean(2).ravel(), self.dim - 3)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)


class UnconfiguredDino(DinoModel):
    def load(self) -> None:
        raise ModelUnavailable(
            f"DINO model '{self.name}' backend '{self.spec.backend}' not implemented in this build. Implement "
            "app.ai.base.DinoModel (e.g. HF DINOv2 via transformers) and register('dino', '<backend>', factory).")

    def embed(self, images):  # pragma: no cover
        raise ModelUnavailable("not loaded")


register("dino", "mock", lambda n, s, d: MockDino(n, s, d))
for _b in ("hf_dino", "dinov2", "dinov3"):
    register("dino", _b, lambda n, s, d: UnconfiguredDino(n, s, d))
