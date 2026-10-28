"""Re-identification: appearance embeddings + a per-camera gallery matcher.

Embedding similarity only says "looks similar to track N seen recently on this camera"; it is never a
claim of global identity."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from app.ai.base import ModelUnavailable, ReidModel, register


class MockReid(ReidModel):
    dim = 32

    def load(self) -> None:
        self.ready = True

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            # colour-histogram-like descriptor: crude but deterministic and appearance-dependent
            m = c.reshape(-1, 3).mean(0)
            out[i, :3] = m
            out[i, 3:] = np.resize(np.asarray(c[:: max(1, c.shape[0] // 4), :: max(1, c.shape[1] // 4)].mean(2).ravel()), self.dim - 3)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-6)


class UnconfiguredReid(ReidModel):
    def load(self) -> None:
        raise ModelUnavailable(
            f"ReID model '{self.name}' backend '{self.spec.backend}' not implemented in this build. Implement "
            "app.ai.base.ReidModel (e.g. OSNet/torchreid) and register('reid', '<backend>', factory).")

    def embed(self, crops):  # pragma: no cover
        raise ModelUnavailable("not loaded")


class AppearanceGallery:
    """Bounded per-camera gallery of recent track embeddings."""

    def __init__(self, threshold: float = 0.85, max_tracks: int = 200):
        self.threshold, self.max_tracks = threshold, max_tracks
        self._g: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def match(self, track_id: int, emb: np.ndarray) -> tuple[int | None, float | None]:
        """Best *other* track with cosine similarity >= threshold, then update gallery."""
        best, best_sim = None, -1.0
        for tid, e in self._g.items():
            if tid == track_id:
                continue
            s = float(np.dot(e, emb))
            if s > best_sim:
                best, best_sim = tid, s
        prev = self._g.get(track_id)
        self._g[track_id] = emb if prev is None else _norm(0.7 * prev + 0.3 * emb)
        self._g.move_to_end(track_id)
        while len(self._g) > self.max_tracks:
            self._g.popitem(last=False)
        return (best, best_sim) if best is not None and best_sim >= self.threshold else (None, None)


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-6)


register("reid", "mock", lambda n, s, d: MockReid(n, s, d))
for _b in ("torchreid", "osnet", "fastreid"):
    register("reid", _b, lambda n, s, d: UnconfiguredReid(n, s, d))
