"""Model adapter interfaces shared by all inference stages."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.core.config import ModelSpec


class ModelUnavailable(RuntimeError):
    """Raised when a configured model cannot be loaded. Never replaced silently by another model."""


class CudaUnavailable(ModelUnavailable):
    pass


@dataclass(slots=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    keypoints: list[list[float]] | None = None  # pose models: [[x, y, conf], ...] per body point


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float


@dataclass(slots=True)
class MaskResult:
    area_px: int
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None  # boolean HxW; kept in-process only, never put on Kafka


class Model(ABC):
    kind: str = ""

    def __init__(self, name: str, spec: ModelSpec, device: str):
        self.name, self.spec, self.device = name, spec, device
        self.version = spec.version
        self.ready = False

    @abstractmethod
    def load(self) -> None: ...

    def warmup(self) -> None:  # optional
        pass

    def close(self) -> None:
        self.ready = False


class Detector(Model):
    kind = "detection"
    class_names: dict[int, str]

    @abstractmethod
    def detect(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        """One result list per input frame, same order."""


class OcrEngine(Model):
    kind = "ocr"

    @abstractmethod
    def read(self, crops: list[np.ndarray]) -> list[OcrResult]: ...


class ReidModel(Model):
    kind = "reid"
    dim: int = 0

    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """(N, D) L2-normalised appearance embeddings."""


class DinoModel(Model):
    kind = "dino"
    dim: int = 0

    @abstractmethod
    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """(N, D) feature embeddings."""


class Segmenter(Model):
    kind = "segmentation"

    @abstractmethod
    def segment(self, image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[MaskResult]: ...


Factory = Callable[[str, ModelSpec, str], Model]
_FACTORIES: dict[tuple[str, str], Factory] = {}


def register(kind: str, backend: str, factory: Factory) -> None:
    _FACTORIES[(kind, backend)] = factory


def create(name: str, spec: ModelSpec, device: str) -> Model:
    f = _FACTORIES.get((spec.kind, spec.backend))
    if f is None:
        raise ModelUnavailable(
            f"model '{name}': no adapter registered for kind={spec.kind} backend={spec.backend}. "
            f"Known: {sorted(k for k in _FACTORIES if k[0] == spec.kind)}")
    return f(name, spec, device)


def crop(img: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.0) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * bw)), int(min(w, x2 + pad * bw))
    y1, y2 = int(max(0, y1 - pad * bh)), int(min(h, y2 + pad * bh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return img[y1:y2, x1:x2]
