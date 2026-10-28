"""Per-worker shared model pool. One instance of each configured model; thread-safe loading;
per-model readiness; CUDA-required mode fails clearly."""
from __future__ import annotations

import logging
import threading
import time

from app.ai import base
from app.ai import detection, dino_embeddings, ocr, reid, segmentation  # noqa: F401  (register adapters)
from app.ai.base import CudaUnavailable, Model, ModelUnavailable
from app.core.config import ModelsConfig, ModelSpec
from app.monitoring import metrics

log = logging.getLogger(__name__)


def cuda_status() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": False, "cuda": False, "devices": 0}
    return {"torch": True, "cuda": bool(torch.cuda.is_available()),
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch_cuda": getattr(torch.version, "cuda", None)}


def resolve_device(require_cuda: bool, gpu_id: int, uses_gpu_backend: bool) -> str:
    """`gpu_id` is the index *as visible inside the container* (after CUDA_VISIBLE_DEVICES / Docker
    device reservations), which is generally not the host index."""
    st = cuda_status()
    if require_cuda:
        if not st["cuda"]:
            raise CudaUnavailable(
                "VMS_REQUIRE_CUDA=true but CUDA is not available to this process "
                f"(torch installed={st['torch']}, cuda={st['cuda']}). Check NVIDIA driver, Container Toolkit, "
                "Docker GPU reservation and that the PyTorch build matches the driver's CUDA version.")
        if gpu_id >= st["devices"]:
            raise CudaUnavailable(f"gpu_id={gpu_id} but only {st['devices']} CUDA device(s) are visible in this container")
        return f"cuda:{gpu_id}"
    return f"cuda:{gpu_id}" if (uses_gpu_backend and st["cuda"] and gpu_id < st["devices"]) else "cpu"


class ModelManager:
    def __init__(self, cfg: ModelsConfig, worker_id: str, require_cuda: bool = False, gpu_id: int = 0,
                 force_mock: bool = False):
        self.cfg, self.worker_id, self.require_cuda, self.gpu_id = cfg, worker_id, require_cuda, gpu_id
        self.force_mock = force_mock
        self._models: dict[str, Model] = {}
        self._status: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.gpu_lock = threading.Lock()  # serialises GPU inference across stages in this worker

    # ---- loading
    def _spec(self, name: str) -> ModelSpec:
        spec = self.cfg.models[name]
        if self.force_mock:
            log.warning("FORCE_MOCK_MODELS active: model '%s' (%s) replaced by mock EXPLICITLY by config", name, spec.backend)
            return spec.model_copy(update={"backend": "mock"})
        return spec

    def load(self, name: str) -> Model:
        with self._lock:  # prevents load races
            if name in self._models:
                return self._models[name]
            if name not in self.cfg.models:
                raise ModelUnavailable(f"unknown model '{name}'")
            spec = self._spec(name)
            uses_gpu = spec.backend not in ("mock",)
            device = resolve_device(self.require_cuda, self.gpu_id, uses_gpu)
            t0 = time.monotonic()
            try:
                m = base.create(name, spec, device)
                m.load()
                m.warmup()
            except Exception as e:  # record, surface, never substitute
                self._status[name] = {"ready": False, "error": str(e), "backend": spec.backend}
                metrics.model_ready.labels(self.worker_id, name).set(0)
                raise
            m.ready = True
            self._models[name] = m
            self._status[name] = {"ready": True, "error": None, "backend": spec.backend, "version": m.version,
                                  "device": device, "load_s": round(time.monotonic() - t0, 2)}
            metrics.model_ready.labels(self.worker_id, name).set(1)
            log.info("model '%s' ready on %s (backend=%s, %.2fs)", name, device, spec.backend, time.monotonic() - t0)
            return m

    def load_for_profiles(self, profiles: set[str] | None = None, strict: bool = True) -> None:
        names: set[str] = set()
        for pn, p in self.cfg.profiles.items():
            if profiles is not None and pn not in profiles:
                continue
            names.add(p.detection)
            for st, ref in (("ocr", p.ocr), ("reid", p.reid), ("dino", p.dino), ("sam", p.sam)):
                if st in p.stages and ref:
                    names.add(ref)
        errors = {}
        for n in sorted(names):
            try:
                self.load(n)
            except Exception as e:
                errors[n] = str(e)
                if strict:
                    raise
        if errors:
            log.error("some models failed to load: %s", errors)

    # ---- access
    def get(self, name: str) -> Model:
        m = self._models.get(name)
        if m is None or not m.ready:
            raise ModelUnavailable(f"model '{name}' is not ready: {self._status.get(name, {}).get('error', 'not loaded')}")
        return m

    def readiness(self) -> dict[str, bool]:
        return {n: bool(self._status.get(n, {}).get("ready")) for n in self.cfg.models if n in self._status}

    def status(self) -> dict[str, dict]:
        return dict(self._status)

    def backend_label(self) -> str:
        bs = {s.get("backend") for s in self._status.values()}
        return "mock" if bs == {"mock"} else ("mixed" if "mock" in bs else (next(iter(bs)) if bs else "none"))

    def close(self) -> None:
        with self._lock:
            for m in self._models.values():
                try:
                    m.close()
                except Exception:  # noqa: BLE001
                    log.exception("closing model %s", m.name)
            self._models.clear()
