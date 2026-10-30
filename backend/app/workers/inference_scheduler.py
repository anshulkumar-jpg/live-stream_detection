"""Central batching scheduler: ONE loop per worker feeds shared model instances.

Cameras never create their own models; frames from all cameras are batched per detection model."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Callable

from app.ai import base
from app.ai.model_manager import ModelManager
from app.core.config import InferenceCfg, ProfileSpec
from app.monitoring import metrics
from app.schemas.event import Detection
from app.workers.frame_router import Frame, FrameRouter
from app.workers.result_processor import FrameResult, ResultProcessor

log = logging.getLogger(__name__)
MAX_OCR_CROPS = 8
MAX_REID_CROPS = 16
MAX_SAM_BOXES = 8


class InferenceScheduler:
    def __init__(self, worker_id: str, router: FrameRouter, models: ModelManager, processor: ResultProcessor,
                 cfg: InferenceCfg, profile_for: Callable[[str], tuple[str, ProfileSpec] | None],
                 is_active: Callable[[str], bool], gpu_mem_fraction: Callable[[], float | None] = lambda: None,
                 classes_for: Callable[[str], set[str] | None] = lambda _cid: None):
        self.worker_id, self.router, self.models, self.processor, self.cfg = worker_id, router, models, processor, cfg
        self.profile_for, self.is_active, self.gpu_mem_fraction = profile_for, is_active, gpu_mem_fraction
        self.classes_for = classes_for  # per-camera class filter (lower-case names) or None = keep all
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._frame_counts: dict[str, int] = defaultdict(int)
        self.batches = 0
        self.last_batch_size = 0
        self.last_error: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.router.close()
        self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self.router.get_batch(self.cfg.batch_size, wait_s=0.25, fill_s=self.cfg.batch_timeout_ms / 1000)
                if batch:
                    self.process_batch(batch)
            except Exception as e:  # noqa: BLE001 - keep the loop alive; surface via status
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("inference loop error")
                time.sleep(0.2)

    # public for tests
    def process_batch(self, frames: list[Frame]) -> list[FrameResult]:
        frames = [f for f in frames if self.is_active(f.camera_id)]  # cancellation of removed cameras
        if not frames:
            return []
        frac = self.gpu_mem_fraction()
        if frac is not None and frac > self.cfg.gpu_mem_limit_fraction:
            for f in frames:
                metrics.dropped_frames_total.labels(self.worker_id, f.camera_id, "gpu_memory").inc()
            log.warning("GPU memory %.0f%% above limit %.0f%%: dropping batch of %d",
                        frac * 100, self.cfg.gpu_mem_limit_fraction * 100, len(frames))
            return []
        groups: dict[str, list[tuple[Frame, str, ProfileSpec]]] = defaultdict(list)
        for f in frames:
            pp = self.profile_for(f.camera_id)
            if pp is None:
                continue
            groups[pp[1].detection].append((f, pp[0], pp[1]))
        out: list[FrameResult] = []
        self.batches += 1
        self.last_batch_size = len(frames)
        for model_name, items in groups.items():
            try:
                out.extend(self._run_group(model_name, items))
            except base.ModelUnavailable as e:
                self.last_error = str(e)
                for f, _, _ in items:
                    metrics.dropped_frames_total.labels(self.worker_id, f.camera_id, "model_unavailable").inc()
                log.error("model unavailable: %s", e)
        return out

    def _run_group(self, model_name: str, items: list[tuple[Frame, str, ProfileSpec]]) -> list[FrameResult]:
        det = self.models.get(model_name)
        assert isinstance(det, base.Detector)
        t0 = time.perf_counter()
        with self.models.gpu_lock:
            raws = det.detect([f.data for f, _, _ in items])
        det_ms = (time.perf_counter() - t0) * 1000
        metrics.inference_latency_seconds.labels(self.worker_id, model_name).observe(det_ms / 1000)
        results = []
        for (f, pname, prof), raw in zip(items, raws):
            if not self.is_active(f.camera_id):
                continue
            allowed = self.classes_for(f.camera_id)
            if allowed:
                raw = [r for r in raw if r.class_name.lower() in allowed]
            timings = {"detect": det_ms / len(items)}  # amortised per frame
            dets = [Detection(class_id=r.class_id, class_name=r.class_name, confidence=max(0.0, min(1.0, r.confidence)),
                              bbox_xyxy=tuple(float(v) for v in r.bbox), keypoints=r.keypoints) for r in raw]  # type: ignore[arg-type]
            res = FrameResult(f.camera_id, f.worker_id, f.session_id, f.sequence_number, f.capture_ts, f.ingest_ts,
                              (f.data.shape[1], f.data.shape[0]), model_name, det.version, dets, det_ms / len(items),
                              timings)
            self._secondary(f, prof, res, raw)
            res.done_ts = time.time()
            self.processor.handle(res)
            results.append(res)
        return results

    def _secondary(self, f: Frame, prof: ProfileSpec, res: FrameResult, raw) -> None:
        stages = set(prof.stages)
        self._frame_counts[f.camera_id] += 1
        n = self._frame_counts[f.camera_id]
        try:
            if "ocr" in stages and prof.ocr and res.detections:
                idx = [i for i, d in enumerate(res.detections)
                       if "*" in prof.ocr_classes or d.class_name in prof.ocr_classes][:MAX_OCR_CROPS]
                crops = [(i, base.crop(f.data, res.detections[i].bbox_xyxy, 0.05)) for i in idx]
                crops = [(i, c) for i, c in crops if c is not None]
                if crops:
                    t = time.perf_counter()
                    m = self.models.get(prof.ocr)
                    with self.models.gpu_lock:
                        o = m.read([c for _, c in crops])  # type: ignore[attr-defined]
                    res.timings_ms["ocr"] = (time.perf_counter() - t) * 1000
                    metrics.inference_latency_seconds.labels(self.worker_id, prof.ocr).observe(res.timings_ms["ocr"] / 1000)
                    for (i, _), r in zip(crops, o):
                        res.detections[i].ocr_text, res.detections[i].ocr_confidence = r.text, round(r.confidence, 4)
            if "reid" in stages and prof.reid and res.detections:
                idx = [i for i, d in enumerate(res.detections) if d.class_name in prof.reid_classes][:MAX_REID_CROPS]
                crops = [(i, base.crop(f.data, res.detections[i].bbox_xyxy)) for i in idx]
                crops = [(i, c) for i, c in crops if c is not None]
                if crops:
                    t = time.perf_counter()
                    m = self.models.get(prof.reid)
                    with self.models.gpu_lock:
                        e = m.embed([c for _, c in crops])  # type: ignore[attr-defined]
                    res.timings_ms["reid"] = (time.perf_counter() - t) * 1000
                    metrics.inference_latency_seconds.labels(self.worker_id, prof.reid).observe(res.timings_ms["reid"] / 1000)
                    for (i, _), vec in zip(crops, e):
                        res.reid_embeddings[i] = vec
            if "dino" in stages and prof.dino and n % max(1, prof.dino_every_n) == 0:
                t = time.perf_counter()
                m = self.models.get(prof.dino)
                with self.models.gpu_lock:
                    e = m.embed([f.data])  # type: ignore[attr-defined]
                res.timings_ms["dino"] = (time.perf_counter() - t) * 1000
                res.extra.update(dino_model=prof.dino, dino_version=m.version, dino_dim=int(e.shape[1]))
            if "sam" in stages and prof.sam and res.detections and n % max(1, prof.sam_every_n) == 0:
                idx = [i for i, d in enumerate(res.detections)
                       if not prof.sam_classes or d.class_name in prof.sam_classes][:MAX_SAM_BOXES]
                if idx:
                    t = time.perf_counter()
                    m = self.models.get(prof.sam)
                    with self.models.gpu_lock:
                        masks = m.segment(f.data, [res.detections[i].bbox_xyxy for i in idx])  # type: ignore[attr-defined]
                    res.timings_ms["sam"] = (time.perf_counter() - t) * 1000
                    for i, mk in zip(idx, masks):
                        res.detections[i].mask_area_px, res.detections[i].mask_model = mk.area_px, prof.sam
        except base.ModelUnavailable as e:
            # optional stage failed: keep detection output flowing, report the problem
            self.last_error = f"secondary stage unavailable: {e}"
            log.error(self.last_error)
