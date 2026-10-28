"""Turns per-frame inference output into tracked, schema-validated events + overlay state."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from app.ai.reid import AppearanceGallery
from app.monitoring import metrics
from app.monitoring.metrics import RateMeter, RollingQuantiles
from app.schemas.event import Detection, DetectionEvent
from app.messaging.topics import Topics

log = logging.getLogger(__name__)
Publish = Callable[[str, str, dict], bool]


@dataclass
class FrameResult:
    camera_id: str
    worker_id: str
    session_id: str | None
    sequence_number: int
    capture_ts: float
    ingest_ts: float
    resolution: tuple[int, int]
    model_name: str
    model_version: str
    detections: list[Detection]
    processing_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    reid_embeddings: dict[int, np.ndarray] = field(default_factory=dict)  # detection index -> vector (in-process only)
    extra: dict = field(default_factory=dict)
    done_ts: float = field(default_factory=time.time)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IouTracker:
    """Greedy class-aware IoU tracker (lightweight; swap for ByteTrack/BoT-SORT if needed)."""

    def __init__(self, iou_thr: float = 0.3, max_age: int = 10):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks: dict[int, dict] = {}
        self._next = 1

    def update(self, dets: list[Detection]) -> list[int]:
        ids = [-1] * len(dets)
        cand = sorted(((iou(t["box"], d.bbox_xyxy), tid, i) for tid, t in self.tracks.items()
                       for i, d in enumerate(dets) if t["cls"] == d.class_id), reverse=True)
        used_t, used_d = set(), set()
        for sc, tid, i in cand:
            if sc < self.iou_thr:
                break
            if tid in used_t or i in used_d:
                continue
            used_t.add(tid), used_d.add(i)
            ids[i] = tid
            self.tracks[tid].update(box=dets[i].bbox_xyxy, age=0)
        for i, d in enumerate(dets):
            if ids[i] == -1:
                ids[i] = self._next
                self.tracks[self._next] = {"box": d.bbox_xyxy, "cls": d.class_id, "age": 0}
                self._next += 1
        for tid in list(self.tracks):
            if tid not in used_t and tid not in ids:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        return ids


class ResultStore:
    """Latest result per camera, read by overlay encoders (independent of Kafka/streaming)."""

    def __init__(self):
        self._r: dict[str, FrameResult] = {}
        self._l = threading.Lock()

    def put(self, r: FrameResult) -> None:
        with self._l:
            self._r[r.camera_id] = r

    def get(self, camera_id: str) -> FrameResult | None:
        with self._l:
            return self._r.get(camera_id)

    def drop(self, camera_id: str) -> None:
        with self._l:
            self._r.pop(camera_id, None)


class EmbeddingStore:
    """Bounded in-process LRU of embeddings referenced from events by `embedding_ref`.
    A production deployment would back this with a vector DB / object store."""

    def __init__(self, max_items: int = 5000):
        self.max, self._d = max_items, OrderedDict()
        self._l = threading.Lock()

    def put(self, ref: str, v: np.ndarray) -> None:
        with self._l:
            self._d[ref] = v
            while len(self._d) > self.max:
                self._d.popitem(last=False)

    def get(self, ref: str):
        with self._l:
            return self._d.get(ref)


class ResultProcessor:
    def __init__(self, worker_id: str, publish: Publish, store: ResultStore, emit_empty: bool = False,
                 reid_threshold: Callable[[str], float] | None = None):
        self.worker_id, self.publish, self.store, self.emit_empty = worker_id, publish, store, emit_empty
        self._trackers: dict[str, IouTracker] = {}
        self._galleries: dict[str, AppearanceGallery] = {}
        self.embeddings = EmbeddingStore()
        self.latency = RollingQuantiles()
        self.infer_rate: dict[str, RateMeter] = {}
        self.total_rate = RateMeter()
        self._reid_thr = reid_threshold
        self._lock = threading.Lock()

    def forget(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)
            self._galleries.pop(camera_id, None)
            self.infer_rate.pop(camera_id, None)
        self.store.drop(camera_id)
        for g in (metrics.inference_fps,):
            try:
                g.remove(self.worker_id, camera_id)
            except KeyError:
                pass

    def inference_fps(self, camera_id: str) -> float:
        m = self.infer_rate.get(camera_id)
        return m.rate() if m else 0.0

    def handle(self, r: FrameResult) -> None:
        cid = r.camera_id
        with self._lock:
            tracker = self._trackers.setdefault(cid, IouTracker())
            gallery = self._galleries.setdefault(cid, AppearanceGallery(self._reid_thr(cid) if self._reid_thr else 0.85))
            meter = self.infer_rate.setdefault(cid, RateMeter())
        ids = tracker.update(r.detections)
        for i, (d, tid) in enumerate(zip(r.detections, ids)):
            d.track_id = tid
            emb = r.reid_embeddings.get(i)
            if emb is not None:
                mt, sim = gallery.match(tid, emb)
                d.appearance_match_track_id, d.appearance_similarity = mt, (round(sim, 4) if sim is not None else None)
                ref = f"reid:{cid}:{r.sequence_number}:{i}"
                self.embeddings.put(ref, emb)
                d.embedding_ref = ref
            metrics.detections_total.labels(self.worker_id, cid, d.class_name).inc()
        self.store.put(r)
        e2e = r.done_ts - r.capture_ts
        self.latency.add(e2e)
        metrics.end_to_end_latency_seconds.labels(self.worker_id, cid).observe(e2e)
        meter.mark()
        self.total_rate.mark()
        metrics.inference_fps.labels(self.worker_id, cid).set(meter.rate())
        if not r.detections and not self.emit_empty:
            return
        ev = DetectionEvent(
            idempotency_key=f"{cid}:{r.session_id or 'nosession'}:{r.sequence_number}:{r.model_name}",
            camera_id=cid, worker_id=r.worker_id, session_id=r.session_id,
            timestamp=datetime.fromtimestamp(r.capture_ts, timezone.utc), sequence_number=r.sequence_number,
            model_name=r.model_name, model_version=r.model_version, detections=r.detections,
            frame_resolution=r.resolution, processing_ms=round(r.processing_ms, 3),
            end_to_end_ms=round(e2e * 1000, 3), stage_timings_ms={k: round(v, 3) for k, v in r.timings_ms.items()},
            extra=r.extra)
        # non-blocking: returns False (and counts a drop) if the bus buffer is full / broker down
        self.publish(Topics.DETECTION_EVENTS, cid, ev.model_dump(mode="json"))
