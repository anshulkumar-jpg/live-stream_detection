"""Bounded per-camera frame queues with FPS sampling, stale-drop, priority + fair batching."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.core.config import RouterCfg
from app.monitoring import metrics


@dataclass(slots=True)
class Frame:
    """Immutable-by-convention frame envelope. `data` is a read-only BGR uint8 HxWx3 array; stages that
    need to draw must copy (overlay does). Timestamps: capture_ts/ingest_ts are epoch seconds (UTC)."""

    camera_id: str
    worker_id: str
    sequence_number: int
    capture_ts: float
    ingest_ts: float
    source_resolution: tuple[int, int]
    data: np.ndarray
    session_id: str | None = None


@dataclass
class _CamQ:
    fps: float
    priority: int
    q: deque = field(default_factory=deque)
    last_accept: float = 0.0
    last_served: float = 0.0
    dropped_stale: int = 0
    dropped_full: int = 0
    sampled_out: int = 0
    accepted: int = 0


class FrameRouter:
    def __init__(self, cfg: RouterCfg, worker_id: str):
        self.cfg, self.worker_id = cfg, worker_id
        self._cams: dict[str, _CamQ] = {}
        self._cv = threading.Condition()
        self._closed = False

    # ---- lifecycle
    def register(self, camera_id: str, fps: float | None = None, priority: int = 0) -> None:
        with self._cv:
            self._cams[camera_id] = _CamQ(fps or self.cfg.default_fps, priority)

    def unregister(self, camera_id: str) -> None:
        with self._cv:
            self._cams.pop(camera_id, None)
        try:
            metrics.frame_queue_depth.remove(self.worker_id, camera_id)
        except KeyError:
            pass

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    # ---- producer side
    def offer(self, frame: Frame) -> bool:
        """Apply target-FPS sampling, then enqueue. Drops the OLDEST frame when full (keeps freshest)."""
        with self._cv:
            c = self._cams.get(frame.camera_id)
            if c is None:
                return False
            if frame.capture_ts - c.last_accept < (1.0 / c.fps) * 0.9:
                c.sampled_out += 1
                return False
            c.last_accept = frame.capture_ts
            if len(c.q) >= self.cfg.queue_size:
                c.q.popleft()
                c.dropped_full += 1
                metrics.dropped_frames_total.labels(self.worker_id, frame.camera_id, "queue_full").inc()
            c.q.append(frame)
            c.accepted += 1
            metrics.frame_queue_depth.labels(self.worker_id, frame.camera_id).set(len(c.q))
            self._cv.notify()
            return True

    # ---- consumer side
    def _pop_batch_locked(self, max_n: int) -> list[Frame]:
        now = time.time()
        stale_s = self.cfg.stale_ms / 1000.0
        # drop stale first
        for cid, c in self._cams.items():
            while c.q and now - c.q[0].capture_ts > stale_s:
                c.q.popleft()
                c.dropped_stale += 1
                metrics.dropped_frames_total.labels(self.worker_id, cid, "stale").inc()
        # priority desc, then least-recently-served (fairness)
        order = sorted((cid for cid, c in self._cams.items() if c.q),
                       key=lambda k: (-self._cams[k].priority, self._cams[k].last_served))
        batch: list[Frame] = []
        while order and len(batch) < max_n:
            progressed = False
            for cid in list(order):
                c = self._cams[cid]
                took = 0
                while c.q and took < self.cfg.fair_burst and len(batch) < max_n:
                    batch.append(c.q.popleft())
                    took += 1
                    progressed = True
                    if c.priority == 0:
                        break  # non-critical cameras: one frame per round => round-robin fairness
                if took:
                    c.last_served = now
                if not c.q:
                    order.remove(cid)
                if len(batch) >= max_n:
                    break
            if not progressed:
                break
        for cid, c in self._cams.items():
            metrics.frame_queue_depth.labels(self.worker_id, cid).set(len(c.q))
        return batch

    def get_batch(self, max_n: int, wait_s: float = 0.25, fill_s: float = 0.025) -> list[Frame]:
        """Block up to wait_s for the first frame, then up to fill_s to let the batch fill."""
        deadline = time.monotonic() + wait_s
        with self._cv:
            while not self._closed and not any(c.q for c in self._cams.values()):
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return []
                self._cv.wait(rem)
            if self._closed:
                return []
            fill_deadline = time.monotonic() + fill_s
            while sum(len(c.q) for c in self._cams.values()) < max_n:
                rem = fill_deadline - time.monotonic()
                if rem <= 0:
                    break
                self._cv.wait(rem)
            return self._pop_batch_locked(max_n)

    # ---- introspection
    def depths(self) -> dict[str, int]:
        with self._cv:
            return {cid: len(c.q) for cid, c in self._cams.items()}

    def stats(self) -> dict[str, dict]:
        with self._cv:
            return {cid: {"depth": len(c.q), "accepted": c.accepted, "dropped_stale": c.dropped_stale,
                          "dropped_full": c.dropped_full, "sampled_out": c.sampled_out, "fps": c.fps,
                          "priority": c.priority} for cid, c in self._cams.items()}

    def set_fps(self, camera_id: str, fps: float) -> None:
        with self._cv:
            if camera_id in self._cams:
                self._cams[camera_id].fps = fps
