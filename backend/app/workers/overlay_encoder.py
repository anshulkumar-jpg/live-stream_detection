"""Overlay drawing + FFmpeg encoding to a sink (MediaMTX RTSP publish URL, file, or null).

The encoder runs at a constant output FPS: each tick it takes the newest decoded frame (older ones are
discarded, order preserved) or repeats the last one, so the stream stays smooth regardless of
inference rate. Detection boxes come from the latest inference result (held for `detection_hold_ms`)."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np

from app.core.config import EncoderCfg, OverlayCfg
from app.core.security import redact_url
from app.monitoring import metrics
from app.monitoring.metrics import RateMeter
from app.utils import ffmpeg as ff
from app.utils.subprocess_manager import ManagedProcess
from app.workers.frame_router import Frame
from app.workers.result_processor import FrameResult

log = logging.getLogger(__name__)
_PALETTE = [(0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255), (0, 0, 255), (255, 255, 0)]


def _color(cfg: OverlayCfg, class_name: str, class_id: int) -> tuple[int, int, int]:
    c = cfg.class_colors.get(class_name)
    return tuple(int(v) for v in c) if c else _PALETTE[class_id % len(_PALETTE)]  # type: ignore[return-value]


def draw_overlay(img: np.ndarray, result: FrameResult | None, cfg: OverlayCfg, camera_id: str,
                 ts: float, hold_ms: int | None = None, scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Return an annotated COPY (the shared frame is never mutated). Handles result=None / empty detections."""
    out = img.copy()
    if not cfg.enabled:
        return out
    hold = (cfg.detection_hold_ms if hold_ms is None else hold_ms) / 1000
    if result is not None and (ts - result.capture_ts) <= hold:
        sx, sy = scale
        for d in result.detections:
            x1, y1, x2, y2 = (int(d.bbox_xyxy[0] * sx), int(d.bbox_xyxy[1] * sy),
                              int(d.bbox_xyxy[2] * sx), int(d.bbox_xyxy[3] * sy))
            col = _color(cfg, d.class_name, d.class_id)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, cfg.thickness)
            if cfg.draw_labels:
                label = d.class_name
                if cfg.draw_confidence:
                    label += f" {d.confidence:.2f}"
                if d.track_id is not None:
                    label += f" #{d.track_id}"
                if d.ocr_text:
                    label += f" [{d.ocr_text}]"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(th + 4, y1)
                cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + bl - 2), col, -1)
                cv2.putText(out, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if cfg.draw_timestamp:
        text = f"{camera_id}  {datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z"
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class OverlayEncoder:
    def __init__(self, camera_id: str, worker_id: str, enc_cfg: EncoderCfg, overlay_cfg: OverlayCfg, sink: str,
                 results_get: Callable[[str], FrameResult | None], rtsp_transport: str = "tcp",
                 source_size: tuple[int, int] = (1280, 720)):
        self.camera_id, self.worker_id, self.cfg, self.overlay, self.sink = camera_id, worker_id, enc_cfg, overlay_cfg, sink
        self.results_get, self.rtsp_transport = results_get, rtsp_transport
        self.src_size = source_size  # decoded frame size (boxes are in this coordinate space)
        self._q: deque[Frame] = deque(maxlen=max(1, enc_cfg.queue_size))
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._proc: ManagedProcess | None = None
        self._thread = threading.Thread(target=self._run, name=f"enc-{camera_id}", daemon=True)
        self.restarts = 0
        self.frames_written = 0
        self.last_error: str | None = None
        self.publishing = False
        self.encoder_name = ""
        self.fps_meter = RateMeter()
        self._lbl = (worker_id, camera_id)
        self.gave_up = False

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._proc:
            self._proc.stop()
        self._thread.join(timeout)
        self.publishing = False
        for g in (metrics.stream_publish_status, metrics.output_fps):
            try:
                g.remove(*self._lbl)
            except KeyError:
                pass

    def push(self, frame: Frame) -> None:
        """Called from the ingest thread: O(1), never blocks; oldest frame dropped when full."""
        with self._cv:
            if len(self._q) == self._q.maxlen:
                metrics.dropped_frames_total.labels(*self._lbl, "encoder_skip").inc()
            self._q.append(frame)
            self._cv.notify()

    # ----------------------------------------------------------------
    def _start_proc(self) -> None:
        plan = ff.choose_encoder(self.cfg)
        self.encoder_name = plan.encoder_name
        args = ff.build_encode_args(plan, self.cfg, self.sink, self.rtsp_transport)
        log.info("starting encoder %s -> %s", plan.encoder_name, redact_url(self.sink), extra={"camera_id": self.camera_id})
        self._proc = ManagedProcess(args, f"enc-{self.camera_id}", stdin=True, stdout=False).start()

    def _run(self) -> None:
        period = 1.0 / self.cfg.fps
        last_frame: Frame | None = None
        next_tick = time.monotonic()
        backoff = 1.0
        w, h = self.cfg.width, self.cfg.height
        while not self._stop.is_set():
            if self._proc is None or not self._proc.alive:
                if self._proc is not None:
                    self.last_error = f"encoder exited rc={self._proc.returncode}: {self._proc.last_error()}"
                    self.publishing = False
                    metrics.stream_publish_status.labels(*self._lbl).set(0)
                    self._proc.stop()
                    self._proc = None
                    self.restarts += 1
                    metrics.ffmpeg_restart_total.labels(*self._lbl, "encoder").inc()
                    if self.restarts > self.cfg.max_restarts:
                        self.gave_up = True
                        log.error("encoder giving up after %d restarts: %s", self.restarts, self.last_error,
                                  extra={"camera_id": self.camera_id})
                        return
                    if self._stop.wait(backoff):
                        return
                    backoff = min(backoff * 2, 15.0)
                try:
                    self._start_proc()
                except ff.FFmpegUnavailable as e:
                    self.last_error = str(e)
                    self.gave_up = True
                    log.error("encoder unavailable: %s", e, extra={"camera_id": self.camera_id})
                    return
                next_tick = time.monotonic()
            # wait for next tick, collecting newest frame
            with self._cv:
                while not self._stop.is_set():
                    rem = next_tick - time.monotonic()
                    if rem <= 0:
                        break
                    self._cv.wait(rem)
                if self._q:
                    skipped = len(self._q) - 1
                    last_frame = self._q[-1]
                    self._q.clear()
                    if skipped:
                        metrics.dropped_frames_total.labels(*self._lbl, "encoder_skip").inc(skipped)
            if last_frame is None:
                next_tick += period
                continue
            now = time.time()
            res = self.results_get(self.camera_id)
            img = last_frame.data
            sx = sy = 1.0
            if img.shape[1] != w or img.shape[0] != h:
                sx, sy = w / img.shape[1], h / img.shape[0]
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            annotated = draw_overlay(img, res, self.overlay, self.camera_id, last_frame.capture_ts, scale=(sx, sy)) \
                if self.overlay.enabled else img
            try:
                assert self._proc and self._proc.proc and self._proc.proc.stdin
                self._proc.proc.stdin.write(np.ascontiguousarray(annotated).data)
                self.frames_written += 1
                self.fps_meter.mark()
                metrics.output_fps.labels(*self._lbl).set(self.fps_meter.rate())
                if self.frames_written > 3 and self._proc.alive:
                    self.publishing = True
                    metrics.stream_publish_status.labels(*self._lbl).set(1)
                    if time.monotonic() - next_tick < 5:
                        backoff = 1.0
            except (BrokenPipeError, OSError, ValueError):
                self.publishing = False  # loop top handles restart
                time.sleep(0.05)
                continue
            next_tick += period
            if time.monotonic() - next_tick > 1.0:  # fell behind (e.g. restart): resync, don't burst
                next_tick = time.monotonic()
