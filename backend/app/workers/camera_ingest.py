"""One independent ingest lifecycle (thread + FFmpeg subprocess) per camera.

Failure isolation: each camera owns its own subprocess, thread, backoff and state; nothing is shared
except the thread-safe router."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from app.core.config import DecodeCfg
from app.core.security import redact_text, redact_url
from app.monitoring import metrics
from app.monitoring.metrics import RateMeter
from app.utils import ffmpeg as ff
from app.utils.subprocess_manager import ManagedProcess
from app.workers.frame_router import Frame

log = logging.getLogger(__name__)
StateCb = Callable[[str, str | None], None]


class CameraIngest:
    def __init__(self, camera_id: str, source_url: str, worker_id: str, cfg: DecodeCfg,
                 on_frame: Callable[[Frame], None], on_state: StateCb | None = None,
                 session_id: str | None = None, transport: str | None = None, realtime_synthetic: bool = True):
        self.camera_id, self._url, self.worker_id = camera_id, source_url, worker_id
        self.cfg = cfg.model_copy(update={"rtsp_transport": transport or cfg.rtsp_transport})
        self.on_frame, self.on_state, self.session_id = on_frame, on_state, session_id
        self.realtime_synthetic = realtime_synthetic
        self.state = "starting"
        self.last_error: str | None = None
        self.restarts = 0
        self.frames = 0
        self.seq = 0
        self.source_resolution: tuple[int, int] | None = None
        self.decoder = "cpu"
        self._cuda_failures = 0
        self._force_cpu = False
        self.fps_meter = RateMeter()
        self.last_frame_mono: float | None = None
        self._stop = threading.Event()
        self._proc: ManagedProcess | None = None
        self._thread = threading.Thread(target=self._run, name=f"ingest-{camera_id}", daemon=True)
        self._lbl = (worker_id, camera_id)

    # ---- control
    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._proc:
            self._proc.stop()
        self._thread.join(timeout)
        self._set_state("stopped")
        for g in (metrics.camera_connection_status, metrics.decoder_fps):
            try:
                g.remove(*self._lbl)
            except KeyError:
                pass

    def last_frame_age(self) -> float | None:
        return None if self.last_frame_mono is None else time.monotonic() - self.last_frame_mono

    def _set_state(self, st: str, err: str | None = None) -> None:
        if err:
            self.last_error = redact_text(err)
        if st != self.state:
            self.state = st
            log.info("camera state -> %s%s", st, f" ({self.last_error})" if err else "",
                     extra={"camera_id": self.camera_id, "worker_id": self.worker_id})
            if self.on_state:
                try:
                    self.on_state(st, self.last_error)
                except Exception:  # noqa: BLE001
                    log.exception("on_state callback failed")
        metrics.camera_connection_status.labels(*self._lbl).set(1 if st == "running" else 0)

    # ---- main loop
    def _run(self) -> None:
        backoff = self.cfg.reconnect_initial_s
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._set_state("connecting" if self.restarts == 0 else "reconnecting")
                self._session()
            except Exception as e:  # noqa: BLE001 - one camera must never kill the worker
                self._set_state(self.state, f"ingest error: {type(e).__name__}: {e}")
                log.exception("ingest loop error", extra={"camera_id": self.camera_id})
            if self._stop.is_set():
                break
            self.restarts += 1
            metrics.ffmpeg_restart_total.labels(*self._lbl, "decoder").inc()
            self._set_state("reconnecting", self.last_error)
            if time.monotonic() - started >= self.cfg.stable_after_s:
                backoff = self.cfg.reconnect_initial_s
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self.cfg.reconnect_max_s)

    def _session(self) -> None:
        probe = ff.probe_stream(self._url, self.cfg.rtsp_transport, self.cfg.read_timeout_s)
        if not probe.ok:
            self._set_state("reconnecting", f"probe failed: {probe.error}")
            return
        self.source_resolution = (probe.width, probe.height)
        self.decoder = "cpu" if self._force_cpu else ff.choose_decoder(self.cfg)
        args = ff.build_decode_args(self._url, self.cfg, self.decoder, self.realtime_synthetic)
        log.info("decoder=%s source=%s %dx%d %s@%.1ffps", self.decoder, redact_url(self._url), probe.width,
                 probe.height, probe.codec, probe.fps, extra={"camera_id": self.camera_id})
        proc = ManagedProcess(args, f"dec-{self.camera_id}").start()
        self._proc = proc
        w, h = self.cfg.width, self.cfg.height
        nbytes = w * h * 3
        wd_stop = threading.Event()
        threading.Thread(target=self._watchdog, args=(proc, wd_stop), daemon=True,
                         name=f"wd-{self.camera_id}").start()
        self.last_frame_mono = time.monotonic()
        try:
            assert proc.proc and proc.proc.stdout
            out = proc.proc.stdout
            while not self._stop.is_set():
                buf = np.empty(nbytes, np.uint8)  # fresh buffer per frame: no copy, no aliasing
                mv, got = memoryview(buf), 0
                while got < nbytes:
                    n = out.readinto(mv[got:])
                    if not n:
                        break
                    got += n
                if got < nbytes:  # EOF / killed / truncated frame -> restart decoder
                    if self.decoder == "cuda" and self.frames == 0:
                        self._cuda_failures += 1
                        if self._cuda_failures >= 3 and self.cfg.allow_cpu_fallback and self.cfg.hwaccel != "cuda-strict":
                            self._force_cpu = True
                            log.warning("NVDEC decode failed %d times; falling back to CPU decode (policy allows)",
                                        self._cuda_failures, extra={"camera_id": self.camera_id})
                    if not self._stop.is_set():
                        self._set_state("reconnecting", f"decoder exited rc={proc.proc.poll()}: {proc.last_error()}")
                    return
                now = time.time()
                self.last_frame_mono = time.monotonic()
                frame_arr = buf.reshape(h, w, 3)
                frame_arr.setflags(write=False)
                self.seq += 1
                self.frames += 1
                self.fps_meter.mark()
                metrics.frames_ingested_total.labels(*self._lbl).inc()
                if self.state != "running":
                    self._set_state("running")
                self.on_frame(Frame(self.camera_id, self.worker_id, self.seq, now, now,
                                    self.source_resolution, frame_arr, self.session_id))
                metrics.decoder_fps.labels(*self._lbl).set(self.fps_meter.rate())
        finally:
            wd_stop.set()
            proc.stop()
            self._proc = None

    def _watchdog(self, proc: ManagedProcess, done: threading.Event) -> None:
        """Kill a decoder that stalls (no frame within read_timeout_s) so the read unblocks."""
        limit = max(self.cfg.read_timeout_s, 2.0)
        while not done.wait(1.0):
            if self.last_frame_mono is not None and time.monotonic() - self.last_frame_mono > limit:
                self.last_error = f"no frames for {limit:.0f}s (stalled); restarting decoder"
                log.warning(self.last_error, extra={"camera_id": self.camera_id})
                proc.stop(1.0)
                return
