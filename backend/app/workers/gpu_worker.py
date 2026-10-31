"""GPU worker process: owns camera sessions, shared models and the inference loop for ONE GPU.

Controlled exclusively through the Kafka command interface (or `--standalone` with cameras.yaml)."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from app.ai.model_manager import ModelManager, cuda_status, resolve_device
from app.core.config import (ModelsConfig, PipelineConfig, ProfileSpec, Settings, get_settings, load_models_config,
                             load_pipeline_config)
from app.core.logging import setup_logging
from app.core.security import redact_url
from app.messaging.kafka_consumer import SeenCache
from app.messaging.kafka_producer import EventBus, make_bus
from app.messaging.topics import Topics
from app.monitoring import metrics
from app.schemas.event import CameraHeartbeat, CameraStatusEvent, PipelineMetricsEvent, WorkerHeartbeat
from app.schemas.worker import CameraSpec, Command
from app.utils import ffmpeg as ff
from app.utils.subprocess_manager import terminate_all
from app.workers.camera_ingest import CameraIngest
from app.workers.frame_router import Frame, FrameRouter
from app.workers.inference_scheduler import InferenceScheduler
from app.workers.mediamtx_publisher import build_publish_url
from app.workers.overlay_encoder import OverlayEncoder
from app.workers.result_processor import ResultProcessor, ResultStore

log = logging.getLogger(__name__)


class GpuMonitor:
    """GPU utilisation/memory via NVML if installed, else torch memory info; None when unknown.
    `gpu_id` is the device index as seen INSIDE the container."""

    def __init__(self, gpu_id: int):
        self.gpu_id, self._h, self._nvml = gpu_id, None, None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self._nvml = pynvml
        except Exception:  # noqa: BLE001 - NVML optional
            pass

    def sample(self) -> dict:
        if self._nvml and self._h:
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._h)
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._h)
                return {"util": float(u.gpu), "used": int(m.used), "total": int(m.total)}
            except Exception:  # noqa: BLE001
                pass
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(self.gpu_id)
                return {"util": None, "used": int(total - free), "total": int(total)}
        except Exception:  # noqa: BLE001
            pass
        return {"util": None, "used": None, "total": None}

    def mem_fraction(self) -> float | None:
        s = self.sample()
        return None if not s["total"] else s["used"] / s["total"]


@dataclass
class Session:
    spec: CameraSpec
    session_id: str
    profile_name: str
    profile: ProfileSpec
    ingest: CameraIngest
    encoder: OverlayEncoder | None = None
    stream_enabled: bool = False
    started: float = field(default_factory=time.time)


class NullBus(EventBus):
    async def start(self): ...
    async def stop(self): ...
    def publish(self, topic, key, value): return True
    async def publish_critical(self, topic, key, value): ...
    async def subscribe(self, topics, group, handler, from_beginning=False): ...


class GpuWorker:
    def __init__(self, settings: Settings, pipeline: PipelineConfig, models_cfg: ModelsConfig, bus: EventBus | None = None,
                 sink_factory: Callable[[str], str] | None = None, realtime_synthetic: bool = True):
        self.s, self.p, self.mc = settings, pipeline, models_cfg
        self.worker_id = settings.worker_id
        self.bus = bus or make_bus(settings, f"-{self.worker_id}")
        self.sink_factory = sink_factory or (lambda cid: build_publish_url(settings, cid))
        self.realtime_synthetic = realtime_synthetic
        self.status = "starting"
        self.sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self.router = FrameRouter(pipeline.router, self.worker_id)
        self.store = ResultStore()
        self.models = ModelManager(models_cfg, self.worker_id, settings.require_cuda, settings.gpu_id, settings.force_mock_models)
        self.gpu = GpuMonitor(settings.gpu_id)
        self.processor = ResultProcessor(self.worker_id, self._publish, self.store)
        self.scheduler = InferenceScheduler(self.worker_id, self.router, self.models, self.processor, pipeline.inference,
                                            self._profile_for, self._is_active, self.gpu.mem_fraction, self._classes_for)
        self._stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._seen_cmds = SeenCache(5000)
        self._hostname = socket.gethostname()

    # ------------------------------------------------------------ lifecycle
    def start(self, metrics_server: bool = True) -> None:
        resolve_device(self.s.require_cuda, self.s.gpu_id, True)  # fail fast & clearly if CUDA is required
        caps = ff.detect_capabilities()
        if not caps.ffmpeg_found:
            raise RuntimeError("ffmpeg/ffprobe not found in PATH")
        log.info("ffmpeg: %s nvdec_possible=%s nvenc=%s", caps.version, caps.nvdec_possible, caps.nvenc_usable)
        self.models.load_for_profiles(strict=True)
        self._start_loop()
        self._run_async(self.bus.start())
        if not self.s.standalone:
            self._run_async(self.bus.subscribe([Topics.PIPELINE_COMMANDS], f"worker-{self.worker_id}", self._on_command))
        self.scheduler.start()
        if metrics_server:
            from prometheus_client import start_http_server

            start_http_server(self.s.worker_metrics_port)
            log.info("worker metrics on :%d/metrics", self.s.worker_metrics_port)
        self.status = "ready"
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        self._hb_thread.start()
        self._heartbeat_once()
        log.info("worker %s ready (gpu_id=%d, capacity=%d)", self.worker_id, self.s.gpu_id, self.p.worker.max_streams)

    def shutdown(self) -> None:
        if self._stop.is_set():
            return
        log.info("worker %s shutting down", self.worker_id)
        self.status = "stopping"
        self._heartbeat_once()
        self._stop.set()
        for cid in list(self.sessions):
            self.stop_camera(cid, publish=True)
        self.scheduler.stop()
        terminate_all()
        time.sleep(0.3)  # let final status events flush
        if self._loop:
            try:
                self._run_async(self.bus.stop(), timeout=8)
            except Exception:  # noqa: BLE001
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.models.close()

    # ------------------------------------------------------------ async plumbing
    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="worker-asyncio", daemon=True)
        self._loop_thread.start()

    def _run_async(self, coro, timeout: float = 15):
        assert self._loop
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _publish(self, topic: str, key: str, value: dict) -> bool:
        return self.bus.publish(topic, key, value)

    # ------------------------------------------------------------ session management
    def _profile_for(self, camera_id: str):
        s = self.sessions.get(camera_id)
        return None if s is None else (s.profile_name, s.profile)

    def _classes_for(self, camera_id: str) -> set[str] | None:
        s = self.sessions.get(camera_id)
        c = s.spec.detection_classes if s else None
        return {x.lower() for x in c} if c else None

    def _is_active(self, camera_id: str) -> bool:
        return camera_id in self.sessions

    def _status_event(self, camera_id: str, session_id: str | None, state: str, err: str | None = None,
                      command_id: str | None = None) -> None:
        ev = CameraStatusEvent(camera_id=camera_id, worker_id=self.worker_id, session_id=session_id, actual_state=state,  # type: ignore[arg-type]
                               error=err, command_id=command_id)
        self._publish(Topics.CAMERA_STATUS, camera_id, ev.model_dump(mode="json"))

    def start_camera(self, spec: CameraSpec, session_id: str | None = None, command_id: str | None = None) -> str | None:
        """Idempotent. Returns error string or None."""
        with self._lock:
            if self.status in ("draining", "stopping"):
                return f"worker is {self.status}; not accepting cameras"
            cur = self.sessions.get(spec.camera_id)
            if cur is not None:
                if session_id and cur.session_id == session_id:
                    return None  # duplicate command: no-op
                self.stop_camera(spec.camera_id, publish=False)
            if len(self.sessions) >= self.p.worker.max_streams:
                return f"worker at capacity ({self.p.worker.max_streams})"
            profile = self.mc.profiles.get(spec.model_profile)
            if profile is None:
                return f"unknown model_profile '{spec.model_profile}'"
            try:
                for ref in {profile.detection, *(getattr(profile, st) for st in ("ocr", "reid", "dino", "sam") if st in profile.stages)}:
                    if ref:
                        self.models.get(ref)
            except Exception as e:  # noqa: BLE001
                return f"model not ready for profile '{spec.model_profile}': {e}"
            fps = spec.target_fps or profile.fps or self.p.router.default_fps
            prio = spec.priority or profile.priority
            self.router.register(spec.camera_id, fps, prio)
            sess_holder: dict[str, Session] = {}

            def on_frame(fr: Frame) -> None:
                self.router.offer(fr)
                enc = sess_holder["s"].encoder
                if enc is not None:
                    enc.push(fr)

            def on_state(st: str, err: str | None) -> None:
                mapped = {"connecting": "starting", "starting": "starting"}.get(st, st)
                self._status_event(spec.camera_id, session_id, mapped, err)

            ing = CameraIngest(spec.camera_id, spec.source_url, self.worker_id, self.p.decode, on_frame, on_state,
                               session_id, spec.transport, self.realtime_synthetic)
            sess = Session(spec, session_id or "local", spec.model_profile, profile, ing)
            sess_holder["s"] = sess
            self.sessions[spec.camera_id] = sess
            ing.start()
            metrics.active_cameras.labels(self.worker_id).set(len(self.sessions))
            if spec.stream_enabled:
                self._set_stream_locked(sess, True)
            log.info("camera started: %r profile=%s fps=%.1f stream=%s", spec, spec.model_profile, fps, spec.stream_enabled,
                     extra={"camera_id": spec.camera_id})
            return None

    def _set_stream_locked(self, sess: Session, enabled: bool) -> None:
        if enabled and sess.encoder is None:
            enc = OverlayEncoder(sess.spec.camera_id, self.worker_id, self.p.encoder, sess.profile.overlay or self.p.overlay,
                                 self.sink_factory(sess.spec.camera_id), self.store.get,
                                 source_size=(self.p.decode.width, self.p.decode.height))
            sess.encoder = enc
            enc.start()
        elif not enabled and sess.encoder is not None:
            enc, sess.encoder = sess.encoder, None
            enc.stop()
        sess.stream_enabled = enabled and sess.encoder is not None

    def set_stream(self, camera_id: str, enabled: bool) -> str | None:
        with self._lock:
            s = self.sessions.get(camera_id)
            if s is None:
                return "camera not running on this worker"
            self._set_stream_locked(s, enabled)
            return None

    def stop_camera(self, camera_id: str, publish: bool = True, command_id: str | None = None) -> None:
        with self._lock:
            s = self.sessions.pop(camera_id, None)
        if s is None:
            if publish:
                self._status_event(camera_id, None, "stopped", command_id=command_id)
            return
        s.ingest.stop()  # also kills the decoder subprocess
        if s.encoder:
            s.encoder.stop()
        self.router.unregister(camera_id)
        self.processor.forget(camera_id)
        for g in (metrics.frame_queue_depth,):
            try:
                g.remove(self.worker_id, camera_id)
            except KeyError:
                pass
        metrics.active_cameras.labels(self.worker_id).set(len(self.sessions))
        if publish:
            self._status_event(camera_id, s.session_id, "stopped", command_id=command_id)
        log.info("camera stopped", extra={"camera_id": camera_id})

    def drain(self) -> None:
        self.status = "draining"
        self._heartbeat_once()

    # ------------------------------------------------------------ command interface
    async def _on_command(self, topic: str, key: str | None, value: dict) -> None:
        try:
            cmd = Command.model_validate(value)
        except Exception as e:  # noqa: BLE001
            log.error("invalid command ignored: %s", e)
            return
        if cmd.worker_id != self.worker_id or self._seen_cmds.seen(cmd.command_id):
            return
        err: str | None = None
        cid = cmd.camera_id
        try:
            if cmd.type == "start_camera" and cmd.camera:
                err = await asyncio.to_thread(self.start_camera, cmd.camera, cmd.session_id, cmd.command_id)
            elif cmd.type == "stop_camera" and cid:
                await asyncio.to_thread(self.stop_camera, cid, True, cmd.command_id)
            elif cmd.type == "update_camera" and cmd.camera:
                err = await asyncio.to_thread(self.start_camera, cmd.camera, cmd.session_id, cmd.command_id)
            elif cmd.type == "set_stream" and cid:
                err = await asyncio.to_thread(self.set_stream, cid, bool(cmd.params.get("enabled")))
            elif cmd.type == "drain_worker":
                self.drain()
            else:
                err = f"unsupported command {cmd.type}"
        except Exception as e:  # noqa: BLE001
            log.exception("command %s failed", cmd.type)
            err = f"{type(e).__name__}: {e}"
        if err and cid:
            self._status_event(cid, cmd.session_id, "error", err, cmd.command_id)
        elif cid and cmd.type in ("start_camera", "update_camera"):
            self._status_event(cid, cmd.session_id, "starting", None, cmd.command_id)  # ack; running follows from ingest

    # ------------------------------------------------------------ heartbeats
    def _heartbeat_loop(self) -> None:
        last_metrics = 0.0
        while not self._stop.wait(self.p.worker.heartbeat_interval_s):
            try:
                self._heartbeat_once()
                if time.monotonic() - last_metrics >= self.p.worker.metrics_interval_s:
                    last_metrics = time.monotonic()
                    self._metrics_once()
            except Exception:  # noqa: BLE001
                log.exception("heartbeat failed")

    def camera_state(self, sess: Session) -> CameraHeartbeat:
        i, e = sess.ingest, sess.encoder
        st = i.state if i.state in ("starting", "connecting", "running", "reconnecting", "stopped", "error") else "error"
        stats = self.router.stats().get(sess.spec.camera_id, {})
        if e and e.gave_up:  # publishing to MediaMTX failed for good (e.g. 401 publish password): surface it
            st = "error"
        age = i.last_frame_age()
        return CameraHeartbeat(
            camera_id=sess.spec.camera_id, worker_id=self.worker_id, session_id=sess.session_id, state=st,  # type: ignore[arg-type]
            stream_publishing=bool(e and e.publishing), decoder_fps=round(i.fps_meter.rate(), 2),
            inference_fps=round(self.processor.inference_fps(sess.spec.camera_id), 2),
            output_fps=round(e.fps_meter.rate(), 2) if e else 0.0, queue_depth=stats.get("depth", 0),
            dropped_frames=stats.get("dropped_stale", 0) + stats.get("dropped_full", 0),
            ffmpeg_restarts=i.restarts + (e.restarts if e else 0),
            last_frame_age_s=None if age is None else round(age, 2),
            last_error=i.last_error or (e.last_error if e else None), source_resolution=i.source_resolution,
            decoder=i.decoder, encoder=e.encoder_name if e else None)

    def worker_heartbeat(self) -> WorkerHeartbeat:
        g = self.gpu.sample()
        q = self.processor.latency.quantiles()
        cuda = cuda_status()
        caps = ff.detect_capabilities()
        gid = str(self.s.gpu_id)
        if g["util"] is not None:
            metrics.gpu_utilization_percent.labels(self.worker_id, gid).set(g["util"])
        if g["used"] is not None:
            metrics.gpu_memory_used_bytes.labels(self.worker_id, gid).set(g["used"])
            metrics.gpu_memory_total_bytes.labels(self.worker_id, gid).set(g["total"])
        return WorkerHeartbeat(
            worker_id=self.worker_id, hostname=self._hostname, gpu_id=self.s.gpu_id, status=self.status,  # type: ignore[arg-type]
            capacity_limit=self.p.worker.max_streams, active_cameras=sorted(self.sessions),
            queue_depths=self.router.depths(), gpu_utilization_pct=g["util"], gpu_memory_used_bytes=g["used"],
            gpu_memory_total_bytes=g["total"], inference_latency_p50_s=q["p50"], inference_latency_p95_s=q["p95"],
            inference_latency_p99_s=q["p99"], models_ready=self.models.readiness(), cuda_available=cuda["cuda"],
            inference_backend=self.models.backend_label(),
            decoders={"nvdec_possible": caps.nvdec_possible},
            encoders={k: v for k, v in caps.nvenc_usable.items()} | {"libx264": "libx264" in caps.encoders})

    def _heartbeat_once(self) -> None:
        self._publish(Topics.WORKER_HEARTBEAT, self.worker_id, self.worker_heartbeat().model_dump(mode="json"))
        for sess in list(self.sessions.values()):
            self._publish(Topics.CAMERA_HEARTBEAT, sess.spec.camera_id, self.camera_state(sess).model_dump(mode="json"))

    def _metrics_once(self) -> None:
        q = self.processor.latency.quantiles()
        dec = sum(s.ingest.fps_meter.rate() for s in self.sessions.values())
        out = sum(s.encoder.fps_meter.rate() for s in self.sessions.values() if s.encoder)
        ev = PipelineMetricsEvent(worker_id=self.worker_id, input_fps=round(dec, 2),
                                  inference_fps=round(self.processor.total_rate.rate(), 2), output_fps=round(out, 2),
                                  dropped_frames_total=sum(v["dropped_stale"] + v["dropped_full"] for v in self.router.stats().values()),
                                  latency_p50_s=q["p50"], latency_p95_s=q["p95"], latency_p99_s=q["p99"])
        self._publish(Topics.PIPELINE_METRICS, self.worker_id, ev.model_dump(mode="json"))


# ---------------------------------------------------------------------- entrypoint
def load_cameras_yaml(path: str) -> list[CameraSpec]:
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    return [CameraSpec(**_cam(c)) for c in raw.get("cameras", []) if c.get("enabled", True)]


def _cam(c: dict) -> dict:
    url = os.path.expandvars(c.get("source_url") or c.get("rtsp_url", ""))
    return {"camera_id": c["camera_id"], "source_url": url, "model_profile": c.get("model_profile", "default"),
            "stream_enabled": c.get("stream_enabled", True), "target_fps": c.get("target_fps"),
            "priority": c.get("priority", 0), "transport": c.get("transport", "tcp"),
            "detection_classes": c.get("detection_classes")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VMS GPU worker")
    ap.add_argument("--standalone", action="store_true", help="run cameras from cameras.yaml without the control plane")
    ap.add_argument("--sink-dir", help="standalone: write encoded output to files in this dir instead of MediaMTX")
    a = ap.parse_args(argv)
    s = get_settings()
    if a.standalone:
        s = s.model_copy(update={"standalone": True})
    setup_logging(s.log_level)
    sink_factory = (lambda cid: str(Path(a.sink_dir) / f"{cid}.ts")) if a.sink_dir else None
    if a.sink_dir:
        Path(a.sink_dir).mkdir(parents=True, exist_ok=True)
    bus = NullBus() if a.standalone and s.bus == "memory" else None
    w = GpuWorker(s, load_pipeline_config(s.pipeline_config), load_models_config(s.models_config), bus, sink_factory)
    try:
        w.start()
    except Exception as e:  # noqa: BLE001
        log.error("worker failed to start: %s", e)
        return 2
    if a.standalone:
        for i, spec in enumerate(load_cameras_yaml(s.cameras_config)):
            err = w.start_camera(spec, f"standalone-{spec.camera_id}")
            if err:
                log.error("camera %s not started: %s", spec.camera_id, err)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    w.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
