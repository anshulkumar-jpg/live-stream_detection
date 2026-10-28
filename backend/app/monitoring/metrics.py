"""Prometheus metrics. All metrics are *process-local*: each worker/API process exposes its own.
Cluster-wide totals must be computed in PromQL (sum by ...) over per-worker series, never assumed."""
from __future__ import annotations

import threading
from collections import deque

from prometheus_client import Counter, Gauge, Histogram

LAT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

active_cameras = Gauge("active_cameras", "Cameras with a running session on this worker", ["worker_id"])
camera_connection_status = Gauge("camera_connection_status", "1=connected/running, 0=not", ["worker_id", "camera_id"])
worker_heartbeat_age_seconds = Gauge("worker_heartbeat_age_seconds", "Age of last heartbeat as seen by control plane", ["worker_id"])
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization", ["worker_id", "gpu_id"])
gpu_memory_used_bytes = Gauge("gpu_memory_used_bytes", "GPU memory used", ["worker_id", "gpu_id"])
gpu_memory_total_bytes = Gauge("gpu_memory_total_bytes", "GPU memory total", ["worker_id", "gpu_id"])
decoder_fps = Gauge("decoder_fps", "Decoded frames per second", ["worker_id", "camera_id"])
inference_fps = Gauge("inference_fps", "Inferred frames per second", ["worker_id", "camera_id"])
output_fps = Gauge("output_fps", "Encoded output frames per second", ["worker_id", "camera_id"])
inference_latency_seconds = Histogram("inference_latency_seconds", "Model inference time per batch stage",
                                      ["worker_id", "model"], buckets=LAT_BUCKETS)
end_to_end_latency_seconds = Histogram("end_to_end_latency_seconds", "Capture -> result latency",
                                       ["worker_id", "camera_id"], buckets=LAT_BUCKETS)
frame_queue_depth = Gauge("frame_queue_depth", "Frames waiting in per-camera router queue", ["worker_id", "camera_id"])
dropped_frames_total = Counter("dropped_frames_total", "Frames dropped", ["worker_id", "camera_id", "reason"])
ffmpeg_restart_total = Counter("ffmpeg_restart_total", "FFmpeg process restarts", ["worker_id", "camera_id", "role"])
detections_total = Counter("detections_total", "Detections produced", ["worker_id", "camera_id", "class_name"])
kafka_publish_errors_total = Counter("kafka_publish_errors_total", "Kafka publish failures/drops", ["component", "reason"])
stream_publish_status = Gauge("stream_publish_status", "1 if encoder is publishing to MediaMTX", ["worker_id", "camera_id"])
model_ready = Gauge("model_ready", "1 if model loaded and ready", ["worker_id", "model"])
frames_ingested_total = Counter("frames_ingested_total", "Frames decoded", ["worker_id", "camera_id"])
api_commands_total = Counter("control_plane_commands_total", "Commands published by control plane", ["type", "result"])


class RollingQuantiles:
    """Thread-safe sliding window for P50/P95/P99 reporting (heartbeats / UI)."""

    def __init__(self, size: int = 1000):
        self._d: deque[float] = deque(maxlen=size)
        self._l = threading.Lock()

    def add(self, v: float) -> None:
        with self._l:
            self._d.append(v)

    def quantiles(self) -> dict[str, float | None]:
        with self._l:
            data = sorted(self._d)
        if not data:
            return {"p50": None, "p95": None, "p99": None}

        def q(p: float) -> float:
            return data[min(len(data) - 1, int(p * len(data)))]

        return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99)}


class RateMeter:
    """Events/sec over a short sliding window."""

    def __init__(self, window_s: float = 5.0):
        import time

        self._t = time
        self.window = window_s
        self._d: deque[float] = deque()
        self._l = threading.Lock()

    def mark(self, n: int = 1) -> None:
        now = self._t.monotonic()
        with self._l:
            for _ in range(n):
                self._d.append(now)
            self._trim(now)

    def _trim(self, now: float) -> None:
        while self._d and now - self._d[0] > self.window:
            self._d.popleft()

    def rate(self) -> float:
        now = self._t.monotonic()
        with self._l:
            self._trim(now)
            return len(self._d) / self.window
