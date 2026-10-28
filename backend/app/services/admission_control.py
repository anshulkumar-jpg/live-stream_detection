"""Admission control: may this worker take one more stream right now?"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.config import Settings
from app.db.models import Worker


@dataclass
class Admission:
    ok: bool
    reason: str = "ok"


def heartbeat_age(w: Worker, now: datetime | None = None) -> float | None:
    if w.last_heartbeat is None:
        return None
    return ((now or datetime.now(timezone.utc)) - w.last_heartbeat).total_seconds()


def is_healthy(w: Worker, s: Settings, now: datetime | None = None) -> bool:
    age = heartbeat_age(w, now)
    return age is not None and age <= s.worker_heartbeat_timeout_s and w.status in ("ready", "degraded", "draining")


def check(w: Worker, assigned_count: int, s: Settings) -> Admission:
    """`assigned_count` comes from the DB (cameras assigned to the worker), NOT from the last heartbeat,
    so back-to-back admissions between heartbeats cannot overshoot capacity."""
    if not is_healthy(w, s):
        return Admission(False, "worker unhealthy or heartbeat stale")
    if w.status != "ready":
        return Admission(False, f"worker status is {w.status}")
    cap = min(w.capacity_limit, s.max_streams_per_worker)
    if assigned_count >= cap:
        return Admission(False, f"at capacity ({assigned_count}/{cap})")
    d = w.details or {}
    used, total = d.get("gpu_memory_used_bytes"), d.get("gpu_memory_total_bytes")
    if used and total and 100.0 * used / total >= s.gpu_mem_high_pct:
        return Admission(False, f"GPU memory {100.0 * used / total:.0f}% >= {s.gpu_mem_high_pct:.0f}%")
    if sum((d.get("queue_depths") or {}).values()) >= s.queue_backlog_high:
        return Admission(False, "queue backlog above threshold")
    p95 = d.get("inference_latency_p95_s")
    if p95 is not None and p95 >= s.p95_latency_high_s:
        return Admission(False, f"p95 latency {p95:.2f}s above threshold")
    ready = d.get("models_ready") or {}
    if ready and not any(ready.values()):
        return Admission(False, "no models ready")
    return Admission(True)
