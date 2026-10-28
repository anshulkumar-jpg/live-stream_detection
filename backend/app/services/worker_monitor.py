"""Consumes worker/camera telemetry from Kafka, maintains actual state in PostgreSQL, detects dead
workers and fences duplicate sessions."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Camera, PipelineSession, Worker, now
from app.messaging.topics import Topics
from app.monitoring import metrics
from app.services import admission_control as ac
from app.services.command_service import CommandService
from app.services.event_service import Hub

log = logging.getLogger(__name__)
_STATE_MAP = {"starting": "starting", "connecting": "starting", "running": "running", "reconnecting": "reconnecting",
              "stopped": "stopped", "error": "error"}


class WorkerMonitor:
    def __init__(self, sf: async_sessionmaker[AsyncSession], s: Settings, commands: CommandService, hub: Hub):
        self.sf, self.s, self.commands, self.hub = sf, s, commands, hub
        self.pipeline_metrics: dict[str, dict] = {}  # latest per worker (in-memory, rebuilt from the topic)
        self._last_fence: dict[tuple[str, str], float] = {}

    async def handle(self, topic: str, key: str | None, v: dict) -> None:
        if topic == Topics.WORKER_HEARTBEAT:
            await self.on_worker_heartbeat(v)
        elif topic == Topics.CAMERA_HEARTBEAT:
            await self.on_camera_heartbeat(v)
        elif topic == Topics.CAMERA_STATUS:
            await self.on_camera_status(v)
        elif topic == Topics.PIPELINE_METRICS:
            self.pipeline_metrics[v.get("worker_id", "?")] = v

    # ------------------------------------------------------------------
    async def on_worker_heartbeat(self, hb: dict) -> None:
        wid = hb["worker_id"]
        active = set(hb.get("active_cameras", []))
        async with self.sf() as db:
            w = await db.get(Worker, wid)
            if w is None:
                w = Worker(worker_id=wid)
                db.add(w)
            w.hostname, w.gpu_id = hb.get("hostname", ""), hb.get("gpu_id", 0)
            w.status = hb.get("status", "ready")
            w.last_heartbeat = now()  # receipt time: immune to worker clock skew
            w.active_camera_count = len(active)
            w.capacity_limit = hb.get("capacity_limit", 16)
            w.details = hb
            # fencing: a worker running a camera that is NOT assigned to it must stop it (prevents duplicate processing)
            assigned = {c.camera_id: c for c in (await db.execute(select(Camera).where(Camera.worker_id == wid))).scalars()
                        if c.desired_state == "running"}
            rogue = [c for c in active if c not in assigned and c not in self._migrating_ids()]
            await db.commit()
        metrics.worker_heartbeat_age_seconds.labels(wid).set(0)
        t = datetime.now().timestamp()
        for cid in rogue:
            if t - self._last_fence.get((wid, cid), 0) > 10:
                self._last_fence[(wid, cid)] = t
                log.warning("fencing: worker %s runs unassigned camera %s -> stop", wid, cid)
                await self.commands.stop_camera(wid, cid)
        await self.hub.broadcast({"type": "worker", "worker_id": wid, "status": hb.get("status"),
                                  "cameras": len(active)})

    def _migrating_ids(self) -> set[str]:
        return getattr(self, "migrating", set())

    async def on_camera_heartbeat(self, hb: dict) -> None:
        cid = hb["camera_id"]
        async with self.sf() as db:
            cam = await db.get(Camera, cid)
            if cam is None or cam.worker_id != hb["worker_id"] or (hb.get("session_id") and cam.current_session_id
                                                                    and hb["session_id"] != cam.current_session_id):
                return  # stale / foreign heartbeat: ignore (fencing happens via worker heartbeat)
            cam.last_heartbeat = now()
            cam.actual_state = _STATE_MAP.get(hb.get("state", ""), cam.actual_state)
            cam.last_error = hb.get("last_error") if cam.actual_state != "running" else None
            cam.heartbeat_detail = hb
            if cam.actual_state == "running" and cam.current_session_id:
                ps = await db.get(PipelineSession, cam.current_session_id)
                if ps and ps.status == "starting":
                    ps.status = "running"
            await db.commit()
        await self.hub.broadcast({"type": "camera", "camera_id": cid, "state": cam.actual_state,
                                  "decoder_fps": hb.get("decoder_fps"), "inference_fps": hb.get("inference_fps"),
                                  "queue_depth": hb.get("queue_depth"), "publishing": hb.get("stream_publishing")})

    async def on_camera_status(self, ev: dict) -> None:
        self.commands.on_ack(ev.get("command_id"), ev["actual_state"], ev.get("error"))
        cid = ev["camera_id"]
        async with self.sf() as db:
            cam = await db.get(Camera, cid)
            if cam is None or cam.worker_id != ev["worker_id"]:
                return
            if ev.get("session_id") and cam.current_session_id and ev["session_id"] != cam.current_session_id:
                return
            if ev["actual_state"] == "stopped" and cam.desired_state == "running":
                cam.actual_state = "stopped"  # worker stopped it unexpectedly; reconcile will restart
            else:
                cam.actual_state = ev["actual_state"]
            if ev.get("error"):
                cam.last_error = ev["error"]
            if cam.current_session_id and ev["actual_state"] in ("running", "error"):
                ps = await db.get(PipelineSession, cam.current_session_id)
                if ps and ps.stop_time is None:
                    ps.status = "running" if ev["actual_state"] == "running" else "error"
                    ps.error = ev.get("error") if ev["actual_state"] == "error" else None
            await db.commit()
        await self.hub.broadcast({"type": "camera_status", "camera_id": cid, "state": ev["actual_state"],
                                  "error": ev.get("error")})

    # ------------------------------------------------------------------
    async def sweep(self) -> list[str]:
        """Mark stale workers offline, refresh age gauges. Returns ids newly marked offline."""
        newly = []
        async with self.sf() as db:
            for w in (await db.execute(select(Worker))).scalars():
                age = ac.heartbeat_age(w)
                if age is not None:
                    metrics.worker_heartbeat_age_seconds.labels(w.worker_id).set(age)
                if w.status != "offline" and (age is None or age > self.s.worker_heartbeat_timeout_s):
                    w.status = "offline"
                    newly.append(w.worker_id)
                    log.error("worker %s heartbeat lost (age=%s)", w.worker_id, age)
            await db.commit()
        return newly
