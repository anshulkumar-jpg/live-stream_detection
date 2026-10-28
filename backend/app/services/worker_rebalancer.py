"""Reassignment of cameras from unhealthy/draining workers, load rebalancing and the optional autoscaler hook.

All moves use CameraManager.migrate / restart (stop -> ack -> start with a new session): never a blind move."""
from __future__ import annotations

import logging
import time
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Camera, PipelineSession, Worker, now
from app.db.repositories import SessionRepo
from app.services import admission_control as ac
from app.services.camera_manager import CameraManager
from app.services.worker_scheduler import WorkerScheduler

log = logging.getLogger(__name__)
MIGRATION_COOLDOWN_S = 60.0


class WorkerRebalancer:
    def __init__(self, sf: async_sessionmaker[AsyncSession], s: Settings, cams: CameraManager, scheduler: WorkerScheduler,
                 scale_hook: Callable[[str, dict], None] | None = None):
        self.sf, self.s, self.cams, self.scheduler, self.scale_hook = sf, s, cams, scheduler, scale_hook
        self._last_move: dict[str, float] = {}

    # ---- 1. dead workers: their sessions are lost; re-place cameras elsewhere
    async def reassign_from_unhealthy(self) -> int:
        moved = 0
        async with self.sf() as db:
            workers = {w.worker_id: w for w in (await db.execute(select(Worker))).scalars()}
            cams = [c for c in (await db.execute(select(Camera).where(Camera.worker_id.is_not(None),
                                                                      Camera.desired_state == "running"))).scalars()]
            dead = [(c.camera_id, c.worker_id, c.current_session_id) for c in cams
                    if c.worker_id not in workers or not ac.is_healthy(workers[c.worker_id], self.s)]
            for cid, wid, sid in dead:
                cam = await db.get(Camera, cid)
                if sid:
                    await SessionRepo(db).close(sid, "lost", f"worker {wid} unhealthy")
                cam.worker_id, cam.current_session_id, cam.actual_state = None, None, "pending"
                cam.last_error = f"worker {wid} unhealthy; reassigning"
            await db.commit()
        for cid, wid, _ in dead:
            ok, reason = await self.cams.assign_pending(cid, exclude={wid})
            log.warning("camera %s moved off unhealthy worker %s: %s (%s)", cid, wid, "started" if ok else "pending", reason)
            moved += int(ok)
        return moved

    # ---- 2. unplaced cameras
    async def place_pending(self) -> int:
        async with self.sf() as db:
            ids = [c.camera_id for c in (await db.execute(select(Camera).where(
                Camera.worker_id.is_(None), Camera.desired_state == "running", Camera.enabled.is_(True)))).scalars()]
        n = 0
        for cid in ids:
            if cid in self.cams.migrating:
                continue
            ok, _ = await self.cams.assign_pending(cid)
            n += int(ok)
        return n

    # ---- 3. restart sessions lost by a worker restart (worker healthy but not running the camera)
    async def heal_missing_sessions(self, grace_s: float = 15.0) -> int:
        healed = 0
        async with self.sf() as db:
            todo = []
            for c in (await db.execute(select(Camera).where(Camera.worker_id.is_not(None),
                                                            Camera.desired_state == "running"))).scalars():
                w = await db.get(Worker, c.worker_id)
                if not w or not ac.is_healthy(w, self.s) or c.camera_id in self.cams.migrating:
                    continue
                active = set((w.details or {}).get("active_cameras", []))
                hb_after_change = w.last_heartbeat and (w.last_heartbeat - c.updated_at).total_seconds() > grace_s
                if c.camera_id not in active and hb_after_change:
                    todo.append(c.camera_id)
        for cid in todo:
            log.warning("camera %s missing on its worker; starting a fresh session", cid)
            ok, _ = await self.cams.restart_on_same_or_other(cid)
            healed += int(ok)
        return healed

    # ---- 4. drain + load rebalance
    async def rebalance_once(self) -> int:
        async with self.sf() as db:
            workers = list((await db.execute(select(Worker))).scalars())
            counts = await self.scheduler.assigned_counts(db)
            cams = list((await db.execute(select(Camera).where(Camera.desired_state == "running",
                                                               Camera.worker_id.is_not(None)))).scalars())
        healthy = [w for w in workers if ac.is_healthy(w, self.s)]
        moves = 0
        t = time.monotonic()

        def pick_target(src: str) -> Worker | None:
            best = None
            for w in healthy:
                if w.worker_id == src or not ac.check(w, counts.get(w.worker_id, 0), self.s).ok:
                    continue
                if best is None or counts.get(w.worker_id, 0) < counts.get(best.worker_id, 0):
                    best = w
            return best

        def movable(wid: str) -> list[Camera]:
            return sorted([c for c in cams if c.worker_id == wid and t - self._last_move.get(c.camera_id, -1e9) > MIGRATION_COOLDOWN_S
                           and c.camera_id not in self.cams.migrating], key=lambda c: c.priority)

        # drain: move everything off draining workers (as capacity allows)
        for w in healthy:
            if w.status == "draining":
                for c in movable(w.worker_id):
                    tgt = pick_target(w.worker_id)
                    if tgt is None:
                        break
                    ok, _ = await self.cams.migrate(c.camera_id, tgt.worker_id)
                    self._last_move[c.camera_id] = t
                    counts[tgt.worker_id] = counts.get(tgt.worker_id, 0) + 1
                    counts[w.worker_id] = max(0, counts.get(w.worker_id, 1) - 1)
                    moves += int(ok)
        # pressure: one move per overloaded worker per pass
        for w in healthy:
            if w.status != "ready":
                continue
            d = w.details or {}
            used, total = d.get("gpu_memory_used_bytes"), d.get("gpu_memory_total_bytes")
            hot = bool(used and total and 100.0 * used / total >= self.s.gpu_mem_high_pct) \
                or sum((d.get("queue_depths") or {}).values()) >= self.s.queue_backlog_high \
                or (d.get("inference_latency_p95_s") or 0) >= self.s.p95_latency_high_s
            skew = counts.get(w.worker_id, 0) - min([counts.get(x.worker_id, 0) for x in healthy] or [0]) >= 3
            if hot or skew:
                mv, tgt = movable(w.worker_id), pick_target(w.worker_id)
                if mv and tgt and counts.get(w.worker_id, 0) > counts.get(tgt.worker_id, 0) + (0 if hot else 1):
                    ok, _ = await self.cams.migrate(mv[0].camera_id, tgt.worker_id)
                    self._last_move[mv[0].camera_id] = t
                    moves += int(ok)
        return moves

    # ---- 5. autoscaler hook (disabled by default)
    async def evaluate_autoscale(self) -> dict | None:
        if not self.s.autoscaler_enabled:
            return None
        async with self.sf() as db:
            workers = list((await db.execute(select(Worker))).scalars())
            counts = await self.scheduler.assigned_counts(db)
        healthy = [w for w in workers if ac.is_healthy(w, self.s)]
        cap = sum(min(w.capacity_limit, self.s.max_streams_per_worker) for w in healthy)
        used = sum(counts.values())
        rec = {"healthy_workers": len(healthy), "streams": used, "capacity": cap,
               "action": "scale_up" if cap and used / cap > 0.85 else ("scale_down" if len(healthy) > 1 and cap and used / cap < 0.3 else "none")}
        if rec["action"] != "none" and self.scale_hook:
            self.scale_hook(rec["action"], rec)  # the hook (K8s/ASG/compose) is deployment-specific
        return rec
