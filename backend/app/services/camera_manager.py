"""Camera lifecycle: desired state in PostgreSQL, assignment, commands. Actual state arrives from workers."""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.security import SecretBox, ValidationError, redact_url, validate_source_url
from app.db.models import Camera, ModelProfile, PipelineSession, now
from app.db.repositories import CameraRepo, SessionRepo
from app.schemas.camera import CameraCreate, CameraUpdate
from app.schemas.worker import CameraSpec
from app.services.command_service import CommandService
from app.services.worker_scheduler import WorkerScheduler

log = logging.getLogger(__name__)


class ConflictError(Exception):
    pass


class NotFoundError(Exception):
    pass


class CameraManager:
    def __init__(self, sf: async_sessionmaker[AsyncSession], s: Settings, box: SecretBox, scheduler: WorkerScheduler,
                 commands: CommandService):
        self.sf, self.s, self.box, self.scheduler, self.commands = sf, s, box, scheduler, commands
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.migrating: set[str] = set()

    # ---- helpers
    def spec_for(self, cam: Camera) -> CameraSpec:
        return CameraSpec(camera_id=cam.camera_id, source_url=self.box.decrypt(cam.rtsp_url_encrypted),
                          model_profile=cam.model_profile,
                          stream_enabled=cam.stream_enabled and self.s.stream_policy == "always",
                          target_fps=cam.target_fps, priority=cam.priority, transport=cam.transport)

    def _validate_url(self, url: str) -> None:
        validate_source_url(url, self.s.camera_allowed_cidrs, self.s.allow_synthetic_sources)

    async def _require(self, db: AsyncSession, camera_id: str) -> Camera:
        cam = await CameraRepo(db).get(camera_id)
        if cam is None:
            raise NotFoundError(camera_id)
        return cam

    async def _check_profile(self, db: AsyncSession, profile: str) -> None:
        p = await db.get(ModelProfile, profile)
        if p is None or not p.enabled:
            raise ValidationError(f"unknown or disabled model_profile '{profile}'")

    # ---- CRUD
    async def create(self, body: CameraCreate) -> Camera:
        self._validate_url(body.rtsp_url)
        async with self.sf() as db:
            if await CameraRepo(db).get(body.camera_id):
                raise ConflictError(f"camera '{body.camera_id}' already exists")
            await self._check_profile(db, body.model_profile)
            cam = Camera(camera_id=body.camera_id, name=body.name, rtsp_url_encrypted=self.box.encrypt(body.rtsp_url),
                         source_display=redact_url(body.rtsp_url), model_profile=body.model_profile,
                         enabled=body.enabled, desired_state="stopped", actual_state="stopped",
                         stream_enabled=body.stream_enabled, target_fps=body.target_fps, priority=body.priority,
                         transport=body.transport)
            db.add(cam)
            await db.commit()
        if body.autostart and body.enabled:
            return await self.start(body.camera_id)
        async with self.sf() as db:
            return await self._require(db, body.camera_id)

    async def update(self, camera_id: str, body: CameraUpdate) -> Camera:
        changes = body.model_dump(exclude_unset=True)
        async with self._locks[camera_id]:
            async with self.sf() as db:
                cam = await self._require(db, camera_id)
                restart = False
                if "rtsp_url" in changes and changes["rtsp_url"]:
                    self._validate_url(changes["rtsp_url"])
                    cam.rtsp_url_encrypted = self.box.encrypt(changes["rtsp_url"])
                    cam.source_display = redact_url(changes["rtsp_url"])
                    restart = True
                if "model_profile" in changes and changes["model_profile"] != cam.model_profile:
                    await self._check_profile(db, changes["model_profile"])
                    cam.model_profile, restart = changes["model_profile"], True
                for f in ("target_fps", "priority", "transport"):
                    if f in changes and changes[f] != getattr(cam, f):
                        setattr(cam, f, changes[f]), None
                        restart = True
                if "name" in changes and changes["name"]:
                    cam.name = changes["name"]
                stream_changed = "stream_enabled" in changes and changes["stream_enabled"] != cam.stream_enabled
                if stream_changed:
                    cam.stream_enabled = changes["stream_enabled"]
                enabled_changed = "enabled" in changes and changes["enabled"] != cam.enabled
                if enabled_changed:
                    cam.enabled = changes["enabled"]
                await db.commit()
                running = cam.desired_state == "running" and cam.worker_id is not None
                wid, enabled = cam.worker_id, cam.enabled
        if enabled_changed and not enabled and running:
            return await self.stop(camera_id)
        if running and restart:
            await self._restart_session(camera_id)
        elif running and stream_changed and wid:
            async with self.sf() as db:
                cam = await self._require(db, camera_id)
            await self.commands.set_stream(wid, camera_id, cam.stream_enabled)
        async with self.sf() as db:
            return await self._require(db, camera_id)

    async def delete(self, camera_id: str) -> None:
        async with self.sf() as db:
            await self._require(db, camera_id)
        await self.stop(camera_id)
        async with self.sf() as db:
            cam = await self._require(db, camera_id)
            await db.delete(cam)
            await db.commit()
        self._locks.pop(camera_id, None)

    # ---- lifecycle
    async def start(self, camera_id: str) -> Camera:
        async with self._locks[camera_id]:
            async with self.sf() as db:
                cam = await self._require(db, camera_id)
                if not cam.enabled:
                    raise ConflictError("camera is disabled")
                cam.desired_state = "running"
                await db.commit()
            await self._assign_and_start(camera_id)
        async with self.sf() as db:
            return await self._require(db, camera_id)

    async def stop(self, camera_id: str) -> Camera:
        async with self._locks[camera_id]:
            async with self.sf() as db:
                cam = await self._require(db, camera_id)
                wid, sid = cam.worker_id, cam.current_session_id
                cam.desired_state = "stopped"
                cam.actual_state = "stopped"
                cam.worker_id = None
                cam.current_session_id = None
                if sid:
                    await SessionRepo(db).close(sid, "stopped")
                await db.commit()
            if wid:
                await self.commands.stop_camera(wid, camera_id)
        async with self.sf() as db:
            return await self._require(db, camera_id)

    async def _restart_session(self, camera_id: str) -> None:
        async with self._locks[camera_id]:
            await self._assign_and_start(camera_id, force_new_session=True)

    async def _assign_and_start(self, camera_id: str, force_new_session: bool = False, exclude: set[str] = frozenset(),  # type: ignore[assignment]
                                target_worker: str | None = None) -> tuple[bool, str]:
        """Caller holds the camera lock. Returns (started, reason)."""
        async with self.sf() as db:
            cam = await self._require(db, camera_id)
            if cam.worker_id and not force_new_session and target_worker is None:
                return True, "already assigned"
            old_wid = cam.worker_id
            if target_worker:
                from app.db.models import Worker

                worker = await db.get(Worker, target_worker)
                reason = "ok" if worker else "unknown target worker"
            else:
                worker, reason = await self.scheduler.pick_worker(db, exclude=set(exclude))
                if force_new_session and old_wid and not exclude:
                    # restart in place if the current worker is still admissible, else pick another
                    from app.db.models import Worker

                    cur = await db.get(Worker, old_wid)
                    from app.services import admission_control as ac

                    counts = await self.scheduler.assigned_counts(db)
                    if cur and ac.is_healthy(cur, self.s) and counts.get(old_wid, 0) <= cur.capacity_limit:
                        worker = cur
            if worker is None:
                cam.actual_state, cam.last_error = "pending", f"no worker admitted: {reason}"
                cam.worker_id, cam.current_session_id = None, None
                await db.commit()
                return False, cam.last_error
            if cam.current_session_id:
                await SessionRepo(db).close(cam.current_session_id, "stopped")
            sid = str(uuid.uuid4())
            db.add(PipelineSession(session_id=sid, camera_id=camera_id, worker_id=worker.worker_id, status="starting"))
            cam.worker_id, cam.current_session_id = worker.worker_id, sid
            cam.actual_state, cam.last_error = "starting", None
            await db.commit()
            spec = self.spec_for(cam)
            wid = worker.worker_id
        if old_wid and old_wid != wid:
            await self.commands.stop_camera(old_wid, camera_id)
        err = await self.commands.start_camera(wid, spec, sid)
        if err:
            async with self.sf() as db:
                cam = await self._require(db, camera_id)
                cam.actual_state, cam.last_error = "error", err
                await SessionRepo(db).close(sid, "error", err)
                await db.commit()
            return False, err
        return True, "ok"

    # used by reconcile/rebalancer (they take the lock themselves)
    async def assign_pending(self, camera_id: str, exclude: set[str] = frozenset()) -> tuple[bool, str]:  # type: ignore[assignment]
        async with self._locks[camera_id]:
            return await self._assign_and_start(camera_id, exclude=exclude)

    async def restart_on_same_or_other(self, camera_id: str) -> tuple[bool, str]:
        async with self._locks[camera_id]:
            return await self._assign_and_start(camera_id, force_new_session=True)

    async def migrate(self, camera_id: str, target_worker: str) -> tuple[bool, str]:
        """Controlled handoff: stop on source (await ack) -> reassign -> start on target with new session."""
        async with self._locks[camera_id]:
            async with self.sf() as db:
                cam = await self._require(db, camera_id)
                src, sid = cam.worker_id, cam.current_session_id
                if cam.desired_state != "running" or not src or src == target_worker:
                    return False, "not migratable"
            self.migrating.add(camera_id)
            try:
                err = await self.commands.stop_camera(src, camera_id, wait_ack=True)
                async with self.sf() as db:
                    cam = await self._require(db, camera_id)
                    if sid:
                        await SessionRepo(db).close(sid, "migrated", err)
                    cam.current_session_id = None
                    cam.worker_id = None  # never assigned to two workers at once
                    cam.actual_state = "starting"
                    await db.commit()
                ok, reason = await self._assign_and_start(camera_id, target_worker=target_worker)
                if not ok:
                    log.warning("migration of %s to %s failed (%s); scheduler will re-place it", camera_id, target_worker, reason)
                return ok, reason
            finally:
                self.migrating.discard(camera_id)
