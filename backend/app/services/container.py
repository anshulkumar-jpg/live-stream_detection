"""Dependency container + background control loops for the control plane."""
from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings, load_models_config
from app.core.security import LoginThrottle, SecretBox, TokenBucketLimiter, hash_password, verify_password
from app.db.models import Base, ModelProfile
from app.db.session import make_engine, make_session_factory
from app.messaging.kafka_producer import EventBus, ensure_topics, make_bus
from app.messaging.topics import Topics
from app.services.camera_manager import CameraManager
from app.services.command_service import CommandService
from app.services.event_service import EventService, Hub
from app.services.stream_service import StreamService
from app.services.worker_monitor import WorkerMonitor
from app.services.worker_rebalancer import WorkerRebalancer
from app.services.worker_scheduler import WorkerScheduler

log = logging.getLogger(__name__)


class Container:
    def __init__(self, s: Settings, bus: EventBus | None = None, create_schema: bool = False):
        self.s = s
        self.create_schema = create_schema  # tests/dev only; production uses `alembic upgrade head`
        self.engine: AsyncEngine = make_engine(s)
        self.sf = make_session_factory(self.engine)
        self.bus = bus or make_bus(s, "-api")
        self.box = SecretBox(s.fernet_key)
        self.hub = Hub()
        self.commands = CommandService(self.bus, s)
        self.scheduler = WorkerScheduler(s)
        self.cams = CameraManager(self.sf, s, self.box, self.scheduler, self.commands)
        self.monitor = WorkerMonitor(self.sf, s, self.commands, self.hub)
        self.monitor.migrating = self.cams.migrating  # type: ignore[attr-defined]
        self.events = EventService(self.sf, self.hub, retention_days=s.event_retention_days)
        self.streams = StreamService(s, self.commands)
        self.rebalancer = WorkerRebalancer(self.sf, s, self.cams, self.scheduler)
        self.limiter = TokenBucketLimiter(s.rate_limit_per_sec, s.rate_limit_burst)
        self._tasks: list[asyncio.Task] = []
        self.users: dict[str, tuple[str, str]] = {}  # username -> (bcrypt hash, role)
        if s.auth_enabled:
            if s.admin_password:
                self.users[s.admin_username] = (hash_password(s.admin_password), "admin")
            if s.viewer_password:
                self.users[s.viewer_username] = (hash_password(s.viewer_password), "viewer")
        self.last_loop_ok: float = 0.0
        self._dummy_hash = hash_password("not-a-real-password") if s.auth_enabled else ""
        self.login_throttle = LoginThrottle()

    async def authenticate(self, username: str, password: str) -> str | None:
        """bcrypt runs in a worker thread (it would block the event loop); unknown users burn the same time as
        wrong passwords, so response timing does not reveal which usernames exist."""
        u = self.users.get(username)
        ok = await asyncio.to_thread(verify_password, password, u[0] if u else self._dummy_hash)
        return u[1] if u and ok else None

    async def seed_profiles(self) -> None:
        mc = load_models_config(self.s.models_config)
        async with self.sf() as db:
            for pid, p in mc.profiles.items():
                row = await db.get(ModelProfile, pid)
                cfg = p.model_dump()
                if row is None:
                    db.add(ModelProfile(profile_id=pid, config=cfg, enabled_stages=list(p.stages)))
                else:
                    row.config, row.enabled_stages = cfg, list(p.stages)
            await db.commit()

    async def start(self) -> None:
        if self.create_schema:
            async with self.engine.begin() as c:
                await c.run_sync(Base.metadata.create_all)
        await self.seed_profiles()
        await self.bus.start()
        if self.s.bus == "kafka":
            try:
                await ensure_topics(self.s)
            except Exception as e:  # noqa: BLE001 - broker may not be up yet; consumers/producers retry
                log.warning("topic creation skipped (%s)", e)
        await self.bus.subscribe([Topics.WORKER_HEARTBEAT, Topics.CAMERA_HEARTBEAT, Topics.CAMERA_STATUS,
                                  Topics.PIPELINE_METRICS], "control-plane", self.monitor.handle)
        await self.bus.subscribe([Topics.DETECTION_EVENTS], "control-plane-events", self.events.handle, from_beginning=True)
        await self.events.start()
        self._tasks.append(asyncio.create_task(self._control_loop(), name="control-loop"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self.events.stop()
        await self.bus.stop()
        await self.engine.dispose()

    async def reconcile_once(self) -> None:
        await self.monitor.sweep()
        await self.rebalancer.reassign_from_unhealthy()
        await self.rebalancer.place_pending()
        await self.rebalancer.heal_missing_sessions()
        await self.rebalancer.rebalance_once()

        async def lookup(cid: str):
            async with self.sf() as db:
                from app.db.models import Camera

                return await db.get(Camera, cid)

        await self.streams.expire_leases(lookup)
        await self.rebalancer.evaluate_autoscale()
        self.last_loop_ok = time.monotonic()

    async def _control_loop(self) -> None:
        last_purge = 0.0
        while True:
            try:
                await self.reconcile_once()
                if time.monotonic() - last_purge > 3600:
                    last_purge = time.monotonic()
                    n = await self.events.purge()
                    if n:
                        log.info("purged %d old events", n)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("control loop iteration failed")
            await asyncio.sleep(self.s.scheduler_interval_s)

    async def db_ok(self) -> bool:
        try:
            async with self.engine.connect() as c:
                await c.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
