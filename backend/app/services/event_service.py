"""Detection events: Kafka -> PostgreSQL (idempotent, batched) + live fan-out to WebSocket clients."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Camera, DetectionEventRow
from app.db.repositories import EventRepo
from app.messaging.kafka_consumer import SeenCache
from app.schemas.event import DetectionEvent

log = logging.getLogger(__name__)


class Hub:
    """WebSocket fan-out with bounded per-client queues (slow clients drop messages, never block)."""

    def __init__(self):
        self._clients: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(200)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    async def broadcast(self, msg: dict) -> None:
        for q in list(self._clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


class EventService:
    def __init__(self, sf: async_sessionmaker[AsyncSession], hub: Hub, flush_interval_s: float = 0.5, batch: int = 200,
                 retention_days: int = 14):
        self.sf, self.hub, self.flush_s, self.batch, self.retention_days = sf, hub, flush_interval_s, batch, retention_days
        self._buf: list[dict] = []
        self._seen = SeenCache()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.stored = 0
        self.rejected = 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._flusher())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def handle(self, topic: str, key: str | None, value: dict) -> None:
        try:
            ev = DetectionEvent.model_validate(value)  # schema validation on ingest
        except Exception as e:  # noqa: BLE001
            self.rejected += 1
            log.warning("invalid detection event dropped: %s", str(e)[:200])
            return
        if self._seen.seen(ev.idempotency_key):
            return
        async with self._lock:
            self._buf.append(ev.model_dump(mode="json"))
            full = len(self._buf) >= self.batch
        await self.hub.broadcast({"type": "detection", "camera_id": ev.camera_id, "timestamp": ev.timestamp.isoformat(),
                                  "sequence_number": ev.sequence_number, "model_name": ev.model_name,
                                  "detections": [{"class_name": d.class_name, "confidence": d.confidence}
                                                 for d in ev.detections][:10]})
        if full:
            await self.flush()

    async def _flusher(self) -> None:
        while True:
            await asyncio.sleep(self.flush_s)
            try:
                await self.flush()
            except Exception:  # noqa: BLE001
                log.exception("event flush failed")

    async def flush(self) -> None:
        async with self._lock:
            rows, self._buf = self._buf, []
        if not rows:
            return
        async with self.sf() as db:
            known = set((await db.execute(select(Camera.camera_id).where(
                Camera.camera_id.in_({r["camera_id"] for r in rows})))).scalars())
            values = [dict(event_id=r["event_id"], idempotency_key=r["idempotency_key"], camera_id=r["camera_id"],
                           worker_id=r["worker_id"], timestamp=datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")),
                           event_type=r["event_type"], payload=r, schema_version=r["schema_version"])
                      for r in rows if r["camera_id"] in known]
            if values:
                dialect = db.bind.dialect.name  # type: ignore[union-attr]
                ins = (pg_insert if dialect == "postgresql" else sqlite_insert)(DetectionEventRow).values(values)
                await db.execute(ins.on_conflict_do_nothing(index_elements=["idempotency_key"]))
                await db.commit()
                self.stored += len(values)

    async def purge(self) -> int:
        async with self.sf() as db:
            n = await EventRepo(db).purge_older_than(self.retention_days)
            await db.commit()
            return n
