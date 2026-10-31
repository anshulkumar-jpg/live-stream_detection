"""Thin async repositories. All take an AsyncSession; callers own the transaction."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Camera, DetectionEventRow, ModelProfile, PipelineSession, Worker, now


class CameraRepo:
    def __init__(self, s: AsyncSession):
        self.s = s

    async def get(self, camera_id: str) -> Camera | None:
        return await self.s.get(Camera, camera_id)

    async def list(self, offset: int = 0, limit: int = 100) -> list[Camera]:
        r = await self.s.execute(select(Camera).order_by(Camera.camera_id).offset(offset).limit(limit))
        return list(r.scalars())

    async def by_worker(self, worker_id: str) -> list[Camera]:
        return list((await self.s.execute(select(Camera).where(Camera.worker_id == worker_id))).scalars())

    async def desired_running(self) -> list[Camera]:
        r = await self.s.execute(select(Camera).where(Camera.desired_state == "running", Camera.enabled.is_(True)))
        return list(r.scalars())

    async def count(self) -> int:
        return (await self.s.execute(select(func.count()).select_from(Camera))).scalar_one()


class WorkerRepo:
    def __init__(self, s: AsyncSession):
        self.s = s

    async def get(self, worker_id: str) -> Worker | None:
        return await self.s.get(Worker, worker_id)

    async def list(self) -> list[Worker]:
        return list((await self.s.execute(select(Worker).order_by(Worker.worker_id))).scalars())


class EventRepo:
    def __init__(self, s: AsyncSession):
        self.s = s

    async def page(self, page: int, page_size: int, camera_id: str | None = None, since: datetime | None = None,
                   until: datetime | None = None, class_name: str | None = None) -> tuple[list[DetectionEventRow], int]:
        q = select(DetectionEventRow)
        cq = select(func.count()).select_from(DetectionEventRow)
        for cond in (
            (DetectionEventRow.camera_id == camera_id) if camera_id else None,
            (DetectionEventRow.timestamp >= since) if since else None,
            (DetectionEventRow.timestamp <= until) if until else None,
            self._class_cond(class_name) if class_name else None,
        ):
            if cond is not None:
                q, cq = q.where(cond), cq.where(cond)
        total = (await self.s.execute(cq)).scalar_one()
        rows = (await self.s.execute(q.order_by(DetectionEventRow.timestamp.desc())
                                     .offset((page - 1) * page_size).limit(page_size))).scalars().all()
        return list(rows), total

    def _class_cond(self, class_name: str):
        """Filter in SQL (before count and paging): events having a detection of this class."""
        import json

        from sqlalchemy import String, cast, literal
        from sqlalchemy.dialects.postgresql import JSONB

        if self.s.get_bind().dialect.name == "postgresql":
            doc = {"detections": [{"class_name": class_name}]}  # a dict: the JSONB type serialises it (a str would become a JSON string)
            return cast(DetectionEventRow.payload, JSONB).op("@>")(literal(doc, JSONB))
        return cast(DetectionEventRow.payload, String).like(f'%"class_name": {json.dumps(class_name)}%')  # sqlite (tests)

    async def purge_older_than(self, days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        r = await self.s.execute(delete(DetectionEventRow).where(DetectionEventRow.timestamp < cutoff))
        return r.rowcount or 0


class SessionRepo:
    def __init__(self, s: AsyncSession):
        self.s = s

    async def open_for_camera(self, camera_id: str) -> list[PipelineSession]:
        r = await self.s.execute(select(PipelineSession).where(
            PipelineSession.camera_id == camera_id, PipelineSession.status.in_(("starting", "running"))))
        return list(r.scalars())

    async def close(self, session_id: str, status: str, error: str | None = None) -> None:
        ps = await self.s.get(PipelineSession, session_id)
        if ps and ps.stop_time is None:
            ps.stop_time, ps.status, ps.error = now(), status, error
