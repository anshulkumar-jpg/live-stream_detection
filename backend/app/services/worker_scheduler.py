"""Chooses a worker for a camera (least-loaded among admitted workers)."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Camera, Worker
from app.services import admission_control as ac


class WorkerScheduler:
    def __init__(self, settings: Settings):
        self.s = settings

    async def assigned_counts(self, db: AsyncSession) -> dict[str, int]:
        r = await db.execute(select(Camera.worker_id, func.count()).where(
            Camera.worker_id.is_not(None), Camera.desired_state == "running").group_by(Camera.worker_id))
        return {w: n for w, n in r.all()}

    async def pick_worker(self, db: AsyncSession, exclude: set[str] = frozenset()) -> tuple[Worker | None, str]:  # type: ignore[assignment]
        workers = (await db.execute(select(Worker))).scalars().all()
        counts = await self.assigned_counts(db)
        best, best_load, reasons = None, 1e9, []
        for w in workers:
            if w.worker_id in exclude:
                continue
            n = counts.get(w.worker_id, 0)
            adm = ac.check(w, n, self.s)
            if not adm.ok:
                reasons.append(f"{w.worker_id}: {adm.reason}")
                continue
            load = n / max(1, min(w.capacity_limit, self.s.max_streams_per_worker))
            if load < best_load:
                best, best_load = w, load
        if best is None:
            return None, "; ".join(reasons) or "no workers registered"
        return best, "ok"
