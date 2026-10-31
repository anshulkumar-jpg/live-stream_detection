from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import User, current_user, get_container, require_admin
from app.db.repositories import CameraRepo, WorkerRepo
from app.schemas.worker import WorkerOut
from app.services import admission_control as ac
from app.services.container import Container

router = APIRouter(prefix="/api/v1/workers", tags=["workers"])


async def _to_out(c: Container, w, cams=None) -> WorkerOut:
    if cams is None:
        async with c.sf() as db:
            cams = [x.camera_id for x in await CameraRepo(db).by_worker(w.worker_id)]
    out = WorkerOut(worker_id=w.worker_id, hostname=w.hostname, gpu_id=w.gpu_id, status=w.status,
                    last_heartbeat=w.last_heartbeat, active_camera_count=w.active_camera_count,
                    capacity_limit=w.capacity_limit, created_at=w.created_at)
    out.heartbeat_age_s = ac.heartbeat_age(w)
    out.healthy = ac.is_healthy(w, c.s)
    out.cameras = cams
    out.details = w.details or {}
    return out


@router.get("", response_model=list[WorkerOut])
async def list_workers(c: Container = Depends(get_container), _: User = Depends(current_user)):
    async with c.sf() as db:
        ws = await WorkerRepo(db).list()
        repo = CameraRepo(db)
        return [await _to_out(c, w, [x.camera_id for x in await repo.by_worker(w.worker_id)]) for w in ws]


@router.get("/{worker_id}", response_model=WorkerOut)
async def get_worker(worker_id: str, c: Container = Depends(get_container), _: User = Depends(current_user)):
    async with c.sf() as db:
        w = await WorkerRepo(db).get(worker_id)
    if w is None:
        raise HTTPException(404, f"worker '{worker_id}' not found")
    return await _to_out(c, w)


@router.get("/{worker_id}/metrics")
async def worker_metrics(worker_id: str, c: Container = Depends(get_container), _: User = Depends(current_user)):
    """Latest reported snapshot (heartbeat + pipeline metrics). Time series live in Prometheus."""
    async with c.sf() as db:
        w = await WorkerRepo(db).get(worker_id)
    if w is None:
        raise HTTPException(404, f"worker '{worker_id}' not found")
    d = w.details or {}
    return {"worker_id": worker_id, "heartbeat_age_s": ac.heartbeat_age(w), "heartbeat": d,
            "pipeline": c.monitor.pipeline_metrics.get(worker_id, {})}


@router.post("/{worker_id}/drain", status_code=202)
async def drain_worker(worker_id: str, c: Container = Depends(get_container), _: User = Depends(require_admin)):
    async with c.sf() as db:
        w = await WorkerRepo(db).get(worker_id)
        if w is None:
            raise HTTPException(404, f"worker '{worker_id}' not found")
        previous = w.status
        w.status = "draining"  # stops admission immediately; worker confirms via heartbeat
        await db.commit()
    err = await c.commands.drain(worker_id)
    if err:
        async with c.sf() as db:  # the worker never got the command: do not leave it stuck in 'draining'
            w = await WorkerRepo(db).get(worker_id)
            if w is not None and w.status == "draining":
                w.status = previous
                await db.commit()
        raise HTTPException(502, err)
    return {"worker_id": worker_id, "status": "draining", "note": "cameras are migrated by the rebalancer as capacity allows"}
