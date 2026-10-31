from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.api.deps import User, current_user, get_container, require_admin
from app.core.security import ValidationError
from app.schemas.camera import CameraCreate, CameraHealth, CameraOut, CameraUpdate, StreamInfo
from app.services.camera_manager import ConflictError, NotFoundError
from app.services.container import Container

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])


def _out(cam) -> CameraOut:
    return CameraOut.model_validate(cam)


async def _guard(coro):
    try:
        return await coro
    except NotFoundError as e:
        raise HTTPException(404, f"camera '{e}' not found")
    except ConflictError as e:
        raise HTTPException(409, str(e))
    except ValidationError as e:
        raise HTTPException(422, str(e))


@router.post("", response_model=CameraOut, status_code=201)
async def create_camera(body: CameraCreate, c: Container = Depends(get_container), _: User = Depends(require_admin)):
    return _out(await _guard(c.cams.create(body)))


@router.get("", response_model=list[CameraOut])
async def list_cameras(offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500),
                       c: Container = Depends(get_container), _: User = Depends(current_user)):
    from app.db.repositories import CameraRepo

    async with c.sf() as db:
        return [_out(x) for x in await CameraRepo(db).list(offset, limit)]


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: str, c: Container = Depends(get_container), _: User = Depends(current_user)):
    from app.db.repositories import CameraRepo

    async with c.sf() as db:
        cam = await CameraRepo(db).get(camera_id)
    if cam is None:
        raise HTTPException(404, f"camera '{camera_id}' not found")
    return _out(cam)


@router.patch("/{camera_id}", response_model=CameraOut)
async def patch_camera(camera_id: str, body: CameraUpdate, c: Container = Depends(get_container), _: User = Depends(require_admin)):
    return _out(await _guard(c.cams.update(camera_id, body)))


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, c: Container = Depends(get_container), _: User = Depends(require_admin)):
    await _guard(c.cams.delete(camera_id))
    return Response(status_code=204)


@router.post("/{camera_id}/start", response_model=CameraOut)
async def start_camera(camera_id: str, c: Container = Depends(get_container), _: User = Depends(require_admin)):
    return _out(await _guard(c.cams.start(camera_id)))


@router.post("/{camera_id}/stop", response_model=CameraOut)
async def stop_camera(camera_id: str, c: Container = Depends(get_container), _: User = Depends(require_admin)):
    return _out(await _guard(c.cams.stop(camera_id)))


@router.get("/{camera_id}/health", response_model=CameraHealth)
async def camera_health(camera_id: str, c: Container = Depends(get_container), _: User = Depends(current_user)):
    from app.db.repositories import CameraRepo

    async with c.sf() as db:
        cam = await CameraRepo(db).get(camera_id)
    if cam is None:
        raise HTTPException(404, f"camera '{camera_id}' not found")
    age = (datetime.now(timezone.utc) - cam.last_heartbeat).total_seconds() if cam.last_heartbeat else None
    ok = cam.desired_state == "stopped" and cam.actual_state == "stopped" or (
        cam.actual_state == "running" and age is not None and age < c.s.worker_heartbeat_timeout_s)
    if ok and cam.actual_state == "running" and cam.stream_enabled and c.s.stream_policy == "always" \
            and (cam.heartbeat_detail or {}).get("stream_publishing") is False:
        ok = False  # decoding but the live stream is not reaching MediaMTX: not healthy
    return CameraHealth(camera_id=cam.camera_id, desired_state=cam.desired_state, actual_state=cam.actual_state,
                        worker_id=cam.worker_id, last_heartbeat=cam.last_heartbeat,
                        heartbeat_age_s=None if age is None else round(age, 2), healthy=bool(ok),
                        detail=cam.heartbeat_detail or {})


@router.get("/{camera_id}/stream", response_model=StreamInfo)
async def camera_stream(camera_id: str, c: Container = Depends(get_container), _: User = Depends(current_user)):
    from app.db.repositories import CameraRepo

    async with c.sf() as db:
        cam = await CameraRepo(db).get(camera_id)
    if cam is None:
        raise HTTPException(404, f"camera '{camera_id}' not found")
    return await c.streams.info(cam)
