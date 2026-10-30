from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app.api.deps import User, current_user, get_container
from app.db.repositories import CameraRepo, WorkerRepo
from app.services import admission_control as ac
from app.services.container import Container

router = APIRouter(tags=["health"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.post("/api/v1/auth/login")
async def login(body: LoginIn, c: Container = Depends(get_container)):
    from app.core.security import create_token

    role = c.authenticate(body.username, body.password)
    if role is None:
        raise HTTPException(401, "invalid credentials")
    return {"access_token": create_token(body.username, role, c.s.jwt_secret), "token_type": "bearer", "role": role}


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(c: Container = Depends(get_container)):
    db = await c.db_ok()
    bus = c.bus.healthy()
    body = {"database": db, "message_bus": bus}
    if not (db and bus):
        raise HTTPException(503, detail=body)
    return {"status": "ready", **body}


@router.get("/metrics")
async def metrics_endpoint():
    """Control-plane process metrics (heartbeat ages, command counters). Worker metrics are scraped from workers."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/api/v1/profiles")
async def profiles(c: Container = Depends(get_container), _: User = Depends(current_user)):
    from sqlalchemy import select

    from app.db.models import ModelProfile

    async with c.sf() as db:
        rows = (await db.execute(select(ModelProfile))).scalars().all()
    return [{"profile_id": r.profile_id, "stages": r.enabled_stages, "version": r.version, "enabled": r.enabled,
             "config": r.config} for r in rows]


@router.get("/api/v1/detection-classes")
async def detection_classes(model_profile: str = "default", c: Container = Depends(get_container),
                            _: User = Depends(current_user)):
    """Classes a user can choose from when adding/editing a camera (detection_classes field)."""
    from app.core.config import load_models_config

    names = load_models_config(c.s.models_config).class_names(model_profile)
    if names is None:
        raise HTTPException(404, f"profile '{model_profile}' does not declare its classes")
    return {"model_profile": model_profile, "classes": names}


@router.get("/api/v1/system/summary")
async def system_summary(c: Container = Depends(get_container), _: User = Depends(current_user)):
    async with c.sf() as db:
        ws = await WorkerRepo(db).list()
        cams = await CameraRepo(db).list(0, 1000)
    by_state: dict[str, int] = {}
    for x in cams:
        by_state[x.actual_state] = by_state.get(x.actual_state, 0) + 1
    return {"cameras_total": len(cams), "cameras_by_state": by_state,
            "workers_total": len(ws), "workers_healthy": sum(ac.is_healthy(w, c.s) for w in ws),
            "database": await c.db_ok(), "message_bus": c.bus.healthy(), "stream_policy": c.s.stream_policy,
            "autoscaler_enabled": c.s.autoscaler_enabled}
