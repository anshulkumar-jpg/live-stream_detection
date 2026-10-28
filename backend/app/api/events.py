from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.api.deps import User, current_user, get_container
from app.core.security import ValidationError, decode_token
from app.db.repositories import EventRepo
from app.schemas.event import EventOut, EventPage
from app.services.container import Container

router = APIRouter(tags=["events"])


@router.get("/api/v1/events", response_model=EventPage)
async def list_events(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200), camera_id: str | None = None,
                      since: datetime | None = None, until: datetime | None = None, class_name: str | None = None,
                      c: Container = Depends(get_container), _: User = Depends(current_user)):
    async with c.sf() as db:
        rows, total = await EventRepo(db).page(page, page_size, camera_id, since, until, class_name)
    return EventPage(items=[EventOut.model_validate(r) for r in rows], total=total, page=page, page_size=page_size)


@router.websocket("/api/v1/ws")
async def ws(websocket: WebSocket, token: str | None = None):
    """Live UI updates (camera/worker state, detections). Auth: ?token=<jwt> (browsers cannot set WS headers)."""
    c: Container = websocket.app.state.container
    if c.s.auth_enabled:
        try:
            decode_token(token or "", c.s.jwt_secret)
        except ValidationError:
            await websocket.close(code=4401)
            return
    await websocket.accept()
    q = c.hub.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), 20)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        c.hub.unsubscribe(q)
