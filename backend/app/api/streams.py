"""Stream endpoints live on the camera router (GET /api/v1/cameras/{id}/stream); this module adds
the MediaMTX gateway status."""
from __future__ import annotations

import asyncio
import hmac
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.security import ValidationError, decode_token

from app.api.deps import User, current_user, get_container
from app.services.container import Container

router = APIRouter(prefix="/api/v1/streams", tags=["streams"])


@router.get("/gateway")
async def gateway_status(c: Container = Depends(get_container), _: User = Depends(current_user)):
    ok = await asyncio.to_thread(c.streams.mtx.healthy)
    return {"mediamtx_api_reachable": ok, "policy": c.s.stream_policy, "webrtc_base": c.s.public_webrtc_base}


class MtxAuthRequest(BaseModel):
    """Body sent by MediaMTX's HTTP authentication (authMethod: http). Extra fields are ignored."""

    user: str | None = ""
    password: str | None = ""
    token: str | None = ""
    ip: str | None = ""
    action: str
    path: str | None = ""
    protocol: str | None = ""
    query: str | None = ""


def _eq(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and hmac.compare_digest(a.encode(), b.encode())


@router.post("/mediamtx-auth", include_in_schema=False)
async def mediamtx_auth(body: MtxAuthRequest, c: Container = Depends(get_container)):
    """Authorises MediaMTX requests. publish: worker credentials; read/playback: a valid control-plane JWT
    (Bearer / password / ?token=); api+metrics: dedicated API credentials. Returns 200 allow / 401 deny."""
    s = c.s
    a = body.action
    if a == "publish":
        if _eq(body.user, s.mediamtx_publish_user) and _eq(body.password, s.mediamtx_publish_pass):
            return {}
    elif a in ("read", "playback"):
        if not s.auth_enabled:
            return {}
        cands = [body.token, body.password, (parse_qs(body.query or "").get("token") or [""])[0]]
        for t in cands:
            if t:
                try:
                    decode_token(t, s.jwt_secret)
                    return {}
                except ValidationError:
                    pass
    elif a in ("api", "metrics", "pprof"):
        if _eq(body.user, s.mediamtx_api_user) and _eq(body.password, s.mediamtx_api_pass):
            return {}
    raise HTTPException(401, "unauthorized")
