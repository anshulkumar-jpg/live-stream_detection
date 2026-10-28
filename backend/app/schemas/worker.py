from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.event import SCHEMA_VERSION, new_id, utcnow

CommandType = Literal["start_camera", "stop_camera", "update_camera", "drain_worker", "set_stream"]


class CameraSpec(BaseModel):
    """What a worker needs to run one camera. Carries the SECRET url ONLY on the internal
    command topic; the topic must be access-controlled (see README security section)."""

    camera_id: str
    source_url: str
    model_profile: str = "default"
    stream_enabled: bool = True
    target_fps: float | None = None
    priority: int = 0
    transport: str = "tcp"

    def __repr__(self) -> str:  # never leak credentials via repr/logging
        from app.core.security import redact_url

        return f"CameraSpec(camera_id={self.camera_id!r}, source={redact_url(self.source_url)!r})"

    __str__ = __repr__


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    command_id: str = Field(default_factory=new_id)
    type: CommandType
    worker_id: str  # target worker
    camera_id: str | None = None
    session_id: str | None = None  # control plane allocates the session id
    camera: CameraSpec | None = None
    params: dict[str, Any] = {}
    issued_at: datetime = Field(default_factory=utcnow)
    # Commands are idempotent: repeating start_camera with the same session_id is a no-op.


class WorkerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    worker_id: str
    hostname: str
    gpu_id: int
    status: str
    last_heartbeat: datetime | None
    active_camera_count: int
    capacity_limit: int
    created_at: datetime
    heartbeat_age_s: float | None = None
    healthy: bool = False
    cameras: list[str] = []
    details: dict = {}
