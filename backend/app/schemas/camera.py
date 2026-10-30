from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.security import ValidationError, validate_camera_id


def _classes_schema(schema: dict) -> None:
    """Show the model's class names as the allowed values (and a ready example) in Swagger/OpenAPI."""
    try:
        from app.core.config import get_settings, load_models_config

        names = load_models_config(get_settings().models_config).class_names("default")
    except Exception:  # docs only: never break startup over this
        return
    if not names:
        return
    for opt in schema.get("anyOf", []):
        if opt.get("type") == "array":
            opt["items"] = {"type": "string", "enum": names}
    schema["examples"] = [names[:1], names[:3:2] if len(names) > 2 else names, names]
    schema["description"] = (f"Classes to detect, any of {names}. Omit, null or [] = detect all. "
                             "Same list: GET /api/v1/detection-classes")


class CameraCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str | None = None  # optional: generated from `name` when omitted
    name: str = Field(min_length=1, max_length=128)
    rtsp_url: str = Field(min_length=8, max_length=2048)
    model_profile: str = "default"
    stream_enabled: bool = True
    enabled: bool = True
    target_fps: float | None = Field(default=None, gt=0, le=60)
    priority: int = Field(default=0, ge=0, le=10)  # >0 = critical camera, scheduled first
    transport: Literal["tcp", "udp"] = "tcp"
    autostart: bool = True
    detection_classes: list[str] | None = Field(default=None, json_schema_extra=_classes_schema)  # omitted/null = all

    @field_validator("camera_id")
    @classmethod
    def _id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            return validate_camera_id(v)
        except ValidationError as e:
            raise ValueError(str(e)) from e


class CameraUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=128)
    rtsp_url: str | None = Field(default=None, min_length=8, max_length=2048)
    model_profile: str | None = None
    stream_enabled: bool | None = None
    enabled: bool | None = None
    target_fps: float | None = Field(default=None, gt=0, le=60)
    priority: int | None = Field(default=None, ge=0, le=10)
    transport: Literal["tcp", "udp"] | None = None
    detection_classes: list[str] | None = Field(default=None, json_schema_extra=_classes_schema)  # [] or null = all


class CameraOut(BaseModel):
    """Never contains the RTSP URL or credentials."""

    model_config = ConfigDict(from_attributes=True)
    camera_id: str
    name: str
    source_display: str  # redacted URL (credentials replaced by ***)
    model_profile: str
    enabled: bool
    desired_state: str
    actual_state: str
    worker_id: str | None
    stream_enabled: bool
    target_fps: float | None
    priority: int
    transport: str
    detection_classes: list[str] | None = None  # null = all classes
    last_error: str | None = None
    last_heartbeat: datetime | None = None
    created_at: datetime
    updated_at: datetime


class CameraHealth(BaseModel):
    camera_id: str
    desired_state: str
    actual_state: str
    worker_id: str | None
    last_heartbeat: datetime | None
    heartbeat_age_s: float | None
    healthy: bool
    detail: dict = {}


class StreamInfo(BaseModel):
    camera_id: str
    path: str
    ready: bool
    publishing: bool
    webrtc_url: str
    hls_master_url: str | None = None
    qualities: list[int] = []
    note: str | None = None
