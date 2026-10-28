"""Kafka/API event schemas. All timestamps are timezone-aware UTC. Payloads are kept compact:
no raw frames and no embeddings (only references / dimensions)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION

    @field_validator("*", mode="after")
    @classmethod
    def _utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        return v


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    # ReID: appearance similarity to a previously seen track (NOT a global identity claim)
    appearance_match_track_id: int | None = None
    appearance_similarity: float | None = None
    embedding_ref: str | None = None  # pointer to external store; embeddings never inline
    mask_area_px: int | None = None
    mask_model: str | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def _bbox(cls, v):
        if v[2] < v[0] or v[3] < v[1]:
            raise ValueError("bbox_xyxy must be x1<=x2, y1<=y2")
        return v


class DetectionEvent(_Base):
    event_id: str = Field(default_factory=new_id)
    idempotency_key: str
    event_type: Literal["detection"] = "detection"
    camera_id: str
    worker_id: str
    session_id: str | None = None
    timestamp: datetime  # capture timestamp (UTC)
    sequence_number: int
    model_name: str
    model_version: str
    detections: list[Detection]
    frame_resolution: tuple[int, int] | None = None
    processing_ms: float
    end_to_end_ms: float | None = None
    stage_timings_ms: dict[str, float] = {}
    snapshot_ref: str | None = None
    extra: dict[str, str | int | float | bool] = {}  # e.g. dino_model/dino_dim (no vectors)


class CameraHeartbeat(_Base):
    event_id: str = Field(default_factory=new_id)
    camera_id: str
    worker_id: str
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    state: Literal["starting", "connecting", "running", "reconnecting", "stopped", "error"]
    stream_publishing: bool = False
    decoder_fps: float = 0.0
    inference_fps: float = 0.0
    output_fps: float = 0.0
    queue_depth: int = 0
    dropped_frames: int = 0
    ffmpeg_restarts: int = 0
    last_frame_age_s: float | None = None
    last_error: str | None = None  # already redacted
    source_resolution: tuple[int, int] | None = None
    decoder: str | None = None  # "cuda"|"cpu"
    encoder: str | None = None


class WorkerHeartbeat(_Base):
    event_id: str = Field(default_factory=new_id)
    worker_id: str
    hostname: str
    gpu_id: int
    timestamp: datetime = Field(default_factory=utcnow)
    status: Literal["starting", "ready", "draining", "stopping", "degraded"]
    capacity_limit: int
    active_cameras: list[str] = []
    queue_depths: dict[str, int] = {}
    gpu_utilization_pct: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    inference_latency_p50_s: float | None = None
    inference_latency_p95_s: float | None = None
    inference_latency_p99_s: float | None = None
    models_ready: dict[str, bool] = {}
    cuda_available: bool = False
    inference_backend: str = "mock"
    metrics_url: str | None = None
    decoders: dict[str, bool] = {}  # capability detection result: {"nvdec": bool}
    encoders: dict[str, bool] = {}  # {"nvenc_h264": bool, ...}


class PipelineMetricsEvent(_Base):
    event_id: str = Field(default_factory=new_id)
    worker_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    input_fps: float = 0.0
    inference_fps: float = 0.0
    output_fps: float = 0.0
    dropped_frames_total: int = 0
    latency_p50_s: float | None = None
    latency_p95_s: float | None = None
    latency_p99_s: float | None = None


class CameraStatusEvent(_Base):
    """Actual-state change reported by a worker (separate from the desired state in the DB)."""

    event_id: str = Field(default_factory=new_id)
    camera_id: str
    worker_id: str
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    actual_state: Literal["starting", "running", "reconnecting", "stopped", "error"]
    error: str | None = None
    command_id: str | None = None  # ack correlation


class EventOut(BaseModel):
    """API representation of a stored event."""

    model_config = ConfigDict(from_attributes=True)
    event_id: str
    camera_id: str
    worker_id: str | None
    timestamp: datetime
    event_type: str
    payload: dict
    schema_version: str


class EventPage(BaseModel):
    items: list[EventOut]
    total: int
    page: int
    page_size: int
