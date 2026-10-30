"""SQLAlchemy 2.x models. Portable across PostgreSQL (prod) and SQLite (tests)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

JsonType = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator):
    """Store timezone-aware UTC; return aware datetimes on every backend (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            raise ValueError("naive datetime not allowed")
        return value.astimezone(timezone.utc) if value is not None else None

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelProfile(Base):
    __tablename__ = "model_profiles"
    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    config: Mapped[dict] = mapped_column(JsonType, default=dict)  # model refs + inference settings
    enabled_stages: Mapped[list] = mapped_column(JsonType, default=list)
    version: Mapped[str] = mapped_column(String(32), default="1")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now, onupdate=now)


class Worker(Base):
    __tablename__ = "workers"
    worker_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(255), default="")
    gpu_id: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="offline")  # ready|draining|stopping|degraded|offline
    last_heartbeat: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    active_camera_count: Mapped[int] = mapped_column(Integer, default=0)
    capacity_limit: Mapped[int] = mapped_column(Integer, default=16)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)
    details: Mapped[dict] = mapped_column(JsonType, default=dict)  # last heartbeat snapshot (gpu, latency, ...)

    cameras: Mapped[list["Camera"]] = relationship(back_populates="worker")


class Camera(Base):
    __tablename__ = "cameras"
    camera_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    rtsp_url_encrypted: Mapped[str] = mapped_column(Text)  # Fernet token; never plaintext
    source_display: Mapped[str] = mapped_column(String(512))  # redacted URL for UI
    model_profile: Mapped[str] = mapped_column(ForeignKey("model_profiles.profile_id"), default="default")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    desired_state: Mapped[str] = mapped_column(String(16), default="stopped")  # running|stopped
    actual_state: Mapped[str] = mapped_column(String(16), default="stopped")  # worker-reported
    worker_id: Mapped[str | None] = mapped_column(ForeignKey("workers.worker_id", ondelete="SET NULL"), nullable=True)
    current_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    stream_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    target_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    transport: Mapped[str] = mapped_column(String(3), default="tcp")
    detection_classes: Mapped[list | None] = mapped_column(JsonType, nullable=True)  # None = detect every class
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_heartbeat: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    heartbeat_detail: Mapped[dict] = mapped_column(JsonType, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=now, onupdate=now)

    worker: Mapped[Worker | None] = relationship(back_populates="cameras")

    __table_args__ = (Index("ix_cameras_worker", "worker_id"), Index("ix_cameras_desired", "desired_state"))


class DetectionEventRow(Base):
    __tablename__ = "detection_events"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id", ondelete="CASCADE"))
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime)
    event_type: Mapped[str] = mapped_column(String(32), default="detection")
    payload: Mapped[dict] = mapped_column(JsonType)
    schema_version: Mapped[str] = mapped_column(String(8), default="1.0")

    __table_args__ = (Index("ix_events_camera_ts", "camera_id", "timestamp"), Index("ix_events_ts", "timestamp"))


class PipelineSession(Base):
    __tablename__ = "pipeline_sessions"
    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id", ondelete="CASCADE"))
    worker_id: Mapped[str] = mapped_column(String(64))
    start_time: Mapped[datetime] = mapped_column(UTCDateTime, default=now)
    stop_time: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="starting")  # starting|running|stopped|lost|error|migrated
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_sessions_camera", "camera_id", "status"),)
