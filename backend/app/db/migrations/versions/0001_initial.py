"""initial schema

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONT = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "model_profiles",
        sa.Column("profile_id", sa.String(64), primary_key=True),
        sa.Column("config", JSONT, nullable=False),
        sa.Column("enabled_stages", JSONT, nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_table(
        "workers",
        sa.Column("worker_id", sa.String(64), primary_key=True),
        sa.Column("hostname", sa.String(255), nullable=False),
        sa.Column("gpu_id", sa.Integer, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("last_heartbeat", TS),
        sa.Column("active_camera_count", sa.Integer, nullable=False),
        sa.Column("capacity_limit", sa.Integer, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("details", JSONT, nullable=False),
    )
    op.create_table(
        "cameras",
        sa.Column("camera_id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("rtsp_url_encrypted", sa.Text, nullable=False),
        sa.Column("source_display", sa.String(512), nullable=False),
        sa.Column("model_profile", sa.String(64), sa.ForeignKey("model_profiles.profile_id"), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False),
        sa.Column("desired_state", sa.String(16), nullable=False),
        sa.Column("actual_state", sa.String(16), nullable=False),
        sa.Column("worker_id", sa.String(64), sa.ForeignKey("workers.worker_id", ondelete="SET NULL")),
        sa.Column("current_session_id", sa.String(36)),
        sa.Column("stream_enabled", sa.Boolean, nullable=False),
        sa.Column("target_fps", sa.Float),
        sa.Column("priority", sa.Integer, nullable=False),
        sa.Column("transport", sa.String(3), nullable=False),
        sa.Column("last_error", sa.Text),
        sa.Column("last_heartbeat", TS),
        sa.Column("heartbeat_detail", JSONT, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_cameras_worker", "cameras", ["worker_id"])
    op.create_index("ix_cameras_desired", "cameras", ["desired_state"])
    op.create_table(
        "detection_events",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column("idempotency_key", sa.String(200), nullable=False, unique=True),
        sa.Column("camera_id", sa.String(64), sa.ForeignKey("cameras.camera_id", ondelete="CASCADE"), nullable=False),
        sa.Column("worker_id", sa.String(64)),
        sa.Column("timestamp", TS, nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("payload", JSONT, nullable=False),
        sa.Column("schema_version", sa.String(8), nullable=False),
    )
    op.create_index("ix_events_camera_ts", "detection_events", ["camera_id", "timestamp"])
    op.create_index("ix_events_ts", "detection_events", ["timestamp"])
    op.create_table(
        "pipeline_sessions",
        sa.Column("session_id", sa.String(36), primary_key=True),
        sa.Column("camera_id", sa.String(64), sa.ForeignKey("cameras.camera_id", ondelete="CASCADE"), nullable=False),
        sa.Column("worker_id", sa.String(64), nullable=False),
        sa.Column("start_time", TS, nullable=False),
        sa.Column("stop_time", TS),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error", sa.Text),
    )
    op.create_index("ix_sessions_camera", "pipeline_sessions", ["camera_id", "status"])


def downgrade() -> None:
    for t in ("pipeline_sessions", "detection_events", "cameras", "workers", "model_profiles"):
        op.drop_table(t)
