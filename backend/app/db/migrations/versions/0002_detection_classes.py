"""per-camera detection class selection

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

JSONT = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("cameras", sa.Column("detection_classes", JSONT, nullable=True))


def downgrade() -> None:
    op.drop_column("cameras", "detection_classes")
