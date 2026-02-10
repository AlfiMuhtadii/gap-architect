"""add processing timestamps to gap_analyses

Revision ID: 20260210231000
Revises: 20260209240000
Create Date: 2026-02-10 23:10:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260210231000"
down_revision = "20260209240000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "gap_analyses",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "gap_analyses",
        sa.Column("last_error_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("gap_analyses", "last_error_at")
    op.drop_column("gap_analyses", "processing_started_at")
