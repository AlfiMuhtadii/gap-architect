"""add jd_skills_override to gap_analyses

Revision ID: 20260209203000
Revises: 20260209200000
Create Date: 2026-02-09 20:30:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260209203000"
down_revision = "20260209200000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gap_analyses", sa.Column("jd_skills_override", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("gap_analyses", "jd_skills_override")
