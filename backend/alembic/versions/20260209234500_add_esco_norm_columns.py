"""add normalized label columns to esco_skills

Revision ID: 20260209234500
Revises: 20260209230000
Create Date: 2026-02-09 23:45:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260209234500"
down_revision = "20260209230000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("esco_skills", sa.Column("preferred_label_norm", sa.Text(), nullable=True))
    op.add_column("esco_skills", sa.Column("alt_labels_norm", postgresql.JSONB(), nullable=True))
    op.create_index("ix_esco_skills_preferred_label_norm", "esco_skills", ["preferred_label_norm"])


def downgrade() -> None:
    op.drop_index("ix_esco_skills_preferred_label_norm", table_name="esco_skills")
    op.drop_column("esco_skills", "alt_labels_norm")
    op.drop_column("esco_skills", "preferred_label_norm")
