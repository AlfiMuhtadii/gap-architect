"""add skill categories

Revision ID: 20260209240000
Revises: 20260209234500
Create Date: 2026-02-09 24:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260209240000"
down_revision = "20260209234500"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gap_results", sa.Column("technical_skills_missing", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("gap_results", sa.Column("transversal_soft_skills_missing", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("gap_results", sa.Column("language_skills_missing", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("gap_results", "language_skills_missing")
    op.drop_column("gap_results", "transversal_soft_skills_missing")
    op.drop_column("gap_results", "technical_skills_missing")
