"""add esco tables

Revision ID: 20260209230000
Revises: 20260209213000
Create Date: 2026-02-09 23:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260209230000"
down_revision = "20260209213000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "esco_skills",
        sa.Column("concept_uri", sa.Text(), primary_key=True, nullable=False),
        sa.Column("concept_type", sa.String(length=64), nullable=True),
        sa.Column("skill_type", sa.String(length=64), nullable=True),
        sa.Column("reuse_level", sa.String(length=64), nullable=True),
        sa.Column("preferred_label", sa.Text(), nullable=True),
        sa.Column("preferred_label_norm", sa.Text(), nullable=True),
        sa.Column("alt_labels", postgresql.JSONB(), nullable=True),
        sa.Column("alt_labels_norm", postgresql.JSONB(), nullable=True),
        sa.Column("hidden_labels", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("modified_date", sa.String(length=32), nullable=True),
        sa.Column("scope_note", sa.Text(), nullable=True),
        sa.Column("definition", sa.Text(), nullable=True),
        sa.Column("in_scheme", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
    )
    op.create_index("ix_esco_skills_preferred_label", "esco_skills", ["preferred_label"], unique=False)
    op.create_index(
        "ix_esco_skills_preferred_label_norm", "esco_skills", ["preferred_label_norm"], unique=False
    )

    op.create_table(
        "esco_occupations",
        sa.Column("concept_uri", sa.Text(), primary_key=True, nullable=False),
        sa.Column("concept_type", sa.String(length=64), nullable=True),
        sa.Column("isco_group", sa.String(length=32), nullable=True),
        sa.Column("preferred_label", sa.Text(), nullable=True),
        sa.Column("alt_labels", postgresql.JSONB(), nullable=True),
        sa.Column("hidden_labels", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("modified_date", sa.String(length=32), nullable=True),
        sa.Column("regulated_profession_note", sa.Text(), nullable=True),
        sa.Column("scope_note", sa.Text(), nullable=True),
        sa.Column("definition", sa.Text(), nullable=True),
        sa.Column("in_scheme", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("code", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_esco_occupations_preferred_label", "esco_occupations", ["preferred_label"], unique=False
    )

    op.create_table(
        "esco_skill_groups",
        sa.Column("concept_uri", sa.Text(), primary_key=True, nullable=False),
        sa.Column("concept_type", sa.String(length=64), nullable=True),
        sa.Column("preferred_label", sa.Text(), nullable=True),
        sa.Column("alt_labels", postgresql.JSONB(), nullable=True),
        sa.Column("hidden_labels", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("modified_date", sa.String(length=32), nullable=True),
        sa.Column("scope_note", sa.Text(), nullable=True),
        sa.Column("in_scheme", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("code", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_esco_skill_groups_preferred_label", "esco_skill_groups", ["preferred_label"], unique=False
    )

    op.create_table(
        "esco_skill_hierarchy",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
        sa.Column("level0_uri", sa.Text(), nullable=True),
        sa.Column("level0_preferred_term", sa.Text(), nullable=True),
        sa.Column("level1_uri", sa.Text(), nullable=True),
        sa.Column("level1_preferred_term", sa.Text(), nullable=True),
        sa.Column("level2_uri", sa.Text(), nullable=True),
        sa.Column("level2_preferred_term", sa.Text(), nullable=True),
        sa.Column("level3_uri", sa.Text(), nullable=True),
        sa.Column("level3_preferred_term", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("scope_note", sa.Text(), nullable=True),
        sa.Column("level0_code", sa.String(length=32), nullable=True),
        sa.Column("level1_code", sa.String(length=32), nullable=True),
        sa.Column("level2_code", sa.String(length=32), nullable=True),
        sa.Column("level3_code", sa.String(length=32), nullable=True),
    )
    op.create_index("ix_esco_skill_hierarchy_level0_uri", "esco_skill_hierarchy", ["level0_uri"])
    op.create_index("ix_esco_skill_hierarchy_level1_uri", "esco_skill_hierarchy", ["level1_uri"])
    op.create_index("ix_esco_skill_hierarchy_level2_uri", "esco_skill_hierarchy", ["level2_uri"])
    op.create_index("ix_esco_skill_hierarchy_level3_uri", "esco_skill_hierarchy", ["level3_uri"])

    op.create_table(
        "esco_language_skills",
        sa.Column("concept_uri", sa.Text(), primary_key=True, nullable=False),
        sa.Column("concept_type", sa.String(length=64), nullable=True),
        sa.Column("skill_type", sa.String(length=64), nullable=True),
        sa.Column("reuse_level", sa.String(length=64), nullable=True),
        sa.Column("preferred_label", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("alt_labels", postgresql.JSONB(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("broader_concept_uri", sa.Text(), nullable=True),
        sa.Column("broader_concept_pt", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_esco_language_skills_preferred_label",
        "esco_language_skills",
        ["preferred_label"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_esco_language_skills_preferred_label", table_name="esco_language_skills")
    op.drop_table("esco_language_skills")

    op.drop_index("ix_esco_skill_hierarchy_level3_uri", table_name="esco_skill_hierarchy")
    op.drop_index("ix_esco_skill_hierarchy_level2_uri", table_name="esco_skill_hierarchy")
    op.drop_index("ix_esco_skill_hierarchy_level1_uri", table_name="esco_skill_hierarchy")
    op.drop_index("ix_esco_skill_hierarchy_level0_uri", table_name="esco_skill_hierarchy")
    op.drop_table("esco_skill_hierarchy")

    op.drop_index("ix_esco_skill_groups_preferred_label", table_name="esco_skill_groups")
    op.drop_table("esco_skill_groups")

    op.drop_index("ix_esco_occupations_preferred_label", table_name="esco_occupations")
    op.drop_table("esco_occupations")

    op.drop_index("ix_esco_skills_preferred_label", table_name="esco_skills")
    op.drop_table("esco_skills")
