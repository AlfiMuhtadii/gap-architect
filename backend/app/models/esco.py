from __future__ import annotations

from sqlalchemy import Text, String, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class EscoSkill(Base):
    __tablename__ = "esco_skills"

    concept_uri: Mapped[str] = mapped_column(Text, primary_key=True)
    concept_type: Mapped[str | None] = mapped_column(String(64))
    skill_type: Mapped[str | None] = mapped_column(String(64))
    reuse_level: Mapped[str | None] = mapped_column(String(64))
    preferred_label: Mapped[str | None] = mapped_column(Text)
    preferred_label_norm: Mapped[str | None] = mapped_column(Text)
    alt_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    alt_labels_norm: Mapped[list[str] | None] = mapped_column(JSONB)
    hidden_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    status: Mapped[str | None] = mapped_column(String(32))
    modified_date: Mapped[str | None] = mapped_column(String(32))
    scope_note: Mapped[str | None] = mapped_column(Text)
    definition: Mapped[str | None] = mapped_column(Text)
    in_scheme: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_esco_skills_preferred_label", "preferred_label"),
        Index("ix_esco_skills_preferred_label_norm", "preferred_label_norm"),
    )


class EscoOccupation(Base):
    __tablename__ = "esco_occupations"

    concept_uri: Mapped[str] = mapped_column(Text, primary_key=True)
    concept_type: Mapped[str | None] = mapped_column(String(64))
    isco_group: Mapped[str | None] = mapped_column(String(32))
    preferred_label: Mapped[str | None] = mapped_column(Text)
    alt_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    hidden_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    status: Mapped[str | None] = mapped_column(String(32))
    modified_date: Mapped[str | None] = mapped_column(String(32))
    regulated_profession_note: Mapped[str | None] = mapped_column(Text)
    scope_note: Mapped[str | None] = mapped_column(Text)
    definition: Mapped[str | None] = mapped_column(Text)
    in_scheme: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    code: Mapped[str | None] = mapped_column(String(32))

    __table_args__ = (Index("ix_esco_occupations_preferred_label", "preferred_label"),)


class EscoSkillGroup(Base):
    __tablename__ = "esco_skill_groups"

    concept_uri: Mapped[str] = mapped_column(Text, primary_key=True)
    concept_type: Mapped[str | None] = mapped_column(String(64))
    preferred_label: Mapped[str | None] = mapped_column(Text)
    alt_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    hidden_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    status: Mapped[str | None] = mapped_column(String(32))
    modified_date: Mapped[str | None] = mapped_column(String(32))
    scope_note: Mapped[str | None] = mapped_column(Text)
    in_scheme: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    code: Mapped[str | None] = mapped_column(String(32))

    __table_args__ = (Index("ix_esco_skill_groups_preferred_label", "preferred_label"),)


class EscoSkillHierarchy(Base):
    __tablename__ = "esco_skill_hierarchy"

    level0_uri: Mapped[str | None] = mapped_column(Text)
    level0_preferred_term: Mapped[str | None] = mapped_column(Text)
    level1_uri: Mapped[str | None] = mapped_column(Text)
    level1_preferred_term: Mapped[str | None] = mapped_column(Text)
    level2_uri: Mapped[str | None] = mapped_column(Text)
    level2_preferred_term: Mapped[str | None] = mapped_column(Text)
    level3_uri: Mapped[str | None] = mapped_column(Text)
    level3_preferred_term: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    scope_note: Mapped[str | None] = mapped_column(Text)
    level0_code: Mapped[str | None] = mapped_column(String(32))
    level1_code: Mapped[str | None] = mapped_column(String(32))
    level2_code: Mapped[str | None] = mapped_column(String(32))
    level3_code: Mapped[str | None] = mapped_column(String(32))

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    __table_args__ = (
        Index("ix_esco_skill_hierarchy_level0_uri", "level0_uri"),
        Index("ix_esco_skill_hierarchy_level1_uri", "level1_uri"),
        Index("ix_esco_skill_hierarchy_level2_uri", "level2_uri"),
        Index("ix_esco_skill_hierarchy_level3_uri", "level3_uri"),
    )


class EscoLanguageSkill(Base):
    __tablename__ = "esco_language_skills"

    concept_uri: Mapped[str] = mapped_column(Text, primary_key=True)
    concept_type: Mapped[str | None] = mapped_column(String(64))
    skill_type: Mapped[str | None] = mapped_column(String(64))
    reuse_level: Mapped[str | None] = mapped_column(String(64))
    preferred_label: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(32))
    alt_labels: Mapped[list[str] | None] = mapped_column(JSONB)
    description: Mapped[str | None] = mapped_column(Text)
    broader_concept_uri: Mapped[str | None] = mapped_column(Text)
    broader_concept_pt: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_esco_language_skills_preferred_label", "preferred_label"),)
