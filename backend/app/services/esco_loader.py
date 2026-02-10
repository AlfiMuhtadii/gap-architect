from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.esco import (
    EscoSkill,
    EscoOccupation,
    EscoSkillGroup,
    EscoSkillHierarchy,
    EscoLanguageSkill,
)


def _split_labels(value: str | None) -> list[str] | None:
    if not value:
        return None
    items: list[str] = []
    for part in value.replace("|", "\n").splitlines():
        part = part.strip()
        if part:
            items.append(part)
    return items or None


def _norm(s: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


async def load_esco_skills(session: AsyncSession, csv_path: str) -> None:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower(): v for k, v in row.items() if k}
            concept_uri = (lower.get("concepturi") or "").strip()
            if not concept_uri:
                continue
            alt = _split_labels(lower.get("altlabels"))
            hidden = _split_labels(lower.get("hiddenlabels"))
            session.add(
                EscoSkill(
                    concept_uri=concept_uri,
                    concept_type=lower.get("concepttype"),
                    skill_type=lower.get("skilltype"),
                    reuse_level=lower.get("reuselevel"),
                    preferred_label=lower.get("preferredlabel"),
                    preferred_label_norm=_norm(lower.get("preferredlabel") or ""),
                    alt_labels=alt,
                    alt_labels_norm=[_norm(a) for a in alt] if alt else None,
                    hidden_labels=hidden,
                    status=lower.get("status"),
                    modified_date=lower.get("modifieddate"),
                    scope_note=lower.get("scopenote"),
                    definition=lower.get("definition"),
                    in_scheme=lower.get("inscheme"),
                    description=lower.get("description"),
                )
            )
    await session.commit()


async def load_esco_occupations(session: AsyncSession, csv_path: str) -> None:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower(): v for k, v in row.items() if k}
            concept_uri = (lower.get("concepturi") or "").strip()
            if not concept_uri:
                continue
            session.add(
                EscoOccupation(
                    concept_uri=concept_uri,
                    concept_type=lower.get("concepttype"),
                    isco_group=lower.get("iscogroup"),
                    preferred_label=lower.get("preferredlabel"),
                    alt_labels=_split_labels(lower.get("altlabels")),
                    hidden_labels=_split_labels(lower.get("hiddenlabels")),
                    status=lower.get("status"),
                    modified_date=lower.get("modifieddate"),
                    regulated_profession_note=lower.get("regulatedprofessionnote"),
                    scope_note=lower.get("scopenote"),
                    definition=lower.get("definition"),
                    in_scheme=lower.get("inscheme"),
                    description=lower.get("description"),
                    code=lower.get("code"),
                )
            )
    await session.commit()


async def load_esco_skill_groups(session: AsyncSession, csv_path: str) -> None:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower(): v for k, v in row.items() if k}
            concept_uri = (lower.get("concepturi") or "").strip()
            if not concept_uri:
                continue
            session.add(
                EscoSkillGroup(
                    concept_uri=concept_uri,
                    concept_type=lower.get("concepttype"),
                    preferred_label=lower.get("preferredlabel"),
                    alt_labels=_split_labels(lower.get("altlabels")),
                    hidden_labels=_split_labels(lower.get("hiddenlabels")),
                    status=lower.get("status"),
                    modified_date=lower.get("modifieddate"),
                    scope_note=lower.get("scopenote"),
                    in_scheme=lower.get("inscheme"),
                    description=lower.get("description"),
                    code=lower.get("code"),
                )
            )
    await session.commit()


async def load_esco_skill_hierarchy(session: AsyncSession, csv_path: str) -> None:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            session.add(
                EscoSkillHierarchy(
                    level0_uri=row.get("Level 0 URI"),
                    level0_preferred_term=row.get("Level 0 preferred term"),
                    level1_uri=row.get("Level 1 URI"),
                    level1_preferred_term=row.get("Level 1 preferred term"),
                    level2_uri=row.get("Level 2 URI"),
                    level2_preferred_term=row.get("Level 2 preferred term"),
                    level3_uri=row.get("Level 3 URI"),
                    level3_preferred_term=row.get("Level 3 preferred term"),
                    description=row.get("Description"),
                    scope_note=row.get("Scope note"),
                    level0_code=row.get("Level 0 code"),
                    level1_code=row.get("Level 1 code"),
                    level2_code=row.get("Level 2 code"),
                    level3_code=row.get("Level 3 code"),
                )
            )
    await session.commit()


async def load_esco_language_skills(session: AsyncSession, csv_path: str) -> None:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower(): v for k, v in row.items() if k}
            concept_uri = (lower.get("concepturi") or "").strip()
            if not concept_uri:
                continue
            session.add(
                EscoLanguageSkill(
                    concept_uri=concept_uri,
                    concept_type=lower.get("concepttype"),
                    skill_type=lower.get("skilltype"),
                    reuse_level=lower.get("reuselevel"),
                    preferred_label=lower.get("preferredlabel"),
                    status=lower.get("status"),
                    alt_labels=_split_labels(lower.get("altlabels")),
                    description=lower.get("description"),
                    broader_concept_uri=lower.get("broaderconcepturi"),
                    broader_concept_pt=lower.get("broaderconceptpt"),
                )
            )
    await session.commit()
