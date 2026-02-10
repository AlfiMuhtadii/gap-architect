from __future__ import annotations

import csv
import logging
import os
import re
from functools import lru_cache
from typing import Iterable

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from sqlalchemy import select
from app.models.esco import EscoSkill

logger = logging.getLogger("app.skill_taxonomy")


_WORD_RE = re.compile(r"[a-z0-9\+\#\-\.]+")
_STOPWORDS = {
    "and",
    "the",
    "with",
    "for",
    "to",
    "in",
    "of",
    "a",
    "on",
    "is",
    "are",
    "be",
    "as",
    "or",
    "we",
    "you",
    "our",
    "looking",
    "seeking",
    "hire",
    "hiring",
    "need",
    "needs",
    "require",
    "required",
    "requirements",
    "role",
    "position",
    "candidate",
    "candidates",
    "responsibilities",
    "responsibility",
    "about",
    "team",
    "company",
    "job",
    "description",
    "purpose",
    "maintain",
    "supporting",
    "environment",
    "key",
}


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _load_esco_skills(csv_path: str) -> list[str]:
    skills: list[str] = []
    if not csv_path:
        return skills
    if not os.path.exists(csv_path):
        return skills
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lower = {k.lower(): v for k, v in row.items() if k}
            skill_type = (lower.get("skilltype") or "").lower()
            status = (lower.get("status") or "").lower()
            if skill_type and "skill" not in skill_type:
                continue
            if status and status not in {"released", "published", "active"}:
                continue
            label = (
                lower.get("preferredlabel")
                or lower.get("skilllabel")
                or lower.get("label")
                or lower.get("preferred_label")
                or lower.get("preferredlabel@en")
                or ""
            )
            label = label.strip()
            if not label:
                continue
            norm = _normalize(label)
            if not norm:
                continue
            if len(norm) < 3:
                continue
            if norm in _STOPWORDS:
                continue
            if len(norm.split()) == 1 and norm in _STOPWORDS:
                continue
            skills.append(label)

            alt_labels = (lower.get("altlabels") or "").strip()
            if alt_labels:
                for alt in re.split(r"[\n;|,]+", alt_labels):
                    alt = alt.strip()
                    if alt:
                        skills.append(alt)

            hidden_labels = (lower.get("hiddenlabels") or "").strip()
            if hidden_labels:
                for alt in re.split(r"[\n;|,]+", hidden_labels):
                    alt = alt.strip()
                    if alt:
                        skills.append(alt)
    return skills


def _load_canonical_skills(path: str) -> list[str]:
    skills: list[str] = []
    if not path:
        return skills
    if not os.path.exists(path):
        return skills
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            norm = _normalize(line)
            if not norm:
                continue
            if len(norm) < 3:
                continue
            if norm in _STOPWORDS:
                continue
            skills.append(line)
    return skills


@lru_cache(maxsize=1)
def get_skill_taxonomy() -> list[str]:
    if settings.use_esco:
        return _load_esco_skills(settings.esco_skills_csv)
    return _load_canonical_skills(settings.canonical_skills_path)


async def get_skill_taxonomy_db() -> list[str]:
    if not settings.use_esco:
        return []
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(EscoSkill.preferred_label))
        return [r[0] for r in res.fetchall() if r[0]]


@lru_cache(maxsize=1)
def get_skill_taxonomy_map() -> dict[str, str]:
    skills = get_skill_taxonomy()
    mapping: dict[str, str] = {}
    for s in skills:
        norm = _normalize(s)
        if norm:
            mapping[norm] = s
    return mapping


async def get_skill_taxonomy_map_db() -> dict[str, str]:
    if not settings.use_esco:
        return {}
    mapping: dict[str, str] = {}
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(EscoSkill.preferred_label, EscoSkill.preferred_label_norm))
        for pref, pref_norm in res.fetchall():
            if pref and pref_norm:
                mapping[pref_norm] = pref
        res2 = await session.execute(select(EscoSkill.alt_labels, EscoSkill.alt_labels_norm))
        for alt, alt_norm in res2.fetchall():
            if isinstance(alt, list) and isinstance(alt_norm, list):
                if len(alt) != len(alt_norm):
                    logger.warning(
                        f"Mismatched alt_labels lengths: alt={len(alt)}, alt_norm={len(alt_norm)}",
                        extra={"alt_length": len(alt), "alt_norm_length": len(alt_norm)},
                    )
                    # Use minimum length to avoid data loss
                    min_len = min(len(alt), len(alt_norm))
                    alt = alt[:min_len]
                    alt_norm = alt_norm[:min_len]
                for a, n in zip(alt, alt_norm):
                    if a and n and n not in mapping:
                        mapping[n] = a
    return mapping


def extract_skills(text: str) -> list[str]:
    mapping = get_skill_taxonomy_map()
    if not mapping:
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []
    token_set = set(tokens)

    max_n = 4
    skill_set = set(mapping.keys())
    found: list[str] = []
    seen = set()

    for i in range(len(tokens)):
        for n in range(max_n, 0, -1):
            if i + n > len(tokens):
                continue
            phrase = " ".join(tokens[i : i + n])
            if phrase in skill_set and phrase not in seen:
                found.append(mapping[phrase])
                seen.add(phrase)

    # Non-contiguous match for multi-word skills (improves recall)
    for phrase in skill_set:
        if phrase in seen:
            continue
        parts = phrase.split()
        if len(parts) < 2:
            continue
        if all(p in token_set for p in parts):
            found.append(mapping[phrase])
            seen.add(phrase)
    return found


def extract_missing_skills(resume_text: str, jd_text: str) -> list[str]:
    resume_skills = set(extract_skills(resume_text))
    jd_skills = extract_skills(jd_text)
    return [s for s in jd_skills if s not in resume_skills]
