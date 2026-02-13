from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Iterable

from app.core.config import settings

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
    return _load_canonical_skills(settings.canonical_skills_path)


async def get_skill_taxonomy_db() -> list[str]:
    return []


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
    return {}


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
