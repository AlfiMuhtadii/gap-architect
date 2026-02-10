from __future__ import annotations

from sqlalchemy import select
import time
import re
from sqlalchemy.ext.asyncio import AsyncSession
import threading

from app.models.esco import EscoSkill, EscoSkillHierarchy, EscoLanguageSkill
from app.core.config import settings

_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def _get_cached(key: str):
    ttl = settings.esco_cache_ttl_seconds
    if ttl <= 0:
        return None
    with _cache_lock:
        entry = _cache.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > ttl:
        with _cache_lock:
            _cache.pop(key, None)
        return None
    return value


def _set_cached(key: str, value: object):
    ttl = settings.esco_cache_ttl_seconds
    if ttl <= 0:
        return
    with _cache_lock:
        _cache[key] = (time.time(), value)


SOFT_GROUP_HINTS = {
    "language skills and knowledge",
    "personal skills",
    "social skills and competences",
    "communication, collaboration and creativity",
    "transversal skills and competences",
}

SOFT_REUSE_LEVELS = {
    "transversal",
    "cross-sector",
}


async def get_esco_skill_labels(session: AsyncSession) -> set[str]:
    cached = _get_cached("esco_skill_labels")
    if cached is not None:
        return cached  # type: ignore[return-value]
    result = await session.execute(select(EscoSkill.preferred_label))
    labels = {r[0] for r in result.fetchall() if r[0]}
    data = {l.lower() for l in labels}
    _set_cached("esco_skill_labels", data)
    return data


async def get_soft_skill_labels(session: AsyncSession) -> set[str]:
    cached = _get_cached("soft_skill_labels")
    if cached is not None:
        return cached  # type: ignore[return-value]
    soft: set[str] = set()

    result = await session.execute(
        select(EscoSkill.preferred_label_norm, EscoSkill.reuse_level)
    )
    for label_norm, reuse in result.fetchall():
        if not label_norm:
            continue
        if reuse and reuse.lower() in SOFT_REUSE_LEVELS:
            soft.add(label_norm)

    if soft:
        _set_cached("soft_skill_labels", soft)
        return soft

    result = await session.execute(
        select(
            EscoSkillHierarchy.level0_preferred_term,
            EscoSkillHierarchy.level1_preferred_term,
            EscoSkillHierarchy.level2_preferred_term,
            EscoSkillHierarchy.level3_preferred_term,
        )
    )
    for l0, l1, l2, l3 in result.fetchall():
        labels = [l0, l1, l2, l3]
        if any(label and label.lower() in SOFT_GROUP_HINTS for label in labels):
            for label in labels:
                if label:
                    soft.add(_norm(label))
    _set_cached("soft_skill_labels", soft)
    return soft


async def get_language_skill_labels(session: AsyncSession) -> set[str]:
    cached = _get_cached("language_skill_labels")
    if cached is not None:
        return cached  # type: ignore[return-value]
    result = await session.execute(
        select(EscoLanguageSkill.preferred_label, EscoLanguageSkill.alt_labels, EscoLanguageSkill.broader_concept_pt)
    )
    labels: set[str] = set()
    for pref, alt, lang in result.fetchall():
        if pref:
            labels.add(_norm(pref))
        if lang:
            labels.add(_norm(lang))
        if isinstance(alt, list):
            for a in alt:
                if a:
                    labels.add(_norm(str(a)))
    _set_cached("language_skill_labels", labels)
    return labels


async def get_skill_depth_map(session: AsyncSession) -> dict[str, int]:
    cached = _get_cached("skill_depth_map")
    if cached is not None:
        return cached  # type: ignore[return-value]
    result = await session.execute(
        select(
            EscoSkillHierarchy.level0_preferred_term,
            EscoSkillHierarchy.level1_preferred_term,
            EscoSkillHierarchy.level2_preferred_term,
            EscoSkillHierarchy.level3_preferred_term,
        )
    )
    depth_map: dict[str, int] = {}
    for l0, l1, l2, l3 in result.fetchall():
        if l0:
            depth_map[_norm(l0)] = max(depth_map.get(_norm(l0), 0), 0)
        if l1:
            depth_map[_norm(l1)] = max(depth_map.get(_norm(l1), 0), 1)
        if l2:
            depth_map[_norm(l2)] = max(depth_map.get(_norm(l2), 0), 2)
        if l3:
            depth_map[_norm(l3)] = max(depth_map.get(_norm(l3), 0), 3)
    _set_cached("skill_depth_map", depth_map)
    return depth_map
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))
