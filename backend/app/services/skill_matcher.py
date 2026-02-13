from __future__ import annotations

import math
import re
from typing import Iterable

from app.core.config import settings
from app.services.skill_taxonomy import extract_missing_skills, extract_skills, get_skill_taxonomy, get_skill_taxonomy_map_db
from app.db.session import AsyncSessionLocal


_WORD_RE = re.compile(r"[a-z0-9\+\#\-\.]+")
_SKILL_STOPWORDS = {"on", "of", "and", "the", "for", "to", "in", "with", "as", "at"}


def _tokens(text: str) -> list[str]:
    raw = _WORD_RE.findall(text.lower())
    return [t for t in raw if len(t) >= 3]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def match_skills(
    resume_text: str,
    jd_text: str,
    jd_skills_override: list[str] | None = None,
) -> tuple[list[str], float, str, list[str], list[str], list[str], list[str]]:
    if jd_skills_override:
        jd_skills = [s.strip() for s in jd_skills_override if s.strip()]
        resume_skills = set(extract_skills(resume_text))
        matched = [s for s in jd_skills if s in resume_skills]
        missing = [s for s in jd_skills if s not in resume_skills]
        missing, removed = _filter_missing_by_resume_text(missing, resume_text)
        total = max(len(jd_skills), 1)
        match_percent = round((1 - (len(missing) / total)) * 100, 2)
        reason = "HR override used for JD skills."
        if removed:
            reason = f"{reason} Resume text filter applied."
        top = _top_priority_from_matches(matched, jd_text)
        technical, transversal_soft, language = (missing, [], [])
        missing, match_percent, reason = await _apply_language_requirements(
            missing, total, jd_text, resume_text, reason
        )
        technical, transversal_soft, language = (missing, [], [])
        return missing, match_percent, reason, top, technical, transversal_soft, language
    taxonomy_map = await get_skill_taxonomy_map_db()
    if taxonomy_map:
        taxonomy_jd = _extract_skills_from_map(jd_text, taxonomy_map)
    else:
        taxonomy_jd = extract_skills(jd_text)
    if taxonomy_jd:
        if taxonomy_map:
            taxonomy_resume = set(_extract_skills_from_map(resume_text, taxonomy_map))
        else:
            taxonomy_resume = set(extract_skills(resume_text))
        if taxonomy_resume:
            matched = [s for s in taxonomy_jd if s in taxonomy_resume]
            missing = [s for s in taxonomy_jd if s not in taxonomy_resume]
            missing, removed_jd = _filter_missing_by_jd_text(missing, jd_text)
            missing, removed = _filter_missing_by_resume_text(missing, resume_text)
            total = max(len(taxonomy_jd), 1)
            match_percent = round((1 - (len(missing) / total)) * 100, 2)
            reason = f"ESCO taxonomy matched {total - len(missing)} of {total} JD skills."
            if removed or removed_jd:
                reason = f"{reason} Resume text filter applied."
            top = _top_priority_from_matches(matched, jd_text)
            missing, match_percent, reason = await _apply_language_requirements(
                missing, total, jd_text, resume_text, reason
            )
            technical, transversal_soft, language = (missing, [], [])
            return missing, match_percent, reason, top, technical, transversal_soft, language

        # Fallback: token overlap between JD skills and resume text
        resume_tokens = set(_tokens(resume_text))
        matched = [s for s in taxonomy_jd if _skill_in_tokens(s, resume_tokens)]
        missing = [s for s in taxonomy_jd if s not in matched]
        missing, removed_jd = _filter_missing_by_jd_text(missing, jd_text)
        missing, removed = _filter_missing_by_resume_text(missing, resume_text)
        total = max(len(taxonomy_jd), 1)
        match_percent = round((1 - (len(missing) / total)) * 100, 2)
        reason = f"ESCO JD skills matched by token overlap; {total - len(missing)} of {total}."
        if removed or removed_jd:
            reason = f"{reason} Resume text filter applied."
        top = _top_priority_from_matches(matched, jd_text)
        missing, match_percent, reason = await _apply_language_requirements(
            missing, total, jd_text, resume_text, reason
        )
        technical, transversal_soft, language = (missing, [], [])
        return missing, match_percent, reason, top, technical, transversal_soft, language

    return await _fallback_token_match(resume_text, jd_text)


async def _fallback_token_match(
    resume_text: str, jd_text: str
) -> tuple[list[str], float, str, list[str], list[str], list[str], list[str]]:
    taxonomy_map = await get_skill_taxonomy_map_db()
    if taxonomy_map:
        jd_skills = _extract_skills_from_map(jd_text, taxonomy_map)
        resume_skills = set(_extract_skills_from_map(resume_text, taxonomy_map))
    else:
        jd_skills = extract_skills(jd_text)
        resume_skills = set(extract_skills(resume_text))

    if not jd_skills:
        jd_skills = list(dict.fromkeys(_tokens(jd_text)))
    if not resume_skills:
        resume_skills = set(_tokens(resume_text))

    matched = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]
    missing, removed_jd = _filter_missing_by_jd_text(missing, jd_text)
    missing, removed = _filter_missing_by_resume_text(missing, resume_text)
    total = max(len(jd_skills), 1)
    match_percent = round((1 - (len(missing) / total)) * 100, 2)
    reason = f"Fallback token match on {total} JD skills; {len(missing)} missing identified."
    if removed or removed_jd:
        reason = f"{reason} Resume text filter applied."
    top = _top_priority_from_matches(matched, jd_text)
    technical, transversal_soft, language = (missing, [], [])
    missing, match_percent, reason = await _apply_language_requirements(
        missing, total, jd_text, resume_text, reason
    )
    technical, transversal_soft, language = (missing, [], [])
    return missing, match_percent, reason, top, technical, transversal_soft, language


def _skill_in_tokens(skill: str, tokens: set[str]) -> bool:
    parts = [p for p in _tokens(skill) if p not in _SKILL_STOPWORDS]
    if not parts:
        return False
    return all(p in tokens for p in parts)


def _contains_skill_phrase(text: str, skill: str) -> bool:
    parts = [p for p in _tokens(skill) if p not in _SKILL_STOPWORDS]
    if not parts:
        return False
    if len(parts) == 1:
        pattern = r"\b" + re.escape(parts[0]) + r"\b"
        return re.search(pattern, text) is not None
    phrase = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
    return re.search(phrase, text) is not None


def _filter_missing_by_resume_text(missing: list[str], resume_text: str) -> tuple[list[str], int]:
    if not missing:
        return missing, 0
    resume_tokens = set(_tokens(resume_text))
    resume_text_norm = " ".join(_WORD_RE.findall(resume_text.lower()))
    filtered: list[str] = []
    removed = 0
    for s in missing:
        if _skill_in_tokens(s, resume_tokens) or _contains_skill_phrase(resume_text_norm, s):
            removed += 1
            continue
        filtered.append(s)
    return filtered, removed


def _filter_missing_by_jd_text(missing: list[str], jd_text: str) -> tuple[list[str], int]:
    if not missing:
        return missing, 0
    jd_tokens = set(_tokens(jd_text))
    jd_text_norm = " ".join(_WORD_RE.findall(jd_text.lower()))
    filtered: list[str] = []
    removed = 0
    for s in missing:
        if not _skill_in_tokens(s, jd_tokens) and not _contains_skill_phrase(jd_text_norm, s):
            removed += 1
            continue
        filtered.append(s)
    return filtered, removed


async def detect_skills(resume_text: str, jd_text: str) -> tuple[list[str], list[str]]:
    taxonomy_map = await get_skill_taxonomy_map_db()
    if taxonomy_map:
        jd_skills = _extract_skills_from_map(jd_text, taxonomy_map)
        resume_skills = _extract_skills_from_map(resume_text, taxonomy_map)
    else:
        jd_skills = extract_skills(jd_text)
        resume_skills = extract_skills(resume_text)

    if not jd_skills:
        jd_skills = list(dict.fromkeys(_tokens(jd_text)))
    if not resume_skills:
        resume_skills = list(dict.fromkeys(_tokens(resume_text)))

    return jd_skills, resume_skills


def suggest_occupations(jd_text: str, occupations: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    jd_tokens = set(_tokens(jd_text))
    if not jd_tokens:
        return []
    scored: list[tuple[str, str, float]] = []
    for concept_uri, label in occupations:
        if not label:
            continue
        label_tokens = set(_tokens(label))
        if not label_tokens:
            continue
        overlap = len(jd_tokens & label_tokens)
        score = overlap / max(len(label_tokens), 1)
        if score > 0:
            scored.append((concept_uri, label, round(score, 3)))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:3]


async def suggest_occupations_semantic(
    resume_text: str,
    occupations: list[tuple[str, str]],
    candidate_limit: int = 200,
    top_k: int = 3,
) -> list[tuple[str, str, float]]:
    _ = candidate_limit, top_k
    return suggest_occupations(resume_text, occupations)


def _extract_skills_from_map(text: str, mapping: dict[str, str]) -> list[str]:
    tokens = _tokens(text)
    if not tokens:
        return []
    max_n = 4
    found: list[str] = []
    seen = set()
    words = list(tokens)
    for i in range(len(words)):
        for n in range(max_n, 0, -1):
            if i + n > len(words):
                continue
            phrase = " ".join(words[i : i + n])
            if phrase in mapping and phrase not in seen:
                found.append(mapping[phrase])
                seen.add(phrase)
    return found


def _top_priority_from_matches(matched_skills: list[str], jd_text: str, limit: int = 5) -> list[str]:
    if not matched_skills:
        return []
    lower = jd_text.lower()
    scores: list[tuple[str, int]] = []
    for s in matched_skills:
        count = lower.count(s.lower())
        scores.append((s, count))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, c in scores if c > 0]
    return top[:limit] if top else matched_skills[:limit]


async def _split_skill_categories(
    skills: list[str],
) -> tuple[list[str], list[str], list[str]]:
    return skills, [], []


async def _detect_language_skills(text: str) -> list[str]:
    _ = text
    return []


async def _apply_language_requirements(
    missing: list[str],
    total: int,
    jd_text: str,
    resume_text: str,
    reason: str,
) -> tuple[list[str], float, str]:
    _ = jd_text, resume_text
    match_percent = round((1 - (len(missing) / max(total, 1))) * 100, 2)
    return missing, match_percent, reason


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = prev[j] + 1
            delete = curr[j - 1] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _fuzzy_contains(text: str, label: str) -> bool:
    t_tokens = text.split()
    l_tokens = label.split()
    if not l_tokens:
        return False
    if len(l_tokens) == 1:
        target = l_tokens[0]
        for token in t_tokens:
            if abs(len(token) - len(target)) <= 1 and _levenshtein(token, target) <= 1:
                return True
        return False
    for lt in l_tokens:
        matched = False
        for token in t_tokens:
            if abs(len(token) - len(lt)) <= 1 and _levenshtein(token, lt) <= 1:
                matched = True
                break
        if not matched:
            return False
    return True
