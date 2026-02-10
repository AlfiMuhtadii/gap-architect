from __future__ import annotations

import math
import re
from typing import Iterable

from app.core.config import settings
from app.services.embedding_provider import get_embedding_provider
from app.services.skill_taxonomy import extract_missing_skills, extract_skills, get_skill_taxonomy, get_skill_taxonomy_map_db
from app.services.esco_repository import get_soft_skill_labels, get_language_skill_labels
from app.db.session import AsyncSessionLocal


_WORD_RE = re.compile(r"[a-z0-9\+\#\-\.]+")
_SKILL_STOPWORDS = {"on", "of", "and", "the", "for", "to", "in", "with", "as", "at"}
_MAX_EMBEDDING_CANDIDATES = 500


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
        technical, transversal_soft, language = (
            await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
        )
        missing, match_percent, reason = await _apply_language_requirements(
            missing, total, jd_text, resume_text, reason
        )
        technical, transversal_soft, language = (
            await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
        )
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
            technical, transversal_soft, language = (
                await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
            )
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
        technical, transversal_soft, language = (
            await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
        )
        return missing, match_percent, reason, top, technical, transversal_soft, language

    provider = get_embedding_provider()
    if provider is None:
        return await _fallback_token_match(resume_text, jd_text)

    candidates = _candidate_skills(jd_text)
    if not candidates:
        return await _fallback_token_match(resume_text, jd_text)

    texts = [jd_text, resume_text] + candidates
    try:
        embeddings = await provider.embed_texts(texts)
    except Exception:  # noqa: BLE001
        return await _fallback_token_match(resume_text, jd_text)
    if not embeddings or len(embeddings) != len(texts):
        return await _fallback_token_match(resume_text, jd_text)
    jd_vec = embeddings[0]
    resume_vec = embeddings[1]
    skill_vecs = embeddings[2:]

    threshold = settings.embedding_similarity_threshold
    matched = 0
    missing: list[str] = []
    matched_skills: list[str] = []
    for skill, vec in zip(candidates, skill_vecs):
        sim_jd = _cosine(jd_vec, vec)
        sim_resume = _cosine(resume_vec, vec)
        if sim_jd >= threshold and sim_resume >= threshold:
            matched += 1
            matched_skills.append(skill)
        elif sim_jd >= threshold and sim_resume < threshold:
            missing.append(skill)

    missing, removed_jd = _filter_missing_by_jd_text(missing, jd_text)
    missing, removed = _filter_missing_by_resume_text(missing, resume_text)
    total = max(matched + len(missing), 1)
    match_percent = round((matched / total) * 100, 2)
    reason = f"Embedding match on {total} candidate skills; {len(missing)} missing identified."
    if removed or removed_jd:
        reason = f"{reason} Resume text filter applied."
    top = _top_priority_from_matches(matched_skills, jd_text)
    missing, match_percent, reason = await _apply_language_requirements(
        missing, total, jd_text, resume_text, reason
    )
    technical, transversal_soft, language = (
        await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
    )
    return missing, match_percent, reason, top, technical, transversal_soft, language


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
    technical, transversal_soft, language = (
        await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
    )
    missing, match_percent, reason = await _apply_language_requirements(
        missing, total, jd_text, resume_text, reason
    )
    technical, transversal_soft, language = (
        await _split_skill_categories(missing) if settings.use_esco else (missing, [], [])
    )
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

    if settings.use_esco:
        jd_lang = await _detect_language_skills(jd_text)
        resume_lang = await _detect_language_skills(resume_text)
        jd_skills = list(dict.fromkeys(jd_skills + jd_lang))
        resume_skills = list(dict.fromkeys(resume_skills + resume_lang))

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
    provider = get_embedding_provider()
    if provider is None:
        return suggest_occupations(resume_text, occupations)
    candidate_limit = max(0, min(candidate_limit, _MAX_EMBEDDING_CANDIDATES))

    resume_tokens = set(_tokens(resume_text))
    candidates: list[tuple[str, str]] = []
    if resume_tokens:
        for concept_uri, label in occupations:
            if not label:
                continue
            label_tokens = set(_tokens(label))
            if label_tokens & resume_tokens:
                candidates.append((concept_uri, label))
            if len(candidates) >= candidate_limit:
                break
    else:
        candidates = occupations[:candidate_limit]

    if not candidates:
        return []

    texts = [resume_text] + [label for _, label in candidates]
    embeddings = await provider.embed_texts(texts)
    if not embeddings or len(embeddings) != len(texts):
        return suggest_occupations(resume_text, occupations)
    resume_vec = embeddings[0]
    scores: list[tuple[str, str, float]] = []
    for (concept_uri, label), vec in zip(candidates, embeddings[1:]):
        score = _cosine(resume_vec, vec)
        scores.append((concept_uri, label, round(score, 4)))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def _candidate_skills(jd_text: str) -> list[str]:
    taxonomy = get_skill_taxonomy()
    if not taxonomy:
        return []
    jd_tokens = set(_tokens(jd_text))
    if not jd_tokens:
        return []
    limit = min(settings.embedding_candidate_limit, _MAX_EMBEDDING_CANDIDATES)
    candidates: list[str] = []
    for skill in taxonomy:
        skill_tokens = set(_tokens(skill))
        if skill_tokens & jd_tokens:
            candidates.append(skill)
        if len(candidates) >= limit:
            break
    return candidates


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
    if not skills:
        return [], [], []
    async with AsyncSessionLocal() as session:
        soft_labels = await get_soft_skill_labels(session)
        lang_labels = await get_language_skill_labels(session)
        soft_labels = soft_labels | lang_labels
    transversal_soft: list[str] = []
    technical: list[str] = []
    language: list[str] = []
    for s in skills:
        sl = " ".join(_WORD_RE.findall(s.lower()))
        if sl in lang_labels or any(l in sl or sl in l for l in lang_labels):
            language.append(s)
        elif sl in soft_labels:
            transversal_soft.append(s)
        else:
            technical.append(s)
    return technical, transversal_soft, language


async def _detect_language_skills(text: str) -> list[str]:
    t = " ".join(_WORD_RE.findall(text.lower()))
    if not t:
        return []
    t_tokens = t.split()
    t_token_set = set(t_tokens)
    if len(t_tokens) > 1200:
        # skip fuzzy matching for very long texts to avoid heavy Levenshtein cost
        t_tokens = t_tokens[:1200]
        t_token_set = set(t_tokens)
    async with AsyncSessionLocal() as session:
        lang_labels = await get_language_skill_labels(session)
    if not lang_labels:
        return []
    found: list[str] = []
    for label in lang_labels:
        if not label:
            continue
        if len(label) < 4:
            continue
        l_tokens = label.split()
        if len(l_tokens) > 1:
            if all(lt in t_token_set for lt in l_tokens):
                found.append(label)
            continue
        target = l_tokens[0]
        if target in t_token_set:
            found.append(label)
            continue
        if len(t_tokens) <= 1200 and len(target) >= 6 and _fuzzy_contains(t, label):
            found.append(label)
    return list(dict.fromkeys(found))


async def _apply_language_requirements(
    missing: list[str],
    total: int,
    jd_text: str,
    resume_text: str,
    reason: str,
) -> tuple[list[str], float, str]:
    jd_lang = await _detect_language_skills(jd_text)
    resume_lang = await _detect_language_skills(resume_text)
    if not jd_lang:
        return missing, round((1 - (len(missing) / max(total, 1))) * 100, 2), reason
    missing_lang = [l for l in jd_lang if l not in resume_lang]
    if missing_lang:
        missing = missing + [l for l in missing_lang if l not in missing]
        total = total + len(jd_lang)
        reason = f"{reason} Language requirements applied."
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
