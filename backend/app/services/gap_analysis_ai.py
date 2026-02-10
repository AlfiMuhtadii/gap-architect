from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
import asyncio
import logging
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError, field_validator, conlist
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult, LlmRun, LlmRunStatus
from app.db.session import AsyncSessionLocal
from app.core.config import settings
from app.services.skill_matcher import match_skills
from app.services.esco_repository import get_skill_depth_map

logger = logging.getLogger("app.gap_analysis_ai")


class LlmProvider(Protocol):
    name: str
    model: str

    async def generate(self, prompt: str) -> str:
        ...


class ActionStep(BaseModel):
    title: str
    why: str
    deliverable: str

    @field_validator("title", "why", "deliverable")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


class InterviewQuestion(BaseModel):
    question: str
    focus_gap: str
    what_good_looks_like: str

    @field_validator("question", "focus_gap", "what_good_looks_like")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


class GapAnalysisAIResult(BaseModel):
    missing_skills: list[str]
    top_priority_skills: list[str]
    hard_skills_missing: list[str]
    soft_skills_missing: list[str]
    technical_skills_missing: list[str] | None = None
    transversal_soft_skills_missing: list[str] | None = None
    language_skills_missing: list[str] | None = None
    action_steps: conlist(ActionStep, min_length=0, max_length=3)
    interview_questions: conlist(InterviewQuestion, min_length=0, max_length=3)
    roadmap_markdown: str
    match_percent: float
    match_reason: str

    @field_validator("missing_skills")
    @classmethod
    def _skills_non_empty(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("missing_skills must contain non-empty strings")
        return v

    @field_validator("roadmap_markdown")
    @classmethod
    def _roadmap_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("roadmap_markdown must be non-empty")
        _validate_roadmap_markdown(v)
        return v

    @field_validator("match_percent")
    @classmethod
    def _match_percent_range(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError("match_percent must be between 0 and 100")
        return v

    @field_validator("match_reason")
    @classmethod
    def _match_reason_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("match_reason must be non-empty")
        return v

    @field_validator("top_priority_skills", "hard_skills_missing", "soft_skills_missing")
    @classmethod
    def _list_non_null(cls, v: list[str]) -> list[str]:
        if v is None:
            raise ValueError("must be a list")
        return v


class LlmCallError(RuntimeError):
    pass


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _parse_json(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("response must be a JSON string")
    text = raw.strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        fence = text.split("```")
        if len(fence) >= 3:
            text = fence[1]
            if text.strip().lower().startswith("json"):
                text = text.strip()[4:]
            text = text.strip()
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("raw"), str):
        return _parse_json(data["raw"])
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    return data


def _fallback_roadmap_markdown(missing_skills: list[str], steps: list[dict[str, str]]) -> str:
    missing = [s for s in missing_skills if s.strip()]
    def _step_fields(step: Any) -> dict[str, str]:
        if isinstance(step, dict):
            return {
                "title": str(step.get("title", "")).strip(),
                "why": str(step.get("why", "")).strip(),
                "deliverable": str(step.get("deliverable", "")).strip(),
            }
        if hasattr(step, "title") and hasattr(step, "why") and hasattr(step, "deliverable"):
            return {
                "title": str(getattr(step, "title", "")).strip(),
                "why": str(getattr(step, "why", "")).strip(),
                "deliverable": str(getattr(step, "deliverable", "")).strip(),
            }
        return {"title": "", "why": "", "deliverable": ""}

    steps_used = [_step_fields(s) for s in steps[:3] if s]
    gap_summary = (
        "The resume covers the core requirements, with only minor gaps."
        if not missing
        else (
            "The resume aligns with the role, but several JD skills are missing. "
            f"Priority gaps include {missing[0]}"
            + (f" and {missing[1]}." if len(missing) > 1 else ".")
        )
    )
    priority = "\n".join([f"- {s}" for s in missing[:3]]) if missing else "- None"
    steps_md = (
        "\n".join(
            [
                f"### Step {i+1}  {step['title']}\n**Why:** {step['why']}\n**Deliverable:** {step['deliverable']}"
                for i, step in enumerate(steps_used)
            ]
        )
        if steps_used
        else "No concrete steps generated."
    )
    outcomes = (
        "\n".join(
            [
                f"- Demonstrates applied skill in {missing[0]}.",
                f"- Produces reviewable evidence for {missing[1] if len(missing) > 1 else missing[0]}.",
                "- Shows readiness for senior-level implementation work.",
            ]
        )
        if missing
        else "- No outcomes generated."
    )
    learning_order = (
        "\n".join([f"{i+1}. {s}" for i, s in enumerate(missing[:3])]) if missing else "1. None"
    )
    return (
        "## Gap Summary\n"
        f"{gap_summary}\n\n## Priority Skills to Learn\n"
        f"{priority}\n\n## Concrete Steps\n"
        f"{steps_md}\n\n## Expected Outcomes / Readiness\n"
        f"{outcomes}\n\n## Suggested Learning Order\n"
        f"{learning_order}"
    )


def _inject_steps_into_roadmap(roadmap: str, steps: list[dict[str, str]]) -> str:
    import re

    if not roadmap:
        return roadmap
    if not steps:
        return roadmap

    steps_md = "\n".join(
        [
            f"### Step {idx+1}  {step.get('title','')}\n**Why:** {step.get('why','')}\n**Deliverable:** {step.get('deliverable','')}"
            for idx, step in enumerate(steps)
        ]
    )

    pattern = r"(## Concrete Steps\s*)(.*?)(\n## |\Z)"
    match = re.search(pattern, roadmap, flags=re.S)
    if not match:
        return roadmap
    prefix = match.group(1)
    suffix = match.group(3)
    return re.sub(pattern, f"{prefix}{steps_md}\n{suffix}", roadmap, flags=re.S)

def _fallback_action_steps(missing_skills: list[str]) -> list[dict[str, str]]:
    return []


def _fallback_interview_questions(missing_skills: list[str]) -> list[dict[str, str]]:
    return []


def _coerce_local_payload(parsed: dict[str, Any]) -> None:
    def _to_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return cleaned

    missing = _to_str_list(parsed.get("missing_skills"))
    parsed["missing_skills"] = missing

    def _ensure_list(name: str) -> None:
        parsed[name] = _to_str_list(parsed.get(name))

    _ensure_list("top_priority_skills")
    _ensure_list("hard_skills_missing")
    _ensure_list("soft_skills_missing")

    if "technical_skills_missing" not in parsed:
        parsed["technical_skills_missing"] = None
    if "transversal_soft_skills_missing" not in parsed:
        parsed["transversal_soft_skills_missing"] = None
    if "language_skills_missing" not in parsed:
        parsed["language_skills_missing"] = None

    steps = parsed.get("action_steps")
    if not isinstance(steps, list):
        parsed["action_steps"] = _fallback_action_steps(missing)
    else:
        fixed_steps = []
        for step in steps[:3]:
            title = str(step.get("title", "")).strip()
            why = str(step.get("why", "")).strip()
            deliverable = str(step.get("deliverable", "")).strip()
            if not title or not why or not deliverable:
                fixed_steps = _fallback_action_steps(missing)
                break
            fixed_steps.append({"title": title, "why": why, "deliverable": deliverable})
        if fixed_steps:
            parsed["action_steps"] = fixed_steps

    questions = parsed.get("interview_questions")
    if not isinstance(questions, list):
        parsed["interview_questions"] = _fallback_interview_questions(missing)
    else:
        fixed_questions = []
        for q in questions[:3]:
            question = str(q.get("question", "")).strip()
            focus = str(q.get("focus_gap", "")).strip()
            good = str(q.get("what_good_looks_like", "")).strip()
            if not question or not focus or not good:
                fixed_questions = _fallback_interview_questions(missing)
                break
            fixed_questions.append(
                {"question": question, "focus_gap": focus, "what_good_looks_like": good}
            )
        if fixed_questions:
            parsed["interview_questions"] = fixed_questions

    if not isinstance(parsed.get("match_percent"), (int, float)):
        parsed["match_percent"] = 0.0
    if not isinstance(parsed.get("match_reason"), str) or not parsed.get("match_reason"):
        parsed["match_reason"] = "Local LLM response normalized."
    if not parsed.get("top_priority_skills"):
        parsed["top_priority_skills"] = [s for s in missing[:3]]
    if parsed.get("top_priority_skills"):
        parsed["top_priority_skills"] = _to_str_list(parsed.get("top_priority_skills"))[:3]
    if not isinstance(parsed.get("hard_skills_missing"), list):
        parsed["hard_skills_missing"] = []
    if not isinstance(parsed.get("soft_skills_missing"), list):
        parsed["soft_skills_missing"] = []
    if not parsed["hard_skills_missing"] and missing:
        parsed["hard_skills_missing"] = missing[:]


def _filter_missing_against_jd_and_resume(
    missing: list[str], jd_text: str, resume_text: str
) -> list[str]:
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()
    filtered: list[str] = []
    for skill in missing:
        s = str(skill).strip()
        if not s:
            continue
        s_lower = s.lower()
        if s_lower in jd_lower and s_lower not in resume_lower:
            filtered.append(s)
    return filtered


def _validate_roadmap_markdown(text: str) -> None:
    import re

    # Normalize common heading variants from LLMs
    if "# Gap Summary" in text and "## Gap Summary" not in text:
        text = text.replace("# Gap Summary", "## Gap Summary", 1)
    if "# Priority Skills to Learn" in text and "## Priority Skills to Learn" not in text:
        text = text.replace("# Priority Skills to Learn", "## Priority Skills to Learn", 1)
    if "# Concrete Steps" in text and "## Concrete Steps" not in text:
        text = text.replace("# Concrete Steps", "## Concrete Steps", 1)
    if "# Expected Outcomes / Readiness" in text and "## Expected Outcomes / Readiness" not in text:
        text = text.replace("# Expected Outcomes / Readiness", "## Expected Outcomes / Readiness", 1)
    if "# Suggested Learning Order" in text and "## Suggested Learning Order" not in text:
        text = text.replace("# Suggested Learning Order", "## Suggested Learning Order", 1)

    required_headings = [
        "## Gap Summary",
        "## Priority Skills to Learn",
        "## Concrete Steps",
        "## Expected Outcomes / Readiness",
        "## Suggested Learning Order",
    ]
    positions: list[int] = []
    for heading in required_headings:
        idx = text.find(heading)
        if idx == -1:
            raise ValueError(f"roadmap_markdown missing heading: {heading}")
        positions.append(idx)
    if positions != sorted(positions):
        raise ValueError("roadmap_markdown headings out of order")


    def _section_slice(start: str, end: str | None) -> str:
        s = text.split(start, 1)[1]
        if end:
            s = s.split(end, 1)[0]
        return s.strip()

    gap_summary = _section_slice("## Gap Summary", "## Priority Skills to Learn")
    sentences = [s for s in re.split(r"[.!?]+", gap_summary) if s.strip()]
    if len(sentences) < 1:
        raise ValueError("roadmap_markdown Gap Summary must have at least 1 sentence")
    placeholder_markers = [
        "write 23",
        "provide exactly",
        "one concise sentence",
        "one measurable artifact",
        "clear action title",
    ]
    if any(marker in gap_summary.lower() for marker in placeholder_markers):
        raise ValueError("roadmap_markdown Gap Summary contains placeholders")

    priority = _section_slice("## Priority Skills to Learn", "## Concrete Steps")
    priority_items = [
        line
        for line in priority.splitlines()
        if line.strip().startswith(("- ", "* ", "+ "))
    ]
    if not (1 <= len(priority_items) <= 10):
        raise ValueError("roadmap_markdown Priority Skills must have 1-10 bullets")
    if any("provide exactly" in line.lower() for line in priority_items):
        raise ValueError("roadmap_markdown Priority Skills contains placeholders")

    steps = _section_slice("## Concrete Steps", "## Expected Outcomes / Readiness")
    step_titles = re.findall(r"^### Step [1-3][\s:]+.+$", steps, flags=re.M)
    if len(step_titles) < 1 or len(step_titles) > 3:
        raise ValueError("roadmap_markdown Concrete Steps must have 1-3 steps")
    if steps.count("**Why:**") < len(step_titles) or steps.count("**Deliverable:**") < len(step_titles):
        raise ValueError("roadmap_markdown Concrete Steps must include Why and Deliverable")
    if any(marker in steps.lower() for marker in placeholder_markers):
        raise ValueError("roadmap_markdown Concrete Steps contains placeholders")

    outcomes = _section_slice("## Expected Outcomes / Readiness", "## Suggested Learning Order")
    outcome_items = [
        line
        for line in outcomes.splitlines()
        if line.strip().startswith(("- ", "* ", "+ "))
    ]
    if len(outcome_items) < 1:
        raise ValueError("roadmap_markdown Expected Outcomes must have at least 1 bullet")
    if any("provide" in line.lower() for line in outcome_items):
        raise ValueError("roadmap_markdown Expected Outcomes contains placeholders")

    order = _section_slice("## Suggested Learning Order", None)
    order_items = [line for line in order.splitlines() if re.match(r"^[1-3]\.\s+", line.strip())]
    if len(order_items) < 1 or len(order_items) > 3:
        raise ValueError("roadmap_markdown Suggested Learning Order must have 1-3 items")
    if any("none" == line.strip().lower() for line in order_items):
        raise ValueError("roadmap_markdown Suggested Learning Order contains placeholders")


def _repair_prompt(base_prompt: str, raw_response: str, error: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "The previous response did not match the required JSON schema.\n"
        f"Validation error: {error}\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{"
        "\"missing_skills\": [\"string\"], "
        "\"top_priority_skills\": [\"string\"], "
        "\"hard_skills_missing\": [\"string\"], "
        "\"soft_skills_missing\": [\"string\"], "
        "\"action_steps\": [{\"title\":\"\",\"why\":\"\",\"deliverable\":\"\"}, ... x3], "
        "\"interview_questions\": [{\"question\":\"\",\"focus_gap\":\"\",\"what_good_looks_like\":\"\"}, ... x3], "
        "\"roadmap_markdown\": \"string\", "
        "\"match_percent\": 0, "
        "\"match_reason\": \"string\""
        "}\n\n"
        "Here is the previous response to repair:\n"
        f"{raw_response}"
    )


def _rank_prompt(missing_skills: list[str], jd_text: str) -> str:
    skills = ", ".join(missing_skills)
    if len(jd_text) > settings.max_prompt_chars:
        jd_text = jd_text[: settings.max_prompt_chars]
    return (
        "You are ranking missing skills by importance for the target job description.\n"
        f"Job description:\n{jd_text}\n\n"
        f"Missing skills: {skills}\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\"ranked\": [\"skill1\", \"skill2\", \"skill3\"]}\n"
        "Rules:\n"
        "- Output exactly 3 items.\n"
        "- Each item must be copied from the Missing skills list.\n"
    )


def _infer_context(jd_text: str) -> str:
    jd = jd_text.lower()
    tags: list[str] = []
    if "saas" in jd:
        tags.append("SaaS")
    if "erp" in jd:
        tags.append("ERP")
    if "microservice" in jd or "microservices" in jd:
        tags.append("microservices")
    if "api" in jd or "grpc" in jd or "rest" in jd:
        tags.append("API services")
    if "data" in jd:
        tags.append("data workflows")
    if "cloud" in jd or "aws" in jd or "gcp" in jd or "azure" in jd:
        tags.append("cloud deployment")
    if tags:
        return " and ".join(tags[:3])
    return "backend systems"


def _build_action_steps(missing_skills: list[str], jd_text: str) -> list[dict[str, str]]:
    context = _infer_context(jd_text)
    steps: list[dict[str, str]] = []
    for skill in missing_skills[:3]:
        steps.append(
            {
                "title": f"Build a {context} module using {skill}",
                "why": f"Demonstrates practical use of {skill} in a {context} context.",
                "deliverable": f"A small, reviewable project or case study showing {skill} applied to {context}.",
            }
        )
    return steps


def _build_interview_questions(missing_skills: list[str], jd_text: str) -> list[dict[str, str]]:
    context = _infer_context(jd_text)
    questions: list[dict[str, str]] = []
    for skill in missing_skills[:3]:
        questions.append(
            {
                "question": f"Explain how you would apply {skill} in a {context} system.",
                "focus_gap": skill,
                "what_good_looks_like": f"Clear tradeoffs, implementation details, and impact using {skill}.",
            }
        )
    return questions


def _count_skill_occurrences(skill: str, jd_text: str) -> int:
    import re

    tokens = re.findall(r"[a-z0-9\+\#\-\.]+", skill.lower())
    if not tokens:
        return 0
    if len(tokens) == 1:
        pattern = r"\b" + re.escape(tokens[0]) + r"\b"
        return len(re.findall(pattern, jd_text.lower()))
    phrase = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
    return len(re.findall(phrase, jd_text.lower()))


async def _pre_rank_missing_skills(
    session: AsyncSession, missing_skills: list[str], jd_text: str
) -> list[str]:
    if not missing_skills:
        return []
    import re
    depth_map = await get_skill_depth_map(session)
    scored: list[tuple[float, int, str]] = []
    stopwords = {"and", "or", "the", "a", "an", "of", "to", "in", "on", "for", "with", "by", "at"}
    for idx, skill in enumerate(missing_skills):
        tokens = re.findall(r"[a-z0-9\+\#\-\.]+", skill.lower())
        if not tokens:
            continue
        if len(tokens) == 1 and (tokens[0] in stopwords or len(tokens[0]) < 3):
            continue
        freq = _count_skill_occurrences(skill, jd_text)
        depth = depth_map.get(skill.lower(), 0)
        length_bonus = min(len(skill) / 10.0, 2.0)
        score = (freq * 2.0) + (depth * 1.0) + length_bonus
        scored.append((score, idx, skill))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored]


async def _rank_missing_skills(
    session: AsyncSession, missing_skills: list[str], jd_text: str
) -> list[str]:
    pre_ranked = await _pre_rank_missing_skills(session, missing_skills, jd_text)
    return pre_ranked[:3]



async def _llm_classify_skills(
    provider: LlmProvider,
    skills: list[str],
) -> tuple[list[str], list[str]] | None:
    if getattr(provider, "name", "") == "heuristic":
        return None
    if getattr(provider, "name", "") == "local_llm":
        return None
    if not skills:
        return [], []
    prompt = (
        "Classify the following skills into hard vs soft.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\"hard\": [\"...\"], \"soft\": [\"...\"]}\n"
        "Rules:\n"
        "- Only use skills from the input list.\n"
        "- Do not invent new skills.\n\n"
        f"SKILLS:\n{', '.join(skills)}"
    )
    try:
        raw = await provider.generate(prompt)
        data = _parse_json(raw)
        hard = [str(s).strip() for s in data.get("hard", []) if str(s).strip() in skills]
        soft = [str(s).strip() for s in data.get("soft", []) if str(s).strip() in skills]
        return list(dict.fromkeys(hard)), list(dict.fromkeys(soft))
    except Exception:
        return None

async def run_gap_analysis_ai(
    session: AsyncSession,
    gap_analysis_id: UUID,
    provider: LlmProvider,
    prompt: str,
) -> None:
    try:
        analysis = (await session.execute(select(GapAnalysis).where(GapAnalysis.id == gap_analysis_id))).scalars().first()
    except SQLAlchemyError as e:
        logger.error(
            f"Database error fetching gap_analysis {gap_analysis_id}",
            extra={"gap_analysis_id": str(gap_analysis_id), "error": str(e)},
            exc_info=True,
        )
        raise ValueError(f"Database error: {e}") from e
    
    if analysis is None:
        logger.warning(f"Gap analysis {gap_analysis_id} not found")
        raise ValueError("gap_analysis not found")
    analysis.processing_started_at = datetime.now(timezone.utc)
    await session.commit()

    def _provider_timeout() -> float:
        return (
            settings.local_llm_timeout_seconds
            if getattr(provider, "name", "") == "local_llm"
            else settings.llm_timeout_seconds
        )

    async def _call_llm(call_prompt: str) -> tuple[dict[str, Any] | None, str | None, int]:
        start = time.perf_counter()
        try:
            raw = await asyncio.wait_for(provider.generate(call_prompt), timeout=_provider_timeout())
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.perf_counter() - start) * 1000)
            error_text = str(exc) or repr(exc)
            raise LlmCallError(error_text) from exc

        duration_ms = int((time.perf_counter() - start) * 1000)
        try:
            parsed = _parse_json(raw)
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc) or repr(exc)
            return None, error_text, duration_ms

        return parsed, None, duration_ms

    try:
        parsed, error, duration_ms = await _call_llm(prompt)
        if parsed is None:
            if getattr(provider, "name", "") == "local_llm":
                raise ValueError("local_invalid_json")
            repaired_prompt = _repair_prompt(prompt, "", error or "invalid JSON")
            parsed, error, duration_ms = await _call_llm(repaired_prompt)
            if parsed is None:
                await _log_llm_run(
                    session=session,
                    gap_analysis_id=gap_analysis_id,
                    provider=provider.name,
                    model=provider.model,
                    request_hash=_hash_prompt(repaired_prompt),
                    response_json={"error": error or "invalid JSON after repair"},
                    status=LlmRunStatus.FAILED,
                    error_message=error or "invalid JSON after repair",
                    duration_ms=duration_ms,
                )
                await _mark_failed_validation(session, analysis, error or "invalid JSON after repair")
                return

        if getattr(provider, "name", "") == "local_llm":
            _coerce_local_payload(parsed)
            roadmap = parsed.get("roadmap_markdown") if isinstance(parsed, dict) else None
            try:
                if isinstance(roadmap, str):
                    _validate_roadmap_markdown(roadmap)
                else:
                    raise ValueError("roadmap missing")
            except Exception:
                parsed["roadmap_markdown"] = _fallback_roadmap_markdown(
                    parsed.get("missing_skills") or [],
                    parsed.get("action_steps") or [],
                )

        try:
            validated = GapAnalysisAIResult.model_validate(parsed)
        except ValidationError as exc:
            if getattr(provider, "name", "") == "local_llm":
                _coerce_local_payload(parsed)
                try:
                    validated = GapAnalysisAIResult.model_validate(parsed)
                except ValidationError as exc_local:
                    raise ValueError("local_invalid_json") from exc_local
            repaired_prompt = _repair_prompt(prompt, json.dumps(parsed, ensure_ascii=True), str(exc))
            parsed2, error2, duration_ms = await _call_llm(repaired_prompt)
            if parsed2 is None:
                await _log_llm_run(
                    session=session,
                    gap_analysis_id=gap_analysis_id,
                    provider=provider.name,
                    model=provider.model,
                    request_hash=_hash_prompt(repaired_prompt),
                    response_json={"error": error2 or "invalid JSON after repair"},
                    status=LlmRunStatus.FAILED,
                    error_message=error2 or "invalid JSON after repair",
                    duration_ms=duration_ms,
                )
                await _mark_failed_validation(session, analysis, error2 or "invalid JSON after repair")
                return
            try:
                validated = GapAnalysisAIResult.model_validate(parsed2)
                parsed = parsed2
            except ValidationError as exc2:
                await _mark_failed_validation(session, analysis, str(exc2))
                return
        await _log_llm_run(
            session=session,
            gap_analysis_id=gap_analysis_id,
            provider=provider.name,
            model=provider.model,
            request_hash=_hash_prompt(prompt),
            response_json=parsed,
            status=LlmRunStatus.SUCCESS,
            error_message=None,
            duration_ms=duration_ms,
        )

        try:
            await _persist_success(session, analysis, validated, provider)
        except Exception as exc:  # noqa: BLE001
            await session.refresh(analysis)
            if analysis.status == GapAnalysisStatus.DONE:
                return
            analysis.status = GapAnalysisStatus.FAILED_LLM
            analysis.error_message = str(exc)[:1000]
            analysis.last_error_at = datetime.now(timezone.utc)
            await session.commit()
            return
    except LlmCallError as exc:
        await _log_llm_run(
            session=session,
            gap_analysis_id=gap_analysis_id,
            provider=provider.name,
            model=provider.model,
            request_hash=_hash_prompt(prompt),
            response_json={"error": str(exc) or "llm_failed"},
            status=LlmRunStatus.FAILED,
            error_message=str(exc) or "llm_failed",
            duration_ms=0,
        )
        if "rate_limited" in str(exc):
            raise ValueError("rate_limited") from exc
        if getattr(provider, "name", "") == "local_llm":
            if "timeout" in str(exc).lower():
                raise ValueError("local_timeout") from exc
            raise ValueError("local_llm_failed") from exc
        if getattr(provider, "name", "") != "heuristic":
            raise ValueError("llm_failed") from exc
        await session.refresh(analysis)
        if analysis.status == GapAnalysisStatus.DONE:
            return
        analysis.status = GapAnalysisStatus.FAILED_LLM
        analysis.error_message = str(exc)
        analysis.last_error_at = datetime.now(timezone.utc)
        await session.commit()


async def _persist_success(
    session: AsyncSession,
    analysis: GapAnalysis,
    result: GapAnalysisAIResult,
    provider: LlmProvider,
) -> None:
    override = analysis.jd_skills_override if isinstance(analysis.jd_skills_override, list) else None
    missing, match_percent, match_reason, top, technical, transversal_soft, language = await match_skills(
        resume_text=analysis.resume_text,
        jd_text=analysis.jd_text,
        jd_skills_override=override,
    )
    use_matcher = settings.use_esco or getattr(provider, "name", "") == "heuristic"
    if not settings.use_esco and getattr(provider, "name", "") != "heuristic":
        missing = result.missing_skills
        match_percent = result.match_percent
        match_reason = result.match_reason
        top = result.top_priority_skills
        technical = result.technical_skills_missing or []
        transversal_soft = result.transversal_soft_skills_missing or []
        language = result.language_skills_missing or []

    if use_matcher:
        ranked = await _rank_missing_skills(session, missing, analysis.jd_text)
        missing_for_steps = ranked if ranked else (missing[:3] if missing else [])

        action_steps = _build_action_steps(missing_for_steps, analysis.jd_text)
        interview_questions = _build_interview_questions(missing_for_steps, analysis.jd_text)
    else:
        action_steps = result.action_steps if isinstance(result.action_steps, list) else []
        interview_questions = result.interview_questions

    if not action_steps:
        action_steps = _fallback_action_steps(missing)

    def _steps_to_dicts(steps: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for step in steps:
            if isinstance(step, dict):
                out.append(
                    {
                        "title": str(step.get("title", "")).strip(),
                        "why": str(step.get("why", "")).strip(),
                        "deliverable": str(step.get("deliverable", "")).strip(),
                    }
                )
            elif hasattr(step, "model_dump"):
                data = step.model_dump()
                out.append(
                    {
                        "title": str(data.get("title", "")).strip(),
                        "why": str(data.get("why", "")).strip(),
                        "deliverable": str(data.get("deliverable", "")).strip(),
                    }
                )
        return out

    def _questions_to_dicts(questions: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for q in questions:
            if isinstance(q, dict):
                out.append(
                    {
                        "question": str(q.get("question", "")).strip(),
                        "focus_gap": str(q.get("focus_gap", "")).strip(),
                        "what_good_looks_like": str(q.get("what_good_looks_like", "")).strip(),
                    }
                )
            elif hasattr(q, "model_dump"):
                data = q.model_dump()
                out.append(
                    {
                        "question": str(data.get("question", "")).strip(),
                        "focus_gap": str(data.get("focus_gap", "")).strip(),
                        "what_good_looks_like": str(data.get("what_good_looks_like", "")).strip(),
                    }
                )
        return out

    action_steps = _steps_to_dicts(action_steps)
    interview_questions = _questions_to_dicts(interview_questions)

    def _step_to_dict(step: Any) -> dict[str, str]:
        if isinstance(step, dict):
            return {
                "title": str(step.get("title", "")).strip(),
                "why": str(step.get("why", "")).strip(),
                "deliverable": str(step.get("deliverable", "")).strip(),
            }
        if hasattr(step, "title") and hasattr(step, "why") and hasattr(step, "deliverable"):
            return {
                "title": str(getattr(step, "title", "")).strip(),
                "why": str(getattr(step, "why", "")).strip(),
                "deliverable": str(getattr(step, "deliverable", "")).strip(),
            }
        return {"title": "", "why": "", "deliverable": ""}

    generated_roadmap = "\n".join(
        [
            f"{i+1}. **{_step_to_dict(step)['title']}**\n   - Why: {_step_to_dict(step)['why']}\n   - Deliverable: {_step_to_dict(step)['deliverable']}"
            for i, step in enumerate(action_steps)
        ]
    )
    roadmap_markdown = result.roadmap_markdown
    if not roadmap_markdown:
        roadmap_markdown = generated_roadmap
    else:
        if len(action_steps) >= 1:
            roadmap_markdown = _inject_steps_into_roadmap(roadmap_markdown, action_steps)

    if use_matcher:
        llm_classified = await _llm_classify_skills(provider, missing)
        if llm_classified:
            llm_hard, llm_soft = llm_classified
            def _merge_case_insensitive(base: list[str], extra: list[str]) -> list[str]:
                seen = {b.lower(): b for b in base}
                for s in extra:
                    key = s.lower()
                    if key not in seen:
                        seen[key] = s
                return list(seen.values())

            if llm_hard:
                filtered = [
                    s
                    for s in llm_hard
                    if s.lower() not in {x.lower() for x in transversal_soft + language}
                ]
                technical = _merge_case_insensitive(technical, filtered)
            if llm_soft:
                filtered = [
                    s
                    for s in llm_soft
                    if s.lower() not in {x.lower() for x in technical + language}
                ]
                transversal_soft = _merge_case_insensitive(transversal_soft, filtered)

    gap_result = GapResult(
        gap_analysis_id=analysis.id,
        missing_skills=missing,
        action_steps=action_steps,
        interview_questions=interview_questions,
        roadmap_markdown=roadmap_markdown,
        match_percent=match_percent,
        match_reason=match_reason,
        top_priority_skills=top,
        hard_skills_missing=technical,
        soft_skills_missing=(transversal_soft + language),
        technical_skills_missing=technical if technical else None,
        transversal_soft_skills_missing=transversal_soft if transversal_soft else None,
        language_skills_missing=language if language else None,
    )
    await session.refresh(analysis)
    if analysis.status == GapAnalysisStatus.DONE and analysis.gap_result is not None:
        return
    analysis.status = GapAnalysisStatus.DONE
    analysis.error_message = None
    analysis.last_error_at = None
    session.add(gap_result)
    try:
        await session.commit()
    except Exception:  # noqa: BLE001
        await session.rollback()
        existing = (
            await session.execute(
                select(GapResult).where(GapResult.gap_analysis_id == analysis.id)
            )
        ).scalars().first()
        if existing is not None:
            await session.refresh(analysis)
            analysis.status = GapAnalysisStatus.DONE
            analysis.error_message = None
            analysis.last_error_at = None
            await session.commit()
            return
        raise


async def _mark_failed_validation(session: AsyncSession, analysis: GapAnalysis, error_message: str) -> None:
    await session.refresh(analysis)
    if analysis.status == GapAnalysisStatus.DONE:
        return
    analysis.status = GapAnalysisStatus.FAILED_VALIDATION
    original_length = len(error_message)
    truncated_message = error_message[:1000]
    if original_length > 1000:
        logger.warning(
            f"Error message truncated for gap_analysis {analysis.id}",
            extra={
                "gap_analysis_id": str(analysis.id),
                "original_length": original_length,
                "truncated_length": len(truncated_message),
            },
        )
    analysis.error_message = truncated_message
    analysis.last_error_at = datetime.now(timezone.utc)
    await session.commit()


async def _log_llm_run(
    *,
    session: AsyncSession,
    gap_analysis_id: UUID,
    provider: str,
    model: str,
    request_hash: str,
    response_json: dict[str, Any],
    status: LlmRunStatus,
    error_message: str | None,
    duration_ms: int,
) -> None:
    run = LlmRun(
        gap_analysis_id=gap_analysis_id,
        provider=provider,
        model=model,
        request_hash=request_hash,
        response_json=response_json,
        status=status,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    async with AsyncSessionLocal() as log_session:
        log_session.add(run)
        await log_session.commit()
