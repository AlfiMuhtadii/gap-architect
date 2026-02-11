import hashlib
from typing import Optional
from datetime import datetime, timezone
from uuid import UUID
from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult
from app.schemas.gap_analysis import GapAnalysisCreate, GapAnalysisOut, GapResultOut
from app.services import llm_service
from app.core.config import settings


def _dispatch_processing(gap_analysis_id: UUID, background_tasks: Optional[BackgroundTasks]) -> None:
    if settings.task_queue == "celery":
        from app.tasks.gap_analysis import process_gap_analysis_task

        process_gap_analysis_task.delay(str(gap_analysis_id))
        return
    if background_tasks is not None:
        background_tasks.add_task(llm_service.process_gap_analysis, gap_analysis_id)
        return
    import asyncio

    asyncio.create_task(llm_service.process_gap_analysis(gap_analysis_id))


def _normalize_text(text: str) -> str:
    import re

    lowered = text.lower()
    # remove punctuation noise but keep alphanumerics and spaces
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return " ".join(cleaned.split()).strip()


def _fingerprint(resume_text: str, jd_text: str, jd_skills_override: list[str] | None = None) -> str:
    override_part = ""
    if jd_skills_override:
        override_part = "\n---\n" + "|".join(sorted(s.strip().lower() for s in jd_skills_override if s.strip()))
    payload = f"{resume_text}\n---\n{jd_text}{override_part}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_result_out(result: GapResult | None) -> GapResultOut | None:
    if not isinstance(result, GapResult):
        return None
    return GapResultOut(
        missing_skills=result.missing_skills,
        action_steps=result.action_steps,
        interview_questions=result.interview_questions,
        roadmap_markdown=result.roadmap_markdown,
        match_percent=result.match_percent,
        match_reason=result.match_reason,
        top_priority_skills=result.top_priority_skills,
        hard_skills_missing=result.hard_skills_missing,
        soft_skills_missing=result.soft_skills_missing,
        technical_skills_missing=result.technical_skills_missing,
        transversal_soft_skills_missing=result.transversal_soft_skills_missing,
        language_skills_missing=result.language_skills_missing,
        grouping_enabled=settings.use_esco,
    )


def _user_message_from_status(status: GapAnalysisStatus, error_message: str | None) -> str | None:
    if status == GapAnalysisStatus.FAILED_VALIDATION:
        return "AI response invalid. Please retry."
    if status == GapAnalysisStatus.FAILED_LLM:
        if error_message and "rate_limited" in error_message:
            return "AI rate limited. Please retry in a moment."
        return "AI processing failed. Please retry."
    return None


async def create_or_get_gap_analysis(
    session: AsyncSession,
    payload: GapAnalysisCreate,
    background_tasks: Optional[BackgroundTasks] = None,
) -> GapAnalysisOut:
    normalized_resume = _normalize_text(payload.resume_text)
    normalized_jd = _normalize_text(payload.jd_text)
    if not normalized_resume or not normalized_jd:
        raise ValueError("Resume and JD must contain alphanumeric characters")
    fingerprint = _fingerprint(normalized_resume, normalized_jd, payload.jd_skills_override)

    stmt = (
        select(GapAnalysis)
        .where(GapAnalysis.fingerprint == fingerprint)
        .options(selectinload(GapAnalysis.gap_result))
    )
    existing = (await session.execute(stmt)).scalars().first()

    if existing and existing.status == GapAnalysisStatus.DONE:
        return GapAnalysisOut(
            id=existing.id,
            status=existing.status,
            result=_build_result_out(existing.gap_result),
            error_message=existing.error_message,
            user_message=_user_message_from_status(existing.status, existing.error_message),
        )

    if existing and existing.status == GapAnalysisStatus.PENDING:
        return GapAnalysisOut(
            id=existing.id,
            status=existing.status,
            result=None,
            error_message=existing.error_message,
            user_message=_user_message_from_status(existing.status, existing.error_message),
        )

    if existing and existing.status in (GapAnalysisStatus.FAILED_LLM, GapAnalysisStatus.FAILED_VALIDATION):
        existing.status = GapAnalysisStatus.PENDING
        existing.error_message = None
        await session.commit()
        _dispatch_processing(existing.id, background_tasks)
        return GapAnalysisOut(id=existing.id, status=existing.status, result=None)

    gap_analysis = GapAnalysis(
        fingerprint=fingerprint,
        resume_text=normalized_resume,
        jd_text=normalized_jd,
        status=GapAnalysisStatus.PENDING,
        model=payload.model,
        prompt_version=payload.prompt_version,
        jd_skills_override=payload.jd_skills_override,
    )
    session.add(gap_analysis)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = (await session.execute(stmt)).scalars().first()
        if existing and existing.status == GapAnalysisStatus.DONE:
            return GapAnalysisOut(
                id=existing.id,
                status=existing.status,
                result=_build_result_out(existing.gap_result),
                error_message=existing.error_message,
                user_message=_user_message_from_status(existing.status, existing.error_message),
            )
        if existing:
            return GapAnalysisOut(
                id=existing.id,
                status=existing.status,
                result=None,
                error_message=existing.error_message,
                user_message=_user_message_from_status(existing.status, existing.error_message),
            )
        raise
    await session.refresh(gap_analysis)

    _dispatch_processing(gap_analysis.id, background_tasks)

    return GapAnalysisOut(
        id=gap_analysis.id,
        status=gap_analysis.status,
        result=None,
        error_message=gap_analysis.error_message,
        user_message=_user_message_from_status(gap_analysis.status, gap_analysis.error_message),
    )


async def get_gap_analysis(session: AsyncSession, gap_analysis_id: UUID) -> GapAnalysisOut | None:
    stmt = (
        select(GapAnalysis)
        .where(GapAnalysis.id == gap_analysis_id)
        .options(selectinload(GapAnalysis.gap_result))
    )
    analysis = (await session.execute(stmt)).scalars().first()
    if analysis is None:
        return None
    if (
        analysis.status == GapAnalysisStatus.PENDING
        and analysis.processing_started_at is not None
        and (datetime.now(timezone.utc) - analysis.processing_started_at).total_seconds()
        > settings.processing_timeout_seconds
    ):
        analysis.status = GapAnalysisStatus.FAILED_LLM
        analysis.error_message = "processing_timeout"
        analysis.last_error_at = datetime.now(timezone.utc)
        await session.commit()
    result = analysis.gap_result
    result_out = (
        GapResultOut(
            missing_skills=result.missing_skills,
            action_steps=result.action_steps,
            interview_questions=result.interview_questions,
            roadmap_markdown=result.roadmap_markdown,
            match_percent=result.match_percent,
            match_reason=result.match_reason,
            top_priority_skills=result.top_priority_skills,
            hard_skills_missing=result.hard_skills_missing,
            soft_skills_missing=result.soft_skills_missing,
            technical_skills_missing=result.technical_skills_missing,
            transversal_soft_skills_missing=result.transversal_soft_skills_missing,
            language_skills_missing=result.language_skills_missing,
            grouping_enabled=settings.use_esco,
        )
        if isinstance(result, GapResult)
        else None
    )
    return GapAnalysisOut(
        id=analysis.id,
        status=analysis.status,
        result=result_out,
        error_message=analysis.error_message,
        user_message=_user_message_from_status(analysis.status, analysis.error_message),
    )
