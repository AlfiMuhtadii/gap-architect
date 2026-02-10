from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_async_session
from app.schemas.gap_analysis import (
    GapAnalysisCreate,
    GapAnalysisOut,
    SkillDetectRequest,
    SkillDetectResponse,
    OccupationSuggestResponse,
    OccupationSuggestion,
)
from app.services.skill_matcher import detect_skills, suggest_occupations_semantic
from sqlalchemy import select
from app.models.esco import EscoOccupation
from app.services.gap_analysis_service import create_or_get_gap_analysis, get_gap_analysis


router = APIRouter()


@router.post("/gap-analyses", response_model=GapAnalysisOut)
async def create_gap_analysis(
    payload: GapAnalysisCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
) -> GapAnalysisOut:
    try:
        return await create_or_get_gap_analysis(session=session, payload=payload, background_tasks=background_tasks)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/gap-analyses/detect-skills", response_model=SkillDetectResponse)
async def detect_gap_skills(payload: SkillDetectRequest) -> SkillDetectResponse:
    jd_skills, resume_skills = await detect_skills(payload.resume_text, payload.jd_text)
    return SkillDetectResponse(jd_skills=jd_skills, resume_skills=resume_skills)


@router.post("/gap-analyses/suggest-occupations", response_model=OccupationSuggestResponse)
async def suggest_gap_occupations(
    payload: SkillDetectRequest, session: AsyncSession = Depends(get_async_session)
) -> OccupationSuggestResponse:
    rows = await session.execute(select(EscoOccupation.concept_uri, EscoOccupation.preferred_label))
    suggestions = await suggest_occupations_semantic(payload.resume_text, rows.fetchall())
    return OccupationSuggestResponse(
        suggestions=[OccupationSuggestion(concept_uri=uri, preferred_label=label, score=score) for uri, label, score in suggestions]
    )


@router.get("/gap-analyses/{gap_analysis_id}", response_model=GapAnalysisOut)
async def fetch_gap_analysis(
    gap_analysis_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> GapAnalysisOut:
    result = await get_gap_analysis(session=session, gap_analysis_id=gap_analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Not found")
    return result
