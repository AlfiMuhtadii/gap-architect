from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_async_session
from app.schemas.gap_analysis import (
    GapAnalysisCreate,
    GapAnalysisOut,
    SkillDetectRequest,
    SkillDetectResponse,
)
from app.models.gap import GapAnalysisStatus
from app.services.skill_matcher import detect_skills
from app.services.gap_analysis_service import create_or_get_gap_analysis, get_gap_analysis


router = APIRouter()


@router.post("/gap-analyses", response_model=GapAnalysisOut)
async def create_gap_analysis(
    payload: GapAnalysisCreate,
    background_tasks: BackgroundTasks,
    response: Response,
    session: AsyncSession = Depends(get_async_session),
) -> GapAnalysisOut:
    try:
        result = await create_or_get_gap_analysis(session=session, payload=payload, background_tasks=background_tasks)
        response.status_code = 200 if result.status == GapAnalysisStatus.DONE else 201
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/gap-analyses/detect-skills", response_model=SkillDetectResponse)
async def detect_gap_skills(payload: SkillDetectRequest) -> SkillDetectResponse:
    jd_skills, resume_skills = await detect_skills(payload.resume_text, payload.jd_text)
    return SkillDetectResponse(jd_skills=jd_skills, resume_skills=resume_skills)


@router.get("/gap-analyses/{gap_analysis_id}", response_model=GapAnalysisOut)
async def fetch_gap_analysis(
    gap_analysis_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> GapAnalysisOut:
    result = await get_gap_analysis(session=session, gap_analysis_id=gap_analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Not found")
    return result
