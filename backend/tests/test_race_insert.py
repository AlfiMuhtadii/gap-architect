import asyncio
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.services.gap_analysis_service import create_or_get_gap_analysis
from app.schemas.gap_analysis import GapAnalysisCreate
from app.services import llm_service


@pytest.mark.asyncio
async def test_race_insert_same_fingerprint(engine, monkeypatch):
    async def _noop(_id):
        return None

    monkeypatch.setattr(llm_service, "process_gap_analysis", _noop)

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    payload = GapAnalysisCreate(
        resume_text="Senior Python Engineer",
        jd_text="Looking for a Python engineer",
        model="m",
        prompt_version="v1",
    )

    async def _create():
        async with maker() as session:
            return await create_or_get_gap_analysis(session, payload, background_tasks=None)

    out1, out2 = await asyncio.gather(_create(), _create())
    assert out1.id == out2.id
