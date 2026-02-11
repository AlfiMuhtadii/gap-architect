import asyncio
from datetime import datetime, timedelta, timezone
import pytest

from app.core import config as config_module
from app.models.gap import GapAnalysis, GapAnalysisStatus
from app.services.gap_analysis_ai import run_gap_analysis_ai
from .factories import GapAnalysisPayloadFactory, make_gap_analysis
from app.services.gap_analysis_service import create_or_get_gap_analysis
from app.schemas.gap_analysis import GapAnalysisCreate
from app.services.gap_analysis_service import get_gap_analysis


@pytest.mark.asyncio
async def test_empty_input_422(client):
    payload = {"resume_text": "", "jd_text": "", "model": "m", "prompt_version": "v1"}
    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_large_input_rejected(client):
    payload = GapAnalysisPayloadFactory(
        resume_text="a" * (config_module.settings.max_resume_chars + 1),
        jd_text="b" * (config_module.settings.max_jd_chars + 1),
    ).build()
    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 422


class SlowProvider:
    name = "slow"
    model = "slow-1"

    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(0.2)
        return "{}"


@pytest.mark.asyncio
async def test_llm_timeout_failed_llm(db_session, monkeypatch):
    monkeypatch.setattr(config_module.settings, "llm_timeout_seconds", 0.05)
    analysis = make_gap_analysis()
    db_session.add(analysis)
    await db_session.commit()

    await run_gap_analysis_ai(db_session, analysis.id, SlowProvider(), prompt="x")

    assert analysis.status == GapAnalysisStatus.FAILED_LLM


@pytest.mark.asyncio
async def test_normalization_same_fingerprint(db_session):
    payload1 = GapAnalysisCreate(
        resume_text="Senior Python! Engineer",
        jd_text="Looking for python engineer.",
        model="m",
        prompt_version="v1",
    )
    out1 = await create_or_get_gap_analysis(db_session, payload1, background_tasks=None)
    payload2 = GapAnalysisCreate(
        resume_text="senior   python engineer",
        jd_text="looking for PYTHON engineer",
        model="m",
        prompt_version="v1",
    )
    out2 = await create_or_get_gap_analysis(db_session, payload2, background_tasks=None)
    assert out1.id == out2.id


@pytest.mark.asyncio
async def test_stuck_pending_moves_to_failed(db_session, monkeypatch):
    monkeypatch.setattr(config_module.settings, "processing_timeout_seconds", 1)
    analysis = make_gap_analysis(status=GapAnalysisStatus.PENDING)
    analysis.processing_started_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    db_session.add(analysis)
    await db_session.commit()

    res = await get_gap_analysis(db_session, analysis.id)
    assert res is not None
    assert res.status == GapAnalysisStatus.FAILED_LLM
