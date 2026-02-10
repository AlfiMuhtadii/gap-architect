import asyncio
import pytest
from sqlalchemy import select

from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult
from app.services.gap_analysis_ai import run_gap_analysis_ai
from .factories import make_gap_analysis


class ValidProvider:
    name = "valid"
    model = "valid-1"

    async def generate(self, prompt: str) -> str:
        return (
            '{'
            '"missing_skills":["a","b"],'
            '"top_priority_skills":["a"],'
            '"hard_skills_missing":["a"],'
            '"soft_skills_missing":["b"],'
            '"action_steps":[{"title":"t1","why":"w1","deliverable":"d1"},'
            '{"title":"t2","why":"w2","deliverable":"d2"},'
            '{"title":"t3","why":"w3","deliverable":"d3"}],'
            '"interview_questions":[{"question":"q1","focus_gap":"g1","what_good_looks_like":"w1"},'
            '{"question":"q2","focus_gap":"g2","what_good_looks_like":"w2"},'
            '{"question":"q3","focus_gap":"g3","what_good_looks_like":"w3"}],'
            '"roadmap_markdown":"rm",'
            '"match_percent":75.5,'
            '"match_reason":"Matched 3 of 4 skills"'
            '}'
        )


class InvalidProvider:
    name = "invalid"
    model = "invalid-1"

    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return "not-json"


@pytest.mark.asyncio
async def test_llm_json_valid_stores_gap_results(db_session):
    analysis = make_gap_analysis()
    db_session.add(analysis)
    await db_session.commit()

    await run_gap_analysis_ai(db_session, analysis.id, ValidProvider(), prompt="x")

    row = (await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis.id))).scalars().first()
    assert row.status == GapAnalysisStatus.DONE
    res = (await db_session.execute(select(GapResult).where(GapResult.gap_analysis_id == analysis.id))).scalars().first()
    assert res is not None
    assert res.roadmap_markdown


@pytest.mark.asyncio
async def test_llm_json_invalid_retry_then_failed_validation(db_session):
    analysis = make_gap_analysis()
    db_session.add(analysis)
    await db_session.commit()

    provider = InvalidProvider()
    await run_gap_analysis_ai(db_session, analysis.id, provider, prompt="x")

    row = (await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis.id))).scalars().first()
    assert row.status == GapAnalysisStatus.FAILED_VALIDATION
    assert provider.calls == 2
