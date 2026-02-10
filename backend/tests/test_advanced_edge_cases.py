import pytest
from sqlalchemy import select

from app.core import config as config_module
from app.models.gap import GapAnalysis, GapAnalysisStatus
from app.services import llm_service
from app.services.gap_analysis_service import _fingerprint, _normalize_text
from app.services.skill_matcher import match_skills


@pytest.mark.asyncio
async def test_retry_failed_status_resets_and_reprocesses(client, db_session, monkeypatch):
    called = {"count": 0}

    async def _fake_process(_gap_analysis_id):
        called["count"] += 1

    monkeypatch.setattr(llm_service, "process_gap_analysis", _fake_process)

    payload = {
        "resume_text": "A",
        "jd_text": "B",
        "model": "m",
        "prompt_version": "v1",
    }
    fp = _fingerprint(_normalize_text(payload["resume_text"]), _normalize_text(payload["jd_text"]), None)
    analysis = GapAnalysis(
        fingerprint=fp,
        resume_text=_normalize_text(payload["resume_text"]),
        jd_text=_normalize_text(payload["jd_text"]),
        status=GapAnalysisStatus.FAILED_LLM,
        model="m",
        prompt_version="v1",
    )
    db_session.add(analysis)
    await db_session.commit()

    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "PENDING"
    assert called["count"] == 1


class EmptyEmbeddingProvider:
    name = "empty"
    model = "empty-1"

    async def embed_texts(self, texts):
        return []


@pytest.mark.asyncio
async def test_embedding_empty_falls_back(monkeypatch):
    from app.services import skill_matcher as skill_matcher_module

    monkeypatch.setattr(skill_matcher_module, "get_embedding_provider", lambda: EmptyEmbeddingProvider())
    async def _empty_map():
        return {}
    monkeypatch.setattr(skill_matcher_module, "get_skill_taxonomy_map_db", _empty_map)
    monkeypatch.setattr(skill_matcher_module, "extract_skills", lambda _text: [])
    missing, match_percent, reason, *_ = await match_skills(
        resume_text="python docker",
        jd_text="python docker kubernetes",
        jd_skills_override=None,
    )
    assert "Fallback token match" in reason
    assert isinstance(missing, list)
    assert match_percent >= 0


@pytest.mark.asyncio
async def test_suggest_occupations_empty_embeddings_fallback(monkeypatch, db_session):
    from app.services import skill_matcher as skill_matcher_module

    monkeypatch.setattr(skill_matcher_module, "get_embedding_provider", lambda: EmptyEmbeddingProvider())
    rows = [("1", "software engineer"), ("2", "data engineer")]
    res = await skill_matcher_module.suggest_occupations_semantic("software engineer", rows)
    assert isinstance(res, list)


def test_prompt_truncation(monkeypatch):
    from app.services import llm_service as llm_module

    original = config_module.settings.max_prompt_chars
    monkeypatch.setattr(config_module.settings, "max_prompt_chars", 100)
    prompt = llm_module._build_prompt("a" * 200, "b" * 200, None)
    assert len(prompt) <= 100
    monkeypatch.setattr(config_module.settings, "max_prompt_chars", original)


@pytest.mark.asyncio
async def test_translation_empty_raises(monkeypatch, db_session):
    monkeypatch.setattr(config_module.settings, "translate_enabled", True)

    async def _empty_translate(_text, _provider):
        return ""

    monkeypatch.setattr(llm_service, "_translate_text", _empty_translate)
    class _SessionCtx:
        async def __aenter__(self):
            return db_session
        async def __aexit__(self, exc_type, exc, tb):
            return False
    monkeypatch.setattr(llm_service, "AsyncSessionLocal", lambda: _SessionCtx())

    analysis = GapAnalysis(
        fingerprint="x" * 64,
        resume_text="resume",
        jd_text="jd",
        status=GapAnalysisStatus.PENDING,
        model="m",
        prompt_version="v1",
    )
    db_session.add(analysis)
    await db_session.commit()

    await llm_service.process_gap_analysis(analysis.id)
    row = (await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis.id))).scalars().first()
    assert row.status == GapAnalysisStatus.FAILED_VALIDATION
