import pytest
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
        "resume_text": " ".join(["python"] * 55),
        "jd_text": " ".join(["docker"] * 55),
        "model": "m",
        "prompt_version": "v1",
    }
    fp = _fingerprint(_normalize_text(payload["resume_text"]), _normalize_text(payload["jd_text"]))
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
    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "PENDING"
    assert called["count"] == 1


@pytest.mark.asyncio
async def test_match_skills_token_fallback_without_taxonomy(monkeypatch):
    monkeypatch.setattr("app.services.skill_matcher.extract_skills", lambda _text: [])
    missing, match_percent, reason, *_ = match_skills(
        resume_text="python docker",
        jd_text="python docker kubernetes",
    )
    assert "Fallback token match" in reason
    assert isinstance(missing, list)
    assert match_percent >= 0


def test_prompt_truncation(monkeypatch):
    from app.services import llm_service as llm_module

    original = config_module.settings.max_prompt_chars
    monkeypatch.setattr(config_module.settings, "max_prompt_chars", 100)
    prompt = llm_module._build_prompt("a" * 200, "b" * 200)
    assert len(prompt) <= 100
    monkeypatch.setattr(config_module.settings, "max_prompt_chars", original)
