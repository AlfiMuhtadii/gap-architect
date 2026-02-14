import pytest


@pytest.mark.asyncio
async def test_validate_input_rejects_short_non_tech(client):
    payload = {
        "resume_text": "Saya kerja di tim.",
        "jd_text": "Lowongan backend engineer.",
    }
    res = await client.post("/api/v1/gap-analyses/validate-input", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["is_valid"] is False
    assert "insufficient" in (body.get("error_message") or "").lower()


@pytest.mark.asyncio
async def test_validate_input_accepts_minimalist_tech_dense(client):
    payload = {
        "resume_text": "Python SQL AWS Docker Kubernetes FastAPI React PostgreSQL CI/CD testing",
        "jd_text": "Python SQL AWS Docker Kubernetes FastAPI React PostgreSQL observability security",
    }
    res = await client.post("/api/v1/gap-analyses/validate-input", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["is_valid"] is True
    assert body["resume_tech_entities"] >= 5
    assert body["jd_tech_entities"] >= 5


@pytest.mark.asyncio
async def test_validate_input_accepts_long_text_even_low_tech(client):
    resume = " ".join(["engineer"] * 55)
    jd = " ".join(["requirement"] * 60)
    res = await client.post(
        "/api/v1/gap-analyses/validate-input",
        json={"resume_text": resume, "jd_text": jd},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["is_valid"] is True
    assert body["resume_word_count"] >= 50
    assert body["jd_word_count"] >= 50
