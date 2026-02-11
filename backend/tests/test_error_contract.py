import pytest


@pytest.mark.asyncio
async def test_validation_error_contract(client):
    res = await client.post(
        "/api/v1/gap-analyses",
        json={"resume_text": "", "jd_text": "", "model": "m", "prompt_version": "v1"},
    )
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"
    assert data["error"]["message"]


@pytest.mark.asyncio
async def test_not_found_error_contract(client):
    res = await client.get("/api/v1/gap-analyses/00000000-0000-0000-0000-000000000000")
    assert res.status_code == 404
    data = res.json()
    assert data["error"]["code"] == "not_found"
