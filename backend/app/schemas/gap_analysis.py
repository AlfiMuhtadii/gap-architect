from pydantic import BaseModel, Field, field_validator
from typing import Any
from uuid import UUID
from app.models.gap import GapAnalysisStatus
from app.core.config import settings


class GapAnalysisCreate(BaseModel):
    resume_text: str = Field(min_length=1, max_length=settings.max_resume_chars)
    jd_text: str = Field(min_length=1, max_length=settings.max_jd_chars)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    jd_skills_override: list[str] | None = None

    @field_validator("resume_text", "jd_text", mode="before")
    @classmethod
    def _strip_and_require_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        cleaned = " ".join(v.split())
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

    @field_validator("model", "prompt_version", mode="before")
    @classmethod
    def _validate_identifiers(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        import re
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", cleaned):
            raise ValueError("invalid format")
        return cleaned

    @field_validator("jd_skills_override", mode="before")
    @classmethod
    def _normalize_override(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("must be a list")
        cleaned = [s.strip() for s in v if isinstance(s, str)]
        cleaned = [s for s in cleaned if s]
        if not cleaned:
            return None
        # de-dup while preserving order
        seen = set()
        uniq = []
        for s in cleaned:
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            uniq.append(s)
        return uniq


class GapResultOut(BaseModel):
    missing_skills: Any
    top_priority_skills: list[str] | None = None
    action_steps: Any
    interview_questions: Any
    roadmap_markdown: str
    match_percent: float | None = None
    match_reason: str | None = None
    generation_meta: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class GapAnalysisOut(BaseModel):
    id: UUID
    status: GapAnalysisStatus
    result: GapResultOut | None = None

    class Config:
        from_attributes = True


class SkillDetectRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=settings.max_resume_chars)
    jd_text: str = Field(min_length=1, max_length=settings.max_jd_chars)


class SkillDetectResponse(BaseModel):
    jd_skills: list[str]
    resume_skills: list[str]
