from pydantic import BaseModel
import os
from urllib.parse import quote_plus
from pathlib import Path
from dotenv import load_dotenv


def _build_database_url(prefix: str, host: str, port: str, db: str, user: str, password: str) -> str:
    safe_password = quote_plus(password)
    return f"{prefix}://{user}:{safe_password}@{host}:{port}/{db}"


def _getenv_nonempty(key: str) -> str | None:
    value = os.getenv(key)
    if value is None:
        return None
    if not str(value).strip():
        return None
    return value


def _strip_inline_comment(value: str) -> str:
    if "#" in value:
        value = value.split("#", 1)[0]
    return value.strip()


_base_dir = Path(__file__).resolve().parents[2]
load_dotenv(_base_dir / ".env")
load_dotenv(_base_dir.parent / ".env")


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "gap-architect")
    api_v1_str: str = os.getenv("API_V1_STR", "/api/v1")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    postgres_host: str = _getenv_nonempty("POSTGRES_HOST") or "localhost"
    postgres_port: str = _getenv_nonempty("POSTGRES_PORT") or "5432"
    postgres_db: str = _getenv_nonempty("POSTGRES_DB") or "gap_architect"
    postgres_user: str = _getenv_nonempty("POSTGRES_USER") or "postgres"
    postgres_password: str = _getenv_nonempty("POSTGRES_PASSWORD") or ""

    database_url: str = _getenv_nonempty("DATABASE_URL") or _build_database_url(
        "postgresql+psycopg",
        postgres_host,
        postgres_port,
        postgres_db,
        postgres_user,
        postgres_password,
    )
    async_database_url: str = _getenv_nonempty("ASYNC_DATABASE_URL") or _build_database_url(
        "postgresql+asyncpg",
        postgres_host,
        postgres_port,
        postgres_db,
        postgres_user,
        postgres_password,
    )
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]

    max_resume_chars: int = int(os.getenv("MAX_RESUME_CHARS", "20000"))
    max_jd_chars: int = int(os.getenv("MAX_JD_CHARS", "20000"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    local_llm_timeout_seconds: float = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120"))
    processing_timeout_seconds: int = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "900"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    rate_limit_max_keys: int = int(os.getenv("RATE_LIMIT_MAX_KEYS", "10000"))
    rate_limit_key_header: str = os.getenv("RATE_LIMIT_KEY_HEADER", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    task_queue: str = os.getenv("TASK_QUEUE", "background").lower()
    celery_broker_url: str = os.getenv("CELERY_BROKER_URL", redis_url)
    celery_result_backend: str = os.getenv("CELERY_RESULT_BACKEND", redis_url)
    llm_provider: str = _strip_inline_comment(os.getenv("LLM_PROVIDER", "heuristic"))
    llm_api_key: str = _strip_inline_comment(os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = _strip_inline_comment(
        os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com"))
    )
    llm_model: str = _strip_inline_comment(
        os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "llama3.1")
    _base_dir: Path = _base_dir
    esco_skills_csv: str = os.getenv(
        "ESCO_SKILLS_CSV",
        str(_base_dir / "datasets" / "skill_en.csv"),
    )
    canonical_skills_path: str = os.getenv(
        "CANONICAL_SKILLS_PATH",
        str(_base_dir / "datasets" / "canonical_skills.txt"),
    )
    use_esco: bool = os.getenv("USE_ESCO", "false").lower() == "true"
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "none")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_similarity_threshold: float = float(os.getenv("EMBEDDING_SIM_THRESHOLD", "0.55"))
    embedding_candidate_limit: int = int(os.getenv("EMBEDDING_CANDIDATE_LIMIT", "250"))
    translate_enabled: bool = os.getenv("TRANSLATE_ENABLED", "false").lower() == "true"
    translate_target_lang: str = os.getenv("TRANSLATE_TARGET_LANG", "en")
    esco_cache_ttl_seconds: int = int(os.getenv("ESCO_CACHE_TTL_SECONDS", "300"))
    max_prompt_chars: int = int(os.getenv("MAX_PROMPT_CHARS", "60000"))


settings = Settings()
