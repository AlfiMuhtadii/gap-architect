from __future__ import annotations

from typing import Protocol
import httpx

from app.core.config import settings


class EmbeddingProvider(Protocol):
    name: str
    model: str

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for openai embedding provider")
        url = f"{self.base_url}/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "input": texts,
        }
        timeout = httpx.Timeout(settings.llm_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or "data" not in data:
            raise ValueError("Embedding response missing data")
        embeddings_data = data.get("data")
        if not isinstance(embeddings_data, list) or not embeddings_data:
            raise ValueError("Embedding response data empty")
        if len(embeddings_data) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings_data)}")
        embeddings: list[list[float]] = []
        for item in embeddings_data:
            if not isinstance(item, dict) or "embedding" not in item:
                raise ValueError("Embedding item missing 'embedding'")
            embeddings.append(item["embedding"])
        return embeddings


def get_embedding_provider() -> EmbeddingProvider | None:
    if settings.embedding_provider.lower() == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_embedding_model,
        )
    return None
