"""OpenAI embedding provider with lazy import."""

from __future__ import annotations

import os
from typing import Any

from surreal_memory.engine.embedding.provider import EmbeddingProvider
from surreal_memory.engine.embedding.retry import call_with_retry

# Known dimensions per model
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Local llama.cpp / llamastash models served over the OpenAI-compatible API.
    "bge-m3": 1024,
    "bge-large-en-v1.5": 1024,
}

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API.

    The ``openai`` package is imported lazily on first use so that the
    dependency is only required when this provider is actually selected.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        provider_label: str = "OpenAI",
    ) -> None:
        self._model = model
        # Default to a local OpenAI-compatible embedding endpoint when
        # SURREAL_MEMORY_EMBEDDING_ENDPOINT is set and no explicit base_url is
        # passed (e.g. llamastash bge-m3 at http://127.0.0.1:11435/v1). Mirrors the
        # reranker's SURREAL_MEMORY_RERANKER_ENDPOINT so embeddings can be served by
        # a local server with no cloud round-trip. Subclasses that pass their own
        # base_url (e.g. OpenRouter) are unaffected.
        env_endpoint = os.environ.get("SURREAL_MEMORY_EMBEDDING_ENDPOINT", "").strip()
        resolved_base = base_url or env_endpoint
        self._base_url = resolved_base.rstrip("/") if resolved_base else None
        self._api_key_env = api_key_env
        self._provider_label = provider_label
        self._api_key = api_key or os.getenv(api_key_env)
        # A locally-configured endpoint (SURREAL_MEMORY_EMBEDDING_ENDPOINT, e.g.
        # llamastash / llama.cpp bge-m3) needs no real key, but the OpenAI SDK still
        # requires a non-empty string — fall back to a placeholder instead of failing
        # hard. This applies ONLY to that local-endpoint env knob: a subclass that
        # passes its own base_url (e.g. OpenRouter) still requires a real key.
        if not self._api_key and env_endpoint and not base_url:
            self._api_key = "sk-local"
        if not self._api_key:
            raise ValueError(
                f"A {provider_label} API key is required. Pass it directly or set "
                f"the {api_key_env} environment variable."
            )
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        """Lazy-initialise the async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required for OpenAIEmbedding. Install it with: pip install openai"
                ) from exc

            client_kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**client_kwargs)

        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed a single text via the OpenAI API."""
        client = self._ensure_client()
        response = await call_with_retry(
            lambda: client.embeddings.create(input=[text], model=self._model),
            provider=self._provider_label,
        )
        return list(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        The OpenAI API natively supports batch input, making this more
        efficient than the default sequential fallback.
        """
        if not texts:
            return []

        client = self._ensure_client()
        response = await call_with_retry(
            lambda: client.embeddings.create(input=texts, model=self._model),
            provider=self._provider_label,
        )
        # The API returns embeddings in the same order as the input
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [list(item.embedding) for item in sorted_data]

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors for the configured model."""
        return _MODEL_DIMENSIONS.get(self._model, 1536)
