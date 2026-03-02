"""Gemini embedding provider with lazy import."""

from __future__ import annotations

import logging
import os
from typing import Any

from neural_memory.engine.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

# Known dimensions per model
_MODEL_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,
}

_DEFAULT_MODEL = "gemini-embedding-001"


class GeminiEmbedding(EmbeddingProvider):
    """Embedding provider backed by the Google Gemini Embeddings API.

    The ``google-genai`` package is imported lazily on first use so that the
    dependency is only required when this provider is actually selected.

    Supports task types for better retrieval quality:
    - RETRIEVAL_QUERY: for queries at recall time
    - RETRIEVAL_DOCUMENT: for neurons at encode/training time
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "A Gemini API key is required. Pass it directly or set "
                "the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        self._task_type = task_type
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        """Lazy-initialise the google-genai client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai is required for GeminiEmbedding. "
                    "Install it with: pip install 'neural-memory[embeddings-gemini]'"
                ) from exc

            self._client = genai.Client(api_key=self._api_key)

        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed a single text via the Gemini API."""
        client = self._ensure_client()
        response = await client.aio.models.embed_content(
            model=self._model,
            contents=text,
            config={"task_type": self._task_type},
        )
        return list(response.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via the Gemini API.

        The Gemini API supports batch input natively. Batches are capped
        at 100 items per request (API limit) and chunked automatically.
        """
        if not texts:
            return []

        client = self._ensure_client()
        all_embeddings: list[list[float]] = []

        # Gemini API allows at most 100 texts per batch request
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            response = await client.aio.models.embed_content(
                model=self._model,
                contents=chunk,
                config={"task_type": self._task_type},
            )
            all_embeddings.extend(list(emb.values) for emb in response.embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors for the configured model."""
        if self._model not in _MODEL_DIMENSIONS:
            logger.warning(
                "Unknown Gemini model %r — assuming 3072D. "
                "If wrong, existing embeddings will be incompatible.",
                self._model,
            )
        return _MODEL_DIMENSIONS.get(self._model, 3072)
