"""BGE-M3 embedding provider (HTTP, dense, L2-normalized).

Talks to a self-hosted BGE-M3 FastAPI service exposed on a local endpoint:

    POST {base_url}/embed   body {"texts": [...]}   header Authorization: Bearer <key>
    -> {"embeddings": [[... 1024 floats ...], ...]}   (already L2-normalized)

Zero-vec guard (non-negotiable): on empty/failed input this RAISES rather than
returning a fabricated ``[0.0] * dim`` vector — a zero vector poisons top-k KNN.
Callers (content_worker / smem reindex) treat a raise as "leave neuron pending"
and back-fill on the next cycle.
"""

from __future__ import annotations

import os
from typing import Any

from surreal_memory.engine.embedding.provider import EmbeddingProvider
from surreal_memory.engine.embedding.retry import call_with_retry

_DEFAULT_MODEL = "bge-m3"
_DEFAULT_DIMENSION = 1024
_DEFAULT_BASE_URL = "http://127.0.0.1:18100"


class _TransientHTTPError(Exception):
    """Carries a status_code so retry.is_transient() can classify 429/5xx."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class BGEM3Embedding(EmbeddingProvider):
    """Embedding provider backed by a self-hosted BGE-M3 HTTP service.

    ``httpx`` is imported lazily so the dependency is only needed when this
    provider is actually selected.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        dimension: int | None = None,
        timeout: float = 60.0,
        base_url_env: str = "SURREAL_MEMORY_EMBEDDING_BASE_URL",
        api_key_env: str = "BGE_M3_API_KEY",
    ) -> None:
        self._model = model
        # Endpoint resolution order: explicit arg > SURREAL_MEMORY_EMBEDDING_ENDPOINT
        # > SURREAL_MEMORY_EMBEDDING_BASE_URL > default.
        #
        # ENDPOINT is the name the rest of the codebase already uses (the
        # OpenAI-compatible provider, the Stop hook, the reasoning distiller), so it
        # wins here too: two similarly-named variables for "where the embedding
        # service lives" is a setup where someone sets the wrong one and gets a
        # silent fallback to the default URL instead of an error. BASE_URL stays
        # supported so anyone who configured this provider earlier keeps working.
        self._base_url = (
            base_url
            or os.getenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT")
            or os.getenv(base_url_env)
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = (
            api_key or os.getenv(api_key_env) or os.getenv("SURREAL_MEMORY_EMBEDDING_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                f"A BGE-M3 API key is required. Pass it directly or set the "
                f"{api_key_env} environment variable."
            )
        env_dim = os.getenv("SURREAL_MEMORY_EMBEDDING_DIMENSION")
        resolved = dimension or (int(env_dim) if env_dim and int(env_dim) > 0 else 0)
        self._dimension = int(resolved) if resolved else _DEFAULT_DIMENSION
        self._timeout = timeout
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "httpx is required for BGEM3Embedding. Install it with: pip install httpx"
                ) from exc
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def _post_embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()

        async def _do() -> Any:
            resp = await client.post(f"{self._base_url}/embed", json={"texts": texts})
            if resp.status_code in (429, 500, 502, 503, 504):
                raise _TransientHTTPError(resp.status_code, f"BGE-M3 {resp.status_code}")
            resp.raise_for_status()
            return resp.json()

        data = await call_with_retry(_do, provider="BGE-M3")
        vecs = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            got = len(vecs) if isinstance(vecs, list) else "?"
            raise RuntimeError(f"BGE-M3 returned {got} vectors for {len(texts)} texts")
        for v in vecs:
            if not isinstance(v, list) or len(v) != self._dimension:
                raise RuntimeError(
                    f"BGE-M3 returned {len(v) if isinstance(v, list) else '?'}D, "
                    f"expected {self._dimension}D (zero-vec guard: reject mismatched vectors)"
                )
        return [list(v) for v in vecs]

    async def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError(
                "BGE-M3 embed called with empty text (zero-vec guard: never fabricate)"
            )
        return (await self._post_embed([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any((not t or not t.strip()) for t in texts):
            raise ValueError("BGE-M3 embed_batch received an empty text (zero-vec guard)")
        return await self._post_embed(texts)

    @property
    def dimension(self) -> int:
        return self._dimension
