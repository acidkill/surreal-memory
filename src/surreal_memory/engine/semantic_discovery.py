"""Semantic synapse discovery — offline consolidation via embeddings.

Discovers SIMILAR_TO synapses between unconnected CONCEPT and ENTITY
neurons by computing cosine similarity on their embedding vectors.

This is an **offline consolidation** step, not a recall-time operation.
It enriches the neural graph so that spreading activation can later
traverse the discovered semantic links.

Optional: silently skips if sentence-transformers is not installed.
Discovered synapses decay 2x faster during pruning unless reinforced,
preventing stale semantic links from accumulating.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.edge_identity import deterministic_edge_id

if TYPE_CHECKING:
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Caps that bound work on very large brains. Semantic discovery now reads the
# STORED embedding on each neuron (no re-embedding), so it scales well.
MAX_NEURONS_TO_LINK = 10000  # max CONCEPT/ENTITY neurons considered per run
MAX_PAIRS_HARD_CAP = 5000  # absolute cap on synapses created per run
SEMANTIC_TOP_K = 5  # link each neuron to its K most-similar peers

# How many synapse rows to pull per page when snapshotting existing pairs.
# The snapshot has to cover the whole table -- it is what stops a second edge
# being laid over a pair already joined by any type -- but asking for it in one
# response is how the LIFECYCLE pass earned "[Errno 104] Connection reset by
# peer". Matches the neuron scan's page size just above.
_SYNAPSE_PAGE_SIZE = 5000


@dataclass(frozen=True)
class SemanticDiscoveryResult:
    """Result of a semantic discovery run."""

    neurons_embedded: int = 0
    pairs_evaluated: int = 0
    synapses_created: int = 0
    skipped_existing: int = 0
    synapses: list[Synapse] = field(default_factory=list)
    truncated: bool = False
    """True when the run stopped at ``semantic_discovery_max_pairs``.

    Without this a capped run and a saturated brain print the same number, so a
    constant "2000" reads as a stuck system when it is actually a backlog
    draining one capped run at a time.
    """


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _auto_detect_provider() -> tuple[str, str]:
    """Auto-detect the best available embedding provider.

    Detection order (first available wins):
    1. Ollama running locally (free, no install needed if Ollama app present)
    2. sentence-transformers installed (free, local, but ~440MB download)
    3. Gemini API key set (free tier available)
    4. OpenAI API key set (paid)
    5. OpenRouter API key set (OpenAI-compatible)

    Returns:
        Tuple of (provider_name, model_name).

    Raises:
        RuntimeError: If no provider is available.
    """
    # 1. Check Ollama (local server)
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            # Prefer multilingual embedding models
            preferred = ["bge-m3", "nomic-embed-text", "mxbai-embed-large", "all-minilm"]
            available_names = [m.get("name", "").split(":")[0] for m in models]
            for pref in preferred:
                if pref in available_names:
                    return ("ollama", pref)
            # Any model works as fallback
            if models:
                return ("ollama", "bge-m3")
    except Exception:
        logger.debug("Ollama embedding probe failed", exc_info=True)

    # 2. Check sentence-transformers (local)
    try:
        import sentence_transformers  # noqa: F401

        return ("sentence_transformer", "paraphrase-multilingual-MiniLM-L12-v2")
    except ImportError:
        pass

    # 3. Check Gemini API key (free tier)
    import os

    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        # Use the live default model, never a hardcoded literal that can drift to a
        # decommissioned model (text-embedding-004 now 404s and was 768-dim, a
        # dimension-mismatch landmine vs the 3072-dim gemini-embedding-001 default).
        from surreal_memory.engine.embedding.gemini_embedding import _DEFAULT_MODEL

        return ("gemini", _DEFAULT_MODEL)

    # 4. Check OpenAI API key (paid)
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", "text-embedding-3-small")

    # 5. Check OpenRouter API key (OpenAI-compatible)
    if os.environ.get("OPENROUTER_API_KEY"):
        return ("openrouter", "openai/text-embedding-3-small")

    raise RuntimeError(
        "No embedding provider available. Install one of:\n"
        "  pip install surreal-memory[embeddings]    # sentence-transformers (local, free)\n"
        "  ollama pull bge-m3                       # Ollama (local, free)\n"
        "  export GEMINI_API_KEY=...                # Google Gemini (free tier)\n"
        "  export OPENAI_API_KEY=...                # OpenAI (paid)\n"
        "  export OPENROUTER_API_KEY=...            # OpenRouter (OpenAI-compatible)"
    )


# Module-level singleton cache — avoids reloading models per tool call (#100)
# Keyed by (provider, model, endpoint) — see _create_provider's cache_key.
_provider_cache: dict[tuple[str, str, str], Any] = {}


def _effective_embedding(config: BrainConfig) -> tuple[bool, str, str]:
    """Resolve the EFFECTIVE embedding (enabled, provider, model).

    The stored ``brain.config`` is often stale (it keeps the defaults it was
    created with and is never re-synced when the user edits config.toml/env).
    "Effective config wins": prefer the unified embedding config
    (config.toml + env overrides) and fall back to the flat brain config only
    when the unified config cannot be loaded.
    """
    try:
        from surreal_memory.unified_config import get_config

        embedding = get_config().embedding
        return (embedding.enabled, embedding.provider, embedding.model)
    except Exception:
        logger.debug("Could not load unified embedding config — using brain config", exc_info=True)
        return (
            config.embedding_enabled,
            config.embedding_provider,
            config.embedding_model,
        )


def _effective_embedding_endpoint() -> str:
    """Resolve the EFFECTIVE embedding endpoint (config.toml + env override).

    Mirrors ``_effective_embedding``'s "effective config wins" pattern —
    ``EmbeddingSettings.resolved_endpoint()`` prefers ``[embedding] endpoint``
    in config.toml over ``SURREAL_MEMORY_EMBEDDING_ENDPOINT``. Without this,
    ``_create_provider`` never read the config value at all: it constructed
    ``OpenAIEmbedding(model=model_name)`` with no ``base_url``, so the config
    field was silently dead — only the env var ever reached the provider.
    Returns "" if the unified config cannot be loaded; callers already fall
    back to reading the env var themselves in that case (see
    ``OpenAIEmbedding.__init__``), so failing open to "no override" reproduces
    the exact pre-existing behaviour.
    """
    try:
        from surreal_memory.unified_config import get_config

        return get_config().embedding.resolved_endpoint()
    except Exception:
        logger.debug("Could not load unified embedding config for endpoint", exc_info=True)
        return ""


def _create_provider(config: BrainConfig, task_type: str = "RETRIEVAL_QUERY") -> Any:
    """Create or retrieve a cached embedding provider from BrainConfig.

    Providers are cached by (provider_name, model_name) so the model is loaded
    once per MCP process lifetime instead of once per tool call.

    Args:
        config: Brain configuration with embedding_provider and embedding_model.
        task_type: Task type hint for providers that support it (e.g. Gemini).

    Raises ImportError if the required package is not installed.
    """
    # "Effective config wins" — the stored brain config can be stale, so resolve
    # the provider/model from the unified config (config.toml + env overrides).
    _, provider_name, model_name = _effective_embedding(config)

    # Auto-detect best available provider
    if provider_name == "auto":
        provider_name, model_name = _auto_detect_provider()
        logger.info("Auto-detected embedding provider: %s (model: %s)", provider_name, model_name)

    endpoint = _effective_embedding_endpoint()
    # Endpoint rides in the cache key too: changing [embedding] endpoint in
    # config.toml (or the env override) must return a freshly built provider
    # pointed at the new address, not a provider cached under the old one.
    cache_key = (provider_name, model_name, endpoint)
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    provider: Any
    if provider_name == "sentence_transformer":
        from surreal_memory.engine.embedding.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        provider = SentenceTransformerEmbedding(model_name=model_name)
    elif provider_name == "openai":
        from surreal_memory.engine.embedding.openai_embedding import OpenAIEmbedding

        provider = OpenAIEmbedding(model=model_name, base_url=endpoint or None)
    elif provider_name == "openrouter":
        from surreal_memory.engine.embedding.openrouter_embedding import OpenRouterEmbedding

        provider = OpenRouterEmbedding(model=model_name)
    elif provider_name == "gemini":
        from surreal_memory.engine.embedding.gemini_embedding import GeminiEmbedding

        provider = GeminiEmbedding(model=model_name, task_type=task_type)
    elif provider_name == "ollama":
        from surreal_memory.engine.embedding.ollama_embedding import OllamaEmbedding

        provider = OllamaEmbedding(model=model_name, **({"base_url": endpoint} if endpoint else {}))
    elif provider_name in ("bge_m3", "bge-m3"):
        from surreal_memory.engine.embedding.bge_m3_embedding import BGEM3Embedding

        # base_url / api_key / dimension resolved from env (SURREAL_MEMORY_EMBEDDING_BASE_URL,
        # BGE_M3_API_KEY, SURREAL_MEMORY_EMBEDDING_DIMENSION) — see BGEM3Embedding.
        provider = BGEM3Embedding(model=model_name or "bge-m3")
    else:
        raise ValueError(f"Unknown embedding provider: {provider_name}")

    _provider_cache[cache_key] = provider
    return provider


async def discover_semantic_synapses(
    storage: NeuralStorage,
    config: BrainConfig,
) -> SemanticDiscoveryResult:
    """Discover SIMILAR_TO synapses between CONCEPT/ENTITY neurons.

    Uses the embedding vectors ALREADY STORED on each neuron
    (``metadata["_embedding"]`` / ``embedding_vec``) — it does NOT re-embed —
    so this is fast, does not depend on the embedding backend being reachable,
    and is cheap enough to run inside automatic consolidation. For each eligible
    neuron it links its top-K most-similar peers above the configured cosine
    threshold, skipping pairs that already share a synapse.

    Steps:
        1. Collect CONCEPT+ENTITY neurons that carry a stored embedding.
        2. Compute cosine similarity (vectorised via numpy when available,
           pure-python otherwise).
        3. Create SIMILAR_TO synapses for each neuron's top-K peers above
           threshold, up to ``semantic_discovery_max_pairs``.
    """
    effective_enabled, _, _ = _effective_embedding(config)
    if not effective_enabled:
        logger.debug("Embedding disabled — skipping semantic discovery")
        return SemanticDiscoveryResult()

    # Collect eligible neurons that already carry a stored embedding (no re-embed).
    batch_size = 5000
    offset = 0
    eligible: list[Neuron] = []
    vectors: list[list[float]] = []
    while True:
        batch = await storage.find_neurons(limit=batch_size, offset=offset)
        if not batch:
            break
        for n in batch:
            if n.type in (NeuronType.CONCEPT, NeuronType.ENTITY) and n.content.strip():
                emb = n.metadata.get("_embedding")
                if emb:
                    eligible.append(n)
                    vectors.append([float(x) for x in emb])
        offset += len(batch)
        if len(batch) < batch_size:
            break

    if len(eligible) < 2:
        return SemanticDiscoveryResult()

    # Safety cap on very large brains.
    if len(eligible) > MAX_NEURONS_TO_LINK:
        eligible = eligible[:MAX_NEURONS_TO_LINK]
        vectors = vectors[:MAX_NEURONS_TO_LINK]
    neurons_embedded = len(vectors)

    # Existing pairs (any synapse type) so we never duplicate a connection.
    # Paged: an unbounded read of this table is the "[Errno 104]" failure mode.
    existing_pairs: set[frozenset[str]] = set()
    synapse_offset = 0
    while True:
        page = await storage.get_synapses(limit=_SYNAPSE_PAGE_SIZE, offset=synapse_offset)
        if not page:
            break
        existing_pairs.update(frozenset({s.source_id, s.target_id}) for s in page)
        synapse_offset += len(page)
        if len(page) < _SYNAPSE_PAGE_SIZE:
            break

    logger.debug(
        "semantic discovery: %d eligible neurons, %d existing pairs",
        len(eligible),
        len(existing_pairs),
    )

    threshold = config.semantic_discovery_similarity_threshold
    max_pairs = min(config.semantic_discovery_max_pairs, MAX_PAIRS_HARD_CAP)
    top_k = SEMANTIC_TOP_K

    new_synapses: list[Synapse] = []
    skipped = 0
    pairs_evaluated = 0

    def _link(i: int, j: int, sim: float) -> bool:
        """Create one SIMILAR_TO synapse if the pair is new. Returns True if added."""
        nonlocal skipped
        pair = frozenset({eligible[i].id, eligible[j].id})
        if pair in existing_pairs:
            skipped += 1
            return False
        new_synapses.append(
            Synapse.create(
                source_id=eligible[i].id,
                target_id=eligible[j].id,
                type=SynapseType.SIMILAR_TO,
                weight=sim * 0.6,  # scale down so semantic links don't dominate
                metadata={"_semantic_discovery": True, "cosine_similarity": round(sim, 4)},
                # SIMILAR_TO is bidirectional, so the id is derived from the
                # sorted pair: (A,B) and (B,A) are one edge, and a writer that
                # slips past the snapshot collides on the primary key instead
                # of laying down a twin row.
                synapse_id=deterministic_edge_id(
                    SynapseType.SIMILAR_TO, eligible[i].id, eligible[j].id
                ),
            )
        )
        existing_pairs.add(pair)
        return True

    try:
        import numpy as np

        mat = np.asarray(vectors, dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        for i in range(len(eligible)):
            sims = mat @ mat[i]
            sims[i] = -1.0
            order = np.argsort(-sims)[:top_k]
            for jj in order:
                j = int(jj)
                sim = float(sims[j])
                pairs_evaluated += 1
                if sim < threshold:
                    break
                _link(i, j, sim)
                if len(new_synapses) >= max_pairs:
                    break
            if len(new_synapses) >= max_pairs:
                break
    except ImportError:
        # Pure-python fallback (slower) — bounded by the caps above.
        for i in range(len(eligible)):
            row = sorted(
                (
                    (j, _cosine_similarity(vectors[i], vectors[j]))
                    for j in range(len(eligible))
                    if j != i
                ),
                key=lambda t: t[1],
                reverse=True,
            )[:top_k]
            for j, sim in row:
                pairs_evaluated += 1
                if sim < threshold:
                    break
                _link(i, j, sim)
                if len(new_synapses) >= max_pairs:
                    break
            if len(new_synapses) >= max_pairs:
                break

    return SemanticDiscoveryResult(
        neurons_embedded=neurons_embedded,
        pairs_evaluated=pairs_evaluated,
        synapses_created=len(new_synapses),
        skipped_existing=skipped,
        synapses=new_synapses,
        truncated=len(new_synapses) >= max_pairs,
    )
