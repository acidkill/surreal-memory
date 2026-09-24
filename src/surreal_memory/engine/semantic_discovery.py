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

import asyncio
import hashlib
import heapq
import json
import logging
import math
import struct
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from surreal_memory.core.constants import GRAPH_ONLY_PLACEHOLDER
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.edge_identity import deterministic_edge_id

if TYPE_CHECKING:
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

SemanticDiscoveryCheckpoint = Callable[[str, str | None, dict[str, Any]], Awaitable[None]]
SemanticDiscoveryBudgetCheck = Callable[[], Awaitable[None]]


class _SHA256Digest(Protocol):
    def update(self, data: bytes) -> None: ...

    def copy(self) -> _SHA256Digest: ...

    def hexdigest(self) -> str: ...


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

# Page size for the neuron scan. Unlike every other scan in the engine, these
# rows MUST carry their embedding vector — the similarity pass is the one place
# that needs it — so a page is ~1024 floats per row and has to stay far smaller
# than the synapse page above.
_EMBEDDING_PAGE_SIZE = 1000

# Similarity rows between event-loop yields in the legacy discovery path.
# Resumable discovery yields every row because each NumPy operation scales with
# the candidate count and must not starve SurrealDB WebSocket keepalives.
_YIELD_EVERY_ROWS = 100

# Keep the durable similarity prefix bounded in both bytes and checkpoint writes.
# The separate per-row budget callback checks the lease/deadline before more work;
# this interval only controls how often the compact result prefix is persisted.
_MAX_SIMILARITY_CHECKPOINTS = 64
_MAX_SIMILARITY_CHECKPOINT_BYTES = 512 * 1024
_MAX_NEURON_CHECKPOINT_BYTES = 512 * 1024
_MAX_REBUILD_STATE_BYTES = 1_000_000

# Ceiling for the pure-python similarity fallback. The vectorised path is O(n*d)
# per row in C; the fallback is O(n*d) per row in interpreted Python, roughly
# three orders of magnitude slower, so at MAX_NEURONS_TO_LINK it does not finish
# in any useful time — and because it never awaits, the per-strategy timeout
# cannot interrupt it either. Bound it and say so, rather than hang.
_FALLBACK_MAX_NEURONS = 500


@dataclass(frozen=True)
class SemanticDiscoveryResult:
    """Result of a semantic discovery run."""

    neurons_embedded: int = 0
    pairs_evaluated: int = 0
    synapses_created: int = 0
    skipped_existing: int = 0
    skipped_created_this_run: int = 0
    """Pairs skipped because THIS pass had just created them.

    Kept apart from ``skipped_existing``: a bidirectional top-K neighbourhood reaches
    each pair twice, so counting the second visit as pre-existing made the report
    overstate how much of the graph was already linked.
    """

    eligible_total: int = 0
    """Candidates before ``MAX_NEURONS_TO_LINK`` resampling — 0 when nothing was cut.

    Each pass deliberately targets the least-connected slice, so consecutive runs
    describe different candidate sets and their counters are not comparable. Saying so
    is the difference between an honest metric and a confusing one.
    """

    synapses: list[Synapse] = field(default_factory=list)
    truncated: bool = False
    """True when the run stopped at ``semantic_discovery_max_pairs``.

    Without this a capped run and a saturated brain print the same number, so a
    constant "2000" reads as a stuck system when it is actually a backlog
    draining one capped run at a time.
    """
    source_fingerprint: str | None = None
    source_token: str | None = None
    reference_time_rebuilt: bool = False


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


async def _select_candidates(
    storage: NeuralStorage,
    eligible: list[Neuron],
    vectors: list[list[float]],
) -> tuple[list[Neuron], list[list[float]]]:
    """Choose which ``MAX_NEURONS_TO_LINK`` neurons this pass considers.

    ``eligible`` always arrives longer than the cap here (the caller only
    calls this when it is). Without this, the cut kept whichever prefix
    ``find_neurons``'s pagination happened to return — the same ~26% of a
    large brain, every single run, because scan order does not change
    between runs. Sorting ascending by synapse degree and keeping the
    least-connected neurons means each pass attacks a different, genuinely
    under-connected part of the graph, so the connectivity gap actually
    closes across repeated runs instead of the same neurons resaturating.

    Capability probe, matching the pattern in ``server/app.py``'s graph
    ranking step: ``get_synapse_degrees`` only exists on backends that can
    answer it cheaply (SurrealDB, via a DB-side GROUP BY). Its absence, or
    any failure while calling it, falls back to the prior first-N behavior —
    this is a prioritisation improvement, not something a degraded backend
    should ever fail a consolidation pass over.
    """
    get_degrees = getattr(storage, "get_synapse_degrees", None)
    degree: dict[str, int] | None = None
    if get_degrees is not None:
        try:
            degree = await get_degrees()
        except Exception:
            logger.debug(
                "get_synapse_degrees failed; falling back to scan-order selection",
                exc_info=True,
            )
            degree = None

    if not degree:
        return eligible[:MAX_NEURONS_TO_LINK], vectors[:MAX_NEURONS_TO_LINK]

    order = sorted(range(len(eligible)), key=lambda idx: degree.get(eligible[idx].id, 0))
    kept = order[:MAX_NEURONS_TO_LINK]
    return [eligible[i] for i in kept], [vectors[i] for i in kept]


async def discover_semantic_synapses(
    storage: NeuralStorage,
    config: BrainConfig,
    *,
    checkpoint: SemanticDiscoveryCheckpoint | None = None,
    resume_state: Mapping[str, Any] | None = None,
    budget_check: SemanticDiscoveryBudgetCheck | None = None,
    run_id: str | None = None,
    owner_token: str | None = None,
    reference_time: datetime | None = None,
    allow_reference_time_rebuild: bool = False,
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

    With durable checkpoints and a supplied ``budget_check``, the callback
    runs before each similarity row. A pause commits the latest completed row
    so a restart can
    skip its scoring work after replaying and validating source data.
    """
    effective_enabled, _, _ = _effective_embedding(config)
    if not effective_enabled:
        logger.debug("Embedding disabled — skipping semantic discovery")
        return SemanticDiscoveryResult()

    if checkpoint is not None:
        return await _discover_semantic_synapses_resumable(
            storage,
            config,
            checkpoint=checkpoint,
            resume_state=resume_state,
            budget_check=budget_check,
            run_id=run_id,
            owner_token=owner_token,
            reference_time=reference_time,
            allow_reference_time_rebuild=allow_reference_time_rebuild,
        )
    if reference_time is not None:
        raise RuntimeError("frozen semantic discovery requires durable keyset checkpoints")

    # Collect eligible neurons that already carry a stored embedding (no re-embed).
    # Ask for the two eligible types separately so the filter runs in the DB's
    # composite (brain_id, type) index instead of dragging every neuron in the
    # brain across the wire and discarding most of them here. On a large brain
    # the type-agnostic scan fetched the whole table — vectors included — to keep
    # the CONCEPT/ENTITY minority.
    per_type: list[tuple[list[Neuron], list[list[float]]]] = []
    for neuron_type in (NeuronType.CONCEPT, NeuronType.ENTITY):
        type_neurons: list[Neuron] = []
        type_vectors: list[list[float]] = []
        offset = 0
        while True:
            batch = await storage.find_neurons(
                type=neuron_type, limit=_EMBEDDING_PAGE_SIZE, offset=offset
            )
            if not batch:
                break
            for n in batch:
                if not n.content.strip():
                    continue
                # GRAPH_ONLY tombstones all share the literal placeholder as
                # content, so their stored vectors are identical brain-wide -
                # cosine 1.0 between memories that have nothing in common.
                # Linking them would persist a false SIMILAR_TO edge on every
                # consolidation pass; the dedup census skips these anchors for
                # the same reason (their content_hash carries the 0 sentinel).
                if n.content == GRAPH_ONLY_PLACEHOLDER:
                    continue
                emb = n.metadata.get("_embedding")
                if emb:
                    type_neurons.append(n)
                    type_vectors.append([float(x) for x in emb])
            offset += len(batch)
            if len(batch) < _EMBEDDING_PAGE_SIZE:
                break
        per_type.append((type_neurons, type_vectors))

    # Interleave the two types instead of concatenating them. Two things below
    # cut on list order, and both would otherwise systematically starve whichever
    # type came second: the `eligible[:MAX_NEURONS_TO_LINK]` slice that
    # `_select_candidates` falls back to whenever it has no degree data (a brain
    # with no synapses yet, or any backend that cannot answer
    # `get_synapse_degrees`), and the stable sort it uses otherwise, where
    # never-linked neurons of both types tie at degree 0. The old type-agnostic
    # scan mixed them by construction; per-type paging has to mix them on purpose.
    eligible: list[Neuron] = []
    vectors: list[list[float]] = []
    longest = max((len(type_neurons) for type_neurons, _ in per_type), default=0)
    for index in range(longest):
        for type_neurons, type_vectors in per_type:
            if index < len(type_neurons):
                eligible.append(type_neurons[index])
                vectors.append(type_vectors[index])

    if len(eligible) < 2:
        return SemanticDiscoveryResult()

    # Safety cap on very large brains. Below the cap, ordering doesn't matter
    # (everything is considered); above it, WHICH neurons get dropped decides
    # whether repeated runs make progress or just resaturate the same slice.
    eligible_before_resample = len(eligible)
    if len(eligible) > MAX_NEURONS_TO_LINK:
        eligible, vectors = await _select_candidates(storage, eligible, vectors)
    else:
        # Nothing was cut, so there is no candidate-set change to report.
        eligible_before_resample = 0
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

    logger.debug("semantic discovery: %d eligible neurons", len(eligible))

    threshold = config.semantic_discovery_similarity_threshold
    max_pairs = min(config.semantic_discovery_max_pairs, MAX_PAIRS_HARD_CAP)
    top_k = SEMANTIC_TOP_K

    new_synapses: list[Synapse] = []
    skipped = 0
    skipped_created_this_run = 0
    pairs_evaluated = 0
    # Pairs this pass itself created. With a bidirectional top-K neighbourhood the same
    # pair is reached twice — once from each end — so without this set the second visit
    # was counted as "already existed", inflating a number the report labels as
    # pre-existing links.
    created_this_run: set[frozenset[str]] = set()

    def _link(i: int, j: int, sim: float) -> bool:
        """Create one SIMILAR_TO synapse if the pair is new. Returns True if added."""
        nonlocal skipped, skipped_created_this_run
        pair = frozenset({eligible[i].id, eligible[j].id})
        if pair in created_this_run:
            skipped_created_this_run += 1
            return False
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
        created_this_run.add(pair)
        return True

    try:
        import numpy as np

        mat = np.asarray(vectors, dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        for i in range(len(eligible)):
            if i and i % _YIELD_EVERY_ROWS == 0:
                # Hand the loop back. Nothing in this pass awaits on its own, so
                # without this the connection's keepalive never gets to run.
                await asyncio.sleep(0)
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
        # Pure-python fallback. It is ~3 orders of magnitude slower than the
        # vectorised path, so MAX_NEURONS_TO_LINK is far too generous for it:
        # at that size the pass never finishes, and since it never awaits, the
        # per-strategy timeout cannot cut it short either. Do a bounded slice
        # and say plainly that the full pass needs numpy.
        considered = min(len(eligible), _FALLBACK_MAX_NEURONS)
        if len(eligible) > considered:
            logger.warning(
                "numpy is not installed: semantic discovery is running its pure-python "
                "fallback and will consider only %d of %d eligible neurons this pass. "
                "Install numpy to link the whole set.",
                considered,
                len(eligible),
            )
        for i in range(considered):
            if i and i % _YIELD_EVERY_ROWS == 0:
                await asyncio.sleep(0)
            row = sorted(
                (
                    (j, _cosine_similarity(vectors[i], vectors[j]))
                    for j in range(considered)
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
        skipped_created_this_run=skipped_created_this_run,
        eligible_total=eligible_before_resample,
        synapses=new_synapses,
        truncated=len(new_synapses) >= max_pairs,
    )


class _SerializableSHA256:
    """Small SHA-256 implementation whose state can survive a checkpoint."""

    _INITIAL = (
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    )
    _K = (
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    )

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        if state is None:
            self._words = list(self._INITIAL)
            self._count = 0
            self._buffer = b""
            return
        words = state.get("h")
        count = state.get("n")
        tail = state.get("b")
        if (
            not isinstance(words, list)
            or len(words) != 8
            or any(type(word) is not int or word < 0 or word > 0xFFFFFFFF for word in words)
            or type(count) is not int
            or count < 0
            or not isinstance(tail, str)
        ):
            raise RuntimeError("invalid semantic_link serialized SHA-256 state")
        try:
            buffer = bytes.fromhex(tail)
        except ValueError as exc:
            raise RuntimeError("invalid semantic_link serialized SHA-256 buffer") from exc
        if len(buffer) >= 64 or count % 64 != len(buffer):
            raise RuntimeError("invalid semantic_link serialized SHA-256 length")
        self._words = list(words)
        self._count = count
        self._buffer = buffer

    @staticmethod
    def _ror(value: int, bits: int) -> int:
        return ((value >> bits) | (value << (32 - bits))) & 0xFFFFFFFF

    def _compress(self, block: bytes) -> None:
        schedule = list(struct.unpack(">16I", block))
        for index in range(16, 64):
            x = schedule[index - 15]
            y = schedule[index - 2]
            sigma0 = self._ror(x, 7) ^ self._ror(x, 18) ^ (x >> 3)
            sigma1 = self._ror(y, 17) ^ self._ror(y, 19) ^ (y >> 10)
            schedule.append(
                (schedule[index - 16] + sigma0 + schedule[index - 7] + sigma1) & 0xFFFFFFFF
            )

        a, b, c, d, e, f, g, h = self._words
        for index in range(64):
            sum1 = self._ror(e, 6) ^ self._ror(e, 11) ^ self._ror(e, 25)
            choose = (e & f) ^ (~e & g)
            t1 = (h + sum1 + choose + self._K[index] + schedule[index]) & 0xFFFFFFFF
            sum0 = self._ror(a, 2) ^ self._ror(a, 13) ^ self._ror(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            t2 = (sum0 + majority) & 0xFFFFFFFF
            a, b, c, d, e, f, g, h = (
                (t1 + t2) & 0xFFFFFFFF,
                a,
                b,
                c,
                (d + t1) & 0xFFFFFFFF,
                e,
                f,
                g,
            )
        self._words = [
            (left + right) & 0xFFFFFFFF
            for left, right in zip(self._words, (a, b, c, d, e, f, g, h), strict=True)
        ]

    def update(self, data: bytes) -> None:
        if not data:
            return
        self._count += len(data)
        data = self._buffer + data
        block_end = len(data) - (len(data) % 64)
        for offset in range(0, block_end, 64):
            self._compress(data[offset : offset + 64])
        self._buffer = data[block_end:]

    def copy(self) -> _SerializableSHA256:
        clone = object.__new__(_SerializableSHA256)
        clone._words = self._words.copy()
        clone._count = self._count
        clone._buffer = self._buffer
        return clone

    def export(self) -> dict[str, Any]:
        return {"h": self._words.copy(), "n": self._count, "b": self._buffer.hex()}

    def hexdigest(self) -> str:
        clone = self.copy()
        bit_count = clone._count * 8
        padding = b"\x80" + b"\x00" * ((55 - clone._count) % 64) + bit_count.to_bytes(8, "big")
        clone.update(padding)
        return "".join(f"{word:08x}" for word in clone._words)


class _SemanticFingerprintBuilder:
    """Stream the same canonical source payload used by consolidation snapshots."""

    def __init__(
        self,
        brain_id: str | None,
        config: BrainConfig,
        *,
        serialized_state: Mapping[str, Any] | None = None,
        serializable: bool = False,
    ) -> None:
        self._digest: _SHA256Digest
        if serialized_state is not None:
            self._digest = _SerializableSHA256(serialized_state)
        elif serializable:
            self._digest = _SerializableSHA256()
        else:
            self._digest = hashlib.sha256()
        if serialized_state is None:
            self._digest.update(b'{"brain_id":')
            self._write(brain_id)
            self._digest.update(b',"config":')
            self._write(repr(config))
            self._digest.update(b',"neurons":[')
        self._neuron_count = 0
        self._edge_count = 0
        self._edges_started = False
        self._finished = False

    def _write(self, value: Any) -> None:
        self._digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        )

    def add_neuron(self, neuron: Neuron) -> None:
        if neuron.type not in (NeuronType.CONCEPT, NeuronType.ENTITY):
            return
        if self._neuron_count:
            self._digest.update(b",")
        self._write(
            {
                "id": neuron.id,
                "type": neuron.type.value,
                "content": neuron.content,
                "embedding": neuron.metadata.get("_embedding"),
            }
        )
        self._neuron_count += 1

    def begin_edges(self) -> None:
        if not self._edges_started:
            self._digest.update(b'],"synapses":[')
            self._edges_started = True

    def add_synapse(self, synapse: Synapse, excluded_ids: set[str] | None = None) -> None:
        self.begin_edges()
        if excluded_ids and synapse.id in excluded_ids:
            return
        if self._edge_count:
            self._digest.update(b",")
        self._write(
            {
                "id": synapse.id,
                "source_id": synapse.source_id,
                "target_id": synapse.target_id,
                "type": synapse.type.value,
            }
        )
        self._edge_count += 1

    def prefix_digest(self) -> str:
        return self._digest.copy().hexdigest()

    def export_state(self) -> dict[str, Any]:
        if not isinstance(self._digest, _SerializableSHA256):
            raise RuntimeError("semantic_link fingerprint is not serializable")
        return {
            "sha256": self._digest.export(),
            "neuron_count": self._neuron_count,
            "edge_count": self._edge_count,
            "edges_started": self._edges_started,
            "finished": self._finished,
        }

    def restore_counts(self, state: Mapping[str, Any]) -> None:
        neuron_count = state.get("neuron_count")
        edge_count = state.get("edge_count")
        edges_started = state.get("edges_started")
        finished = state.get("finished")
        if (
            type(neuron_count) is not int
            or neuron_count < 0
            or type(edge_count) is not int
            or edge_count < 0
            or type(edges_started) is not bool
            or type(finished) is not bool
        ):
            raise RuntimeError("invalid semantic_link fingerprint counters")
        self._neuron_count = neuron_count
        self._edge_count = edge_count
        self._edges_started = edges_started
        self._finished = finished

    def finish(self) -> str:
        if not self._finished:
            self.begin_edges()
            self._digest.update(b"]}")
            self._finished = True
        return self._digest.hexdigest()


async def _capture_source_token(storage: NeuralStorage) -> Any:
    capture = getattr(storage, "capture_semantic_source_token", None)
    if not callable(capture):
        raise RuntimeError("storage does not support semantic source mutation fencing")
    token = await capture()
    if not isinstance(token, str) or not token:
        raise RuntimeError("storage returned an invalid semantic source mutation token")
    return token


async def _assert_source_unchanged(
    storage: NeuralStorage, token: Any, *, created_before: datetime | None = None
) -> None:
    verify = getattr(storage, "assert_semantic_source_unchanged", None)
    if not callable(verify):
        raise RuntimeError("storage does not support semantic source mutation fencing")
    if not isinstance(token, str) or not token:
        raise RuntimeError("semantic source rebuild checkpoint has no valid mutation token")
    if created_before is None:
        await verify(token)
    else:
        await verify(token, created_before=created_before)


async def _load_source_state(
    storage: NeuralStorage,
    state_ref: Any,
    *,
    brain_id: str | None,
    run_id: str | None,
) -> tuple[dict[str, Any], str, int]:
    if not isinstance(state_ref, Mapping):
        raise RuntimeError("invalid semantic_link staged source-state reference")
    state_id = state_ref.get("state_id")
    revision = state_ref.get("revision")
    if not isinstance(state_id, str) or not state_id or type(revision) is not int or revision < 1:
        raise RuntimeError("invalid semantic_link staged source-state reference")
    load = getattr(storage, "load_semantic_discovery_state", None)
    if not callable(load):
        raise RuntimeError("storage cannot load semantic_link staged source state")
    payload = await load(state_id, revision)
    if not isinstance(payload, Mapping):
        raise RuntimeError("semantic_link staged source state is unavailable")
    if payload.get("brain_id") != brain_id or payload.get("state_id") != state_id:
        raise RuntimeError("semantic_link staged source state belongs to another run")
    if payload.get("revision") != revision:
        raise RuntimeError("semantic_link staged source-state revision mismatch")
    if run_id is not None and payload.get("run_id") != run_id:
        raise RuntimeError("semantic_link staged source state belongs to another run")
    return dict(payload), state_id, revision


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


async def _discover_semantic_synapses_resumable(
    storage: NeuralStorage,
    config: BrainConfig,
    *,
    checkpoint: SemanticDiscoveryCheckpoint,
    resume_state: Mapping[str, Any] | None,
    budget_check: SemanticDiscoveryBudgetCheck | None,
    run_id: str | None = None,
    owner_token: str | None = None,
    reference_time: datetime | None = None,
    allow_reference_time_rebuild: bool = False,
) -> SemanticDiscoveryResult:
    """Discover links in durable pages; replay only the committed prefix on resume."""
    for name, value in (("run_id", run_id), ("owner_token", owner_token)):
        if value is not None and (not isinstance(value, str) or not value):
            raise RuntimeError(f"invalid semantic_link {name}")
    neuron_fetch = getattr(storage, "find_neurons_after_id", None)
    synapse_fetch = getattr(storage, "get_synapses_after_id", None)
    pair_lookup = getattr(storage, "find_existing_synapse_pairs", None)
    if not callable(neuron_fetch) or not callable(synapse_fetch) or not callable(pair_lookup):
        raise RuntimeError(
            "semantic_link discovery requires keyset scans and indexed pair lookup support"
        )

    async def _get_degrees() -> dict[str, int]:
        get_degrees = getattr(storage, "get_synapse_degrees", None)
        if not callable(get_degrees):
            raise RuntimeError("storage cannot restore semantic_link degree ranking")
        if reference_time is None:
            return cast("dict[str, int]", await get_degrees())
        return cast("dict[str, int]", await get_degrees(created_before=reference_time))

    manifest = dict(resume_state or {})
    manifest_stage = str(manifest.get("stage") or "")
    manifest_version = manifest.get("version")
    if manifest and (
        manifest.get("kind") != "semantic_link_discovery"
        or manifest_version not in (1, 2, 3)
        or (manifest_version == 2 and manifest_stage != "similarity")
        or (manifest_version == 3 and manifest_stage not in {"neurons", "synapses", "similarity"})
    ):
        raise RuntimeError("invalid semantic_link discovery checkpoint")
    brain_id = getattr(storage, "current_brain_id", None)
    durable_neuron_state: dict[str, Any] | None = None
    state_id: str | None = None
    state_revision = 0
    state_ref = manifest.get("source_state_ref")
    if state_ref is not None:
        durable_neuron_state, state_id, state_revision = await _load_source_state(
            storage, state_ref, brain_id=brain_id, run_id=run_id
        )
        legacy_checkpoint = durable_neuron_state.get("legacy_checkpoint")
        if legacy_checkpoint is not None and not isinstance(legacy_checkpoint, Mapping):
            raise RuntimeError("invalid semantic_link legacy checkpoint in staged state")
        saved = dict(legacy_checkpoint or {})
    elif manifest_version == 3:
        inline_state = manifest.get("source_rebuild")
        if not isinstance(inline_state, Mapping):
            raise RuntimeError("invalid semantic_link source rebuild state")
        durable_neuron_state = dict(inline_state)
        state_id = durable_neuron_state.get("state_id")
        state_revision = durable_neuron_state.get("state_revision", 0)
        legacy_checkpoint = durable_neuron_state.get("legacy_checkpoint")
        if not isinstance(legacy_checkpoint, Mapping):
            raise RuntimeError("invalid semantic_link legacy checkpoint")
        saved = dict(legacy_checkpoint)
    else:
        saved = manifest
    saved_stage = str(saved.get("stage") or "")
    saved_version = saved.get("version")
    saved_cursor = saved.get("cursor")
    if saved and (
        saved.get("kind") != "semantic_link_discovery"
        or saved_version not in (1, 2)
        or (saved_version == 2 and saved_stage != "similarity")
    ):
        raise RuntimeError("invalid semantic_link legacy checkpoint")
    if saved_stage and saved_stage not in {"neurons", "selection", "synapses", "similarity"}:
        raise RuntimeError("invalid semantic_link legacy checkpoint stage")
    token = saved.get("source_token")
    if isinstance(durable_neuron_state, Mapping):
        token = durable_neuron_state.get("source_token")
    reference_time_rebuilt = False
    if reference_time is not None:
        expected_reference_time = reference_time.isoformat(timespec="microseconds")
        staged_reference_time = (
            durable_neuron_state.get("reference_time")
            if isinstance(durable_neuron_state, Mapping)
            else None
        )
        has_staged_checkpoint = bool(manifest or saved or durable_neuron_state)
        if has_staged_checkpoint and staged_reference_time is None:
            if not allow_reference_time_rebuild:
                raise RuntimeError(
                    "legacy semantic_link checkpoint has no frozen reference-time marker"
                )
            if not isinstance(token, str) or not token:
                raise RuntimeError(
                    "legacy semantic_link checkpoint has no source token to validate"
                )
            # Prove that only post-reference records changed before replacing the
            # staged source. Pre-reference updates/deletes still invalidate it.
            await _assert_source_unchanged(storage, token, created_before=reference_time)
            token = await _capture_source_token(storage)
            reference_time_rebuilt = True
            durable_neuron_state = None
            state_id = None
            state_revision = 0
            saved = {}
            saved_stage = ""
            saved_cursor = None
        elif has_staged_checkpoint and staged_reference_time != expected_reference_time:
            raise RuntimeError("semantic_link checkpoint reference_time changed")
    capture_fence = getattr(storage, "capture_semantic_source_token", None)
    verify_fence = getattr(storage, "assert_semantic_source_unchanged", None)
    fence_available = callable(capture_fence) and callable(verify_fence)
    if token is not None:
        await _assert_source_unchanged(storage, token, created_before=reference_time)
    elif fence_available:
        token = await _capture_source_token(storage)
    else:
        raise RuntimeError("storage does not support semantic source mutation fencing")
    fingerprint_state = (
        durable_neuron_state.get("fingerprint")
        if isinstance(durable_neuron_state, Mapping)
        else None
    )
    if fingerprint_state is not None and not isinstance(fingerprint_state, Mapping):
        raise RuntimeError("invalid semantic_link source fingerprint state")
    migrating_legacy_source = True
    fingerprint = _SemanticFingerprintBuilder(
        brain_id,
        config,
        serialized_state=(
            fingerprint_state.get("sha256") if isinstance(fingerprint_state, Mapping) else None
        ),
        serializable=migrating_legacy_source,
    )
    if isinstance(fingerprint_state, Mapping):
        fingerprint.restore_counts(fingerprint_state)
    eligible: list[Neuron] = []
    vectors: list[list[float]] = []
    candidate_priorities: list[int] = []
    candidate_heap: list[tuple[int, int, str]] = []
    candidate_map: dict[str, tuple[int, Neuron, list[float]]] = {}
    degree_by_id: dict[str, int] | None = None
    degree_probe_complete = False
    eligible_total = 0
    type_ranks = {NeuronType.CONCEPT: 0, NeuronType.ENTITY: 0}

    if isinstance(durable_neuron_state, Mapping):
        raw_total = durable_neuron_state.get("eligible_total")
        raw_ranks = durable_neuron_state.get("type_ranks")
        raw_candidates = durable_neuron_state.get("candidates")
        if (
            type(raw_total) is not int
            or raw_total < 0
            or not isinstance(raw_ranks, Mapping)
            or not isinstance(raw_candidates, list)
            or len(raw_candidates) > MAX_NEURONS_TO_LINK
        ):
            raise RuntimeError("invalid semantic_link candidate rebuild state")
        eligible_total = raw_total
        try:
            type_ranks = {
                NeuronType.CONCEPT: int(raw_ranks[NeuronType.CONCEPT.value]),
                NeuronType.ENTITY: int(raw_ranks[NeuronType.ENTITY.value]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid semantic_link candidate type ranks") from exc
        if any(value < 0 for value in type_ranks.values()):
            raise RuntimeError("invalid semantic_link candidate type ranks")
        restore_candidate_ids: list[str] = []
        candidate_state: list[tuple[str, int, int | None]] = []
        for item in raw_candidates:
            if (
                not isinstance(item, list)
                or len(item) != 3
                or not isinstance(item[0], str)
                or type(item[1]) is not int
                or item[1] < 0
                or (item[2] is not None and (type(item[2]) is not int or item[2] < 0))
            ):
                raise RuntimeError("invalid semantic_link candidate entry")
            restore_candidate_ids.append(item[0])
            candidate_state.append((item[0], item[1], item[2]))
        get_batch = getattr(storage, "get_neurons_batch", None)
        if callable(get_batch):
            restored = await get_batch(restore_candidate_ids)
        else:
            get_one = getattr(storage, "get_neuron", None)
            if not callable(get_one):
                raise RuntimeError("storage cannot restore semantic_link candidate neurons")
            restored = {}
            for candidate_id in restore_candidate_ids:
                neuron = await get_one(candidate_id)
                if neuron is not None:
                    restored[candidate_id] = neuron
        if set(restored) != set(restore_candidate_ids):
            raise RuntimeError("semantic_link candidate source changed during rebuild")
        bounded = bool(durable_neuron_state.get("bounded"))
        degree_mode = bool(durable_neuron_state.get("degree_mode"))
        if bounded and degree_mode:
            degree_by_id = await _get_degrees()
        if bounded:
            degree_probe_complete = True
            for candidate_id, priority, stored_degree in candidate_state:
                neuron = restored[candidate_id]
                embedding = neuron.metadata.get("_embedding")
                if not embedding:
                    raise RuntimeError("semantic_link candidate embedding disappeared")
                vector = (
                    embedding
                    if isinstance(embedding, list)
                    and all(type(value) is float for value in embedding)
                    else [float(value) for value in embedding]
                )
                candidate_map[candidate_id] = (priority, neuron, vector)
                if degree_mode:
                    degree = degree_by_id.get(candidate_id, 0) if degree_by_id else 0
                    if stored_degree != degree:
                        raise RuntimeError("semantic_link candidate degree changed during rebuild")
                    candidate_heap.append((-degree, -priority, candidate_id))
                else:
                    candidate_heap.append((0, -priority, candidate_id))
            heapq.heapify(candidate_heap)
        else:
            for candidate_id, priority, _stored_degree in candidate_state:
                neuron = restored[candidate_id]
                embedding = neuron.metadata.get("_embedding")
                if not embedding:
                    raise RuntimeError("semantic_link candidate embedding disappeared")
                eligible.append(neuron)
                vectors.append(
                    embedding
                    if isinstance(embedding, list)
                    and all(type(value) is float for value in embedding)
                    else [float(value) for value in embedding]
                )
                candidate_priorities.append(priority)

    if state_id is None:
        state_id = str(uuid.uuid4())
    source_rebuild: dict[str, Any]
    if isinstance(durable_neuron_state, Mapping):
        source_rebuild = dict(durable_neuron_state)
        source_rebuild.setdefault("state_id", state_id)
        source_rebuild.setdefault("state_revision", state_revision)
    else:
        source_rebuild = {
            "state_id": state_id,
            "state_revision": state_revision,
            "brain_id": brain_id,
            "source_token": token,
            "legacy_checkpoint": saved or None,
            "legacy_neuron_cursor": saved_cursor if saved_stage == "neurons" else None,
            "legacy_neuron_digest": saved.get("prefix_digest")
            if saved_stage == "neurons"
            else None,
            "legacy_neuron_verified": saved_stage != "neurons" or saved_cursor is None,
            "legacy_edge_cursor": saved_cursor if saved_stage == "synapses" else None,
            "legacy_edge_digest": saved.get("prefix_digest") if saved_stage == "synapses" else None,
            "legacy_edge_verified": saved_stage != "synapses" or saved_cursor is None,
            "neuron_cursor": None,
            "edge_cursor": None,
            "neuron_complete": False,
            "edge_complete": False,
            "source_complete": False,
            "similarity_rebuild": None,
        }

    saved_run_id = source_rebuild.get("run_id")
    if run_id is not None:
        if saved_run_id not in (None, run_id):
            raise RuntimeError("semantic_link staged source state belongs to another run")
        source_rebuild["run_id"] = run_id
    saved_owner_token = source_rebuild.get("owner_token")
    if saved_owner_token is not None and (
        not isinstance(saved_owner_token, str) or not saved_owner_token
    ):
        raise RuntimeError("invalid semantic_link staged source owner token")
    if saved_owner_token is None and owner_token is not None:
        source_rebuild["owner_token"] = owner_token

    def source_rebuild_snapshot() -> dict[str, Any]:
        if source_rebuild is None:
            raise RuntimeError("semantic_link source rebuild state is unavailable")
        if candidate_map:
            candidates = [
                [candidate_id, priority, degree_by_id.get(candidate_id, 0) if degree_by_id else 0]
                for candidate_id, (priority, _neuron, _vector) in sorted(candidate_map.items())
            ]
            bounded = True
        else:
            candidates = [
                [neuron.id, priority, None]
                for neuron, priority in zip(eligible, candidate_priorities, strict=True)
            ]
            bounded = False
        return {
            **source_rebuild,
            "reference_time": (
                reference_time.isoformat(timespec="microseconds")
                if reference_time is not None
                else None
            ),
            "fingerprint": fingerprint.export_state(),
            "eligible_total": eligible_total,
            "type_ranks": {kind.value: rank for kind, rank in type_ranks.items()},
            "candidates": candidates,
            "bounded": bounded,
            "degree_mode": bool(degree_by_id),
        }

    async def write_source_checkpoint(
        stage: str, cursor: str | None, *, preserve_legacy_cursor: bool = False
    ) -> None:
        nonlocal state_revision
        save = getattr(storage, "save_semantic_discovery_state", None)
        if not callable(save):
            raise RuntimeError("storage cannot save semantic_link staged source state")
        next_revision = state_revision + 1
        source_snapshot = source_rebuild_snapshot()
        source_snapshot["state_id"] = state_id
        source_snapshot["state_revision"] = next_revision
        source_snapshot["revision"] = next_revision
        encoded_source = json.dumps(
            source_snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded_source) > _MAX_REBUILD_STATE_BYTES:
            raise RuntimeError(
                f"semantic_link staged source state exceeds {_MAX_REBUILD_STATE_BYTES} bytes"
            )
        await save(state_id, next_revision, source_snapshot)
        state_revision = next_revision
        source_rebuild["state_revision"] = next_revision
        checkpoint_stage = saved_stage if preserve_legacy_cursor and saved_stage else stage
        checkpoint_cursor = saved_cursor if preserve_legacy_cursor and saved_stage else cursor
        details = {
            "kind": "semantic_link_discovery",
            "version": 3,
            "stage": checkpoint_stage or stage,
            "cursor": checkpoint_cursor,
            "source_state_ref": {
                "state_id": state_id,
                "revision": next_revision,
                "state_revision": next_revision,
                "source_token": token,
            },
        }
        encoded_manifest = json.dumps(
            [json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"))],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded_manifest) > _MAX_NEURON_CHECKPOINT_BYTES:
            raise RuntimeError("semantic_link staged source manifest exceeds 512 KiB")
        await checkpoint(checkpoint_stage or stage, checkpoint_cursor, details)

    async def record_neuron(neuron: Neuron) -> None:
        nonlocal eligible_total, degree_by_id, degree_probe_complete, candidate_heap
        fingerprint.add_neuron(neuron)
        if neuron.type not in type_ranks or not neuron.content.strip():
            return
        # GRAPH_ONLY placeholders all share an artificial vector and must not
        # produce semantic edges.
        if neuron.content == GRAPH_ONLY_PLACEHOLDER:
            return
        embedding = neuron.metadata.get("_embedding")
        if not embedding:
            return

        # This is the exact position the old per-type lists occupied after
        # interleaving: CONCEPT at 2*r, ENTITY at 2*r+1. The key preserves
        # stable cross-type fairness when candidate degrees tie.
        type_rank = type_ranks[neuron.type]
        priority = type_rank * 2 + (0 if neuron.type == NeuronType.CONCEPT else 1)
        type_ranks[neuron.type] += 1
        eligible_total += 1

        if eligible_total <= MAX_NEURONS_TO_LINK:
            eligible.append(neuron)
            vectors.append(
                embedding
                if isinstance(embedding, list) and all(type(value) is float for value in embedding)
                else [float(value) for value in embedding]
            )
            candidate_priorities.append(priority)
            return

        # Do not retain an unbounded neuron/vector population. Probe only once
        # the existing safety cap is crossed, matching the legacy behavior on
        # small graphs. Backends without degree support retain the original
        # interleaved prefix; capable backends retain the least-connected cap.
        if not degree_probe_complete:
            degree_probe_complete = True
            get_degrees = getattr(storage, "get_synapse_degrees", None)
            if callable(get_degrees):
                try:
                    degree_by_id = await _get_degrees()
                except Exception:
                    logger.debug(
                        "get_synapse_degrees failed; falling back to scan-order selection",
                        exc_info=True,
                    )
                    degree_by_id = None

        if not candidate_heap:
            candidate_map.update(
                {
                    candidate.id: (candidate_priority, candidate, vector)
                    for candidate, vector, candidate_priority in zip(
                        eligible, vectors, candidate_priorities, strict=True
                    )
                }
            )
            if degree_by_id:
                candidate_heap = [
                    (-degree_by_id.get(candidate.id, 0), -candidate_priority, candidate.id)
                    for candidate, candidate_priority in zip(
                        eligible, candidate_priorities, strict=True
                    )
                ]
            else:
                candidate_heap = [
                    (0, -candidate_priority, candidate.id)
                    for candidate, candidate_priority in zip(
                        eligible, candidate_priorities, strict=True
                    )
                ]
            heapq.heapify(candidate_heap)
            # The map now owns the bounded candidates. Drop the parallel list
            # containers so replacement does not leave an extra retained prefix.
            eligible.clear()
            vectors.clear()
            candidate_priorities.clear()

        entry = (
            (-degree_by_id.get(neuron.id, 0), -priority, neuron.id)
            if degree_by_id
            else (0, -priority, neuron.id)
        )
        if entry <= candidate_heap[0]:
            return
        evicted = heapq.heapreplace(candidate_heap, entry)
        candidate_map.pop(evicted[2])
        candidate_map[neuron.id] = (
            priority,
            neuron,
            (
                embedding
                if isinstance(embedding, list) and all(type(value) is float for value in embedding)
                else [float(value) for value in embedding]
            ),
        )

    async def emit(
        stage: str,
        cursor: str | None,
        *,
        prefix_digest: str,
        **details: Any,
    ) -> None:
        source_rebuild["last_scan_stage"] = stage
        source_rebuild["last_scan_cursor"] = cursor
        source_rebuild["last_prefix_digest"] = prefix_digest
        source_rebuild["last_scan_details"] = details
        await write_source_checkpoint(
            stage,
            cursor,
            preserve_legacy_cursor=saved_stage == "similarity",
        )

    neuron_cursor = source_rebuild.get("neuron_cursor")
    if neuron_cursor is not None and not isinstance(neuron_cursor, str):
        raise RuntimeError("invalid semantic_link staged neuron cursor")
    legacy_neuron_cursor = source_rebuild.get(
        "legacy_neuron_cursor", source_rebuild.get("legacy_cursor")
    )
    legacy_neuron_digest = source_rebuild.get(
        "legacy_neuron_digest", source_rebuild.get("legacy_prefix_digest")
    )
    legacy_prefix_verified = bool(
        source_rebuild.get(
            "legacy_neuron_verified", source_rebuild.get("legacy_prefix_verified", False)
        )
    )
    source_scan_complete_on_entry = bool(source_rebuild.get("source_complete"))
    replay_neuron_cursor = (
        str(legacy_neuron_cursor)
        if source_rebuild is not None and legacy_neuron_cursor is not None
        else None
    )
    replay_neuron = replay_neuron_cursor is not None and not legacy_prefix_verified
    replay_neuron_found = not replay_neuron
    last_neuron_id: str | None = neuron_cursor
    while not bool(source_rebuild.get("neuron_complete")):
        if not replay_neuron:
            await emit(
                "neurons",
                neuron_cursor,
                prefix_digest=fingerprint.prefix_digest(),
                eligible_neurons=eligible_total,
            )
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
        if budget_check is not None and source_rebuild is not None:
            await budget_check()
        neuron_page_options: dict[str, Any] = {
            "limit": _EMBEDDING_PAGE_SIZE,
            "ephemeral": None,
            "include_embedding": True,
        }
        if reference_time is not None:
            neuron_page_options["created_before"] = reference_time
        page = await neuron_fetch(neuron_cursor, **neuron_page_options)
        if not page:
            if replay_neuron and not replay_neuron_found:
                raise RuntimeError("semantic_link neuron cursor disappeared during resume")
            source_rebuild["neuron_complete"] = True
            break
        replayed_this_page = False
        for neuron in page:
            neuron_id = str(neuron.id)
            if (
                replay_neuron
                and replay_neuron_cursor is not None
                and neuron_id > replay_neuron_cursor
            ):
                raise RuntimeError("semantic_link neuron cursor changed during resume")
            await record_neuron(neuron)
            neuron_cursor = neuron_id
            source_rebuild["neuron_cursor"] = neuron_cursor
            last_neuron_id = neuron_id
            if replay_neuron and neuron_id == replay_neuron_cursor:
                replay_neuron_found = True
                if fingerprint.prefix_digest() != legacy_neuron_digest:
                    raise RuntimeError("semantic_link neuron source changed before its cursor")
                replay_neuron = False
                legacy_prefix_verified = True
                source_rebuild["legacy_neuron_verified"] = True
                replayed_this_page = True
                break
        if not replay_neuron and not replay_neuron_found:
            raise RuntimeError("semantic_link neuron cursor disappeared during resume")
        if replay_neuron:
            if token is not None:
                await _assert_source_unchanged(storage, token, created_before=reference_time)
                await emit(
                    "neurons",
                    neuron_cursor,
                    prefix_digest=fingerprint.prefix_digest(),
                    eligible_neurons=eligible_total,
                )
            continue
        if not replay_neuron_found:
            continue
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
        await emit(
            "neurons",
            neuron_cursor,
            prefix_digest=fingerprint.prefix_digest(),
            eligible_neurons=eligible_total,
        )
        if len(page) < _EMBEDDING_PAGE_SIZE and not replayed_this_page:
            source_rebuild["neuron_complete"] = True
            break

    neuron_digest = (
        str(source_rebuild["neuron_digest"])
        if source_rebuild.get("neuron_digest") is not None
        else fingerprint.prefix_digest()
    )
    source_rebuild["neuron_digest"] = neuron_digest
    if (
        saved_stage in {"selection", "synapses", "similarity"}
        and saved.get("neuron_digest") != neuron_digest
    ):
        raise RuntimeError("semantic_link neuron source changed after its checkpoint")

    # Restore the original per-type interleave when the complete set fits the
    # cap. Above the cap, match _select_candidates: degree order with stable
    # interleaved ties, or the first interleaved cap when degree lookup fails.
    if candidate_map:
        if degree_by_id:
            selected = sorted(
                candidate_map.values(),
                key=lambda item: (degree_by_id.get(item[1].id, 0), item[0]),
            )
        else:
            selected = sorted(candidate_map.values(), key=lambda item: item[0])
        eligible = [item[1] for item in selected]
        vectors = [item[2] for item in selected]
    elif eligible:
        interleaved = sorted(
            zip(candidate_priorities, eligible, vectors, strict=True),
            key=lambda item: item[0],
        )
        eligible = [item[1] for item in interleaved]
        vectors = [item[2] for item in interleaved]

    eligible_before_resample = eligible_total if eligible_total > MAX_NEURONS_TO_LINK else 0
    if eligible_before_resample and saved_stage not in {"synapses", "similarity"}:
        await emit(
            "selection",
            last_neuron_id,
            prefix_digest=neuron_digest,
            neuron_digest=neuron_digest,
            eligible_neurons=eligible_total,
        )

    edge_cursor = source_rebuild.get("edge_cursor")
    if edge_cursor is not None and not isinstance(edge_cursor, str):
        raise RuntimeError("invalid semantic_link staged synapse cursor")
    if not source_rebuild.get("edge_started"):
        fingerprint.begin_edges()
        source_rebuild["edge_started"] = True
    legacy_edge_cursor = source_rebuild.get("legacy_edge_cursor")
    legacy_edge_digest = source_rebuild.get("legacy_edge_digest")
    legacy_edge_verified = bool(source_rebuild.get("legacy_edge_verified", True))
    replay_edge_cursor = (
        str(legacy_edge_cursor)
        if saved_stage == "synapses" and legacy_edge_cursor is not None and not legacy_edge_verified
        else None
    )
    replay_edge = replay_edge_cursor is not None
    replay_edge_found = not replay_edge
    if (
        saved_stage == "synapses"
        and legacy_edge_cursor is None
        and not legacy_edge_verified
        and fingerprint.prefix_digest() != legacy_edge_digest
    ):
        raise RuntimeError("semantic_link neuron source changed before synapse scan")
    while not bool(source_rebuild.get("edge_complete")):
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
        if budget_check is not None:
            await budget_check()
        synapse_page_options: dict[str, Any] = {"limit": min(_SYNAPSE_PAGE_SIZE, 2000)}
        if reference_time is not None:
            synapse_page_options["created_before"] = reference_time
        page = await synapse_fetch(edge_cursor, **synapse_page_options)
        if not page:
            if replay_edge and not replay_edge_found:
                raise RuntimeError("semantic_link synapse cursor disappeared during resume")
            source_rebuild["edge_complete"] = True
            break
        for synapse in page:
            synapse_id = str(synapse.id)
            if replay_edge and replay_edge_cursor is not None and synapse_id > replay_edge_cursor:
                raise RuntimeError("semantic_link synapse cursor changed during resume")
            fingerprint.add_synapse(synapse)
            edge_cursor = synapse_id
            source_rebuild["edge_cursor"] = edge_cursor
            if replay_edge and synapse_id == replay_edge_cursor:
                replay_edge_found = True
                if fingerprint.prefix_digest() != legacy_edge_digest:
                    raise RuntimeError("semantic_link synapse source changed before its cursor")
                replay_edge = False
                source_rebuild["legacy_edge_verified"] = True
                break
        if replay_edge and not replay_edge_found:
            # Even an unvalidated legacy prefix advances only the staged source
            # cursor; the pending manifest continues to name its old cursor.
            if token is not None:
                await _assert_source_unchanged(storage, token, created_before=reference_time)
            await emit(
                "synapses",
                edge_cursor,
                prefix_digest=fingerprint.prefix_digest(),
                neuron_digest=neuron_digest,
                existing_pairs=0,
            )
            continue
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
        await emit(
            "synapses",
            edge_cursor,
            prefix_digest=fingerprint.prefix_digest(),
            neuron_digest=neuron_digest,
            existing_pairs=0,
        )
        if len(page) < min(_SYNAPSE_PAGE_SIZE, 2000):
            source_rebuild["edge_complete"] = True
            break

    if token is not None:
        await _assert_source_unchanged(storage, token, created_before=reference_time)
    source_fingerprint = fingerprint.finish()
    source_rebuild["source_complete"] = True
    source_rebuild["source_fingerprint"] = source_fingerprint
    if not source_scan_complete_on_entry:
        await emit(
            "synapses",
            edge_cursor,
            prefix_digest=source_fingerprint,
            neuron_digest=neuron_digest,
            existing_pairs=0,
        )
    if saved_stage == "similarity" and saved.get("source_fingerprint") != source_fingerprint:
        raise RuntimeError("semantic_link source changed after its similarity checkpoint")

    if len(eligible) < 2:
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
        return SemanticDiscoveryResult(
            source_fingerprint=source_fingerprint,
            source_token=token,
            reference_time_rebuilt=reference_time_rebuilt,
        )

    logger.debug(
        "semantic discovery: %d eligible neurons",
        len(eligible),
    )
    threshold = config.semantic_discovery_similarity_threshold
    max_pairs = min(config.semantic_discovery_max_pairs, MAX_PAIRS_HARD_CAP)
    top_k = SEMANTIC_TOP_K
    new_synapses: list[Synapse] = []
    skipped = 0
    skipped_created_this_run = 0
    pairs_evaluated = 0
    created_this_run: set[frozenset[str]] = set()
    created_digest = hashlib.sha256(b"semantic-link-created-v1")
    checkpoint_results: list[tuple[int, int, str]] = []

    def _make_synapse(i: int, j: int, sim: float) -> Synapse:
        return Synapse.create(
            source_id=eligible[i].id,
            target_id=eligible[j].id,
            type=SynapseType.SIMILAR_TO,
            weight=sim * 0.6,
            metadata={"_semantic_discovery": True, "cosine_similarity": round(sim, 4)},
            synapse_id=deterministic_edge_id(
                SynapseType.SIMILAR_TO, eligible[i].id, eligible[j].id
            ),
        )

    def _update_created_digest(synapse: Synapse) -> None:
        created_digest.update(
            json.dumps(
                {
                    "id": synapse.id,
                    "source_id": synapse.source_id,
                    "target_id": synapse.target_id,
                    "weight": synapse.weight,
                    "metadata": synapse.metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )

    def _record_created_synapse(i: int, j: int, sim: float) -> None:
        pair = frozenset({eligible[i].id, eligible[j].id})
        synapse = _make_synapse(i, j, sim)
        new_synapses.append(synapse)
        checkpoint_results.append((i, j, sim.hex()))
        _update_created_digest(synapse)
        created_this_run.add(pair)

    async def _existing_pairs(pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        if not pairs:
            return set()
        if reference_time is None:
            found = await pair_lookup(pairs)
        else:
            found = await pair_lookup(pairs, created_before=reference_time)
        if not isinstance(found, (set, frozenset, list, tuple)):
            raise RuntimeError("storage returned invalid semantic-link pair lookup results")
        normalized: set[tuple[str, str]] = set()
        requested = {_canonical_pair(first, second) for first, second in pairs}
        for item in found:
            if (
                not isinstance(item, (tuple, list))
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
            ):
                raise RuntimeError("storage returned invalid semantic-link pair lookup results")
            pair = _canonical_pair(item[0], item[1])
            if pair not in requested:
                raise RuntimeError("storage returned an unrequested semantic-link pair")
            normalized.add(pair)
        return normalized

    def _link(i: int, j: int, sim: float, *, already_exists: bool) -> bool:
        nonlocal skipped, skipped_created_this_run
        pair = frozenset({eligible[i].id, eligible[j].id})
        if pair in created_this_run:
            skipped_created_this_run += 1
            return False
        if already_exists:
            skipped += 1
            return False
        _record_created_synapse(i, j, sim)
        return True

    row_count = len(eligible)
    resume_row: int | None = None
    resume_v2 = saved_stage == "similarity" and saved_version == 2
    resume_v1_similarity = saved_stage == "similarity" and saved_version == 1
    if saved_stage == "similarity" and saved_cursor is not None:
        if not isinstance(saved_cursor, str) or not saved_cursor.isdecimal():
            raise RuntimeError("invalid semantic_link similarity row cursor")
        resume_row = int(saved_cursor)
        if str(resume_row) != saved_cursor or resume_row < 0 or resume_row >= row_count:
            raise RuntimeError("semantic_link similarity cursor is beyond the candidate rows")
    elif resume_v2:
        raise RuntimeError("invalid semantic_link similarity result cursor")

    legacy_similarity_rebuild = source_rebuild.get("similarity_rebuild")
    if legacy_similarity_rebuild is not None and not isinstance(legacy_similarity_rebuild, Mapping):
        raise RuntimeError("invalid semantic_link legacy similarity rebuild state")
    resumed_row_found = resume_row is None or resume_v2
    start_row = resume_row + 1 if resume_v2 and resume_row is not None else 0
    if (
        resume_v1_similarity
        and resume_row is not None
        and isinstance(legacy_similarity_rebuild, Mapping)
    ):
        raw_next_row = legacy_similarity_rebuild.get("next_row")
        raw_verified = legacy_similarity_rebuild.get("cursor_verified")
        if (
            type(raw_next_row) is not int
            or raw_next_row < 0
            or raw_next_row > resume_row + 1
            or type(raw_verified) is not bool
            or (raw_next_row > resume_row and not raw_verified)
        ):
            raise RuntimeError("invalid semantic_link legacy similarity rebuild cursor")
        start_row = raw_next_row
        resumed_row_found = raw_verified
    if resume_v2:
        if resume_row is None:
            raise RuntimeError("invalid semantic_link similarity result cursor")
        raw_results = saved.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > max_pairs:
            raise RuntimeError("invalid semantic_link similarity results")
        result_pairs: list[tuple[str, str]] = []
        for item in raw_results:
            if (
                isinstance(item, list)
                and len(item) == 3
                and type(item[0]) is int
                and type(item[1]) is int
                and 0 <= item[0] < row_count
                and 0 <= item[1] < row_count
            ):
                result_pairs.append((eligible[item[0]].id, eligible[item[1]].id))
        existing_result_pairs = await _existing_pairs(result_pairs)
        last_source_index = -1
        results_per_source: dict[int, int] = {}
        for item in raw_results:
            if not isinstance(item, list) or len(item) != 3:
                raise RuntimeError("invalid semantic_link similarity result entry")
            source_index, target_index, similarity_hex = item
            if (
                type(source_index) is not int
                or type(target_index) is not int
                or not isinstance(similarity_hex, str)
                or len(similarity_hex) > 32
            ):
                raise RuntimeError("invalid semantic_link similarity result entry")
            if (
                source_index < last_source_index
                or source_index < 0
                or source_index > resume_row
                or target_index < 0
                or target_index >= row_count
                or source_index == target_index
            ):
                raise RuntimeError("semantic_link similarity result index is out of range")
            last_source_index = source_index
            results_per_source[source_index] = results_per_source.get(source_index, 0) + 1
            if results_per_source[source_index] > top_k:
                raise RuntimeError("semantic_link similarity result exceeds top-K")
            try:
                sim = float.fromhex(similarity_hex)
            except ValueError as exc:
                raise RuntimeError("invalid semantic_link similarity value") from exc
            if sim.hex() != similarity_hex or sim < threshold:
                raise RuntimeError("semantic_link similarity result changed")
            pair = frozenset({eligible[source_index].id, eligible[target_index].id})
            if (
                _canonical_pair(eligible[source_index].id, eligible[target_index].id)
                in existing_result_pairs
                or pair in created_this_run
            ):
                raise RuntimeError("semantic_link checkpoint contains a duplicate pair")
            _record_created_synapse(source_index, target_index, sim)

        def _saved_counter(name: str) -> int:
            value = saved.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"invalid semantic_link similarity counter: {name}")
            return value

        skipped = _saved_counter("skipped_existing")
        skipped_created_this_run = _saved_counter("skipped_created_this_run")
        pairs_evaluated = _saved_counter("pairs_evaluated")
        synapses_created = _saved_counter("synapses_created")
        if (
            synapses_created != len(new_synapses)
            or synapses_created > max_pairs
            or pairs_evaluated > (resume_row + 1) * top_k
            or synapses_created + skipped + skipped_created_this_run > pairs_evaluated
            or created_digest.hexdigest() != saved.get("result_prefix_digest")
        ):
            raise RuntimeError("semantic_link similarity checkpoint failed validation")

    if resume_v1_similarity and legacy_similarity_rebuild is not None:
        if not isinstance(legacy_similarity_rebuild, Mapping):
            raise RuntimeError("invalid semantic_link legacy similarity rebuild state")
        raw_results = legacy_similarity_rebuild.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > max_pairs:
            raise RuntimeError("invalid semantic_link legacy similarity rebuild results")
        legacy_result_pairs: list[tuple[str, str]] = []
        for item in raw_results:
            if (
                not isinstance(item, list)
                or len(item) != 3
                or type(item[0]) is not int
                or type(item[1]) is not int
                or item[0] < 0
                or item[0] >= start_row
                or item[1] < 0
                or item[1] >= row_count
                or item[0] == item[1]
                or not isinstance(item[2], str)
            ):
                raise RuntimeError("invalid semantic_link legacy similarity result entry")
            legacy_result_pairs.append((eligible[item[0]].id, eligible[item[1]].id))
        if await _existing_pairs(legacy_result_pairs):
            raise RuntimeError("semantic_link legacy similarity results now have existing pairs")
        for source_index, target_index, similarity_hex in raw_results:
            try:
                similarity = float.fromhex(similarity_hex)
            except ValueError as exc:
                raise RuntimeError("invalid semantic_link legacy similarity value") from exc
            if similarity.hex() != similarity_hex or similarity < threshold:
                raise RuntimeError("semantic_link legacy similarity result changed")
            _record_created_synapse(source_index, target_index, similarity)

        def _rebuild_counter(name: str) -> int:
            value = legacy_similarity_rebuild.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"invalid semantic_link legacy similarity counter: {name}")
            return value

        skipped = _rebuild_counter("skipped_existing")
        skipped_created_this_run = _rebuild_counter("skipped_created_this_run")
        pairs_evaluated = _rebuild_counter("pairs_evaluated")
        if created_digest.hexdigest() != legacy_similarity_rebuild.get("result_prefix_digest"):
            raise RuntimeError("semantic_link legacy similarity rebuild digest mismatch")

    v1_similarity_cursor_prevalidated = bool(
        resume_v1_similarity
        and resume_row is not None
        and isinstance(legacy_similarity_rebuild, Mapping)
        and legacy_similarity_rebuild.get("cursor_verified") is True
        and legacy_similarity_rebuild.get("next_row") == resume_row + 1
    )
    checkpoint_interval = max(
        _YIELD_EVERY_ROWS,
        math.ceil(row_count / max(1, _MAX_SIMILARITY_CHECKPOINTS - 1)),
    )
    durable_row_cursor = (
        resume_row
        if resume_v2
        else (start_row - 1 if resume_v1_similarity and start_row > 0 else None)
    )

    async def checkpoint_row(index: int, similarity_backend: str) -> None:
        nonlocal durable_row_cursor
        details: dict[str, Any] = {
            "kind": "semantic_link_discovery",
            "version": 2,
            "stage": "similarity",
            "cursor": str(index),
            "neuron_digest": neuron_digest,
            "source_fingerprint": source_fingerprint,
            "similarity_backend": similarity_backend,
            "results": [[i, j, sim_hex] for i, j, sim_hex in checkpoint_results],
            "result_prefix_digest": created_digest.copy().hexdigest(),
            "synapses_created": len(new_synapses),
            "skipped_existing": skipped,
            "skipped_created_this_run": skipped_created_this_run,
            "pairs_evaluated": pairs_evaluated,
        }
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
            details["source_token"] = token
        source_rebuild["similarity_rebuild"] = None
        source_rebuild["legacy_checkpoint"] = details
        await write_source_checkpoint("similarity", str(index))
        durable_row_cursor = index

    async def persist_v1_similarity_rebuild(next_row: int, *, cursor_verified: bool) -> None:
        if token is not None:
            await _assert_source_unchanged(storage, token, created_before=reference_time)
        source_rebuild["similarity_rebuild"] = {
            "next_row": next_row,
            "cursor_verified": cursor_verified,
            "results": [[i, j, sim_hex] for i, j, sim_hex in checkpoint_results],
            "result_prefix_digest": created_digest.copy().hexdigest(),
            "synapses_created": len(new_synapses),
            "skipped_existing": skipped,
            "skipped_created_this_run": skipped_created_this_run,
            "pairs_evaluated": pairs_evaluated,
        }
        await write_source_checkpoint("similarity", saved_cursor, preserve_legacy_cursor=True)

    async def check_budget_before_row(next_index: int, similarity_backend: str) -> None:
        if budget_check is None:
            return
        try:
            await budget_check()
        except ConsolidationPausedError:
            completed_index = next_index - 1
            legacy_cursor_validated = resume_row is None or resume_v2 or resumed_row_found
            if (
                legacy_cursor_validated
                and completed_index >= 0
                and (durable_row_cursor is None or completed_index > durable_row_cursor)
            ):
                # Persist all completed rows before yielding at the run budget. A
                # restart then replays source pages but skips this similarity prefix.
                await checkpoint_row(completed_index, similarity_backend)
            raise

    async def run_row(index: int, similarities: Any, similarity_backend: str) -> bool:
        nonlocal pairs_evaluated, resumed_row_found
        row_similarities = similarities(index)
        pair_candidates = [
            (eligible[index].id, eligible[j].id) for j, sim in row_similarities if sim >= threshold
        ]
        existing_for_row = await _existing_pairs(pair_candidates)
        for candidate_index in row_similarities:
            j, sim = candidate_index
            pairs_evaluated += 1
            if sim < threshold:
                break
            existing_pair = _canonical_pair(eligible[index].id, eligible[j].id)
            _link(index, j, sim, already_exists=existing_pair in existing_for_row)
            if len(new_synapses) >= max_pairs:
                break
        if resume_row is not None and not resume_v2 and index == resume_row:
            if created_digest.hexdigest() != saved.get("result_prefix_digest"):
                raise RuntimeError("semantic_link result changed before its similarity cursor")
            if (
                len(new_synapses) != int(saved.get("synapses_created", -1))
                or skipped != int(saved.get("skipped_existing", -1))
                or skipped_created_this_run != int(saved.get("skipped_created_this_run", -1))
                or pairs_evaluated != int(saved.get("pairs_evaluated", -1))
            ):
                raise RuntimeError("semantic_link counters changed before its similarity cursor")
            resumed_row_found = True
        if resume_v1_similarity and resume_row is not None and index <= resume_row:
            verified_cursor = bool(resume_row == index and resumed_row_found)
            await persist_v1_similarity_rebuild(index + 1, cursor_verified=verified_cursor)
            if verified_cursor:
                # Only replace the v1 commit point after its result prefix and
                # counters have been reconstructed and validated.
                await checkpoint_row(index, similarity_backend)
        completed_rows = index + 1
        if budget_check is not None or completed_rows % _YIELD_EVERY_ROWS == 0:
            # Row cost scales with candidate count, so resumable runs with a
            # budget guard also yield once per row for database keepalives.
            await asyncio.sleep(0)
        reached_cap = len(new_synapses) >= max_pairs
        if (
            (durable_row_cursor is None or index > durable_row_cursor)
            and not (resume_v1_similarity and resume_row is not None and index < resume_row)
            and (
                completed_rows % checkpoint_interval == 0
                or reached_cap
                or completed_rows == row_count
            )
        ):
            await checkpoint_row(index, similarity_backend)
        return reached_cap

    if saved_stage != "similarity" or resume_row is None:
        await emit(
            "similarity",
            None,
            prefix_digest=source_fingerprint,
            neuron_digest=neuron_digest,
            source_fingerprint=source_fingerprint,
            result_prefix_digest=created_digest.hexdigest(),
            synapses_created=0,
            skipped_existing=0,
            skipped_created_this_run=0,
            pairs_evaluated=0,
        )

    try:
        import numpy as np

        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9

        def numpy_similarities(index: int) -> list[tuple[int, float]]:
            scores = matrix @ matrix[index]
            scores[index] = -1.0
            return [(int(j), float(scores[j])) for j in np.argsort(-scores)[:top_k]]

        similarity_backend = f"numpy_float32:{np.__version__}"
        if v1_similarity_cursor_prevalidated and resume_row is not None:
            await checkpoint_row(resume_row, similarity_backend)
        if resume_v2 and saved.get("similarity_backend") != similarity_backend:
            raise RuntimeError("semantic_link similarity backend changed after its checkpoint")
        row_limit = start_row if resume_v2 and len(new_synapses) >= max_pairs else row_count
        for row_index in range(start_row, row_limit):
            await check_budget_before_row(row_index, similarity_backend)
            if (
                resume_row is not None
                and not resume_v2
                and row_index > resume_row
                and not resumed_row_found
            ):
                raise RuntimeError("semantic_link similarity cursor is beyond the candidate rows")
            if await run_row(row_index, numpy_similarities, similarity_backend):
                break
    except ImportError:
        considered = min(len(eligible), _FALLBACK_MAX_NEURONS)
        if len(eligible) > considered:
            logger.warning(
                "numpy is not installed: semantic discovery is running its pure-python "
                "fallback and will consider only %d of %d eligible neurons this pass. "
                "Install numpy to link the whole set.",
                considered,
                len(eligible),
            )

        similarity_backend = "python_float64"
        if v1_similarity_cursor_prevalidated and resume_row is not None:
            await checkpoint_row(resume_row, similarity_backend)
        if resume_v2 and saved.get("similarity_backend") != similarity_backend:
            raise RuntimeError("semantic_link similarity backend changed after its checkpoint")
        if resume_row is not None and resume_v2 and resume_row >= considered:
            raise RuntimeError("semantic_link similarity cursor exceeds fallback candidate rows")

        def python_similarities(index: int) -> list[tuple[int, float]]:
            return sorted(
                (
                    (j, _cosine_similarity(vectors[index], vectors[j]))
                    for j in range(considered)
                    if j != index
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]

        row_limit = start_row if resume_v2 and len(new_synapses) >= max_pairs else considered
        for row_index in range(start_row, row_limit):
            await check_budget_before_row(row_index, similarity_backend)
            if await run_row(row_index, python_similarities, similarity_backend):
                break

    if resume_row is not None and not resumed_row_found:
        raise RuntimeError("semantic_link similarity cursor is beyond the candidate rows")
    if token is not None:
        await _assert_source_unchanged(storage, token, created_before=reference_time)
    return SemanticDiscoveryResult(
        neurons_embedded=len(vectors),
        pairs_evaluated=pairs_evaluated,
        synapses_created=len(new_synapses),
        skipped_existing=skipped,
        skipped_created_this_run=skipped_created_this_run,
        eligible_total=eligible_before_resample,
        synapses=new_synapses,
        truncated=len(new_synapses) >= max_pairs,
        source_fingerprint=source_fingerprint,
        source_token=token,
        reference_time_rebuilt=reference_time_rebuilt,
    )
