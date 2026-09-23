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
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
        )

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


class _SemanticFingerprintBuilder:
    """Stream the same canonical source payload used by consolidation snapshots."""

    def __init__(self, brain_id: str | None, config: BrainConfig) -> None:
        self._digest = hashlib.sha256()
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

    def finish(self) -> str:
        if not self._finished:
            self.begin_edges()
            self._digest.update(b"]}")
            self._finished = True
        return self._digest.hexdigest()


async def _discover_semantic_synapses_resumable(
    storage: NeuralStorage,
    config: BrainConfig,
    *,
    checkpoint: SemanticDiscoveryCheckpoint,
    resume_state: Mapping[str, Any] | None,
    budget_check: SemanticDiscoveryBudgetCheck | None,
) -> SemanticDiscoveryResult:
    """Discover links in durable pages; replay only the committed prefix on resume."""
    neuron_fetch = getattr(storage, "find_neurons_after_id", None)
    synapse_fetch = getattr(storage, "get_synapses_after_id", None)
    if not callable(neuron_fetch) or not callable(synapse_fetch):
        raise RuntimeError(
            "semantic_link discovery requires neuron and synapse keyset scan support"
        )

    saved = dict(resume_state or {})
    saved_stage = str(saved.get("stage") or "")
    saved_version = saved.get("version")
    if saved and (
        saved.get("kind") != "semantic_link_discovery"
        or saved_version not in (1, 2)
        or (saved_version == 2 and saved_stage != "similarity")
    ):
        raise RuntimeError("invalid semantic_link discovery checkpoint")
    saved_cursor = saved.get("cursor")
    brain_id = getattr(storage, "current_brain_id", None)
    fingerprint = _SemanticFingerprintBuilder(brain_id, config)
    per_type: dict[NeuronType, tuple[list[Neuron], list[list[float]]]] = {
        NeuronType.CONCEPT: ([], []),
        NeuronType.ENTITY: ([], []),
    }

    def record_neuron(neuron: Neuron) -> None:
        fingerprint.add_neuron(neuron)
        if neuron.type not in per_type or not neuron.content.strip():
            return
        # GRAPH_ONLY placeholders all share an artificial vector and must not
        # produce semantic edges.
        if neuron.content == GRAPH_ONLY_PLACEHOLDER:
            return
        embedding = neuron.metadata.get("_embedding")
        if not embedding:
            return
        neurons, vectors = per_type[neuron.type]
        neurons.append(neuron)
        vectors.append([float(value) for value in embedding])

    async def emit(
        stage: str,
        cursor: str | None,
        *,
        prefix_digest: str,
        **details: Any,
    ) -> None:
        await checkpoint(
            stage,
            cursor,
            {
                "kind": "semantic_link_discovery",
                "version": 1,
                "stage": stage,
                "cursor": cursor,
                "prefix_digest": prefix_digest,
                **details,
            },
        )

    neuron_cursor: str | None = None
    replay_neuron_cursor = (
        str(saved_cursor) if saved_stage == "neurons" and saved_cursor is not None else None
    )
    replay_neuron = replay_neuron_cursor is not None
    replay_neuron_found = not replay_neuron
    last_neuron_id: str | None = None
    while True:
        if not replay_neuron:
            await emit(
                "neurons",
                neuron_cursor,
                prefix_digest=fingerprint.prefix_digest(),
                eligible_neurons=sum(len(value[0]) for value in per_type.values()),
            )
        page = await neuron_fetch(
            neuron_cursor,
            limit=_EMBEDDING_PAGE_SIZE,
            ephemeral=None,
            include_embedding=True,
        )
        if not page:
            if replay_neuron and not replay_neuron_found:
                raise RuntimeError("semantic_link neuron cursor disappeared during resume")
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
            record_neuron(neuron)
            neuron_cursor = neuron_id
            last_neuron_id = neuron_id
            if replay_neuron and neuron_id == replay_neuron_cursor:
                replay_neuron_found = True
                if fingerprint.prefix_digest() != saved.get("prefix_digest"):
                    raise RuntimeError("semantic_link neuron source changed before its cursor")
                replay_neuron = False
                replayed_this_page = True
                break
        if not replay_neuron and not replay_neuron_found:
            raise RuntimeError("semantic_link neuron cursor disappeared during resume")
        if replay_neuron:
            continue
        if not replay_neuron_found:
            continue
        await emit(
            "neurons",
            neuron_cursor,
            prefix_digest=fingerprint.prefix_digest(),
            eligible_neurons=sum(len(value[0]) for value in per_type.values()),
        )
        if len(page) < _EMBEDDING_PAGE_SIZE and not replayed_this_page:
            break

    neuron_digest = fingerprint.prefix_digest()
    if (
        saved_stage in {"selection", "synapses", "similarity"}
        and saved.get("neuron_digest") != neuron_digest
    ):
        raise RuntimeError("semantic_link neuron source changed after its checkpoint")

    eligible: list[Neuron] = []
    vectors: list[list[float]] = []
    concept_neurons, concept_vectors = per_type[NeuronType.CONCEPT]
    entity_neurons, entity_vectors = per_type[NeuronType.ENTITY]
    longest = max(len(concept_neurons), len(entity_neurons))
    for index in range(longest):
        if index < len(concept_neurons):
            eligible.append(concept_neurons[index])
            vectors.append(concept_vectors[index])
        if index < len(entity_neurons):
            eligible.append(entity_neurons[index])
            vectors.append(entity_vectors[index])

    eligible_total = len(eligible)
    eligible_before_resample = eligible_total if eligible_total > MAX_NEURONS_TO_LINK else 0
    if len(eligible) > MAX_NEURONS_TO_LINK:
        if saved_stage not in {"synapses", "similarity"}:
            await emit(
                "selection",
                last_neuron_id,
                prefix_digest=neuron_digest,
                neuron_digest=neuron_digest,
                eligible_neurons=eligible_total,
            )
        eligible, vectors = await _select_candidates(storage, eligible, vectors)

    candidate_ids = {neuron.id for neuron in eligible}
    existing_pairs: set[frozenset[str]] = set()
    fingerprint.begin_edges()

    edge_cursor: str | None = None
    replay_edge_cursor = (
        str(saved_cursor) if saved_stage == "synapses" and saved_cursor is not None else None
    )
    replay_edge = replay_edge_cursor is not None
    replay_edge_found = not replay_edge
    if saved_stage == "synapses" and saved_cursor is None:
        if fingerprint.prefix_digest() != saved.get("prefix_digest"):
            raise RuntimeError("semantic_link neuron source changed before synapse scan")
    while True:
        if not replay_edge:
            await emit(
                "synapses",
                edge_cursor,
                prefix_digest=fingerprint.prefix_digest(),
                neuron_digest=neuron_digest,
                existing_pairs=len(existing_pairs),
            )
        page = await synapse_fetch(
            edge_cursor,
            limit=min(_SYNAPSE_PAGE_SIZE, 2000),
        )
        if not page:
            if replay_edge and not replay_edge_found:
                raise RuntimeError("semantic_link synapse cursor disappeared during resume")
            break
        replayed_this_page = False
        for synapse in page:
            synapse_id = str(synapse.id)
            if replay_edge and replay_edge_cursor is not None and synapse_id > replay_edge_cursor:
                raise RuntimeError("semantic_link synapse cursor changed during resume")
            fingerprint.add_synapse(synapse)
            if synapse.source_id in candidate_ids and synapse.target_id in candidate_ids:
                existing_pairs.add(frozenset({synapse.source_id, synapse.target_id}))
            edge_cursor = synapse_id
            if replay_edge and synapse_id == replay_edge_cursor:
                replay_edge_found = True
                if fingerprint.prefix_digest() != saved.get("prefix_digest"):
                    raise RuntimeError("semantic_link synapse source changed before its cursor")
                replay_edge = False
                replayed_this_page = True
                break
        if replay_edge and not replay_edge_found:
            continue
        await emit(
            "synapses",
            edge_cursor,
            prefix_digest=fingerprint.prefix_digest(),
            neuron_digest=neuron_digest,
            existing_pairs=len(existing_pairs),
        )
        if len(page) < min(_SYNAPSE_PAGE_SIZE, 2000) and not replayed_this_page:
            break

    source_fingerprint = fingerprint.finish()
    if saved_stage == "similarity" and saved.get("source_fingerprint") != source_fingerprint:
        raise RuntimeError("semantic_link source changed after its similarity checkpoint")

    if len(eligible) < 2:
        return SemanticDiscoveryResult(source_fingerprint=source_fingerprint)

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
        existing_pairs.add(pair)
        created_this_run.add(pair)

    def _link(i: int, j: int, sim: float) -> bool:
        nonlocal skipped, skipped_created_this_run
        pair = frozenset({eligible[i].id, eligible[j].id})
        if pair in created_this_run:
            skipped_created_this_run += 1
            return False
        if pair in existing_pairs:
            skipped += 1
            return False
        _record_created_synapse(i, j, sim)
        return True

    row_count = len(eligible)
    resume_row: int | None = None
    resume_v2 = saved_stage == "similarity" and saved_version == 2
    if saved_stage == "similarity" and saved_cursor is not None:
        if not isinstance(saved_cursor, str) or not saved_cursor.isdecimal():
            raise RuntimeError("invalid semantic_link similarity row cursor")
        resume_row = int(saved_cursor)
        if str(resume_row) != saved_cursor or resume_row < 0 or resume_row >= row_count:
            raise RuntimeError("semantic_link similarity cursor is beyond the candidate rows")
    elif resume_v2:
        raise RuntimeError("invalid semantic_link similarity result cursor")

    resumed_row_found = resume_row is None or resume_v2
    start_row = resume_row + 1 if resume_v2 and resume_row is not None else 0
    if resume_v2:
        if resume_row is None:
            raise RuntimeError("invalid semantic_link similarity result cursor")
        raw_results = saved.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > max_pairs:
            raise RuntimeError("invalid semantic_link similarity results")
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
            if pair in existing_pairs or pair in created_this_run:
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

    checkpoint_interval = max(
        _YIELD_EVERY_ROWS,
        math.ceil(row_count / max(1, _MAX_SIMILARITY_CHECKPOINTS - 1)),
    )
    durable_row_cursor = resume_row if resume_v2 else None

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
        snapshot = json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # The caller stores this JSON text as one string in a pending list, so
        # measure the outer serialized payload, including quote escaping.
        encoded_size = len(
            json.dumps([snapshot], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if encoded_size > _MAX_SIMILARITY_CHECKPOINT_BYTES:
            raise RuntimeError(
                "semantic_link similarity checkpoint payload exceeds "
                f"{_MAX_SIMILARITY_CHECKPOINT_BYTES} bytes; optimize durable result staging"
            )
        await checkpoint("similarity", str(index), details)
        durable_row_cursor = index

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
        for candidate_index in similarities(index):
            j, sim = candidate_index
            pairs_evaluated += 1
            if sim < threshold:
                break
            _link(index, j, sim)
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
        completed_rows = index + 1
        if budget_check is not None or completed_rows % _YIELD_EVERY_ROWS == 0:
            # Row cost scales with candidate count, so resumable runs with a
            # budget guard also yield once per row for database keepalives.
            await asyncio.sleep(0)
        reached_cap = len(new_synapses) >= max_pairs
        if completed_rows % checkpoint_interval == 0 or reached_cap or completed_rows == row_count:
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
    )
