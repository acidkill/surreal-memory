"""Reasoning distiller: staged reasoning_traces -> ReasoningBank patterns.

Heuristic (no-LLM) distillation, run inside consolidation (strategy
``LEARN_REASONING``, SUMMARIZE tier, after ``PROCESS_REASONING_TRACES`` ingest):

  batch unprocessed traces per model
    -> segment each thinking into ~12 closed-vocabulary reasoning "moves"
    -> classify a category (bge-m3 embedding vs seed centroids; keyword fallback)
    -> cluster within (model, category) by cosine (embeddings) or move-set Jaccard
    -> for each cluster >= min_cluster_support: build a ReasoningBank pattern
       (title / description / strategy / confidence / frequency) and materialize
       it as a fiber (_reasoning_pattern) + CONCEPT neuron + EFFECTIVE_FOR synapse
    -> mark traces processed; prune + cap the staging table.

Fully fail-soft: with the embedding provider DOWN, classification falls back to
keywords and clustering to move-set Jaccard, so distillation still produces
patterns (never raises on a missing provider).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Direction, Synapse, SynapseType
from surreal_memory.engine.clustering import UnionFind
from surreal_memory.engine.reasoning_naming import _is_loopback, build_namer
from surreal_memory.engine.reasoning_progress import PHASE_DISTILLING, MiningProgress

if TYPE_CHECKING:
    from surreal_memory.engine.embedding.provider import EmbeddingProvider
    from surreal_memory.engine.reasoning_naming import PatternNamer
    from surreal_memory.engine.reasoning_progress import ProgressCallback
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

logger = logging.getLogger(__name__)

_CLUSTER_COSINE = 0.75  # fallback only; reasoning_training.cluster_cosine wins
_CATEGORY_COS_THRESHOLD = 0.35
_MOVE_JACCARD = 0.6
_CLASSIFY_CHARS = 500
_BATCH_PER_MODEL = 200
# Ceiling on existing pattern fibers fetched for dedup/existing-count/coverage.
# Raised from 5000 for full-corpus mining across many models (u008).
_PATTERN_FETCH_LIMIT = 20_000

# ── Reasoning moves (closed vocabulary; regex discourse markers) ──────────────
_REASONING_MOVES: dict[str, re.Pattern[str]] = {
    "restate-goal": re.compile(
        r"(?i)\b(the goal is|the task is|i need to|we need to|objective is)\b"
    ),
    "decompose": re.compile(
        r"(?i)\b(break (this|it) down|decompose|sub-?problem|break into|the steps are)\b"
    ),
    "hypothesize": re.compile(
        r"(?i)\b(hypothes\w*|i suspect|maybe|might be|could be|perhaps|likely because)\b"
    ),
    "gather-evidence": re.compile(
        r"(?i)\b(let me (check|look|read|grep)|looking at|the evidence|i see that|the code shows|confirmed that)\b"
    ),
    "verify": re.compile(r"(?i)\b(verify|confirm|double-?check|make sure|ensure that|validate)\b"),
    "test-first": re.compile(
        r"(?i)\b(write a test|test first|failing test|red test|tdd|add a test)\b"
    ),
    "check-edge-cases": re.compile(
        r"(?i)\b(edge case|corner case|what if|boundary|off by one|empty (list|input)|null case|none case)\b"
    ),
    "backtrack": re.compile(
        r"(?i)\b(actually|wait|let me reconsider|scratch that|on second thought|rethink|hold on)\b"
    ),
    "compare-alternatives": re.compile(
        r"(?i)\b(option [a-z0-9]|alternative|versus|vs\.?|instead of|trade-?off|on the other hand|compared to)\b"
    ),
    "plan-steps": re.compile(
        r"(?i)\b(my plan|the approach|step \d|next,? i|i'?ll (start|do|then)|let me first)\b"
    ),
    "self-correct": re.compile(
        r"(?i)\b(i was wrong|correction|that'?s incorrect|my mistake|got it wrong|oops)\b"
    ),
    "summarize-decision": re.compile(
        r"(?i)\b(in summary|to summarize|conclusion|so i'?ll|decided to|the decision|therefore)\b"
    ),
}

# ── Category classification: seed descriptions (embeddings) + keyword fallback ─
_CATEGORY_SEEDS: dict[str, str] = {
    "debugging": "debugging errors, root cause analysis, stack traces, fixing bugs and failures",
    "planning": "planning steps, breaking down a task, deciding the approach and order of actions",
    "implementation": "implementing code, writing functions, adding a feature, coding the solution",
    "refactoring": "refactoring, cleaning up and restructuring code, improving readability, no behavior change",
    "research": "researching, reading documentation, exploring the codebase, understanding how something works",
    "verification": "verifying and testing, confirming correctness, checking outputs and edge cases",
    "architecture": "architecture and system design, module boundaries, data flow, design decisions",
    "data-analysis": "analyzing data, computing statistics, aggregating metrics, interpreting results",
}
_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "debugging": (
        "bug",
        "error",
        "exception",
        "stack trace",
        "root cause",
        "traceback",
        "crash",
        "failing",
        "debug",
    ),
    "refactoring": (
        "refactor",
        "clean up",
        "rename",
        "restructure",
        "simplify",
        "extract",
        "dead code",
        "tidy up",
    ),
    "verification": (
        "verify",
        "test",
        "confirm",
        "validate",
        "assert",
        "make sure",
        "edge case",
        "double-check",
    ),
    "architecture": (
        "architecture",
        "design",
        "module",
        "boundary",
        "data flow",
        "interface",
        "layer",
        "component",
    ),
    "data-analysis": (
        "analyze",
        "statistics",
        "metric",
        "aggregate",
        "distribution",
        "compute",
        "measure",
    ),
    "research": (
        "read",
        "documentation",
        "docs",
        "explore",
        "investigate",
        "grep",
        "find out",
        "look up",
    ),
    "planning": ("plan", "approach", "steps", "strategy", "break down", "sequence", "outline"),
    "implementation": (
        "implement",
        "write the",
        "add a",
        "function",
        "build",
        "create the",
        "feature",
        "method",
    ),
}
# Keyword-fallback precedence (specific -> generic).
_CATEGORY_ORDER: tuple[str, ...] = (
    "debugging",
    "refactoring",
    "verification",
    "architecture",
    "data-analysis",
    "research",
    "planning",
    "implementation",
)

_OTHER = "other"


def _has_keyword(content_lower: str, keyword: str) -> bool:
    """Whole-word (single token) / substring (phrase) keyword match."""
    if " " in keyword or "-" in keyword:
        return keyword in content_lower
    return re.search(rf"\b{re.escape(keyword)}\b", content_lower) is not None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def segment_moves(text: str) -> list[str]:
    """Return the reasoning moves present in *text*, in closed-vocabulary order."""
    if not text:
        return []
    return [move for move, pattern in _REASONING_MOVES.items() if pattern.search(text)]


def _classify_by_vector(vec: list[float], seeds: dict[str, list[float]]) -> str:
    best_cat, best_sim = _OTHER, _CATEGORY_COS_THRESHOLD
    for cat, cvec in seeds.items():
        sim = _cosine(vec, cvec)
        if sim >= best_sim:
            best_sim, best_cat = sim, cat
    return best_cat


def _classify_by_keywords(text: str, categories: tuple[str, ...]) -> str:
    low = text.lower()
    for cat in _CATEGORY_ORDER:
        if cat not in categories:
            continue
        if any(_has_keyword(low, kw) for kw in _CATEGORY_KEYWORDS.get(cat, ())):
            return cat
    return _OTHER


def _lcs_two(a: list[str], b: list[str]) -> list[str]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out: list[str] = []
    i = j = 0
    while i < m and j < n:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _lcs_all(seqs: list[list[str]]) -> list[str]:
    seqs = [s for s in seqs if s]
    if not seqs:
        return []
    acc = seqs[0]
    for s in seqs[1:]:
        acc = _lcs_two(acc, s)
        if not acc:
            break
    return acc


def _medoid_index(vectors: list[list[float]]) -> int:
    best_i, best_score = 0, -2.0
    for i in range(len(vectors)):
        score = sum(_cosine(vectors[i], vectors[j]) for j in range(len(vectors)) if j != i)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def _cluster(
    vectors: list[list[float]] | None,
    moves: list[list[str]],
    cosine_threshold: float = _CLUSTER_COSINE,
) -> list[list[int]]:
    """Cluster items by cosine (vectors) or move-set Jaccard (fallback).

    ``cosine_threshold`` belongs to the configured embedder, not to this
    module: two models embedding the same pair of traces disagree on the
    absolute cosine, so a value tuned for one silently clusters nothing under
    another.
    """
    n = len(moves)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if vectors is not None:
                if _cosine(vectors[i], vectors[j]) >= cosine_threshold:
                    uf.union(i, j)
            else:
                a, b = set(moves[i]), set(moves[j])
                union = a | b
                if union and len(a & b) / len(union) >= _MOVE_JACCARD:
                    uf.union(i, j)
    return list(uf.groups().values())


def _build_pattern(
    cluster_traces: list[dict[str, Any]],
    cluster_vectors: list[list[float]] | None,
    cluster_moves: list[list[str]],
    model: str,
    category: str,
    traces_in_category: int,
) -> dict[str, Any]:
    size = len(cluster_traces)
    if cluster_vectors:
        mi = _medoid_index(cluster_vectors)
    else:
        mi = max(range(size), key=lambda i: len(str(cluster_traces[i].get("content", ""))))
    medoid_content = str(cluster_traces[mi].get("content", ""))

    move_counts = Counter(m for moves in cluster_moves for m in moves)
    top_moves = [m for m, _ in move_counts.most_common(3)]
    title = f"{category}: " + ", ".join(top_moves) if top_moves else category

    lcs = _lcs_all(cluster_moves)
    strategy_moves = " -> ".join(lcs) if lcs else " -> ".join(top_moves)
    strategy = f"Moves: {strategy_moves}\n{medoid_content[:400]}"[:600]

    confidence = min(1.0, size / traces_in_category) if traces_in_category else 0.0
    # Signature keys on the cluster's exact trace set (not the display title) so
    # two distinct clusters that share the same top-moves title never collide.
    trace_key = ",".join(sorted(str(t.get("trace_hash", "")) for t in cluster_traces))
    signature = hashlib.sha256(f"{model}:{category}:{trace_key}".encode()).hexdigest()
    return {
        "model": model,
        "category": category,
        "title": title,
        "description": medoid_content[:200],
        "strategy": strategy,
        "confidence": round(confidence, 4),
        "frequency": size,
        "signature": signature,
    }


async def _find_or_create_concept(
    storage: NeuralStorage, content: str, metadata: dict[str, Any] | None = None
) -> str:
    existing = await storage.find_neurons(content_exact=content, limit=1)
    if existing:
        return existing[0].id
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content, metadata=metadata or {})
    await storage.add_neuron(neuron)
    return neuron.id


async def _materialize_pattern(
    storage: NeuralStorage,
    pattern: dict[str, Any],
    existing_sigs: set[str],
) -> bool:
    """Create a fiber + CONCEPT neuron + EFFECTIVE_FOR synapse for *pattern*.

    Idempotent by ``_reasoning_signature``: returns False (no-op) if the pattern
    was already materialized.
    """
    sig = pattern["signature"]
    if sig in existing_sigs:
        return False

    category_nid = await _find_or_create_concept(
        storage, f"reasoning_category:{pattern['category']}", {"_reasoning_category_concept": True}
    )
    pattern_nid = await _find_or_create_concept(
        storage, pattern["title"], {"_reasoning_pattern_concept": True}
    )

    existing_syn = await storage.get_synapses(
        source_id=pattern_nid, target_id=category_nid, type=SynapseType.EFFECTIVE_FOR
    )
    if existing_syn:
        synapse_id = existing_syn[0].id
    else:
        synapse = Synapse.create(
            source_id=pattern_nid,
            target_id=category_nid,
            type=SynapseType.EFFECTIVE_FOR,
            weight=min(1.0, float(pattern["confidence"])),
            direction=Direction.UNIDIRECTIONAL,
        )
        await storage.add_synapse(synapse)
        synapse_id = synapse.id

    fiber = Fiber.create(
        neuron_ids={pattern_nid, category_nid},
        synapse_ids={synapse_id},
        anchor_neuron_id=pattern_nid,
        pathway=[pattern_nid, category_nid],
        summary=pattern["title"],
        tags=set(),
        metadata={
            "_reasoning_pattern": True,
            "_source_model": pattern["model"],
            "_reasoning_category": pattern["category"],
            "_reasoning_title": pattern["title"],
            "_reasoning_description": pattern["description"],
            "_reasoning_strategy": pattern["strategy"],
            "_reasoning_frequency": pattern["frequency"],
            "_reasoning_confidence": pattern["confidence"],
            "_reasoning_signature": sig,
        },
    )
    # Patterns are activated only by injection (which may be OFF); unpinned they
    # are dead weight to decay/prune and vanish between sessions — pin them like
    # trained KB (doc_trainer) so lifecycle skips their neurons and fibers.
    fiber = replace(fiber, pinned=True)
    await storage.add_fiber(fiber)
    return True


_LOCAL_SAFE_PROVIDERS: tuple[str, ...] = ("openai", "openrouter", "bge_m3")
_OLLAMA_DEFAULT_BASE = "http://localhost:11434"


def _warn_remote_endpoint(endpoint: str) -> None:
    """Say the embedder was refused, rather than degrading silently.

    A silent fallback to keyword classification is indistinguishable from
    "embeddings are off" from the outside, which is how a misconfigured
    endpoint went unnoticed long enough to freeze category coverage.
    """
    logger.warning(
        "reasoning distiller: embedding endpoint %r is not loopback — "
        "reasoning traces never leave this machine, so classification "
        "falls back to keywords. Point SURREAL_MEMORY_EMBEDDING_ENDPOINT "
        "or [embedding] endpoint at a local server, or set "
        "[reasoning_training] allow_remote_endpoints to allow a remote "
        "gateway.",
        endpoint or "<unset>",
    )


def _endpoint_is_loopback(endpoint: str) -> bool:
    """True when *endpoint* is a URL whose host is genuinely loopback.

    ``_is_loopback`` takes a HOST, not a URL — parsing is the caller's job (see
    ``resolve_llm_endpoint``), so handing it a full URL always answers False.
    The host test itself lives there and is shared, so both the distill-LLM and
    the embedder reach a remote endpoint under exactly one rule.
    """
    from urllib.parse import urlsplit

    if not endpoint.strip():
        return False
    try:
        return _is_loopback(urlsplit(endpoint.strip()).hostname)
    except ValueError:
        return False


def _get_embedder(config: UnifiedConfig | None = None) -> EmbeddingProvider | None:
    """Best-effort LOCAL embedding provider; None if unavailable (fail-soft).

    Only a local Ollama or a loopback OpenAI-compatible endpoint (llamastash
    bge-m3) is used, so distillation stays local + fast and never blocks on a
    remote/heavy provider -- unless ``reasoning_training.allow_remote_endpoints``
    widens the gate to a configured remote http(s) gateway. Any failure -> None
    and the caller falls back to keyword classification + move-set clustering.

    The CONFIGURED provider wins when embeddings are enabled. Deciding by
    environment probe alone meant an unrelated GEMINI_API_KEY export shadowed a
    correctly configured loopback endpoint: the probe answered "gemini", which
    this function cannot build, so it returned None and every trace was
    classified by keyword while a working bge-m3 sat idle. Delegating to the
    canonical factory also picks up the configured model name and the shared
    provider cache, neither of which the hand-rolled construction had.
    """
    if config is not None and config.embedding.enabled:
        provider = (config.embedding.provider or "").strip().lower()
        endpoint = config.embedding.resolved_endpoint()
        # The same opt-in the naming LLM uses: remote endpoints (a LiteLLM
        # gateway commonly serves both roles) are accepted only when the
        # operator explicitly widened the loopback invariant.
        allow_remote = bool(getattr(config.reasoning_training, "allow_remote_endpoints", False))
        if provider == "auto":
            provider = ""  # fall through to the probe below
        elif provider == "ollama":
            # Ollama's own base URL, NOT the embedding endpoint: it is the value
            # this provider actually connects to, so it is the one that has to
            # clear the gate. Checking anything else would validate a string
            # that never reaches the socket.
            ollama_base = endpoint or os.environ.get("OLLAMA_BASE_URL", _OLLAMA_DEFAULT_BASE)
            if not (allow_remote or _endpoint_is_loopback(ollama_base)):
                _warn_remote_endpoint(ollama_base)
                return None
            try:
                from surreal_memory.engine.embedding.ollama_embedding import OllamaEmbedding

                return OllamaEmbedding(model=config.embedding.model, base_url=ollama_base)
            except Exception:
                logger.debug("reasoning distiller: ollama embedder could not be built")
                return None
        elif provider in _LOCAL_SAFE_PROVIDERS and (
            allow_remote or _endpoint_is_loopback(endpoint)
        ):
            try:
                from surreal_memory.engine.embedding.openai_embedding import OpenAIEmbedding

                # base_url is passed EXPLICITLY so the endpoint that cleared the
                # gate is the endpoint the client connects to. Delegating to the
                # provider factory instead re-resolved it independently: an
                # openrouter provider carries a hardcoded remote default, and an
                # openai one reads only the env var, so a loopback endpoint set
                # in config.toml passed the check while traces went to the cloud.
                return OpenAIEmbedding(model=config.embedding.model, base_url=endpoint)
            except Exception:
                logger.debug(
                    "reasoning distiller: configured provider %r could not be built", provider
                )
                return None
        elif provider in _LOCAL_SAFE_PROVIDERS:
            _warn_remote_endpoint(endpoint)
            return None

    try:
        from surreal_memory.engine.semantic_discovery import _auto_detect_provider

        provider_name, model_name = _auto_detect_provider()
    except Exception:
        logger.debug("reasoning distiller: no embedding provider detected", exc_info=True)
        return None

    try:
        endpoint = os.environ.get("SURREAL_MEMORY_EMBEDDING_ENDPOINT", "")
        if provider_name == "ollama":
            # Same rule as the configured path: gate the URL this provider will
            # really open, which is OLLAMA_BASE_URL, not the embedding endpoint.
            ollama_base = os.environ.get("OLLAMA_BASE_URL", _OLLAMA_DEFAULT_BASE)
            if not _endpoint_is_loopback(ollama_base):
                _warn_remote_endpoint(ollama_base)
                return None
            from surreal_memory.engine.embedding.ollama_embedding import OllamaEmbedding

            return OllamaEmbedding(model=model_name, base_url=ollama_base)
        if provider_name in ("openai", "openrouter") and _endpoint_is_loopback(endpoint):
            from surreal_memory.engine.embedding.openai_embedding import OpenAIEmbedding

            return OpenAIEmbedding(model=model_name, base_url=endpoint)
    except Exception:
        logger.debug("reasoning distiller: embedding provider construction failed", exc_info=True)
    return None


async def _seed_centroids(
    embedder: EmbeddingProvider, categories: tuple[str, ...]
) -> dict[str, list[float]] | None:
    try:
        descriptions = [_CATEGORY_SEEDS.get(c, c) for c in categories]
        vectors = await embedder.embed_batch(descriptions)
        return {c: list(v) for c, v in zip(categories, vectors, strict=False)}
    except Exception:
        logger.debug("reasoning distiller: seed embedding failed", exc_info=True)
        return None


async def _embed_texts(
    embedder: EmbeddingProvider | None, texts: list[str]
) -> list[list[float]] | None:
    if embedder is None or not texts:
        return None
    try:
        return [list(v) for v in await embedder.embed_batch(texts)]
    except Exception:
        logger.debug("reasoning distiller: trace embedding failed", exc_info=True)
        return None


@dataclass
class DistillResult:
    """Outcome of a distillation pass."""

    patterns_learned: int = 0
    traces_processed: int = 0
    models_seen: int = 0


async def _process_model_batch(
    storage: NeuralStorage,
    brain_id: str,
    rt: ReasoningTrainingConfig,
    model: str,
    traces: list[dict[str, Any]],
    embedder: EmbeddingProvider | None,
    seeds: dict[str, list[float]] | None,
    existing_sigs: set[str],
    budget: int,
    namer: PatternNamer | None = None,
) -> tuple[int, list[Any]]:
    """Distill one model's trace batch. Returns (patterns_created, consumed_ids).

    ``consumed_ids`` are the traces safe to mark processed: ``other`` traces,
    under-support categories, and every category fully clustered before this
    model's remaining per-target ``budget`` ran out. A category left unreached —
    or cut off mid-cluster — by the budget is NOT consumed, so the next run
    revisits it (already-materialized patterns are skipped by signature).
    """
    clf_texts = [
        f"{t.get('task_context', '')} {str(t.get('content', ''))[:_CLASSIFY_CHARS]}".strip()
        for t in traces
    ]
    moves_list = [segment_moves(str(t.get("content", ""))) for t in traces]
    vectors = await _embed_texts(embedder, clf_texts)
    categories = [
        _classify_by_vector(vectors[i], seeds)
        if (vectors is not None and seeds)
        else _classify_by_keywords(clf_texts[i], rt.categories)
        for i in range(len(traces))
    ]
    await storage.set_trace_categories(
        brain_id, {t["id"]: categories[i] for i, t in enumerate(traces)}
    )

    consumed: list[Any] = [traces[i]["id"] for i, c in enumerate(categories) if c == _OTHER]
    by_category: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        if cat != _OTHER:
            by_category.setdefault(cat, []).append(i)

    created = 0
    for category, idxs in by_category.items():
        if created >= budget:
            break  # budget reached before this category → leave its traces unprocessed
        if len(idxs) < rt.min_cluster_support:
            consumed.extend(traces[i]["id"] for i in idxs)  # too few to cluster; done
            continue
        sub_vectors = [vectors[i] for i in idxs] if vectors is not None else None
        sub_moves = [moves_list[i] for i in idxs]
        sub_traces = [traces[i] for i in idxs]
        capped_mid_category = False
        for local_cluster in _cluster(sub_vectors, sub_moves, rt.cluster_cosine):
            if len(local_cluster) < rt.min_cluster_support:
                continue
            if created >= budget:
                capped_mid_category = True
                break
            cluster_traces = [sub_traces[k] for k in local_cluster]
            cluster_vecs = [sub_vectors[k] for k in local_cluster] if sub_vectors else None
            cluster_moves = [sub_moves[k] for k in local_cluster]
            pattern = _build_pattern(
                cluster_traces, cluster_vecs, cluster_moves, model, category, len(idxs)
            )
            if namer is not None:
                # Prose only: the signature is already fixed by the cluster's
                # trace hashes, so naming cannot fork a pattern into a duplicate.
                pattern = await namer.rename(pattern, cluster_traces)
            if await _materialize_pattern(storage, pattern, existing_sigs):
                existing_sigs.add(pattern["signature"])
                created += 1
        if capped_mid_category:
            break  # do NOT consume this category's traces — revisit next run
        consumed.extend(traces[i]["id"] for i in idxs)
    return created, consumed


async def distill_reasoning_patterns(
    storage: NeuralStorage,
    brain_id: str,
    config: UnifiedConfig,
    *,
    embedder: EmbeddingProvider | None = None,
    drain: bool = False,
    progress: ProgressCallback | None = None,
) -> DistillResult:
    """Distill unprocessed reasoning traces into ReasoningBank pattern fibers.

    ``storage`` must already be on ``brain_id`` (graph writes use the current
    brain). Distillation is governed by per-model targets
    (``reasoning_training.pattern_targets``): for each detected source model the
    budget is ``max(0, target - existing_patterns_for_that_model)``. A model with
    budget 0 (its target is unset/0, or already met) is SKIPPED entirely — its
    traces stay unprocessed until a target is raised, so a preliminary Mine with
    no targets set only DETECTS models without distilling anything.

    ``drain=True`` (a manual ``POST /mine``) keeps fetching batches for a model
    until its budget is spent or its backlog is exhausted; ``drain=False``
    (background consolidation) processes at most one batch per model per run.
    Consumed traces are marked processed per batch so the next fetch returns
    fresh work. ``progress`` receives a distilling snapshot as each model
    advances.
    """
    rt = config.reasoning_training
    embedder = embedder or _get_embedder(config)
    seeds = await _seed_centroids(embedder, rt.categories) if embedder is not None else None
    # None unless distill_use_llm is on AND a local endpoint and model are set.
    # acquire() explicitly loads the chat model when distill_llm_load_cmd is
    # configured (a no-op otherwise, falling back to the first rename pulling
    # it in implicitly as before); released in the finally below either way,
    # so it is resident for this run only.
    namer = build_namer(rt)
    if namer is not None:
        await namer.acquire()

    existing = await storage.find_fibers(
        metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
    )
    existing_sigs = {
        str(f.metadata.get("_reasoning_signature"))
        for f in existing
        if f.metadata.get("_reasoning_signature")
    }
    existing_by_model: dict[str, int] = {}
    for f in existing:
        source_model = f.metadata.get("_source_model")
        if source_model:
            existing_by_model[str(source_model)] = existing_by_model.get(str(source_model), 0) + 1

    patterns_created = 0
    processed_ids: list[Any] = []
    models = await storage.get_reasoning_trace_models(brain_id)
    if rt.mining_models:
        # Honor the configured source-model globs so distillation is restricted to
        # the same models as ingestion (and to POST /mine's models= override). An
        # empty mining_models means "all models" (unchanged default behavior).
        models = [m for m in models if any(fnmatch(m, pat) for pat in rt.mining_models)]
    models_total = len(models)

    def _emit(current_model: str | None, models_done: int) -> None:
        if progress is not None:
            progress(
                MiningProgress(
                    phase=PHASE_DISTILLING,
                    traces_processed=len(processed_ids),
                    patterns_learned=patterns_created,
                    current_model=current_model,
                    models_done=models_done,
                    models_total=models_total,
                )
            )

    try:
        for idx, model in enumerate(models):
            budget = max(0, rt.pattern_targets.get(model, 0) - existing_by_model.get(model, 0))
            if budget <= 0:
                # Target unset/0 or already met → leave this model's traces unprocessed.
                _emit(model, idx + 1)
                continue
            while budget > 0:
                traces = await storage.get_unprocessed_reasoning_traces(
                    brain_id, limit=_BATCH_PER_MODEL, model=model
                )
                traces = traces[:_BATCH_PER_MODEL]
                if not traces:
                    break  # backlog for this model exhausted
                created, consumed = await _process_model_batch(
                    storage,
                    brain_id,
                    rt,
                    model,
                    traces,
                    embedder,
                    seeds,
                    existing_sigs,
                    budget,
                    namer,
                )
                patterns_created += created
                budget -= created
                if consumed:
                    processed_ids.extend(consumed)
                    # Mark consumed processed NOW so the next fetch returns fresh
                    # traces — otherwise a drain loop re-fetches the same batch forever.
                    await storage.mark_reasoning_traces_processed(brain_id, consumed)
                _emit(model, idx)
                # Termination guard: a batch that consumes nothing makes no forward
                # progress (budget hit 0 mid-category), so stop draining this model.
                if not consumed:
                    break
                if not drain:
                    break  # background consolidation: one batch per model per run
            _emit(model, idx + 1)
    finally:
        # Unconditional: an aborted or failed run must not leave the chat model
        # parked in VRAM either.
        if namer is not None:
            await namer.release()

    if processed_ids:
        await storage.prune_reasoning_traces(brain_id, rt.retention_days)
        await storage.cap_reasoning_traces(brain_id, rt.max_traces_total)

    return DistillResult(
        patterns_learned=patterns_created,
        traces_processed=len(processed_ids),
        models_seen=models_total,
    )


async def reasoning_coverage(
    storage: NeuralStorage,
    model: str,
    config: UnifiedConfig,
) -> dict[str, Any]:
    """Per-category coverage for *model*.

    A category is covered iff it has >= ``min_patterns_per_category`` pattern
    fibers with ``_source_model == model`` and confidence >= ``min_confidence``.
    ``coverage_percent`` = covered / len(categories) * 100 (``other`` excluded —
    it is never in ``categories``). ``storage`` must be on the target brain.
    """
    rt = config.reasoning_training
    fibers = await storage.find_fibers(
        metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
    )
    counts: dict[str, int] = dict.fromkeys(rt.categories, 0)
    for f in fibers:
        md = f.metadata
        if md.get("_source_model") != model:
            continue
        if float(md.get("_reasoning_confidence", 0.0)) < rt.min_confidence:
            continue
        cat = md.get("_reasoning_category")
        if cat in counts:
            counts[cat] += 1
    covered = {c: counts[c] >= rt.min_patterns_per_category for c in rt.categories}
    n_covered = sum(1 for v in covered.values() if v)
    percent = (n_covered / len(rt.categories) * 100.0) if rt.categories else 0.0
    return {"by_category": counts, "covered": covered, "coverage_percent": round(percent, 1)}
