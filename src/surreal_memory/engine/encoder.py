"""Memory encoder for converting experiences into neural structures."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from surreal_memory.core.neuron import Neuron
from surreal_memory.core.synapse import Synapse
from surreal_memory.engine.arousal import ArousalStep
from surreal_memory.engine.context_retrieval import ContextFingerprintStep
from surreal_memory.engine.pipeline import Pipeline, PipelineContext
from surreal_memory.engine.pipeline_steps import (
    AutoTagStep,
    BuildFiberStep,
    ConfirmatoryBoostStep,
    ConflictDetectionStep,
    CoOccurrenceStep,
    CreateAnchorStep,
    CreateSynapsesStep,
    CrossMemoryLinkStep,
    DecisionComponentStep,
    DedupCheckStep,
    EmotionStep,
    ExtractActionNeuronsStep,
    ExtractConceptNeuronsStep,
    ExtractEntityNeuronsStep,
    ExtractIntentNeuronsStep,
    ExtractTimeNeuronsStep,
    RelationExtractionStep,
    SemanticLinkingStep,
    StructuredDataEncoderStep,
    StructureDetectionStep,
    TemporalLinkingStep,
)
from surreal_memory.engine.prediction_error import PredictionErrorStep
from surreal_memory.engine.temporal_binding import TemporalBindingStep
from surreal_memory.extraction.entities import EntityExtractor
from surreal_memory.extraction.relations import RelationExtractor
from surreal_memory.extraction.sentiment import SentimentExtractor
from surreal_memory.extraction.temporal import TemporalExtractor
from surreal_memory.utils.tag_normalizer import TagNormalizer
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _inline_embed_timeout() -> float:
    """Seconds the write path will wait for inline embeddings before giving up.

    Default 10s: comfortably above a local endpoint (bge-m3 via llamastash answers
    in ~15 ms) and comfortably below the 30s tool-call cap that MCP hosts such as
    Claude Code and Claude Desktop enforce, so a slow provider can never turn a
    successful write into a client-side timeout. Override with
    ``SURREAL_MEMORY_INLINE_EMBED_TIMEOUT`` (<= 0 disables the bound).
    """
    raw = os.environ.get("SURREAL_MEMORY_INLINE_EMBED_TIMEOUT", "").strip()
    if not raw:
        return 10.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid SURREAL_MEMORY_INLINE_EMBED_TIMEOUT=%r — falling back to 10s", raw)
        return 10.0
    return value if value > 0 else 0.0


if TYPE_CHECKING:
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.core.fiber import Fiber
    from surreal_memory.engine.dedup.pipeline import DedupPipeline
    from surreal_memory.storage.base import NeuralStorage


@dataclass
class EncodingResult:
    """
    Result of encoding a memory.

    Attributes:
        fiber: The created memory fiber
        neurons_created: List of newly created neurons
        neurons_linked: List of existing neuron IDs that were linked
        synapses_created: List of newly created synapses
        extraction_stats: Optional concept-extraction counters when callers
            opt in via verbose_extraction. Surface schema:
            ``{"dropped_short", "dropped_noise", "dropped_duplicate_entity"}``.
    """

    fiber: Fiber
    neurons_created: list[Neuron]
    neurons_linked: list[str]
    synapses_created: list[Synapse]
    conflicts_detected: int = 0
    extraction_stats: dict[str, int] | None = None
    # U3: old anchor-neuron ids this memory supersedes (applied post-save by the
    # caller, which owns the freshly-created fiber_id). Empty in the common case.
    pending_supersessions: list[str] = field(default_factory=list)


def build_default_pipeline(
    temporal_extractor: TemporalExtractor,
    entity_extractor: EntityExtractor,
    relation_extractor: RelationExtractor,
    sentiment_extractor: SentimentExtractor,
    tag_normalizer: TagNormalizer,
    dedup_pipeline: DedupPipeline | None = None,
) -> Pipeline:
    """Build the default encoding pipeline with all 15 steps.

    This is the standard pipeline that reproduces the original monolithic
    ``encode()`` behavior. Users can customize by removing, replacing,
    or reordering steps.

    Args:
        temporal_extractor: Temporal extraction instance
        entity_extractor: Entity extraction instance
        relation_extractor: Relation extraction instance
        sentiment_extractor: Sentiment extraction instance
        tag_normalizer: Tag normalization instance
        dedup_pipeline: Optional dedup pipeline

    Returns:
        A Pipeline with all default steps.
    """
    return Pipeline(
        [
            ExtractTimeNeuronsStep(temporal_extractor=temporal_extractor),
            ExtractEntityNeuronsStep(entity_extractor=entity_extractor),
            ExtractConceptNeuronsStep(),
            ExtractActionNeuronsStep(),
            ExtractIntentNeuronsStep(),
            StructureDetectionStep(),
            DecisionComponentStep(),
            AutoTagStep(tag_normalizer=tag_normalizer),
            DedupCheckStep(dedup_pipeline=dedup_pipeline),
            PredictionErrorStep(),
            CreateAnchorStep(),
            StructuredDataEncoderStep(),
            CreateSynapsesStep(),
            CoOccurrenceStep(),
            EmotionStep(sentiment_extractor=sentiment_extractor),
            ArousalStep(),
            RelationExtractionStep(relation_extractor=relation_extractor),
            ConfirmatoryBoostStep(),
            ConflictDetectionStep(),
            TemporalLinkingStep(),
            SemanticLinkingStep(),
            CrossMemoryLinkStep(),
            ContextFingerprintStep(),
            BuildFiberStep(),
            TemporalBindingStep(),
        ]
    )


_INSTRUCTION_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "my",
        "our",
        "your",
        "their",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "such",
        "just",
        "than",
        "then",
        "when",
        "where",
        "which",
        "who",
        "what",
        "how",
        "why",
        "also",
        "very",
        "too",
        "up",
        "out",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "only",
        "own",
        "same",
        "other",
        "there",
        "here",
        "new",
        "now",
        "always",
        "never",
    }
)


def _extract_trigger_patterns(content: str, max_patterns: int = 5) -> list[str]:
    """Extract significant keywords from instruction content as trigger patterns.

    Takes the top N significant keywords (filtered against stop words) from
    the instruction text. These words will be used to boost the instruction
    during recall when the query overlaps with them.

    Args:
        content: The instruction text.
        max_patterns: Maximum number of trigger patterns to return.

    Returns:
        List of lowercase keyword strings.
    """
    # Normalize: lowercase, remove punctuation except apostrophes
    normalized = re.sub(r"[^\w\s']", " ", content.lower())
    words = normalized.split()
    seen: dict[str, int] = {}
    for word in words:
        word = word.strip("'")
        if len(word) >= 4 and word not in _INSTRUCTION_STOP_WORDS:
            seen[word] = seen.get(word, 0) + 1
    # Sort by frequency descending, then alphabetically for stability
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
    return [kw for kw, _ in ranked[:max_patterns]]


def _inject_instruction_metadata(
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Merge instruction tracking fields into metadata (non-destructive).

    Only adds keys that are not already present — preserves any existing
    instruction metadata set by the caller.

    Args:
        content: Instruction text (used for trigger extraction).
        metadata: Existing metadata dict.

    Returns:
        New metadata dict with instruction fields merged in.
    """
    defaults: dict[str, Any] = {
        "version": 1,
        "execution_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "success_rate": None,
        "last_executed_at": None,
        "failure_modes": [],
        "trigger_patterns": _extract_trigger_patterns(content),
        "refinement_history": [],
    }
    # Merge: existing keys win; only add missing ones
    merged = {**defaults, **metadata}
    return merged


class MemoryEncoder:
    """
    Encoder for converting experiences into neural structures.

    The encoder:
    1. Extracts neurons from content (time, entities, actions, concepts)
    2. Finds existing similar neurons for de-duplication
    3. Creates synapses based on relationships
    4. Bundles everything into a Fiber
    5. Auto-links with nearby temporal neurons

    Internally delegates to a composable :class:`Pipeline` of steps.
    The default pipeline reproduces the original behavior. Pass a custom
    ``pipeline`` to ``__init__`` to customize encoding.
    """

    def __init__(
        self,
        storage: NeuralStorage,
        config: BrainConfig,
        temporal_extractor: TemporalExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        dedup_pipeline: DedupPipeline | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        """
        Initialize the encoder.

        Args:
            storage: Storage backend
            config: Brain configuration
            temporal_extractor: Custom temporal extractor
            entity_extractor: Custom entity extractor
            relation_extractor: Custom relation extractor
            dedup_pipeline: Optional DedupPipeline for anchor deduplication
            pipeline: Custom pipeline (overrides default step composition)
        """
        self._storage = storage
        self._config = config
        self._temporal = temporal_extractor or TemporalExtractor()
        self._entity = entity_extractor or EntityExtractor()
        self._relation = relation_extractor or RelationExtractor()
        self._sentiment = SentimentExtractor()
        self._tag_normalizer = TagNormalizer()
        self._dedup_pipeline = dedup_pipeline

        self._pipeline = pipeline or build_default_pipeline(
            temporal_extractor=self._temporal,
            entity_extractor=self._entity,
            relation_extractor=self._relation,
            sentiment_extractor=self._sentiment,
            tag_normalizer=self._tag_normalizer,
            dedup_pipeline=self._dedup_pipeline,
        )

    @property
    def pipeline(self) -> Pipeline:
        """Access the encoding pipeline (read-only)."""
        return self._pipeline

    async def encode(
        self,
        content: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        tags: set[str] | None = None,
        language: str = "auto",
        *,
        skip_conflicts: bool = False,
        skip_time_neurons: bool = False,
        initial_stage: str = "",
        salience_ceiling: float = 0.0,
    ) -> EncodingResult:
        """
        Encode content into neural structures.

        Args:
            content: The text content to encode
            timestamp: When this memory occurred (default: now)
            metadata: Additional metadata to attach
            tags: Optional tags for the fiber
            language: Language hint ("vi", "en", or "auto")
            skip_conflicts: Skip conflict detection (for bulk doc training).
            skip_time_neurons: Skip TIME neuron creation (for bulk doc training).
            initial_stage: Override maturation stage (e.g. "episodic" for doc training).
            salience_ceiling: Cap initial fiber salience (0 = no cap).

        Returns:
            EncodingResult with created structures
        """
        if timestamp is None:
            timestamp = utcnow()

        # Auto-populate instruction metadata for instruction/workflow types
        merged_metadata = dict(metadata or {})
        mem_type = merged_metadata.get("type", "")
        if mem_type in ("instruction", "workflow"):
            merged_metadata = _inject_instruction_metadata(content, merged_metadata)

        ctx = PipelineContext(
            content=content,
            timestamp=timestamp,
            metadata=merged_metadata,
            tags=tags or set(),
            language=language,
            skip_conflicts=skip_conflicts,
            skip_time_neurons=skip_time_neurons,
            initial_stage=initial_stage,
            salience_ceiling=salience_ceiling,
        )

        ctx = await self._pipeline.run(ctx, self._storage, self._config)

        # Extract fiber from context (set by BuildFiberStep) — use .get() to avoid mutation
        fiber = ctx.effective_metadata.get("_pipeline_fiber")
        if fiber is None:
            msg = "Pipeline did not produce a fiber (missing BuildFiberStep?)"
            raise RuntimeError(msg)

        # Post-encode: schema assimilation + interference detection (non-critical)
        if ctx.anchor_neuron is not None:
            await self._post_encode_neuro(ctx.anchor_neuron)

        # Embed the freshly-created neurons inline so semantic recall works on
        # them immediately, instead of only after a batch ``smem reindex``.
        await self._embed_created_neurons(ctx)

        return EncodingResult(
            fiber=fiber,
            neurons_created=ctx.neurons_created,
            neurons_linked=ctx.neurons_linked,
            synapses_created=ctx.synapses_created,
            conflicts_detected=ctx.conflicts_detected,
            extraction_stats={
                "dropped_short": ctx.dropped_short,
                "dropped_noise": ctx.dropped_noise,
                "dropped_duplicate_entity": ctx.dropped_duplicate_entity,
            },
            pending_supersessions=list(ctx.pending_supersessions),
        )

    async def _embed_created_neurons(self, ctx: PipelineContext) -> None:
        """Embed the non-ephemeral neurons this encode just created so semantic
        recall works on them immediately.

        Embeddings used to be populated only by a batch ``smem reindex``, so a
        freshly-saved memory was keyword-recallable but NOT semantically
        recallable until the next reindex ran. One ``embed_batch`` call covers
        the whole memory; against a local endpoint (bge-m3 via llamastash,
        ~15 ms) this adds negligible latency. Fully fail-soft: if embeddings are
        disabled or no provider is available, memories stay keyword-only — no
        error, no slowdown — exactly the prior behaviour. Ephemeral and TIME
        neurons are skipped (disposable / not semantically meaningful).
        """
        logger = logging.getLogger(__name__)
        try:
            from surreal_memory.engine.semantic_discovery import (
                _create_provider,
                _effective_embedding,
            )

            enabled, _, _ = _effective_embedding(self._config)
        except Exception:
            return
        if not enabled:
            return

        from surreal_memory.core.neuron import NeuronType

        seen: set[str] = set()
        candidates: list[Neuron] = []
        for n in [*ctx.neurons_created, ctx.anchor_neuron]:
            if n is None or n.id in seen:
                continue
            if getattr(n, "ephemeral", False):
                continue  # disposable (24 h) — not worth an embedding
            if n.type == NeuronType.TIME:
                continue  # timestamps are not semantically meaningful
            if not n.content or len(n.content) < 3:
                continue
            if "_embedding" in n.metadata:
                continue
            seen.add(n.id)
            candidates.append(n)
        if not candidates:
            return

        try:
            provider = _create_provider(self._config, task_type="RETRIEVAL_DOCUMENT")
            # Bound the wait. This method is fail-soft by design — "no provider"
            # already means "keyword-only memory, no error" — but an *unavailable*
            # provider and a *slow* one used to be treated differently: a remote or
            # rate-limited endpoint blocked the write until its own timeout, pushing
            # `smem_remember` past the 30s tool-call cap MCP hosts impose. The write
            # itself had already succeeded, so the caller saw a timeout and could not
            # tell whether the memory landed — the exact failure this tool exists to
            # prevent. A local endpoint (bge-m3 via llamastash, ~15 ms) is nowhere
            # near this budget; anything that is pays with a slightly later vector
            # instead of a lost write. `smem reindex` back-fills.
            budget = _inline_embed_timeout()
            embed = provider.embed_batch([n.embedding_text() for n in candidates])
            vectors = await (asyncio.wait_for(embed, timeout=budget) if budget else embed)
        except TimeoutError:
            logger.warning(
                "Inline embedding exceeded %.0fs for %d neuron(s) — memory saved "
                "keyword-only; run `smem reindex` to back-fill the vectors.",
                _inline_embed_timeout(),
                len(candidates),
            )
            return
        except Exception:
            logger.debug("Inline embedding skipped (provider unavailable)", exc_info=True)
            return

        # Prefer a single batched write (SurrealDB collapses this into one
        # multi-statement UPDATE per chunk): inline embedding otherwise costs
        # one round-trip per created neuron.
        pairs = [(n.id, list(v)) for n, v in zip(candidates, vectors, strict=False)]
        try:
            await self._storage.update_neuron_embeddings(pairs)
            return
        except Exception:
            logger.debug("Batch inline embed update failed; falling back", exc_info=True)
        for neuron, vector in zip(candidates, vectors, strict=False):
            try:
                await self._storage.update_neuron(neuron.with_metadata(_embedding=list(vector)))
            except Exception:
                logger.debug("Inline embed update failed: %s", neuron.id, exc_info=True)

    async def _post_encode_neuro(self, anchor: Neuron) -> None:
        """Run post-encode neuroscience hooks (schema assimilation + interference).

        Non-critical: failures are logged and swallowed so encoding always succeeds.
        Schema assimilation skips small brains (< schema_min_cluster_size) since
        there aren't enough neurons to form meaningful schemas.
        """
        # Schema assimilation: auto-wire when brain has enough memories
        schema_enabled = getattr(self._config, "schema_assimilation_enabled", False)
        if isinstance(schema_enabled, bool) and schema_enabled:
            try:
                # Skip small brains — not enough neurons for schema formation
                min_cluster = getattr(self._config, "schema_min_cluster_size", 10)
                min_cluster = int(min_cluster) if isinstance(min_cluster, (int, float)) else 10
                stats = await self._storage.get_stats(self._storage.brain_id or "")
                neuron_count = stats.get("neuron_count", 0)
                if neuron_count >= min_cluster:
                    from surreal_memory.engine.schema_assimilation import assimilate_or_accommodate

                    await assimilate_or_accommodate(anchor, self._storage, self._config)
            except Exception:
                logger.debug("Post-encode schema assimilation failed (non-critical)", exc_info=True)

        # Interference detection: detect and resolve competing memories
        interference_enabled = getattr(self._config, "interference_detection_enabled", False)
        if isinstance(interference_enabled, bool) and interference_enabled:
            try:
                from surreal_memory.engine.interference import (
                    detect_interference,
                    resolve_interference,
                )

                results = await detect_interference(anchor, self._storage, self._config)
                if results:
                    await resolve_interference(results, anchor, self._storage, self._config)
            except Exception:
                logger.debug(
                    "Post-encode interference detection failed (non-critical)", exc_info=True
                )
