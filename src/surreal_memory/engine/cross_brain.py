"""Cross-brain recall — parallel spreading activation across multiple brains.

Resolves brain names against the active storage backend's brain listing
(SQLite or SurrealDB), fetches backend-aware shared storage for each, runs SA
in parallel via asyncio.gather, deduplicates results by SimHash, and merges
by confidence.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from surreal_memory.engine.retrieval_types import DepthLevel
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.unified_config import UnifiedConfig
    from surreal_memory.utils.geo import GeoFilter

logger = logging.getLogger(__name__)

# Hard cap on number of brains to query
MAX_CROSS_BRAINS = 5


@dataclass(frozen=True)
class CrossBrainFiber:
    """A fiber result from a cross-brain query."""

    fiber_id: str
    source_brain: str
    summary: str
    confidence: float
    content_hash: int = 0


@dataclass(frozen=True)
class CrossBrainResult:
    """Aggregated result from cross-brain recall."""

    query: str
    brains_queried: list[str]
    fibers: list[CrossBrainFiber]
    total_neurons_activated: int = 0
    merged_context: str = ""
    errors: dict[str, str] = field(default_factory=dict)


async def _query_single_brain(
    brain_name: str,
    query: str,
    depth: DepthLevel,
    max_tokens: int,
    tags: set[str] | None = None,
    near: GeoFilter | None = None,
) -> tuple[str, list[CrossBrainFiber], int, str, str | None]:
    """Query a single brain and return its results.

    Uses the backend-aware ``get_shared_storage`` accessor so this respects
    ``config.storage_backend`` ("sqlite" or "surrealdb") instead of always
    opening a throwaway local SQLite file — on a SurrealDB deployment that
    file is typically empty or nonexistent, which silently made cross-brain
    recall a no-op.

    The storage instance returned by ``get_shared_storage`` is cached/shared
    (per its own docstring, "to avoid connection leaks") and may be in
    concurrent use by the primary, non-cross-brain recall path running in
    this same process. It is intentionally NOT closed here — its lifecycle is
    owned by the shared-storage cache, not by an individual cross-brain
    query; closing it here could tear down a connection another concurrent
    caller still needs.

    Returns:
        Tuple of (brain_name, fibers, neurons_activated, context, error).
        ``error`` is None on success (including "queried fine, zero
        matches"); it carries a short message when the query itself raised.
    """
    from surreal_memory.unified_config import get_shared_storage

    try:
        storage = await get_shared_storage(brain_name)

        # Find the brain by name in the DB
        brain = await storage.find_brain_by_name(brain_name)
        if not brain:
            return brain_name, [], 0, "", None

        # Rows are scoped by the brain *name*; brain.id is a UUID for brains
        # created by older versions and would select an empty scope.
        storage.set_brain(brain.name)

        from surreal_memory.engine.retrieval import ReflexPipeline

        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=query,
            depth=depth,
            max_tokens=max_tokens,
            reference_time=utcnow(),
            near=near,
            tags=tags,
        )

        fibers: list[CrossBrainFiber] = []
        for fid in result.fibers_matched:
            fiber = await storage.get_fiber(fid)
            if fiber:
                fibers.append(
                    CrossBrainFiber(
                        fiber_id=fid,
                        source_brain=brain_name,
                        summary=fiber.summary or "",
                        confidence=result.confidence,
                        content_hash=getattr(fiber, "content_hash", 0) or 0,
                    )
                )

        return brain_name, fibers, result.neurons_activated, result.context or "", None
    except Exception as exc:
        logger.warning("Cross-brain query failed for '%s'", brain_name, exc_info=True)
        return brain_name, [], 0, "", str(exc) or exc.__class__.__name__


def _dedup_fibers(fibers: list[CrossBrainFiber]) -> list[CrossBrainFiber]:
    """Deduplicate fibers by SimHash proximity.

    Keeps the fiber with the highest confidence when duplicates are found.
    """
    from surreal_memory.utils.simhash import is_near_duplicate

    result: list[CrossBrainFiber] = []
    seen_hashes: list[tuple[int, int]] = []  # (hash, index in result)

    for fiber in fibers:
        if fiber.content_hash == 0:
            result.append(fiber)
            continue

        is_dup = False
        for existing_hash, idx in seen_hashes:
            if existing_hash != 0 and is_near_duplicate(fiber.content_hash, existing_hash):
                # Keep the one with higher confidence
                if fiber.confidence > result[idx].confidence:
                    result[idx] = fiber
                is_dup = True
                break

        if not is_dup:
            seen_hashes.append((fiber.content_hash, len(result)))
            result.append(fiber)

    return result


async def cross_brain_recall(
    config: UnifiedConfig,
    brain_names: list[str],
    query: str,
    depth: int = 1,
    max_tokens: int = 500,
    tags: set[str] | None = None,
    near: GeoFilter | None = None,
) -> CrossBrainResult:
    """Run recall across multiple brains in parallel.

    Args:
        config: Unified configuration (kept for signature compatibility;
            brain resolution itself goes through the process-wide config
            singleton via ``list_available_brains``/``get_shared_storage``)
        brain_names: List of brain names to query (max 5)
        query: The recall query
        depth: Depth level (0-3)
        max_tokens: Max tokens per brain query

    Returns:
        CrossBrainResult with merged, deduplicated results
    """
    from surreal_memory.unified_config import list_available_brains

    # Cap brain count
    brain_names = brain_names[:MAX_CROSS_BRAINS]

    # Resolve valid brain names via the active storage backend's own brain
    # listing. UnifiedConfig.list_brains() only inspects local sqlite *.db
    # fixture files, so it returns nothing for a SurrealDB deployment (or
    # stale leftovers from an old SQLite install) — list_available_brains()
    # is backend-aware and queries the real SurrealDB brain table when
    # storage_backend == "surrealdb". There is no per-brain file to check on
    # that backend, so membership in this listing is the only validity check.
    available = set(await list_available_brains())

    valid_brain_names: list[str] = [name for name in brain_names if name in available]
    for name in brain_names:
        if name not in available:
            logger.debug("Brain '%s' not found, skipping", name)

    if not valid_brain_names:
        return CrossBrainResult(
            query=query,
            brains_queried=[],
            fibers=[],
            merged_context="No valid brains found to query.",
        )

    try:
        depth_level = DepthLevel(depth)
    except ValueError:
        depth_level = DepthLevel.CONTEXT

    # Query all brains in parallel
    tasks = [
        _query_single_brain(name, query, depth_level, max_tokens, tags=tags, near=near)
        for name in valid_brain_names
    ]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    all_fibers: list[CrossBrainFiber] = []
    total_neurons = 0
    context_parts: list[str] = []
    queried: list[str] = []
    errors: dict[str, str] = {}

    for brain_name, fibers, neurons_activated, context, error in results:
        queried.append(brain_name)
        all_fibers.extend(fibers)
        total_neurons += neurons_activated
        if context:
            context_parts.append(f"[{brain_name}] {context}")
        if error:
            errors[brain_name] = error

    # Sort by confidence descending
    all_fibers.sort(key=lambda f: f.confidence, reverse=True)

    # Deduplicate by SimHash
    deduped = _dedup_fibers(all_fibers)

    # Merge context
    merged_context = "\n\n".join(context_parts) if context_parts else "No relevant memories found."

    return CrossBrainResult(
        query=query,
        brains_queried=queried,
        fibers=deduped,
        total_neurons_activated=total_neurons,
        merged_context=merged_context,
        errors=errors,
    )
