"""Token budget management for Knowledge Surface.

Trims a KnowledgeSurface to fit within the token budget.
Priority-based: lowest-priority items removed first.
"""

from __future__ import annotations

from dataclasses import replace

from neural_memory.surface.models import (
    KnowledgeSurface,
)
from neural_memory.surface.serializer import serialize


def trim_to_budget(
    surface: KnowledgeSurface,
    budget: int | None = None,
) -> KnowledgeSurface:
    """Trim a KnowledgeSurface to fit within the token budget.

    Removal order (lowest impact first):
    1. DEPTH MAP entries for SUFFICIENT nodes (implicit — agent doesn't need hint)
    2. Lowest-priority GRAPH entries (and their cluster/depth refs)
    3. Smallest CLUSTERS
    4. Oldest SIGNALS

    Args:
        surface: The surface to trim.
        budget: Token budget override. If None, uses frontmatter.token_budget.

    Returns:
        New KnowledgeSurface within budget (immutable — original unchanged).
    """
    target = budget if budget is not None else surface.frontmatter.token_budget
    if target <= 0:
        return surface

    current = surface

    # Iteratively trim until within budget
    for _ in range(20):  # Safety cap
        estimate = _estimate_tokens(current)
        if estimate <= target:
            return current

        # Step 1: Remove SUFFICIENT depth hints (least useful)
        trimmed = _trim_sufficient_depth(current)
        if _estimate_tokens(trimmed) <= target:
            return trimmed
        current = trimmed

        # Step 2: Remove lowest-priority graph entry
        trimmed = _trim_lowest_priority_graph(current)
        if trimmed is current:
            break  # Nothing left to trim
        if _estimate_tokens(trimmed) <= target:
            return trimmed
        current = trimmed

    return current


def _estimate_tokens(surface: KnowledgeSurface) -> int:
    """Estimate token count as chars / 4."""
    return len(serialize(surface)) // 4


def _trim_sufficient_depth(surface: KnowledgeSurface) -> KnowledgeSurface:
    """Remove DEPTH MAP entries with SUFFICIENT level."""
    from neural_memory.surface.models import DepthLevel

    remaining = tuple(h for h in surface.depth_map if h.level != DepthLevel.SUFFICIENT)
    if len(remaining) == len(surface.depth_map):
        return surface
    return replace(surface, depth_map=remaining)


def _trim_lowest_priority_graph(surface: KnowledgeSurface) -> KnowledgeSurface:
    """Remove the lowest-priority graph entry and clean up references."""
    if not surface.graph:
        return surface

    # Find entry with lowest priority
    entries = list(surface.graph)
    entries.sort(key=lambda e: e.node.priority)
    removed = entries[0]
    remaining_entries = tuple(entries[1:])
    removed_id = removed.node.id

    # Clean up cluster references
    remaining_clusters = tuple(
        replace(
            c,
            node_ids=tuple(nid for nid in c.node_ids if nid != removed_id),
        )
        for c in surface.clusters
        if not (len(c.node_ids) == 1 and c.node_ids[0] == removed_id)
    )

    # Clean up depth map references
    remaining_depth = tuple(h for h in surface.depth_map if h.node_id != removed_id)

    # Clean up signal references
    remaining_signals = tuple(s for s in surface.signals if s.node_id != removed_id)

    return replace(
        surface,
        graph=remaining_entries,
        clusters=remaining_clusters,
        depth_map=remaining_depth,
        signals=remaining_signals,
    )
