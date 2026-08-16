"""Semantic drift detection — find tag clusters that should be merged.

Port of the tag-cooccurrence + Jaccard clustering half of
``engine/drift_detection.py``, removed in 20dbe5f6 alongside the SQLite
backend. Uses tag co-occurrence + Jaccard similarity to detect when different
tags refer to the same concept, and outputs clusters with a confidence-based
suggestion: merge, alias, or review.

Deliberately NOT ported: Wasserstein-1 activation drift and cross-session
temporal drift. Both had no working data source even before the SQLite
removal (see PR #151) — resurrecting them here would ship the same "always
empty" dead code this run exists to stop shipping.

Runs during consolidation (not hot path). Zero LLM, pure statistics.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from surreal_memory.engine.clustering import UnionFind

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Constants (unchanged from the removed mixin — this is a port) ──────────

JACCARD_MERGE_THRESHOLD = 0.7  # Jaccard >= 0.7 → likely synonyms (auto-merge)
JACCARD_ALIAS_THRESHOLD = 0.4  # Jaccard >= 0.4 → related concepts (alias)
JACCARD_REVIEW_THRESHOLD = 0.3  # Jaccard >= 0.3 → possible drift (review)
MIN_COOCCURRENCE_COUNT = 3  # Minimum co-occurrences to consider
MAX_CLUSTER_SIZE = 10  # Max tags in a single cluster
MIN_TAG_FIBERS = 2  # Tag must appear in >= 2 fibers to be considered


@dataclass(frozen=True)
class TagCluster:
    """A cluster of tags detected as potentially referring to the same concept."""

    canonical: str  # Most-used tag in the cluster
    members: frozenset[str]  # All tags in the cluster (including canonical)
    confidence: float  # Average Jaccard similarity within cluster
    evidence: str = ""  # Human-readable explanation


@dataclass(frozen=True)
class DriftReport:
    """A single drift detection result with action suggestion."""

    cluster: TagCluster
    suggestion: str  # "merge" | "alias" | "review"
    cluster_id: str = ""  # Stable ID for persistence


def compute_jaccard(
    tag_a: str,
    tag_b: str,
    tag_fiber_counts: dict[str, int],
    cooccurrence_count: int,
) -> float:
    """Jaccard similarity between two tags: intersection over union.

    = cooccurrence / (count_a + count_b - cooccurrence)
    """
    count_a = tag_fiber_counts.get(tag_a, 0)
    count_b = tag_fiber_counts.get(tag_b, 0)

    if count_a == 0 or count_b == 0:
        return 0.0

    union = count_a + count_b - cooccurrence_count
    if union <= 0:
        return 0.0

    return cooccurrence_count / union


def _cluster_id(members: frozenset[str]) -> str:
    """Stable cluster ID from sorted member tags."""
    key = "|".join(sorted(members))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def detect_clusters(
    cooccurrences: list[tuple[str, str, int]],
    tag_fiber_counts: dict[str, int],
) -> list[DriftReport]:
    """Detect tag clusters using Union-Find on Jaccard-similar pairs.

    Args:
        cooccurrences: List of (tag_a, tag_b, count) pairs.
        tag_fiber_counts: Dict of {tag: fiber_count} for Jaccard's denominator.

    Returns:
        List of DriftReport, confidence descending.
    """
    if not cooccurrences:
        return []

    all_tags: list[str] = []
    tag_index: dict[str, int] = {}
    for tag_a, tag_b, _count in cooccurrences:
        for tag in (tag_a, tag_b):
            if tag not in tag_index:
                tag_index[tag] = len(all_tags)
                all_tags.append(tag)

    if len(all_tags) < 2:
        return []

    uf = UnionFind(len(all_tags))
    pair_jaccards: dict[tuple[int, int], float] = {}

    for tag_a, tag_b, count in cooccurrences:
        if count < MIN_COOCCURRENCE_COUNT:
            continue
        if tag_fiber_counts.get(tag_a, 0) < MIN_TAG_FIBERS:
            continue
        if tag_fiber_counts.get(tag_b, 0) < MIN_TAG_FIBERS:
            continue

        jaccard = compute_jaccard(tag_a, tag_b, tag_fiber_counts, count)

        if jaccard >= JACCARD_REVIEW_THRESHOLD:
            idx_a = tag_index[tag_a]
            idx_b = tag_index[tag_b]
            pair_jaccards[(idx_a, idx_b)] = jaccard

            # Only union above alias threshold (review pairs stay separate).
            if jaccard >= JACCARD_ALIAS_THRESHOLD:
                uf.union(idx_a, idx_b)

    if not pair_jaccards:
        return []

    groups = uf.groups()

    reports: list[DriftReport] = []
    for member_indices in groups.values():
        if len(member_indices) < 2:
            continue
        if len(member_indices) > MAX_CLUSTER_SIZE:
            member_indices = member_indices[:MAX_CLUSTER_SIZE]

        member_tags = frozenset(all_tags[i] for i in member_indices)

        jaccard_sum = 0.0
        jaccard_count = 0
        for i in member_indices:
            for j in member_indices:
                if i < j:
                    j_val = pair_jaccards.get((i, j), pair_jaccards.get((j, i), 0.0))
                    if j_val > 0:
                        jaccard_sum += j_val
                        jaccard_count += 1

        avg_jaccard = jaccard_sum / jaccard_count if jaccard_count > 0 else 0.0

        canonical = max(member_tags, key=lambda t: tag_fiber_counts.get(t, 0))

        if avg_jaccard >= JACCARD_MERGE_THRESHOLD:
            suggestion = "merge"
        elif avg_jaccard >= JACCARD_ALIAS_THRESHOLD:
            suggestion = "alias"
        else:
            suggestion = "review"

        others = sorted(member_tags - {canonical})
        evidence = (
            f"Tags {others} co-occur with '{canonical}' "
            f"(avg Jaccard={avg_jaccard:.2f}, "
            f"fibers: {', '.join(f'{t}={tag_fiber_counts.get(t, 0)}' for t in sorted(member_tags))})"
        )

        cluster = TagCluster(
            canonical=canonical,
            members=member_tags,
            confidence=round(avg_jaccard, 4),
            evidence=evidence,
        )

        reports.append(
            DriftReport(
                cluster=cluster,
                suggestion=suggestion,
                cluster_id=_cluster_id(member_tags),
            )
        )

    reports.sort(key=lambda r: r.cluster.confidence, reverse=True)
    return reports


async def refresh_drift_clusters(
    storage: NeuralStorage, *, persist: bool = True
) -> tuple[int, int]:
    """Recompute drift clusters from the current tag_cooccurrence table.

    Read-detect-persist, all fail-soft on the reads (an empty result degrades
    to "no clusters found" rather than raising) — this is a consolidation
    step, not something that should abort the whole pass over a transient
    storage hiccup.

    ``persist=False`` still reads and still clusters; it only skips the writes,
    so a dry run reports the number of clusters it WOULD have saved. Returning
    a flat 0 without looking would make "preview of a clean brain" and "preview
    of a brain with five clusters" indistinguishable — the same ambiguity this
    whole feature exists to remove. Mirrors ``ConsolidationEngine._dedup``,
    which likewise censuses unconditionally and gates only its writes.

    Returns ``(detected, persisted)``. The two are reported separately because a
    failing write used to be indistinguishable from a cluster that was never found:
    the count returned was the SAVED count while the report called it "found".
    """
    # WARNING, not debug: a failed read here makes the pass report zero clusters,
    # which is indistinguishable from an honest "no drift found" — the exact
    # ambiguity this feature was rebuilt to remove. A transient hiccup deserves
    # to degrade gracefully; it does not deserve to do so silently. (Proven
    # necessary: a malformed ORDER BY in get_tag_fiber_counts was swallowed here
    # as a bare 0 and only surfaced because a live test asserted a positive.)
    try:
        cooccurrences = await storage.get_tag_cooccurrence(min_count=MIN_COOCCURRENCE_COUNT)
    except Exception:
        logger.warning("Failed to read tag_cooccurrence for drift detection", exc_info=True)
        return (0, 0)

    try:
        tag_fiber_counts = await storage.get_tag_fiber_counts()
    except Exception:
        logger.warning("Failed to read tag fiber counts for drift detection", exc_info=True)
        return (0, 0)

    reports = detect_clusters(cooccurrences, tag_fiber_counts)
    detected = len(reports)

    if not persist:
        return (detected, detected)

    saved = 0
    for report in reports:
        try:
            await storage.save_drift_cluster(
                cluster_id=report.cluster_id,
                canonical=report.cluster.canonical,
                members=sorted(report.cluster.members),
                confidence=report.cluster.confidence,
                status="detected",
            )
            saved += 1
        except Exception:
            # WARNING, not debug: a swallowed write silently lowers the number the
            # report shows, so the operator sees "fewer clusters" instead of "a write
            # failed".
            logger.warning("Failed to persist drift cluster %s", report.cluster_id, exc_info=True)

    return (detected, saved)
