"""In-memory semantic drift detection persistence mixin.

Dict-based, 1:1 semantics with the SurrealDB mixin (``storage/surrealdb/drift.py``)
so the in-memory backend stays a faithful stand-in for tests — the same gap
that let drift detection ship SQLite-only for a whole release (see
``storage/memory_pinning.py``'s docstring for the pattern this repeats).
"""

from __future__ import annotations

import logging
from typing import Any

from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

_MAX_PAIRS_PER_CALL = 100

# Must stay equal to the SurrealDB mixin's cap. If this backend counted an
# unbounded number of fibers while production stopped at 10000, every test
# written against it would prove something untrue about production — which is
# the specific failure this file's docstring promises not to have.
_MAX_FIBER_SCAN = 10000


class InMemoryDriftMixin:
    """Mixin providing tag_cooccurrence and drift_clusters operations."""

    _tag_cooccurrence: dict[str, dict[tuple[str, str], dict[str, Any]]]
    _drift_clusters: dict[str, dict[str, dict[str, Any]]]
    _fibers: dict[str, dict[str, Any]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ========== Tag Co-occurrence ==========

    async def record_tag_cooccurrence(self, tags: set[str]) -> None:
        if len(tags) < 2:
            return

        brain_id = self._get_brain_id()
        store = self._tag_cooccurrence[brain_id]
        now = utcnow()

        sorted_tags = sorted(tags)
        pairs: list[tuple[str, str]] = []
        for i in range(len(sorted_tags)):
            for j in range(i + 1, len(sorted_tags)):
                pairs.append((sorted_tags[i], sorted_tags[j]))
        pairs = pairs[:_MAX_PAIRS_PER_CALL]

        for pair in pairs:
            existing = store.get(pair)
            count = (existing["pair_count"] if existing else 0) + 1
            store[pair] = {"pair_count": count, "last_seen": now}

    async def get_tag_cooccurrence(
        self, min_count: int = 2, limit: int = 500
    ) -> list[tuple[str, str, int]]:
        brain_id = self._get_brain_id()
        store = self._tag_cooccurrence[brain_id]
        capped = min(int(limit), 2000)

        rows = [
            (pair[0], pair[1], data["pair_count"])
            for pair, data in store.items()
            if data["pair_count"] >= min_count
        ]
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:capped]

    async def get_tag_fiber_counts(self) -> dict[str, int]:
        """Fiber count per tag, capped and logged exactly like the SurrealDB mixin.

        Insertion order is this backend's stable analogue of the SurrealDB
        mixin's ``ORDER BY id`` — the point in both is that the sample must not
        change between two passes over an unchanged brain.
        """
        brain_id = self._get_brain_id()
        fibers = list(self._fibers[brain_id].values())[:_MAX_FIBER_SCAN]
        if len(fibers) >= _MAX_FIBER_SCAN:
            logger.warning(
                "Drift detection scanned the %d-fiber cap on brain %r; tag counts are a "
                "truncated sample while co-occurrence counts are cumulative, so cluster "
                "confidences past this point are approximate.",
                _MAX_FIBER_SCAN,
                brain_id,
            )
        tag_counts: dict[str, int] = {}
        for fiber in fibers:
            all_tags = fiber.auto_tags | fiber.agent_tags
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    # ========== Drift Clusters ==========

    async def save_drift_cluster(
        self,
        cluster_id: str,
        canonical: str,
        members: list[str],
        confidence: float,
        status: str = "detected",
    ) -> None:
        brain_id = self._get_brain_id()
        store = self._drift_clusters[brain_id]
        existing = store.get(cluster_id)

        store[cluster_id] = {
            "id": cluster_id,
            "canonical": canonical,
            "members": list(members),
            "confidence": float(confidence),
            "status": status,
            "created_at": existing["created_at"] if existing else utcnow(),
            "resolved_at": None,
        }

    async def get_drift_clusters(
        self, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        capped = min(int(limit), 200)

        clusters = list(self._drift_clusters[brain_id].values())
        if status:
            clusters = [c for c in clusters if c["status"] == status]
        clusters.sort(key=lambda c: c["confidence"], reverse=True)
        return [dict(c) for c in clusters[:capped]]

    async def resolve_drift_cluster(self, cluster_id: str, status: str) -> bool:
        brain_id = self._get_brain_id()
        store = self._drift_clusters[brain_id]
        cluster = store.get(cluster_id)
        if cluster is None:
            return False
        cluster["status"] = status
        cluster["resolved_at"] = utcnow()
        return True
