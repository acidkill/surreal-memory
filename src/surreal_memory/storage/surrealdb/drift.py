"""SurrealDB mixin for semantic drift detection persistence.

Port of the SQLite-only ``SQLiteDriftMixin`` removed in 3524066d (the
SurrealDB-only migration never carried this feature forward, so every caller
probed for it with ``getattr`` and quietly returned nothing — see
``engine/uncertainty_report.py``). Two tables: ``tag_cooccurrence`` (encode-time
input, see the producer in ``engine/pipeline_steps.py``) and ``drift_clusters``
(consolidation output, see ``engine/drift_clusters.py``).

Record-id construction here follows the U3 lesson (BUG-004): a raw f-string
resource passed to ``conn.select/merge/delete`` is inlined verbatim into
SurrealQL text, and a part starting with a digit is parsed as a *number* —
hard-failing the instant it hits a non-digit character. Every id here is
either bound via ``RecordID(table, id)`` or via ``type::record('table', $var)``
with the id as a bound parameter — never spliced into query text directly.

``brain_id`` is a BOUND parameter in the reads below, not an inlined literal.
This repo documents a gotcha that SurrealDB only uses a ``brain_id`` index for
an inline literal — but that was *measured* not to apply to these query shapes:
``EXPLAIN`` on all four of them reports ``IndexScan`` against the composite
index either way. The gotcha reproduces where a second filter makes the planner
choose between two candidate indexes (``find_neurons``' extra ``type`` filter);
these queries have no such ambiguity. Binding is kept as the safer default.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

# Tag-pair generation is O(n^2); this caps it on a fiber with an unusually
# large tag set, mirroring the removed SQLite mixin's cap exactly.
_MAX_PAIRS_PER_CALL = 100

# Upper bound on the fiber scan behind Jaccard's denominator. Same value the
# removed SQLite mixin used; see get_tag_fiber_counts for what happens at it.
_MAX_FIBER_SCAN = 10000


def _pair_record_id(brain_id: str, tag_a: str, tag_b: str) -> str:
    """Stable, collision-free id for one (brain, tag_a, tag_b) pair.

    Hashing the pair (rather than folding the tag text through ``_to_surreal_id``
    and concatenating) avoids separator collisions: two distinct pairs whose
    tags differ only in where an underscore falls (e.g. ("a_b", "c") vs
    ("a", "b_c")) would otherwise fold to the same joined string.
    """
    digest = hashlib.sha256(f"{brain_id}\x00{tag_a}\x00{tag_b}".encode()).hexdigest()
    return digest[:32]


class SurrealDBDriftMixin:
    """Mixin providing CRUD for tag_cooccurrence and drift_clusters tables."""

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Tag co-occurrence
    # ------------------------------------------------------------------

    async def record_tag_cooccurrence(self, tags: set[str]) -> None:
        """Record tag co-occurrence pairs from a single fiber.

        For each canonical pair (a, b) with a < b, UPSERT the pair count and
        last-seen timestamp in one batched multi-statement round-trip —
        pattern mirrors ``SurrealDBKeywordEntityMixin.increment_keyword_df``,
        with each statement's bound parameter given its own real per-index
        name (``$sid0``, ``$sid1``, ...), not a shared literal placeholder.
        """
        if len(tags) < 2:
            return

        brain_id = self._get_brain_id()
        now = utcnow()

        sorted_tags = sorted(tags)
        pairs: list[tuple[str, str]] = []
        for i in range(len(sorted_tags)):
            for j in range(i + 1, len(sorted_tags)):
                pairs.append((sorted_tags[i], sorted_tags[j]))
        pairs = pairs[:_MAX_PAIRS_PER_CALL]

        params: dict[str, Any] = {"bid": brain_id, "now": now}
        stmts: list[str] = []
        for i, (tag_a, tag_b) in enumerate(pairs):
            sid_key = f"sid{i}"
            a_key = f"a{i}"
            b_key = f"b{i}"
            params[sid_key] = _pair_record_id(brain_id, tag_a, tag_b)
            params[a_key] = tag_a
            params[b_key] = tag_b
            stmts.append(
                f"UPSERT type::record('tag_cooccurrence', ${sid_key})"
                f" SET brain_id = $bid, tag_a = ${a_key}, tag_b = ${b_key},"
                f" pair_count = (pair_count ?? 0) + 1, last_seen = $now"
            )
        await self._query(";\n".join(stmts) + ";", **params)

    async def get_tag_cooccurrence(
        self,
        min_count: int = 2,
        limit: int = 500,
    ) -> list[tuple[str, str, int]]:
        """Get tag co-occurrence pairs above threshold, count descending."""
        brain_id = self._get_brain_id()
        capped_limit = min(int(limit), 2000)

        rows = await self._query(
            "SELECT tag_a, tag_b, pair_count FROM tag_cooccurrence"
            " WHERE brain_id = $bid AND pair_count >= $min_count"
            " ORDER BY pair_count DESC LIMIT $limit",
            bid=brain_id,
            min_count=int(min_count),
            limit=capped_limit,
        )
        return [(str(r["tag_a"]), str(r["tag_b"]), int(r["pair_count"])) for r in rows]

    async def get_tag_fiber_counts(self) -> dict[str, int]:
        """Get fiber count per tag for Jaccard's denominator.

        Scans up to ``_MAX_FIBER_SCAN`` fibers and counts in Python — no
        dedicated tag-fiber index exists (or is worth building for a
        consolidation-time-only read).

        The cap is inherited from the removed SQLite mixin, but two things are
        NOT inherited, because the original was silently wrong past it:

        * ``ORDER BY id`` makes the sample deterministic. Without it the slice
          was storage-order dependent, so two consecutive passes over an
          unchanged brain could read different counts and emit different
          confidences for the same tag pair.
        * hitting the cap is logged. Past it, Jaccard's numerator
          (``tag_cooccurrence.pair_count``, cumulative and uncapped) and its
          denominator (this truncated sample) describe different populations,
          so confidences skew in BOTH directions — an undercounted denominator
          inflates a score, while a tag missing from the sample entirely
          collapses it to 0.0 via ``compute_jaccard``'s zero-count guard. The
          caller cannot detect that from the returned dict, so it is surfaced
          here rather than left to be discovered on a large brain.
        """
        brain_id = self._get_brain_id()
        rows = await self._query(
            # `id` is projected only because SurrealDB requires every ORDER BY
            # idiom to appear in the selection ("Missing order idiom `id` in
            # statement selection" otherwise) — the value itself is unused.
            "SELECT id, auto_tags, agent_tags FROM fiber WHERE brain_id = $bid"
            " ORDER BY id LIMIT $limit",
            bid=brain_id,
            limit=_MAX_FIBER_SCAN,
        )
        if len(rows) >= _MAX_FIBER_SCAN:
            logger.warning(
                "Drift detection scanned the %d-fiber cap on brain %r; tag counts are a "
                "truncated sample while co-occurrence counts are cumulative, so cluster "
                "confidences past this point are approximate.",
                _MAX_FIBER_SCAN,
                brain_id,
            )
        tag_counts: dict[str, int] = {}
        for r in rows:
            auto = r.get("auto_tags") or []
            agent = r.get("agent_tags") or []
            all_tags = {str(t) for t in (*auto, *agent)}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts

    # ------------------------------------------------------------------
    # Drift clusters
    # ------------------------------------------------------------------

    async def save_drift_cluster(
        self,
        cluster_id: str,
        canonical: str,
        members: list[str],
        confidence: float,
        status: str = "detected",
    ) -> None:
        """Upsert a drift cluster detection result.

        Re-saving an existing cluster_id resets ``resolved_at`` to none — a
        cluster that reappears in a fresh detection pass is active again,
        regardless of a prior resolution.
        """
        brain_id = self._get_brain_id()
        sid = f"{brain_id}\x00{cluster_id}"
        record_id = hashlib.sha256(sid.encode()).hexdigest()[:32]

        await self._query(
            "UPSERT type::record('drift_clusters', $rid)"
            " SET brain_id = $bid, cluster_id = $cid, canonical = $canonical,"
            " members = $members, confidence = $confidence, status = $status,"
            " resolved_at = NONE",
            rid=record_id,
            bid=brain_id,
            cid=cluster_id,
            canonical=canonical,
            members=list(members),
            confidence=float(confidence),
            status=status,
        )

    async def get_drift_clusters(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get drift clusters, optionally filtered by status, confidence descending."""
        brain_id = self._get_brain_id()
        capped = min(int(limit), 200)

        if status:
            rows = await self._query(
                "SELECT cluster_id, canonical, members, confidence, status,"
                " created_at, resolved_at FROM drift_clusters"
                " WHERE brain_id = $bid AND status = $status"
                " ORDER BY confidence DESC LIMIT $limit",
                bid=brain_id,
                status=status,
                limit=capped,
            )
        else:
            rows = await self._query(
                "SELECT cluster_id, canonical, members, confidence, status,"
                " created_at, resolved_at FROM drift_clusters"
                " WHERE brain_id = $bid ORDER BY confidence DESC LIMIT $limit",
                bid=brain_id,
                limit=capped,
            )

        return [
            {
                "id": str(r.get("cluster_id", "")),
                "canonical": r.get("canonical"),
                "members": list(r.get("members") or []),
                "confidence": r.get("confidence"),
                "status": r.get("status"),
                "created_at": r.get("created_at"),
                "resolved_at": r.get("resolved_at"),
            }
            for r in rows
        ]

    async def resolve_drift_cluster(
        self,
        cluster_id: str,
        status: str,
    ) -> bool:
        """Update drift cluster status (merged/aliased/dismissed). True if a row changed."""
        brain_id = self._get_brain_id()
        now = utcnow()

        rows = await self._query(
            "UPDATE drift_clusters SET status = $status, resolved_at = $now"
            " WHERE brain_id = $bid AND cluster_id = $cid",
            bid=brain_id,
            cid=cluster_id,
            status=status,
            now=now,
        )
        return bool(rows)
