"""SQLite mixin for cognitive layer state persistence.

Covers three tables:
- cognitive_state: hypothesis/prediction confidence tracking
- hot_index: ranked summary of active cognitive items
- knowledge_gaps: metacognition — what the brain doesn't know
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Cap cognitive entries per brain to prevent unbounded growth
_MAX_COGNITIVE_PER_BRAIN = 5_000

# Hot index slot limits
_MAX_HOT_SLOTS = 20


class SQLiteCognitiveMixin:
    """Mixin providing CRUD for the cognitive_state table."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def upsert_cognitive_state(
        self,
        neuron_id: str,
        *,
        confidence: float = 0.5,
        evidence_for_count: int = 0,
        evidence_against_count: int = 0,
        status: str = "active",
        predicted_at: str | None = None,
        resolved_at: str | None = None,
        schema_version: int = 1,
        parent_schema_id: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Insert or update a cognitive state record."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT INTO cognitive_state
               (brain_id, neuron_id, confidence, evidence_for_count,
                evidence_against_count, status, predicted_at, resolved_at,
                schema_version, parent_schema_id, last_evidence_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(brain_id, neuron_id) DO UPDATE SET
                 confidence = excluded.confidence,
                 evidence_for_count = excluded.evidence_for_count,
                 evidence_against_count = excluded.evidence_against_count,
                 status = excluded.status,
                 predicted_at = excluded.predicted_at,
                 resolved_at = excluded.resolved_at,
                 schema_version = excluded.schema_version,
                 parent_schema_id = excluded.parent_schema_id,
                 last_evidence_at = excluded.last_evidence_at""",
            (
                brain_id,
                neuron_id,
                max(0.01, min(0.99, confidence)),
                evidence_for_count,
                evidence_against_count,
                status,
                predicted_at,
                resolved_at,
                schema_version,
                parent_schema_id,
                last_evidence_at,
                utcnow().isoformat(),
            ),
        )
        await conn.commit()

    async def get_cognitive_state(self, neuron_id: str) -> dict[str, Any] | None:
        """Get cognitive state for a neuron."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                      status, predicted_at, resolved_at, schema_version,
                      parent_schema_id, last_evidence_at, created_at
               FROM cognitive_state
               WHERE brain_id = ? AND neuron_id = ?""",
            (brain_id, neuron_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        col_names = [d[0] for d in (cursor.description or [])]
        return dict(zip(col_names, row, strict=False))

    async def list_cognitive_states(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List cognitive states, optionally filtered by status."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if status:
            cursor = await conn.execute(
                """SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = ? AND status = ?
                   ORDER BY confidence DESC LIMIT ?""",
                (brain_id, status, capped_limit),
            )
        else:
            cursor = await conn.execute(
                """SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = ?
                   ORDER BY confidence DESC LIMIT ?""",
                (brain_id, capped_limit),
            )

        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [dict(zip(col_names, r, strict=False)) for r in rows]

    async def update_cognitive_evidence(
        self,
        neuron_id: str,
        *,
        confidence: float,
        evidence_for_count: int,
        evidence_against_count: int,
        status: str,
        resolved_at: str | None = None,
        last_evidence_at: str | None = None,
    ) -> None:
        """Update only evidence-related fields of a cognitive state.

        Unlike upsert_cognitive_state, this preserves predicted_at,
        schema_version, parent_schema_id, and created_at unchanged.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """UPDATE cognitive_state SET
                 confidence = ?,
                 evidence_for_count = ?,
                 evidence_against_count = ?,
                 status = ?,
                 resolved_at = ?,
                 last_evidence_at = ?
               WHERE brain_id = ? AND neuron_id = ?""",
            (
                max(0.01, min(0.99, confidence)),
                evidence_for_count,
                evidence_against_count,
                status,
                resolved_at,
                last_evidence_at,
                brain_id,
                neuron_id,
            ),
        )
        await conn.commit()

    async def list_predictions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List predictions (cognitive states with predicted_at set)."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if status:
            cursor = await conn.execute(
                """SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = ? AND predicted_at IS NOT NULL AND status = ?
                   ORDER BY predicted_at ASC LIMIT ?""",
                (brain_id, status, capped_limit),
            )
        else:
            cursor = await conn.execute(
                """SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, predicted_at, resolved_at, last_evidence_at, created_at
                   FROM cognitive_state
                   WHERE brain_id = ? AND predicted_at IS NOT NULL
                   ORDER BY predicted_at ASC LIMIT ?""",
                (brain_id, capped_limit),
            )

        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [dict(zip(col_names, r, strict=False)) for r in rows]

    async def get_calibration_stats(self) -> dict[str, int]:
        """Get prediction calibration statistics.

        Returns:
            Dict with correct_count, wrong_count, total_resolved, pending_count.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT
                 SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) AS correct,
                 SUM(CASE WHEN status = 'refuted' THEN 1 ELSE 0 END) AS wrong,
                 SUM(CASE WHEN status IN ('confirmed', 'refuted') THEN 1 ELSE 0 END) AS resolved,
                 SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending
               FROM cognitive_state
               WHERE brain_id = ? AND predicted_at IS NOT NULL""",
            (brain_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
        return {
            "correct_count": row[0] or 0,
            "wrong_count": row[1] or 0,
            "total_resolved": row[2] or 0,
            "pending_count": row[3] or 0,
        }

    # ──────────────────── Hot Index ────────────────────

    async def refresh_hot_index(
        self,
        items: list[dict[str, Any]],
    ) -> int:
        """Replace the hot index with freshly scored items.

        Args:
            items: List of dicts with keys: slot, category, neuron_id,
                   summary, confidence, score.

        Returns:
            Number of items written.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        # Clear existing index for this brain
        await conn.execute("DELETE FROM hot_index WHERE brain_id = ?", (brain_id,))

        count = 0
        for item in items[:_MAX_HOT_SLOTS]:
            await conn.execute(
                """INSERT INTO hot_index
                   (brain_id, slot, category, neuron_id, summary, confidence, score, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    brain_id,
                    item["slot"],
                    item["category"],
                    item["neuron_id"],
                    item["summary"][:500],
                    item.get("confidence"),
                    item["score"],
                    now,
                ),
            )
            count += 1
        await conn.commit()
        return count

    async def get_hot_index(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the current hot index items, sorted by score descending."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_HOT_SLOTS)

        cursor = await conn.execute(
            """SELECT slot, category, neuron_id, summary, confidence, score, updated_at
               FROM hot_index
               WHERE brain_id = ?
               ORDER BY score DESC LIMIT ?""",
            (brain_id, capped),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [dict(zip(col_names, r, strict=False)) for r in rows]

    # ──────────────────── Knowledge Gaps ────────────────────

    async def add_knowledge_gap(
        self,
        *,
        topic: str,
        detection_source: str,
        priority: float = 0.5,
        related_neuron_ids: list[str] | None = None,
    ) -> str:
        """Create a new knowledge gap record.

        Returns:
            The generated gap ID.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        gap_id = str(uuid4())

        await conn.execute(
            """INSERT INTO knowledge_gaps
               (id, brain_id, topic, detected_at, detection_source,
                related_neuron_ids, priority)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                gap_id,
                brain_id,
                topic[:500],
                utcnow().isoformat(),
                detection_source,
                json.dumps(related_neuron_ids or []),
                max(0.0, min(1.0, priority)),
            ),
        )
        await conn.commit()
        return gap_id

    async def list_knowledge_gaps(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List knowledge gaps sorted by priority descending."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped = min(limit, 200)

        if include_resolved:
            cursor = await conn.execute(
                """SELECT id, topic, detected_at, detection_source,
                          related_neuron_ids, resolved_at, resolved_by_neuron_id, priority
                   FROM knowledge_gaps
                   WHERE brain_id = ?
                   ORDER BY priority DESC LIMIT ?""",
                (brain_id, capped),
            )
        else:
            cursor = await conn.execute(
                """SELECT id, topic, detected_at, detection_source,
                          related_neuron_ids, resolved_at, resolved_by_neuron_id, priority
                   FROM knowledge_gaps
                   WHERE brain_id = ? AND resolved_at IS NULL
                   ORDER BY priority DESC LIMIT ?""",
                (brain_id, capped),
            )

        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        results = []
        for r in rows:
            d = dict(zip(col_names, r, strict=False))
            # Parse JSON array
            try:
                d["related_neuron_ids"] = json.loads(d.get("related_neuron_ids", "[]"))
            except (json.JSONDecodeError, TypeError):
                d["related_neuron_ids"] = []
            results.append(d)
        return results

    async def get_knowledge_gap(self, gap_id: str) -> dict[str, Any] | None:
        """Get a single knowledge gap by ID."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT id, topic, detected_at, detection_source,
                      related_neuron_ids, resolved_at, resolved_by_neuron_id, priority
               FROM knowledge_gaps
               WHERE brain_id = ? AND id = ?""",
            (brain_id, gap_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        col_names = [d[0] for d in (cursor.description or [])]
        d = dict(zip(col_names, row, strict=False))
        try:
            d["related_neuron_ids"] = json.loads(d.get("related_neuron_ids", "[]"))
        except (json.JSONDecodeError, TypeError):
            d["related_neuron_ids"] = []
        return d

    async def resolve_knowledge_gap(
        self,
        gap_id: str,
        *,
        resolved_by_neuron_id: str | None = None,
    ) -> bool:
        """Mark a knowledge gap as resolved.

        Returns:
            True if the gap was found and resolved, False otherwise.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE knowledge_gaps SET
                 resolved_at = ?,
                 resolved_by_neuron_id = ?
               WHERE brain_id = ? AND id = ? AND resolved_at IS NULL""",
            (utcnow().isoformat(), resolved_by_neuron_id, brain_id, gap_id),
        )
        await conn.commit()
        return (cursor.rowcount or 0) > 0

    # ──────────────────── Schema Evolution ────────────────────

    async def get_schema_history(
        self,
        neuron_id: str,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Walk the version chain from a hypothesis back through parent_schema_id.

        Returns a list ordered newest-first, with the given neuron_id at index 0.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        history: list[dict[str, Any]] = []
        current_id: str | None = neuron_id
        seen: set[str] = set()

        while current_id and len(history) < max_depth:
            if current_id in seen:
                break  # Cycle guard
            seen.add(current_id)

            cursor = await conn.execute(
                """SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,
                          status, schema_version, parent_schema_id, created_at
                   FROM cognitive_state
                   WHERE brain_id = ? AND neuron_id = ?""",
                (brain_id, current_id),
            )
            row = await cursor.fetchone()
            if not row:
                break
            col_names = [d[0] for d in (cursor.description or [])]
            entry = dict(zip(col_names, row, strict=False))
            history.append(entry)
            current_id = entry.get("parent_schema_id")

        return history
