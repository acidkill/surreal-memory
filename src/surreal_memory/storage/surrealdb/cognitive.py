"""SurrealDB cognitive-state storage mixin.

Covers cognitive_state (with predictions), hot_index, knowledge_gaps,
and the schema-history walk.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

_MAX_HOT_SLOTS = 20
_MAX_LIST_LIMIT = 200
_TOPIC_MAX_LEN = 500
_SUMMARY_MAX_LEN = 500


def _parse_datetime(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo is not None else val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except (ValueError, AttributeError):
            return None
    return None


def _clamp_confidence(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _clamp_priority(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _row_to_cognitive_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Project a SurrealDB cognitive_state record to the SQLite return shape."""
    return {
        "neuron_id": row.get("neuron_id"),
        "confidence": float(row.get("confidence", 0.5)),
        "evidence_for_count": int(row.get("evidence_for_count", 0)),
        "evidence_against_count": int(row.get("evidence_against_count", 0)),
        "status": str(row.get("status", "active")),
        "predicted_at": _parse_datetime(row.get("predicted_at")),
        "resolved_at": _parse_datetime(row.get("resolved_at")),
        "last_evidence_at": _parse_datetime(row.get("last_evidence_at")),
        "schema_version": int(row.get("schema_version", 1)),
        "parent_schema_id": row.get("parent_schema_id"),
        "created_at": _parse_datetime(row.get("created_at")),
    }


class SurrealDBCognitiveMixin:
    """Mixin providing cognitive state, predictions, hot index, and knowledge-gap CRUD."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ---------------- cognitive_state CRUD ----------------

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
        """Insert or update a cognitive state record (composite key: brain_id+neuron_id)."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        sid = f"{_to_surreal_id(brain_id)}_{_to_surreal_id(neuron_id)}"

        record_data: dict[str, Any] = {
            "brain_id": brain_id,
            "neuron_id": neuron_id,
            "confidence": _clamp_confidence(confidence),
            "evidence_for_count": int(evidence_for_count),
            "evidence_against_count": int(evidence_against_count),
            "status": status,
            "predicted_at": _parse_datetime(predicted_at),
            "resolved_at": _parse_datetime(resolved_at),
            "schema_version": int(schema_version),
            "parent_schema_id": parent_schema_id,
            "last_evidence_at": _parse_datetime(last_evidence_at),
        }

        existing = await self._query(
            "SELECT id FROM cognitive_state"
            " WHERE brain_id = $brain_id AND neuron_id = $neuron_id LIMIT 1",
            brain_id=brain_id,
            neuron_id=neuron_id,
        )
        if existing:
            # Merge by existing record id (not recomputed sid) — survives a brain
            # rename where the id keeps the old brain prefix (else silent no-op).
            await conn.merge(existing[0]["id"], record_data)
        else:
            insert_data = dict(record_data)
            insert_data["id"] = sid
            insert_data["created_at"] = utcnow()
            await conn.insert("cognitive_state", insert_data)

    async def get_cognitive_state(self, neuron_id: str) -> dict[str, Any] | None:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM cognitive_state"
            " WHERE brain_id = $brain_id AND neuron_id = $neuron_id LIMIT 1",
            brain_id=brain_id,
            neuron_id=neuron_id,
        )
        if not rows:
            return None
        return _row_to_cognitive_dict(rows[0])

    async def list_cognitive_states(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_LIST_LIMIT)

        if status is not None:
            rows = await self._query(
                "SELECT * FROM cognitive_state"
                " WHERE brain_id = $brain_id AND status = $status"
                " ORDER BY confidence DESC LIMIT $limit",
                brain_id=brain_id,
                status=status,
                limit=capped,
            )
        else:
            rows = await self._query(
                "SELECT * FROM cognitive_state"
                " WHERE brain_id = $brain_id"
                " ORDER BY confidence DESC LIMIT $limit",
                brain_id=brain_id,
                limit=capped,
            )

        return [_row_to_cognitive_dict(r) for r in rows]

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
        """Update only evidence fields, preserving predicted_at/schema_version/parent/created_at."""
        brain_id = self._get_brain_id()
        existing = await self._query(
            "SELECT id FROM cognitive_state"
            " WHERE brain_id = $brain_id AND neuron_id = $neuron_id LIMIT 1",
            brain_id=brain_id,
            neuron_id=neuron_id,
        )
        if not existing:
            return

        conn = self._ensure_conn()
        # Merge by the id the SELECT above returned, for the same reason the
        # upsert path does: a recomputed sid misses a row whose id still carries
        # an older brain prefix, and the merge then silently writes nothing.
        await conn.merge(
            existing[0]["id"],
            {
                "confidence": _clamp_confidence(confidence),
                "evidence_for_count": int(evidence_for_count),
                "evidence_against_count": int(evidence_against_count),
                "status": status,
                "resolved_at": _parse_datetime(resolved_at),
                "last_evidence_at": _parse_datetime(last_evidence_at),
            },
        )

    # ---------------- predictions ----------------

    async def list_predictions(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Predictions = cognitive_state rows where predicted_at IS NOT NULL."""
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_LIST_LIMIT)

        if status is not None:
            rows = await self._query(
                "SELECT * FROM cognitive_state"
                " WHERE brain_id = $brain_id"
                " AND predicted_at IS NOT NONE AND status = $status"
                " ORDER BY predicted_at ASC LIMIT $limit",
                brain_id=brain_id,
                status=status,
                limit=capped,
            )
        else:
            rows = await self._query(
                "SELECT * FROM cognitive_state"
                " WHERE brain_id = $brain_id AND predicted_at IS NOT NONE"
                " ORDER BY predicted_at ASC LIMIT $limit",
                brain_id=brain_id,
                limit=capped,
            )
        return [_row_to_cognitive_dict(r) for r in rows]

    async def get_calibration_stats(self) -> dict[str, int]:
        """Tally prediction outcomes from cognitive_state.

        Returns: correct_count, wrong_count, total_resolved, pending_count.
        """
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT status, count() AS n FROM cognitive_state"
            " WHERE brain_id = $brain_id AND predicted_at IS NOT NONE"
            " GROUP BY status",
            brain_id=brain_id,
        )

        counts = {"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
        for r in rows:
            n = int(r.get("n", 0))
            st = str(r.get("status", ""))
            if st == "confirmed":
                counts["correct_count"] += n
                counts["total_resolved"] += n
            elif st == "refuted":
                counts["wrong_count"] += n
                counts["total_resolved"] += n
            elif st == "pending":
                counts["pending_count"] += n
        return counts

    # ---------------- hot_index ----------------

    async def refresh_hot_index(self, items: list[dict[str, Any]]) -> int:
        """Atomically replace this brain's hot index slots with fresh top-N entries."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()

        await conn.query(
            "DELETE hot_index WHERE brain_id = $brain_id",
            {"brain_id": brain_id},
        )

        now = utcnow()
        count = 0
        bid_safe = _to_surreal_id(brain_id)

        for item in items[:_MAX_HOT_SLOTS]:
            slot = int(item["slot"])
            sid = f"{bid_safe}_{slot:03d}"
            summary = str(item.get("summary", ""))[:_SUMMARY_MAX_LEN]
            await conn.insert(
                "hot_index",
                {
                    "id": sid,
                    "brain_id": brain_id,
                    "slot": slot,
                    "category": str(item.get("category", "")),
                    "neuron_id": str(item["neuron_id"]),
                    "summary": summary,
                    "confidence": item.get("confidence"),
                    "score": float(item.get("score", 0.0)),
                    "updated_at": now,
                },
            )
            count += 1
        return count

    async def get_hot_index(self, limit: int = 10) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_HOT_SLOTS)

        rows = await self._query(
            "SELECT slot, category, neuron_id, summary, confidence, score, updated_at"
            " FROM hot_index"
            " WHERE brain_id = $brain_id"
            " ORDER BY score DESC LIMIT $limit",
            brain_id=brain_id,
            limit=capped,
        )

        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "slot": int(r.get("slot", 0)),
                    "category": str(r.get("category", "")),
                    "neuron_id": str(r.get("neuron_id", "")),
                    "summary": str(r.get("summary", "")),
                    "confidence": (
                        float(r["confidence"]) if r.get("confidence") is not None else None
                    ),
                    "score": float(r.get("score", 0.0)),
                    "updated_at": _parse_datetime(r.get("updated_at")),
                }
            )
        return results

    # ---------------- knowledge_gaps ----------------

    async def add_knowledge_gap(
        self,
        *,
        topic: str,
        detection_source: str,
        priority: float = 0.5,
        related_neuron_ids: list[str] | None = None,
    ) -> str:
        """Create a new knowledge gap, returning its generated ID."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()

        gap_id = str(uuid4())
        sid = _to_surreal_id(gap_id)

        await conn.insert(
            "knowledge_gaps",
            {
                "id": sid,
                "brain_id": brain_id,
                "topic": topic[:_TOPIC_MAX_LEN],
                "detected_at": utcnow(),
                "detection_source": detection_source,
                "related_neuron_ids": list(related_neuron_ids or []),
                "priority": _clamp_priority(priority),
            },
        )
        return gap_id

    async def list_knowledge_gaps(
        self,
        *,
        include_resolved: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_LIST_LIMIT)

        if include_resolved:
            rows = await self._query(
                "SELECT id, topic, detected_at, detection_source,"
                " related_neuron_ids, resolved_at, resolved_by_neuron_id, priority"
                " FROM knowledge_gaps"
                " WHERE brain_id = $brain_id"
                " ORDER BY priority DESC LIMIT $limit",
                brain_id=brain_id,
                limit=capped,
            )
        else:
            rows = await self._query(
                "SELECT id, topic, detected_at, detection_source,"
                " related_neuron_ids, resolved_at, resolved_by_neuron_id, priority"
                " FROM knowledge_gaps"
                " WHERE brain_id = $brain_id AND resolved_at IS NONE"
                " ORDER BY priority DESC LIMIT $limit",
                brain_id=brain_id,
                limit=capped,
            )

        results: list[dict[str, Any]] = []
        for r in rows:
            raw_id = str(r.get("id", ""))
            gap_id = _record_id_part(raw_id)
            results.append(
                {
                    "id": gap_id,
                    "topic": str(r.get("topic", "")),
                    "detected_at": _parse_datetime(r.get("detected_at")),
                    "detection_source": str(r.get("detection_source", "")),
                    "related_neuron_ids": list(r.get("related_neuron_ids") or []),
                    "resolved_at": _parse_datetime(r.get("resolved_at")),
                    "resolved_by_neuron_id": r.get("resolved_by_neuron_id"),
                    "priority": float(r.get("priority", 0.5)),
                }
            )
        return results

    async def get_knowledge_gap(self, gap_id: str) -> dict[str, Any] | None:
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(gap_id)

        rows = await self._query(
            # Rebuild the record id in SurrealQL rather than comparing `id`
            # with a "knowledge_gaps:<sid>" *string*: `id` holds a record id,
            # so the string form is unconditionally false and this lookup
            # returned None for every gap that existed. Same trap as the
            # typed_memory / fiber / alerts lookups in this package.
            "SELECT id, topic, detected_at, detection_source,"
            " related_neuron_ids, resolved_at, resolved_by_neuron_id, priority"
            " FROM knowledge_gaps"
            " WHERE brain_id = $brain_id AND id = type::record('knowledge_gaps', $sid)"
            " LIMIT 1",
            brain_id=brain_id,
            sid=sid,
        )
        if not rows:
            return None

        r = rows[0]
        raw_id = str(r.get("id", ""))
        return {
            "id": _record_id_part(raw_id),
            "topic": str(r.get("topic", "")),
            "detected_at": _parse_datetime(r.get("detected_at")),
            "detection_source": str(r.get("detection_source", "")),
            "related_neuron_ids": list(r.get("related_neuron_ids") or []),
            "resolved_at": _parse_datetime(r.get("resolved_at")),
            "resolved_by_neuron_id": r.get("resolved_by_neuron_id"),
            "priority": float(r.get("priority", 0.5)),
        }

    async def resolve_knowledge_gap(
        self,
        gap_id: str,
        *,
        resolved_by_neuron_id: str | None = None,
    ) -> bool:
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(gap_id)

        existing = await self._query(
            "SELECT id FROM knowledge_gaps"
            " WHERE brain_id = $brain_id AND id = type::record('knowledge_gaps', $sid)"
            " AND resolved_at IS NONE LIMIT 1",
            brain_id=brain_id,
            sid=sid,
        )
        if not existing:
            return False

        conn = self._ensure_conn()
        await conn.merge(
            existing[0]["id"],
            {
                "resolved_at": utcnow(),
                "resolved_by_neuron_id": resolved_by_neuron_id,
            },
        )
        return True

    # ---------------- schema-history walk ----------------

    async def get_schema_history(
        self,
        neuron_id: str,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Walk the parent_schema_id chain newest-first, with cycle protection."""
        brain_id = self._get_brain_id()

        history: list[dict[str, Any]] = []
        current_id: str | None = neuron_id
        seen: set[str] = set()

        while current_id and len(history) < max_depth:
            if current_id in seen:
                break
            seen.add(current_id)

            rows = await self._query(
                "SELECT neuron_id, confidence, evidence_for_count, evidence_against_count,"
                " status, schema_version, parent_schema_id, created_at"
                " FROM cognitive_state"
                " WHERE brain_id = $brain_id AND neuron_id = $neuron_id LIMIT 1",
                brain_id=brain_id,
                neuron_id=current_id,
            )
            if not rows:
                break

            r = rows[0]
            parent_raw = r.get("parent_schema_id")
            entry = {
                "neuron_id": str(r.get("neuron_id", "")),
                "confidence": float(r.get("confidence", 0.5)),
                "evidence_for_count": int(r.get("evidence_for_count", 0)),
                "evidence_against_count": int(r.get("evidence_against_count", 0)),
                "status": str(r.get("status", "")),
                "schema_version": int(r.get("schema_version", 1)),
                "parent_schema_id": parent_raw,
                "created_at": _parse_datetime(r.get("created_at")),
            }
            history.append(entry)
            current_id = str(parent_raw) if parent_raw is not None else None

        return history
