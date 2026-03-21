"""SQLite change log operations mixin for multi-device sync."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangeEntry:
    """A single change log entry."""

    id: int  # Auto-incremented sequence number
    brain_id: str
    entity_type: str  # "neuron", "synapse", "fiber"
    entity_id: str
    operation: str  # "insert", "update", "delete"
    device_id: str
    changed_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    synced: bool = False


class SQLiteChangeLogMixin:
    """Mixin providing change log CRUD operations for incremental sync."""

    # ------------------------------------------------------------------
    # Protocol stubs — satisfied by SQLiteStorage at runtime.
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a change to the log. Returns the sequence number (id)."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        cursor = await conn.execute(
            """INSERT INTO change_log
               (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                brain_id,
                entity_type,
                entity_id,
                operation,
                device_id,
                now,
                json.dumps(payload or {}),
            ),
        )
        await conn.commit()
        return cursor.lastrowid or 0

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[ChangeEntry]:
        """Get changes after a given sequence number, ordered by id ASC."""
        safe_limit = min(limit, 10000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM change_log WHERE brain_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            (brain_id, sequence, safe_limit),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_change_entry(dict(zip(col_names, r, strict=False))) for r in rows]

    async def get_unsynced_changes(self, limit: int = 1000) -> list[ChangeEntry]:
        """Get all unsynced changes, ordered by id ASC."""
        safe_limit = min(limit, 10000)
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT * FROM change_log WHERE brain_id = ? AND synced = 0 ORDER BY id ASC LIMIT ?",
            (brain_id, safe_limit),
        )
        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [_row_to_change_entry(dict(zip(col_names, r, strict=False))) for r in rows]

    async def mark_synced(self, up_to_sequence: int) -> int:
        """Mark all changes up to a sequence number as synced. Returns count marked."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "UPDATE change_log SET synced = 1 WHERE brain_id = ? AND id <= ? AND synced = 0",
            (brain_id, up_to_sequence),
        )
        await conn.commit()
        return cursor.rowcount

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        """Delete synced changes older than N days. Returns count pruned."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        cutoff = (utcnow() - timedelta(days=older_than_days)).isoformat()

        cursor = await conn.execute(
            "DELETE FROM change_log WHERE brain_id = ? AND synced = 1 AND changed_at < ?",
            (brain_id, cutoff),
        )
        await conn.commit()
        return cursor.rowcount

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        """Seed the change log with all existing entities as 'insert' entries.

        This enables initial sync for brains created before sync was enabled.
        Existing change_log entries are preserved — only entities NOT already
        tracked in the log are added.

        Returns:
            Dict with counts: neurons, synapses, fibers seeded.
        """
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()
        counts: dict[str, int] = {"neurons": 0, "synapses": 0, "fibers": 0}

        # Seed neurons not already in change_log
        cursor = await conn.execute(
            """INSERT INTO change_log
               (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
               SELECT ?, 'neuron', n.id, 'insert', ?, ?, json_object(
                   'id', n.id,
                   'type', n.type,
                   'content', n.content,
                   'metadata', n.metadata,
                   'content_hash', n.content_hash,
                   'created_at', n.created_at
               ), 0
               FROM neurons n
               WHERE n.brain_id = ?
                 AND COALESCE(n.ephemeral, 0) = 0
                 AND NOT EXISTS (
                     SELECT 1 FROM change_log cl
                     WHERE cl.brain_id = ? AND cl.entity_type = 'neuron' AND cl.entity_id = n.id
                 )""",
            (brain_id, device_id, now, brain_id, brain_id),
        )
        counts["neurons"] = cursor.rowcount

        # Seed synapses not already in change_log
        cursor = await conn.execute(
            """INSERT INTO change_log
               (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
               SELECT ?, 'synapse', s.id, 'insert', ?, ?, json_object(
                   'id', s.id,
                   'source_id', s.source_id,
                   'target_id', s.target_id,
                   'type', s.type,
                   'weight', s.weight,
                   'direction', s.direction,
                   'metadata', s.metadata,
                   'reinforced_count', s.reinforced_count,
                   'last_activated', s.last_activated,
                   'created_at', s.created_at
               ), 0
               FROM synapses s
               WHERE s.brain_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM change_log cl
                     WHERE cl.brain_id = ? AND cl.entity_type = 'synapse' AND cl.entity_id = s.id
                 )""",
            (brain_id, device_id, now, brain_id, brain_id),
        )
        counts["synapses"] = cursor.rowcount

        # Seed fibers not already in change_log
        cursor = await conn.execute(
            """INSERT INTO change_log
               (brain_id, entity_type, entity_id, operation, device_id, changed_at, payload, synced)
               SELECT ?, 'fiber', f.id, 'insert', ?, ?, json_object(
                   'id', f.id,
                   'neuron_ids', f.neuron_ids,
                   'synapse_ids', f.synapse_ids,
                   'anchor_neuron_id', f.anchor_neuron_id,
                   'pathway', f.pathway,
                   'conductivity', f.conductivity,
                   'last_conducted', f.last_conducted,
                   'time_start', f.time_start,
                   'time_end', f.time_end,
                   'coherence', f.coherence,
                   'salience', f.salience,
                   'frequency', f.frequency,
                   'summary', f.summary,
                   'auto_tags', f.auto_tags,
                   'agent_tags', f.agent_tags,
                   'metadata', f.metadata,
                   'compression_tier', f.compression_tier,
                   'created_at', f.created_at
               ), 0
               FROM fibers f
               WHERE f.brain_id = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM change_log cl
                     WHERE cl.brain_id = ? AND cl.entity_type = 'fiber' AND cl.entity_id = f.id
                 )""",
            (brain_id, device_id, now, brain_id, brain_id),
        )
        counts["fibers"] = cursor.rowcount

        await conn.commit()

        total = counts["neurons"] + counts["synapses"] + counts["fibers"]
        logger.info(
            "Seeded change log for brain %s: %d neurons, %d synapses, %d fibers (%d total)",
            brain_id,
            counts["neurons"],
            counts["synapses"],
            counts["fibers"],
            total,
        )
        return counts

    async def get_change_log_stats(self) -> dict[str, Any]:
        """Get change log statistics."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        async with conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN synced = 0 THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN synced = 1 THEN 1 ELSE 0 END) as synced,
                MAX(id) as last_sequence
               FROM change_log WHERE brain_id = ?""",
            (brain_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return {"total": 0, "pending": 0, "synced": 0, "last_sequence": 0}
            col_names = [d[0] for d in (cursor.description or [])]
            data = dict(zip(col_names, row, strict=False))
        return {
            "total": data["total"] or 0,
            "pending": data["pending"] or 0,
            "synced": data["synced"] or 0,
            "last_sequence": data["last_sequence"] or 0,
        }


def _row_to_change_entry(row: dict[str, Any]) -> ChangeEntry:
    """Convert a database row dict to a ChangeEntry."""
    return ChangeEntry(
        id=int(row["id"]),
        brain_id=str(row["brain_id"]),
        entity_type=str(row["entity_type"]),
        entity_id=str(row["entity_id"]),
        operation=str(row["operation"]),
        device_id=str(row["device_id"] or ""),
        changed_at=datetime.fromisoformat(str(row["changed_at"])),
        payload=json.loads(str(row["payload"])) if row["payload"] else {},
        synced=bool(row["synced"]),
    )
