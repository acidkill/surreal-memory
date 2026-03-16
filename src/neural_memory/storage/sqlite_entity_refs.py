"""SQLite mixin for entity reference tracking (lazy entity promotion)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteEntityRefsMixin:
    """Mixin providing CRUD for the entity_refs table.

    Tracks entity mentions before they are promoted to full neurons.
    Entities need N mentions (default 2) before becoming neurons.
    """

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_entity_ref(
        self, entity_text: str, fiber_id: str, created_at: datetime | None = None
    ) -> None:
        """Record an entity mention for a fiber."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        ts = (created_at or utcnow()).isoformat()

        await conn.execute(
            """INSERT OR IGNORE INTO entity_refs
               (brain_id, entity_text, fiber_id, created_at, promoted)
               VALUES (?, ?, ?, ?, 0)""",
            (brain_id, entity_text, fiber_id, ts),
        )
        await conn.commit()

    async def count_entity_refs(self, entity_text: str) -> int:
        """Count how many fibers mention this entity (unpromoted only)."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT COUNT(*) FROM entity_refs
               WHERE brain_id = ? AND entity_text = ? AND promoted = 0""",
            (brain_id, entity_text),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def get_entity_ref_fiber_ids(self, entity_text: str) -> list[str]:
        """Get fiber IDs that reference this entity (for retroactive linking)."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """SELECT fiber_id FROM entity_refs
               WHERE brain_id = ? AND entity_text = ? AND promoted = 0""",
            (brain_id, entity_text),
        )
        rows = await cursor.fetchall()
        return [str(r[0]) for r in rows]

    async def mark_entity_refs_promoted(self, entity_text: str) -> int:
        """Mark all refs for an entity as promoted. Returns count updated."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            """UPDATE entity_refs SET promoted = 1
               WHERE brain_id = ? AND entity_text = ? AND promoted = 0""",
            (brain_id, entity_text),
        )
        await conn.commit()
        return cursor.rowcount

    async def prune_old_entity_refs(self, max_age_days: int = 90) -> int:
        """Remove unpromoted entity refs older than max_age_days."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        cutoff = utcnow()
        cutoff_iso = cutoff.isoformat()

        # Calculate cutoff by comparing created_at
        cursor = await conn.execute(
            """DELETE FROM entity_refs
               WHERE brain_id = ? AND promoted = 0
               AND julianday(?) - julianday(created_at) > ?""",
            (brain_id, cutoff_iso, max_age_days),
        )
        await conn.commit()
        return cursor.rowcount
