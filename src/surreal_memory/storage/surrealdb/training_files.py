"""SurrealDB mixin for document-training file tracking (dedup and resume).

``smem train`` hashes each file's content and records it here, so a second run
over the same corpus skips files it has already encoded. These four operations
existed only on ``SQLiteStorage``; ``DocTrainer`` probed for them with
``hasattr`` and, finding nothing on SurrealDB, re-encoded every file on every
run — silently duplicating the whole corpus each time. That also made the
migration guide's claim that training progress is "derived state [the] next
training run rebuilds" false on SurrealDB, where it never rebuilt at all.

The ``training_files`` table is purely additive, so it lives in ``SCHEMA_SQL``
rather than behind a ``SCHEMA_VERSION`` bump: ``ensure_schema`` is idempotent and
runs before ``apply_migrations`` on every ``initialize()``, so existing databases
pick the table up on their next start. Bumping the version would have been
actively harmful — ``MIGRATIONS`` keys off ``TARGET_VERSION``, so raising it to
10 would rewrite the ``(8, 9)`` entry to ``(8, 10)`` and strand every v8 database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class SurrealDBTrainingFilesMixin:
    """Mixin providing training file tracking CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_training_file_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Look up a training file record by content hash, or None if untrained."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM training_files WHERE brain_id = $brain_id "
            "AND file_hash = $file_hash LIMIT 1",
            brain_id=brain_id,
            file_hash=file_hash,
        )
        if not rows:
            return None
        return _row_to_record(rows[0])

    async def upsert_training_file(
        self,
        *,
        file_hash: str,
        file_path: str,
        file_size: int,
        chunks_total: int = 0,
        chunks_completed: int = 0,
        status: str = "pending",
        domain_tag: str = "",
    ) -> str:
        """Create or update a training file record. Returns the record ID."""
        brain_id = self._get_brain_id()
        # trained_at is stamped only on completion, matching SQLite: a record that
        # goes back to in_progress keeps its previous completion timestamp rather
        # than claiming it was never finished.
        trained_at = utcnow() if status == "completed" else None

        existing = await self.get_training_file_by_hash(file_hash)
        if existing:
            record_id: str = existing["id"]
            await self._query(
                "UPDATE type::record('training_files', $rid) SET "
                "chunks_total = $chunks_total, chunks_completed = $chunks_completed, "
                "status = $status, trained_at = $trained_at "
                "WHERE brain_id = $brain_id",
                rid=_to_surreal_id(record_id),
                chunks_total=chunks_total,
                chunks_completed=chunks_completed,
                status=status,
                trained_at=trained_at,
                brain_id=brain_id,
            )
            return record_id

        record_id = str(uuid4())
        await self._ensure_conn().insert(
            "training_files",
            {
                "id": _to_surreal_id(record_id),
                "brain_id": brain_id,
                "file_hash": file_hash,
                "file_path": file_path,
                "file_size": file_size,
                "chunks_total": chunks_total,
                "chunks_completed": chunks_completed,
                "status": status,
                "domain_tag": domain_tag,
                "trained_at": trained_at,
                "created_at": utcnow(),
            },
        )
        return record_id

    async def update_training_file_progress(
        self, record_id: str, chunks_completed: int, status: str = "in_progress"
    ) -> None:
        """Update chunk progress for a training file, for resume support."""
        brain_id = self._get_brain_id()
        # COALESCE semantics: only a completing update stamps trained_at, and a
        # later non-completing update must not clear it.
        if status == "completed":
            await self._query(
                "UPDATE type::record('training_files', $rid) SET "
                "chunks_completed = $chunks_completed, status = $status, "
                "trained_at = $trained_at WHERE brain_id = $brain_id",
                rid=_to_surreal_id(record_id),
                chunks_completed=chunks_completed,
                status=status,
                trained_at=utcnow(),
                brain_id=brain_id,
            )
            return

        await self._query(
            "UPDATE type::record('training_files', $rid) SET "
            "chunks_completed = $chunks_completed, status = $status "
            "WHERE brain_id = $brain_id",
            rid=_to_surreal_id(record_id),
            chunks_completed=chunks_completed,
            status=status,
            brain_id=brain_id,
        )

    async def get_training_stats(self) -> dict[str, Any]:
        """Training file counts for the current brain."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT status, count() AS c, math::sum(chunks_completed) AS chunks "
            "FROM training_files WHERE brain_id = $brain_id GROUP BY status",
            brain_id=brain_id,
        )

        stats = {
            "total_files": 0,
            "completed": 0,
            "in_progress": 0,
            "failed": 0,
            "total_chunks": 0,
        }
        for row in rows:
            count = int(row.get("c") or 0)
            stats["total_files"] += count
            stats["total_chunks"] += int(row.get("chunks") or 0)
            status = str(row.get("status") or "")
            if status in stats:
                stats[status] += count
        return stats


def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a SurrealDB row into the plain dict callers expect.

    ``DocTrainer`` reads ``record["status"]`` and the MCP layer serialises the
    whole dict, so the record id is flattened to a bare string and datetimes to
    ISO strings — matching what ``aiosqlite.Row`` produced.
    """
    rid = row.get("id")
    if rid is not None:
        text = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
        row["id"] = text.split(":", 1)[1] if ":" in text else text
    for key in ("created_at", "trained_at"):
        val = row.get(key)
        if isinstance(val, datetime):
            row[key] = val.isoformat()
    return row
