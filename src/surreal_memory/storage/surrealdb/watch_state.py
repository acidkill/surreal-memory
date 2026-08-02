"""SurrealDB mixin for `smem watch` file-ingestion state (issue #138).

``WatchStateTracker`` used to reach into ``SQLiteStorage``'s private
``aiosqlite`` connection directly (``storage._db``). No other backend ever had
that attribute, so every entry point — the MCP ``smem_watch`` tool, the CLI,
and the server's background watcher — raised ``AttributeError`` on SurrealDB.
These five operations existed only on the SQLite-backed tracker; moving them
behind the storage interface (mirroring ``training_files.py``) is the fix.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from surreal_memory.core.watch_record import WatchedFile
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class SurrealDBWatchStateMixin:
    """Mixin providing `smem watch` file-ingestion state for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _get_watch_row(self, file_path: str) -> dict[str, Any] | None:
        """Look up a watch_state row by file_path for the current brain."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM watch_state WHERE brain_id = $brain_id "
            "AND file_path = $file_path LIMIT 1",
            brain_id=brain_id,
            file_path=file_path,
        )
        return rows[0] if rows else None

    async def watch_should_process(self, file_path: str, mtime: float, content_hash: int) -> bool:
        """True if file_path needs (re-)ingestion.

        False only when both mtime is unchanged AND content is a near-duplicate
        of what was last stored (mtime touched without a real content change).
        """
        row = await self._get_watch_row(file_path)
        if row is None:
            return True

        stored_mtime = float(row.get("mtime") or 0.0)
        if mtime <= stored_mtime:
            return False

        from surreal_memory.utils.simhash import is_near_duplicate

        stored_hash = int(row.get("simhash") or 0)
        if is_near_duplicate(content_hash, stored_hash):
            # Content is unchanged — record the new mtime so the next check
            # doesn't re-compare against a stale timestamp, but skip ingestion.
            brain_id = self._get_brain_id()
            await self._query(
                "UPDATE watch_state SET mtime = $mtime "
                "WHERE brain_id = $brain_id AND file_path = $file_path",
                mtime=mtime,
                brain_id=brain_id,
                file_path=file_path,
            )
            return False

        return True

    async def watch_mark_processed(
        self, file_path: str, mtime: float, content_hash: int, neuron_count: int
    ) -> None:
        """Record that file_path was successfully ingested."""
        brain_id = self._get_brain_id()
        now = utcnow()
        existing = await self._get_watch_row(file_path)

        if existing is not None:
            await self._query(
                "UPDATE watch_state SET mtime = $mtime, simhash = $simhash, "
                "neuron_count = $neuron_count, last_ingested = $last_ingested, "
                "status = 'active' WHERE brain_id = $brain_id AND file_path = $file_path",
                mtime=mtime,
                simhash=content_hash,
                neuron_count=neuron_count,
                last_ingested=now,
                brain_id=brain_id,
                file_path=file_path,
            )
            return

        record_id = uuid4().hex
        await self._ensure_conn().insert(
            "watch_state",
            {
                "id": _to_surreal_id(record_id),
                "brain_id": brain_id,
                "file_path": file_path,
                "mtime": mtime,
                "simhash": content_hash,
                "neuron_count": neuron_count,
                "last_ingested": now,
                "status": "active",
            },
        )

    async def watch_mark_deleted(self, file_path: str) -> None:
        """Soft-delete a watched file's record."""
        brain_id = self._get_brain_id()
        await self._query(
            "UPDATE watch_state SET status = 'deleted' "
            "WHERE brain_id = $brain_id AND file_path = $file_path",
            brain_id=brain_id,
            file_path=file_path,
        )

    async def watch_list_files(self, status: str | None = None) -> list[WatchedFile]:
        """List tracked files, optionally filtered by status."""
        brain_id = self._get_brain_id()
        if status:
            rows = await self._query(
                "SELECT * FROM watch_state WHERE brain_id = $brain_id AND status = $status "
                "ORDER BY last_ingested DESC",
                brain_id=brain_id,
                status=status,
            )
        else:
            rows = await self._query(
                "SELECT * FROM watch_state WHERE brain_id = $brain_id ORDER BY last_ingested DESC",
                brain_id=brain_id,
            )
        return [_row_to_watched_file(row) for row in rows]

    async def watch_get_stats(self) -> dict[str, Any]:
        """Aggregate watch-state stats for the current brain."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT status, count() AS c, math::sum(neuron_count) AS neurons "
            "FROM watch_state WHERE brain_id = $brain_id GROUP BY status",
            brain_id=brain_id,
        )

        stats: dict[str, Any] = {"total_files": 0, "total_neurons": 0, "by_status": {}}
        for row in rows:
            count = int(row.get("c") or 0)
            neurons = int(row.get("neurons") or 0)
            status_val = str(row.get("status") or "")
            stats["total_files"] += count
            stats["total_neurons"] += neurons
            stats["by_status"][status_val] = {"files": count, "neurons": neurons}
        return stats


def _row_to_watched_file(row: dict[str, Any]) -> WatchedFile:
    """Normalise a SurrealDB watch_state row into a plain WatchedFile."""
    last_ingested = row.get("last_ingested")
    if isinstance(last_ingested, datetime):
        last_ingested = last_ingested.isoformat()
    return WatchedFile(
        file_path=str(row.get("file_path") or ""),
        mtime=float(row.get("mtime") or 0.0),
        simhash=int(row.get("simhash") or 0),
        neuron_count=int(row.get("neuron_count") or 0),
        last_ingested=str(last_ingested or ""),
        status=str(row.get("status") or "active"),
    )
