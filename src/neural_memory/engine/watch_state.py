"""Watch state tracker — tracks file ingestion state in SQLite.

Stores file path, mtime, simhash, and neuron count to determine
whether a file needs re-ingestion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS watch_state (
    file_path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    simhash INTEGER NOT NULL DEFAULT 0,
    neuron_count INTEGER NOT NULL DEFAULT 0,
    last_ingested TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
)
"""


@dataclass(frozen=True)
class WatchedFile:
    """State of a watched file."""

    file_path: str
    mtime: float
    simhash: int
    neuron_count: int
    last_ingested: str
    status: str = "active"


class WatchStateTracker:
    """Tracks file ingestion state in SQLite."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._initialized = False

    async def initialize(self) -> None:
        """Create watch_state table if not exists."""
        if self._initialized:
            return
        await self._db.execute(CREATE_TABLE_SQL)
        await self._db.commit()
        self._initialized = True

    async def should_process(self, file_path: Path) -> bool:
        """Check if a file needs (re-)processing.

        Returns True if:
        - File is not tracked yet
        - File mtime has changed since last ingestion
        """
        await self.initialize()
        resolved = str(file_path.resolve())

        cursor = await self._db.execute(
            "SELECT mtime FROM watch_state WHERE file_path = ?",
            (resolved,),
        )
        row = await cursor.fetchone()
        if row is None:
            return True

        current_mtime = file_path.stat().st_mtime
        return bool(current_mtime > row[0])

    async def should_process_with_simhash(
        self,
        file_path: Path,
        content_hash: int,
    ) -> bool:
        """Check if file needs processing using both mtime AND simhash.

        Returns False (skip) only if both mtime unchanged AND simhash matches.
        """
        await self.initialize()
        resolved = str(file_path.resolve())

        cursor = await self._db.execute(
            "SELECT mtime, simhash FROM watch_state WHERE file_path = ?",
            (resolved,),
        )
        row = await cursor.fetchone()
        if row is None:
            return True

        current_mtime = file_path.stat().st_mtime
        stored_mtime, stored_hash = row[0], row[1]

        # If mtime hasn't changed, skip
        if current_mtime <= stored_mtime:
            return False

        # Mtime changed but check simhash — content might be same
        from neural_memory.utils.simhash import is_near_duplicate

        if is_near_duplicate(content_hash, stored_hash):
            # Update mtime but skip processing
            await self._db.execute(
                "UPDATE watch_state SET mtime = ? WHERE file_path = ?",
                (current_mtime, resolved),
            )
            await self._db.commit()
            return False

        return True

    async def mark_processed(
        self,
        file_path: Path,
        mtime: float,
        content_hash: int,
        neuron_count: int,
    ) -> None:
        """Record that a file was successfully processed."""
        await self.initialize()
        resolved = str(file_path.resolve())
        now = utcnow().isoformat()

        await self._db.execute(
            """INSERT INTO watch_state (file_path, mtime, simhash, neuron_count, last_ingested, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            ON CONFLICT(file_path) DO UPDATE SET
                mtime = excluded.mtime,
                simhash = excluded.simhash,
                neuron_count = excluded.neuron_count,
                last_ingested = excluded.last_ingested,
                status = 'active'
            """,
            (resolved, mtime, content_hash, neuron_count, now),
        )
        await self._db.commit()

    async def mark_deleted(self, file_path: Path) -> None:
        """Mark a file as deleted (soft delete)."""
        await self.initialize()
        resolved = str(file_path.resolve())
        await self._db.execute(
            "UPDATE watch_state SET status = 'deleted' WHERE file_path = ?",
            (resolved,),
        )
        await self._db.commit()

    async def list_watched_files(
        self,
        *,
        status: str | None = None,
    ) -> list[WatchedFile]:
        """List all tracked files."""
        await self.initialize()

        if status:
            cursor = await self._db.execute(
                "SELECT file_path, mtime, simhash, neuron_count, last_ingested, status "
                "FROM watch_state WHERE status = ? ORDER BY last_ingested DESC",
                (status,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT file_path, mtime, simhash, neuron_count, last_ingested, status "
                "FROM watch_state ORDER BY last_ingested DESC",
            )

        rows = await cursor.fetchall()
        return [
            WatchedFile(
                file_path=row[0],
                mtime=row[1],
                simhash=row[2],
                neuron_count=row[3],
                last_ingested=row[4],
                status=row[5],
            )
            for row in rows
        ]

    async def get_stats(self) -> dict[str, Any]:
        """Get watch state statistics."""
        await self.initialize()

        cursor = await self._db.execute(
            "SELECT status, COUNT(*), SUM(neuron_count) FROM watch_state GROUP BY status",
        )
        rows = await cursor.fetchall()

        stats: dict[str, Any] = {"total_files": 0, "total_neurons": 0, "by_status": {}}
        for status_val, count, neurons in rows:
            stats["total_files"] += count
            stats["total_neurons"] += neurons or 0
            stats["by_status"][status_val] = {"files": count, "neurons": neurons or 0}

        return stats
