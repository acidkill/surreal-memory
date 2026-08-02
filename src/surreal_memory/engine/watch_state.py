"""Watch state tracker — tracks file ingestion state via the storage backend.

Stores file path, mtime, simhash, and neuron count to determine whether a
file needs re-ingestion.

Used to reach into a private SQLiteStorage attribute (``storage._db``, a raw
``aiosqlite.Connection``) directly. No other backend ever had that attribute,
so every entry point raised ``AttributeError`` on SurrealDB (issue #138).
This is now a thin facade over the storage interface's ``watch_*`` methods
(``NeuralStorage.watch_should_process`` etc., implemented per backend) —
callers keep the exact same method names/signatures as before; only the
constructor argument changed from a raw connection to a ``NeuralStorage``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.core.watch_record import WatchedFile

if TYPE_CHECKING:
    from pathlib import Path

    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

__all__ = ["WatchedFile", "WatchStateTracker"]


class WatchStateTracker:
    """Tracks file ingestion state via the active storage backend."""

    def __init__(self, storage: NeuralStorage) -> None:
        self._storage = storage

    async def initialize(self) -> None:
        """No-op: schema/state setup is handled by storage.initialize()."""
        return None

    async def should_process_with_simhash(
        self,
        file_path: Path,
        content_hash: int,
    ) -> bool:
        """Check if file needs processing using both mtime AND simhash.

        Returns False (skip) only if both mtime unchanged AND simhash matches.
        """
        resolved = str(file_path.resolve())
        mtime = file_path.stat().st_mtime
        return await self._storage.watch_should_process(resolved, mtime, content_hash)

    async def mark_processed(
        self,
        file_path: Path,
        mtime: float,
        content_hash: int,
        neuron_count: int,
    ) -> None:
        """Record that a file was successfully processed."""
        resolved = str(file_path.resolve())
        await self._storage.watch_mark_processed(resolved, mtime, content_hash, neuron_count)

    async def mark_deleted(self, file_path: Path) -> None:
        """Mark a file as deleted (soft delete)."""
        resolved = str(file_path.resolve())
        await self._storage.watch_mark_deleted(resolved)

    async def list_watched_files(
        self,
        *,
        status: str | None = None,
    ) -> list[WatchedFile]:
        """List all tracked files."""
        return await self._storage.watch_list_files(status=status)

    async def get_stats(self) -> dict[str, Any]:
        """Get watch state statistics."""
        return await self._storage.watch_get_stats()
