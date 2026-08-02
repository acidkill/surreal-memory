"""In-memory `smem watch` file-ingestion state mixin (issue #138).

Faithful stand-in for the SurrealDB implementation so InMemoryStorage stays a
usable test fixture for the watch feature, mirroring memory_pinning.py's
training-file section.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from surreal_memory.core.watch_record import WatchedFile
from surreal_memory.utils.timeutils import utcnow


class InMemoryWatchStateMixin:
    """Mixin providing `smem watch` file-ingestion state for InMemoryStorage."""

    _watch_files: dict[str, dict[str, dict[str, Any]]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _get_watch_record(self, file_path: str) -> dict[str, Any] | None:
        brain_id = self._get_brain_id()
        for record in self._watch_files[brain_id].values():
            if record["file_path"] == file_path:
                return record
        return None

    async def watch_should_process(self, file_path: str, mtime: float, content_hash: int) -> bool:
        """True if file_path needs (re-)ingestion."""
        record = await self._get_watch_record(file_path)
        if record is None:
            return True

        stored_mtime = record["mtime"]
        if mtime <= stored_mtime:
            return False

        from surreal_memory.utils.simhash import is_near_duplicate

        if is_near_duplicate(content_hash, record["simhash"]):
            record["mtime"] = mtime
            return False

        return True

    async def watch_mark_processed(
        self, file_path: str, mtime: float, content_hash: int, neuron_count: int
    ) -> None:
        """Record that file_path was successfully ingested."""
        brain_id = self._get_brain_id()
        store = self._watch_files[brain_id]

        existing = await self._get_watch_record(file_path)
        now = utcnow().isoformat()
        if existing is not None:
            existing["mtime"] = mtime
            existing["simhash"] = content_hash
            existing["neuron_count"] = neuron_count
            existing["last_ingested"] = now
            existing["status"] = "active"
            return

        record_id = str(uuid4())
        store[record_id] = {
            "id": record_id,
            "brain_id": brain_id,
            "file_path": file_path,
            "mtime": mtime,
            "simhash": content_hash,
            "neuron_count": neuron_count,
            "last_ingested": now,
            "status": "active",
        }

    async def watch_mark_deleted(self, file_path: str) -> None:
        """Soft-delete a watched file's record."""
        record = await self._get_watch_record(file_path)
        if record is not None:
            record["status"] = "deleted"

    async def watch_list_files(self, status: str | None = None) -> list[WatchedFile]:
        """List tracked files, optionally filtered by status."""
        brain_id = self._get_brain_id()
        records = list(self._watch_files[brain_id].values())
        if status:
            records = [r for r in records if r["status"] == status]
        records.sort(key=lambda r: r["last_ingested"], reverse=True)
        return [
            WatchedFile(
                file_path=r["file_path"],
                mtime=r["mtime"],
                simhash=r["simhash"],
                neuron_count=r["neuron_count"],
                last_ingested=r["last_ingested"],
                status=r["status"],
            )
            for r in records
        ]

    async def watch_get_stats(self) -> dict[str, Any]:
        """Aggregate watch-state stats for the current brain."""
        brain_id = self._get_brain_id()
        records = self._watch_files[brain_id].values()

        stats: dict[str, Any] = {"total_files": 0, "total_neurons": 0, "by_status": {}}
        for record in records:
            status_val = record["status"]
            neurons = record["neuron_count"]
            stats["total_files"] += 1
            stats["total_neurons"] += neurons
            bucket = stats["by_status"].setdefault(status_val, {"files": 0, "neurons": 0})
            bucket["files"] += 1
            bucket["neurons"] += neurons
        return stats
