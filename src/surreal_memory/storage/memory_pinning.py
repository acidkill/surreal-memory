"""In-memory pinned-fiber, graph-density and training-file operations mixin.

These used to be SQLite-only, so every caller probed for them with ``hasattr``
and quietly degraded when they were missing — which meant decay and prune
deleted pinned memories, ``smem_pin`` refused to run, ``smem train`` re-encoded
its whole corpus on every pass, and ``activation_strategy="auto"`` never left
classic BFS. Implementing them here keeps the in-memory backend a faithful
stand-in for SurrealDB in tests, which is exactly where the original gap hid:
the acceptance tests only ever ran against SQLite.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any
from uuid import uuid4

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron
from surreal_memory.core.synapse import Synapse
from surreal_memory.utils.timeutils import utcnow

_MAX_LIST_LIMIT = 200


class InMemoryPinningMixin:
    """Mixin providing pinned-fiber, graph-density and training-file operations."""

    _fibers: dict[str, dict[str, Fiber]]
    _neurons: dict[str, dict[str, Neuron]]
    _synapses: dict[str, dict[str, Synapse]]
    _training_files: dict[str, dict[str, dict[str, Any]]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ========== Pinned (KB) Memory ==========

    async def pin_fibers(self, fiber_ids: list[str], pinned: bool = True) -> int:
        """Pin or unpin fibers by ID. Returns the number of fibers updated.

        Counts how many of the requested ids exist in this brain, whether or not
        the flag actually changed — matching the SQLite rowcount, so re-pinning
        an already-pinned fiber still reports it as updated.
        """
        if not fiber_ids:
            return 0

        brain_id = self._get_brain_id()
        store = self._fibers[brain_id]

        updated = 0
        for fiber_id in fiber_ids:
            fiber = store.get(fiber_id)
            if fiber is None:
                continue
            if fiber.pinned != pinned:
                store[fiber_id] = dc_replace(fiber, pinned=pinned)
            updated += 1
        return updated

    async def get_pinned_neuron_ids(self) -> set[str]:
        """Get every neuron ID belonging to a pinned fiber in the current brain."""
        brain_id = self._get_brain_id()
        result: set[str] = set()
        for fiber in self._fibers[brain_id].values():
            if fiber.pinned:
                result.update(fiber.neuron_ids)
        return result

    async def list_pinned_fibers(self, limit: int = 50) -> list[dict[str, Any]]:
        """List pinned fibers for the current brain, newest first."""
        brain_id = self._get_brain_id()
        safe_limit = min(max(int(limit), 0), _MAX_LIST_LIMIT)
        if safe_limit == 0:
            return []

        pinned = [f for f in self._fibers[brain_id].values() if f.pinned]
        pinned.sort(key=lambda f: f.created_at, reverse=True)

        typed = self._typed_memories[brain_id]  # type: ignore[attr-defined]
        results: list[dict[str, Any]] = []
        for fiber in pinned[:safe_limit]:
            # type/priority live on the typed memory, not the fiber.
            tm = typed.get(fiber.id)
            tags = sorted(fiber.auto_tags | fiber.agent_tags)
            results.append(
                {
                    "fiber_id": fiber.id,
                    "summary": fiber.summary or "",
                    "type": tm.memory_type.value if tm else "unknown",
                    "priority": int(tm.priority) if tm else 5,
                    "tags": sorted(tm.tags) if tm and tm.tags else tags,
                    "created_at": fiber.created_at.isoformat(),
                }
            )
        return results

    # ========== Graph Statistics ==========

    async def get_graph_density(self) -> float:
        """Average synapses per neuron for the current brain, or 0.0 if empty."""
        brain_id = self._get_brain_id()
        neuron_count = len(self._neurons[brain_id])
        if neuron_count == 0:
            return 0.0
        return len(self._synapses[brain_id]) / neuron_count

    # ========== Document Training Files ==========

    async def get_training_file_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Look up a training file record by content hash, or None if untrained."""
        brain_id = self._get_brain_id()
        for record in self._training_files[brain_id].values():
            if record["file_hash"] == file_hash:
                return dict(record)
        return None

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
        store = self._training_files[brain_id]

        existing = await self.get_training_file_by_hash(file_hash)
        if existing:
            record_id: str = existing["id"]
            record = store[record_id]
            record["chunks_total"] = chunks_total
            record["chunks_completed"] = chunks_completed
            record["status"] = status
            # Only completion stamps trained_at, matching SQLite.
            record["trained_at"] = utcnow().isoformat() if status == "completed" else None
            return record_id

        record_id = str(uuid4())
        store[record_id] = {
            "id": record_id,
            "brain_id": brain_id,
            "file_hash": file_hash,
            "file_path": file_path,
            "file_size": file_size,
            "chunks_total": chunks_total,
            "chunks_completed": chunks_completed,
            "status": status,
            "domain_tag": domain_tag,
            "trained_at": utcnow().isoformat() if status == "completed" else None,
            "created_at": utcnow().isoformat(),
        }
        return record_id

    async def update_training_file_progress(
        self, record_id: str, chunks_completed: int, status: str = "in_progress"
    ) -> None:
        """Update chunk progress for a training file, for resume support."""
        brain_id = self._get_brain_id()
        record = self._training_files[brain_id].get(record_id)
        if record is None:
            return

        record["chunks_completed"] = chunks_completed
        record["status"] = status
        if status == "completed":
            record["trained_at"] = utcnow().isoformat()

    async def get_training_stats(self) -> dict[str, Any]:
        """Training file counts for the current brain."""
        brain_id = self._get_brain_id()
        records = self._training_files[brain_id].values()

        stats: dict[str, Any] = {
            "total_files": len(records),
            "completed": 0,
            "in_progress": 0,
            "failed": 0,
            "total_chunks": 0,
        }
        for record in records:
            stats["total_chunks"] += record["chunks_completed"]
            status = record["status"]
            if status in stats:
                stats[status] += 1
        return stats
