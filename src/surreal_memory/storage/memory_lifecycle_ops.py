"""In-memory lifecycle, compression, and hot-index storage mixin for testing."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron
from surreal_memory.core.synapse import Synapse
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow

_MAX_HOT_SLOTS = 20
_MAX_HOT_SUMMARY_CHARS = 500
_MAX_PROMOTION_CANDIDATES = 200


class InMemoryLifecycleMixin:
    """Mixin providing compression, snapshot, lifecycle, and hot-index ops."""

    # Declared in InMemoryStorage.__init__
    _neurons: dict[str, dict[str, Neuron]]
    _synapses: dict[str, dict[str, Synapse]]
    _fibers: dict[str, dict[str, Fiber]]
    _typed_memories: dict[str, dict[str, TypedMemory]]
    _compression_backups: dict[str, dict[str, dict[str, Any]]]
    _neuron_snapshots: dict[str, dict[str, dict[str, Any]]]
    _hot_index: dict[str, list[dict[str, Any]]]
    _cognitive_states: dict[str, dict[str, dict[str, Any]]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def delete_neuron(self, neuron_id: str) -> bool:
        raise NotImplementedError

    # ========== Compression Backups ==========

    async def save_compression_backup(
        self,
        fiber_id: str,
        original_content: str,
        compression_tier: int,
        original_token_count: int,
        compressed_token_count: int,
    ) -> None:
        """Save (upsert) a pre-compression content backup for a fiber."""
        brain_id = self._get_brain_id()
        self._compression_backups.setdefault(brain_id, {})[fiber_id] = {
            "fiber_id": fiber_id,
            "brain_id": brain_id,
            "original_content": original_content,
            "compression_tier": compression_tier,
            "compressed_at": utcnow().isoformat(),
            "original_token_count": original_token_count,
            "compressed_token_count": compressed_token_count,
        }

    async def get_compression_backup(self, fiber_id: str) -> dict[str, Any] | None:
        """Retrieve the compression backup for a fiber, if any."""
        brain_id = self._get_brain_id()
        backup = self._compression_backups.get(brain_id, {}).get(fiber_id)
        return dict(backup) if backup is not None else None

    async def delete_compression_backup(self, fiber_id: str) -> bool:
        """Delete the compression backup for a fiber."""
        brain_id = self._get_brain_id()
        backups = self._compression_backups.get(brain_id, {})
        if fiber_id in backups:
            del backups[fiber_id]
            return True
        return False

    async def get_compression_stats(self) -> dict[str, Any]:
        """Return aggregate compression statistics for the current brain."""
        brain_id = self._get_brain_id()
        backups = self._compression_backups.get(brain_id, {}).values()

        counts: dict[int, int] = {}
        originals: dict[int, int] = {}
        compressed: dict[int, int] = {}
        for backup in backups:
            tier = int(backup["compression_tier"])
            counts[tier] = counts.get(tier, 0) + 1
            originals[tier] = originals.get(tier, 0) + int(backup["original_token_count"] or 0)
            compressed[tier] = compressed.get(tier, 0) + int(backup["compressed_token_count"] or 0)

        by_tier: dict[int, int] = {}
        total_backups = 0
        total_tokens_saved = 0
        for tier in sorted(counts):
            by_tier[tier] = counts[tier]
            total_backups += counts[tier]
            total_tokens_saved += max(0, originals[tier] - compressed[tier])

        return {
            "total_backups": total_backups,
            "by_tier": by_tier,
            "total_tokens_saved": total_tokens_saved,
        }

    # ========== Neuron Snapshots ==========

    async def save_neuron_snapshot(
        self,
        neuron_id: str,
        brain_id: str,
        original_content: str,
        compressed_at: str,
        tier: int,
    ) -> None:
        """Save (upsert) a pre-compression content snapshot for a neuron."""
        self._neuron_snapshots.setdefault(brain_id, {})[neuron_id] = {
            "neuron_id": neuron_id,
            "brain_id": brain_id,
            "original_content": original_content,
            "compressed_at": compressed_at,
            "tier": tier,
        }

    async def get_neuron_snapshot(self, neuron_id: str) -> dict[str, Any] | None:
        """Retrieve the snapshot for a neuron, if any."""
        brain_id = self._get_brain_id()
        snapshot = self._neuron_snapshots.get(brain_id, {}).get(neuron_id)
        return dict(snapshot) if snapshot is not None else None

    async def delete_neuron_snapshot(self, neuron_id: str) -> bool:
        """Delete the snapshot for a neuron."""
        brain_id = self._get_brain_id()
        snapshots = self._neuron_snapshots.get(brain_id, {})
        if neuron_id in snapshots:
            del snapshots[neuron_id]
            return True
        return False

    # ========== Lifecycle State and Neuron Flags ==========

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        """Update the lifecycle_state for a neuron."""
        brain_id = self._get_brain_id()
        neuron = self._neurons.get(brain_id, {}).get(neuron_id)
        if neuron is None:
            return
        self._neurons[brain_id][neuron_id] = neuron.with_metadata(lifecycle_state=lifecycle_state)

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        """Return count of neurons by lifecycle_state for the current brain."""
        brain_id = self._get_brain_id()
        distribution: dict[str, int] = {}
        for neuron in self._neurons.get(brain_id, {}).values():
            state = str(neuron.metadata.get("lifecycle_state") or "active")
            distribution[state] = distribution.get(state, 0) + 1
        return distribution

    async def update_neuron_ephemeral(self, neuron_id: str, ephemeral: bool) -> None:
        """Set or clear the ephemeral flag for a neuron."""
        brain_id = self._get_brain_id()
        neuron = self._neurons.get(brain_id, {}).get(neuron_id)
        if neuron is None:
            return
        self._neurons[brain_id][neuron_id] = replace(neuron, ephemeral=ephemeral)

    async def update_neurons_ephemeral_batch(self, neuron_ids: list[str], ephemeral: bool) -> None:
        """Batch-set ephemeral flag for multiple neurons."""
        if not neuron_ids:
            return
        brain_id = self._get_brain_id()
        neurons = self._neurons.get(brain_id, {})
        for neuron_id in neuron_ids:
            neuron = neurons.get(neuron_id)
            if neuron is not None:
                neurons[neuron_id] = replace(neuron, ephemeral=ephemeral)

    async def cleanup_ephemeral_neurons(self, max_age_hours: float = 24.0) -> int:
        """Delete ephemeral neurons older than max_age_hours."""
        brain_id = self._get_brain_id()
        cutoff = utcnow() - timedelta(hours=max_age_hours)

        expired = [
            neuron.id
            for neuron in self._neurons.get(brain_id, {}).values()
            if neuron.ephemeral and ensure_naive_utc(neuron.created_at) < cutoff
        ]

        deleted = 0
        for neuron_id in expired:
            if await self.delete_neuron(neuron_id):
                deleted += 1
        return deleted

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        """Set or clear the frozen flag for a neuron."""
        brain_id = self._get_brain_id()
        neuron = self._neurons.get(brain_id, {}).get(neuron_id)
        if neuron is None:
            return
        self._neurons[brain_id][neuron_id] = neuron.with_metadata(frozen=frozen)

    async def batch_update_ghost_shown(self, fiber_ids: list[str], timestamp: datetime) -> int:
        """Batch update last_ghost_shown_at for multiple fibers."""
        if not fiber_ids:
            return 0
        brain_id = self._get_brain_id()
        fibers = self._fibers.get(brain_id, {})
        for fiber_id in fiber_ids:
            fiber = fibers.get(fiber_id)
            if fiber is not None:
                fibers[fiber_id] = replace(fiber, last_ghost_shown_at=timestamp)
        # SQLite reports the requested count, not the number of rows actually matched.
        return len(fiber_ids)

    # ========== Hot Index ==========

    async def refresh_hot_index(self, items: list[dict[str, Any]]) -> int:
        """Replace the hot index with freshly scored items."""
        brain_id = self._get_brain_id()
        now = utcnow().isoformat()

        rows: list[dict[str, Any]] = []
        for item in items[:_MAX_HOT_SLOTS]:
            rows.append(
                {
                    "slot": item["slot"],
                    "category": item["category"],
                    "neuron_id": item["neuron_id"],
                    "summary": str(item["summary"])[:_MAX_HOT_SUMMARY_CHARS],
                    "confidence": item.get("confidence"),
                    "score": item["score"],
                    "updated_at": now,
                }
            )

        self._hot_index[brain_id] = rows
        return len(rows)

    async def get_hot_index(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get the current hot index items, sorted by score descending."""
        brain_id = self._get_brain_id()
        capped = min(limit, _MAX_HOT_SLOTS)
        rows = sorted(
            self._hot_index.get(brain_id, []),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        return [dict(row) for row in rows[:capped]]

    # ========== Typed Memory Extras ==========

    async def promote_memory_type(
        self,
        fiber_id: str,
        new_type: MemoryType,
        new_expires_at: str | None = None,
    ) -> bool:
        """Promote a memory's type and update its expiry."""
        brain_id = self._get_brain_id()
        typed_memories = self._typed_memories.get(brain_id, {})
        typed_memory = typed_memories.get(fiber_id)
        if typed_memory is None:
            return False

        old_type = str(typed_memory.memory_type.value)
        if old_type == new_type.value:
            return False

        metadata = dict(typed_memory.metadata)
        metadata["auto_promoted"] = True
        metadata["promoted_from"] = old_type
        metadata["promoted_at"] = utcnow().isoformat()

        expires_at = (
            ensure_naive_utc(datetime.fromisoformat(new_expires_at)) if new_expires_at else None
        )
        typed_memories[fiber_id] = replace(
            typed_memory,
            memory_type=new_type,
            expires_at=expires_at,
            metadata=metadata,
        )
        return True

    async def update_typed_memory_source(self, fiber_id: str, source: str) -> bool:
        """Update only the source field on a typed memory."""
        brain_id = self._get_brain_id()
        typed_memories = self._typed_memories.get(brain_id, {})
        typed_memory = typed_memories.get(fiber_id)
        if typed_memory is None:
            return False
        typed_memories[fiber_id] = replace(typed_memory, source=source)
        return True

    async def get_promotion_candidates(
        self,
        min_frequency: int = 5,
        source_type: str = "context",
    ) -> list[dict[str, Any]]:
        """Find typed memories eligible for auto-promotion."""
        brain_id = self._get_brain_id()
        fibers = self._fibers.get(brain_id, {})

        candidates: list[dict[str, Any]] = []
        for typed_memory in self._typed_memories.get(brain_id, {}).values():
            if str(typed_memory.memory_type.value) != source_type:
                continue
            fiber = fibers.get(typed_memory.fiber_id)
            if fiber is None or fiber.frequency < min_frequency or fiber.pinned:
                continue
            candidates.append(
                {
                    "fiber_id": typed_memory.fiber_id,
                    "memory_type": str(typed_memory.memory_type.value),
                    "expires_at": typed_memory.expires_at,
                    "metadata": dict(typed_memory.metadata),
                    "frequency": fiber.frequency,
                    "conductivity": fiber.conductivity,
                }
            )
            if len(candidates) >= _MAX_PROMOTION_CANDIDATES:
                break
        return candidates

    # ========== Merkle Buckets and Schema History ==========

    async def get_bucket_entity_ids(
        self, entity_type: str, prefix: str, *, is_pro: bool = False
    ) -> list[str]:
        """Return all entity IDs in the given bucket prefix for delete detection."""
        if not is_pro:
            return []

        brain_id = self._get_brain_id()
        entity_ids: list[str]
        if entity_type == "neuron":
            entity_ids = list(self._neurons.get(brain_id, {}))
        elif entity_type == "synapse":
            entity_ids = list(self._synapses.get(brain_id, {}))
        elif entity_type == "fiber":
            entity_ids = list(self._fibers.get(brain_id, {}))
        else:
            return []

        parts = prefix.split("/")
        if len(parts) != 2:
            return []
        hex_prefix = parts[1].lower()

        return sorted(eid for eid in entity_ids if eid[:2].lower() == hex_prefix)

    async def get_schema_history(
        self,
        neuron_id: str,
        *,
        max_depth: int = 20,
    ) -> list[dict[str, Any]]:
        """Walk the version chain for a hypothesis, newest-first."""
        brain_id = self._get_brain_id()
        states = self._cognitive_states.get(brain_id, {})

        history: list[dict[str, Any]] = []
        current_id: str | None = neuron_id
        seen: set[str] = set()

        while current_id and len(history) < max_depth:
            if current_id in seen:
                break
            seen.add(current_id)

            state = states.get(current_id)
            if state is None:
                break

            parent_raw = state.get("parent_schema_id")
            history.append(
                {
                    "neuron_id": str(state.get("neuron_id", current_id)),
                    "confidence": float(state.get("confidence", 0.5)),
                    "evidence_for_count": int(state.get("evidence_for_count", 0)),
                    "evidence_against_count": int(state.get("evidence_against_count", 0)),
                    "status": str(state.get("status", "")),
                    "schema_version": int(state.get("schema_version", 1)),
                    "parent_schema_id": parent_raw,
                    "created_at": state.get("created_at"),
                }
            )
            current_id = str(parent_raw) if parent_raw is not None else None

        return history
