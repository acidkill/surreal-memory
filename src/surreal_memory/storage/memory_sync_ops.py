"""In-memory multi-device sync storage mixin for testing.

Covers the device registry, change log, Merkle hash cache, and Bayesian
depth priors, mirroring the semantics of the SQLite backend mixins.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron
from surreal_memory.core.synapse import Synapse
from surreal_memory.core.sync_records import ChangeEntry, DeviceRecord
from surreal_memory.engine.depth_prior import DepthPrior
from surreal_memory.sync.merkle import ENTITY_TYPES, MerkleTreeBuilder
from surreal_memory.utils.timeutils import utcnow

_MAX_CHANGE_LIMIT = 10000


class InMemorySyncMixin:
    """Mixin providing device, change log, Merkle, and depth prior ops."""

    # Declared in InMemoryStorage.__init__
    _neurons: dict[str, dict[str, Neuron]]
    _synapses: dict[str, dict[str, Synapse]]
    _fibers: dict[str, dict[str, Fiber]]
    _devices: dict[str, dict[str, DeviceRecord]]
    _change_log: dict[str, list[ChangeEntry]]
    _change_log_seq: dict[str, int]
    _merkle_hashes: dict[str, dict[str, dict[str, str]]]
    _depth_priors: dict[str, dict[str, dict[int, DepthPrior]]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    # ========== Device Registry Operations ==========

    async def register_device(self, device_id: str, device_name: str = "") -> DeviceRecord:
        """Register a device for the current brain (upsert)."""
        brain_id = self._get_brain_id()
        now = utcnow()

        if brain_id not in self._devices:
            self._devices[brain_id] = {}
        existing = self._devices[brain_id].get(device_id)

        if existing is None:
            self._devices[brain_id][device_id] = DeviceRecord(
                device_id=device_id,
                brain_id=brain_id,
                device_name=device_name,
                last_sync_at=None,
                last_sync_sequence=0,
                registered_at=now,
            )
        else:
            self._devices[brain_id][device_id] = replace(existing, device_name=device_name)

        # Mirrors SQLite: the return value reflects the insert values, not the
        # stored row, so re-registering reports a fresh record.
        return DeviceRecord(
            device_id=device_id,
            brain_id=brain_id,
            device_name=device_name,
            last_sync_at=None,
            last_sync_sequence=0,
            registered_at=now,
        )

    async def get_device(self, device_id: str) -> DeviceRecord | None:
        """Get device info for a specific device."""
        brain_id = self._get_brain_id()
        return self._devices.get(brain_id, {}).get(device_id)

    async def list_devices(self) -> list[DeviceRecord]:
        """List all registered devices for the current brain."""
        brain_id = self._get_brain_id()
        devices = list(self._devices.get(brain_id, {}).values())
        devices.sort(key=lambda d: d.registered_at)
        return devices

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        """Update the last sync timestamp and sequence for a device."""
        brain_id = self._get_brain_id()
        devices = self._devices.get(brain_id, {})
        existing = devices.get(device_id)
        if existing is None:
            return
        devices[device_id] = replace(
            existing,
            last_sync_at=utcnow(),
            last_sync_sequence=last_sync_sequence,
        )

    async def remove_device(self, device_id: str) -> bool:
        """Remove a device from the registry. Returns True if deleted."""
        brain_id = self._get_brain_id()
        devices = self._devices.get(brain_id, {})
        if device_id in devices:
            del devices[device_id]
            return True
        return False

    # ========== Change Log Operations ==========

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append a change to the log. Returns the sequence number (id)."""
        brain_id = self._get_brain_id()
        return self._append_change(
            brain_id=brain_id,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
            device_id=device_id,
            payload=payload or {},
            changed_at=utcnow(),
        )

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[ChangeEntry]:
        """Get changes after a given sequence number, ordered by id ASC."""
        safe_limit = min(limit, _MAX_CHANGE_LIMIT)
        brain_id = self._get_brain_id()
        entries = [e for e in self._change_log.get(brain_id, []) if e.id > sequence]
        entries.sort(key=lambda e: e.id)
        return entries[:safe_limit]

    async def get_unsynced_changes(self, limit: int = 1000) -> list[ChangeEntry]:
        """Get all unsynced changes, ordered by id ASC."""
        safe_limit = min(limit, _MAX_CHANGE_LIMIT)
        brain_id = self._get_brain_id()
        entries = [e for e in self._change_log.get(brain_id, []) if not e.synced]
        entries.sort(key=lambda e: e.id)
        return entries[:safe_limit]

    async def mark_synced(self, up_to_sequence: int) -> int:
        """Mark all changes up to a sequence number as synced. Returns count marked."""
        brain_id = self._get_brain_id()
        entries = self._change_log.get(brain_id, [])
        marked = 0
        for index, entry in enumerate(entries):
            if entry.id <= up_to_sequence and not entry.synced:
                entries[index] = replace(entry, synced=True)
                marked += 1
        return marked

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        """Delete synced changes older than N days. Returns count pruned."""
        brain_id = self._get_brain_id()
        entries = self._change_log.get(brain_id, [])
        if not entries:
            return 0
        cutoff = utcnow() - timedelta(days=older_than_days)
        kept = [e for e in entries if not (e.synced and e.changed_at < cutoff)]
        pruned = len(entries) - len(kept)
        self._change_log[brain_id] = kept
        return pruned

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        """Seed the change log with all existing entities as 'insert' entries."""
        brain_id = self._get_brain_id()
        now = utcnow()
        tracked = {(e.entity_type, e.entity_id) for e in self._change_log.get(brain_id, [])}
        counts: dict[str, int] = {"neurons": 0, "synapses": 0, "fibers": 0}

        for neuron in self._neurons.get(brain_id, {}).values():
            if neuron.ephemeral or ("neuron", neuron.id) in tracked:
                continue
            self._seed_entry(brain_id, "neuron", neuron.id, device_id, _neuron_payload(neuron), now)
            counts["neurons"] += 1

        for synapse in self._synapses.get(brain_id, {}).values():
            if ("synapse", synapse.id) in tracked:
                continue
            payload = _synapse_payload(synapse)
            self._seed_entry(brain_id, "synapse", synapse.id, device_id, payload, now)
            counts["synapses"] += 1

        for fiber in self._fibers.get(brain_id, {}).values():
            if ("fiber", fiber.id) in tracked:
                continue
            self._seed_entry(brain_id, "fiber", fiber.id, device_id, _fiber_payload(fiber), now)
            counts["fibers"] += 1

        return counts

    async def get_change_log_stats(self) -> dict[str, Any]:
        """Get change log statistics for the current brain."""
        brain_id = self._get_brain_id()
        entries = self._change_log.get(brain_id, [])
        synced = sum(1 for e in entries if e.synced)
        return {
            "total": len(entries),
            "pending": len(entries) - synced,
            "synced": synced,
            "last_sequence": max((e.id for e in entries), default=0),
        }

    # ========== Merkle Hash Operations ==========

    async def compute_merkle_root(self, entity_type: str, *, is_pro: bool = False) -> str | None:
        """Compute and cache the Merkle root hash for an entity type."""
        if not is_pro:
            return None

        if entity_type not in ENTITY_TYPES:
            raise ValueError(
                f"Unknown entity_type: {entity_type!r}. Must be one of {ENTITY_TYPES}."
            )

        brain_id = self._get_brain_id()
        tree = MerkleTreeBuilder.build_tree(self._merkle_entities(entity_type), entity_type)

        if brain_id not in self._merkle_hashes:
            self._merkle_hashes[brain_id] = {}
        cached = self._merkle_hashes[brain_id].setdefault(entity_type, {})
        cached[tree.prefix] = tree.hash
        for child in tree.children:
            cached[child.prefix] = child.hash

        return tree.hash

    async def get_merkle_tree(self, entity_type: str, *, is_pro: bool = False) -> dict[str, str]:
        """Return cached {prefix: hash} map for an entity type."""
        if not is_pro:
            return {}
        brain_id = self._get_brain_id()
        return dict(self._merkle_hashes.get(brain_id, {}).get(entity_type, {}))

    async def invalidate_merkle_prefix(
        self, entity_type: str, entity_id: str, *, is_pro: bool = False
    ) -> None:
        """Delete cached hashes for the bucket containing entity_id."""
        if not is_pro:
            return

        brain_id = self._get_brain_id()
        cached = self._merkle_hashes.get(brain_id, {}).get(entity_type)
        if not cached:
            return

        type_prefix = f"{entity_type}s"
        bucket_key = (
            entity_id[:2].lower() if len(entity_id) >= 2 else entity_id.lower().ljust(2, "0")
        )
        cached.pop(type_prefix, None)
        cached.pop(f"{type_prefix}/{bucket_key}", None)

    async def get_merkle_root(self, *, is_pro: bool = False) -> str | None:
        """Get combined root hash across all entity types."""
        if not is_pro:
            return None

        brain_id = self._get_brain_id()
        by_type = self._merkle_hashes.get(brain_id, {})
        hashes: list[str] = []

        for entity_type in ("neuron", "synapse", "fiber"):
            cached = by_type.get(entity_type, {}).get(f"{entity_type}s")
            if cached is None:
                return None
            hashes.append(cached)

        return MerkleTreeBuilder.compute_branch_hash(hashes)

    # ========== Depth Prior Operations ==========

    async def get_depth_priors_batch(
        self,
        entity_texts: list[str],
    ) -> dict[str, list[DepthPrior]]:
        """Batch-fetch Bayesian depth priors for multiple entities."""
        if not entity_texts:
            return {}

        brain_id = self._get_brain_id()
        by_entity = self._depth_priors.get(brain_id, {})
        return {text: list(by_entity.get(text, {}).values()) for text in entity_texts}

    async def upsert_depth_prior(self, prior: DepthPrior) -> None:
        """Insert or update a single depth prior."""
        brain_id = self._get_brain_id()

        if brain_id not in self._depth_priors:
            self._depth_priors[brain_id] = {}
        by_entity = self._depth_priors[brain_id]
        if prior.entity_text not in by_entity:
            by_entity[prior.entity_text] = {}

        level = int(prior.depth_level)
        existing = by_entity[prior.entity_text].get(level)
        # Mirrors SQLite ON CONFLICT: created_at of the original row is retained.
        stored = replace(prior, created_at=existing.created_at) if existing else prior
        by_entity[prior.entity_text][level] = stored

    async def get_stale_priors(self, older_than: datetime) -> list[DepthPrior]:
        """Find depth priors not updated since a given date."""
        brain_id = self._get_brain_id()
        by_entity = self._depth_priors.get(brain_id, {})
        return [
            prior
            for by_level in by_entity.values()
            for prior in by_level.values()
            if prior.last_updated < older_than
        ]

    async def delete_depth_priors(self, entity_text: str) -> int:
        """Delete all depth priors for an entity. Returns count deleted."""
        brain_id = self._get_brain_id()
        by_entity = self._depth_priors.get(brain_id, {})
        removed = by_entity.pop(entity_text, {})
        return len(removed)

    # ========== Internal helpers ==========

    def _next_sequence(self, brain_id: str) -> int:
        sequence = self._change_log_seq.get(brain_id, 0) + 1
        self._change_log_seq[brain_id] = sequence
        return sequence

    def _append_change(
        self,
        *,
        brain_id: str,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str,
        payload: dict[str, Any],
        changed_at: datetime,
    ) -> int:
        if brain_id not in self._change_log:
            self._change_log[brain_id] = []
        entry = ChangeEntry(
            id=self._next_sequence(brain_id),
            brain_id=brain_id,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
            device_id=device_id,
            changed_at=changed_at,
            payload=payload,
            synced=False,
        )
        self._change_log[brain_id].append(entry)
        return entry.id

    def _seed_entry(
        self,
        brain_id: str,
        entity_type: str,
        entity_id: str,
        device_id: str,
        payload: dict[str, Any],
        changed_at: datetime,
    ) -> None:
        self._append_change(
            brain_id=brain_id,
            entity_type=entity_type,
            entity_id=entity_id,
            operation="insert",
            device_id=device_id,
            payload=payload,
            changed_at=changed_at,
        )

    def _merkle_entities(self, entity_type: str) -> list[tuple[str, str, str]]:
        """Return (entity_id, updated_at, content_hash) tuples for hashing.

        The in-memory models carry no ``updated_at`` field, so that component is
        always the empty string — matching SQLite rows written by the current
        code path, which never populates the column either. Synapses have no
        content hash on either backend.
        """
        brain_id = self._get_brain_id()

        if entity_type == "neuron":
            neurons = self._neurons.get(brain_id, {}).values()
            return [(n.id, "", str(n.content_hash)) for n in neurons]
        if entity_type == "synapse":
            return [(s.id, "", "") for s in self._synapses.get(brain_id, {}).values()]
        return [(f.id, "", f.summary or "") for f in self._fibers.get(brain_id, {}).values()]


def _neuron_payload(neuron: Neuron) -> dict[str, Any]:
    """Build a sync payload dict for a neuron."""
    return {
        "id": neuron.id,
        "type": neuron.type.value,
        "content": neuron.content,
        "metadata": dict(neuron.metadata),
        "content_hash": neuron.content_hash,
        "created_at": neuron.created_at.isoformat(),
    }


def _synapse_payload(synapse: Synapse) -> dict[str, Any]:
    """Build a sync payload dict for a synapse."""
    return {
        "id": synapse.id,
        "source_id": synapse.source_id,
        "target_id": synapse.target_id,
        "type": synapse.type.value,
        "weight": synapse.weight,
        "direction": synapse.direction.value,
        "metadata": dict(synapse.metadata),
        "reinforced_count": synapse.reinforced_count,
        "last_activated": synapse.last_activated.isoformat() if synapse.last_activated else None,
        "created_at": synapse.created_at.isoformat(),
    }


def _fiber_payload(fiber: Fiber) -> dict[str, Any]:
    """Build a sync payload dict for a fiber."""
    return {
        "id": fiber.id,
        "neuron_ids": sorted(fiber.neuron_ids),
        "synapse_ids": sorted(fiber.synapse_ids),
        "anchor_neuron_id": fiber.anchor_neuron_id,
        "pathway": list(fiber.pathway),
        "conductivity": fiber.conductivity,
        "last_conducted": fiber.last_conducted.isoformat() if fiber.last_conducted else None,
        "time_start": fiber.time_start.isoformat() if fiber.time_start else None,
        "time_end": fiber.time_end.isoformat() if fiber.time_end else None,
        "coherence": fiber.coherence,
        "salience": fiber.salience,
        "frequency": fiber.frequency,
        "summary": fiber.summary,
        "auto_tags": sorted(fiber.auto_tags),
        "agent_tags": sorted(fiber.agent_tags),
        "metadata": dict(fiber.metadata),
        "compression_tier": fiber.compression_tier,
        "created_at": fiber.created_at.isoformat(),
    }
