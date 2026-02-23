"""Tests for composite FalkorDBStorage — brain ops, stats, export/import."""

from __future__ import annotations

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage


class TestBrainOperations:
    """Brain metadata CRUD."""

    @pytest.mark.asyncio
    async def test_save_and_get_brain(self, storage: FalkorDBStorage) -> None:
        brain = Brain.create(name="new-brain")
        await storage.save_brain(brain)

        retrieved = await storage.get_brain(brain.id)
        assert retrieved is not None
        assert retrieved.name == "new-brain"
        assert retrieved.id == brain.id

    @pytest.mark.asyncio
    async def test_find_brain_by_name(self, storage: FalkorDBStorage) -> None:
        brain = Brain.create(name="findme")
        await storage.save_brain(brain)

        found = await storage.find_brain_by_name("findme")
        assert found is not None
        assert found.id == brain.id

    @pytest.mark.asyncio
    async def test_find_nonexistent_brain_returns_none(
        self, storage: FalkorDBStorage
    ) -> None:
        found = await storage.find_brain_by_name("does-not-exist")
        assert found is None

    @pytest.mark.asyncio
    async def test_brain_upsert(self, storage: FalkorDBStorage) -> None:
        """save_brain should upsert (MERGE) not duplicate."""
        brain = Brain.create(name="upsert-test")
        await storage.save_brain(brain)

        from dataclasses import replace

        updated = replace(brain, name="upsert-updated")
        await storage.save_brain(updated)

        retrieved = await storage.get_brain(brain.id)
        assert retrieved is not None
        assert retrieved.name == "upsert-updated"


class TestStats:
    """Brain statistics."""

    @pytest.mark.asyncio
    async def test_empty_stats(
        self, storage: FalkorDBStorage, brain_id: str
    ) -> None:
        stats = await storage.get_stats(brain_id)
        assert stats["neuron_count"] == 0
        assert stats["synapse_count"] == 0
        assert stats["fiber_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_adding_data(
        self, storage: FalkorDBStorage, brain_id: str
    ) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Stat A")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Stat B")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        await storage.add_synapse(s)

        f = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s.id},
            anchor_neuron_id=n1.id,
        )
        await storage.add_fiber(f)

        stats = await storage.get_stats(brain_id)
        assert stats["neuron_count"] == 2
        assert stats["synapse_count"] == 1
        assert stats["fiber_count"] == 1

    @pytest.mark.asyncio
    async def test_enhanced_stats(
        self, storage: FalkorDBStorage, brain_id: str
    ) -> None:
        n = Neuron.create(type=NeuronType.CONCEPT, content="Enhanced stat")
        await storage.add_neuron(n)

        stats = await storage.get_enhanced_stats(brain_id)
        assert "neuron_count" in stats
        assert "storage_backend" in stats
        assert stats["storage_backend"] == "falkordb"


class TestClear:
    """Brain data clearing."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_data(
        self, storage: FalkorDBStorage, brain_id: str
    ) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="To clear A")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="To clear B")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        s = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        await storage.add_synapse(s)

        await storage.clear(brain_id)

        stats = await storage.get_stats(brain_id)
        assert stats["neuron_count"] == 0
        assert stats["synapse_count"] == 0


class TestExportImport:
    """Brain export/import round-trip."""

    @pytest.mark.asyncio
    async def test_export_import_roundtrip(
        self, storage: FalkorDBStorage, brain_id: str
    ) -> None:
        # Populate brain
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Export A")
        n2 = Neuron.create(type=NeuronType.ENTITY, content="Export B")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        s = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO, weight=0.7)
        await storage.add_synapse(s)

        f = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids={s.id},
            anchor_neuron_id=n1.id,
            pathway=[n1.id, n2.id],
        )
        await storage.add_fiber(f)

        # Export
        snapshot = await storage.export_brain(brain_id)
        assert len(snapshot.neurons) == 2
        assert len(snapshot.synapses) == 1
        assert len(snapshot.fibers) == 1

        # Import into new brain
        target_brain = Brain.create(name="import-target")
        await storage.save_brain(target_brain)
        imported_id = await storage.import_brain(
            snapshot, target_brain_id=target_brain.id
        )

        # Verify imported data
        await storage.set_brain_with_indexes(imported_id)
        stats = await storage.get_stats(imported_id)
        assert stats["neuron_count"] == 2
        assert stats["synapse_count"] == 1
        assert stats["fiber_count"] == 1


class TestBrainContext:
    """Brain context management."""

    @pytest.mark.asyncio
    async def test_set_brain_changes_context(
        self, storage: FalkorDBStorage
    ) -> None:
        brain1 = Brain.create(name="ctx-1")
        brain2 = Brain.create(name="ctx-2")
        await storage.save_brain(brain1)
        await storage.save_brain(brain2)

        await storage.set_brain_with_indexes(brain1.id)
        assert storage.current_brain_id == brain1.id

        await storage.set_brain_with_indexes(brain2.id)
        assert storage.current_brain_id == brain2.id

    @pytest.mark.asyncio
    async def test_data_isolated_between_brains(
        self, storage: FalkorDBStorage
    ) -> None:
        """Each brain should have its own graph."""
        brain1 = Brain.create(name="iso-1")
        brain2 = Brain.create(name="iso-2")
        await storage.save_brain(brain1)
        await storage.save_brain(brain2)

        # Add neuron to brain1
        await storage.set_brain_with_indexes(brain1.id)
        n = Neuron.create(type=NeuronType.CONCEPT, content="Only in brain1")
        await storage.add_neuron(n)

        # Switch to brain2 — should not see brain1's neuron
        await storage.set_brain_with_indexes(brain2.id)
        result = await storage.get_neuron(n.id)
        assert result is None

        # Brain1 still has it
        await storage.set_brain_with_indexes(brain1.id)
        result = await storage.get_neuron(n.id)
        assert result is not None

        # Cleanup
        await storage.clear(brain1.id)
        await storage.clear(brain2.id)


class TestBatchOperations:
    """Batch save/auto-save (no-ops for FalkorDB)."""

    @pytest.mark.asyncio
    async def test_batch_save_noop(self, storage: FalkorDBStorage) -> None:
        """batch_save should not raise."""
        storage.disable_auto_save()
        await storage.batch_save()
        storage.enable_auto_save()
