"""Tests for FK constraint handling during consolidation and recall.

Verifies that operations gracefully handle FK violations caused by
concurrent neuron/synapse deletion (e.g., during consolidation pruning).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage() -> SQLiteStorage:
    """Create a temporary SQLite storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        s = SQLiteStorage(db_path)
        await s.initialize()

        brain = Brain.create(name="test_brain")
        await s.save_brain(brain)
        s.set_brain(brain.id)

        yield s
        await s.close()


class TestUpdateNeuronStateFK:
    """Tests for update_neuron_state FK handling."""

    @pytest.mark.asyncio
    async def test_update_state_for_existing_neuron(self, storage: SQLiteStorage) -> None:
        """Normal case: updating state for existing neuron succeeds."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="test")
        await storage.add_neuron(neuron)

        state = await storage.get_neuron_state(neuron.id)
        assert state is not None

        from dataclasses import replace

        updated = replace(state, activation_level=0.9, access_frequency=5)
        await storage.update_neuron_state(updated)

        result = await storage.get_neuron_state(neuron.id)
        assert result is not None
        assert result.activation_level == 0.9
        assert result.access_frequency == 5

    @pytest.mark.asyncio
    async def test_update_state_for_deleted_neuron_no_error(self, storage: SQLiteStorage) -> None:
        """FK fix: updating state for deleted neuron should not raise."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="will be deleted")
        await storage.add_neuron(neuron)

        state = await storage.get_neuron_state(neuron.id)
        assert state is not None

        # Delete the neuron (simulating consolidation prune)
        await storage.delete_neuron(neuron.id)

        # This should NOT raise — FK constraint gracefully handled
        from dataclasses import replace

        updated = replace(state, activation_level=0.5)
        await storage.update_neuron_state(updated)

        # State should not exist since neuron was deleted
        result = await storage.get_neuron_state(neuron.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_state_for_never_existed_neuron(self, storage: SQLiteStorage) -> None:
        """FK fix: updating state for non-existent neuron should not raise."""
        now = utcnow()
        fake_state = NeuronState(
            neuron_id="non-existent-id",
            activation_level=0.5,
            access_frequency=1,
            last_activated=now,
            decay_rate=0.1,
            created_at=now,
        )
        # Should not raise
        await storage.update_neuron_state(fake_state)


class TestAddSynapseFK:
    """Tests for add_synapse FK handling."""

    @pytest.mark.asyncio
    async def test_add_synapse_with_deleted_source(self, storage: SQLiteStorage) -> None:
        """Adding synapse with deleted source neuron raises ValueError."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="source")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="target")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Delete source
        await storage.delete_neuron(n1.id)

        synapse = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_synapse(synapse)

    @pytest.mark.asyncio
    async def test_add_synapse_with_deleted_target(self, storage: SQLiteStorage) -> None:
        """Adding synapse with deleted target neuron raises ValueError."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="source")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="target")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Delete target
        await storage.delete_neuron(n2.id)

        synapse = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_synapse(synapse)

    @pytest.mark.asyncio
    async def test_add_synapse_fk_error_message(self, storage: SQLiteStorage) -> None:
        """FK IntegrityError during add_synapse gives correct error message."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="source")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="target")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Normal add succeeds
        synapse = Synapse.create(source_id=n1.id, target_id=n2.id, type=SynapseType.RELATED_TO)
        await storage.add_synapse(synapse)

        # Duplicate add raises "already exists"
        with pytest.raises(ValueError, match="already exists"):
            await storage.add_synapse(synapse)


class TestSaveMaturationFK:
    """Tests for save_maturation FK handling."""

    @pytest.mark.asyncio
    async def test_save_maturation_for_deleted_fiber_no_error(self, storage: SQLiteStorage) -> None:
        """FK fix: saving maturation for deleted fiber should not raise."""
        from neural_memory.core.fiber import Fiber
        from neural_memory.engine.memory_stages import MaturationRecord, MemoryStage

        # Create a fiber first
        n = Neuron.create(type=NeuronType.CONCEPT, content="test")
        await storage.add_neuron(n)
        fiber = Fiber.create(neuron_ids={n.id}, synapse_ids=set(), anchor_neuron_id=n.id)
        await storage.add_fiber(fiber)

        # Create maturation record
        record = MaturationRecord(
            fiber_id=fiber.id,
            brain_id="test",
            stage=MemoryStage.SHORT_TERM,
            stage_entered_at=utcnow(),
        )
        await storage.save_maturation(record)

        # Delete the fiber (simulating consolidation merge)
        await storage.delete_fiber(fiber.id)

        # This should NOT raise — FK constraint gracefully handled
        from dataclasses import replace

        updated = replace(record, stage=MemoryStage.WORKING)
        await storage.save_maturation(updated)


class TestAddFiberFK:
    """Tests for add_fiber FK handling with pruned neurons."""

    @pytest.mark.asyncio
    async def test_add_fiber_with_deleted_neuron_raises_fk_error(
        self, storage: SQLiteStorage
    ) -> None:
        """Adding fiber referencing deleted neuron raises FK-specific ValueError."""
        from neural_memory.core.fiber import Fiber

        n1 = Neuron.create(type=NeuronType.CONCEPT, content="existing")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="will be deleted")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Delete n2 (simulating prune)
        await storage.delete_neuron(n2.id)

        # Fiber references both — n2 no longer exists
        fiber = Fiber.create(
            neuron_ids={n1.id, n2.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
        )
        with pytest.raises(ValueError, match="non-existent neurons"):
            await storage.add_fiber(fiber)

    @pytest.mark.asyncio
    async def test_add_fiber_duplicate_raises_already_exists(self, storage: SQLiteStorage) -> None:
        """Adding duplicate fiber still raises 'already exists' error."""
        from neural_memory.core.fiber import Fiber

        n = Neuron.create(type=NeuronType.CONCEPT, content="test")
        await storage.add_neuron(n)

        fiber = Fiber.create(neuron_ids={n.id}, synapse_ids=set(), anchor_neuron_id=n.id)
        await storage.add_fiber(fiber)

        with pytest.raises(ValueError, match="already exists"):
            await storage.add_fiber(fiber)


class TestConsolidationPruneFK:
    """Integration test: consolidation prune + subsequent operations."""

    @pytest.mark.asyncio
    async def test_prune_then_lifecycle_no_fk_error(self, storage: SQLiteStorage) -> None:
        """After prune deletes neurons, lifecycle decay should not FK-crash."""
        from dataclasses import replace

        # Create connected neurons
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="connected")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="orphan")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Get states
        s1 = await storage.get_neuron_state(n1.id)
        s2 = await storage.get_neuron_state(n2.id)
        assert s1 is not None
        assert s2 is not None

        # Delete orphan (simulating prune)
        await storage.delete_neuron(n2.id)

        # Try to update both states (simulating lifecycle decay after prune)
        # s2's neuron is deleted — should not raise
        updated_s1 = replace(s1, activation_level=0.3)
        updated_s2 = replace(s2, activation_level=0.1)

        await storage.update_neuron_state(updated_s1)  # OK
        await storage.update_neuron_state(updated_s2)  # Should not raise

        # Only s1 should have the update
        result = await storage.get_neuron_state(n1.id)
        assert result is not None
        assert result.activation_level == 0.3
