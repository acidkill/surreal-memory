"""Tests for FalkorDB fiber storage operations."""

from __future__ import annotations

import pytest

from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage


@pytest.fixture
async def fiber_data(storage: FalkorDBStorage) -> tuple[list[Neuron], Fiber]:
    """Create neurons and a fiber for testing."""
    neurons = [
        Neuron.create(type=NeuronType.CONCEPT, content=f"Fiber neuron {i}")
        for i in range(4)
    ]
    for n in neurons:
        await storage.add_neuron(n)

    # Create synapses along the pathway
    synapses = []
    for i in range(len(neurons) - 1):
        s = Synapse.create(
            neurons[i].id, neurons[i + 1].id, SynapseType.RELATES_TO
        )
        await storage.add_synapse(s)
        synapses.append(s)

    fiber = Fiber.create(
        neuron_ids={n.id for n in neurons},
        synapse_ids={s.id for s in synapses},
        anchor_neuron_id=neurons[0].id,
        pathway=[n.id for n in neurons],
    )
    fiber = fiber.add_auto_tags("test-tag", "concept")
    fiber = fiber.add_tags("user-tag")

    return neurons, fiber


class TestFiberCRUD:
    """Basic fiber create/read/update/delete."""

    @pytest.mark.asyncio
    async def test_add_and_get_fiber(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        neurons, fiber = fiber_data
        result_id = await storage.add_fiber(fiber)

        assert result_id == fiber.id
        retrieved = await storage.get_fiber(fiber.id)
        assert retrieved is not None
        assert retrieved.id == fiber.id
        assert retrieved.anchor_neuron_id == neurons[0].id
        assert len(retrieved.pathway) == 4

    @pytest.mark.asyncio
    async def test_fiber_tags_preserved(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        _, fiber = fiber_data
        await storage.add_fiber(fiber)

        retrieved = await storage.get_fiber(fiber.id)
        assert retrieved is not None
        assert "test-tag" in retrieved.auto_tags
        assert "concept" in retrieved.auto_tags
        assert "user-tag" in retrieved.agent_tags

    @pytest.mark.asyncio
    async def test_fiber_neuron_ids_preserved(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        neurons, fiber = fiber_data
        await storage.add_fiber(fiber)

        retrieved = await storage.get_fiber(fiber.id)
        assert retrieved is not None
        for n in neurons:
            assert n.id in retrieved.neuron_ids

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(
        self, storage: FalkorDBStorage
    ) -> None:
        result = await storage.get_fiber("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_fiber(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        _, fiber = fiber_data
        await storage.add_fiber(fiber)

        updated = fiber.with_salience(0.9).with_summary("Test summary")
        await storage.update_fiber(updated)

        retrieved = await storage.get_fiber(fiber.id)
        assert retrieved is not None
        assert retrieved.salience == pytest.approx(0.9, abs=0.01)
        assert retrieved.summary == "Test summary"

    @pytest.mark.asyncio
    async def test_delete_fiber(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        _, fiber = fiber_data
        await storage.add_fiber(fiber)

        deleted = await storage.delete_fiber(fiber.id)
        assert deleted is True

        result = await storage.get_fiber(fiber.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(
        self, storage: FalkorDBStorage
    ) -> None:
        deleted = await storage.delete_fiber("nonexistent-id")
        assert deleted is False


class TestFiberSearch:
    """Fiber find/query operations."""

    @pytest.mark.asyncio
    async def test_find_by_neuron(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        neurons, fiber = fiber_data
        await storage.add_fiber(fiber)

        results = await storage.find_fibers(contains_neuron=neurons[0].id)
        assert len(results) >= 1
        assert any(f.id == fiber.id for f in results)

    @pytest.mark.asyncio
    async def test_find_by_tags(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        _, fiber = fiber_data
        await storage.add_fiber(fiber)

        results = await storage.find_fibers(tags={"test-tag"})
        assert len(results) >= 1
        assert any(f.id == fiber.id for f in results)

    @pytest.mark.asyncio
    async def test_find_by_min_salience(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        _, fiber = fiber_data
        high_salience = fiber.with_salience(0.95)
        await storage.add_fiber(high_salience)

        results = await storage.find_fibers(min_salience=0.9)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_find_fibers_batch(
        self,
        storage: FalkorDBStorage,
        fiber_data: tuple[list[Neuron], Fiber],
    ) -> None:
        neurons, fiber = fiber_data
        await storage.add_fiber(fiber)

        results = await storage.find_fibers_batch(
            neuron_ids=[neurons[0].id, neurons[1].id],
            limit_per_neuron=5,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_fibers_ordered(
        self, storage: FalkorDBStorage
    ) -> None:
        """Test get_fibers with ordering."""
        # Create neurons for two fibers
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Ordered A")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Ordered B")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        f1 = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids=set(),
            anchor_neuron_id=n1.id,
        ).with_salience(0.3)
        f2 = Fiber.create(
            neuron_ids={n2.id},
            synapse_ids=set(),
            anchor_neuron_id=n2.id,
        ).with_salience(0.8)

        await storage.add_fiber(f1)
        await storage.add_fiber(f2)

        results = await storage.get_fibers(
            limit=10, order_by="salience", descending=True
        )
        assert len(results) == 2
        assert results[0].salience >= results[1].salience
