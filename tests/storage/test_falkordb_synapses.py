"""Tests for FalkorDB synapse storage operations."""

from __future__ import annotations

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage


@pytest.fixture
async def two_neurons(storage: FalkorDBStorage) -> tuple[Neuron, Neuron]:
    """Create two neurons in storage for synapse tests."""
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="Source neuron")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="Target neuron")
    await storage.add_neuron(n1)
    await storage.add_neuron(n2)
    return n1, n2


class TestSynapseCRUD:
    """Basic synapse create/read/update/delete."""

    @pytest.mark.asyncio
    async def test_add_and_get_synapse(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        synapse = Synapse.create(
            n1.id, n2.id, SynapseType.RELATES_TO, weight=0.7
        )
        result_id = await storage.add_synapse(synapse)

        assert result_id == synapse.id
        retrieved = await storage.get_synapse(synapse.id)
        assert retrieved is not None
        assert retrieved.source_id == n1.id
        assert retrieved.target_id == n2.id
        assert retrieved.weight == pytest.approx(0.7, abs=0.01)
        assert retrieved.type == SynapseType.RELATES_TO

    @pytest.mark.asyncio
    async def test_add_synapse_missing_neuron_raises(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, _ = two_neurons
        synapse = Synapse.create(
            n1.id, "nonexistent-id", SynapseType.RELATES_TO
        )
        with pytest.raises(ValueError):
            await storage.add_synapse(synapse)

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(
        self, storage: FalkorDBStorage
    ) -> None:
        result = await storage.get_synapse("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_synapse(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        synapse = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO, weight=0.5)
        await storage.add_synapse(synapse)

        reinforced = synapse.reinforce(delta=0.1)
        await storage.update_synapse(reinforced)

        retrieved = await storage.get_synapse(synapse.id)
        assert retrieved is not None
        assert retrieved.weight > 0.5
        assert retrieved.reinforced_count == 1

    @pytest.mark.asyncio
    async def test_delete_synapse(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        synapse = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        await storage.add_synapse(synapse)

        deleted = await storage.delete_synapse(synapse.id)
        assert deleted is True

        result = await storage.get_synapse(synapse.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(
        self, storage: FalkorDBStorage
    ) -> None:
        deleted = await storage.delete_synapse("nonexistent-id")
        assert deleted is False


class TestSynapseQuery:
    """Synapse filter/query operations."""

    @pytest.mark.asyncio
    async def test_get_synapses_by_source(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        s1 = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO, weight=0.6)
        await storage.add_synapse(s1)

        results = await storage.get_synapses(source_id=n1.id)
        assert len(results) == 1
        assert results[0].source_id == n1.id

    @pytest.mark.asyncio
    async def test_get_synapses_by_target(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        s1 = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        await storage.add_synapse(s1)

        results = await storage.get_synapses(target_id=n2.id)
        assert len(results) == 1
        assert results[0].target_id == n2.id

    @pytest.mark.asyncio
    async def test_get_synapses_by_type(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        s1 = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        s2 = Synapse.create(n2.id, n1.id, SynapseType.CAUSED_BY)
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        results = await storage.get_synapses(type=SynapseType.CAUSED_BY)
        assert len(results) == 1
        assert results[0].type == SynapseType.CAUSED_BY

    @pytest.mark.asyncio
    async def test_get_synapses_min_weight(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        s1 = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO, weight=0.3)
        s2 = Synapse.create(n2.id, n1.id, SynapseType.RELATES_TO, weight=0.8)
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        results = await storage.get_synapses(min_weight=0.5)
        assert len(results) == 1
        assert results[0].weight >= 0.5

    @pytest.mark.asyncio
    async def test_get_all_synapses(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        s1 = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        s2 = Synapse.create(n2.id, n1.id, SynapseType.CAUSED_BY)
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        results = await storage.get_all_synapses()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_synapses_for_neurons(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        s1 = Synapse.create(n1.id, n2.id, SynapseType.RELATES_TO)
        await storage.add_synapse(s1)

        grouped = await storage.get_synapses_for_neurons([n1.id], direction="out")
        assert n1.id in grouped
        assert len(grouped[n1.id]) == 1


class TestSynapseBidirectional:
    """Bidirectional synapse behavior."""

    @pytest.mark.asyncio
    async def test_bidirectional_synapse(
        self, storage: FalkorDBStorage, two_neurons: tuple[Neuron, Neuron]
    ) -> None:
        n1, n2 = two_neurons
        synapse = Synapse.create(
            n1.id,
            n2.id,
            SynapseType.RELATES_TO,
            weight=0.5,
            direction=Direction.BI,
        )
        await storage.add_synapse(synapse)

        retrieved = await storage.get_synapse(synapse.id)
        assert retrieved is not None
        assert retrieved.direction == Direction.BI
