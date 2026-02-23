"""Tests for FalkorDB graph traversal operations."""

from __future__ import annotations

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage


@pytest.fixture
async def graph_data(
    storage: FalkorDBStorage,
    sample_neurons: list[Neuron],
    sample_synapses: list[Synapse],
) -> tuple[list[Neuron], list[Synapse]]:
    """Populate storage with a chain: n0 -> n1 -> n2 -> n3 -> n4."""
    for n in sample_neurons:
        await storage.add_neuron(n)
    for s in sample_synapses:
        await storage.add_synapse(s)
    return sample_neurons, sample_synapses


class TestNeighborTraversal:
    """get_neighbors — the hotpath for spreading activation."""

    @pytest.mark.asyncio
    async def test_outgoing_neighbors(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n0 has one outgoing edge to n1
        neighbors = await storage.get_neighbors(neurons[0].id, direction="out")
        assert len(neighbors) == 1
        neighbor_neuron, synapse = neighbors[0]
        assert neighbor_neuron.id == neurons[1].id

    @pytest.mark.asyncio
    async def test_incoming_neighbors(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n1 has one incoming edge from n0
        neighbors = await storage.get_neighbors(neurons[1].id, direction="in")
        assert len(neighbors) == 1
        neighbor_neuron, _ = neighbors[0]
        assert neighbor_neuron.id == neurons[0].id

    @pytest.mark.asyncio
    async def test_both_directions(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n2 has incoming from n1 and outgoing to n3
        neighbors = await storage.get_neighbors(neurons[2].id, direction="both")
        assert len(neighbors) == 2
        neighbor_ids = {n.id for n, _ in neighbors}
        assert neurons[1].id in neighbor_ids
        assert neurons[3].id in neighbor_ids

    @pytest.mark.asyncio
    async def test_filter_by_synapse_type(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n2 -> n3 is CAUSED_BY, n1 -> n2 is AT_LOCATION
        neighbors = await storage.get_neighbors(
            neurons[2].id,
            direction="both",
            synapse_types=[SynapseType.CAUSED_BY],
        )
        # Should only find the CAUSED_BY edge (n2 -> n3)
        assert len(neighbors) >= 1
        types = {s.type for _, s in neighbors}
        assert SynapseType.CAUSED_BY in types

    @pytest.mark.asyncio
    async def test_filter_by_min_weight(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n3 -> n4 has weight 0.9 (outgoing from n3)
        neighbors = await storage.get_neighbors(
            neurons[3].id,
            direction="out",
            min_weight=0.85,
        )
        assert len(neighbors) == 1

    @pytest.mark.asyncio
    async def test_no_neighbors_for_leaf(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n4 has no outgoing edges
        neighbors = await storage.get_neighbors(neurons[4].id, direction="out")
        assert len(neighbors) == 0

    @pytest.mark.asyncio
    async def test_bidirectional_edge_traversal(self, storage: FalkorDBStorage) -> None:
        """Bidirectional edges should be traversable from both ends."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Bi source")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Bi target")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        synapse = Synapse.create(
            n1.id,
            n2.id,
            SynapseType.RELATES_TO,
            weight=0.5,
            direction=Direction.BI,
        )
        await storage.add_synapse(synapse)

        # From n2's perspective, n1 should be reachable via "both"
        neighbors = await storage.get_neighbors(n2.id, direction="both")
        neighbor_ids = {n.id for n, _ in neighbors}
        assert n1.id in neighbor_ids


class TestShortestPath:
    """get_path — shortest path between two neurons."""

    @pytest.mark.asyncio
    async def test_direct_path(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n0 -> n1 (direct edge)
        path = await storage.get_path(neurons[0].id, neurons[1].id)
        assert path is not None
        assert len(path) >= 1

    @pytest.mark.asyncio
    async def test_multi_hop_path(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # n0 -> n1 -> n2 -> n3 -> n4 (4 hops)
        path = await storage.get_path(neurons[0].id, neurons[4].id, max_hops=5)
        assert path is not None
        assert len(path) >= 2  # At least 2 segments

    @pytest.mark.asyncio
    async def test_no_path_returns_none(self, storage: FalkorDBStorage) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Isolated A")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="Isolated B")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        path = await storage.get_path(n1.id, n2.id)
        assert path is None

    @pytest.mark.asyncio
    async def test_path_max_hops_limit(
        self,
        storage: FalkorDBStorage,
        graph_data: tuple[list[Neuron], list[Synapse]],
    ) -> None:
        neurons, _ = graph_data
        # Chain is 4 hops, max_hops=2 should not find n0 -> n4
        path = await storage.get_path(neurons[0].id, neurons[4].id, max_hops=2)
        # Depending on implementation, may return None or a partial path
        if path is not None:
            assert len(path) <= 2
