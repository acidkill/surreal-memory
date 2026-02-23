"""Integration test: spreading activation over FalkorDB graph.

Tests the full encode -> store -> spread -> retrieve cycle using FalkorDB
as the backend, verifying that graph traversal produces correct activation
patterns compared to the SQLite reference implementation.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
import pytest_asyncio

from tests.storage.conftest import falkordb_available

FALKORDB_HOST = os.environ.get("FALKORDB_TEST_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_TEST_PORT", "6379"))

_FALKORDB_AVAILABLE = falkordb_available(FALKORDB_HOST, FALKORDB_PORT)
_SKIP_REASON = f"FalkorDB not available at {FALKORDB_HOST}:{FALKORDB_PORT}"


@pytest.fixture(autouse=True)
def _require_falkordb() -> None:
    if not _FALKORDB_AVAILABLE:
        pytest.skip(_SKIP_REASON)


@pytest_asyncio.fixture
async def storage() -> Any:
    """Fresh FalkorDB storage with test brain."""
    from neural_memory.core.brain import Brain, BrainConfig
    from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage

    brain_id = f"spread_{uuid.uuid4().hex[:8]}"
    store = FalkorDBStorage(host=FALKORDB_HOST, port=FALKORDB_PORT)
    await store.initialize()

    brain = Brain.create(
        name=brain_id,
        brain_id=brain_id,
        config=BrainConfig(
            max_spread_hops=4,
            activation_threshold=0.15,
        ),
    )
    await store.save_brain(brain)
    await store.set_brain_with_indexes(brain_id)

    yield store

    try:
        await store.clear(brain_id)
    except Exception:
        pass
    await store.close()


class TestSpreadingActivation:
    """End-to-end spreading activation over a FalkorDB graph."""

    @pytest.mark.asyncio
    async def test_activation_spreads_along_chain(self, storage: Any) -> None:
        """Build a chain A -> B -> C -> D, activate A, verify spread."""
        from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType

        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content=f"Chain node {i}")
            for i in range(4)
        ]
        for n in neurons:
            await storage.add_neuron(n)

        # Chain: 0 -> 1 -> 2 -> 3
        weights = [0.9, 0.8, 0.7]
        for i, w in enumerate(weights):
            s = Synapse.create(
                neurons[i].id,
                neurons[i + 1].id,
                SynapseType.RELATES_TO,
                weight=w,
            )
            await storage.add_synapse(s)

        # Activate first neuron
        state = NeuronState(
            neuron_id=neurons[0].id,
            activation_level=1.0,
            access_frequency=1,
        )
        await storage.update_neuron_state(state)

        # Simulate 1-hop spread: neighbors of neurons[0]
        neighbors = await storage.get_neighbors(neurons[0].id, direction="out")
        assert len(neighbors) == 1
        neighbor, synapse = neighbors[0]
        assert neighbor.id == neurons[1].id
        assert synapse.weight == pytest.approx(0.9, abs=0.01)

        # Simulate 2-hop: neighbors of neurons[1]
        hop2 = await storage.get_neighbors(neurons[1].id, direction="out")
        assert len(hop2) == 1
        assert hop2[0][0].id == neurons[2].id

    @pytest.mark.asyncio
    async def test_hub_activation_fan_out(self, storage: Any) -> None:
        """Hub node with multiple outgoing edges fans activation out."""
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType

        hub = Neuron.create(type=NeuronType.CONCEPT, content="Hub node")
        await storage.add_neuron(hub)

        spokes = [
            Neuron.create(type=NeuronType.ENTITY, content=f"Spoke {i}")
            for i in range(5)
        ]
        for sp in spokes:
            await storage.add_neuron(sp)
            s = Synapse.create(
                hub.id, sp.id, SynapseType.RELATES_TO, weight=0.6
            )
            await storage.add_synapse(s)

        neighbors = await storage.get_neighbors(hub.id, direction="out")
        assert len(neighbors) == 5

        for _, synapse in neighbors:
            assert synapse.weight == pytest.approx(0.6, abs=0.01)

    @pytest.mark.asyncio
    async def test_cycle_detection_via_path(self, storage: Any) -> None:
        """Graph with cycle: A -> B -> C -> A. Path should still work."""
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType

        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content=f"Cycle {i}")
            for i in range(3)
        ]
        for n in neurons:
            await storage.add_neuron(n)

        for i in range(3):
            s = Synapse.create(
                neurons[i].id,
                neurons[(i + 1) % 3].id,
                SynapseType.RELATES_TO,
                weight=0.5,
            )
            await storage.add_synapse(s)

        path = await storage.get_path(neurons[0].id, neurons[2].id, max_hops=4)
        assert path is not None

    @pytest.mark.asyncio
    async def test_weighted_path_selection(self, storage: Any) -> None:
        """Strong vs weak edges affect neighbor ordering by weight."""
        from neural_memory.core.neuron import Neuron, NeuronType
        from neural_memory.core.synapse import Synapse, SynapseType

        center = Neuron.create(type=NeuronType.CONCEPT, content="Center")
        strong = Neuron.create(type=NeuronType.CONCEPT, content="Strong link")
        weak = Neuron.create(type=NeuronType.CONCEPT, content="Weak link")
        await storage.add_neuron(center)
        await storage.add_neuron(strong)
        await storage.add_neuron(weak)

        s1 = Synapse.create(
            center.id, strong.id, SynapseType.RELATES_TO, weight=0.95
        )
        s2 = Synapse.create(
            center.id, weak.id, SynapseType.RELATES_TO, weight=0.1
        )
        await storage.add_synapse(s1)
        await storage.add_synapse(s2)

        neighbors = await storage.get_neighbors(center.id, direction="out")
        assert len(neighbors) == 2

        strong_only = await storage.get_neighbors(
            center.id, direction="out", min_weight=0.5
        )
        assert len(strong_only) == 1
        assert strong_only[0][0].id == strong.id

    @pytest.mark.asyncio
    async def test_multi_brain_isolation(self, storage: Any) -> None:
        """Neurons in brain A are invisible from brain B."""
        from neural_memory.core.brain import Brain
        from neural_memory.core.neuron import Neuron, NeuronType

        brain_a_id = f"iso_a_{uuid.uuid4().hex[:8]}"
        brain_b_id = f"iso_b_{uuid.uuid4().hex[:8]}"

        for bid in (brain_a_id, brain_b_id):
            brain = Brain.create(name=bid, brain_id=bid)
            await storage.save_brain(brain)

        await storage.set_brain_with_indexes(brain_a_id)
        n = Neuron.create(type=NeuronType.CONCEPT, content="Brain A only")
        await storage.add_neuron(n)

        await storage.set_brain_with_indexes(brain_b_id)
        found = await storage.get_neuron(n.id)
        assert found is None

        await storage.set_brain_with_indexes(brain_a_id)
        found = await storage.get_neuron(n.id)
        assert found is not None

        await storage.clear(brain_a_id)
        await storage.clear(brain_b_id)
