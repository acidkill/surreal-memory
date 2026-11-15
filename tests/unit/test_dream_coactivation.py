"""DREAM must only relate neurons coactivated by the same exploration."""

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.dream import dream
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest.mark.asyncio
async def test_independent_seed_explorations_do_not_create_connection() -> None:
    store = InMemoryStorage()
    brain = Brain.create(name="dream-isolated")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    try:
        for neuron_id in ("isolated-a", "isolated-b"):
            await store.add_neuron(
                Neuron.create(NeuronType.CONCEPT, neuron_id, neuron_id=neuron_id)
            )
        result = await dream(store, BrainConfig(dream_neuron_count=2), seed=42)
        assert result.synapses_created == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_same_exploration_can_connect_nonadjacent_neurons() -> None:
    from surreal_memory.core.synapse import Synapse, SynapseType

    store = InMemoryStorage()
    brain = Brain.create(name="dream-connected")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    try:
        for neuron_id in ("a", "b", "c"):
            await store.add_neuron(
                Neuron.create(NeuronType.CONCEPT, neuron_id, neuron_id=neuron_id)
            )
        for source, target in (("a", "b"), ("b", "c")):
            await store.add_synapse(
                Synapse.create(source, target, SynapseType.RELATED_TO, weight=1)
            )
        result = await dream(store, BrainConfig(dream_neuron_count=3), seed=42)
        assert {s.source_id for s in result.synapses_created} | {
            s.target_id for s in result.synapses_created
        } >= {"a", "c"}
        assert any({s.source_id, s.target_id} == {"a", "c"} for s in result.synapses_created)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_checkpoint_resume_and_corruption_rejection() -> None:
    from dataclasses import replace

    from surreal_memory.core.synapse import Synapse, SynapseType
    from surreal_memory.engine.dream import DreamPlanCheckpoint

    store = InMemoryStorage()
    brain = Brain.create(name="dream-checkpoints")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    snapshots: list[DreamPlanCheckpoint] = []
    try:
        for i in range(15):
            neuron_id = f"n{i:02}"
            await store.add_neuron(
                Neuron.create(NeuronType.CONCEPT, neuron_id, neuron_id=neuron_id)
            )
        for i in range(1, 15):
            await store.add_synapse(
                Synapse.create("n00", f"n{i:02}", SynapseType.RELATED_TO, weight=1)
            )

        async def save(state: DreamPlanCheckpoint) -> None:
            snapshots.append(state)

        result = await dream(store, BrainConfig(dream_neuron_count=15), seed=42, checkpoint=save)
        assert {state.pair_cursor for state in snapshots} >= {0, 100, 105}
        pairs = {(s.source_id, s.target_id) for s in result.synapses_created}
        for cursor in (0, 100, 105):
            state = next(item for item in snapshots if item.pair_cursor == cursor)
            resumed = await dream(store, BrainConfig(), resume=state)
            assert {(s.source_id, s.target_id) for s in resumed.synapses_created} == pairs

        state = snapshots[-1]
        for bad in ((-1,), (4999,), (True,), (0, 0), (2, 1)):
            with pytest.raises(RuntimeError):
                await dream(store, BrainConfig(), resume=replace(state, eligible_pair_ordinals=bad))
        with pytest.raises(RuntimeError):
            await dream(store, BrainConfig(), resume=replace(state, eligible_pair_ordinals=None))
        with pytest.raises(RuntimeError):
            await dream(store, BrainConfig(), resume=replace(state, pair_receipts=("bad",)))
        with pytest.raises(RuntimeError):
            await dream(store, BrainConfig(), resume=replace(state, eligible_pair_ordinals=()))
    finally:
        await store.close()
