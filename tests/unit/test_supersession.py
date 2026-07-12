"""Unit tests for engine/supersession.py (U3), on InMemoryStorage."""

from __future__ import annotations

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.supersession import (
    resolve_fibers_for_neurons,
    supersede_typed_memory,
)
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="supersession")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


async def _make_fact(store: InMemoryStorage, hint: str) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"fact-{hint}")
    await store.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=f"fact {hint}",
    )
    await store.add_fiber(fiber)
    await store.add_typed_memory(TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT))
    return fiber


class TestSupersedeTypedMemory:
    async def test_supersede_sets_validity_synapse_and_metadata(
        self, storage: InMemoryStorage
    ) -> None:
        old = await _make_fact(storage, "oslo")
        new = await _make_fact(storage, "bergen")
        outcome = await supersede_typed_memory(
            storage,
            old_fiber_id=old.id,
            new_fiber_id=new.id,
            new_anchor_id=new.anchor_neuron_id,
            old_anchor_id=old.anchor_neuron_id,
            reason="moved",
        )
        assert outcome.superseded is True

        tm = await storage.get_typed_memory(old.id)
        assert tm is not None
        assert tm.is_superseded is True
        assert tm.superseded_by == new.id
        assert tm.valid_until is not None

        synapses = await storage.get_synapses(
            source_id=new.anchor_neuron_id, target_id=old.anchor_neuron_id
        )
        assert any(s.type == SynapseType.SUPERSEDES for s in synapses)

        old_neuron = await storage.get_neuron(old.anchor_neuron_id)
        assert old_neuron is not None
        assert old_neuron.metadata.get("_superseded") is True
        assert old_neuron.metadata.get("_superseded_by") == new.id

    async def test_idempotent(self, storage: InMemoryStorage) -> None:
        old = await _make_fact(storage, "oslo")
        new = await _make_fact(storage, "bergen")
        first = await supersede_typed_memory(
            storage, old.id, new.id, new.anchor_neuron_id, old.anchor_neuron_id
        )
        second = await supersede_typed_memory(
            storage, old.id, new.id, new.anchor_neuron_id, old.anchor_neuron_id
        )
        assert first.superseded is True
        assert second.superseded is False  # no-op on second call
        synapses = await storage.get_synapses(
            source_id=new.anchor_neuron_id, target_id=old.anchor_neuron_id
        )
        assert sum(1 for s in synapses if s.type == SynapseType.SUPERSEDES) == 1

    async def test_missing_old_typed_memory_is_noop(self, storage: InMemoryStorage) -> None:
        new = await _make_fact(storage, "bergen")
        outcome = await supersede_typed_memory(
            storage, "no-such-fiber", new.id, new.anchor_neuron_id, "no-such-neuron"
        )
        assert outcome.superseded is False


class TestResolveFibersForNeurons:
    async def test_unambiguous_neuron_maps(self, storage: InMemoryStorage) -> None:
        fiber = await _make_fact(storage, "x")
        mapping = await resolve_fibers_for_neurons(storage, [fiber.anchor_neuron_id])
        assert mapping.get(fiber.anchor_neuron_id) == fiber.id

    async def test_neuron_with_no_fiber_skipped(self, storage: InMemoryStorage) -> None:
        orphan = Neuron.create(type=NeuronType.CONCEPT, content="orphan")
        await storage.add_neuron(orphan)
        mapping = await resolve_fibers_for_neurons(storage, [orphan.id])
        assert orphan.id not in mapping
