"""U3 manual hook: smem_conflicts resolve keep_new stamps A-side lineage.

A user-driven ``keep_new`` resolution must not only mark the old anchor neuron
``_superseded`` (C-side) but also stamp the fiber's A-side validity so recall
hard-filters it and provenance can trace the replacement.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.mcp.conflict_handler import ConflictHandler
from surreal_memory.storage.memory_store import InMemoryStorage

BRAIN_ID = "test-brain"


class _FakeServer(ConflictHandler):
    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage

    async def get_storage(self) -> InMemoryStorage:
        return self._storage


def _make_storage() -> InMemoryStorage:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN_ID)
    storage.disable_auto_save = lambda: None  # type: ignore[attr-defined]
    storage.enable_auto_save = lambda: None  # type: ignore[attr-defined]

    async def _batch_save() -> None:
        pass

    storage.batch_save = _batch_save  # type: ignore[attr-defined]
    return storage


async def _fact(storage: InMemoryStorage, content: str) -> tuple[Neuron, Fiber]:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content, neuron_id=str(uuid4()))
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=content,
    )
    await storage.add_fiber(fiber)
    await storage.add_typed_memory(
        TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT)
    )
    return neuron, fiber


class TestManualKeepNewSupersession:
    async def test_keep_new_stamps_a_side_lineage(self) -> None:
        storage = _make_storage()
        server = _FakeServer(storage)

        old_neuron, old_fiber = await _fact(storage, "Emma lives in Oslo")
        new_neuron, new_fiber = await _fact(storage, "Emma lives in Bergen")
        # mark old disputed + CONTRADICTS new -> old
        disputed = old_neuron.with_metadata(_disputed=True)
        await storage.update_neuron(disputed)
        synapse = Synapse.create(
            source_id=new_neuron.id,
            target_id=old_neuron.id,
            type=SynapseType.CONTRADICTS,
            weight=0.8,
        )
        await storage.add_synapse(synapse)

        result = await server._conflicts(
            {"action": "resolve", "neuron_id": old_neuron.id, "resolution": "keep_new"}
        )
        assert result["success"] is True

        # A-side lineage stamped on the OLD fiber's typed_memory.
        tm = await storage.get_typed_memory(old_fiber.id)
        assert tm is not None
        assert tm.superseded_by == new_fiber.id
        assert tm.valid_until is not None

        # SUPERSEDES synapse new_anchor -> old_anchor.
        supersedes = await storage.get_synapses(
            source_id=new_neuron.id, target_id=old_neuron.id
        )
        assert any(s.type == SynapseType.SUPERSEDES for s in supersedes)

    async def test_keep_new_without_fibers_is_still_ok(self) -> None:
        # No fibers/typed_memories → lineage is a safe no-op; resolution succeeds.
        storage = _make_storage()
        server = _FakeServer(storage)
        old = Neuron.create(type=NeuronType.CONCEPT, content="old", neuron_id=str(uuid4()))
        await storage.add_neuron(old.with_metadata(_disputed=True))
        new = Neuron.create(type=NeuronType.CONCEPT, content="new", neuron_id=str(uuid4()))
        await storage.add_neuron(new)
        synapse = Synapse.create(
            source_id=new.id, target_id=old.id, type=SynapseType.CONTRADICTS, weight=0.8
        )
        await storage.add_synapse(synapse)

        result = await server._conflicts(
            {"action": "resolve", "neuron_id": old.id, "resolution": "keep_new"}
        )
        assert result["success"] is True
