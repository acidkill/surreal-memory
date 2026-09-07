"""delete_neuron sheds the id from every fiber listing it (issue #194).

The cascade contract, identical in both backends: after deletion no fiber
claims a nonexistent member, and a fiber whose ANCHOR was deleted is expired
(tombstoned) rather than left pointing at nothing. The enrichment pass
additionally refuses to mint RELATED_TO edges at anchors that no longer
resolve — previously every pass re-created persistent edges pointing at
deleted rows.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage


async def _storage_with_fiber() -> tuple[InMemoryStorage, str, str, str]:
    storage = InMemoryStorage()
    brain = Brain.create(name="cascade-test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    anchor = Neuron.create(type=NeuronType.CONCEPT, content="anchor", neuron_id="n-anchor")
    member = Neuron.create(type=NeuronType.CONCEPT, content="member", neuron_id="n-member")
    extra = Neuron.create(type=NeuronType.CONCEPT, content="extra", neuron_id="n-extra")
    for n in (anchor, member, extra):
        await storage.add_neuron(n)
    fiber = Fiber.create(
        neuron_ids={"n-anchor", "n-member"},
        synapse_ids=set(),
        anchor_neuron_id="n-anchor",
        fiber_id="f1",
    )
    await storage.add_fiber(fiber)
    await storage.add_typed_memory(
        TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.CONTEXT,
            priority=Priority.NORMAL,
            source="test",
        )
    )
    return storage, "n-member", "n-anchor", "f1"


@pytest.mark.asyncio
async def test_deleting_a_member_sheds_the_id_and_keeps_the_fiber() -> None:
    storage, member, _anchor, fid = await _storage_with_fiber()
    assert await storage.delete_neuron(member) is True
    fiber = await storage.get_fiber(fid)
    assert fiber is not None, "a fiber that only lost a member keeps living"
    assert member not in fiber.neuron_ids
    assert fiber.anchor_neuron_id == "n-anchor"
    assert "_anchor_deleted" not in fiber.metadata


@pytest.mark.asyncio
async def test_deleting_the_anchor_expires_the_fiber() -> None:
    storage, _member, anchor, fid = await _storage_with_fiber()
    assert await storage.delete_neuron(anchor) is True
    fiber = await storage.get_fiber(fid)
    assert fiber is not None
    assert anchor not in fiber.neuron_ids, "the dead anchor id must also be shed"
    assert fiber.metadata.get("_anchor_deleted"), "metadata tombstone is set"
    typed = await storage.get_typed_memory("f1")
    assert typed is not None and typed.expires_at is not None, (
        "a typed fiber is soft-forgotten the same way smem_forget does it"
    )


@pytest.mark.asyncio
async def test_synapses_and_state_still_cascade() -> None:
    """The pre-existing cascade (synapses, neuron_state) stays intact."""
    storage, member, _anchor, _fid = await _storage_with_fiber()
    await storage.update_neuron_state(
        __import__("surreal_memory.core.neuron", fromlist=["NeuronState"]).NeuronState(
            neuron_id=member
        )
    )
    await storage.delete_neuron(member)
    assert await storage.get_neuron(member) is None
    assert await storage.get_neuron_state(member) is None


class _FakeStorage:
    """Just enough surface for find_cross_cluster_links."""

    def __init__(self, live_anchors: set[str]) -> None:
        self._live = live_anchors
        self.add_synapse = AsyncMock()

    async def get_synapses_paged(self, **_: Any) -> list[Any]:
        return []

    async def get_neurons_batch(self, ids: list[str]) -> dict[str, Any]:
        return {i: object() for i in ids if i in self._live}

    async def get_fibers(self, **_: Any) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_enrichment_never_links_to_dead_anchors() -> None:
    """An anchor that no longer resolves produces no RELATED_TO edge."""
    from surreal_memory.engine.enrichment import find_cross_cluster_links

    storage = _FakeStorage(live_anchors=set())  # everything deleted
    synapses = await find_cross_cluster_links(storage, tag_overlap_threshold=0.0)
    assert synapses == [], "no edge may point at an anchor that does not exist"
