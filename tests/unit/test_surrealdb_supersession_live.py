"""Live-DB regression for U3 supersession against real SurrealDB.

engine.supersession.supersede_typed_memory touches update_typed_memory,
add_synapse (SUPERSEDES) and find_fibers — the exact surfaces where the three
prior ``_to_surreal_id`` round-trip bugs (A/B/C) bit. This guards a hypothetical
fourth: the A-side validity update and the anchor->fiber resolution must persist
and re-read correctly on SurrealDB for BOTH the dash and loaded (underscore) id
forms. Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os

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

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name="u3-supersession-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    try:
        await store.close()
    except Exception:
        pass


async def _fact(storage, hint: str) -> Fiber:  # type: ignore[no-untyped-def]
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"supersession-{hint}")
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=f"supersession {hint}",
    )
    await storage.add_fiber(fiber)
    await storage.add_typed_memory(
        TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT)
    )
    return fiber


class TestSupersessionLive:
    async def test_supersede_persists_and_reads_back(self, storage) -> None:  # type: ignore[no-untyped-def]
        old = await _fact(storage, "oslo")
        new = await _fact(storage, "bergen")

        outcome = await supersede_typed_memory(
            storage,
            old_fiber_id=old.id,
            new_fiber_id=new.id,
            new_anchor_id=new.anchor_neuron_id,
            old_anchor_id=old.anchor_neuron_id,
            reason="moved",
        )
        assert outcome.superseded is True

        # A-side validity update round-trips (dash form).
        tm = await storage.get_typed_memory(old.id)
        assert tm is not None
        assert tm.superseded_by == new.id
        assert tm.valid_until is not None

        # ...and via the loaded (underscore) id form too.
        loaded_old = await storage.get_fiber(old.id)
        assert loaded_old is not None
        tm_underscore = await storage.get_typed_memory(loaded_old.id)
        assert tm_underscore is not None
        assert tm_underscore.superseded_by == new.id

        # SUPERSEDES synapse persisted new_anchor -> old_anchor.
        synapses = await storage.get_synapses(
            source_id=new.anchor_neuron_id, target_id=old.anchor_neuron_id
        )
        assert any(s.type == SynapseType.SUPERSEDES for s in synapses)

    async def test_supersede_is_idempotent(self, storage) -> None:  # type: ignore[no-untyped-def]
        old = await _fact(storage, "a")
        new = await _fact(storage, "b")
        first = await supersede_typed_memory(
            storage,
            old_fiber_id=old.id,
            new_fiber_id=new.id,
            new_anchor_id=new.anchor_neuron_id,
            old_anchor_id=old.anchor_neuron_id,
        )
        assert first.superseded is True
        second = await supersede_typed_memory(
            storage,
            old_fiber_id=old.id,
            new_fiber_id=new.id,
            new_anchor_id=new.anchor_neuron_id,
            old_anchor_id=old.anchor_neuron_id,
        )
        assert second.superseded is False

    async def test_resolve_fibers_for_neurons_live(self, storage) -> None:  # type: ignore[no-untyped-def]
        fiber = await _fact(storage, "solo")
        mapping = await resolve_fibers_for_neurons(storage, [fiber.anchor_neuron_id])
        resolved = mapping.get(fiber.anchor_neuron_id)
        assert resolved is not None
        # resolve returns the loaded (underscore-sanitized) fiber-id form, consistent
        # with get_fiber/find_fibers everywhere (the deferred Fiber.id round-trip, Bug
        # C root). It is functionally equivalent: id-agnostic get_typed_memory (UB2)
        # resolves it AND the original dash id to the same typed_memory, so supersede
        # works whichever form the hooks pass.
        tm_resolved = await storage.get_typed_memory(resolved)
        tm_dash = await storage.get_typed_memory(fiber.id)
        assert tm_resolved is not None
        assert tm_dash is not None
        assert tm_resolved.fiber_id == tm_dash.fiber_id
