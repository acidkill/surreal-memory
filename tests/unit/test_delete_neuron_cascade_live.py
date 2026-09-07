"""Live-DB regression: delete_neuron cascades to fibers (issue #194).

The unit tests pin the contract on the in-memory backend; this file proves the
SurrealQL cascade on the only production backend — the id is shed from
``fiber.neuron_ids`` everywhere, and a fiber losing its ANCHOR is
soft-forgotten (typed_memory ``expires_at`` + ``_anchor_deleted`` metadata
tombstone) instead of being left claiming a member that no longer exists.
Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing at a running SurrealDB",
)

BRAIN_NAME = "delete-cascade-live"


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name=BRAIN_NAME)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store

    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delete_sheds_member_and_soft_forgets_anchor_fiber(storage) -> None:  # type: ignore[no-untyped-def]
    anchor = Neuron.create(type=NeuronType.CONCEPT, content="anchor", neuron_id="dc-anchor")
    member = Neuron.create(type=NeuronType.CONCEPT, content="member", neuron_id="dc-member")
    await storage.add_neuron(anchor)
    await storage.add_neuron(member)
    await storage.add_fiber(
        Fiber.create(
            neuron_ids={"dc-anchor", "dc-member"},
            synapse_ids=set(),
            anchor_neuron_id="dc-anchor",
            fiber_id="dc-fiber",
        )
    )
    await storage.add_typed_memory(
        TypedMemory.create(
            fiber_id="dc-fiber",
            memory_type=MemoryType.CONTEXT,
            priority=Priority.NORMAL,
            source="test",
        )
    )

    # Member delete: id shed, fiber alive.
    assert await storage.delete_neuron("dc-member") is True
    fiber = await storage.get_fiber("dc-fiber")
    assert fiber is not None
    assert "dc-member" not in fiber.neuron_ids
    assert fiber.anchor_neuron_id == "dc-anchor"

    # Anchor delete: soft-forget + tombstone + shed.
    assert await storage.delete_neuron("dc-anchor") is True
    fiber = await storage.get_fiber("dc-fiber")
    assert fiber is not None
    assert "dc-anchor" not in fiber.neuron_ids
    assert fiber.metadata.get("_anchor_deleted"), "tombstone must persist"
    typed = await storage.get_typed_memory("dc-fiber")
    assert typed is not None and typed.expires_at is not None
