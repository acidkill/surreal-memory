"""Live-DB pinning tests: ``Fiber.id`` must survive a save/read round trip.

``Fiber.create`` mints a dash-form uuid4; the store folds it to underscores because a
SurrealDB record name may only contain ``[A-Za-z0-9_]``. ``_row_to_neuron`` and
``_row_to_synapse`` folded it back on the way out; ``_row_to_fiber`` did not, so
``Fiber.id`` came back as a different string than it went in — and at least seventeen
files across the tree grew dual-form compensations to live with that.

These tests fail on the code before the fix (``loaded.id`` carries underscores) and
pass after it. Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]


@pytest_asyncio.fixture
async def store():
    """A fresh store scoped to its own throwaway database."""
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="fiber-id-roundtrip-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed_fiber(store: SurrealDBStorage) -> Fiber:
    anchor = Neuron.create(type=NeuronType.CONCEPT, content="round-trip anchor")
    await store.add_neuron(anchor)
    fiber = Fiber.create(
        neuron_ids={anchor.id},
        synapse_ids=set(),
        anchor_neuron_id=anchor.id,
        summary="round-trip summary",
    )
    await store.add_fiber(fiber)
    return fiber


async def test_fiber_id_round_trips_through_get_fiber(store: SurrealDBStorage) -> None:
    """``get_fiber(id).id == id`` — the identity every other read path is built on."""
    fiber = await _seed_fiber(store)
    assert "-" in fiber.id  # guard the premise: uuid4 really is dash-form

    loaded = await store.get_fiber(fiber.id)

    assert loaded is not None
    assert loaded.id == fiber.id
    assert "_" not in loaded.id


async def test_fiber_id_round_trips_through_find_fibers_and_batch(
    store: SurrealDBStorage,
) -> None:
    """The listing paths must agree with ``get_fiber``, or callers cannot join on the id."""
    fiber = await _seed_fiber(store)

    found = await store.find_fibers(contains_neuron=fiber.anchor_neuron_id, limit=10)
    assert [f.id for f in found] == [fiber.id]

    # find_fibers_batch takes NEURON ids (it fans out over find_fibers), not fiber ids.
    batch = await store.find_fibers_batch([fiber.anchor_neuron_id])
    assert [f.id for f in batch] == [fiber.id]


async def test_fiber_id_from_a_read_is_accepted_by_the_next_read(
    store: SurrealDBStorage,
) -> None:
    """Feeding a read id straight back in must resolve — the property the fold broke.

    Before the fix this was the whole bug in one line: ``get_fiber(get_fiber(x).id)``
    returned ``None``, because the id handed out was not the id accepted back.
    """
    fiber = await _seed_fiber(store)

    once = await store.get_fiber(fiber.id)
    assert once is not None
    twice = await store.get_fiber(once.id)

    assert twice is not None
    assert twice.id == once.id == fiber.id
