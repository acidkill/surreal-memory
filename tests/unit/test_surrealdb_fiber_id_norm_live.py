"""Live-DB regression for the SurrealDB fiber-id normalization fix (UB2).

A Fiber loaded from SurrealDB carries the underscore-sanitized record-id form, but
typed_memory.fiber_id is stored as the original dash uuid. get_typed_memory /
get_typed_memories_batch previously matched the dash field only, so recall-time
enrichment (sources map, trust map) — which iterates loaded (underscore) fiber ids
— resolved nothing. The fix matches on the sanitized record id, so both forms work.
Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType

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
    brain = Brain.create(name="ub2-fiber-id-norm-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    try:
        await store.close()
    except Exception:
        pass


class TestFiberIdNormalization:
    async def test_typed_memory_resolves_both_id_forms(self, storage) -> None:  # type: ignore[no-untyped-def]
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="fid-normalization content")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="fid normalization fiber",
        )
        await storage.add_fiber(fiber)
        await storage.add_typed_memory(
            TypedMemory.create(
                fiber_id=fiber.id, memory_type=MemoryType.FACT, trust_score=0.9
            )
        )

        # Dash form (the original id).
        got_dash = await storage.get_typed_memory(fiber.id)
        assert got_dash is not None

        # Underscore form, exactly as a Fiber loaded from SurrealDB carries it.
        loaded = await storage.get_fiber(fiber.id)
        assert loaded is not None
        got_underscore = await storage.get_typed_memory(loaded.id)
        assert got_underscore is not None  # UB2: previously None

        # Batch lookup with the loaded (underscore) id must resolve + be keyed by it.
        batch = await storage.get_typed_memories_batch([loaded.id])
        assert loaded.id in batch
        assert batch[loaded.id].trust_score == 0.9
