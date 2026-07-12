"""U3 backfill: smem_lifecycle action=backfill_supersession.

Retroactively stamps A-side supersession lineage for facts that were marked
_superseded (C-side) before U3 existed. Idempotent; ambiguous fibers are skipped.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.mcp.tool_handlers import ToolHandler
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="backfill")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


def _handler(storage: InMemoryStorage) -> ToolHandler:
    handler = ToolHandler()

    async def _gs() -> InMemoryStorage:
        return storage

    handler.get_storage = _gs  # type: ignore[method-assign]
    return handler


async def _fact(
    store: InMemoryStorage, hint: str, *, superseded: bool = False
) -> tuple[Neuron, Fiber]:
    meta = {"_superseded": True} if superseded else None
    neuron = Neuron.create(
        type=NeuronType.CONCEPT, content=f"fact-{hint}", neuron_id=str(uuid4()), metadata=meta
    )
    await store.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=hint,
    )
    await store.add_fiber(fiber)
    await store.add_typed_memory(
        TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT)
    )
    return neuron, fiber


async def _contradicts(store: InMemoryStorage, new_anchor: str, old_anchor: str) -> None:
    await store.add_synapse(
        Synapse.create(
            source_id=new_anchor,
            target_id=old_anchor,
            type=SynapseType.CONTRADICTS,
            weight=0.8,
        )
    )


class TestBackfillSupersession:
    async def test_backfills_and_is_idempotent(self, storage: InMemoryStorage) -> None:
        old_n, old_f = await _fact(storage, "oslo", superseded=True)
        new_n, new_f = await _fact(storage, "bergen")
        await _contradicts(storage, new_n.id, old_n.id)

        handler = _handler(storage)
        res = await handler._lifecycle({"action": "backfill_supersession"})

        assert res["backfilled"] == 1
        assert res["skipped_ambiguous"] == 0
        tm = await storage.get_typed_memory(old_f.id)
        assert tm is not None
        assert tm.superseded_by == new_f.id
        assert tm.valid_until is not None
        supersedes = await storage.get_synapses(source_id=new_n.id, target_id=old_n.id)
        assert any(s.type == SynapseType.SUPERSEDES for s in supersedes)

        # Second run is a no-op.
        res2 = await handler._lifecycle({"action": "backfill_supersession"})
        assert res2["backfilled"] == 0
        assert res2["already_linked"] == 1

    async def test_non_superseded_neuron_ignored(self, storage: InMemoryStorage) -> None:
        old_n, old_f = await _fact(storage, "oslo")  # NOT marked _superseded
        new_n, _ = await _fact(storage, "bergen")
        await _contradicts(storage, new_n.id, old_n.id)

        res = await _handler(storage)._lifecycle({"action": "backfill_supersession"})
        assert res["scanned"] == 0
        assert res["backfilled"] == 0
        tm = await storage.get_typed_memory(old_f.id)
        assert tm is not None
        assert tm.superseded_by is None

    async def test_ambiguous_old_anchor_skipped(self, storage: InMemoryStorage) -> None:
        shared = Neuron.create(
            type=NeuronType.CONCEPT,
            content="shared",
            neuron_id=str(uuid4()),
            metadata={"_superseded": True},
        )
        await storage.add_neuron(shared)
        for i in range(2):
            fib = Fiber.create(
                neuron_ids={shared.id},
                synapse_ids=set(),
                anchor_neuron_id=shared.id,
                summary=f"dup {i}",
            )
            await storage.add_fiber(fib)
            await storage.add_typed_memory(
                TypedMemory.create(fiber_id=fib.id, memory_type=MemoryType.FACT)
            )
        new_n, _ = await _fact(storage, "winner")
        await _contradicts(storage, new_n.id, shared.id)

        res = await _handler(storage)._lifecycle({"action": "backfill_supersession"})
        assert res["scanned"] == 1
        assert res["backfilled"] == 0
        assert res["skipped_ambiguous"] == 1
