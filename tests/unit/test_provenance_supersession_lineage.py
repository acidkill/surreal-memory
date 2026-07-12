"""U3 lineage: smem_provenance trace walks the SUPERSEDES chain both directions.

Builds a real supersession chain A <- B <- C on InMemoryStorage via
engine.supersession and asserts the provenance trace surfaces both ancestors
(what a fact superseded) and descendants (what superseded it), with a cycle guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.supersession import supersede_typed_memory
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="lineage")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


async def _fact(store: InMemoryStorage, hint: str) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"fact-{hint}")
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
    return fiber


def _make_handler() -> MagicMock:
    from surreal_memory.mcp.tool_handlers import ToolHandler

    handler = MagicMock(spec=ToolHandler)
    handler._provenance = ToolHandler._provenance.__get__(handler, type(handler))
    handler._provenance_trace = ToolHandler._provenance_trace.__get__(handler, type(handler))
    handler._walk_supersedes = ToolHandler._walk_supersedes.__get__(handler, type(handler))
    return handler


class TestSupersessionLineage:
    async def test_trace_middle_of_chain(self, storage: InMemoryStorage) -> None:
        a = await _fact(storage, "oslo")
        b = await _fact(storage, "bergen")
        c = await _fact(storage, "tromso")
        # A <- B <- C  (B supersedes A; C supersedes B)
        await supersede_typed_memory(
            storage,
            old_fiber_id=a.id,
            new_fiber_id=b.id,
            new_anchor_id=b.anchor_neuron_id,
            old_anchor_id=a.anchor_neuron_id,
            reason="moved to bergen",
        )
        await supersede_typed_memory(
            storage,
            old_fiber_id=b.id,
            new_fiber_id=c.id,
            new_anchor_id=c.anchor_neuron_id,
            old_anchor_id=b.anchor_neuron_id,
            reason="moved to tromso",
        )

        handler = _make_handler()
        handler.get_storage = AsyncMock(return_value=storage)
        res = await handler._provenance({"action": "trace", "neuron_id": b.anchor_neuron_id})

        assert res["is_superseded"] is True
        assert res["supersedes_count"] == 1
        superseded_by = [e for e in res["provenance"] if e["type"] == "superseded_by"]
        supersedes = [e for e in res["provenance"] if e["type"] == "supersedes"]
        assert [e["neuron_id"] for e in superseded_by] == [c.anchor_neuron_id]
        assert [e["neuron_id"] for e in supersedes] == [a.anchor_neuron_id]
        assert supersedes[0]["reason"] == "moved to bergen"

    async def test_cycle_guard_terminates(self, storage: InMemoryStorage) -> None:
        # Pathological A <-> B cycle must not loop forever.
        a = await _fact(storage, "a")
        b = await _fact(storage, "b")
        for src, tgt in ((a, b), (b, a)):
            await storage.add_synapse(
                Synapse.create(
                    source_id=src.anchor_neuron_id,
                    target_id=tgt.anchor_neuron_id,
                    type=SynapseType.SUPERSEDES,
                    weight=1.0,
                )
            )
        handler = _make_handler()
        handler.get_storage = AsyncMock(return_value=storage)
        res = await handler._provenance({"action": "trace", "neuron_id": a.anchor_neuron_id})
        # Terminates; each direction advances at most once before revisiting.
        assert res["is_superseded"] is True
        assert res["supersedes_count"] >= 1
