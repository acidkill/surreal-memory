"""Live-DB regression for the typed-memory delete id-form mismatch (BUG-006).

``find_fibers`` returns ``Fiber.id`` in the ``_to_surreal_id``-folded form (the
SurrealDB record name, dashes folded to underscores), while
``typed_memory.fiber_id`` keeps the original dashed uuid. ``delete_typed_memory``
used to match the dashed *field* only, so a fiber id round-tripped through
``find_fibers``/``get_fiber`` deleted nothing, returned ``False`` and raised no
error — leaving an orphan ``typed_memory`` row behind forever (``smem_forget``
with ``hard=true`` still deleted the fiber, so the orphan became unreachable).
The fix matches on the sanitized record id, exactly like ``get_typed_memory``,
so both id forms resolve. Skipped unless SURREALDB_URL points at a running
SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name="bug006-tm-delete-id-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await store.close()
    except Exception:
        pass


async def _make_memory(storage, summary: str) -> Fiber:  # type: ignore[no-untyped-def]
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"{summary} content")
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=summary,
    )
    await storage.add_fiber(fiber)
    await storage.add_typed_memory(
        TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT, trust_score=0.9)
    )
    return fiber


async def _typed_memory_row_count(storage) -> int:  # type: ignore[no-untyped-def]
    rows = await storage._query(
        "SELECT count() AS cnt FROM typed_memory WHERE brain_id = $brain_id GROUP ALL",
        brain_id=storage._get_brain_id(),
    )
    return int(rows[0]["cnt"]) if rows else 0


class TestDeleteTypedMemoryIdForms:
    async def test_delete_accepts_find_fibers_folded_id(self, storage) -> None:  # type: ignore[no-untyped-def]
        """A fiber id sourced from find_fibers must delete its typed_memory row."""
        fiber = await _make_memory(storage, "bug006 folded-id fiber")

        found = await storage.find_fibers(contains_neuron=fiber.anchor_neuron_id, limit=10)
        assert len(found) == 1
        folded_id = found[0].id
        # Guard the premise: the round-tripped id really is a different string.
        assert "-" in fiber.id
        assert folded_id == fiber.id.replace("-", "_")
        assert folded_id != fiber.id

        assert await storage.delete_typed_memory(folded_id) is True  # BUG-006: was False
        assert await storage.get_typed_memory(fiber.id) is None
        assert await _typed_memory_row_count(storage) == 0  # BUG-006: orphan row remained

    async def test_delete_still_accepts_original_dashed_id(self, storage) -> None:  # type: ignore[no-untyped-def]
        """The pre-existing dashed-id contract (TypedMemory.fiber_id) still holds."""
        fiber = await _make_memory(storage, "bug006 dashed-id fiber")

        assert await storage.delete_typed_memory(fiber.id) is True
        assert await storage.get_typed_memory(fiber.id) is None
        assert await _typed_memory_row_count(storage) == 0

    async def test_delete_unknown_id_returns_false_and_keeps_others(self, storage) -> None:  # type: ignore[no-untyped-def]
        """The looser match must not become an over-broad delete."""
        fiber = await _make_memory(storage, "bug006 survivor fiber")

        assert await storage.delete_typed_memory("no-such-fiber-id") is False
        assert await storage.get_typed_memory(fiber.id) is not None
        assert await _typed_memory_row_count(storage) == 1

    async def test_second_delete_of_same_memory_returns_false(self, storage) -> None:  # type: ignore[no-untyped-def]
        """Deleting twice reports the second call as a no-op, not a phantom success."""
        fiber = await _make_memory(storage, "bug006 double-delete fiber")
        loaded = await storage.get_fiber(fiber.id)
        assert loaded is not None

        assert await storage.delete_typed_memory(loaded.id) is True
        assert await storage.delete_typed_memory(loaded.id) is False
