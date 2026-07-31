"""Regression (#112): expiry cleanup must never return a pinned memory.

get_expired_memories()/get_expired_memory_count() used to filter only on
expires_at, ignoring whether the memory's underlying fiber is pinned=true.
Pinning is the documented "never remove this" mechanism (already honored by
consolidation/orphan-pruning), but the expiry-cleanup background task
(mcp/expiry_cleanup_handler.py) deletes whatever these two methods return —
with no other guard, no confirmation, no dry-run — so this was silent,
automatic data loss of exactly what the user explicitly protected.

The test is skipped when SURREALDB_URL is unset so CI without docker still
passes.
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.utils.timeutils import utcnow
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="pinned-expiry-test-9f3a1c")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    # Best-effort: drop this test's brain (and stale leftovers) from the
    # shared DB so `smem brain list` doesn't accumulate test brains.
    try:
        await cleanup_live_brains(storage, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


async def _make_fiber(storage, idx: int, *, pinned: bool = False) -> Fiber:  # type: ignore[no-untyped-def]
    """Create the neuron + fiber that a TypedMemory row references."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"pinned-expiry-{idx}")
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=f"fiber-{idx}",
    )
    await storage.add_fiber(fiber)
    if pinned:
        fiber = dataclasses.replace(fiber, pinned=True)
        await storage.update_fiber(fiber)
    return fiber


@pytest.mark.asyncio
async def test_get_expired_memories_excludes_pinned_fiber(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """An expired memory whose fiber is pinned must be excluded from
    get_expired_memories() and not counted by get_expired_memory_count().
    """
    fiber = await _make_fiber(surrealdb_storage, idx=1, pinned=True)
    typed_mem = TypedMemory(
        fiber_id=fiber.id,
        memory_type=MemoryType.TODO,
        expires_at=utcnow() - timedelta(days=1),
    )
    await surrealdb_storage.add_typed_memory(typed_mem)

    expired = await surrealdb_storage.get_expired_memories()
    assert fiber.id not in {m.fiber_id for m in expired}
    assert await surrealdb_storage.get_expired_memory_count() == 0


@pytest.mark.asyncio
async def test_get_expired_memories_includes_unpinned_fiber(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """No regression: an expired memory whose fiber is NOT pinned is still
    returned/counted — this is the existing, correct behavior for
    non-pinned expired memories.
    """
    fiber = await _make_fiber(surrealdb_storage, idx=2, pinned=False)
    typed_mem = TypedMemory(
        fiber_id=fiber.id,
        memory_type=MemoryType.TODO,
        expires_at=utcnow() - timedelta(days=1),
    )
    await surrealdb_storage.add_typed_memory(typed_mem)

    expired = await surrealdb_storage.get_expired_memories()
    assert fiber.id in {m.fiber_id for m in expired}
    assert await surrealdb_storage.get_expired_memory_count() == 1


@pytest.mark.asyncio
async def test_get_expired_memories_includes_orphan_typed_memory(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """A typed_memory whose fiber has already been deleted (orphan — no fiber
    row to correlate against) must still be returned/counted, preserving the
    pre-fix orphan-cleanup behavior. Only an *existing* pinned fiber protects
    its memory; a missing fiber has nothing to protect.
    """
    orphan_fiber_id = str(uuid.uuid4())
    typed_mem = TypedMemory(
        fiber_id=orphan_fiber_id,
        memory_type=MemoryType.TODO,
        expires_at=utcnow() - timedelta(days=1),
    )
    # SurrealDB's add_typed_memory does not verify fiber existence (unlike
    # SQLite's), so this legitimately models a fiber deleted after its
    # typed_memory row was created.
    await surrealdb_storage.add_typed_memory(typed_mem)

    expired = await surrealdb_storage.get_expired_memories()
    assert orphan_fiber_id in {m.fiber_id for m in expired}
    assert await surrealdb_storage.get_expired_memory_count() == 1
