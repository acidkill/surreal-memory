"""Issue #36: get_fibers(exclude_expired=True) drops soft-forgotten memories.

Soft-forget sets typed_memory.expires_at=now; recall must exclude such fibers
immediately (before consolidation cleanup) instead of only under hard delete.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _make_store() -> SurrealDBStorage:
    """A SurrealDBStorage with _query/_get_brain_id stubbed (no real connection)."""
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._query = AsyncMock(return_value=[])  # type: ignore[method-assign]
    store._get_brain_id = lambda: "default"  # type: ignore[method-assign,assignment]
    return store


@pytest.mark.asyncio
async def test_exclude_expired_adds_typed_memory_filter() -> None:
    store = _make_store()
    await store.get_fibers(limit=5, exclude_expired=True)
    sql = store._query.call_args.args[0]  # type: ignore[attr-defined]
    assert "typed_memory" in sql
    assert "expires_at" in sql
    assert "time::now()" in sql


@pytest.mark.asyncio
async def test_default_keeps_all_fibers() -> None:
    store = _make_store()
    await store.get_fibers(limit=5)
    sql = store._query.call_args.args[0]  # type: ignore[attr-defined]
    assert "typed_memory" not in sql
    assert "expires_at" not in sql


# ---------------------------------------------------------------------------
# In-memory + SQLite backends: real (non-mock) exclude_expired behaviour.
# ---------------------------------------------------------------------------


async def _add_fiber(storage: NeuralStorage, content: str) -> Fiber:
    """Create an anchor neuron + fiber and persist them."""
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=content,
    )
    await storage.add_fiber(fiber)
    return fiber


async def _mark_expiry(storage: NeuralStorage, fiber_id: str, expires_in_days: int) -> None:
    """Attach a typed_memory with a relative expiry (negative == already expired)."""
    typed = TypedMemory.create(
        fiber_id=fiber_id,
        memory_type=MemoryType.FACT,
        priority=Priority.NORMAL,
        expires_in_days=expires_in_days,
    )
    await storage.add_typed_memory(typed)


@pytest.fixture
async def in_memory() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="test_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


@pytest.fixture
async def sqlite() -> AsyncIterator[InMemoryStorage]:
    storage = InMemoryStorage()
    brain = Brain.create(name="test_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    await storage.close()


async def _assert_exclude_expired(storage: NeuralStorage) -> None:
    expired = await _add_fiber(storage, "expired memory")
    fresh = await _add_fiber(storage, "fresh memory")
    untyped = await _add_fiber(storage, "untyped memory")
    await _mark_expiry(storage, expired.id, expires_in_days=-1)  # past -> expired
    await _mark_expiry(storage, fresh.id, expires_in_days=30)  # future -> live

    # Default keeps everything (no soft-forget filtering).
    all_ids = {f.id for f in await storage.get_fibers(limit=100)}
    assert all_ids == {expired.id, fresh.id, untyped.id}

    # exclude_expired drops only the soft-forgotten fiber; live and un-typed stay.
    kept_ids = {f.id for f in await storage.get_fibers(limit=100, exclude_expired=True)}
    assert expired.id not in kept_ids
    assert fresh.id in kept_ids
    assert untyped.id in kept_ids


@pytest.mark.asyncio
async def test_in_memory_get_fibers_excludes_expired(in_memory: InMemoryStorage) -> None:
    await _assert_exclude_expired(in_memory)


@pytest.mark.asyncio
async def test_sqlite_get_fibers_excludes_expired(sqlite: InMemoryStorage) -> None:
    await _assert_exclude_expired(sqlite)
