"""Fiber keyset pages must not stop at the legacy top-N ceiling."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


@pytest.mark.asyncio
async def test_in_memory_fiber_keyset_crosses_10000_boundary_and_freezes_time() -> None:
    storage = InMemoryStorage()
    storage._current_brain_id = "brain-a"
    before = datetime(2026, 9, 20, tzinfo=UTC)
    for index in range(10005):
        fiber = Fiber(
            id=f"f{index:05}",
            neuron_ids={"anchor"},
            synapse_ids=set(),
            anchor_neuron_id="anchor",
            created_at=before - timedelta(days=1),
        )
        await storage.add_fiber(fiber)
    await storage.add_fiber(
        Fiber("f10005", {"anchor"}, set(), "anchor", created_at=before + timedelta(days=1))
    )
    storage._current_brain_id = "brain-b"
    await storage.add_fiber(Fiber("f10004b", {"anchor"}, set(), "anchor", created_at=before))
    storage._current_brain_id = "brain-a"

    cursor: str | None = None
    seen: list[str] = []
    while page := await storage.get_fibers_after_id(cursor, limit=1700, created_before=before):
        assert len(page) <= 1700
        seen.extend(fiber.id for fiber in page)
        cursor = page[-1].id

    assert seen == [f"f{index:05}" for index in range(10005)]
    assert len(seen) == len(set(seen))
    assert [f.id for f in await storage.get_fibers_after_id("f10000", limit=9999)] == [
        "f10001",
        "f10002",
        "f10003",
        "f10004",
        "f10005",
    ]


@pytest.mark.asyncio
async def test_surrealdb_fiber_page_uses_bound_cursor_brain_and_timestamp() -> None:
    storage = SurrealDBStorage(url="http://localhost:8001")
    storage._current_brain_id = "brain-a"
    storage._conn = AsyncMock()
    storage._query = AsyncMock(return_value=[])  # type: ignore[method-assign]
    before = datetime(2026, 9, 20, tzinfo=UTC)

    assert (
        await storage.get_fibers_after_id("fiber:f-10000", limit=9999, created_before=before) == []
    )

    sql = storage._query.await_args.args[0]
    params = storage._query.await_args.kwargs
    assert "SELECT * FROM fiber WHERE brain_id = $brain_id" in sql
    assert "id > type::record('fiber', $cursor_id)" in sql
    assert "created_at IS NONE OR created_at <= $created_before" in sql
    assert "ORDER BY id ASC LIMIT 2000" in sql
    assert params == {
        "brain_id": "brain-a",
        "cursor_id": "f_10000",
        "created_before": before,
    }
