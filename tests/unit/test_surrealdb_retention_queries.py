"""SurrealDB retention queries use portable bound datetime cutoffs."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow


@pytest.mark.asyncio
async def test_decay_retention_binds_datetime_instead_of_unsupported_time_ago() -> None:
    store = object.__new__(SurrealDBStorage)
    store._get_brain_id = lambda: "default"  # type: ignore[method-assign]
    store._count_decay_passes = AsyncMock(side_effect=[3, 1])  # type: ignore[method-assign]
    store._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

    before = utcnow()
    assert await store.prune_decay_passes(retention_days=90, max_records=2) == 2
    after = utcnow()

    first = store._query.await_args_list[0]
    assert "ran_at < $cutoff" in first.args[0]
    assert "time::ago" not in first.args[0]
    assert isinstance(first.kwargs["cutoff"], datetime)
    assert before - timedelta(days=90) <= first.kwargs["cutoff"] <= after - timedelta(days=90)


@pytest.mark.asyncio
async def test_synced_change_retention_reuses_same_bound_cutoff() -> None:
    store = object.__new__(SurrealDBStorage)
    store._get_brain_id = lambda: "default"  # type: ignore[method-assign]
    store._query = AsyncMock(side_effect=[[{"c": 2}], []])  # type: ignore[method-assign]

    assert await store.prune_synced_changes(older_than_days=30) == 2

    count, delete = store._query.await_args_list
    assert "changed_at < $cutoff" in count.args[0]
    assert "time::ago" not in count.args[0]
    assert isinstance(count.kwargs["cutoff"], datetime)
    assert delete.kwargs["cutoff"] == count.kwargs["cutoff"]
