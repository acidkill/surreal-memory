"""The vector-index query must search at least ``_KNN_EF_MIN`` candidates.

Why a test on a constant: ``find_neurons_by_embedding`` is the only road from a
query embedding to the anchor neurons of a recall, and the HNSW search width it
asks for decides whether the true nearest neighbours are in the answer at all.
At ef=100 the index measurably dropped true top-30 neighbours on a production
brain; the floor was raised to 400 on that evidence. Pinning the SQL the store
emits keeps a future refactor from silently sliding the width back down.
"""

from __future__ import annotations

import pytest

from surreal_memory.storage.surrealdb import store as store_module
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def test_knn_search_width_floor_is_400() -> None:
    assert store_module._KNN_EF_MIN == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "expected_operator"),
    [
        (10, "<|10,400|>"),  # small k: the floor applies
        (30, "<|30,400|>"),  # the recall anchor path (EMBEDDING_ANCHOR_MIN_LIMIT)
        (300, "<|300,600|>"),  # large k: 2*k overtakes the floor, ef never < k
    ],
)
async def test_knn_query_uses_search_width_floor(limit: int, expected_operator: str) -> None:
    storage = SurrealDBStorage()
    storage._current_brain_id = "testbrain"
    captured: list[str] = []

    async def fake_query(sql: str, **params: object) -> list[dict[str, object]]:
        captured.append(sql)
        return []

    storage._query = fake_query  # type: ignore[method-assign]

    rows = await storage.find_neurons_by_embedding([1.0, 0.0, 0.0, 0.0], limit=limit)

    assert rows == []
    assert len(captured) == 1
    assert expected_operator in captured[0], captured[0]
