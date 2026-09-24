from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.activity import SurrealDBActivityMixin


class _ActivityStorage(SurrealDBActivityMixin):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[Any] = []

    def _get_brain_id(self) -> str:
        return "brain"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if "GROUP BY neuron_a, neuron_b" in sql:
            cursor = (params.get("after_a", ""), params.get("after_b", ""))
            grouped = sorted(
                (row for row in self.rows if (row["neuron_a"], row["neuron_b"]) > cursor),
                key=lambda row: (row["neuron_a"], row["neuron_b"]),
            )
            page = grouped[: int(params["limit"])]
            return [
                {
                    "neuron_a": row["neuron_a"],
                    "neuron_b": row["neuron_b"],
                    "pair_count": row["pair_count"],
                    "average_strength": row["average_strength"],
                }
                for row in page
            ]
        if "SELECT id FROM co_activations" in sql:
            return []
        return []

    def _ensure_conn(self) -> Any:
        raise AssertionError("no delete expected in this test")


@pytest.mark.asyncio
async def test_co_activation_aggregate_iterator_uses_bounded_keyset_pages() -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    storage = _ActivityStorage(
        [
            {"neuron_a": "a", "neuron_b": "b", "pair_count": 3, "average_strength": 0.5},
            {"neuron_a": "a", "neuron_b": "c", "pair_count": 5, "average_strength": 0.8},
            {"neuron_a": "b", "neuron_b": "c", "pair_count": 4, "average_strength": 0.7},
        ]
    )

    results = [
        row
        async for row in storage.iter_co_activation_counts(
            since=now - timedelta(days=7),
            until=now,
            min_count=3,
            page_size=2,
        )
    ]

    assert results == [
        ("a", "b", 3, 0.5),
        ("a", "c", 5, 0.8),
        ("b", "c", 4, 0.7),
    ]
    assert len(storage.queries) == 2
    first_sql, first_params = storage.queries[0]
    second_sql, second_params = storage.queries[1]
    assert "GROUP BY neuron_a, neuron_b" in first_sql
    assert first_params["limit"] == 2
    assert second_params["after_a"] == "a"
    assert second_params["after_b"] == "c"
    assert first_params["since"] < first_params["until"]
