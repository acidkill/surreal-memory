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
        if "SELECT id, neuron_a, neuron_b, binding_strength" in sql:
            if "id > type::record('co_activations', $after_id)" in sql:
                a, b, after_id = (
                    str(params["after_a"]),
                    str(params["after_b"]),
                    str(params["after_id"]),
                )
                rows = [
                    row
                    for row in self.rows
                    if row["neuron_a"] == a
                    and row["neuron_b"] == b
                    and row["id"].split(":", 1)[-1] > after_id
                ]
            elif "neuron_a = $after_a AND neuron_b > $after_b" in sql:
                a, b = str(params["after_a"]), str(params["after_b"])
                rows = [row for row in self.rows if row["neuron_a"] == a and row["neuron_b"] > b]
            elif "neuron_a > $after_a" in sql:
                a = str(params["after_a"])
                rows = [row for row in self.rows if row["neuron_a"] > a]
            else:
                rows = list(self.rows)
            rows.sort(key=lambda row: (row["neuron_a"], row["neuron_b"], row["id"]))
            return rows[: int(params["limit"])]
        if "SELECT id FROM co_activations" in sql:
            return []
        return []

    def _ensure_conn(self) -> Any:
        raise AssertionError("no delete expected in this test")


@pytest.mark.asyncio
async def test_co_activation_aggregate_iterator_uses_bounded_raw_event_pages() -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    storage = _ActivityStorage(
        [
            {"id": "co_activations:e1", "neuron_a": "a", "neuron_b": "b", "binding_strength": 0.3},
            {"id": "co_activations:e2", "neuron_a": "a", "neuron_b": "b", "binding_strength": 0.6},
            {"id": "co_activations:e3", "neuron_a": "a", "neuron_b": "b", "binding_strength": 0.6},
            {"id": "co_activations:e4", "neuron_a": "a", "neuron_b": "c", "binding_strength": 0.8},
            {"id": "co_activations:e5", "neuron_a": "b", "neuron_b": "c", "binding_strength": 0.6},
            {"id": "co_activations:e6", "neuron_a": "b", "neuron_b": "c", "binding_strength": 0.8},
        ]
    )

    results = [
        row
        async for row in storage.iter_co_activation_counts(
            since=now - timedelta(days=7),
            until=now,
            min_count=2,
            page_size=2,
        )
    ]

    assert results == [
        ("a", "b", 3, 0.5),
        ("b", "c", 2, 0.7),
    ]
    assert len(storage.queries) == 10
    first_sql, first_params = storage.queries[0]
    second_sql, second_params = storage.queries[1]
    assert "GROUP BY" not in first_sql
    assert "ORDER BY neuron_a, neuron_b, id" in first_sql
    assert first_params["limit"] == 2
    assert second_params["after_a"] == "a"
    assert second_params["after_b"] == "b"
    assert second_params["after_id"] == "e2"
    assert "id > type::record('co_activations', $after_id)" in second_sql
    assert "neuron_b != $after_b" in storage.queries[2][0]
    assert "neuron_a != $after_a" in storage.queries[3][0]
    assert first_params["since"] < first_params["until"]


@pytest.mark.asyncio
async def test_co_activation_iterator_resumes_after_complete_pair() -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    storage = _ActivityStorage(
        [
            {"id": "co_activations:e1", "neuron_a": "a", "neuron_b": "b", "binding_strength": 0.3},
            {"id": "co_activations:e2", "neuron_a": "a", "neuron_b": "c", "binding_strength": 0.8},
            {"id": "co_activations:e3", "neuron_a": "b", "neuron_b": "c", "binding_strength": 0.6},
        ]
    )

    results = [
        row
        async for row in storage.iter_co_activation_counts(
            since=now - timedelta(days=7),
            until=now,
            after_pair=("a", "b"),
            page_size=5,
        )
    ]

    assert results == [("a", "c", 1, 0.8), ("b", "c", 1, 0.6)]
    assert len(storage.queries) == 2
    assert storage.queries[0][1]["after_a"] == "a"
    assert storage.queries[0][1]["after_b"] == "b"
