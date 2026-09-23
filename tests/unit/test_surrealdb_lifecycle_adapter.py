"""The SurrealDB lifecycle column and the engine's metadata view stay aligned."""

from typing import Any

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage, _row_to_neuron


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "neuron:example",
        "type": "concept",
        "content": "example",
        "metadata": {"priority": 2, "lifecycle_state": "active"},
        "lifecycle_state": "warm",
    }
    row.update(overrides)
    return row


def test_lifecycle_reread_uses_authoritative_column_without_changing_other_metadata() -> None:
    row = _row()
    neuron = _row_to_neuron(row)

    assert neuron.metadata == {"priority": 2, "lifecycle_state": "warm"}
    assert row["metadata"] == {"priority": 2, "lifecycle_state": "active"}
    assert neuron.metadata["lifecycle_state"] == "warm"  # a second scan is a no-op
    assert _row_to_neuron(_row(lifecycle_state=None)).metadata["lifecycle_state"] == "active"


@pytest.mark.asyncio
async def test_lifecycle_update_is_brain_scoped_and_reread_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"
    row = _row()
    queries: list[tuple[str, dict[str, Any]]] = []

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        queries.append((sql, params))
        if "UPDATE" in sql and params["brain_id"] == "test-brain":
            row["lifecycle_state"] = params["lifecycle_state"]
            return [row]
        return []

    monkeypatch.setattr(store, "_query", query)
    await store.update_neuron_lifecycle("example", "cool")
    reread = _row_to_neuron(row)

    assert reread.metadata == {"priority": 2, "lifecycle_state": "cool"}
    assert queries == [
        (
            "UPDATE neuron:example SET lifecycle_state = $lifecycle_state "
            "WHERE brain_id = $brain_id RETURN AFTER",
            {"lifecycle_state": "cool", "brain_id": "test-brain"},
        )
    ]


@pytest.mark.asyncio
async def test_lifecycle_update_missing_row_or_query_failure_is_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"

    async def missing(sql: str, **params: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(store, "_query", missing)
    with pytest.raises(LookupError, match="not found in brain test-brain"):
        await store.update_neuron_lifecycle("example", "cool")

    async def failed(sql: str, **params: Any) -> list[dict[str, Any]]:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "_query", failed)
    with pytest.raises(RuntimeError, match="database unavailable"):
        await store.update_neuron_lifecycle("example", "cool")
