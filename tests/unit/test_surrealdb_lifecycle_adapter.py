"""The SurrealDB lifecycle column and the engine's metadata view stay aligned."""

from dataclasses import replace
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
            "UPDATE type::record('neuron', $record_id) SET lifecycle_state = $lifecycle_state "
            "WHERE brain_id = $brain_id RETURN AFTER",
            {"record_id": "example", "lifecycle_state": "cool", "brain_id": "test-brain"},
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


@pytest.mark.asyncio
async def test_generic_update_keeps_concurrent_canonical_lifecycle_and_cleans_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"
    row = _row(metadata={"priority": 2, "lifecycle_state": "active", "source": "test"})
    queries: list[tuple[str, dict[str, Any]]] = []
    logged: list[tuple[Any, ...]] = []

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        queries.append((sql, params))
        if "SET lifecycle_state = $lifecycle_state" in sql:
            row["lifecycle_state"] = params["lifecycle_state"]
            return [dict(row)]

        assert "metadata.lifecycle_state = IF lifecycle_state IS NONE" in sql
        # Model the UPDATE expression against the row state at statement time.
        metadata = dict(params["metadata"])
        if row.get("lifecycle_state") is not None:
            metadata.pop("lifecycle_state", None)
        row.update(
            {
                "type": params["type"],
                "content": params["content"],
                "content_hash": params["content_hash"],
                "metadata": metadata,
                "ephemeral": params["ephemeral"],
                "updated_at": params["updated_at"],
            }
        )
        return [dict(row)]

    async def record_change(*args: Any, **kwargs: Any) -> None:
        logged.append(args)

    monkeypatch.setattr(store, "_query", query)
    monkeypatch.setattr(store, "_record_change_internal", record_change)

    snapshot = _row_to_neuron(row)
    assert snapshot.metadata["lifecycle_state"] == "warm"
    await store.update_neuron_lifecycle(snapshot.id, "cool")
    await store.update_neuron(replace(snapshot, content="edited content"))

    assert row["lifecycle_state"] == "cool"
    assert row["metadata"] == {"priority": 2, "source": "test"}
    assert row["content"] == "edited content"
    assert len(queries) == 2
    assert "WHERE brain_id = $brain_id RETURN AFTER" in queries[1][0]
    assert queries[1][1]["brain_id"] == "test-brain"
    assert logged[0][3].metadata == {"priority": 2, "lifecycle_state": "cool", "source": "test"}


@pytest.mark.asyncio
async def test_generic_update_preserves_legacy_lifecycle_without_canonical_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"
    row = _row(
        metadata={"priority": 2, "lifecycle_state": "archived", "source": "legacy"},
        lifecycle_state=None,
    )

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        assert "metadata.lifecycle_state = IF lifecycle_state IS NONE" in sql
        row["metadata"] = dict(params["metadata"])
        row["content"] = params["content"]
        return [dict(row)]

    async def record_change(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(store, "_query", query)
    monkeypatch.setattr(store, "_record_change_internal", record_change)

    neuron = _row_to_neuron(row)
    await store.update_neuron(replace(neuron, content="legacy row edited"))

    assert row["lifecycle_state"] is None
    assert row["metadata"] == {
        "priority": 2,
        "lifecycle_state": "archived",
        "source": "legacy",
    }


@pytest.mark.asyncio
async def test_generic_update_binds_all_digit_record_id_and_enforces_brain_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"
    row = _row(
        id="neuron:1234567890",
        brain_id="test-brain",
        metadata={"priority": 9, "source": "synthetic"},
        lifecycle_state="cool",
    )
    queries: list[tuple[str, dict[str, Any]]] = []
    logged: list[tuple[Any, ...]] = []

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        queries.append((sql, params))
        if params["brain_id"] != "test-brain":
            return []
        return [dict(row)]

    async def record_change(*args: Any, **kwargs: Any) -> None:
        logged.append(args)

    monkeypatch.setattr(store, "_query", query)
    monkeypatch.setattr(store, "_record_change_internal", record_change)

    neuron = _row_to_neuron(row)
    await store.update_neuron(replace(neuron, content="edited numeric-id neuron"))

    sql, params = queries[0]
    assert "UPDATE type::record('neuron', $record_id) SET" in sql
    assert "WHERE brain_id = $brain_id RETURN AFTER" in sql
    assert params["record_id"] == "1234567890"
    assert isinstance(params["record_id"], str)
    assert params["brain_id"] == "test-brain"
    assert logged[0][3].id == "1234567890"
    assert logged[0][3].metadata["lifecycle_state"] == "cool"

    store._current_brain_id = "other-brain"
    with pytest.raises(LookupError, match="not found in brain other-brain"):
        await store.update_neuron(replace(neuron, content="must not cross brains"))
    assert len(queries) == 2
    assert queries[1][1]["record_id"] == "1234567890"
    assert queries[1][1]["brain_id"] == "other-brain"


@pytest.mark.asyncio
async def test_get_neuron_binds_all_digit_record_id_and_keeps_brain_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"
    row = _row(
        id="neuron:1234567890",
        brain_id="test-brain",
        metadata={"priority": 9},
        lifecycle_state="cool",
    )
    queries: list[tuple[str, dict[str, Any]]] = []

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        queries.append((sql, params))
        if params["brain_id"] != "test-brain":
            return []
        return [dict(row)]

    monkeypatch.setattr(store, "_query", query)

    neuron = await store.get_neuron("1234567890")
    assert neuron is not None
    assert neuron.id == "1234567890"
    assert neuron.metadata["lifecycle_state"] == "cool"
    assert "FROM type::record('neuron', $record_id)" in queries[0][0]
    assert queries[0][1] == {"record_id": "1234567890", "brain_id": "test-brain"}

    store._current_brain_id = "other-brain"
    assert await store.get_neuron("1234567890") is None
    assert queries[1][1] == {"record_id": "1234567890", "brain_id": "other-brain"}
