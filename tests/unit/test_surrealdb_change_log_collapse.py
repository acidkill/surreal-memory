"""Focused guards for bounded, brain-scoped pending update collapse."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


class _RecordID:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


@pytest.mark.asyncio
async def test_collapse_uses_global_max_and_reports_rows_actually_deleted() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    old = _RecordID("change_log:old_update")
    middle = _RecordID("change_log:middle_update")
    newest = _RecordID("change_log:newest_update")
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 3}],
            [
                {"id": old, "entity_type": "synapse", "entity_id": "s-1", "sequence": 1},
                {
                    "id": middle,
                    "entity_type": "synapse",
                    "entity_id": "s-1",
                    "sequence": 2,
                },
                {
                    "id": newest,
                    "entity_type": "synapse",
                    "entity_id": "s-1",
                    "sequence": 3,
                },
            ],
            [{"entity_id": "s-1", "latest_sequence": 3}],
        ]
    )
    # A concurrent sync made one selected candidate ineligible before DELETE.
    # RETURN BEFORE.id still gives the exact number that the server removed.
    store._query_values = AsyncMock(return_value=[old])

    deleted, cursor, exhausted = await store.collapse_pending_updates(max_rows=4)
    assert (deleted, exhausted) == (1, True)
    assert json.loads(cursor or "") == {
        "brain_id": "brain-a",
        "watermark": 3,
        "sequence": 3,
        "id": "newest_update",
    }

    high_water, select, latest = store._query.await_args_list
    assert "ORDER BY sequence DESC LIMIT 1" in high_water.args[0]
    assert "SELECT id, entity_type, entity_id, sequence FROM change_log" in select.args[0]
    assert "sequence <= $watermark" in select.args[0]
    assert "ORDER BY sequence ASC, id ASC LIMIT $cap" in select.args[0]
    assert select.kwargs["cap"] == 4
    assert "math::max(sequence) AS latest_sequence" in latest.args[0]
    assert latest.kwargs["entity_ids"] == ["s-1"]
    store._query_values.assert_awaited_once()
    delete = store._query_values.await_args
    assert delete.args[0] == (
        "DELETE FROM $record_ids WHERE brain_id = $brain_id "
        "AND operation = 'update' AND synced = false RETURN BEFORE.id"
    )
    assert delete.kwargs["record_ids"] == [old, middle]
    assert delete.kwargs["record_ids"][0] is old
    assert delete.kwargs["brain_id"] == "brain-a"


@pytest.mark.asyncio
async def test_collapse_keeps_tied_global_maxima() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    old = _RecordID("change_log:older")
    tied_a = _RecordID("change_log:tied_a")
    tied_b = _RecordID("change_log:tied_b")
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 3}],
            [
                {"id": old, "entity_type": "synapse", "entity_id": "s-1", "sequence": 1},
                {"id": tied_a, "entity_type": "synapse", "entity_id": "s-1", "sequence": 3},
                {"id": tied_b, "entity_type": "synapse", "entity_id": "s-1", "sequence": 3},
            ],
            [{"entity_id": "s-1", "latest_sequence": 3}],
        ]
    )
    store._query_values = AsyncMock(return_value=[old])

    deleted, _, exhausted = await store.collapse_pending_updates(max_rows=4)

    assert (deleted, exhausted) == (1, True)
    assert store._query_values.await_args.kwargs["record_ids"] == [old]


@pytest.mark.asyncio
async def test_collapse_rejects_malformed_cursor_before_querying() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    store._query = AsyncMock()

    with pytest.raises(ValueError, match="invalid change-log collapse cursor"):
        await store.collapse_pending_updates(cursor='{"watermark": 1}')

    store._query.assert_not_awaited()


@pytest.mark.asyncio
async def test_collapse_rejects_cursor_from_another_brain() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    store._query = AsyncMock()

    with pytest.raises(ValueError, match="invalid change-log collapse cursor"):
        await store.collapse_pending_updates(
            cursor=json.dumps({"brain_id": "brain-b", "watermark": 1, "sequence": 1, "id": "row"})
        )

    store._query.assert_not_awaited()


@pytest.mark.asyncio
async def test_collapse_does_not_send_ids_that_fail_the_strict_record_id_check() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 2}],
            [
                {
                    "id": _RecordID("change_log:unsafe'quoted"),
                    "entity_type": "neuron",
                    "entity_id": "n-1",
                    "sequence": 1,
                },
                {
                    "id": _RecordID("change_log:safe_newest"),
                    "entity_type": "neuron",
                    "entity_id": "n-1",
                    "sequence": 2,
                },
            ],
            [{"entity_id": "n-1", "latest_sequence": 2}],
        ]
    )
    store._query_values = AsyncMock()

    deleted, _, exhausted = await store.collapse_pending_updates()
    assert (deleted, exhausted) == (0, True)
    assert store._query.await_count == 3
    store._query_values.assert_not_awaited()
