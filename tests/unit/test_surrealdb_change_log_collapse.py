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
            [{"id": newest, "entity_type": "synapse", "entity_id": "s-1", "sequence": 3}],
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

    high_water, select, maxima = store._query.await_args_list
    assert "ORDER BY sequence DESC LIMIT 1" in high_water.args[0]
    assert "SELECT id, entity_type, entity_id, sequence FROM change_log" in select.args[0]
    assert "sequence <= $watermark" in select.args[0]
    assert "ORDER BY sequence ASC, id ASC LIMIT $cap" in select.args[0]
    assert select.kwargs["cap"] == 4
    assert "FROM change_log WITH INDEX idx_changelog_seq WHERE " in maxima.args[0]
    assert "ORDER BY sequence DESC, id DESC" in maxima.args[0]
    assert maxima.kwargs["upper_sequence"] == 3
    assert "entity_ids" not in maxima.kwargs
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
            [{"id": tied_b, "entity_type": "synapse", "entity_id": "s-1", "sequence": 3}],
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
            [
                {
                    "id": _RecordID("change_log:safe_newest"),
                    "entity_type": "neuron",
                    "entity_id": "n-1",
                    "sequence": 2,
                }
            ],
        ]
    )
    store._query_values = AsyncMock()

    deleted, _, exhausted = await store.collapse_pending_updates()
    assert (deleted, exhausted) == (0, True)
    assert store._query.await_count == 3
    store._query_values.assert_not_awaited()


@pytest.mark.asyncio
async def test_collapse_keyset_scans_global_maxima_across_pages_and_entity_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("surreal_memory.storage.surrealdb.store._COLLAPSE_READ_PAGE_ROWS", 2)
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    neuron_old = _RecordID("change_log:neuron_old")
    synapse_old = _RecordID("change_log:synapse_old")
    neuron_new = _RecordID("change_log:neuron_new")
    synapse_tie_a = _RecordID("change_log:synapse_tie_a")
    synapse_tie_b = _RecordID("change_log:synapse_tie_b")
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 5}],
            [
                {"id": neuron_old, "entity_type": "neuron", "entity_id": "same", "sequence": 1},
                {"id": synapse_old, "entity_type": "synapse", "entity_id": "same", "sequence": 2},
            ],
            [
                {"id": neuron_new, "entity_type": "neuron", "entity_id": "same", "sequence": 5},
                {
                    "id": _RecordID("change_log:other"),
                    "entity_type": "fiber",
                    "entity_id": "other",
                    "sequence": 4,
                },
            ],
            [
                {"id": synapse_tie_b, "entity_type": "synapse", "entity_id": "same", "sequence": 3},
                {"id": synapse_tie_a, "entity_type": "synapse", "entity_id": "same", "sequence": 3},
            ],
        ]
    )
    store._query_values = AsyncMock(return_value=[neuron_old, synapse_old])

    deleted, _, exhausted = await store.collapse_pending_updates(max_rows=2)

    assert (deleted, exhausted) == (2, False)
    first_maxima, second_maxima = store._query.await_args_list[2:]
    assert first_maxima.kwargs["upper_sequence"] == 5
    assert second_maxima.kwargs["upper_sequence"] == 4
    assert str(second_maxima.kwargs["before_id"]) == "change_log:other"
    assert store._query_values.await_args.kwargs["record_ids"] == [neuron_old, synapse_old]


@pytest.mark.asyncio
async def test_collapse_maxima_read_failure_never_deletes_candidates() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 2}],
            [
                {
                    "id": _RecordID("change_log:old"),
                    "entity_type": "synapse",
                    "entity_id": "s-1",
                    "sequence": 1,
                }
            ],
            RuntimeError("read failed"),
        ]
    )
    store._query_values = AsyncMock()

    with pytest.raises(RuntimeError, match="read failed"):
        await store.collapse_pending_updates(max_rows=1)

    store._query_values.assert_not_awaited()


@pytest.mark.asyncio
async def test_collapse_rejects_a_nonadvancing_maxima_page_cursor() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    stalled_page = [
        {
            "id": _RecordID("change_log:other_new"),
            "entity_type": "fiber",
            "entity_id": "other",
            "sequence": 2,
        },
        {
            "id": _RecordID("change_log:other_old"),
            "entity_type": "fiber",
            "entity_id": "other",
            "sequence": 1,
        },
    ]
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 2}],
            [
                {
                    "id": _RecordID("change_log:target"),
                    "entity_type": "synapse",
                    "entity_id": "target",
                    "sequence": 1,
                }
            ],
            stalled_page,
            stalled_page,
        ]
    )
    store._query_values = AsyncMock()

    with pytest.raises(RuntimeError, match="maxima scan cursor did not advance"):
        await store.collapse_pending_updates(max_rows=1)

    store._query_values.assert_not_awaited()


@pytest.mark.asyncio
async def test_candidate_wire_pages_preserve_ties_and_round_trip_quoted_numeric_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("surreal_memory.storage.surrealdb.store._COLLAPSE_READ_PAGE_ROWS", 1)
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    rows = [
        {
            "id": _RecordID("change_log:a"),
            "entity_type": "synapse",
            "entity_id": "a",
            "sequence": 1,
        },
        {
            "id": _RecordID("change_log:b"),
            "entity_type": "synapse",
            "entity_id": "b",
            "sequence": 1,
        },
        {
            "id": _RecordID("change_log:c"),
            "entity_type": "synapse",
            "entity_id": "c",
            "sequence": 1,
        },
    ]
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 1}],
            [rows[0]],
            [rows[1]],
            [rows[2]],
            list(reversed(rows)),
        ]
    )
    store._query_values = AsyncMock()

    deleted, cursor, exhausted = await store.collapse_pending_updates(max_rows=3)

    assert deleted == 0
    assert not exhausted
    state = json.loads(cursor or "")
    assert state["id"] == "c"
    candidate_queries = [
        call
        for call in store._query.await_args_list
        if "ORDER BY sequence ASC, id ASC" in call.args[0]
    ]
    assert len(candidate_queries) == 3
    assert all(call.kwargs["cap"] == 1 for call in candidate_queries)
    assert all("WITH INDEX idx_changelog_seq" in call.args[0] for call in candidate_queries)
    assert candidate_queries[2].kwargs["last_sequence"] == 1
    assert candidate_queries[2].kwargs["last_id"] == "b"

    quoted = {
        "id": _RecordID("change_log:⟨123⟩"),
        "entity_type": "synapse",
        "entity_id": "quoted",
        "sequence": 1,
    }
    store._query.side_effect = [[{"sequence": 1}], [quoted], [quoted]]
    _, quoted_cursor, quoted_exhausted = await store.collapse_pending_updates(max_rows=1)
    assert not quoted_exhausted
    quoted_state = json.loads(quoted_cursor or "")
    assert quoted_state["id"] == "123"

    store._query.side_effect = [[]]
    assert await store.collapse_pending_updates(cursor=quoted_cursor) == (
        0,
        quoted_cursor,
        True,
    )
    assert store._query.await_args.kwargs["last_id"] == "123"
    legacy_cursor = json.dumps({**quoted_state, "id": "⟨123⟩"})
    store._query.side_effect = [[]]
    assert await store.collapse_pending_updates(cursor=legacy_cursor) == (
        0,
        legacy_cursor,
        True,
    )
    assert store._query.await_args.kwargs["last_id"] == "123"
    store._query_values.assert_not_awaited()


@pytest.mark.asyncio
async def test_candidate_wire_page_read_failure_never_deletes_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("surreal_memory.storage.surrealdb.store._COLLAPSE_READ_PAGE_ROWS", 1)
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    store._query = AsyncMock(
        side_effect=[
            [{"sequence": 2}],
            [
                {
                    "id": _RecordID("change_log:first"),
                    "entity_type": "synapse",
                    "entity_id": "s-1",
                    "sequence": 1,
                }
            ],
            RuntimeError("candidate read failed"),
        ]
    )
    store._query_values = AsyncMock()

    with pytest.raises(RuntimeError, match="candidate read failed"):
        await store.collapse_pending_updates(max_rows=2)

    store._query_values.assert_not_awaited()
