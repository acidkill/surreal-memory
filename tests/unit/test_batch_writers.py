"""Unit tests for the SurrealDB batch writers used by ``smem decay``.

``update_synapses_batch`` / ``update_neuron_states_batch`` were stubs — plain
``for`` loops over the single-row methods — so a decay pass over ~57k synapses
issued ~116k sequential HTTP round-trips (98% of a measured 475 s run). These
tests pin the property that made the fix worth doing: ONE query per chunk, not
one per row, including at the chunk boundaries.

Mock/fake based — no live SurrealDB required.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock

from surreal_memory.core.neuron import NeuronState
from surreal_memory.core.synapse import Direction, Synapse, SynapseType
from surreal_memory.storage.surrealdb.store import (
    _BATCH_WRITE_CHUNK,
    SurrealDBStorage,
    _chunked,
)
from surreal_memory.utils.timeutils import utcnow


def _make_storage_spy() -> Any:
    """A storage instance with the transport replaced by recording spies.

    Constructed without connecting: only the query-building code runs.
    """
    os.environ.setdefault("SURREALDB_URL", "http://127.0.0.1:65535")
    os.environ.setdefault("SURREALDB_USER", "root")
    os.environ.setdefault("SURREALDB_PASS", "x")
    os.environ.setdefault("SURREALDB_NS", "ns")
    os.environ.setdefault("SURREALDB_DB", "db")
    s = SurrealDBStorage()
    s._current_brain_id = "default"
    s._skip_change_log = False
    s._query = AsyncMock(return_value=[])
    # change_log rides on conn.insert, not _query — spy the connection too.
    s._conn = AsyncMock()
    return s


def _syn(n: int) -> Synapse:
    return Synapse.create(
        source_id=f"neuron:s{n}",
        target_id=f"neuron:t{n}",
        type=SynapseType.CONTAINS,
    )


def _state(n: int) -> NeuronState:
    return NeuronState(neuron_id=f"neuron:n{n}", activation_level=0.5)


# ---------------------------- chunk helper ----------------------------


def test_chunked_boundaries():
    assert list(_chunked([])) == []
    assert [len(c) for c in _chunked(list(range(1)))] == [1]
    assert [len(c) for c in _chunked(list(range(_BATCH_WRITE_CHUNK)))] == [_BATCH_WRITE_CHUNK]
    assert [len(c) for c in _chunked(list(range(_BATCH_WRITE_CHUNK + 1)))] == [
        _BATCH_WRITE_CHUNK,
        1,
    ]


def test_chunked_preserves_order_and_every_item():
    items = list(range(_BATCH_WRITE_CHUNK * 2 + 7))
    flat = [x for chunk in _chunked(items) for x in chunk]
    assert flat == items


# ---------------------------- synapse batch ----------------------------


def test_update_synapses_batch_is_one_query_per_chunk():
    """The N+1 must be gone: 5 synapses => 1 _query, not 5."""
    s = _make_storage_spy()
    asyncio.run(s.update_synapses_batch([_syn(i) for i in range(5)]))
    assert s._query.await_count == 1, f"expected 1 _query, got {s._query.await_count}"
    sql = s._query.await_args.args[0]
    assert sql.count("UPDATE type::record($tbl, $id") == 5
    assert s._query.await_args.kwargs["tbl"] == "synapse"
    assert sql.rstrip().endswith(";")


def test_update_synapses_batch_binds_ids_and_payloads_as_params():
    """SECURITY: nothing derived from a synapse may reach the SurQL text."""
    s = _make_storage_spy()
    syn = _syn(0)
    syn = Synapse(
        id=syn.id,
        source_id=syn.source_id,
        target_id=syn.target_id,
        type=SynapseType.CONTAINS,
        weight=0.25,
        direction=Direction.UNIDIRECTIONAL,
        metadata={"note": 'x"; DELETE synapse; --'},
        reinforced_count=3,
        created_at=syn.created_at,
    )
    asyncio.run(s.update_synapses_batch([syn]))
    sql, params = s._query.await_args.args[0], s._query.await_args.kwargs
    assert "DELETE synapse" not in sql
    assert params["id0"] == syn.id.replace("-", "_")
    assert params["d0"]["weight"] == 0.25
    assert params["d0"]["reinforced_count"] == 3
    assert params["d0"]["metadata"] == {"note": 'x"; DELETE synapse; --'}


def test_update_synapses_batch_omits_unset_last_activated():
    """Parity with update_synapse: never null out a timestamp it would leave alone."""
    s = _make_storage_spy()
    never = _syn(0)
    assert never.last_activated is None
    asyncio.run(s.update_synapses_batch([never]))
    assert "last_activated" not in s._query.await_args.kwargs["d0"]

    s2 = _make_storage_spy()
    now = utcnow()
    used = Synapse(
        id=never.id,
        source_id=never.source_id,
        target_id=never.target_id,
        type=never.type,
        last_activated=now,
        created_at=never.created_at,
    )
    asyncio.run(s2.update_synapses_batch([used]))
    assert s2._query.await_args.kwargs["d0"]["last_activated"] == now


def test_update_synapses_batch_chunks_at_boundary():
    """500 rows => 1 query; 501 => 2. One statement per row, none lost."""
    for count, expected_queries in ((0, 0), (1, 1), (_BATCH_WRITE_CHUNK, 1), (501, 2)):
        s = _make_storage_spy()
        asyncio.run(s.update_synapses_batch([_syn(i) for i in range(count)]))
        assert s._query.await_count == expected_queries, f"{count} rows"
        statements = sum(
            call.args[0].count("UPDATE type::record($tbl, $id") for call in s._query.await_args_list
        )
        assert statements == count, f"{count} rows lost statements"


def test_update_synapses_batch_empty_is_noop():
    s = _make_storage_spy()
    asyncio.run(s.update_synapses_batch([]))
    assert s._query.await_count == 0
    assert s._conn.insert.await_count == 0


# ---------------------------- change_log batching ----------------------------


def test_change_log_rows_are_bulk_inserted_not_dropped():
    """Every synapse still gets its own change_log row — in ONE insert call."""
    s = _make_storage_spy()
    syns = [_syn(i) for i in range(5)]
    asyncio.run(s.update_synapses_batch(syns))

    assert s._conn.insert.await_count == 1, "change_log must be ONE bulk insert per chunk"
    table, records = s._conn.insert.await_args.args
    assert table == "change_log"
    assert len(records) == 5, "no change_log entry may be silently dropped"
    assert [r["entity_id"] for r in records] == [x.id for x in syns]
    assert {r["operation"] for r in records} == {"update"}
    assert {r["entity_type"] for r in records} == {"synapse"}
    # Sequence numbers stay unique and monotonic — delta sync orders on them.
    seqs = [r["sequence"] for r in records]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == 5
    # Payload carries what a peer needs to replay the write.
    assert records[0]["payload"]["source_id"] == syns[0].source_id


def test_change_log_bulk_insert_per_chunk():
    s = _make_storage_spy()
    asyncio.run(s.update_synapses_batch([_syn(i) for i in range(501)]))
    assert s._conn.insert.await_count == 2
    total = sum(len(call.args[1]) for call in s._conn.insert.await_args_list)
    assert total == 501


def test_change_log_skipped_only_via_explicit_flag():
    s = _make_storage_spy()
    s._skip_change_log = True
    asyncio.run(s.update_synapses_batch([_syn(0)]))
    assert s._query.await_count == 1, "entity write still happens"
    assert s._conn.insert.await_count == 0


def test_change_log_failure_does_not_abort_the_entity_write():
    """Sync bookkeeping is fail-soft — same contract as _record_change_internal."""
    s = _make_storage_spy()
    s._conn.insert = AsyncMock(side_effect=RuntimeError("change_log down"))
    asyncio.run(s.update_synapses_batch([_syn(0)]))  # must not raise
    assert s._query.await_count == 1


# ---------------------------- neuron state batch ----------------------------


def test_update_neuron_states_batch_is_one_query_per_chunk():
    s = _make_storage_spy()
    asyncio.run(s.update_neuron_states_batch([_state(i) for i in range(7)]))
    assert s._query.await_count == 1
    sql = s._query.await_args.args[0]
    assert sql.count("UPDATE type::record($tbl, $id") == 7
    assert s._query.await_args.kwargs["tbl"] == "neuron_state"


def test_update_neuron_states_batch_targets_the_state_record_id():
    """State rows live at neuron_state:state_<sanitised neuron id>."""
    s = _make_storage_spy()
    asyncio.run(s.update_neuron_states_batch([NeuronState(neuron_id="neuron:ab-cd")]))
    assert s._query.await_args.kwargs["id0"] == "state_ab_cd"


def test_update_neuron_states_batch_chunks_at_boundary():
    for count, expected_queries in ((0, 0), (1, 1), (_BATCH_WRITE_CHUNK, 1), (501, 2)):
        s = _make_storage_spy()
        asyncio.run(s.update_neuron_states_batch([_state(i) for i in range(count)]))
        assert s._query.await_count == expected_queries, f"{count} rows"
        statements = sum(
            call.args[0].count("UPDATE type::record($tbl, $id") for call in s._query.await_args_list
        )
        assert statements == count


def test_update_neuron_states_batch_writes_no_change_log():
    """Parity with update_neuron_state, which does not log — activation drift is
    derived local state, not a synced entity."""
    s = _make_storage_spy()
    asyncio.run(s.update_neuron_states_batch([_state(0)]))
    assert s._conn.insert.await_count == 0


def test_update_neuron_states_batch_omits_unset_timestamps():
    s = _make_storage_spy()
    asyncio.run(s.update_neuron_states_batch([NeuronState(neuron_id="neuron:x")]))
    data = s._query.await_args.kwargs["d0"]
    assert "last_activated" not in data
    assert "refractory_until" not in data
    assert data["activation_level"] == 0.0


def test_update_neuron_states_batch_empty_is_noop():
    s = _make_storage_spy()
    asyncio.run(s.update_neuron_states_batch([]))
    assert s._query.await_count == 0


# ------------------ round-trip count vs the old stub ------------------


def test_batch_writers_beat_the_sequential_stub_by_orders_of_magnitude():
    """Guards the whole point of the fix: round-trips must scale with chunks,
    not rows. The old stub issued 2 per synapse (merge + change_log)."""
    rows = 1200
    s = _make_storage_spy()
    asyncio.run(s.update_synapses_batch([_syn(i) for i in range(rows)]))
    round_trips = s._query.await_count + s._conn.insert.await_count
    stub_round_trips = rows * 2
    assert round_trips == 6, f"expected 3 chunks x (1 update + 1 change_log), got {round_trips}"
    assert round_trips * 100 < stub_round_trips
