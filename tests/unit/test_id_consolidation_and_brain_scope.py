"""Single-source id sanitisation + brain-scoped record fetch.

Two follow-ups on the id-sanitiser consolidation:

  * two weak sanitisers still bypassed the ``_ids.py`` single choke point —
    ``migrations._sanitize_id`` (dash-only ``.replace``) and an inline
    ``.replace("-", "_")`` in ``tool_events`` — so both now route through
    ``_to_surreal_id``;
  * every fetch-by-record-id (``get_neuron``, ``get_neuron_state``,
    ``get_synapse``, ``get_fiber``, ``find_neurons_by_ids`` and the two
    concurrent batch fetches ``get_neurons_batch`` / ``get_synapses_batch``)
    issued a bare record select with no ``WHERE brain_id`` — a caller could read
    another brain's record just by knowing its id — so all now scope to the
    current brain (cross-brain IDOR).

The concurrent batch fetches (v2.10.5, bounded ``asyncio.Semaphore`` +
``gather``) fire ONE ``self._query`` per id, so their brain-scope is asserted
over ``conn.query.call_args_list`` (N calls), not a single ``call_args``. The
brain id is resolved once, before any fetch starts, so an unsafe brain context
fails closed (``_safe_brain_id`` raises) before a single query goes out.

The surrealdb SDK is stubbed (repo convention); the connection is an AsyncMock.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()

import surreal_memory.storage.surrealdb.tool_events as tool_events_mod
from surreal_memory.storage.surrealdb import migrations
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _store_with_mock_conn() -> tuple[SurrealDBStorage, AsyncMock]:
    st = SurrealDBStorage(url="http://localhost:8001")
    conn = AsyncMock()
    conn.query = AsyncMock(return_value=[])
    conn.insert = AsyncMock(return_value=None)
    st._conn = conn
    st._current_brain_id = "b1"
    return st, conn


# --------------------------------------------------------------------------- #
# brain-scoped batch fetch
# --------------------------------------------------------------------------- #
class TestBrainScopedBatchFetch:
    async def test_find_neurons_by_ids_scopes_to_current_brain(self):
        st, conn = _store_with_mock_conn()
        await st.find_neurons_by_ids(["abc-123", "def-456"])
        sql, params = conn.query.call_args.args
        # Still a pinned-record fetch (ids in FROM), never a `FROM neuron` scan…
        assert sql.startswith("SELECT * OMIT embedding_vec FROM neuron:abc_123, neuron:def_456")
        # …but now scoped, so a foreign-brain id cannot be read back.
        assert sql.endswith("WHERE brain_id = $brain_id")
        assert params == {"brain_id": "b1"}

    async def test_get_neurons_batch_scopes_to_current_brain(self):
        # v2.10.5 made this concurrent (semaphore + gather): one self._query per
        # id, so assert brain-scope over EVERY recorded query, not just the last.
        st, conn = _store_with_mock_conn()
        await st.get_neurons_batch(["abc-123", "def-456"])
        calls = conn.query.call_args_list
        assert len(calls) == 2
        seen_ids = set()
        for call in calls:
            sql, params = call.args
            assert sql.startswith("SELECT * FROM neuron:")
            assert sql.endswith("WHERE brain_id = $brain_id")
            assert params == {"brain_id": "b1"}
            seen_ids.add(sql)
        assert "SELECT * FROM neuron:abc_123 WHERE brain_id = $brain_id" in seen_ids
        assert "SELECT * FROM neuron:def_456 WHERE brain_id = $brain_id" in seen_ids

    async def test_get_neurons_batch_uses_safe_brain_id(self):
        # An unsafe brain context must fail closed (via _safe_brain_id) rather
        # than inline a hostile id or fire a single unscoped select — the brain
        # id is resolved once, before any concurrent fetch starts.
        st, conn = _store_with_mock_conn()
        st._current_brain_id = 'bad"; DELETE neuron'
        import pytest

        with pytest.raises(ValueError):
            await st.get_neurons_batch(["abc-123"])
        conn.query.assert_not_called()

    async def test_get_synapses_batch_scopes_to_current_brain(self):
        # New in v2.10.5, same concurrent unscoped-fetch pattern as
        # get_neurons_batch — assert brain-scope over every recorded query.
        st, conn = _store_with_mock_conn()
        await st.get_synapses_batch(["s-1", "s-2"])
        calls = conn.query.call_args_list
        assert len(calls) == 2
        seen = set()
        for call in calls:
            sql, params = call.args
            assert sql.startswith("SELECT * FROM synapse:")
            assert sql.endswith("WHERE brain_id = $brain_id")
            assert params == {"brain_id": "b1"}
            seen.add(sql)
        assert "SELECT * FROM synapse:s_1 WHERE brain_id = $brain_id" in seen
        assert "SELECT * FROM synapse:s_2 WHERE brain_id = $brain_id" in seen

    async def test_get_synapses_batch_uses_safe_brain_id(self):
        st, conn = _store_with_mock_conn()
        st._current_brain_id = 'bad"; DELETE synapse'
        import pytest

        with pytest.raises(ValueError):
            await st.get_synapses_batch(["s-1"])
        conn.query.assert_not_called()


# --------------------------------------------------------------------------- #
# single-source sanitisation
# --------------------------------------------------------------------------- #
class TestSanitizerConsolidation:
    def test_migrations_has_no_local_sanitiser(self):
        # The dash-only duplicate is gone; the single source is _ids._to_surreal_id.
        assert not hasattr(migrations, "_sanitize_id")

    def test_relation_row_folds_via_single_source(self, monkeypatch):
        # A dotted endpoint id must fold '.' -> '_' (full [A-Za-z0-9_] fold),
        # which the removed dash-only sanitiser did NOT do. This both proves the
        # single source is used and fixes a latent bug: neuron records are stored
        # folded, so a dotted legacy id used to migrate to a non-matching edge.
        rid_calls: list[tuple[str, str]] = []

        def _fake_rid(table: str, ident: object) -> str:
            rid_calls.append((table, str(ident)))
            return f"{table}:{ident}"

        monkeypatch.setattr(sys.modules["surrealdb"], "RecordID", _fake_rid)
        migrations._to_relation_row(
            {
                "id": "synapse:s1",
                "source_id": "a.b-c",
                "target_id": "x-y",
                "brain_id": "default",
                "type": "assoc",
                "weight": 1.0,
            }
        )
        assert ("neuron", _to_surreal_id("a.b-c")) in rid_calls  # -> a_b_c
        assert ("neuron", "a_b_c") in rid_calls
        assert ("neuron", "x_y") in rid_calls

    async def test_tool_events_id_routes_through_single_source(self, monkeypatch):
        st, conn = _store_with_mock_conn()
        monkeypatch.setattr(tool_events_mod, "_to_surreal_id", lambda s: "SENTINEL")
        await st.insert_tool_events(
            "b1", [{"tool_name": "t", "created_at": "2026-01-01T00:00:00+00:00"}]
        )
        table, doc = conn.insert.call_args.args
        assert table == "tool_events"
        # The record id came from the single-source folder, not a raw .replace.
        assert doc["id"] == "SENTINEL"


# --------------------------------------------------------------------------- #
# single-record fetches close the same read-by-id class
# --------------------------------------------------------------------------- #
class TestSingleRecordBrainScope:
    async def test_get_neuron_scopes_to_brain(self):
        st, conn = _store_with_mock_conn()
        await st.get_neuron("abc-123")
        sql, params = conn.query.call_args.args
        assert sql == "SELECT * FROM neuron:abc_123 WHERE brain_id = $brain_id"
        assert params == {"brain_id": "b1"}

    async def test_get_neuron_state_scopes_to_brain(self):
        st, conn = _store_with_mock_conn()
        await st.get_neuron_state("abc-123")
        sql, params = conn.query.call_args.args
        assert sql == "SELECT * FROM neuron_state:state_abc_123 WHERE brain_id = $brain_id"
        assert params == {"brain_id": "b1"}

    async def test_get_synapse_scopes_to_brain(self):
        st, conn = _store_with_mock_conn()
        await st.get_synapse("s-1")
        sql, params = conn.query.call_args.args
        assert sql == "SELECT * FROM synapse:s_1 WHERE brain_id = $brain_id"
        assert params == {"brain_id": "b1"}

    async def test_get_fiber_scopes_to_brain(self):
        st, conn = _store_with_mock_conn()
        await st.get_fiber("f-1")
        sql, params = conn.query.call_args.args
        assert sql == "SELECT * FROM fiber:f_1 WHERE brain_id = $brain_id"
        assert params == {"brain_id": "b1"}
