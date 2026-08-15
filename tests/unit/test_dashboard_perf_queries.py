"""Unit tests for the 2.7.2 dashboard-performance query shapes.

After the full BGE re-embed every neuron row carries a 1024-float
``embedding_vec``, which made ``SELECT *`` scans the dashboard's bottleneck
(Overview ~27 s, Graph 40 s+). These tests pin the fixes:

  * ``find_neurons(include_embedding=False)`` issues ``SELECT * OMIT embedding_vec``.
  * ``find_neurons_by_ids`` fetches records directly (no full-table scan) and
    omits the vector by default.
  * ``get_synapse_degrees`` aggregates per-endpoint degree with ``GROUP BY in/out``
    (the computed source_id/target_id fields do NOT aggregate).
  * ``get_edges_for_neurons`` uses the indexed ``->synapse`` traversal.
  * ``count_activated_neuron_states`` / ``get_connected_neuron_ids`` replace the
    unbounded neuron_state / synapse scans in diagnostics.

The surrealdb SDK is stubbed (repo convention); the connection is an AsyncMock.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

# Stub the optional surrealdb SDK ONLY when it is genuinely not installed: an
# `if not in sys.modules` guard would shadow an installed SDK for the rest of
# the pytest session and break the live (SURREALDB_URL) tests running later.
try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


class _FakeRID:
    def __init__(self, table: str, ident: str) -> None:
        self.table_name = table
        self.id = ident

    def __str__(self) -> str:
        return f"{self.table_name}:{self.id}"


def _store_with_mock_conn() -> tuple[SurrealDBStorage, AsyncMock]:
    st = SurrealDBStorage(url="http://localhost:8001")
    conn = AsyncMock()
    conn.query = AsyncMock(return_value=[])
    st._conn = conn
    st._current_brain_id = "b1"
    return st, conn


def _queries(conn: AsyncMock) -> list[str]:
    out = []
    for call in conn.query.call_args_list:
        out.append(call.args[0] if call.args else call.kwargs.get("sql", ""))
    return out


# --------------------------------------------------------------------------- #
# find_neurons projection
# --------------------------------------------------------------------------- #
class TestFindNeuronsProjection:
    async def test_default_selects_star(self):
        st, conn = _store_with_mock_conn()
        await st.find_neurons(limit=10)
        (sql,) = _queries(conn)
        assert sql.startswith("SELECT * FROM neuron")
        assert "OMIT" not in sql

    async def test_include_embedding_false_omits_vector(self):
        st, conn = _store_with_mock_conn()
        await st.find_neurons(limit=10, include_embedding=False)
        (sql,) = _queries(conn)
        assert sql.startswith("SELECT * OMIT embedding_vec FROM neuron")

    async def test_omitted_row_has_no_embedding_metadata(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(
            return_value=[
                {
                    "id": _FakeRID("neuron", "n_1"),
                    "type": "concept",
                    "content": "x",
                    "metadata": {},
                }
            ]
        )
        neurons = await st.find_neurons(limit=1, include_embedding=False)
        assert len(neurons) == 1
        assert "_embedding" not in neurons[0].metadata


# --------------------------------------------------------------------------- #
# find_neurons_by_ids
# --------------------------------------------------------------------------- #
class TestFindNeuronsByIds:
    async def test_fetches_records_directly_with_omit(self):
        st, conn = _store_with_mock_conn()
        await st.find_neurons_by_ids(["abc-123", "def-456"])
        (sql,) = _queries(conn)
        assert sql.startswith("SELECT * OMIT embedding_vec FROM neuron:abc_123, neuron:def_456")
        # Still a pinned-record fetch, not a `FROM neuron` table scan…
        assert "FROM neuron WHERE" not in sql
        # …but scoped to the current brain (no cross-brain id read).
        assert sql.endswith("WHERE brain_id = $brain_id")

    async def test_empty_and_unsafe_ids_are_dropped(self):
        st, conn = _store_with_mock_conn()
        assert await st.find_neurons_by_ids([]) == []
        # An injection-shaped id must not reach the query string.
        await st.find_neurons_by_ids(["bad; DELETE neuron", "ok-1"])
        (sql,) = _queries(conn)
        # Fork semantics: the hardened single-source _to_surreal_id FOLDS every
        # char outside [A-Za-z0-9_] to "_", so the hostile id becomes an inert
        # record id instead of being dropped — no breakout char can survive.
        assert "neuron:bad__DELETE_neuron" in sql
        assert ";" not in sql and '"' not in sql and "'" not in sql
        assert "neuron:ok_1" in sql

    async def test_chunks_large_id_lists(self):
        st, conn = _store_with_mock_conn()
        await st.find_neurons_by_ids([f"n-{i}" for i in range(1500)])
        sqls = _queries(conn)
        assert len(sqls) == 2  # 1000 + 500


# --------------------------------------------------------------------------- #
# Degree + traversal aggregates
# --------------------------------------------------------------------------- #
class TestSynapseDegrees:
    async def test_groups_by_native_in_and_out(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(
            side_effect=[
                [{"nid": _FakeRID("neuron", "a_1"), "deg": 3}],  # GROUP BY in
                [{"nid": _FakeRID("neuron", "a_1"), "deg": 2}],  # GROUP BY out
            ]
        )
        degree = await st.get_synapse_degrees()
        sqls = _queries(conn)
        assert any("GROUP BY in" in s for s in sqls)
        assert any("GROUP BY out" in s for s in sqls)
        # source_id/target_id are computed fields and must not be grouped on.
        assert not any("GROUP BY source_id" in s or "GROUP BY target_id" in s for s in sqls)
        assert degree == {"a-1": 5}


class TestEdgesForNeurons:
    async def test_uses_indexed_graph_traversal(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(
            return_value=[
                {
                    "id": _FakeRID("neuron", "src_1"),
                    "edges": [
                        {
                            "id": _FakeRID("synapse", "e_1"),
                            "out": _FakeRID("neuron", "dst_2"),
                            "type": "related_to",
                            "weight": 0.8,
                            "direction": "uni",
                        }
                    ],
                }
            ]
        )
        edges = await st.get_edges_for_neurons(["src-1"])
        (sql,) = _queries(conn)
        assert "->synapse" in sql
        assert sql.strip().startswith("SELECT id, ->synapse")
        assert len(edges) == 1
        assert edges[0].source_id == "src-1"
        assert edges[0].target_id == "dst-2"

    async def test_empty_input_short_circuits(self):
        st, conn = _store_with_mock_conn()
        assert await st.get_edges_for_neurons([]) == []
        conn.query.assert_not_called()


# --------------------------------------------------------------------------- #
# Diagnostics aggregates
# --------------------------------------------------------------------------- #
class TestDiagnosticsAggregates:
    async def test_count_activated_uses_group_all(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(return_value=[{"c": 42}])
        n = await st.count_activated_neuron_states()
        (sql,) = _queries(conn)
        assert "count()" in sql and "access_frequency > 0" in sql and "GROUP ALL" in sql
        assert n == 42

    async def test_connected_ids_groups_endpoints(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(
            side_effect=[
                [str(_FakeRID("neuron", "a_1"))],  # SELECT VALUE in ... GROUP BY in
                [str(_FakeRID("neuron", "b_2"))],  # SELECT VALUE out ... GROUP BY out
            ]
        )
        connected = await st.get_connected_neuron_ids()
        sqls = _queries(conn)
        assert any("SELECT VALUE in" in s and "GROUP BY in" in s for s in sqls)
        assert any("SELECT VALUE out" in s and "GROUP BY out" in s for s in sqls)
        assert connected == {"a-1", "b-2"}


# --------------------------------------------------------------------------- #
# _query_values -- honest SELECT VALUE typing (#154 finding 3)
# --------------------------------------------------------------------------- #
class TestQueryValues:
    """The SDK (>=2.0.0) unwraps the RPC envelope itself: query() returns the
    first statement's result directly, so these mocks use the shapes a live
    server produces (verified against SurrealDB 3.5.0 over ws and http)."""

    async def test_flat_scalar_list(self):
        """`in`/`out` are scalar record links -- one value per row."""
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(return_value=["neuron:a", "neuron:b"])
        values = await st._query_values("SELECT VALUE in FROM synapse")
        assert values == ["neuron:a", "neuron:b"]

    async def test_empty_result_is_empty_list(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(return_value=[])
        assert await st._query_values("SELECT VALUE in FROM synapse") == []

    async def test_a_row_whose_value_is_itself_an_array_is_not_collapsed(self):
        """The #143 trap: SELECT VALUE on an array-typed field. One matching
        row whose value is an array must come back as that one array, not be
        confused for "these array elements are separate rows"."""
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(return_value=[["tag-a", "tag-b"]])
        values = await st._query_values("SELECT VALUE tags FROM neuron")
        assert values == [["tag-a", "tag-b"]]

    async def test_every_rows_array_comes_back_not_just_the_first(self):
        """The #154-finding-3 residual, pinned: the old ``result[0] if
        isinstance(result[0], list)`` unwrap collapsed a list of per-row
        arrays to the FIRST row's array. All rows must survive."""
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(return_value=[["t1", "t2"], ["t3"], None])
        values = await st._query_values("SELECT VALUE tags FROM neuron")
        assert values == [["t1", "t2"], ["t3"], None]

    async def test_scalar_statement_result_reads_as_no_values(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(return_value=5)
        assert await st._query_values("RETURN 5") == []


class TestGetAllSynapsesProjection:
    async def test_include_metadata_false_omits_blob(self):
        st, conn = _store_with_mock_conn()
        await st.get_all_synapses(include_metadata=False)
        (sql,) = _queries(conn)
        assert sql.startswith("SELECT * OMIT metadata FROM synapse")

    async def test_default_keeps_metadata(self):
        st, conn = _store_with_mock_conn()
        await st.get_all_synapses()
        (sql,) = _queries(conn)
        assert sql.startswith("SELECT * FROM synapse")


# --------------------------------------------------------------------------- #
# brain_id inline literal (index usage) — 2.7.3
# --------------------------------------------------------------------------- #
class TestBrainLiteral:
    def test_valid_brain_id_is_quoted(self):
        from surreal_memory.storage.surrealdb.store import _brain_literal

        assert _brain_literal("default") == '"default"'
        assert _brain_literal("it_2c80122dbe75") == '"it_2c80122dbe75"'
        assert _brain_literal("my-brain.v2") == '"my-brain.v2"'

    def test_injection_shaped_brain_id_rejected(self):
        import pytest

        from surreal_memory.storage.surrealdb.store import _brain_literal

        for bad in ['a" OR "1"="1', "a; DELETE neuron", "a b", 'a"']:
            with pytest.raises(ValueError):
                _brain_literal(bad)


class TestGetStatsInline:
    async def test_counts_inline_brain_id_for_index(self):
        st, conn = _store_with_mock_conn()
        await st.get_stats("default")
        sqls = _queries(conn)
        # brain_id inlined as a literal (uses the index) — never parameterized,
        # which in SurrealDB 3.2.0 falls back to a full table scan.
        assert all('brain_id = "default"' in s for s in sqls)
        assert not any("$bid" in s or "$brain_id" in s for s in sqls)
        assert {s.split("FROM ")[1].split(" ")[0] for s in sqls} == {
            "neuron",
            "synapse",
            "fiber",
        }


class TestEnhancedStatsSkipNeuronTypes:
    async def test_skip_neuron_types_omits_neuron_group_by(self):
        st, conn = _store_with_mock_conn()
        await st.get_enhanced_stats("default", include_neuron_types=False)
        sqls = _queries(conn)
        # The pricey `FROM neuron … GROUP BY type` scan must NOT run.
        assert not any("FROM neuron" in s and "GROUP BY type" in s for s in sqls)
        # The synapse-type stats (needed for diversity) still run.
        assert any("FROM synapse" in s and "GROUP BY type" in s for s in sqls)

    async def test_default_includes_neuron_types(self):
        st, conn = _store_with_mock_conn()
        await st.get_enhanced_stats("default")
        sqls = _queries(conn)
        assert any("FROM neuron" in s and "GROUP BY type" in s for s in sqls)
