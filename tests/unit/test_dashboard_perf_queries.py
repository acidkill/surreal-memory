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

if "surrealdb" not in sys.modules:
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
                [
                    {
                        "id": _FakeRID("neuron", "n_1"),
                        "type": "concept",
                        "content": "x",
                        "metadata": {},
                    }
                ]
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
        assert "WHERE" not in sql  # record fetch, not a table scan

    async def test_empty_and_unsafe_ids_are_dropped(self):
        st, conn = _store_with_mock_conn()
        assert await st.find_neurons_by_ids([]) == []
        # An injection-shaped id must not reach the query string.
        await st.find_neurons_by_ids(["bad; DELETE neuron", "ok-1"])
        (sql,) = _queries(conn)
        assert "DELETE" not in sql
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
                [[{"nid": _FakeRID("neuron", "a_1"), "deg": 3}]],  # GROUP BY in
                [[{"nid": _FakeRID("neuron", "a_1"), "deg": 2}]],  # GROUP BY out
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
                [
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
        conn.query = AsyncMock(return_value=[[{"c": 42}]])
        n = await st.count_activated_neuron_states()
        (sql,) = _queries(conn)
        assert "count()" in sql and "access_frequency > 0" in sql and "GROUP ALL" in sql
        assert n == 42

    async def test_connected_ids_groups_endpoints(self):
        st, conn = _store_with_mock_conn()
        conn.query = AsyncMock(
            side_effect=[
                [[str(_FakeRID("neuron", "a_1"))]],  # SELECT VALUE in ... GROUP BY in
                [[str(_FakeRID("neuron", "b_2"))]],  # SELECT VALUE out ... GROUP BY out
            ]
        )
        connected = await st.get_connected_neuron_ids()
        sqls = _queries(conn)
        assert any("SELECT VALUE in" in s and "GROUP BY in" in s for s in sqls)
        assert any("SELECT VALUE out" in s and "GROUP BY out" in s for s in sqls)
        assert connected == {"a-1", "b-2"}


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
