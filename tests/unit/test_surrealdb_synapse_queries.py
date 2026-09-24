"""Unit tests for the v8 RELATION synapse query shapes in SurrealDBStorage.

The surrealdb SDK is stubbed (repo convention); the connection is an AsyncMock
whose query() records calls, so these assert the SurrealQL shape (native in/out,
type::record, INSERT RELATION, in.*/out.* inline) without a live DB. End-to-end
behaviour against a real v3.2.0 DB is covered by the U6 integration test.
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# Stub the optional surrealdb SDK ONLY when it is genuinely not installed: an
# `if not in sys.modules` guard would shadow an installed SDK for the rest of
# the pytest session and break the live (SURREALDB_URL) tests running later.
try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()

from surreal_memory.core.synapse import Direction, Synapse, SynapseType
from surreal_memory.storage.surrealdb.store import (
    SurrealDBStorage,
    _endpoint_to_id,
    _row_to_synapse,
)


class _FakeRID:
    """Minimal RecordID stand-in: str() == 'table:id', with .table_name/.id."""

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


def _find_query(conn: AsyncMock, needle: str) -> tuple[str, dict] | None:
    for call in conn.query.call_args_list:
        sql = call.args[0] if call.args else call.kwargs.get("sql", "")
        params = call.args[1] if len(call.args) > 1 else {}
        if needle in sql:
            return sql, params
    return None


# --------------------------------------------------------------------------- #
# _row_to_synapse endpoint mapping
# --------------------------------------------------------------------------- #
class TestRowToSynapse:
    def test_maps_in_out_recordids_to_source_target(self):
        row = {
            "id": _FakeRID("synapse", "e1"),
            "in": _FakeRID("neuron", "abc_123"),
            "out": _FakeRID("neuron", "def_456"),
            "type": "related_to",
            "weight": 1.0,
            "direction": "uni",
        }
        syn = _row_to_synapse(row)
        assert syn.id == "e1"
        # underscores denormalised back to dashes
        assert syn.source_id == "abc-123"
        assert syn.target_id == "def-456"

    def test_falls_back_to_legacy_source_target(self):
        row = {
            "id": _FakeRID("synapse", "e2"),
            "source_id": "old_src",
            "target_id": "old_tgt",
            "type": "related_to",
        }
        syn = _row_to_synapse(row)
        assert syn.source_id == "old-src"
        assert syn.target_id == "old-tgt"

    def test_endpoint_to_id_prefers_edge_over_legacy(self):
        assert _endpoint_to_id(_FakeRID("neuron", "x_1"), "legacy") == "x-1"
        assert _endpoint_to_id(None, "leg_acy") == "leg-acy"
        assert _endpoint_to_id(None, None) == ""
        # bare string 'neuron:y' strips the table prefix
        assert _endpoint_to_id("neuron:y_2", None) == "y-2"


# --------------------------------------------------------------------------- #
# add_synapse issues INSERT RELATION with in/out RecordIDs
# --------------------------------------------------------------------------- #
class TestAddSynapse:
    @pytest.mark.asyncio
    async def test_uses_insert_relation_with_in_out(self):
        st, conn = _store_with_mock_conn()
        syn = Synapse.create(
            "src-1", "tgt-2", SynapseType.RELATED_TO, direction=Direction.UNIDIRECTIONAL
        )
        await st.add_synapse(syn)

        found = _find_query(conn, "INSERT RELATION INTO synapse")
        assert found is not None, "add_synapse must use INSERT RELATION"
        sql, params = found
        row = params["row"]
        assert {"id", "in", "out"}.issubset(row.keys())
        # flat document columns are gone
        assert "source_id" not in row and "target_id" not in row
        # plain conn.insert must NOT be used for the RELATION table
        assert not conn.insert.called or not any(
            c.args and c.args[0] == "synapse" for c in conn.insert.call_args_list
        )


# --------------------------------------------------------------------------- #
# get_synapses / get_neighbors / delete_neuron query shapes
# --------------------------------------------------------------------------- #
class TestQueryShapes:
    @pytest.mark.asyncio
    async def test_get_synapses_filters_on_in_out_via_type_record(self):
        st, conn = _store_with_mock_conn()
        await st.get_synapses(source_id="a", target_id="b")
        found = _find_query(conn, "FROM synapse WHERE")
        assert found is not None
        sql, params = found
        assert "in = type::record('neuron', $source_id)" in sql
        assert "out = type::record('neuron', $target_id)" in sql
        assert "source_id = " not in sql and "target_id = " not in sql

    @pytest.mark.asyncio
    async def test_alias_pair_lookup_forces_source_index(self):
        st, conn = _store_with_mock_conn()
        await st.get_synapses(
            source_id="duplicate-1", target_id="canonical-2", type=SynapseType.ALIAS, limit=1
        )
        found = _find_query(conn, "FROM synapse WITH INDEX idx_synapse_in WHERE")
        assert found is not None
        sql, params = found
        assert "in = type::record('neuron', $source_id)" in sql
        assert "out = type::record('neuron', $target_id)" in sql
        assert "type = $stype" in sql
        assert params["stype"] == "alias"

    @pytest.mark.asyncio
    async def test_alias_slice_lookup_keeps_default_index_selection(self):
        st, conn = _store_with_mock_conn()
        await st.get_synapses(type=SynapseType.ALIAS, limit=5000)
        found = _find_query(conn, "FROM synapse WHERE")
        assert found is not None
        assert "WITH INDEX idx_synapse_in" not in found[0]

    @pytest.mark.asyncio
    async def test_get_neighbors_inlines_endpoints_to_kill_n_plus_1(self):
        st, conn = _store_with_mock_conn()
        await st.get_neighbors("a", direction="out")
        found = _find_query(conn, "FROM synapse WHERE")
        assert found is not None
        sql, _ = found
        assert "in.* AS in_neuron" in sql
        assert "out.* AS out_neuron" in sql
        assert "in = type::record('neuron', $nid)" in sql

    @pytest.mark.asyncio
    async def test_delete_neuron_cascade_uses_in_out(self):
        """Two single-field DELETEs, not one OR query — and brain-agnostic.

        A single "brain_id = ... AND (in = X OR out = X)" query measured
        ~1.2s/call live on SurrealDB 3.2.0 — the planner doesn't use either
        idx_synapse_in/idx_synapse_out across an OR of two different fields.
        Splitting into two single-field DELETEs (each hits its own index)
        measured ~5ms total.

        The DELETEs match the neuron endpoint ALONE, with no ``brain_id``
        filter (audit DB-01): a synapse pointing at the deleted neuron becomes
        a dangling orphan regardless of its own brain_id, and some write paths
        create synapses with a NULL brain_id that a ``brain_id = '<brain>' AND``
        filter silently skipped. Matching the endpoint alone is both correct and
        index-friendly.
        """
        st, conn = _store_with_mock_conn()
        await st.delete_neuron("a")
        in_query = _find_query(conn, "WHERE in = neuron:")
        out_query = _find_query(conn, "WHERE out = neuron:")
        assert in_query is not None
        assert out_query is not None
        assert " OR " not in in_query[0]
        assert " OR " not in out_query[0]
        # brain-agnostic on purpose (audit DB-01): no brain_id filter, or the
        # NULL-brain_id orphan synapses this fix targets would be skipped again.
        assert "brain_id" not in in_query[0]
        assert "brain_id" not in out_query[0]


class TestBatchedPruneQueries:
    @pytest.mark.asyncio
    async def test_delete_synapses_batch_chunks_and_preserves_change_log(self):
        st, _ = _store_with_mock_conn()
        row = {
            "id": _FakeRID("synapse", "edge_1"),
            "in": _FakeRID("neuron", "source_1"),
            "out": _FakeRID("neuron", "target_1"),
            "type": "related_to",
            "weight": 0.1,
        }
        st._query = AsyncMock(side_effect=[[row], []])  # type: ignore[method-assign]
        st._record_changes_bulk = AsyncMock()  # type: ignore[method-assign]

        count = await st.delete_synapses_batch([f"edge-{i}" for i in range(129)])

        assert count == 1
        assert st._query.await_count == 2
        first_sql = st._query.await_args_list[0].args[0]
        first_params = st._query.await_args_list[0].kwargs
        assert first_sql.startswith("DELETE FROM synapse")
        assert "brain_id = $brain_id" in first_sql
        assert "RETURN BEFORE" in first_sql
        assert "type::record('synapse', $synapse_id_0)" in first_sql
        assert first_params["synapse_id_0"] == "edge_0"
        logged = st._record_changes_bulk.await_args.args
        assert logged[:2] == ("synapse", "delete")
        assert [synapse.id for synapse in logged[2]] == ["edge-1"]

    @pytest.mark.asyncio
    async def test_get_synapses_by_ids_uses_bound_record_ids(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        assert await st.get_synapses_by_ids(["edge-1"]) == []

        sql = st._query.await_args.args[0]
        params = st._query.await_args.kwargs
        assert "id IN [type::record('synapse', $synapse_id_0)]" in sql
        assert params["synapse_id_0"] == "edge_1"

    @pytest.mark.asyncio
    async def test_get_synapses_for_sources_uses_bounded_source_list(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        assert await st.get_synapses_for_sources(["source-1"]) == []

        sql = st._query.await_args.args[0]
        params = st._query.await_args.kwargs
        assert "in IN [type::record('neuron', $source_id_0)]" in sql
        assert params["source_id_0"] == "source_1"

    @pytest.mark.asyncio
    async def test_target_counts_aggregate_without_returning_synapse_rows(self):
        st, _ = _store_with_mock_conn()
        st._query_response = AsyncMock(return_value=[[2]])  # type: ignore[method-assign]

        counts = await st.get_synapse_target_counts_for_sources(["source-1"])

        assert counts == {"source-1": 2}
        sql = st._query_response.await_args.args[0]
        params = st._query_response.await_args.kwargs
        assert "array::len(array::group(out))" in sql
        assert "GROUP ALL" in sql
        assert "in = type::record('neuron', $source_id_0)" in sql
        assert params["source_id_0"] == "source_1"

    @pytest.mark.asyncio
    async def test_target_counts_split_indexed_subqueries_into_bounded_batches(self):
        st, _ = _store_with_mock_conn()
        st._query_response = AsyncMock(
            side_effect=[[[1]] * 128, [[2]]]
        )  # type: ignore[method-assign]
        sources = [f"source-{index}" for index in range(129)]

        counts = await st.get_synapse_target_counts_for_sources(sources)

        assert counts == {source: (1 if index < 128 else 2) for index, source in enumerate(sources)}
        assert st._query_response.await_count == 2
        assert "$source_id_127" in st._query_response.await_args_list[0].args[0]
        assert "$source_id_128" not in st._query_response.await_args_list[0].args[0]

    @pytest.mark.asyncio
    async def test_synapse_keyset_page_binds_frozen_reference_and_cursor(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        page = await st.get_synapses_after_id(
            "edge-10",
            limit=9999,
            created_before=datetime(2026, 9, 20),
        )

        assert page == []
        sql = st._query.await_args.args[0]
        params = st._query.await_args.kwargs
        assert "id > type::record('synapse', $cursor_id)" in sql
        assert "created_at IS NONE OR created_at <= $created_before" in sql
        assert "ORDER BY id ASC LIMIT 2000" in sql
        assert params["cursor_id"] == "edge_10"

    @pytest.mark.asyncio
    async def test_prune_page_uses_projected_tuple_cursor_without_upper_bound(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        cursor_time = datetime(2026, 9, 19, 8, 15)
        page = await st.get_synapse_prune_page(
            cursor_time,
            "edge-10",
            limit=9999,
        )

        assert page == []
        sql = st._query.await_args.args[0]
        params = st._query.await_args.kwargs
        assert sql.startswith(
            "SELECT id, brain_id, in, out, type, weight, direction, metadata, "
            "created_at, last_activated, reinforced_count FROM synapse"
        )
        assert "created_at >= $cursor_time" in sql
        assert "(created_at > $cursor_time OR (created_at = $cursor_time AND " in sql
        assert "id > type::record('synapse', $cursor_id)" in sql
        assert "created_at <= $created_before" not in sql
        assert "ORDER BY created_at ASC, id ASC LIMIT 2000" in sql
        assert params["cursor_id"] == "edge_10"
        assert params["cursor_time"] == cursor_time
        assert params["brain_id"] == "b1"

    @pytest.mark.asyncio
    async def test_prune_first_page_uses_creation_time_order(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await st.get_synapse_prune_page(None, None, limit=100)

        sql = st._query.await_args.args[0]
        assert "brain_id = $brain_id" in sql
        assert "ORDER BY created_at ASC, id ASC LIMIT 100" in sql
        assert "SELECT * FROM synapse" not in sql

    @pytest.mark.asyncio
    async def test_neuron_keyset_page_omits_embeddings_and_is_brain_scoped(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]

        page = await st.find_neurons_after_id(
            "node-10",
            created_before=datetime(2026, 9, 20),
            ephemeral=False,
            include_embedding=False,
        )

        assert page == []
        sql = st._query.await_args.args[0]
        params = st._query.await_args.kwargs
        assert "SELECT * OMIT embedding_vec FROM neuron" in sql
        assert 'brain_id = "b1"' in sql
        assert "id > type::record('neuron', $cursor_id)" in sql
        assert "created_at IS NONE OR created_at <= $created_before" in sql
        assert "ephemeral = $ephemeral" in sql
        assert params["cursor_id"] == "node_10"
        assert params["ephemeral"] is False

    @pytest.mark.asyncio
    async def test_get_connected_neuron_ids_for_only_queries_supplied_endpoints(self):
        st, _ = _store_with_mock_conn()
        st._query_values = AsyncMock(  # type: ignore[method-assign]
            side_effect=[[_FakeRID("neuron", "node_1")], []]
        )

        connected = await st.get_connected_neuron_ids_for(["node-1", "node-2"])

        assert connected == {"node-1"}
        assert st._query_values.await_count == 2
        sql = st._query_values.await_args_list[0].args[0]
        assert "in IN [type::record('neuron', $incoming_id_0)" in sql
        assert st._query_values.await_args_list[0].kwargs["incoming_id_1"] == "node_2"

    @pytest.mark.asyncio
    async def test_fiber_membership_is_unbounded_by_a_global_limit(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"neuron_ids": ["node-1", "unrequested"]}]
        )

        protected = await st.get_fiber_neuron_ids_for(["node-1"], min_salience=0.8)

        assert protected == {"node-1"}
        sql = st._query.await_args.args[0]
        assert "neuron_ids CONTAINSANY $neuron_ids" in sql
        assert "salience > $min_salience" in sql
        assert "LIMIT" not in sql

    @pytest.mark.asyncio
    async def test_remove_synapse_refs_from_fibers_uses_set_difference_in_chunks(self):
        st, _ = _store_with_mock_conn()
        st._query = AsyncMock(return_value=[])  # type: ignore[method-assign]
        st._record_changes_bulk = AsyncMock()  # type: ignore[method-assign]

        updated = await st.remove_synapse_refs_from_fibers([f"edge-{i}" for i in range(129)])

        assert updated == 0
        assert st._query.await_count == 2
        sql = st._query.await_args_list[0].args[0]
        assert "array::difference(synapse_ids, $synapse_ids)" in sql
        assert "synapse_ids CONTAINSANY $synapse_ids" in sql
