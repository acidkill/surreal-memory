"""Unit tests for the v8 RELATION synapse query shapes in SurrealDBStorage.

The surrealdb SDK is stubbed (repo convention); the connection is an AsyncMock
whose query() records calls, so these assert the SurrealQL shape (native in/out,
type::record, INSERT RELATION, in.*/out.* inline) without a live DB. End-to-end
behaviour against a real v3.2.0 DB is covered by the U6 integration test.
"""

from __future__ import annotations

import sys
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
        """Two single-field DELETEs, not one OR query.

        A single "brain_id = ... AND (in = X OR out = X)" query measured
        ~1.2s/call live on SurrealDB 3.2.0 — the planner doesn't use either
        idx_synapse_in/idx_synapse_out across an OR of two different fields.
        Splitting into two single-field DELETEs (each hits its own index)
        measured ~5ms total.
        """
        st, conn = _store_with_mock_conn()
        await st.delete_neuron("a")
        in_query = _find_query(conn, "AND in = neuron:")
        out_query = _find_query(conn, "AND out = neuron:")
        assert in_query is not None
        assert out_query is not None
        assert " OR " not in in_query[0]
        assert " OR " not in out_query[0]
