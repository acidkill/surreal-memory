"""Unit tests for the ISO GQL path fast-path + BFS fallback (RUN-005 U5).

The surrealdb SDK is stubbed and the connection is mocked, so these assert the
routing/parsing logic (probe on/off, GQL used when it yields a verified path,
BFS fallback on GQL error/None, learn-once self-disable). Live behaviour on a
real v3.2.0 server is covered by the U5 live smoke and the U6 integration test.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Stub the optional surrealdb SDK ONLY when it is genuinely not installed: an
# `if not in sys.modules` guard would shadow an installed SDK for the rest of
# the pytest session and break the live (SURREALDB_URL) tests running later.
try:
    import surrealdb  # noqa: F401
except ImportError:  # pragma: no cover - CI unit env has no surrealdb SDK
    sys.modules["surrealdb"] = MagicMock()
    sys.modules["surrealdb.errors"] = MagicMock()

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _store() -> SurrealDBStorage:
    st = SurrealDBStorage(url="http://localhost:8001")
    st._conn = AsyncMock()
    st._current_brain_id = "b1"
    return st


def _neuron(nid: str, content: str = "x") -> Neuron:
    return Neuron.create(type=NeuronType.CONCEPT, content=content, neuron_id=nid)


def _synapse(sid: str, src: str, tgt: str) -> Synapse:
    return Synapse.create(src, tgt, SynapseType.RELATED_TO, synapse_id=sid)


# --------------------------------------------------------------------------- #
# Capability probe (in initialize)
# --------------------------------------------------------------------------- #
class TestGqlProbe:
    @pytest.mark.asyncio
    async def test_probe_enables_gql_on_success(self):
        st = SurrealDBStorage()
        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None
        mock_conn.version.return_value = "surrealdb-3.2.0"
        mock_conn.query.return_value = [{"n": {}}]  # eval::gql probe succeeds

        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await st.initialize()
        assert st.gql_available is True

    @pytest.mark.asyncio
    async def test_probe_disables_gql_on_capability_error(self):
        st = SurrealDBStorage()
        mock_conn = AsyncMock()
        mock_conn.signin.return_value = None
        mock_conn.use.return_value = None
        mock_conn.version.return_value = "surrealdb-3.2.0"
        mock_conn.query.side_effect = RuntimeError("Experimental capability `gql` is not enabled")

        with (
            patch("surrealdb.AsyncSurreal", return_value=mock_conn, create=True),
            patch("surreal_memory.storage.surrealdb.store.ensure_schema", new_callable=AsyncMock),
            patch(
                "surreal_memory.storage.surrealdb.migrations.apply_migrations",
                new_callable=AsyncMock,
            ),
        ):
            await st.initialize()
        assert st.gql_available is False


# --------------------------------------------------------------------------- #
# get_path routing: GQL fast-path vs BFS fallback
# --------------------------------------------------------------------------- #
class TestGetPathRouting:
    @pytest.mark.asyncio
    async def test_uses_gql_when_it_returns_verified_path(self):
        st = _store()
        st._gql_available = True
        expected = [(_neuron("c"), _synapse("e2", "b", "c"))]
        st._get_path_gql = AsyncMock(return_value=expected)
        st.get_neighbors = AsyncMock(side_effect=AssertionError("BFS must not run"))

        result = await st.get_path("a", "c", max_hops=4)
        assert result is expected
        st._get_path_gql.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_bfs_when_gql_raises(self):
        st = _store()
        st._gql_available = True
        st._get_path_gql = AsyncMock(side_effect=RuntimeError("gql boom"))
        # BFS finds a -> c directly.
        st.get_neighbors = AsyncMock(return_value=[(_neuron("c"), _synapse("e", "a", "c"))])

        result = await st.get_path("a", "c", max_hops=4)
        assert result is not None
        assert result[-1][0].id == "c"
        st.get_neighbors.assert_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_bfs_when_gql_returns_none(self):
        st = _store()
        st._gql_available = True
        st._get_path_gql = AsyncMock(return_value=None)
        st.get_neighbors = AsyncMock(return_value=[(_neuron("c"), _synapse("e", "a", "c"))])

        result = await st.get_path("a", "c", max_hops=4)
        assert result is not None and result[-1][0].id == "c"

    @pytest.mark.asyncio
    async def test_gql_not_attempted_when_unavailable(self):
        st = _store()
        st._gql_available = False
        st._get_path_gql = AsyncMock(side_effect=AssertionError("GQL must not be attempted"))
        st.get_neighbors = AsyncMock(return_value=[(_neuron("c"), _synapse("e", "a", "c"))])

        result = await st.get_path("a", "c", max_hops=4)
        assert result is not None and result[-1][0].id == "c"
        st._get_path_gql.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_self_disables_after_repeated_misses(self):
        st = _store()
        st._gql_available = True
        st._get_path_gql = AsyncMock(return_value=None)  # always a miss
        st.get_neighbors = AsyncMock(return_value=[(_neuron("c"), _synapse("e", "a", "c"))])

        for _ in range(3):
            await st.get_path("a", "c", max_hops=4)
        assert st._gql_available is False
        # once disabled, GQL is no longer attempted
        st._get_path_gql.reset_mock()
        await st.get_path("a", "c", max_hops=4)
        st._get_path_gql.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_self_loop_short_circuits_before_gql(self):
        st = _store()
        st._gql_available = True
        st._get_path_gql = AsyncMock(side_effect=AssertionError("self-loop must not hit GQL"))
        st.get_neuron = AsyncMock(return_value=_neuron("a"))

        result = await st.get_path("a", "a")
        assert result is not None and len(result) == 1


# --------------------------------------------------------------------------- #
# _get_path_gql: parsing + endpoint verification
# --------------------------------------------------------------------------- #
class TestGetPathGqlParsing:
    def _node_row(self, nid: str, content: str) -> dict:
        return {"id": _Rid("neuron", nid), "type": "concept", "content": content}

    def _edge_row(self, sid: str, src: str, tgt: str) -> dict:
        return {
            "id": _Rid("synapse", sid),
            "in": _Rid("neuron", src),
            "out": _Rid("neuron", tgt),
            "type": "related_to",
        }

    @pytest.mark.asyncio
    async def test_maps_and_verifies_valid_path(self):
        st = _store()
        # path a -> b -> c: [nodeA, edgeAB, nodeB, edgeBC, nodeC]
        path_seq = [
            self._node_row("a", "a"),
            self._edge_row("e1", "a", "b"),
            self._node_row("b", "b"),
            self._edge_row("e2", "b", "c"),
            self._node_row("c", "c"),
        ]
        st._query = AsyncMock(return_value=[{"p": path_seq}])
        result = await st._get_path_gql("a", "c", 4, False)
        assert result is not None
        assert [n.id for n, _ in result] == ["b", "c"]
        assert result[-1][0].id == "c"

    @pytest.mark.asyncio
    async def test_returns_none_when_endpoint_mismatch(self):
        st = _store()
        # path ends at "d", but caller asked for target "c" -> reject
        path_seq = [
            self._node_row("a", "a"),
            self._edge_row("e1", "a", "d"),
            self._node_row("d", "d"),
        ]
        st._query = AsyncMock(return_value=[{"p": path_seq}])
        result = await st._get_path_gql("a", "c", 4, False)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_gql_result(self):
        st = _store()
        st._query = AsyncMock(return_value=[{"p": None}])
        assert await st._get_path_gql("a", "c", 4, False) is None
        st._query = AsyncMock(return_value=[])
        assert await st._get_path_gql("a", "c", 4, False) is None


class _Rid:
    """Minimal RecordID stand-in with .table_name/.id and str() == 'table:id'."""

    def __init__(self, table: str, ident: str) -> None:
        self.table_name = table
        self.id = ident

    def __str__(self) -> str:
        return f"{self.table_name}:{self.id}"
