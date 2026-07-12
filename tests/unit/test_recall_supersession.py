"""U3: recall-time supersession hard-filter + point-in-time (valid_at) + escape hatch.

The ONE intended default-behaviour change of v2.9.0: facts whose typed_memory has
``valid_until`` set (i.e. they have been superseded) are dropped from recall by
default. Callers opt back in per-call with ``include_superseded=true``; or disable
the hard filter process-wide with the ``SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER``
escape hatch (superseded facts then still surface, demoted 0.25x elsewhere).

Uses a REAL ``RetrievalResult`` dataclass (not a MagicMock) so the post-filter's
``dataclasses.replace`` rewrite of ``fibers_matched`` is exercised on the same
object shape production uses. The pipeline itself is mocked to return a fixed
result; storage returns real ``TypedMemory`` objects so validity logic runs.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.core.memory_types import MemoryType, Priority, Provenance, TypedMemory
from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig
from surreal_memory.utils.timeutils import utcnow


def _emma_typed_memories() -> dict[str, TypedMemory]:
    """Emma lived in Oslo (superseded), now lives in Bergen (current)."""
    now = utcnow()
    t_old = now - timedelta(days=30)
    oslo = TypedMemory(
        fiber_id="f-oslo",
        memory_type=MemoryType.FACT,
        priority=Priority.from_int(5),
        provenance=Provenance(source="test"),
        created_at=t_old,
        valid_from=t_old,
        valid_until=now,  # stopped being true at `now`
        superseded_by="f-bergen",
    )
    bergen = TypedMemory(
        fiber_id="f-bergen",
        memory_type=MemoryType.FACT,
        priority=Priority.from_int(5),
        provenance=Provenance(source="test"),
        created_at=now,
        valid_from=now,
        valid_until=None,  # still true / open-ended
    )
    return {"f-oslo": oslo, "f-bergen": bergen}


def _make_result() -> RetrievalResult:
    """A real RetrievalResult matching both Emma fibers, oslo ranked first."""
    return RetrievalResult(
        answer="Emma lives in Oslo. Emma lives in Bergen.",
        confidence=0.9,
        depth_used=DepthLevel.INSTANT,
        neurons_activated=2,
        fibers_matched=["f-oslo", "f-bergen"],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
        context="Emma lives in Oslo. Emma lives in Bergen.",
        latency_ms=1.0,
        tokens_used=10,
        metadata={},
        score_breakdown=None,
    )


def _make_server() -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
        cfg = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
        )
        cfg.write_gate.enabled = False
        mock_get_config.return_value = cfg
        return MCPServer()


def _fiber(fid: str) -> MagicMock:
    f = MagicMock()
    f.id = fid
    f.anchor_neuron_id = f"anchor-{fid}"
    f.metadata = {}  # real dict → .get("encrypted") is None (no decryption path)
    f.created_at = utcnow()
    return f


def _neuron(nid: str) -> MagicMock:
    n = MagicMock()
    n.id = nid
    n.content = f"content-{nid}"
    n.metadata = {}
    return n


async def _run_recall(
    server: MCPServer, extra_args: dict[str, Any], tms: dict[str, TypedMemory]
) -> dict[str, Any]:
    mock_storage = AsyncMock()
    mock_storage.get_brain = AsyncMock(return_value=MagicMock(id="test-brain", config=MagicMock()))
    mock_storage._current_brain_id = "test-brain"
    mock_storage.brain_id = "test-brain"
    mock_storage.get_typed_memory = AsyncMock(side_effect=lambda fid: tms.get(fid))
    mock_storage.get_fiber = AsyncMock(side_effect=lambda fid: _fiber(fid))
    mock_storage.get_neuron = AsyncMock(side_effect=lambda nid: _neuron(nid))

    result_obj = _make_result()
    with (
        patch.object(server, "get_storage", return_value=mock_storage),
        patch("surreal_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
        patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
        patch.object(server, "_fire_eternal_trigger"),
        patch.object(server, "_record_tool_action", new_callable=AsyncMock),
        patch.object(server, "_passive_capture", new_callable=AsyncMock),
    ):
        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(return_value=result_obj)
        mock_pipeline_cls.return_value = mock_pipeline
        return await server.call_tool(
            "smem_recall", {"query": "where does emma live", **extra_args}
        )


class TestRecallSupersessionFilter:
    """The default-behaviour change: superseded facts drop from recall."""

    @pytest.fixture
    def server(self) -> MCPServer:
        return _make_server()

    @pytest.mark.asyncio
    async def test_superseded_hard_filtered_by_default(self, server: MCPServer) -> None:
        res = await _run_recall(server, {}, _emma_typed_memories())
        assert res["fibers_matched"] == ["f-bergen"]
        assert res["superseded_excluded_count"] == 1

    @pytest.mark.asyncio
    async def test_include_superseded_returns_both(self, server: MCPServer) -> None:
        res = await _run_recall(server, {"include_superseded": True}, _emma_typed_memories())
        assert res["fibers_matched"] == ["f-oslo", "f-bergen"]
        assert "superseded_excluded_count" not in res

    @pytest.mark.asyncio
    async def test_escape_hatch_env_disables_hard_filter(
        self, server: MCPServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER", "1")
        res = await _run_recall(server, {}, _emma_typed_memories())
        # Escape hatch on → superseded fact still surfaces (0.25x demotion applies
        # elsewhere, not here). No hard exclusion.
        assert res["fibers_matched"] == ["f-oslo", "f-bergen"]
        assert "superseded_excluded_count" not in res

    @pytest.mark.asyncio
    async def test_valid_at_point_in_time_keeps_old_fact(self, server: MCPServer) -> None:
        tms = _emma_typed_memories()
        # A moment while Emma still lived in Oslo (1 day into the old fact).
        when = (tms["f-oslo"].valid_from + timedelta(days=1)).isoformat()
        res = await _run_recall(server, {"valid_at": when}, tms)
        # oslo was valid then; bergen (valid_from=now) was not yet true.
        assert res["fibers_matched"] == ["f-oslo"]
        assert res["superseded_excluded_count"] == 1

    @pytest.mark.asyncio
    async def test_exact_mode_exposes_validity_fields(self, server: MCPServer) -> None:
        tms = _emma_typed_memories()
        res = await _run_recall(
            server,
            {"mode": "exact", "include_superseded": True, "include_citations": False},
            tms,
        )
        memories = {m["fiber_id"]: m for m in res["memories"]}
        assert set(memories) == {"f-oslo", "f-bergen"}
        # Superseded fact carries full lineage.
        assert memories["f-oslo"]["valid_from"] == tms["f-oslo"].valid_from.isoformat()
        assert memories["f-oslo"]["valid_until"] == tms["f-oslo"].valid_until.isoformat()
        assert memories["f-oslo"]["superseded_by"] == "f-bergen"
        # Current fact: open-ended → no valid_until / superseded_by keys.
        assert "valid_until" not in memories["f-bergen"]
        assert "superseded_by" not in memories["f-bergen"]

    @pytest.mark.asyncio
    async def test_answer_context_rebuilt_after_hard_filter(
        self, server: MCPServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the hard-filter drops a fiber from a REAL RetrievalResult dataclass,
        # result.context (surfaced as `answer`) must be rebuilt from the survivors so
        # the superseded fact's text does not linger in the primary answer field.
        # (Regression: the rebuild was gated on hasattr(result,"_replace"), a no-op on
        # the production dataclass, so answer kept the dropped fact's prose.)
        async def _fake_format_context(**_: object) -> tuple[str, dict[str, object]]:
            return ("Emma lives in Bergen.", {})

        monkeypatch.setattr(
            "surreal_memory.engine.retrieval_context.format_context", _fake_format_context
        )
        res = await _run_recall(server, {}, _emma_typed_memories())
        assert res["fibers_matched"] == ["f-bergen"]
        # answer regenerated from the surviving fiber only — no "Oslo" text.
        assert res["answer"] == "Emma lives in Bergen."
        assert "Oslo" not in res["answer"]
