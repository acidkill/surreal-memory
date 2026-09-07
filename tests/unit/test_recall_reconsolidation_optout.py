"""smem_recall's reconsolidate opt-out (issue #198).

Recall writes by design — deferred-write flush plus reconsolidation of the top
matched fibers. The per-brain `reconsolidation_enabled` switch is the global
source of truth; these tests pin the per-call opt-out: the MCP tool accepts
`reconsolidate` (bool), threads it into ReflexPipeline.query, defaults to True,
and rejects non-boolean values. The engine-side gate is additionally pinned at
the signature level.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_server() -> Any:
    from surreal_memory.mcp.server import MCPServer

    server = MCPServer.__new__(MCPServer)
    server._config = MagicMock()
    server._config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
    server._config.safety = MagicMock(auto_redact_min_severity=3)
    server._config.auto = MagicMock(enabled=False)
    server._config.dedup = MagicMock(enabled=False)
    server._config.tool_tier = MagicMock(tier="full")
    server._storage = None
    server._hooks = None
    server._eternal_trigger_count = 0
    server.config = MagicMock()
    server.config.auto.enabled = False  # skip passive capture on long queries
    hooks = MagicMock()
    hooks.emit = AsyncMock(return_value=None)
    server.hooks = hooks
    fake_storage = MagicMock()
    fake_storage.current_brain_id = "b1"
    fake_storage.get_brain = AsyncMock(return_value=SimpleNamespace(id="b1", config=MagicMock()))
    server.get_storage = AsyncMock(return_value=fake_storage)
    return server


def _patched_pipeline(monkeypatch: pytest.MonkeyPatch, result: Any) -> dict[str, Any]:
    """Replace ReflexPipeline inside recall_handler with a capturing fake."""
    import surreal_memory.engine.retrieval as retrieval_mod

    captured: dict[str, Any] = {}

    class _FakePipeline:
        def __init__(self, storage: Any, config: Any) -> None:
            captured["storage"] = storage

        async def query(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return result

    # recall_handler imports ReflexPipeline inside the function body, so the
    # local `from ... import` resolves against the SOURCE module each call.
    monkeypatch.setattr(retrieval_mod, "ReflexPipeline", _FakePipeline)
    return captured


_RESULT = SimpleNamespace(
    answer="a",
    confidence=0.9,
    fibers_matched=[],
    neurons_activated=1,
    context="c",
    latency_ms=1.0,
    synthesis_method="test",
    metadata={},
    subgraph=SimpleNamespace(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
    co_activations={},
    depth_used=SimpleNamespace(value=1),
    tokens_used=10,
    score_breakdown=None,
)


@pytest.mark.asyncio
async def test_reconsolidate_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patched_pipeline(monkeypatch, _RESULT)
    server = _make_server()
    await server._recall({"query": "how did we configure the gateway"})
    assert captured["reconsolidate"] is True


@pytest.mark.asyncio
async def test_reconsolidate_false_threads_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patched_pipeline(monkeypatch, _RESULT)
    server = _make_server()
    await server._recall({"query": "how did we configure the gateway", "reconsolidate": False})
    assert captured["reconsolidate"] is False


@pytest.mark.asyncio
async def test_reconsolidate_non_boolean_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patched_pipeline(monkeypatch, _RESULT)
    server = _make_server()
    result = await server._recall({"query": "q", "reconsolidate": "no"})
    assert "error" in result and "reconsolidate" in result["error"]


def test_engine_query_signature_has_the_optout_defaulting_true() -> None:
    """The engine contract: ReflexPipeline.query carries the per-call opt-out."""
    from surreal_memory.engine.retrieval import ReflexPipeline

    params = inspect.signature(ReflexPipeline.query).parameters
    assert "reconsolidate" in params, "query() must expose the per-call opt-out"
    assert params["reconsolidate"].default is True
