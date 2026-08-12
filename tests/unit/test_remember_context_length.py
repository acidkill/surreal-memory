"""smem_remember: `context` must count toward MAX_CONTENT_LENGTH.

`_remember` validated `len(content)` BEFORE `merge_context()` ran, but
`merge_context` appends the caller-supplied `context` dict into the string that
is actually persisted and has no size cap of its own. A call with a tiny
`content` and a huge `context` therefore sailed past the check and stored a
memory far above MAX_CONTENT_LENGTH — the boundary the constant exists to
enforce was bypassed for anyone routing size through `context`.

The batch path already counts `context` toward MAX_BATCH_TOTAL_CHARS
(see TestBulkRemember in test_baby_mi_features.py); these tests cover the
single-item root cause and pin the common case as unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.mcp.constants import MAX_CONTENT_LENGTH
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig, WriteGateConfig


class _StopEncodeError(Exception):
    """Sentinel to stop _remember right after encode is invoked."""


def _make_server() -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
        cfg = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
        )
        cfg.write_gate = WriteGateConfig()  # real config (mode="off") so the write-gate is skipped
        cfg.encryption.enabled = False
        cfg.safety.auto_redact_min_severity = 3
        mock_get_config.return_value = cfg
        return MCPServer()


def _brain_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.get_brain = AsyncMock(return_value=MagicMock(id="test-brain", config=MagicMock()))
    storage._current_brain_id = "test-brain"
    storage.brain_id = "test-brain"
    return storage


async def _run_remember(
    server: MCPServer, args: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run _remember with the encoder stubbed out.

    Returns the handler response and whatever kwargs reached `encoder.encode`
    (empty dict when the handler returned before encoding).
    """
    captured: dict[str, Any] = {}

    async def _capture_encode(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _StopEncodeError  # short-circuit the rest of _remember

    with (
        patch.object(server, "get_storage", return_value=_brain_storage()),
        patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
        patch.object(server, "_fire_eternal_trigger"),
        patch.object(server, "_record_tool_action", new_callable=AsyncMock),
        patch.object(server, "_passive_capture", new_callable=AsyncMock),
        patch("surreal_memory.mcp.remember_handler.MemoryEncoder") as mock_encoder_cls,
    ):
        mock_encoder_cls.return_value.encode = _capture_encode
        try:
            response = await server._remember(args)
        except _StopEncodeError:
            response = {"_encoded": True}
    return response, captured


class TestRememberContextCountsTowardMaxContentLength:
    @pytest.fixture
    def server(self) -> MCPServer:
        return _make_server()

    async def test_oversized_context_is_rejected(self, server: MCPServer) -> None:
        """Tiny content + oversized context must be rejected, not stored."""
        big_context = {"reason": "x" * (MAX_CONTENT_LENGTH + 50_000)}

        response, captured = await _run_remember(
            server, {"content": "tiny note", "type": "decision", "context": big_context}
        )

        assert "error" in response
        assert "too long" in response["error"].lower()
        # Nothing was encoded — the oversized string never reached storage.
        assert captured == {}

    async def test_oversized_context_reports_the_merged_size(self, server: MCPServer) -> None:
        """The error must report the post-merge size, so the caller can see that
        `context` counts toward the limit rather than only `content`."""
        content = "y" * 60_000
        big_context = {"reason": "x" * 60_000}

        response, _ = await _run_remember(
            server, {"content": content, "type": "decision", "context": big_context}
        )

        assert "error" in response
        assert "after merging context" in response["error"]
        assert str(MAX_CONTENT_LENGTH) in response["error"]

    async def test_normal_content_and_context_still_merge_unchanged(
        self, server: MCPServer
    ) -> None:
        """Regression guard: the common case keeps its exact prior behavior."""
        response, captured = await _run_remember(
            server,
            {
                "content": "Chose SurrealDB",
                "type": "decision",
                "context": {
                    "reason": "it fits the graph model",
                    "alternatives": ["sqlite", "postgres"],
                },
            },
        )

        assert response == {"_encoded": True}
        merged = captured["content"]
        assert merged.startswith("Chose SurrealDB")
        assert "because it fits the graph model" in merged
        assert "Alternatives considered: sqlite, postgres" in merged

    async def test_merged_content_just_under_the_limit_is_accepted(self, server: MCPServer) -> None:
        """The guard bounds the merged string only — a large but legal
        content+context combination must still be stored."""
        content = "y" * 40_000
        context = {"reason": "x" * 50_000}

        response, captured = await _run_remember(
            server, {"content": content, "type": "decision", "context": context}
        )

        assert response == {"_encoded": True}
        assert len(captured["content"]) <= MAX_CONTENT_LENGTH

    async def test_content_only_check_still_fires_before_merge(self, server: MCPServer) -> None:
        """The pre-existing content-only guard is untouched."""
        response, captured = await _run_remember(
            server, {"content": "z" * (MAX_CONTENT_LENGTH + 1)}
        )

        assert "error" in response
        assert "too long" in response["error"].lower()
        assert "after merging context" not in response["error"]
        assert captured == {}
