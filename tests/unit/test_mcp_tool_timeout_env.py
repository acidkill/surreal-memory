"""SURREAL_MEMORY_MCP_TOOL_TIMEOUT drives the MCP tool-call budget.

A client talking to a remote SurrealDB over wss can legitimately need more than
the default 30 s per tool call (recall = embedding + vector search + rerank
across the wire; issue #229). These tests pin the resolver contract: seconds
(not milliseconds), a 600 s ceiling against ms-typos, and warn-and-default
instead of refusing to start on garbage.
"""

from __future__ import annotations

import importlib

import pytest

from surreal_memory.mcp import server as mcp_server


class TestResolveToolCallTimeout:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, 30.0),
            ("", 30.0),
            ("   ", 30.0),
            ("180", 180.0),
            ("12.5", 12.5),
            ("0.5", 0.5),
        ],
    )
    def test_valid_and_unset_values(self, raw: str | None, expected: float) -> None:
        assert mcp_server._resolve_tool_call_timeout(raw) == expected

    @pytest.mark.parametrize("raw", ["abc", "30s", "1e", ","])
    def test_non_numeric_falls_back_with_warning(
        self, raw: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="surreal_memory.mcp.server"):
            assert mcp_server._resolve_tool_call_timeout(raw) == 30.0
        assert any("not a number" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("raw", ["0", "-5", "-0.1"])
    def test_non_positive_falls_back_with_warning(
        self, raw: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="surreal_memory.mcp.server"):
            assert mcp_server._resolve_tool_call_timeout(raw) == 30.0
        assert any("must be positive" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("raw", ["601", "180000", "1e9"])
    def test_over_ceiling_clamps_with_warning(
        self, raw: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="surreal_memory.mcp.server"):
            assert mcp_server._resolve_tool_call_timeout(raw) == 600.0
        assert any("ceiling" in r.getMessage() for r in caplog.records)

    def test_ceiling_boundary_is_exact(self) -> None:
        assert mcp_server._resolve_tool_call_timeout("600") == 600.0


class TestEnvWiring:
    def test_module_constant_follows_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The constant is resolved from the environment at import time, so a
        configured deployment (e.g. SURREAL_MEMORY_MCP_TOOL_TIMEOUT=180 for a
        remote-DB client) actually changes the budget the dispatcher uses."""
        monkeypatch.setenv("SURREAL_MEMORY_MCP_TOOL_TIMEOUT", "180")
        reloaded = importlib.reload(mcp_server)
        try:
            assert reloaded._TOOL_CALL_TIMEOUT == 180.0
        finally:
            monkeypatch.delenv("SURREAL_MEMORY_MCP_TOOL_TIMEOUT", raising=False)
            importlib.reload(mcp_server)

    def test_module_constant_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_MCP_TOOL_TIMEOUT", raising=False)
        reloaded = importlib.reload(mcp_server)
        try:
            assert reloaded._TOOL_CALL_TIMEOUT == 30.0
        finally:
            importlib.reload(mcp_server)
