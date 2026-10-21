"""Tests for the PostToolUse hook (perf/light rewrite, ac2a001 subset)."""

from __future__ import annotations

import importlib
import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from surreal_memory.hooks.post_tool_use import (
    _NOISE_TOOLS,
    _append_to_buffer,
    _check_buffer_rotation,
    _format_event,
    _get_blacklist,
    _get_buffer_path,
    _get_session_id,
    _is_enabled,
    _read_stdin,
    _read_tool_memory_config,
    _truncate_args,
    _utcnow_iso,
    main,
)


class TestTruncateArgs:
    def test_short_args(self) -> None:
        result = _truncate_args({"query": "test"})
        assert len(result) <= 200
        assert "query" in result

    def test_long_args(self) -> None:
        big_input = {"data": "x" * 500}
        result = _truncate_args(big_input)
        assert len(result) == 200

    def test_none_args(self) -> None:
        assert _truncate_args(None) == ""

    def test_non_serializable(self) -> None:
        """Falls back to str() for non-serializable objects."""
        result = _truncate_args(object())
        assert len(result) > 0


class TestNoiseTools:
    def test_noise_tools_is_frozenset(self) -> None:
        assert isinstance(_NOISE_TOOLS, frozenset)

    def test_contains_known_noise(self) -> None:
        assert "TodoRead" in _NOISE_TOOLS
        assert "TodoWrite" in _NOISE_TOOLS
        assert "WebSearch" in _NOISE_TOOLS
        assert "smem_context" in _NOISE_TOOLS
        assert "smem_recap" in _NOISE_TOOLS
        assert "smem_explain" in _NOISE_TOOLS

    def test_does_not_contain_read(self) -> None:
        assert "Read" not in _NOISE_TOOLS
        assert "Bash" not in _NOISE_TOOLS


class TestUtcnowIso:
    def test_returns_string(self) -> None:
        result = _utcnow_iso()
        assert isinstance(result, str)
        assert "T" in result

    def test_no_tzinfo_suffix(self) -> None:
        """Naive ISO string — no +00:00 suffix (matches storage convention)."""
        result = _utcnow_iso()
        assert "+" not in result
        assert "Z" not in result


class TestGetSessionId:
    def test_claude_session_id(self) -> None:
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "test-session-123"}, clear=False):
            assert _get_session_id() == "test-session-123"

    def test_codex_session_id_fallback(self) -> None:
        env = {"CODEX_SESSION_ID": "codex-456"}
        with patch.dict(os.environ, env, clear=False):
            # Remove CLAUDE_SESSION_ID if present
            env_without_claude = {k: v for k, v in os.environ.items() if k != "CLAUDE_SESSION_ID"}
            env_without_claude["CODEX_SESSION_ID"] = "codex-456"
            with patch.dict(os.environ, env_without_claude, clear=True):
                assert _get_session_id() == "codex-456"

    def test_claude_takes_priority_over_codex(self) -> None:
        with patch.dict(
            os.environ,
            {"CLAUDE_SESSION_ID": "claude-abc", "CODEX_SESSION_ID": "codex-xyz"},
            clear=False,
        ):
            assert _get_session_id() == "claude-abc"

    def test_empty_when_neither_set(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CLAUDE_SESSION_ID", "CODEX_SESSION_ID")
        }
        with patch.dict(os.environ, env, clear=True):
            assert _get_session_id() == ""


class TestFormatEvent:
    def test_basic_format(self) -> None:
        hook_input = {
            "tool_name": "Read",
            "server_name": "filesystem",
            "tool_input": {"path": "/tmp/test.py"},
            "duration_ms": 50,
        }
        event = _format_event(hook_input)
        assert event["tool_name"] == "Read"
        assert event["server_name"] == "filesystem"
        assert event["success"] is True
        assert event["duration_ms"] == 50
        assert "created_at" in event

    def test_error_event(self) -> None:
        hook_input = {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "tool_error": "Permission denied",
            "duration_ms": 10,
        }
        event = _format_event(hook_input)
        assert event["success"] is False

    def test_missing_fields(self) -> None:
        """Gracefully handles missing optional fields."""
        event = _format_event({"tool_name": "Read"})
        assert event["tool_name"] == "Read"
        assert event["server_name"] == ""
        assert event["duration_ms"] == 0
        assert event["success"] is True

    def test_invalid_duration(self) -> None:
        """Non-numeric duration defaults to 0."""
        event = _format_event({"tool_name": "Read", "duration_ms": "not-a-number"})
        assert event["duration_ms"] == 0

    def test_uses_tool_fallback_key(self) -> None:
        """Falls back from tool_name to tool key."""
        event = _format_event({"tool": "Write"})
        assert event["tool_name"] == "Write"

    def test_session_id_from_env(self) -> None:
        """session_id populated from env without heavy imports."""
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "sess-xyz"}, clear=False):
            event = _format_event({"tool_name": "Bash"})
        assert event["session_id"] == "sess-xyz"

    def test_no_heavy_imports(self) -> None:
        """_format_event must not import surreal_memory sub-modules."""
        import surreal_memory.hooks.post_tool_use as mod

        # Reload to ensure no lazy-import side effects
        importlib.reload(mod)
        event = mod._format_event({"tool_name": "Read"})
        assert "created_at" in event


class TestBufferRotation:
    def test_no_rotation_small_buffer(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        lines = [json.dumps({"tool_name": f"tool-{i}"}) for i in range(10)]
        buf.write_text("\n".join(lines) + "\n")

        _check_buffer_rotation(buf, max_lines=100)
        assert len(buf.read_text().splitlines()) == 10

    def test_rotation_large_buffer(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        lines = [json.dumps({"tool_name": f"tool-{i}"}) for i in range(200)]
        buf.write_text("\n".join(lines) + "\n")

        _check_buffer_rotation(buf, max_lines=100)
        remaining = buf.read_text().splitlines()
        assert len(remaining) == 100  # Kept newest half

    def test_rotation_missing_file(self, tmp_path: Path) -> None:
        buf = tmp_path / "nonexistent.jsonl"
        _check_buffer_rotation(buf)  # Should not raise


class TestReadStdin:
    def test_valid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO('{"tool_name": "Read"}')):
            result = _read_stdin()
        assert result == {"tool_name": "Read"}

    def test_empty_stdin(self) -> None:
        with patch.object(sys, "stdin", StringIO("")):
            result = _read_stdin()
        assert result == {}

    def test_invalid_json(self) -> None:
        with patch.object(sys, "stdin", StringIO("not json")):
            result = _read_stdin()
        assert result == {}


class TestReadToolMemoryConfig:
    def test_no_config_file(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}):
            assert _read_tool_memory_config() == {}

    def test_reads_tool_memory_section(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[tool_memory]\nenabled = false\nblacklist = ["Foo"]\n')
        with patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}):
            cfg = _read_tool_memory_config()
        assert cfg["enabled"] is False
        assert cfg["blacklist"] == ["Foo"]

    def test_missing_section_returns_empty(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[general]\nbrain = 'default'\n")
        with patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}):
            assert _read_tool_memory_config() == {}


class TestIsEnabled:
    def test_empty_cfg_defaults_true(self) -> None:
        assert _is_enabled({}) is True

    def test_enabled_true(self) -> None:
        assert _is_enabled({"enabled": True}) is True

    def test_enabled_false(self) -> None:
        assert _is_enabled({"enabled": False}) is False


class TestGetBlacklist:
    def test_empty_cfg_returns_empty(self) -> None:
        assert _get_blacklist({}) == []

    def test_blacklist_present(self) -> None:
        result = _get_blacklist({"blacklist": ["TodoRead", "TaskList"]})
        assert result == ["TodoRead", "TaskList"]

    def test_invalid_type_returns_empty(self) -> None:
        assert _get_blacklist({"blacklist": "not-a-list"}) == []


class TestGetBufferPath:
    def test_custom_dir(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}):
            path = _get_buffer_path()
        assert path == tmp_path / "tool_events.jsonl"


class TestAppendToBuffer:
    def test_creates_and_appends(self, tmp_path: Path) -> None:
        buf = tmp_path / "sub" / "events.jsonl"
        event = {"tool_name": "Read", "created_at": "2026-01-01T00:00:00"}
        assert _append_to_buffer(event, buf) is True
        assert buf.exists()
        data = json.loads(buf.read_text().strip())
        assert data["tool_name"] == "Read"

    def test_appends_multiple(self, tmp_path: Path) -> None:
        buf = tmp_path / "events.jsonl"
        _append_to_buffer({"tool_name": "Read"}, buf)
        _append_to_buffer({"tool_name": "Write"}, buf)
        lines = buf.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_concurrent_appends_produce_valid_jsonl(self, tmp_path: Path) -> None:
        """Multiple appends stay valid JSONL (each line parseable)."""
        buf = tmp_path / "events.jsonl"
        for i in range(20):
            _append_to_buffer({"tool_name": f"Tool{i}", "seq": i}, buf)
        lines = buf.read_text().strip().splitlines()
        assert len(lines) == 20
        for line in lines:
            parsed = json.loads(line)
            assert "tool_name" in parsed


class TestMain:
    def test_empty_stdin(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO("")),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_no_tool_name(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"server_name": "test"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"

    def test_noise_tool_skipped_without_config_read(self, tmp_path: Path) -> None:
        """_NOISE_TOOLS are skipped before any file I/O (no config needed)."""
        stdout = StringIO()
        # No config file — would default-enable, but noise check runs first
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "TodoRead"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        # No event file should be created
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_websearch_noise_skipped(self, tmp_path: Path) -> None:
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "WebSearch"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_namespaced_smem_read_tools_skipped(self, tmp_path: Path) -> None:
        for tool_name in (
            "smem_recall",
            "smem_context",
            "smem_recap",
            "smem_explain",
            "mcp__server__smem_recall",
            "mcp__server__smem_context",
            "mcp__server__smem_recap",
            "mcp__server__smem_explain",
        ):
            stdout = StringIO()
            with (
                patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
                patch.object(
                    sys,
                    "stdin",
                    StringIO(json.dumps({"tool_name": tool_name})),
                ),
                patch.object(sys, "stdout", stdout),
            ):
                main()
            assert stdout.getvalue().strip() == "{}"
            assert not (tmp_path / "tool_events.jsonl").exists()

    def test_namespaced_blacklist_matches_canonical_name(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text(
            '[tool_memory]\nenabled = true\nblacklist = ["smem_"]\n'
        )
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(
                sys,
                "stdin",
                StringIO('{"tool_name": "mcp__server__smem_custom"}'),
            ),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_preview_full_name_and_legitimate_tools_unchanged(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text("[tool_memory]\nenabled = true\n")
        for tool_name in ("mcp__Claude_Preview__preview_logs", "Bash", "Read"):
            stdout = StringIO()
            with (
                patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
                patch.object(
                    sys,
                    "stdin",
                    StringIO(json.dumps({"tool_name": tool_name})),
                ),
                patch.object(sys, "stdout", stdout),
            ):
                main()
            assert stdout.getvalue().strip() == "{}"
            buffer_path = tmp_path / "tool_events.jsonl"
            if tool_name == "mcp__Claude_Preview__preview_logs":
                assert not buffer_path.exists()
            else:
                event = json.loads(buffer_path.read_text().splitlines()[-1])
                assert event["tool_name"] == tool_name

    def test_blacklisted_tool(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[tool_memory]\nenabled = true\nblacklist = ["CustomNoise"]\n')
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "CustomNoiseRead"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_disabled_exits_fast(self, tmp_path: Path) -> None:
        """When tool memory is disabled, main outputs empty JSON."""
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = false\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Read"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        assert not (tmp_path / "tool_events.jsonl").exists()

    def test_successful_capture(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Read", "duration_ms": 25}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        buf = tmp_path / "tool_events.jsonl"
        assert buf.exists()
        event = json.loads(buf.read_text().strip())
        assert event["tool_name"] == "Read"
        assert event["success"] is True

    def test_codex_session_id_captured(self, tmp_path: Path) -> None:
        """CODEX_SESSION_ID is captured as session_id when CLAUDE_SESSION_ID absent."""
        config = tmp_path / "config.toml"
        config.write_text("[tool_memory]\nenabled = true\n")
        stdout = StringIO()
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_SESSION_ID"}
        env["SURREAL_MEMORY_DIR"] = str(tmp_path)
        env["CODEX_SESSION_ID"] = "codex-test-session"
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Bash"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        buf = tmp_path / "tool_events.jsonl"
        assert buf.exists()
        event = json.loads(buf.read_text().strip())
        assert event["session_id"] == "codex-test-session"

    def test_no_config_defaults_enabled(self, tmp_path: Path) -> None:
        """Without config.toml, tool memory is enabled by default."""
        stdout = StringIO()
        with (
            patch.dict(os.environ, {"SURREAL_MEMORY_DIR": str(tmp_path)}),
            patch.object(sys, "stdin", StringIO('{"tool_name": "Bash"}')),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert stdout.getvalue().strip() == "{}"
        buf = tmp_path / "tool_events.jsonl"
        assert buf.exists()
        event = json.loads(buf.read_text().strip())
        assert event["tool_name"] == "Bash"
