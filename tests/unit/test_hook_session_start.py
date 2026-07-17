"""Tests for the SessionStart Claude Code hook."""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.hooks.session_start import get_recent_memories, main, read_hook_input


@pytest.mark.asyncio
async def test_get_recent_memories_no_project_name() -> None:
    """No resolved project returns empty string — no context injected."""
    mock_storage = AsyncMock()
    mock_storage.close = AsyncMock()

    mock_config = MagicMock()
    mock_config.current_brain = "test"

    with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            return_value=mock_storage,
        ):
            result = await get_recent_memories(None)

    assert result == ""


@pytest.mark.asyncio
async def test_get_recent_memories_no_project_memories() -> None:
    """Project with no memories returns empty string."""
    mock_storage = AsyncMock()
    mock_storage.get_project_memories = AsyncMock(return_value=[])
    mock_storage.close = AsyncMock()

    mock_config = MagicMock()
    mock_config.current_brain = "test"

    with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            return_value=mock_storage,
        ):
            result = await get_recent_memories("myproject")

    assert result == ""


@pytest.mark.asyncio
async def test_get_recent_memories_returns_formatted_bullets() -> None:
    """Project memories' fibers (summary/essence) are formatted as bullets."""
    fiber_with_summary = MagicMock()
    fiber_with_summary.summary = "Fixed auth bug in login.py"
    fiber_with_summary.essence = None

    fiber_with_essence_only = MagicMock()
    fiber_with_essence_only.summary = None
    fiber_with_essence_only.essence = "Auth bug"

    fiber_empty = MagicMock()
    fiber_empty.summary = None
    fiber_empty.essence = None

    tm1 = MagicMock()
    tm1.fiber_id = "f1"
    tm2 = MagicMock()
    tm2.fiber_id = "f2"
    tm3 = MagicMock()
    tm3.fiber_id = "f3"

    fibers_by_id = {"f1": fiber_with_summary, "f2": fiber_with_essence_only, "f3": fiber_empty}

    mock_storage = AsyncMock()
    mock_storage.get_project_memories = AsyncMock(return_value=[tm1, tm2, tm3])
    mock_storage.get_fiber = AsyncMock(side_effect=lambda fid: fibers_by_id.get(fid))
    mock_storage.close = AsyncMock()

    mock_config = MagicMock()
    mock_config.current_brain = "test"

    with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            return_value=mock_storage,
        ):
            result = await get_recent_memories("myproject")

    lines = result.splitlines()
    assert len(lines) == 2
    assert lines[0] == "- Fixed auth bug in login.py"
    assert lines[1] == "- Auth bug"


def test_main_malformed_stdin_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    """Malformed or empty stdin is handled gracefully — hook never blocks Claude Code."""
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch("asyncio.run", side_effect=RuntimeError("storage unavailable")):
            with pytest.raises(SystemExit) as exc_info:
                main()

    # Must exit 0 — never block Claude Code
    assert exc_info.value.code == 0


def test_read_hook_input_empty_stdin() -> None:
    """Empty stdin returns an empty dict without raising."""
    with patch("sys.stdin", io.StringIO("")):
        result = read_hook_input()
    assert result == {}


def test_read_hook_input_valid_json() -> None:
    """Valid JSON on stdin is parsed and returned."""
    payload = {"session_id": "abc123", "turn": 1}
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        result = read_hook_input()
    assert result == payload


def test_main_outputs_context_json_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    """When memories exist, main() writes a hookSpecificOutput JSON response."""
    with patch("sys.stdin", io.StringIO("{}")):
        with patch("asyncio.run", return_value="- Fixed auth bug\n- Added retry logic"):
            main()  # happy path — no sys.exit, just returns

    captured = capsys.readouterr()
    response = json.loads(captured.out.strip())
    hook_out = response["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "SessionStart"
    assert "Fixed auth bug" in hook_out["additionalContext"]


def test_main_exits_silently_when_no_memories(capsys: pytest.CaptureFixture[str]) -> None:
    """When the brain has no memories, main() exits 0 with no stdout output."""
    with patch("sys.stdin", io.StringIO("{}")):
        with patch("asyncio.run", return_value=""):
            with pytest.raises(SystemExit) as exc_info:
                main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""


_REASONING_BLOCK = "## Reasoning strategies (learned from claude-fable-5)\n\n1. **plan**"


def test_main_combines_memories_and_reasoning_blocks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both blocks present → main() emits one context with the memories block
    followed by the reasoning block."""
    with (
        patch("sys.stdin", io.StringIO("{}")),
        patch("surreal_memory.hooks.project_context.derive_project_name", return_value="proj"),
        patch(
            "surreal_memory.hooks.session_start.get_recent_memories",
            new=AsyncMock(return_value="- mem bullet"),
        ),
        patch(
            "surreal_memory.hooks.session_start.get_reasoning_injection",
            new=AsyncMock(return_value=_REASONING_BLOCK),
        ),
    ):
        main()

    response = json.loads(capsys.readouterr().out.strip())
    hook_out = response["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "SessionStart"
    content = hook_out["additionalContext"]
    assert "## Recent Memories — project: proj" in content
    assert "- mem bullet" in content
    assert "## Reasoning strategies (learned from claude-fable-5)" in content
    # Memories block comes before the reasoning block.
    assert content.index("Recent Memories") < content.index("Reasoning strategies")


def test_main_reasoning_failure_isolated_from_memories(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reasoning-block exception must not drop the memories block."""
    with (
        patch("sys.stdin", io.StringIO("{}")),
        patch("surreal_memory.hooks.project_context.derive_project_name", return_value="proj"),
        patch(
            "surreal_memory.hooks.session_start.get_recent_memories",
            new=AsyncMock(return_value="- mem bullet"),
        ),
        patch(
            "surreal_memory.hooks.session_start.get_reasoning_injection",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        main()

    captured = capsys.readouterr()
    content = json.loads(captured.out.strip())["hookSpecificOutput"]["additionalContext"]
    assert "- mem bullet" in content
    assert "Reasoning strategies" not in content
    assert "reasoning injection failed" in captured.err.lower()


def test_main_memories_failure_isolated_from_reasoning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A memories-block exception must not drop the reasoning block."""
    with (
        patch("sys.stdin", io.StringIO("{}")),
        patch("surreal_memory.hooks.project_context.derive_project_name", return_value="proj"),
        patch(
            "surreal_memory.hooks.session_start.get_recent_memories",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
        patch(
            "surreal_memory.hooks.session_start.get_reasoning_injection",
            new=AsyncMock(return_value=_REASONING_BLOCK),
        ),
    ):
        main()

    captured = capsys.readouterr()
    content = json.loads(captured.out.strip())["hookSpecificOutput"]["additionalContext"]
    assert "Reasoning strategies" in content
    assert "Recent Memories" not in content
    assert "context load failed" in captured.err.lower()
