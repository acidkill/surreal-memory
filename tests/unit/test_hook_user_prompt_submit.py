"""Tests for the UserPromptSubmit Claude Code hook.

The hook emits the reasoning-strategies block inside a hookSpecificOutput JSON
envelope (additionalContext) — the only channel Claude Code adds to the model's
context for this event — and always exits 0 so it can never block the prompt.
The block itself is produced by the shared
engine.reasoning_injection.get_reasoning_context orchestrator, which is patched
here — its own behavior (resolve/build/marker) is covered in
test_reasoning_injection.py.
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.hooks.user_prompt_submit import main, read_hook_input

_ORCHESTRATOR = "surreal_memory.engine.reasoning_injection.get_reasoning_context"
_REASONING_BLOCK = "## Reasoning strategies (learned from claude-fable-5)\n\n1. **plan**"


def test_read_hook_input_empty_stdin() -> None:
    with patch("sys.stdin", io.StringIO("")):
        assert read_hook_input() == {}


def test_read_hook_input_valid_json() -> None:
    payload = {"session_id": "s1", "transcript_path": "/x/t.jsonl", "prompt": "hi"}
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        assert read_hook_input() == payload


def test_read_hook_input_malformed_json() -> None:
    with patch("sys.stdin", io.StringIO("not json")):
        assert read_hook_input() == {}


def test_main_emits_hook_specific_output_json(capsys: pytest.CaptureFixture[str]) -> None:
    # Context reaches the model ONLY via hookSpecificOutput.additionalContext —
    # the hook must emit that JSON envelope, not raw stdout.
    with patch("sys.stdin", io.StringIO("{}")):
        with patch(_ORCHESTRATOR, new=AsyncMock(return_value=_REASONING_BLOCK)):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    hook_out = payload["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "UserPromptSubmit"
    assert hook_out["additionalContext"] == _REASONING_BLOCK


def test_main_no_block_prints_nothing_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sys.stdin", io.StringIO("{}")):
        with patch(_ORCHESTRATOR, new=AsyncMock(return_value="")):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # nothing injected into the prompt
    assert "No reasoning strategies" in captured.err


def test_main_injection_failure_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    # An orchestrator exception must never block the prompt — exit 0, no stdout.
    with patch("sys.stdin", io.StringIO("{}")):
        with patch(_ORCHESTRATOR, new=AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "failed" in captured.err.lower()


def test_main_malformed_stdin_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch(_ORCHESTRATOR, new=AsyncMock(return_value="")):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
