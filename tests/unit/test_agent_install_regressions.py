"""Regression coverage for safe lifecycle installation and hook input parsing."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import pytest

from surreal_memory.cli.setup import setup_hooks_claude


@pytest.fixture
def claude_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    directory = tmp_path / ".claude"
    directory.mkdir()
    return directory / "settings.json"


def test_installer_wires_recovery_events_and_is_repeatable(claude_settings: Path) -> None:
    unrelated = {"hooks": [{"type": "command", "command": "echo custom-startup"}]}
    claude_settings.write_text(
        json.dumps({"theme": "dark", "hooks": {"SessionStart": [unrelated]}})
    )

    assert setup_hooks_claude() in {"added", "updated"}
    first = claude_settings.read_bytes()
    installed = json.loads(first)

    assert installed["theme"] == "dark"
    assert unrelated in installed["hooks"]["SessionStart"]
    for event in ("SessionStart", "UserPromptSubmit", "PreCompact", "Stop", "PostToolUse"):
        assert any(
            "smem" in hook.get("command", "") or "surreal_memory" in hook.get("command", "")
            for group in installed["hooks"][event]
            for hook in group["hooks"]
        ), event

    assert setup_hooks_claude() == "exists"
    assert claude_settings.read_bytes() == first


@pytest.mark.parametrize("original", ['{"hooks": ', "[]", "null", '{"hooks": []}'])
def test_installer_preserves_invalid_existing_settings(
    claude_settings: Path, original: str
) -> None:
    claude_settings.write_text(original)

    assert setup_hooks_claude() == "failed"
    assert claude_settings.read_text() == original


def test_installer_replaces_owned_expression_matcher_only(claude_settings: Path) -> None:
    expression = 'tool != "TodoRead" && tool != "TodoWrite" && tool != "TaskList"'
    unrelated = {"matcher": "Read|Edit", "hooks": [{"type": "command", "command": "echo user"}]}
    old = {
        "matcher": expression,
        "hooks": [{"type": "command", "command": "smem-hook-post-tool-use", "timeout": 5}],
    }
    claude_settings.write_text(json.dumps({"hooks": {"PostToolUse": [unrelated, old]}}))

    assert setup_hooks_claude() in {"added", "updated"}
    installed = json.loads(claude_settings.read_text())["hooks"]["PostToolUse"]

    assert unrelated in installed
    assert all(group.get("matcher") != expression for group in installed)


@pytest.mark.parametrize("module_name", ["session_start", "user_prompt_submit"])
@pytest.mark.parametrize("payload", ["[]", "null", '"message"', "42", "false"])
def test_context_hook_rejects_non_object_json(
    module_name: str, payload: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"surreal_memory.hooks.{module_name}")
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    assert module.read_hook_input() == {}
