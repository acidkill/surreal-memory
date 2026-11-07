"""Native client discovery, idempotent install, and packaged references."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from surreal_memory.cli import app


def test_install_skills_client_option_copies_shared_references(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = CliRunner().invoke(app, ["install-skills", "--client", "codex"])
    assert result.exit_code == 0, result.output
    root = tmp_path / ".agents" / "skills" / "coding-memory-codex"
    assert (root / "SKILL.md").exists()
    policy = root / "references" / "protocol.md"
    assert "pending" in policy.read_text()
    policy.write_text("User customized policy")
    again = CliRunner().invoke(app, ["install-skills", "--client", "codex"])
    assert again.exit_code == 0
    assert policy.read_text() == "User customized policy"


def test_codex_hooks_preserve_user_config_and_do_not_bypass_trust(tmp_path, monkeypatch):
    from surreal_memory.cli.agent_setup import setup_client_hooks

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    path = tmp_path / "hooks.json"
    user = {"hooks": [{"type": "command", "command": "echo personal"}]}
    path.write_text(json.dumps({"hooks": {"SessionStart": [user]}}))
    with patch("surreal_memory.cli.agent_setup.native_codex_hooks", return_value=True):
        assert setup_client_hooks("codex") == "added"
        first = path.read_bytes()
        assert setup_client_hooks("codex") == "exists"
    assert path.read_bytes() == first
    assert user in json.loads(first)["hooks"]["SessionStart"]
    assert b"bypass" not in first
    assert "PostCompact" in json.loads(first)["hooks"]


def test_codex_without_hooks_is_explicitly_skills_only(tmp_path, monkeypatch):
    from surreal_memory.cli.agent_setup import setup_client_hooks

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    with patch("surreal_memory.cli.agent_setup.native_codex_hooks", return_value=False):
        assert setup_client_hooks("codex").startswith("skills_only")
    assert not (tmp_path / "hooks.json").exists()


def test_doctor_reports_pending_writes_and_incomplete_client(tmp_path, monkeypatch):
    from surreal_memory.cli.agent_setup import check_agent_integrations
    from surreal_memory.hooks.agent_journal import enqueue

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    (tmp_path / ".codex").mkdir()
    enqueue("capture", {"content": "Chose copper because it reflects infrared"})
    result = check_agent_integrations()
    assert result["status"] == "warn"
    assert "1 pending" in result["detail"]
    assert "Codex" in result["detail"]
