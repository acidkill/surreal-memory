"""Regression tests: `smem brain` subcommands must see SurrealDB-only brains.

CLIConfig.list_brains only globs local *.json/*.db fixture files, so every
`smem brain` subcommand that checked brain existence via config.list_brains()
reported "no brains found" / "not found" on a healthy SurrealDB install (the
only production backend since v2.0.0) even though the brain genuinely exists.
These commands must route through unified_config.list_available_brains()
instead, which queries the active backend directly.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from surreal_memory.cli.main import app

runner = CliRunner()

CLI = "surreal_memory.cli.commands.brain"


def _cli_config(current_brain: str = "default") -> MagicMock:
    config = MagicMock()
    config.current_brain = current_brain
    config.save = MagicMock()
    return config


def test_brain_list_shows_surrealdb_only_brain() -> None:
    """A brain that exists only in SurrealDB (no local .db/.json file) must
    still show up in `smem brain list` — this is the exact bug reported."""
    with (
        patch(f"{CLI}.get_config", return_value=_cli_config()),
        patch(
            f"{CLI}.list_available_brains",
            new=AsyncMock(return_value=["default", "surrealdb-only-brain"]),
        ),
    ):
        result = runner.invoke(app, ["brain", "list"])

    assert result.exit_code == 0, result.output
    assert "surrealdb-only-brain" in result.output


def test_brain_list_json_includes_surrealdb_only_brain() -> None:
    with (
        patch(f"{CLI}.get_config", return_value=_cli_config()),
        patch(
            f"{CLI}.list_available_brains",
            new=AsyncMock(return_value=["default", "surrealdb-only-brain"]),
        ),
    ):
        result = runner.invoke(app, ["brain", "list", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "surrealdb-only-brain" in data["brains"]


def test_brain_use_finds_surrealdb_only_brain() -> None:
    """Switching to a brain that only exists in SurrealDB must not fail with
    'not found' just because there's no local file for it."""
    config = _cli_config()
    with (
        patch(f"{CLI}.get_config", return_value=config),
        patch(
            f"{CLI}.list_available_brains",
            new=AsyncMock(return_value=["surrealdb-only-brain"]),
        ),
    ):
        result = runner.invoke(app, ["brain", "use", "surrealdb-only-brain"])

    assert result.exit_code == 0, result.output
    assert config.current_brain == "surrealdb-only-brain"
    config.save.assert_called_once()


def test_brain_create_rejects_existing_surrealdb_only_name() -> None:
    """Creating a brain whose name already exists in SurrealDB must be
    rejected, even though no local file with that name exists."""
    with (
        patch(f"{CLI}.get_config", return_value=_cli_config()),
        patch(f"{CLI}.list_available_brains", new=AsyncMock(return_value=["taken"])),
        patch(f"{CLI}.get_storage", new=AsyncMock()),
    ):
        result = runner.invoke(app, ["brain", "create", "taken"])

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_brain_delete_on_surrealdb_only_brain_gives_clear_error_not_crash() -> None:
    """Deleting a brain that only exists in SurrealDB (no local file to
    unlink) must fail with a clear message, not an unhandled
    FileNotFoundError from Path.unlink()."""
    config = _cli_config(current_brain="default")
    fake_path = MagicMock()
    fake_path.exists.return_value = False

    with (
        patch(f"{CLI}.get_config", return_value=config),
        patch(
            f"{CLI}.list_available_brains",
            new=AsyncMock(return_value=["surrealdb-only-brain"]),
        ),
        patch(f"{CLI}.get_brain_path_auto", return_value=fake_path),
    ):
        result = runner.invoke(app, ["brain", "delete", "surrealdb-only-brain", "--force"])

    assert result.exit_code == 1
    assert "isn't supported yet" in result.output
    fake_path.unlink.assert_not_called()


def test_brain_delete_still_deletes_local_file_brain() -> None:
    """A brain that DOES have a local file (legacy/test-fixture mode) must
    still be deletable exactly as before."""
    config = _cli_config(current_brain="default")
    fake_path = MagicMock()
    fake_path.exists.return_value = True

    with (
        patch(f"{CLI}.get_config", return_value=config),
        patch(f"{CLI}.list_available_brains", new=AsyncMock(return_value=["local-brain"])),
        patch(f"{CLI}.get_brain_path_auto", return_value=fake_path),
    ):
        result = runner.invoke(app, ["brain", "delete", "local-brain", "--force"])

    assert result.exit_code == 0, result.output
    fake_path.unlink.assert_called_once()
