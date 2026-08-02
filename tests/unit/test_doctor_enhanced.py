"""Tests for enhanced doctor checks (hooks, dedup, surface, auto-fix)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from surreal_memory.cli.doctor import (
    FAIL,
    OK,
    QUICKSTART_URL,
    SKIP,
    WARN,
    _auto_fix,
    _check_dedup,
    _check_hooks,
    _check_surface,
    _fix_dedup,
    _fix_embedding,
    _fix_hooks,
    run_doctor,
)


class TestCheckHooks:
    """Test hooks diagnostic check."""

    def test_all_hooks_present(self, tmp_path: Path) -> None:
        settings = {
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "smem-hook-pre-compact"}]}
                ],
                "Stop": [{"hooks": [{"type": "command", "command": "smem-hook-stop"}]}],
                "PostToolUse": [
                    {"hooks": [{"type": "command", "command": "smem-hook-post-tool-use"}]}
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == OK
            assert "3/3" in result["detail"]

    def test_wrapper_referenced_but_missing_on_path(self, tmp_path: Path) -> None:
        """Hooks wired through smem-hook-env warn if the wrapper is not on PATH."""
        settings = {
            "hooks": {
                "PreCompact": [
                    {
                        "hooks": [
                            {"type": "command", "command": "smem-hook-env smem-hook-pre-compact"}
                        ]
                    }
                ],
                "Stop": [
                    {"hooks": [{"type": "command", "command": "smem-hook-env smem-hook-stop"}]}
                ],
                "PostToolUse": [
                    {
                        "hooks": [
                            {"type": "command", "command": "smem-hook-env smem-hook-post-tool-use"}
                        ]
                    }
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with (
            patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path),
            patch("surreal_memory.cli.doctor.shutil.which", return_value=None),
        ):
            result = _check_hooks()
            assert result["status"] == WARN
            assert "smem-hook-env" in result["detail"]

    def test_wrapper_referenced_and_present(self, tmp_path: Path) -> None:
        """Hooks wired through smem-hook-env pass when the wrapper resolves on PATH."""
        settings = {
            "hooks": {
                "PreCompact": [
                    {
                        "hooks": [
                            {"type": "command", "command": "smem-hook-env smem-hook-pre-compact"}
                        ]
                    }
                ],
                "Stop": [
                    {"hooks": [{"type": "command", "command": "smem-hook-env smem-hook-stop"}]}
                ],
                "PostToolUse": [
                    {
                        "hooks": [
                            {"type": "command", "command": "smem-hook-env smem-hook-post-tool-use"}
                        ]
                    }
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with (
            patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path),
            patch(
                "surreal_memory.cli.doctor.shutil.which",
                return_value="/home/user/.local/bin/smem-hook-env",
            ),
        ):
            result = _check_hooks()
            assert result["status"] == OK
            assert "3/3" in result["detail"]

    def test_missing_hooks(self, tmp_path: Path) -> None:
        settings = {
            "hooks": {
                "PreCompact": [
                    {"hooks": [{"type": "command", "command": "smem-hook-pre-compact"}]}
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == WARN
            assert "Stop" in result["detail"]
            assert "PostToolUse" in result["detail"]

    def test_no_settings_file(self, tmp_path: Path) -> None:
        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == WARN

    def test_corrupt_settings(self, tmp_path: Path) -> None:
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("not json!", encoding="utf-8")

        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == WARN

    def test_detects_python_module_hooks(self, tmp_path: Path) -> None:
        """Hooks using python -m surreal_memory.hooks.* should also be detected."""
        settings = {
            "hooks": {
                "PreCompact": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -m surreal_memory.hooks.pre_compact",
                            }
                        ]
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": "python -m surreal_memory.hooks.stop"}
                        ]
                    }
                ],
                "PostToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -m surreal_memory.hooks.post_tool_use",
                            }
                        ]
                    }
                ],
            }
        }
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_hooks()
            assert result["status"] == OK


class TestCheckDedup:
    """Test dedup diagnostic check."""

    def test_dedup_enabled(self) -> None:
        mock_config = MagicMock()
        mock_config.dedup.enabled = True
        with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
            result = _check_dedup()
            assert result["status"] == OK

    def test_dedup_disabled(self) -> None:
        mock_config = MagicMock()
        mock_config.dedup.enabled = False
        with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
            result = _check_dedup()
            assert result["status"] == WARN
            assert "fixable" in result

    def test_config_not_loaded(self) -> None:
        with patch("surreal_memory.unified_config.get_config", side_effect=Exception("no config")):
            result = _check_dedup()
            assert result["status"] == SKIP


class TestCheckSurface:
    """Test knowledge surface diagnostic check."""

    def test_surface_exists(self, tmp_path: Path) -> None:
        surface_file = tmp_path / "default.nm"
        surface_file.write_text("---\nbrain: default\n---\n# GRAPH", encoding="utf-8")

        mock_config = MagicMock()
        mock_config.current_brain = "default"

        with (
            patch("surreal_memory.unified_config.get_config", return_value=mock_config),
            patch("surreal_memory.surface.resolver.get_surface_path", return_value=surface_file),
        ):
            result = _check_surface()
            assert result["status"] == OK
            assert "KB" in result["detail"]

    def test_surface_missing(self, tmp_path: Path) -> None:
        surface_file = tmp_path / "default.nm"  # doesn't exist

        mock_config = MagicMock()
        mock_config.current_brain = "default"

        with (
            patch("surreal_memory.unified_config.get_config", return_value=mock_config),
            patch("surreal_memory.surface.resolver.get_surface_path", return_value=surface_file),
        ):
            result = _check_surface()
            assert result["status"] == WARN

    def test_surface_module_unavailable(self) -> None:
        """When surface package isn't available, check should skip."""
        with patch(
            "surreal_memory.unified_config.get_config",
            side_effect=Exception("no config"),
        ):
            result = _check_surface()
            assert result["status"] == SKIP


class TestAutoFix:
    """Test auto-fix functionality."""

    def test_fixes_fixable_checks(self) -> None:
        checks = [
            {"name": "Hooks", "status": WARN, "detail": "missing", "fixable": True},
            {"name": "Python version", "status": OK, "detail": "3.12"},
        ]
        with patch("surreal_memory.cli.doctor._try_fix") as mock_fix:
            mock_fix.return_value = {"name": "Hooks", "status": OK, "detail": "auto-fixed"}
            result = _auto_fix(checks)
            assert result[0]["status"] == OK
            assert result[1]["status"] == OK  # unchanged

    def test_skips_non_fixable(self) -> None:
        checks = [
            {"name": "Python version", "status": FAIL, "detail": "3.9 (requires 3.11+)"},
        ]
        result = _auto_fix(checks)
        assert result[0]["status"] == FAIL  # unchanged, not fixable

    def test_fix_hooks_success(self) -> None:
        with patch("surreal_memory.cli.setup.setup_hooks_claude", return_value="added"):
            result = _fix_hooks()
            assert result["status"] == OK

    def test_fix_hooks_failure(self) -> None:
        with patch("surreal_memory.cli.setup.setup_hooks_claude", side_effect=Exception("fail")):
            result = _fix_hooks()
            assert result["status"] == WARN

    def test_fix_dedup_success(self, tmp_path: Path) -> None:
        from dataclasses import replace

        from surreal_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path)
        config = replace(config, dedup=replace(config.dedup, enabled=False))
        with patch("surreal_memory.unified_config.get_config", return_value=config):
            result = _fix_dedup()
            assert result["status"] == OK

    def test_fix_embedding_with_provider(self) -> None:
        provider = {"key": "sentence_transformer", "model": "all-MiniLM-L6-v2", "label": "ST"}
        with (
            patch(
                "surreal_memory.cli.full_setup.detect_embedding_provider",
                return_value=provider,
            ),
            patch("surreal_memory.cli.full_setup.enable_config_defaults"),
        ):
            result = _fix_embedding()
            assert result["status"] == OK

    def test_fix_embedding_no_provider(self) -> None:
        with patch("surreal_memory.cli.full_setup.detect_embedding_provider", return_value=None):
            result = _fix_embedding()
            assert result["status"] == WARN


class TestRunDoctorIntegration:
    """Test run_doctor with new checks."""

    @patch("surreal_memory.cli.doctor._check_pro_plugin")
    @patch("surreal_memory.cli.doctor._check_surface")
    @patch("surreal_memory.cli.doctor._check_dedup")
    @patch("surreal_memory.cli.doctor._check_hooks")
    @patch("surreal_memory.cli.doctor._check_cli_tools")
    @patch("surreal_memory.cli.doctor._check_surrealdb_version")
    @patch("surreal_memory.cli.doctor._check_mcp_env_completeness")
    @patch("surreal_memory.cli.doctor._check_surrealdb_connection")
    @patch("surreal_memory.cli.doctor._check_mcp_connection")
    @patch("surreal_memory.cli.doctor._check_mcp_config")
    @patch("surreal_memory.cli.doctor._check_embedding_provider")
    @patch("surreal_memory.cli.doctor._check_dependencies")
    @patch("surreal_memory.cli.doctor._check_brain")
    @patch("surreal_memory.cli.doctor._check_config_freshness")
    @patch("surreal_memory.cli.doctor._check_storage_backend")
    @patch("surreal_memory.cli.doctor._check_config")
    @patch("surreal_memory.cli.doctor._check_python_version")
    def test_reports_every_registered_check(self, *mocks: MagicMock) -> None:
        for mock in mocks:
            mock.return_value = {"name": "test", "status": OK, "detail": "ok"}

        result = run_doctor(json_output=True)
        # Bump when run_doctor gains a check: config, storage backend, brain,
        # dependencies, embedding, MCP config/connection/env, hooks,
        # dedup, surface, config freshness, CLI tools, SurrealDB
        # connection/version, Pro plugin, Python version.
        assert result["total"] == 17
        assert result["passed"] == 17

    def test_quickstart_url_defined(self) -> None:
        assert "quickstart" in QUICKSTART_URL


class TestCheckSurrealdbConnection:
    """_check_surrealdb_connection() — new TIER_CORE check for auth-fail detection."""

    def test_skip_when_storage_is_not_surrealdb(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_connection

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "memory")
        result = _check_surrealdb_connection()
        assert result["status"] == "skip"

    def test_fail_when_storage_auth_error(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_connection
        from surreal_memory.storage.surrealdb.connection import StorageAuthError

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")

        with patch(
            "surreal_memory.cli.doctor._run_surrealdb_ping",
            side_effect=StorageAuthError("auth failed", hint="Set SURREALDB_PASS"),
        ):
            result = _check_surrealdb_connection()

        assert result["status"] == "fail"
        assert "SURREALDB_PASS" in result.get("fix", "")

    def test_ok_when_connection_succeeds(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_connection

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")

        with patch("surreal_memory.cli.doctor._run_surrealdb_ping", return_value=None):
            result = _check_surrealdb_connection()

        assert result["status"] == "ok"

    def test_warn_on_generic_exception(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_connection

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")

        with patch(
            "surreal_memory.cli.doctor._run_surrealdb_ping",
            side_effect=ConnectionRefusedError("refused"),
        ):
            result = _check_surrealdb_connection()

        assert result["status"] == "warn"


class TestCheckSurrealdbVersion:
    """_check_surrealdb_version() — new TIER_CORE gate for SurrealDB >= 3.2.0."""

    def test_skip_when_storage_is_not_surrealdb(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_version

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "sqlite")
        assert _check_surrealdb_version()["status"] == "skip"

    def test_ok_when_current(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_version

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        with patch(
            "surreal_memory.cli.doctor._run_surrealdb_version_probe",
            return_value="surrealdb-3.2.0",
        ):
            result = _check_surrealdb_version()
        assert result["status"] == "ok"
        assert "3.2.0" in result["detail"]

    def test_fail_when_too_old(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_version

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        with patch(
            "surreal_memory.cli.doctor._run_surrealdb_version_probe",
            return_value="surrealdb-3.1.1",
        ):
            result = _check_surrealdb_version()
        assert result["status"] == "fail"
        assert "fix" in result

    def test_warn_when_probe_fails(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_version

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        with patch(
            "surreal_memory.cli.doctor._run_surrealdb_version_probe",
            side_effect=RuntimeError("unreachable"),
        ):
            result = _check_surrealdb_version()
        assert result["status"] == "warn"

    def test_warn_when_version_unparsable(self, monkeypatch):
        from surreal_memory.cli.doctor import _check_surrealdb_version

        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        with patch(
            "surreal_memory.cli.doctor._run_surrealdb_version_probe",
            return_value="nightly-build",
        ):
            result = _check_surrealdb_version()
        assert result["status"] == "warn"


class TestSynapseMigrationCommand:
    """smem doctor --synapse-migration {status|retry|purge-backup}."""

    def test_rejects_invalid_action(self):
        from surreal_memory.cli.doctor import run_synapse_migration_command

        with pytest.raises(ValueError):
            run_synapse_migration_command("bogus")


class TestCheckMcpEnvCompleteness:
    """_check_mcp_env_completeness() — TIER_RECOMMENDED check for missing env."""

    def test_warn_when_entry_lacks_env(self, tmp_path):
        from surreal_memory.cli.doctor import _check_mcp_env_completeness

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(
            json.dumps({"mcpServers": {"surreal-memory": {"command": "smem-mcp"}}})
        )
        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_mcp_env_completeness()

        assert result["status"] == "warn"
        assert result.get("fixable") is True

    def test_ok_when_env_is_complete(self, tmp_path):
        from surreal_memory.cli.doctor import _check_mcp_env_completeness

        claude_json = tmp_path / ".claude.json"
        claude_json.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "surreal-memory": {
                            "command": "smem-mcp",
                            "env": {"SURREALDB_PASS": "surrealmemory"},
                        }
                    }
                }
            )
        )
        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_mcp_env_completeness()

        assert result["status"] == "ok"

    def test_skip_when_no_claude_json(self, tmp_path):
        from surreal_memory.cli.doctor import _check_mcp_env_completeness

        with patch("surreal_memory.cli.doctor.Path.home", return_value=tmp_path):
            result = _check_mcp_env_completeness()

        assert result["status"] in ("skip", "warn")


class TestFixMcpEnv:
    """_fix_mcp_env() auto-fix handler — calls setup to backfill env."""

    def test_fix_handler_calls_setup_functions(self):
        from surreal_memory.cli.doctor import _fix_mcp_env

        with (
            patch("surreal_memory.cli.setup.setup_mcp_claude", return_value="added") as mock_claude,
            patch(
                "surreal_memory.cli.setup.setup_mcp_claude_desktop", return_value="added"
            ) as mock_desktop,
        ):
            result = _fix_mcp_env()

        assert mock_claude.called
        assert mock_desktop.called
        assert result["status"] == "ok"
