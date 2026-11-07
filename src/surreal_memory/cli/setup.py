"""Zero-config setup for Surreal-Memory.

Handles first-time initialization: config, brain, and MCP auto-configuration.
Called by `smem init` to set up everything in one command.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer


def find_smem_command() -> dict[str, Any]:
    """Find the best command to run the MCP server.

    Priority:
    1. smem-mcp entry point (cleanest)
    2. smem CLI with mcp subcommand
    3. python -m fallback (uses absolute path, normalized for Windows)

    Always includes an ``env`` block so MCP clients receive the full
    SurrealDB connection config and do not start with an empty environment.
    """
    from surreal_memory.storage.surrealdb.connection import build_mcp_env

    env = build_mcp_env()

    smem_mcp = shutil.which("smem-mcp")
    if smem_mcp:
        return {"command": "smem-mcp", "env": env}

    smem = shutil.which("smem")
    if smem:
        return {"command": "smem", "args": ["mcp"], "env": env}

    return {
        "command": _normalize_path(sys.executable),
        "args": ["-m", "surreal_memory.mcp"],
        "env": env,
    }


def setup_config(data_dir: Path, *, force: bool = False) -> bool:
    """Create ~/.surrealmemory/ with config.toml and brains/ directory.

    Returns True if config was created/updated, False if skipped.
    """
    from surreal_memory.unified_config import UnifiedConfig

    config_path = data_dir / "config.toml"

    if config_path.exists() and not force:
        return False

    config = UnifiedConfig(data_dir=data_dir)
    config.save()

    brains_dir = data_dir / "brains"
    brains_dir.mkdir(parents=True, exist_ok=True)

    return True


def setup_brain(data_dir: Path) -> str:
    """Ensure default brain SQLite DB exists.

    Returns the brain name.
    """
    brains_dir = data_dir / "brains"
    brains_dir.mkdir(parents=True, exist_ok=True)

    db_path = brains_dir / "default.db"
    if not db_path.exists():
        db_path.touch()

    return "default"


def _claude_json_has_server(claude_json_path: Path, server_name: str) -> bool:
    """Check if a server is already registered in ~/.claude.json."""
    if not claude_json_path.exists():
        return False
    try:
        raw = claude_json_path.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        data = json.loads(raw)
        servers = data.get("mcpServers", {})
        return server_name in servers
    except (json.JSONDecodeError, OSError):
        return False


def _add_via_claude_cli(scope: str, command_args: list[str]) -> bool:
    """Try to add MCP server via `claude mcp add` CLI.

    Returns True on success, False if claude CLI is not available or fails.
    """
    import subprocess

    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False

    cmd = [claude_bin, "mcp", "add", "-s", scope, "surreal-memory", "--"]
    cmd.extend(command_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=15,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _add_via_claude_json(claude_json_path: Path, mcp_entry: dict[str, Any]) -> bool:
    """Fallback: write MCP config directly to ~/.claude.json."""
    try:
        existing: dict[str, Any] = {}
        if claude_json_path.exists():
            raw = claude_json_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)

        servers: dict[str, Any] = existing.setdefault("mcpServers", {})
        servers["surreal-memory"] = mcp_entry
        claude_json_path.write_text(
            json.dumps(existing, indent=2) + "\n",
            encoding="utf-8",
        )
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _cleanup_stale_mcp_servers_json() -> None:
    """Remove the deprecated ~/.claude/mcp_servers.json if it exists.

    Claude Code does NOT read this file — it was a pre-release path.
    Entries here cause user confusion since they appear configured but don't work.
    """
    stale = Path.home() / ".claude" / "mcp_servers.json"
    if stale.exists():
        try:
            stale.unlink()
        except OSError:
            pass


def setup_mcp_claude() -> str:
    """Auto-configure MCP in Claude Code.

    Strategy:
    Always use direct JSON write (``~/.claude.json``) rather than the
    ``claude mcp add`` CLI path, because the CLI does not support the ``env``
    block required for SurrealDB credential injection. The JSON write is
    idempotent and includes the full ``env`` dict.

    If an entry already exists but lacks ``env``, the env block is backfilled
    so upgraded installations receive the credential config.

    Returns status string: "added", "exists", "failed", or "not_found".
    """
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        return "not_found"

    claude_json = claude_dir.parent / ".claude.json"
    mcp_entry = find_smem_command()

    # Check for existing entry
    if claude_json.exists():
        try:
            raw = claude_json.read_text(encoding="utf-8").strip()
            if raw:
                data: dict[str, Any] = json.loads(raw)
                servers = data.get("mcpServers", {})
                if "surreal-memory" in servers:
                    existing = servers["surreal-memory"]
                    # If env already present and populated → exists
                    if existing.get("env", {}).get("SURREALDB_PASS"):
                        _cleanup_stale_mcp_servers_json()
                        return "exists"
                    # Backfill env for upgraded installations — backup first
                    try:
                        backup = claude_json.with_suffix(".json.bak")
                        backup.write_bytes(claude_json.read_bytes())
                    except OSError:
                        pass
                    existing["env"] = mcp_entry.get("env", {})
                    claude_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                    _cleanup_stale_mcp_servers_json()
                    return "added"
        except (json.JSONDecodeError, OSError):
            pass

    # Write full entry (includes env)
    if _add_via_claude_json(claude_json, mcp_entry):
        _cleanup_stale_mcp_servers_json()
        return "added"

    return "failed"


def setup_mcp_cursor() -> str:
    """Auto-configure MCP in Cursor (~/.cursor/mcp.json).

    Returns status string: "added", "exists", "failed", or "not_found".
    """
    cursor_dir = Path.home() / ".cursor"
    if not cursor_dir.exists():
        return "not_found"

    config_path = cursor_dir / "mcp.json"
    mcp_entry = find_smem_command()

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = config_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            existing = {}

    servers = existing.get("mcpServers", {})
    if "surreal-memory" in servers:
        return "exists"

    try:
        servers["surreal-memory"] = mcp_entry
        existing["mcpServers"] = servers
        config_path.write_text(
            json.dumps(existing, indent=2) + "\n",
            encoding="utf-8",
        )
        return "added"
    except OSError:
        return "failed"


def _claude_desktop_config_path() -> Path | None:
    """Return the Claude Desktop config file path for the current OS.

    Returns None when the parent directory does not exist (Desktop not installed).
    """
    import os

    if sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            return None
        config_dir = Path(appdata) / "Claude"
    else:
        config_dir = Path.home() / ".config" / "Claude"

    if not config_dir.exists():
        return None
    return config_dir / "claude_desktop_config.json"


def setup_mcp_claude_desktop() -> str:
    """Auto-configure MCP in Claude Desktop.

    Writes or updates ``claude_desktop_config.json`` with a ``surreal-memory``
    entry including the full ``env`` block so the Desktop app never starts the
    MCP server with an empty environment.

    Idempotent: if the entry already exists and has ``env.SURREALDB_PASS``,
    returns ``"exists"``.  If the entry exists without ``env``, backfills it
    and returns ``"added"``.  Creates a ``.bak`` backup before any write.

    Returns status string: ``"added"``, ``"exists"``, ``"failed"``,
    or ``"not_found"`` (Desktop not installed).
    """
    config_path = _claude_desktop_config_path()
    if config_path is None:
        return "not_found"

    mcp_entry = find_smem_command()

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            raw = config_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            existing = {}

        servers = existing.get("mcpServers", {})
        if "surreal-memory" in servers:
            current = servers["surreal-memory"]
            if current.get("env", {}).get("SURREALDB_PASS"):
                return "exists"
            # Backfill env for upgraded installations — fall through to write

        # Backup before modifying
        try:
            backup = config_path.with_suffix(".json.bak")
            backup.write_bytes(config_path.read_bytes())
        except OSError:
            pass

    try:
        servers = existing.setdefault("mcpServers", {})
        servers["surreal-memory"] = mcp_entry
        config_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return "added"
    except OSError:
        return "failed"


def _normalize_path(path: str) -> str:
    """Normalize a filesystem path for use in shell commands.

    On Windows, convert backslashes to forward slashes and quote if
    the path contains spaces. This ensures commands work in both
    CMD/PowerShell and Git Bash/MSYS2 environments.
    """
    normalized = path.replace("\\", "/")
    if " " in normalized:
        normalized = f'"{normalized}"'
    return normalized


def _find_hook_command(entry_point: str, cli_subcommand: str, module: str) -> str:
    """Find the best shell command for a Surreal-Memory hook.

    Priority:
    1. Dedicated pip entry point  — cleanest, cross-platform
    2. smem <subcommand>          — works if smem CLI is on PATH
    3. python -m <module>         — always-available fallback (uses absolute path)
    """
    if shutil.which(entry_point):
        return entry_point
    if shutil.which("smem"):
        return f"smem {cli_subcommand}"
    python = _normalize_path(sys.executable)
    return f"{python} -m {module}"


def _find_pre_compact_command() -> str:
    return _find_hook_command("smem-hook-pre-compact", "flush", "surreal_memory.hooks.pre_compact")


def _find_stop_command() -> str:
    return _find_hook_command("smem-hook-stop", "stop-hook", "surreal_memory.hooks.stop")


def _find_post_tool_use_command() -> str:
    return _find_hook_command(
        "smem-hook-post-tool-use", "post-tool-use-hook", "surreal_memory.hooks.post_tool_use"
    )


def _is_smem_hook_present(entries: list[dict[str, Any]]) -> bool:
    """Return True if any Surreal-Memory hook is already registered in the entry list."""
    for entry in entries:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if "surreal_memory" in cmd or "smem" in cmd:
                return True
    return False


def _load_settings(settings_path: Path) -> dict[str, Any]:
    if not settings_path.exists():
        return {}
    try:
        raw = settings_path.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {}


def setup_hooks_claude() -> str:
    """Install the shared fail-open lifecycle adapter for Claude Code."""
    from surreal_memory.cli.agent_setup import setup_client_hooks

    return setup_client_hooks("claude")


def _discover_bundled_skills() -> dict[str, Path]:
    """Find all bundled skills (dirs containing SKILL.md).

    Returns mapping of skill name to SKILL.md path.
    """
    from surreal_memory.skills import SKILLS_DIR

    skills: dict[str, Path] = {}
    for child in sorted(SKILLS_DIR.iterdir()):
        skill_file = child / "SKILL.md"
        if child.is_dir() and skill_file.is_file():
            skills[child.name] = skill_file
    return skills


def setup_skills(*, force: bool = False, client: str | None = None) -> dict[str, str]:
    """Install packaged skill trees, preserving user edits unless --force is explicit."""
    from surreal_memory.cli.agent_setup import selected_clients

    clients = selected_clients(client)
    if not clients:
        return {"Skills": "not_found (no detected coding clients)"}
    bundled = _discover_bundled_skills()
    results: dict[str, str] = {}
    for target in clients:
        root = (
            (Path.home() / ".agents" / "skills")
            if target == "codex"
            else Path.home() / ".claude" / "skills"
        )
        for name, source in bundled.items():
            if name.startswith("coding-memory-") and name != "coding-memory-" + target:
                continue
            key = target + "/" + name if len(clients) > 1 else name
            status = "exists"
            try:
                for src in sorted(source.parent.rglob("*")):
                    if not src.is_file():
                        continue
                    dest = root / name / src.relative_to(source.parent)
                    if dest.exists() and dest.read_bytes() == src.read_bytes():
                        continue
                    if dest.exists() and not force:
                        status = "update available"
                        continue
                    was_present = dest.exists()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(src.read_bytes())
                    if status != "update available":
                        status = "updated" if was_present else "installed"
                if name.startswith("coding-memory-"):
                    common = source.parent.parent / "coding-memory-common" / "protocol.md"
                    dest = root / name / "references" / "protocol.md"
                    if dest.exists() and dest.read_bytes() != common.read_bytes() and not force:
                        status = "update available"
                    elif not dest.exists() or force:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(common.read_bytes())
                results[key] = status
            except OSError:
                results[key] = "failed"
    return results


def print_summary(results: dict[str, str]) -> None:
    """Print formatted setup summary."""
    typer.echo()
    typer.secho("  Surreal-Memory Setup", bold=True)
    typer.echo()

    status_icons = {
        "ok": typer.style("[OK]", fg=typer.colors.GREEN),
        "skip": typer.style("[--]", fg=typer.colors.YELLOW),
        "fail": typer.style("[!!]", fg=typer.colors.RED),
    }

    for label, detail in results.items():
        icon = status_icons.get(_classify_status(detail), status_icons["skip"])
        typer.echo(f"  {icon} {label:<16}{detail}")

    typer.echo()


def _classify_status(detail: str) -> str:
    """Classify a result detail string into ok/skip/fail."""
    lower = detail.lower()
    if any(word in lower for word in ("created", "added", "ready", "installed", "updated")):
        return "ok"
    if any(word in lower for word in ("exists", "already")):
        return "ok"
    if any(word in lower for word in ("not detected", "skipped", "not found")):
        return "skip"
    return "fail"
