"""Conservative user-level registration for local coding clients."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

EVENTS = ("SessionStart", "UserPromptSubmit", "PreCompact", "Stop", "PostToolUse")


def codex_home() -> Path:
    """Respect Codex's supported config override."""
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def detected_clients() -> list[str]:
    """Auto-install only into existing client configurations."""
    result = []
    if (Path.home() / ".claude").is_dir():
        result.append("claude")
    if codex_home().is_dir():
        result.append("codex")
    return result


def selected_clients(client: str | None = None) -> list[str]:
    """Validate explicit targets; auto mode never creates unrelated client roots."""
    if client is None:
        return detected_clients()
    if client not in {"codex", "claude", "all"}:
        raise ValueError("client must be codex, claude or all")
    return ["codex", "claude"] if client == "all" else [client]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Retain a content-addressed backup and replace settings atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_bytes()
        backup = path.with_name(
            path.name + ".smem-" + hashlib.sha256(old).hexdigest()[:12] + ".bak"
        )
        if not backup.exists():
            fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(old)
    fd, temporary = tempfile.mkstemp(prefix=".smem-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def owned(command: str) -> bool:
    """Match smem hook commands, not arbitrary user commands mentioning smem."""
    return "surreal_memory.hooks." in command or any(
        name in command
        for name in ("smem-hook-", "smem stop-hook", "smem flush", "smem post-tool-use-hook")
    )


def native_codex_hooks() -> bool:
    """Detect capability instead of assuming hooks from a version string."""
    binary = shutil.which("codex")
    if not binary:
        return False
    try:
        result = subprocess.run(
            [binary, "features", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and any(
            line.split()[:1] == ["hooks"] and line.split()[-1:] == ["true"]
            for line in result.stdout.splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        return False


def setup_client_hooks(client: str, *, create: bool = False) -> str:
    """Replace only owned hooks; preserve malformed and unrelated configuration."""
    root = codex_home() if client == "codex" else Path.home() / ".claude"
    if not root.exists() and not create:
        return "not_found"
    if client == "codex" and not native_codex_hooks():
        return "skills_only (native hooks unavailable; upgrade Codex)"
    path = root / ("hooks.json" if client == "codex" else "settings.json")
    try:
        original = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(original, dict) or not isinstance(original.get("hooks", {}), dict):
            return "failed"
        # Clone to compare without mutating the original or losing unrelated keys.
        value = json.loads(json.dumps(original))
        hooks = value.setdefault("hooks", {})
        for groups in hooks.values():
            if not isinstance(groups, list):
                return "failed"
            for group in groups:
                if not isinstance(group, dict) or not isinstance(group.get("hooks", []), list):
                    return "failed"
                if any(
                    not isinstance(hook, dict) or not isinstance(hook.get("command", ""), str)
                    for hook in group.get("hooks", [])
                ):
                    return "failed"
        argv = [sys.executable, "-m", "surreal_memory.hooks.agent_lifecycle", "--client", client]
        command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
        for event in (
            *EVENTS,
            "SubagentStart",
            "SubagentStop",
            *(("PostCompact", "SessionEnd") if client == "codex" else ()),
        ):
            retained = []
            for group in hooks.get(event, []):
                remaining = [
                    hook for hook in group.get("hooks", []) if not owned(hook.get("command", ""))
                ]
                if remaining or not group.get("hooks"):
                    retained.append({**group, "hooks": remaining})
            retained.append(
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": command,
                            "timeout": 2 if event == "PostToolUse" else 12,
                        }
                    ]
                }
            )
            hooks[event] = retained
        if value == original:
            return "exists"
        atomic_json(path, value)
        return "added"
    except (ValueError, OSError, TypeError):
        return "failed"


def setup_mcp_codex(*, create: bool = False) -> str:
    """Use the native CLI to preserve arbitrary TOML and credential configuration."""
    if not codex_home().exists() and not create:
        return "not_found"
    binary = shutil.which("codex")
    if not binary:
        return "not_found (Codex CLI needed for MCP registration)"
    try:
        import tomllib

        path = codex_home() / "config.toml"
        if path.exists():
            config = tomllib.loads(path.read_text(encoding="utf-8"))
            if "surreal-memory" in config.get("mcp_servers", {}):
                return "exists"
        from surreal_memory.cli.setup import find_smem_command

        entry = find_smem_command()
        command = [binary, "mcp", "add", "surreal-memory"]
        for key, value in entry.get("env", {}).items():
            command.extend(["--env", key + "=" + value])
        command.extend(["--", entry["command"], *entry.get("args", [])])
        result = subprocess.run(command, capture_output=True, timeout=10)
        return "added" if result.returncode == 0 else "failed"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "failed"


def setup_agents(
    *, client: str | None = None, force: bool = False, skills: bool = True, mcp: bool = True
) -> dict[str, str]:
    """Install explicitly selected or detected local agent integrations."""
    from surreal_memory.cli.setup import setup_mcp_claude, setup_skills

    results = {}
    for target in selected_clients(client):
        if client is not None:
            (codex_home() if target == "codex" else Path.home() / ".claude").mkdir(
                parents=True, exist_ok=True
            )
        if mcp:
            results[target + " MCP"] = (
                setup_mcp_codex(create=True) if target == "codex" else setup_mcp_claude()
            )
            results[target + " hooks"] = setup_client_hooks(target, create=True)
        if skills:
            results.update(
                {
                    target + "/" + key: value
                    for key, value in setup_skills(force=force, client=target).items()
                }
            )
    return results


def check_agent_integrations() -> dict[str, Any]:
    """Inspect installation and pending writes without claiming unobserved hook trust."""
    from surreal_memory.hooks.agent_journal import pending
    from surreal_memory.unified_config import get_config

    details = []
    incomplete = False
    for client in detected_clients():
        root = codex_home() if client == "codex" else Path.home() / ".claude"
        path = root / ("hooks.json" if client == "codex" else "settings.json")
        try:
            value = json.loads(path.read_text())
            events = value.get("hooks", {})
            missing = [
                event
                for event in EVENTS
                if not any(
                    owned(hook.get("command", ""))
                    for group in events.get(event, [])
                    for hook in group.get("hooks", [])
                )
            ]
            skill = (
                (Path.home() / ".agents" / "skills") if client == "codex" else root / "skills"
            ) / ("coding-memory-" + client)
            if (
                not (skill / "SKILL.md").is_file()
                or not (skill / "references" / "protocol.md").is_file()
            ):
                missing.append("coding-memory skill/references")
            if client == "codex":
                import tomllib

                config = tomllib.loads((root / "config.toml").read_text())
                if "surreal-memory" not in config.get("mcp_servers", {}):
                    missing.append("MCP server")
                if not native_codex_hooks():
                    missing.append("native hooks (skills-only fallback)")
                # Registered commands and observed execution/trust are different states.
                missing.append("verify enabled/trusted hooks in Codex /hooks")
            elif not (Path.home() / ".claude.json").is_file():
                missing.append("MCP registration")
            if missing:
                incomplete = True
            details.append(
                client.title()
                + ": "
                + (
                    ", ".join(missing)
                    if missing
                    else "registered; exercise recovery to verify runtime"
                )
            )
        except (OSError, ValueError, AttributeError, TypeError):
            incomplete = True
            details.append(client.title() + ": incomplete or malformed integration settings")
    try:
        count = len(pending())
        details.append(f"{count} pending writes (not confirmed brain persistence)")
        incomplete |= count > 0
    except (OSError, ValueError, KeyError):
        incomplete = True
        details.append("retry queue unreadable")
    try:
        if get_config().tool_tier.tier == "minimal":
            incomplete = True
            details.append("checkpoint tool requires standard/full tool tier")
    except (OSError, ValueError):
        incomplete = True
    return {
        "name": "Coding agent memory",
        "status": "warn" if incomplete else "ok",
        "detail": "; ".join(details),
        "fix": "Run smem init --client codex|claude|all; verify hook trust and recovery in the client",
    }
