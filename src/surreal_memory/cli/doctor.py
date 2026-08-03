"""System health diagnostic — smem doctor.

Checks Python version, dependencies, config validity, brain accessibility,
embedding provider, storage integrity, schema version, hooks, dedup,
and knowledge surface. Produces green/yellow/red status per check
with actionable fix suggestions. Supports --fix for auto-remediation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

from surreal_memory.cli._helpers import run_async

# Check result constants
OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

# Priority tiers (used to group output so users know which warnings actually matter)
TIER_CORE = "core"  # required for basic operation
TIER_RECOMMENDED = "recommended"  # recommended for full power (embeddings, MCP, hooks)
TIER_OPTIONAL = "optional"  # nice-to-have (surface snapshot, pro plugin, etc.)
TIER_DEV = "dev"  # contributor/source checkout diagnostics

# Check name → tier mapping. Unassigned defaults to RECOMMENDED.
_CHECK_TIERS: dict[str, str] = {
    "Python version": TIER_CORE,
    "Configuration": TIER_CORE,
    "Storage backend": TIER_CORE,
    "Brain database": TIER_CORE,
    "Dependencies": TIER_CORE,
    "CLI tools": TIER_CORE,
    "SurrealDB connection": TIER_CORE,
    "SurrealDB version": TIER_CORE,
    "Embedding provider": TIER_RECOMMENDED,
    "MCP env completeness": TIER_RECOMMENDED,
    "MCP configuration": TIER_RECOMMENDED,
    "MCP server": TIER_RECOMMENDED,
    "Hooks": TIER_RECOMMENDED,
    "Dedup": TIER_OPTIONAL,
    "Knowledge surface": TIER_OPTIONAL,
    "Config freshness": TIER_OPTIONAL,
    "Pro features": TIER_OPTIONAL,
    "Orphan fibers": TIER_OPTIONAL,
    "Source checkout": TIER_DEV,
    "Editable install": TIER_DEV,
    "Dev dependencies": TIER_DEV,
    "Checkout version": TIER_DEV,
}

QUICKSTART_URL = "https://acidkill.github.io/surreal-memory/guides/quickstart/"


def run_doctor(
    *,
    json_output: bool = False,
    fix: bool = False,
    dev: bool = False,
) -> dict[str, Any]:
    """Run all diagnostic checks and return results.

    Args:
        json_output: Return machine-readable output.
        fix: Auto-fix what's possible (enable config flags, install hooks).
        dev: Include source checkout diagnostics for contributors.
    """
    checks: list[dict[str, Any]] = []

    checks.append(_check_python_version())
    checks.append(_check_config())
    checks.append(_check_storage_backend())
    checks.append(_check_brain())
    checks.append(_check_dependencies())
    checks.append(_check_embedding_provider())
    checks.append(_check_mcp_config())
    checks.append(_check_mcp_connection())
    checks.append(_check_hooks())
    checks.append(_check_dedup())
    checks.append(_check_surface())
    checks.append(_check_config_freshness())
    checks.append(_check_cli_tools())
    checks.append(_check_surrealdb_connection())
    checks.append(_check_surrealdb_version())
    checks.append(_check_mcp_env_completeness())
    checks.append(_check_pro_plugin())
    if dev:
        checks.extend(_check_dev_environment())

    # Annotate every check with its priority tier (see _CHECK_TIERS).
    for check in checks:
        check["tier"] = _CHECK_TIERS.get(check.get("name", ""), TIER_RECOMMENDED)

    # Auto-fix pass
    if fix:
        checks = _auto_fix(checks)

    result = {
        "checks": checks,
        "passed": sum(1 for c in checks if c["status"] == OK),
        "warnings": sum(1 for c in checks if c["status"] == WARN),
        "failed": sum(1 for c in checks if c["status"] == FAIL),
        "total": len(checks),
    }

    if not json_output:
        _render_results(result)

    return result


def _source_checkout_root() -> Path | None:
    """Return the repository root when running from a source checkout."""
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").exists() and (root / "src" / "surreal_memory").exists():
        return root
    return None


def _read_checkout_version(root: Path) -> str | None:
    """Read ``project.version`` from a source checkout."""
    try:
        import tomllib

        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        version = data.get("project", {}).get("version")
        return str(version) if version else None
    except Exception:
        return None


def _check_dev_environment() -> list[dict[str, Any]]:
    """Contributor-focused diagnostics for source checkouts."""
    root = _source_checkout_root()
    return [
        _check_dev_source_checkout(root),
        _check_dev_editable_install(root),
        _check_dev_dependencies(),
        _check_dev_checkout_version(root),
    ]


def _check_dev_source_checkout(root: Path | None = None) -> dict[str, Any]:
    """Check whether doctor is running from a repository checkout."""
    root = _source_checkout_root() if root is None else root
    if root is None:
        return {
            "name": "Source checkout",
            "status": SKIP,
            "detail": "not running from a source checkout",
        }
    return {
        "name": "Source checkout",
        "status": OK,
        "detail": str(root),
    }


def _check_dev_editable_install(root: Path | None = None) -> dict[str, Any]:
    """Check whether imports resolve to the current checkout."""
    root = _source_checkout_root() if root is None else root
    if root is None:
        return {
            "name": "Editable install",
            "status": SKIP,
            "detail": "source checkout not detected",
        }

    try:
        import surreal_memory

        package_path = Path(surreal_memory.__file__).resolve()
        package_path.relative_to(root / "src")
        return {
            "name": "Editable install",
            "status": OK,
            "detail": "importing from checkout src/",
        }
    except ValueError:
        return {
            "name": "Editable install",
            "status": WARN,
            "detail": "importing installed package, not checkout src/",
            "fix": 'Run: pip install -e ".[dev]"',
        }
    except Exception:
        return {
            "name": "Editable install",
            "status": WARN,
            "detail": "could not inspect import path",
            "fix": 'Run: pip install -e ".[dev]"',
        }


def _check_dev_dependencies() -> dict[str, Any]:
    """Check contributor tooling dependencies."""
    required = {
        "pytest": "pytest",
        "pytest-asyncio": "pytest_asyncio",
        "ruff": "ruff",
        "mypy": "mypy",
    }
    missing = []
    for package_name, module_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        return {
            "name": "Dev dependencies",
            "status": FAIL,
            "detail": f"missing: {', '.join(missing)}",
            "fix": 'Run: pip install -e ".[dev]"',
        }

    return {
        "name": "Dev dependencies",
        "status": OK,
        "detail": "pytest, pytest-asyncio, ruff, mypy available",
    }


def _check_dev_checkout_version(root: Path | None = None) -> dict[str, Any]:
    """Compare checkout version with the installed package metadata."""
    root = _source_checkout_root() if root is None else root
    if root is None:
        return {
            "name": "Checkout version",
            "status": SKIP,
            "detail": "source checkout not detected",
        }

    checkout_version = _read_checkout_version(root)
    if checkout_version is None:
        return {
            "name": "Checkout version",
            "status": WARN,
            "detail": "could not read pyproject.toml version",
        }

    try:
        installed_version = importlib.metadata.version("surreal-memory")
    except importlib.metadata.PackageNotFoundError:
        return {
            "name": "Checkout version",
            "status": WARN,
            "detail": f"checkout {checkout_version}; package not installed",
            "fix": 'Run: pip install -e ".[dev]"',
        }

    if installed_version == checkout_version:
        return {
            "name": "Checkout version",
            "status": OK,
            "detail": f"checkout {checkout_version}; installed {installed_version}",
        }

    return {
        "name": "Checkout version",
        "status": WARN,
        "detail": f"checkout {checkout_version}; installed {installed_version}",
        "fix": 'Run: pip install -e ".[dev]"',
    }


def _check_python_version() -> dict[str, Any]:
    """Check Python version is 3.11+."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 11):
        return {"name": "Python version", "status": OK, "detail": version_str}

    return {
        "name": "Python version",
        "status": FAIL,
        "detail": f"{version_str} (requires 3.11+)",
        "fix": "Install Python 3.11 or newer",
    }


def _check_config() -> dict[str, Any]:
    """Check config.toml exists and is valid."""
    from surreal_memory.unified_config import get_surrealmemory_dir

    data_dir = get_surrealmemory_dir()
    config_path = data_dir / "config.toml"

    if not config_path.exists():
        return {
            "name": "Configuration",
            "status": FAIL,
            "detail": f"{config_path} not found",
            "fix": "Run: smem init",
        }

    try:
        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
        return {
            "name": "Configuration",
            "status": OK,
            "detail": f"{config_path} (brain: {config.current_brain})",
        }
    except Exception:
        return {
            "name": "Configuration",
            "status": FAIL,
            "detail": "parse error — run: smem init --force",
            "fix": "Run: smem init --force",
        }


def _check_storage_backend() -> dict[str, Any]:
    """Report the active backend, warning when it is one that 3.0.0 drops."""
    try:
        from surreal_memory.unified_config import get_config

        backend = get_config(reload=True).storage_backend
    except Exception as e:
        return {
            "name": "Storage backend",
            "status": FAIL,
            "detail": f"could not resolve backend: {e}",
            "fix": "Run: smem init --force",
        }

    if backend == "surrealdb":
        return {"name": "Storage backend", "status": OK, "detail": "surrealdb"}

    return {
        "name": "Storage backend",
        "status": WARN,
        "detail": "memory — nothing is persisted; every memory is lost on exit",
        "fix": "Set SURREAL_MEMORY_STORAGE=surrealdb for a durable store",
    }


def _run_surrealdb_get_brain(brain_name: str) -> Any:
    """Look up a brain by name via a short-lived SurrealDB connection."""
    import asyncio

    from surreal_memory.storage.surrealdb.connection import SurrealSettings
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    async def _lookup() -> Any:
        settings = SurrealSettings.from_env()
        storage = SurrealDBStorage(
            url=settings.url,
            user=settings.user,
            password=settings.password,
            namespace=settings.namespace,
            database=settings.database,
        )
        try:
            await asyncio.wait_for(storage.initialize(), timeout=5)
            return await storage.get_brain(brain_name)
        finally:
            await storage.close()

    return run_async(_lookup())


def _check_brain() -> dict[str, Any]:
    """Check the configured brain exists and is accessible."""
    try:
        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
    except Exception as exc:
        return {
            "name": "Brain database",
            "status": WARN,
            "detail": f"could not load config to check brain: {type(exc).__name__}",
        }

    brain_name = config.current_brain

    if config.storage_backend == "memory":
        return {
            "name": "Brain database",
            "status": SKIP,
            "detail": "memory backend — non-persistent, no brain to check",
        }

    try:
        brain = _run_surrealdb_get_brain(brain_name)
    except Exception as exc:
        return {
            "name": "Brain database",
            "status": WARN,
            "detail": f"could not reach SurrealDB: {type(exc).__name__}",
            "fix": "Ensure SurrealDB is running and SURREALDB_URL is correct",
        }

    if brain is None:
        return {
            "name": "Brain database",
            "status": WARN,
            "detail": f"brain '{brain_name}' not created yet — will be created on first remember",
        }

    return {
        "name": "Brain database",
        "status": OK,
        "detail": f"{brain_name} (created {brain.created_at.date()})",
    }


def _check_dependencies() -> dict[str, Any]:
    """Check core dependencies are importable."""
    required = ["aiosqlite", "typer"]
    missing = []

    for dep in required:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        return {
            "name": "Dependencies",
            "status": FAIL,
            "detail": f"Missing: {', '.join(missing)}",
            "fix": "Run: pip install surreal-memory",
        }

    return {"name": "Dependencies", "status": OK, "detail": "all core deps available"}


def _check_embedding_provider() -> dict[str, Any]:
    """Check embedding provider availability."""
    try:
        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
    except Exception:
        return {
            "name": "Embedding provider",
            "status": SKIP,
            "detail": "config not loaded",
        }

    if not config.embedding.enabled:
        return {
            "name": "Embedding provider",
            "status": WARN,
            "detail": "disabled (semantic search unavailable)",
            "fix": "Run: smem setup embeddings",
        }

    provider = config.embedding.provider

    # Check if provider package is importable
    provider_checks: dict[str, str] = {
        "sentence_transformer": "sentence_transformers",
        "openai": "openai",
        "openrouter": "openai",
        "gemini": "google.genai",
        "ollama": "ollama",
    }

    module_name = provider_checks.get(provider)
    if module_name:
        try:
            importlib.import_module(module_name)
            # An importable package says nothing about the configuration it is
            # handed. A provider aimed at another provider's model silently
            # assumes a dimension, and the vector index then rejects every
            # write — so an incoherent triple is a failure, not an OK.
            from surreal_memory.engine.embedding.capability import check_embedding_coherence

            try:
                configured_dim = int(getattr(config.embedding, "dimension", 0) or 0)
            except (TypeError, ValueError):
                configured_dim = 0
            mismatch = check_embedding_coherence(
                provider, str(config.embedding.model or ""), configured_dim
            )
            if mismatch is not None:
                return {
                    "name": "Embedding provider",
                    "status": FAIL,
                    "detail": mismatch.summary,
                    "fix": mismatch.fix,
                }
            return {
                "name": "Embedding provider",
                "status": OK,
                "detail": f"{provider} (model: {config.embedding.model})",
            }
        except ImportError:
            install_hint = {
                "sentence_transformer": "pip install surreal-memory[embeddings]",
                "openai": "pip install surreal-memory[embeddings-openai]",
                "openrouter": "pip install surreal-memory[embeddings-openrouter]",
                "gemini": "pip install surreal-memory[embeddings-gemini]",
                "ollama": "pip install surreal-memory[embeddings]",
            }
            return {
                "name": "Embedding provider",
                "status": FAIL,
                "detail": f"{provider} configured but not installed",
                "fix": f"Run: {install_hint.get(provider, 'pip install surreal-memory[embeddings]')}",
            }

    return {
        "name": "Embedding provider",
        "status": OK,
        "detail": f"{provider} (model: {config.embedding.model})",
    }


def _check_mcp_config() -> dict[str, Any]:
    """Check MCP server is configured in Claude Code."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        return {
            "name": "MCP configuration",
            "status": WARN,
            "detail": "~/.claude.json not found",
            "fix": "Run: smem init",
        }

    try:
        data = json.loads(claude_json.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        if "surreal-memory" in servers:
            return {
                "name": "MCP configuration",
                "status": OK,
                "detail": "surreal-memory registered in Claude Code",
            }
        return {
            "name": "MCP configuration",
            "status": WARN,
            "detail": "surreal-memory not found in ~/.claude.json",
            "fix": "Run: smem init",
        }
    except (json.JSONDecodeError, OSError):
        return {
            "name": "MCP configuration",
            "status": WARN,
            "detail": "could not parse ~/.claude.json",
            "fix": "Run: smem init",
        }


def _check_mcp_connection() -> dict[str, Any]:
    """Test that the MCP server can actually start."""
    import subprocess

    smem_mcp = shutil.which("smem-mcp")
    if not smem_mcp:
        # Fallback to module execution
        smem_mcp = None

    try:
        cmd = [smem_mcp] if smem_mcp else [sys.executable, "-m", "surreal_memory.mcp"]
        result = subprocess.run(
            cmd,
            input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"doctor","version":"1.0"}}}\n',
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        # MCP server over stdio will output JSON-RPC response
        if result.stdout and "result" in result.stdout:
            return {
                "name": "MCP server",
                "status": OK,
                "detail": "server responds to initialize",
            }
        if result.returncode == 0 or result.stdout:
            return {
                "name": "MCP server",
                "status": OK,
                "detail": "server starts successfully",
            }
        return {
            "name": "MCP server",
            "status": WARN,
            "detail": f"server exited with code {result.returncode}",
            "fix": "Check: smem-mcp or python -m surreal_memory.mcp",
        }
    except subprocess.TimeoutExpired:
        # Timeout is actually expected — MCP servers run indefinitely on stdio
        return {
            "name": "MCP server",
            "status": OK,
            "detail": "server starts (stdio mode)",
        }
    except FileNotFoundError:
        return {
            "name": "MCP server",
            "status": WARN,
            "detail": "smem-mcp not found on PATH",
            "fix": "Run: pip install surreal-memory",
        }
    except Exception:
        return {
            "name": "MCP server",
            "status": WARN,
            "detail": "could not test MCP server",
        }


def _check_cli_tools() -> dict[str, Any]:
    """Check CLI tools are on PATH."""
    tools = ["smem", "smem-mcp"]
    found = [t for t in tools if shutil.which(t)]
    missing = [t for t in tools if t not in found]

    if not missing:
        return {
            "name": "CLI tools",
            "status": OK,
            "detail": "smem + smem-mcp on PATH",
        }

    if "smem" in missing:
        return {
            "name": "CLI tools",
            "status": FAIL,
            "detail": f"missing: {', '.join(missing)}",
            "fix": "Run: pip install surreal-memory",
        }

    return {
        "name": "CLI tools",
        "status": WARN,
        "detail": f"missing: {', '.join(missing)} (smem mcp fallback available)",
    }


def _check_pro_plugin() -> dict[str, Any]:
    """Check if Surreal-Memory Pro plugin is installed and active."""
    try:
        from surreal_memory.plugins import get_plugins, has_pro

        if has_pro():
            plugins = get_plugins()
            names = [f"{p.name} v{p.version}" for p in plugins]
            return {
                "name": "Pro plugin",
                "status": OK,
                "detail": ", ".join(names),
            }

        # Check if Pro package is importable but not registered
        try:
            import surreal_memory_pro  # noqa: F401

            return {
                "name": "Pro plugin",
                "status": WARN,
                "detail": "Package installed but not registered — check entry_points",
            }
        except ImportError:
            pass

        # Check license
        from surreal_memory.unified_config import get_config

        config = get_config()
        if config.is_pro():
            return {
                "name": "Pro plugin",
                "status": WARN,
                "detail": "License active but plugin not installed",
                "fix": "Run: pip install surreal-memory-pro",
            }

        return {
            "name": "Pro plugin",
            "status": SKIP,
            "detail": "Not installed (free tier)",
        }
    except Exception:
        return {
            "name": "Pro plugin",
            "status": SKIP,
            "detail": "Could not check",
        }


def _check_hooks() -> dict[str, Any]:
    """Check Claude Code hooks are installed."""
    claude_dir = Path.home() / ".claude"
    settings_path = claude_dir / "settings.json"

    if not settings_path.exists():
        return {
            "name": "Hooks",
            "status": WARN,
            "detail": "~/.claude/settings.json not found",
            "fix": "Run: smem init",
            "fixable": True,
        }

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "name": "Hooks",
            "status": WARN,
            "detail": "could not parse settings.json",
        }

    hooks_section = data.get("hooks", {})
    expected = ["PreCompact", "Stop", "PostToolUse"]
    found: list[str] = []

    for event in expected:
        entries = hooks_section.get(event, [])
        for entry in entries:
            for hook in entry.get("hooks", []):
                cmd = hook.get("command", "")
                if "surreal_memory" in cmd or "smem" in cmd:
                    found.append(event)
                    break

    # If hooks are wired through the smem-hook-env wrapper, that wrapper must be
    # resolvable on PATH — otherwise every hook fails with "command not found"
    # (regression seen when the local wrapper script went missing).
    uses_wrapper = any(
        "smem-hook-env" in hook.get("command", "")
        for event in expected
        for entry in hooks_section.get(event, [])
        for hook in entry.get("hooks", [])
    )
    if uses_wrapper and shutil.which("smem-hook-env") is None:
        return {
            "name": "Hooks",
            "status": WARN,
            "detail": "hooks call 'smem-hook-env' but it is not on PATH (hooks will fail)",
            "fix": (
                "Install it: cp scripts/smem-hook-env ~/.local/bin/ "
                "&& chmod +x ~/.local/bin/smem-hook-env"
            ),
            "fixable": False,
        }

    if len(found) == len(expected):
        return {
            "name": "Hooks",
            "status": OK,
            "detail": f"{len(found)}/{len(expected)} installed ({', '.join(found)})",
        }

    missing = [e for e in expected if e not in found]
    return {
        "name": "Hooks",
        "status": WARN,
        "detail": f"{len(found)}/{len(expected)} — missing: {', '.join(missing)}",
        "fix": "Run: smem init",
        "fixable": True,
    }


def _check_dedup() -> dict[str, Any]:
    """Check dedup is enabled in config."""
    try:
        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
    except Exception:
        return {"name": "Dedup", "status": SKIP, "detail": "config not loaded"}

    if config.dedup.enabled:
        return {"name": "Dedup", "status": OK, "detail": "enabled"}

    return {
        "name": "Dedup",
        "status": WARN,
        "detail": "disabled (duplicate memories not caught)",
        "fix": "Run: smem init --full",
        "fixable": True,
    }


def _check_surface() -> dict[str, Any]:
    """Check knowledge surface (.nm file) exists."""
    try:
        from surreal_memory.surface.resolver import get_surface_path
        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
        surface_path = get_surface_path(config.current_brain)

        if surface_path.exists():
            size_kb = surface_path.stat().st_size / 1024
            return {
                "name": "Knowledge surface",
                "status": OK,
                "detail": f"{surface_path.name} ({size_kb:.1f} KB)",
            }

        return {
            "name": "Knowledge surface",
            "status": WARN,
            "detail": "not generated yet",
            "fix": "Run: smem surface generate (via MCP or after first session)",
        }
    except Exception:
        return {
            "name": "Knowledge surface",
            "status": SKIP,
            "detail": "surface module not available",
        }


def _check_config_freshness() -> dict[str, Any]:
    """Check if config.toml has all sections from current version."""
    try:
        import tomllib

        from surreal_memory.unified_config import get_surrealmemory_dir

        config_path = get_surrealmemory_dir() / "config.toml"
        if not config_path.exists():
            return {
                "name": "Config freshness",
                "status": SKIP,
                "detail": "no config.toml",
            }

        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        expected_sections = [
            "brain",
            "embedding",
            "auto",
            "eternal",
            "maintenance",
            "conflict",
            "safety",
            "encryption",
            "write_gate",
            "dedup",
            "tool_memory",
        ]
        missing = [s for s in expected_sections if s not in raw]
        if missing:
            return {
                "name": "Config freshness",
                "status": WARN,
                "detail": f"missing sections: {', '.join(missing)}",
                "fix": "Run: smem doctor --fix",
                "fixable": True,
            }

        return {
            "name": "Config freshness",
            "status": OK,
            "detail": "all sections present",
        }
    except Exception:
        return {
            "name": "Config freshness",
            "status": SKIP,
            "detail": "could not check config freshness",
        }


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


def _auto_fix(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attempt to auto-fix fixable issues. Returns updated checks."""
    fixed_checks: list[dict[str, Any]] = []

    for check in checks:
        if check.get("fixable") and check["status"] in (WARN, FAIL):
            fixed = _try_fix(check)
            if fixed:
                fixed_checks.append(fixed)
                continue
        fixed_checks.append(check)

    return fixed_checks


def _try_fix(check: dict[str, Any]) -> dict[str, Any] | None:
    """Try to fix a single check. Returns updated check or None."""
    name = check["name"]

    handler = _FIX_HANDLERS.get(name)
    if handler is None:
        return None
    if name == "Embedding provider" and "disabled" not in check.get("detail", ""):
        return None
    result: dict[str, Any] | None = handler()
    if result:
        result["_fixed"] = True
    return result


_FIX_HANDLERS: dict[str, Any] = {
    "Hooks": lambda: _fix_hooks(),
    "Dedup": lambda: _fix_dedup(),
    "Embedding provider": lambda: _fix_embedding(),
    "Config freshness": lambda: _fix_config_freshness(),
    "MCP env completeness": lambda: _fix_mcp_env(),
    "SurrealDB connection": lambda: _fix_mcp_env(),
}


def _fix_hooks() -> dict[str, Any]:
    """Auto-fix: install missing hooks."""
    try:
        from surreal_memory.cli.setup import setup_hooks_claude

        status = setup_hooks_claude()
        if status in ("added", "exists"):
            return {
                "name": "Hooks",
                "status": OK,
                "detail": "auto-fixed: hooks installed",
            }
    except Exception:
        pass
    return {
        "name": "Hooks",
        "status": WARN,
        "detail": "auto-fix failed",
        "fix": "Run: smem init",
    }


def _fix_dedup() -> dict[str, Any]:
    """Auto-fix: enable dedup in config."""
    try:
        from dataclasses import replace

        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
        updated = replace(config, dedup=replace(config.dedup, enabled=True))
        updated.save()
        return {
            "name": "Dedup",
            "status": OK,
            "detail": "auto-fixed: enabled",
        }
    except Exception:
        pass
    return {
        "name": "Dedup",
        "status": WARN,
        "detail": "auto-fix failed",
    }


def _fix_embedding() -> dict[str, Any]:
    """Auto-fix: detect and enable embedding provider."""
    try:
        from surreal_memory.cli.full_setup import detect_embedding_provider, enable_config_defaults

        provider = detect_embedding_provider()
        if provider:
            enable_config_defaults(embedding_provider=provider)
            return {
                "name": "Embedding provider",
                "status": OK,
                "detail": f"auto-fixed: {provider['key']} enabled",
            }
    except Exception:
        pass
    return {
        "name": "Embedding provider",
        "status": WARN,
        "detail": "no provider available to auto-enable",
        "fix": "Run: smem setup embeddings",
    }


def _fix_config_freshness() -> dict[str, Any]:
    """Auto-fix: re-save config.toml to add missing sections with defaults."""
    try:
        from surreal_memory.unified_config import get_config

        config = get_config(reload=True)
        config.save()
        return {
            "name": "Config freshness",
            "status": OK,
            "detail": "auto-fixed: config.toml updated with new sections",
        }
    except Exception:
        pass
    return {
        "name": "Config freshness",
        "status": WARN,
        "detail": "auto-fix failed",
    }


# ---------------------------------------------------------------------------
# SurrealDB-specific checks
# ---------------------------------------------------------------------------


def _run_surrealdb_ping() -> None:
    """Attempt a real SurrealDB connection with a short timeout.

    Raises StorageAuthError on bad credentials, any other exception on
    connectivity issues.  Returns None on success.
    """
    import asyncio

    from surreal_memory.storage.surrealdb.connection import SurrealSettings
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    async def _ping() -> None:
        settings = SurrealSettings.from_env()
        storage = SurrealDBStorage(
            url=settings.url,
            user=settings.user,
            password=settings.password,
            namespace=settings.namespace,
            database=settings.database,
        )
        try:
            await asyncio.wait_for(storage.initialize(), timeout=5)
        finally:
            await storage.close()

    run_async(_ping())


def _check_surrealdb_connection() -> dict[str, Any]:
    """Check that the SurrealDB connection works (TIER_CORE for surrealdb backend).

    Returns SKIP when storage backend is not surrealdb.
    Returns FAIL with actionable hint when authentication fails.
    Returns WARN when SurrealDB is unreachable (not running, wrong URL, etc.).
    Returns OK on success.
    """
    import os

    from surreal_memory.storage.surrealdb.connection import StorageAuthError

    if os.environ.get("SURREAL_MEMORY_STORAGE") != "surrealdb":
        try:
            from surreal_memory.unified_config import get_config

            config = get_config(reload=True)
            if config.storage_backend != "surrealdb":
                return {
                    "name": "SurrealDB connection",
                    "status": SKIP,
                    "detail": "surrealdb backend not active",
                }
        except Exception as cfg_exc:
            return {
                "name": "SurrealDB connection",
                "status": WARN,
                "detail": f"could not load config to determine backend: {type(cfg_exc).__name__}",
                "fix": "Check config.toml is valid; run: smem init",
            }

    try:
        _run_surrealdb_ping()
        return {
            "name": "SurrealDB connection",
            "status": OK,
            "detail": "authenticated and connected",
        }
    except StorageAuthError:
        return {
            "name": "SurrealDB connection",
            "status": FAIL,
            "detail": "authentication failed (wrong password or user)",
            "fix": "Set SURREALDB_PASS in your MCP client env or run: smem doctor --fix",
            "fixable": True,
        }
    except Exception as exc:
        return {
            "name": "SurrealDB connection",
            "status": WARN,
            "detail": f"could not reach SurrealDB: {type(exc).__name__}",
            "fix": "Ensure SurrealDB is running and SURREALDB_URL is correct",
        }


def _run_surrealdb_version_probe() -> str:
    """Sign in to SurrealDB and return the raw server version string.

    Raises on connectivity/auth failure so the caller can WARN.
    """
    import asyncio

    from surrealdb import AsyncSurreal

    from surreal_memory.storage.surrealdb.connection import SurrealSettings

    async def _probe() -> str:
        settings = SurrealSettings.from_env()
        conn = AsyncSurreal(settings.url)
        await conn.signin({"username": settings.user, "password": settings.password})
        return str(await asyncio.wait_for(conn.version(), timeout=5))

    return run_async(_probe())


def _check_surrealdb_version() -> dict[str, Any]:
    """Check the SurrealDB server is >= MIN_SERVER_VERSION (TIER_CORE for surrealdb).

    SKIP when the surrealdb backend is not active; WARN on unreachable/unparsable
    version; FAIL on a confirmed too-old server; OK otherwise.
    """
    import os

    from surreal_memory.storage.surrealdb.connection import (
        MIN_SERVER_VERSION,
        parse_server_version,
    )

    if os.environ.get("SURREAL_MEMORY_STORAGE") != "surrealdb":
        try:
            from surreal_memory.unified_config import get_config

            if get_config(reload=True).storage_backend != "surrealdb":
                return {
                    "name": "SurrealDB version",
                    "status": SKIP,
                    "detail": "surrealdb backend not active",
                }
        except Exception:
            return {
                "name": "SurrealDB version",
                "status": SKIP,
                "detail": "storage backend could not be determined",
            }

    min_str = ".".join(str(p) for p in MIN_SERVER_VERSION)
    try:
        raw = _run_surrealdb_version_probe()
    except Exception as exc:
        return {
            "name": "SurrealDB version",
            "status": WARN,
            "detail": f"could not read version: {type(exc).__name__}",
            "fix": "Ensure SurrealDB is running and SURREALDB_URL is correct",
        }

    parsed = parse_server_version(raw)
    if parsed is None:
        return {
            "name": "SurrealDB version",
            "status": WARN,
            "detail": f"unrecognised version string '{raw}'",
        }
    if parsed < MIN_SERVER_VERSION:
        return {
            "name": "SurrealDB version",
            "status": FAIL,
            "detail": f"{raw} is older than the required {min_str}",
            "fix": (
                "Upgrade: docker compose -f docker-compose.surrealdb.yml pull && "
                "docker compose -f docker-compose.surrealdb.yml up -d "
                "(the surrealdb_data volume is preserved — back it up first)"
            ),
        }
    return {
        "name": "SurrealDB version",
        "status": OK,
        "detail": f"{raw} (>= {min_str})",
    }


def run_synapse_migration_command(action: str, *, json_output: bool = False) -> dict[str, Any]:
    """Handle ``smem doctor --synapse-migration {status|retry|purge-backup}``.

    - status: report schema_meta:version, the migration state, and backup row count.
    - retry: re-run apply_migrations (resumes a partial/failed synapse->RELATE migration).
    - purge-backup: drop the synapse_migration_backup table (post-migration cleanup).
    """
    valid = {"status", "retry", "purge-backup"}
    if action not in valid:
        raise ValueError(f"--synapse-migration must be one of {sorted(valid)}, got {action!r}")

    import json as json_mod

    from surrealdb import AsyncSurreal

    from surreal_memory.storage.surrealdb import migrations as migrations_mod
    from surreal_memory.storage.surrealdb.connection import SurrealSettings

    async def _run() -> dict[str, Any]:
        settings = SurrealSettings.from_env()
        conn = AsyncSurreal(settings.url)
        await conn.signin({"username": settings.user, "password": settings.password})
        await conn.use(settings.namespace, settings.database)

        if action == "status":
            version = await migrations_mod._read_stamped_version(conn)
            state = await migrations_mod._get_state(conn)
            backup_rows = await migrations_mod._count(conn, migrations_mod.BACKUP_TABLE)
            return {
                "action": "status",
                "schema_version": version,
                "migration_state": state,
                "backup_rows": backup_rows,
            }
        if action == "retry":
            final = await migrations_mod.apply_migrations(conn)
            return {"action": "retry", "schema_version": final}
        # purge-backup
        await conn.query(f"REMOVE TABLE IF EXISTS {migrations_mod.BACKUP_TABLE}")
        return {"action": "purge-backup", "dropped": migrations_mod.BACKUP_TABLE}

    result = run_async(_run())

    if json_output:
        print(json_mod.dumps(result, indent=2, default=str))
    elif action == "status":
        state = result.get("migration_state") or {}
        phase = state.get("phase", "n/a") if isinstance(state, dict) else "n/a"
        print(f"synapse->RELATE migration: schema_version={result['schema_version']} phase={phase}")
        print(f"  backup rows: {result['backup_rows']} (table {migrations_mod.BACKUP_TABLE})")
    elif action == "retry":
        print(
            f"synapse->RELATE migration re-run complete: schema_version={result['schema_version']}"
        )
    else:
        print(f"Dropped migration backup table: {result['dropped']}")

    return result


def _check_mcp_env_completeness() -> dict[str, Any]:
    """Check that MCP entries contain the required env block (TIER_RECOMMENDED).

    Reads ~/.claude.json (Claude Code) and claude_desktop_config.json to verify
    that the surreal-memory entry has SURREALDB_PASS in its env block.
    """
    configs_to_check: list[tuple[str, Path]] = [
        ("Claude Code (~/.claude.json)", Path.home() / ".claude.json"),
    ]

    # Also check Claude Desktop config
    try:
        from surreal_memory.cli.setup import _claude_desktop_config_path

        desktop_path = _claude_desktop_config_path()
        if desktop_path is not None:
            configs_to_check.append(("Claude Desktop", desktop_path))
    except Exception:
        pass

    missing_env: list[str] = []

    for label, config_path in configs_to_check:
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            entry = data.get("mcpServers", {}).get("surreal-memory")
            if entry is None:
                continue
            if not entry.get("env", {}).get("SURREALDB_PASS"):
                missing_env.append(label)
        except (json.JSONDecodeError, OSError):
            continue

    if not any(p.exists() for _, p in configs_to_check):
        return {
            "name": "MCP env completeness",
            "status": SKIP,
            "detail": "no MCP client config files found",
        }

    if missing_env:
        return {
            "name": "MCP env completeness",
            "status": WARN,
            "detail": f"surreal-memory entry missing env in: {', '.join(missing_env)}",
            "fix": "Run: smem doctor --fix",
            "fixable": True,
        }

    return {
        "name": "MCP env completeness",
        "status": OK,
        "detail": "env block with SURREALDB_PASS present",
    }


def _fix_mcp_env() -> dict[str, Any]:
    """Auto-fix: backfill env into MCP client configs."""
    from surreal_memory.cli.setup import setup_mcp_claude, setup_mcp_claude_desktop

    try:
        setup_mcp_claude()
        setup_mcp_claude_desktop()
        return {
            "name": "MCP env completeness",
            "status": OK,
            "detail": "auto-fixed: env backfilled in MCP client configs",
        }
    except Exception:
        pass
    return {
        "name": "MCP env completeness",
        "status": WARN,
        "detail": "auto-fix failed",
        "fix": "Run: smem init",
    }


def _render_results(result: dict[str, Any]) -> None:
    """Render diagnostic results to terminal."""
    typer.echo()
    typer.secho("  Surreal-Memory Doctor", bold=True)
    typer.secho("  ───────────────────", dim=True)
    typer.echo()

    icons = {
        OK: typer.style("[OK]", fg=typer.colors.GREEN),
        WARN: typer.style("[!!]", fg=typer.colors.YELLOW),
        FAIL: typer.style("[XX]", fg=typer.colors.RED),
        SKIP: typer.style("[--]", fg=typer.colors.BRIGHT_BLACK),
    }

    tier_labels = {
        TIER_CORE: ("CORE", "required for basic operation"),
        TIER_RECOMMENDED: ("RECOMMENDED", "full-power setup"),
        TIER_OPTIONAL: ("OPTIONAL", "nice-to-have, not needed for basic use"),
        TIER_DEV: ("DEV", "source checkout and contributor tooling"),
    }

    for tier in (TIER_CORE, TIER_RECOMMENDED, TIER_OPTIONAL, TIER_DEV):
        tier_checks = [c for c in result["checks"] if c.get("tier") == tier]
        if not tier_checks:
            continue
        label, hint = tier_labels[tier]
        typer.secho(f"  [{label}] ", fg=typer.colors.CYAN, bold=True, nl=False)
        typer.secho(hint, dim=True)
        for check in tier_checks:
            icon = icons.get(check["status"], icons[SKIP])
            typer.echo(f"    {icon} {check['name']:<22}{check['detail']}")
            if "fix" in check:
                typer.secho(f"         Fix: {check['fix']}", dim=True)
        typer.echo()

    typer.echo()
    passed = result["passed"]
    total = result["total"]
    warns = result["warnings"]
    fails = result["failed"]

    summary_parts = [f"{passed}/{total} passed"]
    if warns:
        summary_parts.append(f"{warns} warnings")
    if fails:
        summary_parts.append(f"{fails} failed")

    color = (
        typer.colors.RED
        if fails > 0
        else (typer.colors.YELLOW if warns > 0 else typer.colors.GREEN)
    )
    typer.secho(f"  {', '.join(summary_parts)}", fg=color, bold=True)

    # Suggest guide if there are issues
    if warns > 0 or fails > 0:
        typer.echo()
        typer.secho(f"  See full setup guide: {QUICKSTART_URL}", dim=True)
        if not any(c.get("_fixed") for c in result["checks"]):
            typer.secho("  Auto-fix available issues: smem doctor --fix", dim=True)

    typer.echo()
