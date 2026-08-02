"""Shared CLI helpers for configuration, storage, and output formatting."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import typer

from surreal_memory.cli.config import CLIConfig
from surreal_memory.cli.storage import PersistentStorage

logger = logging.getLogger(__name__)

T = TypeVar("T")


def resolve_brain(brain: str | None, config: CLIConfig, *, default: str | None = None) -> str:
    """Resolve the effective brain name: explicit arg > env var > default/config.

    Every CLI command that needs the brain name before calling ``get_storage``
    (e.g. to echo progress) must resolve it through this helper rather than
    ``brain or config.current_brain`` — that pattern always produces a
    non-``None`` name, which makes ``get_storage`` skip its own env-var check
    (see its docstring) and silently ignores ``SURREAL_MEMORY_BRAIN``.
    """
    if brain is not None:
        return brain
    env_brain = os.environ.get("SURREAL_MEMORY_BRAIN")
    if env_brain:
        return env_brain
    return default if default is not None else config.current_brain


# Track storages created during a CLI command so we can close them before
# the event loop shuts down (prevents "Event loop is closed" noise from
# aiosqlite's background thread).
_active_storages: list[Any] = []


def get_config() -> CLIConfig:
    """Get CLI configuration."""
    return CLIConfig.load()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async CLI command with proper storage cleanup.

    Replaces bare ``asyncio.run()`` to ensure aiosqlite connections are
    closed *before* the event loop is torn down.
    """
    from surreal_memory.utils.sandbox import ensure_aiosqlite_or_exit_cli

    try:
        ensure_aiosqlite_or_exit_cli()
    except BaseException:
        coro.close()
        raise

    async def _with_cleanup() -> T:
        try:
            return await coro
        finally:
            for storage in _active_storages:
                if hasattr(storage, "close"):
                    try:
                        await storage.close()
                    except (OSError, RuntimeError):
                        logger.debug("Failed to close storage during cleanup", exc_info=True)
            _active_storages.clear()
            # Yield once so any pending callbacks from aiosqlite worker
            # threads are drained before asyncio.run() tears down the loop.
            # Prevents "Event loop is closed" on Python 3.12+.
            await asyncio.sleep(0)

    return asyncio.run(_with_cleanup())


async def get_storage(
    config: CLIConfig,
    *,
    brain_name: str | None = None,
    force_shared: bool = False,
    force_local: bool = False,
    force_unified: bool = False,
) -> PersistentStorage:
    """
    Get storage for current brain.

    Args:
        config: CLI configuration
        brain_name: Brain name override (default: config.current_brain)
        force_shared: Override config to use remote shared mode
        force_local: Override config to use local JSON mode
        force_unified: Override config to use the unified surrealdb/memory backend

    Returns:
        Storage instance (local JSON, unified surrealdb/memory, or remote shared)
    """
    # Priority: explicit arg > env var > config file
    name = resolve_brain(brain_name, config)

    # Remote shared mode (via server)
    use_shared = (config.is_shared_mode or force_shared) and not force_local and not force_unified
    if use_shared:
        from surreal_memory.storage.shared_store import SharedStorage

        storage = SharedStorage(
            server_url=config.shared.server_url,
            brain_id=name,
            timeout=config.shared.timeout,
            api_key=config.shared.api_key,
        )
        await storage.connect()
        _active_storages.append(storage)
        return storage  # type: ignore[return-value]

    # Unified config mode (surrealdb/memory backend)
    if config.use_unified_storage or force_unified:
        from surreal_memory.unified_config import get_shared_storage

        unified_storage = await get_shared_storage(name)
        _active_storages.append(unified_storage)
        return unified_storage  # type: ignore[return-value]

    # Legacy JSON mode
    brain_path = config.get_brain_path(name)
    return await PersistentStorage.load(brain_path)


def get_brain_path_auto(config: CLIConfig, brain_name: str | None = None) -> Path:
    """Get brain file path, choosing .db or .json based on storage mode."""
    if config.use_unified_storage:
        return config.get_brain_db_path(brain_name)
    return config.get_brain_path(brain_name)


def output_result(data: dict[str, Any], as_json: bool = False) -> None:
    """Output result in appropriate format."""
    if as_json:
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        # Human-readable format
        if "error" in data:
            typer.secho(f"Error: {data['error']}", fg=typer.colors.RED)
        elif "answer" in data:
            typer.echo(data["answer"])

            # Show freshness warnings
            if data.get("freshness_warnings"):
                typer.echo("")
                for warning in data["freshness_warnings"]:
                    typer.secho(warning, fg=typer.colors.YELLOW)

            # Show metadata
            meta_parts = []
            if data.get("confidence") is not None:
                meta_parts.append(f"confidence: {data['confidence']:.2f}")
            if data.get("neurons_activated"):
                meta_parts.append(f"neurons: {data['neurons_activated']}")
            if data.get("oldest_memory_age"):
                meta_parts.append(f"oldest: {data['oldest_memory_age']}")

            if meta_parts:
                typer.secho(f"\n[{', '.join(meta_parts)}]", fg=typer.colors.BRIGHT_BLACK)

            # Show routing info if present
            if data.get("routing"):
                r = data["routing"]
                typer.secho(
                    f"\n[routing: {r['query_type']}, depth: {r['suggested_depth']}, "
                    f"confidence: {r['confidence']}]",
                    fg=typer.colors.BRIGHT_BLACK,
                )

        elif "message" in data:
            typer.secho(data["message"], fg=typer.colors.GREEN)

            # Show memory type info
            type_parts = []
            if data.get("memory_type"):
                type_parts.append(f"type: {data['memory_type']}")
            if data.get("priority"):
                type_parts.append(f"priority: {data['priority']}")
            if data.get("expires_in_days") is not None:
                type_parts.append(f"expires: {data['expires_in_days']}d")
            if data.get("project"):
                type_parts.append(f"project: {data['project']}")
            if type_parts:
                typer.secho(f"  [{', '.join(type_parts)}]", fg=typer.colors.BRIGHT_BLACK)

            # Show warnings if any
            if data.get("warnings"):
                for warning in data["warnings"]:
                    typer.secho(warning, fg=typer.colors.YELLOW)

        elif "context" in data:
            typer.echo(data["context"])
        else:
            typer.echo(str(data))
