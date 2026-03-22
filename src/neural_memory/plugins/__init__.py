"""Neural Memory plugin registry.

Extension point for neural-memory-pro and third-party plugins.
Plugins are auto-discovered via Python entry_points (group: neural_memory.plugins).

Usage (in your plugin's __init__.py):
    from neural_memory.plugins import register
    register(MyPlugin())

Or via pyproject.toml entry_points for auto-discovery:
    [project.entry-points."neural_memory.plugins"]
    pro = "neural_memory_pro:auto_register"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from neural_memory.plugins.base import ProPlugin

logger = logging.getLogger(__name__)

_plugins: list[ProPlugin] = []
_discovered = False


def register(plugin: ProPlugin) -> None:
    """Register a plugin instance."""
    if any(p.name == plugin.name for p in _plugins):
        logger.debug("Plugin '%s' already registered, skipping", plugin.name)
        return
    _plugins.append(plugin)
    logger.info("Registered plugin: %s v%s", plugin.name, plugin.version)


def discover() -> list[ProPlugin]:
    """Auto-discover plugins via entry_points. Idempotent."""
    global _discovered
    if _discovered:
        return _plugins

    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="neural_memory.plugins")
        for ep in eps:
            try:
                loader = ep.load()
                loader()  # Each entry point should call register()
                logger.debug("Loaded plugin entry point: %s", ep.name)
            except Exception:
                logger.warning("Failed to load plugin: %s", ep.name, exc_info=True)
    except Exception:
        logger.debug("Plugin discovery failed", exc_info=True)

    _discovered = True
    return _plugins


def get_plugins() -> list[ProPlugin]:
    """Get all registered plugins (triggers discovery on first call)."""
    if not _discovered:
        discover()
    return list(_plugins)


def get_retrieval_strategy(name: str) -> Callable[..., Any] | None:
    """Look up a retrieval strategy from any registered plugin."""
    for plugin in get_plugins():
        strategies = plugin.get_retrieval_strategies()
        if name in strategies:
            return strategies[name]
    return None


def get_compression_fn() -> Callable[..., Any] | None:
    """Get Pro compression function if available."""
    for plugin in get_plugins():
        fn = plugin.get_compression_fn()
        if fn is not None:
            return fn
    return None


def get_consolidation_strategy(name: str) -> Callable[..., Any] | None:
    """Look up a consolidation strategy from any registered plugin."""
    for plugin in get_plugins():
        strategies = plugin.get_consolidation_strategies()
        if name in strategies:
            return strategies[name]
    return None


def has_pro() -> bool:
    """Check if any Pro plugin is registered."""
    return len(get_plugins()) > 0


__all__ = [
    "ProPlugin",
    "discover",
    "get_compression_fn",
    "get_consolidation_strategy",
    "get_plugins",
    "get_retrieval_strategy",
    "has_pro",
    "register",
]
