"""Surface file resolver — finds the correct .nm file for the current context.

Searches project-level first, then global. Handles Windows/Unix paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_surface_path(brain_name: str = "default", *, for_write: bool = False) -> Path:
    """Resolve the surface.nm file path for a given brain.

    Priority:
    1. Project-level: ``<project_root>/.neuralmemory/surface.nm``
    2. Global: ``~/.neuralmemory/surfaces/<brain_name>.nm``

    For reads, project-level is returned only if the file exists.
    For writes (``for_write=True``), project-level is returned whenever
    a project root is detected — this ensures the first save creates
    the file at project level instead of always falling through to global.

    Args:
        brain_name: Brain to resolve surface for.
        for_write: If True, prefer project path even when file doesn't exist yet.

    Returns:
        Path to the surface file (may not exist yet for writes).
    """
    project_root = detect_project_root()
    if project_root is not None:
        project_surface = project_root / ".neuralmemory" / "surface.nm"
        if for_write or project_surface.exists():
            return project_surface

    return _global_surface_path(brain_name)


def _global_surface_path(brain_name: str = "default") -> Path:
    """Return the global surface path for a brain."""
    from neural_memory.unified_config import get_neuralmemory_dir

    return get_neuralmemory_dir() / "surfaces" / f"{brain_name}.nm"


def detect_project_root() -> Path | None:
    """Detect the project root directory.

    Walks up from CWD looking for common project markers.
    Returns None if no project root found (e.g., in home directory).

    Markers (first match wins):
    - ``.neuralmemory/`` directory (NM project config)
    - ``.git/`` directory
    - ``pyproject.toml``
    - ``package.json``
    - ``Cargo.toml``
    - ``go.mod``
    """
    markers = (
        ".neuralmemory",
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
    )

    try:
        current = Path.cwd().resolve()
    except OSError:
        return None

    home = Path.home().resolve()

    # Walk up, but don't go above home or root
    for _ in range(20):  # safety cap
        for marker in markers:
            if (current / marker).exists():
                return current

        parent = current.parent
        if parent == current or current == home:
            break
        current = parent

    return None


def load_surface_text(brain_name: str = "default") -> str | None:
    """Load surface.nm content if it exists.

    Args:
        brain_name: Brain name to resolve.

    Returns:
        File content as string, or None if not found/unreadable.
    """
    path = get_surface_path(brain_name)
    if not path.exists():
        return None

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("Failed to read surface file: %s", path, exc_info=True)
        return None


def save_surface_text(text: str, brain_name: str = "default") -> Path:
    """Write surface.nm content atomically.

    Creates parent directories if needed. Uses write-then-rename
    for atomic updates on supported platforms.

    Args:
        text: The .nm file content to write.
        brain_name: Brain name to resolve path for.

    Returns:
        Path where the file was written.
    """
    path = get_surface_path(brain_name, for_write=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Warn once if both project and global surfaces exist (stale global copy)
    global_path = _global_surface_path(brain_name)
    if path != global_path and global_path.exists():
        logger.info(
            "Project surface at %s takes priority; global copy at %s is now stale",
            path,
            global_path,
        )

    # Atomic write: write to temp, then rename
    tmp_path = path.with_suffix(".nm.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        # On Windows, target must not exist for rename
        if os.name == "nt" and path.exists():
            path.unlink()
        tmp_path.rename(path)
    except OSError:
        # Fallback: direct write
        logger.debug("Atomic write failed, falling back to direct write", exc_info=True)
        path.write_text(text, encoding="utf-8")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return path
