"""Explicit coding-memory scopes shared by MCP and lifecycle adapters."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def project_identity(cwd: str | None = None) -> str:
    """Use an override, sanitized Git origin, or canonical repository path."""
    override = os.environ.get("SMEM_PROJECT", "").strip()
    if override:
        return override
    path = Path(cwd or os.getcwd()).resolve()
    git = shutil.which("git")
    if git is None:
        return str(path)
    try:
        root = subprocess.run(  # noqa: S603 — resolved git binary, fixed arguments
            [git, "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        ).stdout.strip()
        path = Path(root).resolve()
        remote = subprocess.run(  # noqa: S603 — resolved git binary, fixed arguments
            [git, "remote", "get-url", "origin"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        ).stdout.strip()
        if "://" not in remote and ":" in remote and "@" in remote:
            host, repo = remote.split(":", 1)
            remote = "ssh://" + host + "/" + repo
        url = urlsplit(remote)
        if url.hostname and url.path.strip("/"):
            return url.hostname.lower() + "/" + url.path.strip("/").removesuffix(".git")
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return str(path)


def scope_tag(args: dict[str, Any]) -> str | None:
    """Validate an explicitly requested scope; absent scope preserves legacy APIs."""
    scope = args.get("scope")
    if scope is None:
        if args.get("project_id") is not None or args.get("include_global_preferences"):
            raise ValueError("scope='project' is required for scoped memory")
        return None
    if scope == "global_preference":
        return "coding_global_preference"
    if scope != "project":
        raise ValueError("scope must be project or global_preference")
    project = args.get("project_id")
    if project is None:
        project = project_identity()
    if not isinstance(project, str) or not project.strip() or len(project) > 4096:
        raise ValueError("project_id must be a nonempty identifier (at most 4096 characters)")
    return "coding_project:" + hashlib.sha256(project.encode()).hexdigest()
