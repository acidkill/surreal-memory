"""Environment/sandbox guards for CLI execution.

Reconstructed module: ``cli/_helpers.run_async`` imports
``ensure_sqlite_or_exit_cli`` from here, but the module was absent from the
tree — a regression from the SurrealDB-only refactor (commit 1f6fe80) that
broke every CLI command routed through ``run_async`` with a ``ModuleNotFoundError``.
This restores a tolerant guard so the CLI runs again.
"""

from __future__ import annotations

import sys


def ensure_sqlite_or_exit_cli() -> None:
    """Verify the local persistence stack can run, else exit the CLI cleanly.

    SQLite (stdlib ``sqlite3``) is the minimum requirement for any local-mode
    CLI command. In rare restricted sandboxes it is unavailable; fail fast with
    a clear message instead of a cryptic crash mid-command. This is a no-op in
    normal environments.
    """
    try:
        import sqlite3  # noqa: F401
    except Exception as exc:  # pragma: no cover - only in restricted sandboxes
        print(  # noqa: T201
            "surreal-memory: local SQLite (sqlite3) is unavailable in this "
            "environment; cannot run local-mode CLI commands.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
