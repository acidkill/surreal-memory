"""Surreal-Memory CLI main entry point."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Annotated

import typer

# Main app
app = typer.Typer(
    name="smem",
    help="Surreal-Memory - Reflex-based memory for AI agents",
    no_args_is_help=True,
)

_MACHINE_ORIENTED_UPDATE_CHECK_SKIP_COMMANDS = frozenset(
    {
        "context",
        "recall",
        "stats",
        "status",
    }
)


def _version_callback(value: bool) -> None:
    """Print version and exit when --version is passed."""
    if value:
        from surreal_memory import __version__

        typer.echo(f"surreal-memory {__version__}")
        raise typer.Exit()


def _warn_if_not_initialized(ctx: typer.Context) -> None:
    """Print a one-line hint if Surreal-Memory has never been initialized.

    Skipped for the 'init' command itself to avoid noise during setup.
    """
    if ctx.invoked_subcommand == "init":
        return
    from surreal_memory.unified_config import get_surrealmemory_dir

    config_path = get_surrealmemory_dir() / "config.toml"
    if not config_path.exists():
        typer.secho(
            "Tip: Surreal-Memory not set up yet. Run 'smem init' to get started.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _args_request_json(args: Sequence[str]) -> bool:
    """Return True when the raw CLI invocation requests JSON output."""
    return any(arg in {"--json", "-j"} or arg.startswith("--json=") for arg in args)


def _should_run_update_check(ctx: typer.Context, args: Sequence[str] | None = None) -> bool:
    """Decide whether this command should emit opportunistic update notices."""
    if ctx.invoked_subcommand in _MACHINE_ORIENTED_UPDATE_CHECK_SKIP_COMMANDS:
        return False

    raw_args = sys.argv[1:] if args is None else args
    return not _args_request_json(raw_args)


@app.callback(invoke_without_command=True)
def _app_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Global callback: runs before every command."""
    if ctx.invoked_subcommand is None:
        return

    _warn_if_not_initialized(ctx)

    if not _should_run_update_check(ctx):
        return

    from surreal_memory.cli.update_check import run_update_check_background

    run_update_check_background()


# Register sub-apps (brain, project, shared)
from surreal_memory.cli.commands.brain import brain_app  # noqa: E402
from surreal_memory.cli.commands.config_cmd import config_app  # noqa: E402
from surreal_memory.cli.commands.habits import habits_app  # noqa: E402
from surreal_memory.cli.commands.project import project_app  # noqa: E402
from surreal_memory.cli.commands.reasoning import reasoning_app  # noqa: E402
from surreal_memory.cli.commands.shared import shared_app  # noqa: E402
from surreal_memory.cli.commands.storage import storage_app  # noqa: E402
from surreal_memory.cli.commands.telegram import app as telegram_app  # noqa: E402
from surreal_memory.cli.commands.version import version_app  # noqa: E402

app.add_typer(brain_app, name="brain")
app.add_typer(config_app, name="config")
app.add_typer(project_app, name="project")
app.add_typer(shared_app, name="shared")
app.add_typer(storage_app, name="storage")
app.add_typer(habits_app, name="habits")
app.add_typer(reasoning_app, name="reasoning")
app.add_typer(version_app, name="version")
app.add_typer(telegram_app, name="telegram")

# "sync" is an alias for "shared" — users expect `smem sync activate`
app.add_typer(shared_app, name="sync")

# Register top-level commands
from surreal_memory.cli.commands import (  # noqa: E402
    codebase,
    info,
    listing,
    memory,
    reindex,
    shortcuts,
    tools,
    train,
    update,
    watch,
)

memory.register(app)
listing.register(app)
info.register(app)
tools.register(app)
shortcuts.register(app)
codebase.register(app)
train.register(app)
reindex.register(app)
update.register(app)
watch.register(app)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
