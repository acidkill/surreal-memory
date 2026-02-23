"""CLI command for migrating between storage backends."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer


def migrate(
    target: Annotated[
        str,
        typer.Argument(help="Target backend: 'falkordb' or 'sqlite'"),
    ],
    brain: Annotated[
        str | None,
        typer.Option("--brain", "-b", help="Specific brain to migrate (default: current)"),
    ] = None,
    falkordb_host: Annotated[
        str,
        typer.Option("--falkordb-host", help="FalkorDB host"),
    ] = "localhost",
    falkordb_port: Annotated[
        int,
        typer.Option("--falkordb-port", help="FalkorDB port"),
    ] = 6379,
) -> None:
    """Migrate brain data between storage backends.

    Example: nmem migrate falkordb --brain default
    """
    if target not in ("falkordb", "sqlite"):
        typer.secho(f"Unknown target backend: {target}", fg=typer.colors.RED)
        typer.echo("Supported targets: falkordb, sqlite")
        raise typer.Exit(1)

    if target == "falkordb":
        asyncio.run(
            _migrate_to_falkordb(
                brain_name=brain,
                host=falkordb_host,
                port=falkordb_port,
            )
        )
    else:
        typer.secho("SQLite -> SQLite migration not needed.", fg=typer.colors.YELLOW)
        raise typer.Exit(0)


async def _migrate_to_falkordb(
    brain_name: str | None,
    host: str,
    port: int,
) -> None:
    """Run the SQLite -> FalkorDB migration."""
    from neural_memory.storage.falkordb.falkordb_migration import (
        migrate_sqlite_to_falkordb,
    )
    from neural_memory.unified_config import get_config

    config = get_config()
    name = brain_name or config.current_brain
    db_path = str(config.get_brain_db_path(name))

    typer.secho(f"Migrating brain '{name}' from SQLite -> FalkorDB", bold=True)
    typer.echo(f"  Source: {db_path}")
    typer.echo(f"  Target: {host}:{port}")

    result = await migrate_sqlite_to_falkordb(
        sqlite_db_path=db_path,
        falkordb_host=host,
        falkordb_port=port,
        brain_name=name,
    )

    if result.get("success"):
        for brain_info in result.get("brains", []):
            typer.echo(
                f"  {brain_info['name']}: "
                f"{brain_info['neurons']} neurons, "
                f"{brain_info['synapses']} synapses, "
                f"{brain_info['fibers']} fibers"
            )
        typer.secho("Migration complete!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Migration failed: {result.get('error')}", fg=typer.colors.RED)
        raise typer.Exit(1)


def register(app: typer.Typer) -> None:
    """Register migrate command with app."""
    app.command()(migrate)
