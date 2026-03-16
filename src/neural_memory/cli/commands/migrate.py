"""CLI command for migrating between storage backends."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import typer


def migrate(
    target: Annotated[
        str,
        typer.Argument(help="Target backend: 'falkordb', 'postgres', or 'sqlite'"),
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
    pg_host: Annotated[
        str,
        typer.Option("--pg-host", help="PostgreSQL host"),
    ] = "localhost",
    pg_port: Annotated[
        int,
        typer.Option("--pg-port", help="PostgreSQL port"),
    ] = 5432,
    pg_database: Annotated[
        str,
        typer.Option("--pg-database", help="PostgreSQL database name"),
    ] = "neuralmemory",
    pg_user: Annotated[
        str,
        typer.Option("--pg-user", help="PostgreSQL user"),
    ] = "postgres",
    pg_password: Annotated[
        str,
        typer.Option("--pg-password", help="PostgreSQL password (or use NEURAL_MEMORY_POSTGRES_PASSWORD env)"),
    ] = "",
) -> None:
    """Migrate brain data between storage backends.

    Examples:
        nmem migrate falkordb --brain default
        nmem migrate postgres --brain default --pg-host localhost --pg-database neuralmem
    """
    supported = ("falkordb", "postgres", "sqlite")
    if target not in supported:
        typer.secho(f"Unknown target backend: {target}", fg=typer.colors.RED)
        typer.echo(f"Supported targets: {', '.join(supported)}")
        raise typer.Exit(1)

    if target == "falkordb":
        asyncio.run(
            _migrate_to_falkordb(
                brain_name=brain,
                host=falkordb_host,
                port=falkordb_port,
            )
        )
    elif target == "postgres":
        asyncio.run(
            _migrate_to_postgres(
                brain_name=brain,
                host=pg_host,
                port=pg_port,
                database=pg_database,
                user=pg_user,
                password=pg_password,
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

    _print_result(result)


async def _migrate_to_postgres(
    brain_name: str | None,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
) -> None:
    """Run the SQLite -> PostgreSQL migration."""
    import os

    from neural_memory.storage.postgres.postgres_migration import (
        migrate_sqlite_to_postgres,
    )
    from neural_memory.unified_config import get_config

    config = get_config()
    name = brain_name or config.current_brain
    db_path = str(config.get_brain_db_path(name))

    # Allow env var override for password
    effective_password = password or os.environ.get("NEURAL_MEMORY_POSTGRES_PASSWORD", "")

    typer.secho(f"Migrating brain '{name}' from SQLite -> PostgreSQL", bold=True)
    typer.echo(f"  Source: {db_path}")
    typer.echo(f"  Target: {user}@{host}:{port}/{database}")

    result = await migrate_sqlite_to_postgres(
        sqlite_db_path=db_path,
        pg_host=host,
        pg_port=port,
        pg_database=database,
        pg_user=user,
        pg_password=effective_password,
        brain_name=name,
    )

    _print_result(result)


def _print_result(result: dict[str, Any]) -> None:
    """Print migration result."""
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
