"""Knowledge Surface commands.

``smem doctor`` has always prescribed ``smem surface generate`` when a brain has
no surface, but the command did not exist -- following the advice produced
``No such command 'surface'``. This module makes the prescription true.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from surreal_memory.cli._helpers import (
    get_config,
    get_storage,
    output_result,
    resolve_brain,
    run_async,
)

surface_app = typer.Typer(help="Knowledge Surface commands")


def _resolved_brain(brain: str | None) -> tuple[Any, str]:
    """Return ``(config, brain_name)`` with the name validated for a file path."""
    from surreal_memory.surface.resolver import validate_brain_name

    config = get_config()
    name = resolve_brain(brain, config)
    try:
        return config, validate_brain_name(name)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@surface_app.command("generate")
def surface_generate(
    brain: Annotated[str | None, typer.Option("--brain", "-b", help="Brain name")] = None,
    global_path: Annotated[
        bool,
        typer.Option(
            "--global-path",
            help="Write to ~/.surrealmemory/surfaces/<brain>.nm regardless of the directory",
        ),
    ] = False,
    token_budget: Annotated[
        int, typer.Option("--token-budget", help="Max tokens for the surface")
    ] = 1200,
    max_graph_nodes: Annotated[
        int, typer.Option("--max-nodes", help="Max graph nodes to include")
    ] = 30,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Regenerate the Knowledge Surface from the brain.

    Without --global-path the surface lands next to the project you are standing
    in, which means the file you get depends on your shell's current directory.
    Use --global-path when you want one predictable location -- notably when
    satisfying `smem doctor`, which is usually run from more than one directory.

    Examples:
        smem surface generate
        smem surface generate --global-path
        smem surface generate --brain work --global-path --json
    """

    async def _generate() -> None:
        from surreal_memory.surface.lifecycle import regenerate_surface
        from surreal_memory.surface.resolver import get_surface_path

        config, brain_name = _resolved_brain(brain)
        storage = await get_storage(config, brain_name=brain_name)

        surface = await regenerate_surface(
            storage=storage,
            brain_name=brain_name,
            token_budget=token_budget,
            max_graph_nodes=max_graph_nodes,
            global_only=global_path,
        )
        path = get_surface_path(brain_name, for_write=True, global_only=global_path)

        output_result(
            {
                "brain": brain_name,
                "path": str(path),
                "graph_nodes": len(surface.graph),
                "clusters": len(surface.clusters),
                "signals": len(surface.signals),
                "token_estimate": surface.token_estimate(),
            },
            json_output,
        )

    run_async(_generate())


@surface_app.command("show")
def surface_show(
    brain: Annotated[str | None, typer.Option("--brain", "-b", help="Brain name")] = None,
    global_path: Annotated[
        bool,
        typer.Option("--global-path", help="Read the global surface instead of a project one"),
    ] = False,
    raw: Annotated[bool, typer.Option("--raw", help="Print the raw .nm text")] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show the current Knowledge Surface.

    Examples:
        smem surface show
        smem surface show --raw
        smem surface show --global-path --json
    """

    async def _show() -> None:
        from surreal_memory.surface.lifecycle import show_surface
        from surreal_memory.surface.resolver import get_surface_path

        _config, brain_name = _resolved_brain(brain)
        info = await show_surface(brain_name, global_only=global_path)
        text = info.pop("surface_text", "")
        info["path"] = str(get_surface_path(brain_name, global_only=global_path))

        if raw and text:
            typer.echo(text)
            return

        output_result(info, json_output)

    run_async(_show())
