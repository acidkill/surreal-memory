"""Reasoning-training CLI commands (status / mine / patterns / clear).

Thin wrappers over the reasoning engine + storage (NOT the HTTP dashboard router):
mine model ``thinking`` into ReasoningBank patterns and inspect/clear them.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from surreal_memory.cli._helpers import get_config, get_storage, output_result, run_async

reasoning_app = typer.Typer(
    help="Reasoning-training commands (mine model thinking into strategies)"
)

_PATTERN_FETCH_LIMIT = 5000


def _split_models(models: str | None) -> tuple[str, ...]:
    """Parse a comma-separated --models value into a tuple of model names."""
    if not models:
        return ()
    return tuple(m.strip() for m in models.split(",") if m.strip())


@reasoning_app.command("status")
def reasoning_status(
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show reasoning-training status: trace counts, learned patterns, per-model coverage.

    Examples:
        smem reasoning status
        smem reasoning status --json
    """

    async def _status() -> None:
        from surreal_memory.engine.reasoning_distiller import reasoning_coverage
        from surreal_memory.unified_config import get_config as get_unified_config

        config = get_config()
        storage = await get_storage(config)
        try:
            brain_id = storage.brain_id or ""
            if not brain_id:
                typer.secho("No brain configured.", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)

            ucfg = get_unified_config()
            stats = await storage.get_reasoning_stats(brain_id)
            fibers = await storage.find_fibers(
                metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
            )
            pattern_counts: dict[str, int] = {}
            for f in fibers:
                m = str(f.metadata.get("_source_model", ""))
                if m:
                    pattern_counts[m] = pattern_counts.get(m, 0) + 1

            models = sorted(set(stats.get("by_model", {})) | set(pattern_counts))
            per_model: list[dict[str, Any]] = []
            for m in models:
                cov = await reasoning_coverage(storage, m, ucfg)
                tstats = stats.get("by_model", {}).get(m, {})
                per_model.append(
                    {
                        "model": m,
                        "trace_count": tstats.get("trace_count", 0),
                        "unprocessed": tstats.get("unprocessed", 0),
                        "pattern_count": pattern_counts.get(m, 0),
                        "coverage_percent": cov["coverage_percent"],
                    }
                )

            result = {
                "total_traces": stats.get("total", 0),
                "unprocessed_traces": stats.get("unprocessed", 0),
                "total_patterns": len(fibers),
                "mining_enabled": ucfg.reasoning_training.mining_enabled,
                "injection_enabled": ucfg.reasoning_training.injection_enabled,
                "models": per_model,
            }

            if json_output:
                output_result(result, True)
            else:
                typer.echo(
                    f"Reasoning traces: {result['total_traces']} "
                    f"({result['unprocessed_traces']} unprocessed)"
                )
                typer.echo(f"Learned patterns: {result['total_patterns']}")
                typer.echo(
                    f"Mining: {'on' if result['mining_enabled'] else 'off'} | "
                    f"Injection: {'on' if result['injection_enabled'] else 'off'}"
                )
                if per_model:
                    typer.echo("\nPer model:")
                    for pm in per_model:
                        typer.echo(
                            f"  {pm['model']}: {pm['trace_count']} traces, "
                            f"{pm['pattern_count']} patterns, {pm['coverage_percent']}% coverage"
                        )
        finally:
            await storage.close()

    run_async(_status())


@reasoning_app.command("mine")
def reasoning_mine(
    backfill: Annotated[
        bool, typer.Option("--backfill", help="Scan the full history (scan_lookback_days=0)")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-n", help="Don't write — report a no-op")
    ] = False,
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated source models to restrict to")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Run even if mining is disabled in config")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Mine reasoning traces from ~/.claude transcripts and distill patterns.

    Examples:
        smem reasoning mine
        smem reasoning mine --backfill --models claude-fable-5,claude-sonnet-5
    """

    async def _mine() -> None:
        from dataclasses import replace as dc_replace

        from surreal_memory.engine.reasoning_distiller import distill_reasoning_patterns
        from surreal_memory.engine.reasoning_miner import ingest_reasoning_traces
        from surreal_memory.unified_config import get_config as get_unified_config

        config = get_config()
        storage = await get_storage(config)
        try:
            brain_id = storage.brain_id or ""
            if not brain_id:
                typer.secho("No brain configured.", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)

            ucfg = get_unified_config()
            if not ucfg.reasoning_training.mining_enabled and not force:
                # Privacy: don't scan ~/.claude transcripts unless mining is enabled.
                typer.secho(
                    "Mining is disabled. Enable it in config or pass --force.",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
                raise typer.Exit(1)

            if dry_run:
                result = {"traces_ingested": 0, "patterns_learned": 0, "dry_run": True}
                if json_output:
                    output_result(result, True)
                else:
                    typer.echo("Dry run: no changes.")
                return

            rt = ucfg.reasoning_training
            overrides: dict[str, Any] = {}
            if backfill:
                overrides["scan_lookback_days"] = 0
            model_list = _split_models(models)
            if model_list:
                overrides["mining_models"] = model_list
            run_cfg = dc_replace(ucfg, reasoning_training=dc_replace(rt, **overrides))

            ingest = await ingest_reasoning_traces(storage, brain_id, run_cfg)
            distill = await distill_reasoning_patterns(storage, brain_id, run_cfg)
            result = {
                "traces_ingested": ingest.traces_ingested,
                "patterns_learned": distill.patterns_learned,
            }
            if json_output:
                output_result(result, True)
            else:
                typer.echo(
                    f"Mined {result['traces_ingested']} traces, "
                    f"learned {result['patterns_learned']} patterns."
                )
        finally:
            await storage.close()

    run_async(_mine())


@reasoning_app.command("patterns")
def reasoning_patterns(
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Filter by source model")
    ] = None,
    category: Annotated[
        str | None, typer.Option("--category", "-c", help="Filter by category")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List learned reasoning patterns.

    Examples:
        smem reasoning patterns
        smem reasoning patterns --model claude-fable-5 --category debugging
    """

    async def _patterns() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            if not (storage.brain_id or ""):
                typer.secho("No brain configured.", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)
            fibers = await storage.find_fibers(
                metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
            )
            rows: list[dict[str, Any]] = []
            for f in fibers:
                md = f.metadata
                if model and md.get("_source_model") != model:
                    continue
                if category and md.get("_reasoning_category") != category:
                    continue
                rows.append(
                    {
                        "id": f.id,
                        "source_model": md.get("_source_model", ""),
                        "category": md.get("_reasoning_category", ""),
                        "title": md.get("_reasoning_title", ""),
                        "confidence": md.get("_reasoning_confidence", 0.0),
                        "frequency": md.get("_reasoning_frequency", 0),
                    }
                )
            rows.sort(
                key=lambda r: float(r["confidence"] or 0.0) * float(r["frequency"] or 0),
                reverse=True,
            )

            if json_output:
                output_result({"patterns": rows, "count": len(rows)}, True)
            else:
                if not rows:
                    typer.echo("No learned patterns.")
                    return
                typer.echo(f"Learned patterns ({len(rows)}):")
                for r in rows:
                    typer.echo(
                        f"  [{r['category']}] {r['title']} — {r['source_model']} "
                        f"(conf {float(r['confidence'] or 0.0):.2f}, freq {r['frequency']})"
                    )
        finally:
            await storage.close()

    run_async(_patterns())


@reasoning_app.command("clear")
def reasoning_clear(
    model: Annotated[str, typer.Option("--model", "-m", help="Model whose staged traces to wipe")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Wipe staged reasoning traces for a model (privacy).

    Examples:
        smem reasoning clear --model claude-fable-5
        smem reasoning clear --model claude-fable-5 --force
    """

    async def _clear() -> None:
        config = get_config()
        storage = await get_storage(config)
        try:
            brain_id = storage.brain_id or ""
            if not brain_id:
                typer.secho("No brain configured.", fg=typer.colors.RED, err=True)
                raise typer.Exit(1)

            if not force and not typer.confirm(f"Wipe all staged reasoning traces for {model!r}?"):
                typer.echo("Cancelled.")
                return

            deleted = await storage.delete_reasoning_traces_by_model(brain_id, model)
            if json_output:
                output_result({"deleted": deleted}, True)
            else:
                typer.echo(f"Wiped {deleted} reasoning trace(s) for {model}.")
        finally:
            await storage.close()

    run_async(_clear())
