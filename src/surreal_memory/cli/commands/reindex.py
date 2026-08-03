"""``smem reindex`` — (re)compute embedding vectors for the current brain.

Embeddings can be silently off (or stale) because the stored ``brain.config``
keeps the defaults it was created with. This command re-embeds neurons using
the EFFECTIVE embedding provider (config.toml + env overrides), independent of
whatever the stored brain config says.

Design notes:
    - Idempotent: ``--missing-only`` (default) only embeds neurons that lack a
      vector; ``--all`` re-embeds everything.
    - ``--dry-run`` (default off) reports how many neurons *would* be embedded
      and writes nothing.
    - Fail-soft per neuron: a single embedding/update failure never aborts the
      run.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from surreal_memory.cli._helpers import get_config, get_storage, output_result, run_async

_BATCH_SIZE_DEFAULT = 64
# Consecutive embedding-batch failures tolerated while NOTHING has succeeded
# before giving up. A misconfigured endpoint fails every batch identically, so
# continuing only reprints the same error thousands of times; a genuinely
# transient blip clears well inside this margin (each batch already retries
# internally with backoff).
_ABORT_AFTER_FAILURES = 3


def reindex(
    brain: Annotated[
        str,
        typer.Option("--brain", "-b", help="Target brain name (default: current)"),
    ] = "",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report how many would be embedded; write nothing"),
    ] = False,
    all_neurons: Annotated[
        bool,
        typer.Option("--all", help="Re-embed every neuron (default: only missing vectors)"),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Neurons per embedding batch"),
    ] = _BATCH_SIZE_DEFAULT,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output as JSON"),
    ] = False,
) -> None:
    """Re-embed neurons for the current brain using the effective provider."""
    run_async(_reindex_async(brain, dry_run, all_neurons, batch_size, json_output))


def _needs_embedding(neuron: Any, *, all_neurons: bool) -> bool:
    """Return True when a neuron should be embedded for this run."""
    if not neuron.content.strip():
        return False
    if all_neurons:
        return True
    return not neuron.metadata.get("_embedding")


async def _reindex_async(
    brain: str,
    dry_run: bool,
    all_neurons: bool,
    batch_size: int,
    json_output: bool,
) -> None:
    """Async implementation of the reindex command."""
    if batch_size < 1:
        typer.echo("Error: --batch-size must be >= 1", err=True)
        raise typer.Exit(code=1)

    config = get_config()
    storage = await get_storage(config, brain_name=brain or None)

    brain_data = await storage.get_brain(storage.brain_id or "")
    if not brain_data:
        typer.echo("Error: No brain configured", err=True)
        raise typer.Exit(code=1)

    # Provider resolution honors the effective embedding config (config.toml +
    # env overrides), so a stale brain.config does not disable embeddings.
    from surreal_memory.engine.semantic_discovery import _create_provider, _effective_embedding

    enabled, provider_name, model_name = _effective_embedding(brain_data.config)
    if not enabled:
        typer.echo(
            "Error: embeddings are disabled in the effective config — "
            "enable them (config.toml [embedding] or SURREAL_MEMORY_EMBEDDING_ENABLED=true)",
            err=True,
        )
        raise typer.Exit(code=1)

    # Collect candidate neurons (paginate to avoid loading everything at once).
    candidates: list[Any] = []
    page = 5000
    offset = 0
    while True:
        page_neurons = await storage.find_neurons(limit=page, offset=offset)
        if not page_neurons:
            break
        candidates.extend(n for n in page_neurons if _needs_embedding(n, all_neurons=all_neurons))
        offset += len(page_neurons)
        if len(page_neurons) < page:
            break

    to_embed = len(candidates)

    if dry_run:
        _emit(
            json_output,
            {
                "dry_run": True,
                "provider": provider_name,
                "model": model_name,
                "mode": "all" if all_neurons else "missing-only",
                "would_embed": to_embed,
            },
            human=(
                f"[dry-run] provider={provider_name} model={model_name} "
                f"mode={'all' if all_neurons else 'missing-only'} "
                f"would embed {to_embed} neuron(s)"
            ),
        )
        return

    if to_embed == 0:
        _emit(
            json_output,
            {
                "dry_run": False,
                "provider": provider_name,
                "model": model_name,
                "embedded": 0,
                "failed": 0,
            },
            human="Nothing to embed — all neurons already have vectors.",
        )
        return

    try:
        provider = _create_provider(brain_data.config, task_type="RETRIEVAL_DOCUMENT")
    except Exception as exc:  # pragma: no cover - depends on optional package
        typer.echo(f"Error: embedding provider unavailable: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    embedded = 0
    failed = 0
    first_error = ""
    consecutive_failures = 0
    aborted = False
    for start in range(0, to_embed, batch_size):
        batch = candidates[start : start + batch_size]
        texts = [n.embedding_text() for n in batch]
        try:
            vectors = await provider.embed_batch(texts)
        except Exception as exc:
            failed += len(batch)
            consecutive_failures += 1
            # Report the REAL error once. Printing a bare "failed (skipped)" per
            # batch buried the only useful line (e.g. an HTTP 400 naming the
            # wrong host, or a physical-batch-size limit) under N identical
            # ones, turning a one-glance diagnosis into a debugging session.
            if not first_error:
                first_error = f"{type(exc).__name__}: {exc}"
                if not json_output:
                    typer.echo(f"  embedding failed: {first_error}", err=True)
            # Nothing has succeeded and the failures keep coming: the cause is
            # configuration, not this particular batch. Grinding through the
            # remaining thousands of neurons only repeats it.
            if embedded == 0 and consecutive_failures >= _ABORT_AFTER_FAILURES:
                aborted = True
                remaining = to_embed - (start + len(batch))
                failed += remaining
                if not json_output:
                    typer.echo(
                        f"  aborting after {consecutive_failures} consecutive failures "
                        f"with nothing embedded ({remaining} neuron(s) not attempted)",
                        err=True,
                    )
                break
            continue
        consecutive_failures = 0

        pairs = [(neuron.id, vector) for neuron, vector in zip(batch, vectors, strict=True)]
        try:
            await storage.update_neuron_embeddings(pairs)
            embedded += len(pairs)
        except Exception:
            failed += len(pairs)
            if not json_output:
                typer.echo(f"  batch {start}-{start + len(batch)} write failed (skipped)", err=True)
            continue

        if not json_output:
            typer.echo(f"  embedded {min(start + batch_size, to_embed)}/{to_embed}")

    payload: dict[str, Any] = {
        "dry_run": False,
        "provider": provider_name,
        "model": model_name,
        "mode": "all" if all_neurons else "missing-only",
        "embedded": embedded,
        "failed": failed,
    }
    if first_error:
        payload["error"] = first_error
    if aborted:
        payload["aborted"] = True
    _emit(
        json_output,
        payload,
        human=f"Embedded {embedded} neuron(s), {failed} failed.",
    )

    # A run that embedded nothing is a failure, not a quiet success: exiting 0
    # let `smem reindex` report "0 embedded, 4096 failed" while any script or
    # scheduler calling it recorded a clean run.
    if failed and embedded == 0:
        raise typer.Exit(code=1)


def _emit(json_output: bool, data: dict[str, Any], *, human: str) -> None:
    """Print either JSON or a human-readable line."""
    if json_output:
        output_result(data, as_json=True)
    else:
        typer.echo(human)


def register(app: typer.Typer) -> None:
    """Register the reindex command with the CLI app."""
    app.command(name="reindex")(reindex)
