#!/usr/bin/env python
"""Compute and store `fiber.fiber_vec` for the current brain — the one-time backfill
`storage/base.py::find_fibers_by_embedding`'s docstring refers to.

Text embedded per fiber: `summary` when non-empty, else the content of its first eight member
neurons joined (truncated to 2000 chars) — same fallback measured on a copy of the production
brain, where 21 of 40 sampled fibers had no `summary`. A fiber with neither is skipped (never
embeds an empty string — `EmbeddingProvider.embed`/`embed_batch` already reject that).

This does NOT wire fiber embedding into the encode path (a new fiber gets `fiber_vec = NONE`
until this script runs again) — that is a deliberate follow-up, not part of this fix (see
`engine/retrieval.py`'s "FIBER VECTOR ANCHORS" step, which is a no-op for any fiber whose
`fiber_vec` is still unset).

Dry-run by default (report only, touches nothing); pass `--apply` to actually write. `--all`
selects every fiber instead of only those missing `fiber_vec` — combine with `--dry-run`
(default) to see how many a full re-embed would touch, or with `--apply` to actually do it.

``--brain`` is always required — there is no "current brain" default.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from surreal_memory.unified_config import get_shared_storage

_PAGE_SIZE = 500
_BATCH_SIZE = 50
_MAX_NEURONS_PER_FIBER = 8
_MAX_TEXT_LEN = 2000


async def _fiber_page(storage: Any, offset: int, limit: int) -> list[dict[str, Any]]:
    brain_id = storage.brain_id
    rows = await storage._query(
        "SELECT meta::id(id) AS id, summary, neuron_ids, fiber_vec FROM fiber "
        "WHERE brain_id = $bid ORDER BY id LIMIT $limit START $offset",
        bid=brain_id,
        limit=limit,
        offset=offset,
    )
    return list(rows)


async def _neuron_contents(storage: Any, neuron_ids: list[str]) -> dict[str, str]:
    if not neuron_ids:
        return {}
    fetched = await storage.get_neurons_batch(neuron_ids)
    return {nid: n.content for nid, n in fetched.items()}


def _fiber_text(row: dict[str, Any], contents: dict[str, str]) -> str:
    summary = (row.get("summary") or "").strip()
    if summary:
        return summary
    nids = list(row.get("neuron_ids") or [])[:_MAX_NEURONS_PER_FIBER]
    parts = [contents.get(nid, "") for nid in nids]
    return "\n".join(p for p in parts if p.strip())[:_MAX_TEXT_LEN].strip()


async def _run(brain: str, dry_run: bool, all_fibers: bool) -> int:
    storage = await get_shared_storage(brain_name=brain)
    try:
        brain_data = await storage.get_brain(storage.brain_id or "")
        if not brain_data:
            print(f"ERROR: no brain named {brain!r}", file=sys.stderr)
            return 1

        from surreal_memory.engine.semantic_discovery import _create_provider, _effective_embedding

        enabled, provider_name, model_name = _effective_embedding(brain_data.config)
        if not enabled:
            print(
                "ERROR: embeddings are disabled in the effective config — "
                "enable them before backfilling fiber vectors",
                file=sys.stderr,
            )
            return 1

        candidates: list[tuple[str, str]] = []  # (fiber_id, text)
        skipped_no_text = 0
        skipped_has_vector = 0
        offset = 0
        while True:
            page = await _fiber_page(storage, offset, _PAGE_SIZE)
            if not page:
                break
            offset += len(page)
            needs_neurons = [r for r in page if not (r.get("summary") or "").strip()]
            contents: dict[str, str] = {}
            for r in needs_neurons:
                contents.update(
                    await _neuron_contents(
                        storage, list(r.get("neuron_ids") or [])[:_MAX_NEURONS_PER_FIBER]
                    )
                )
            for row in page:
                if row.get("fiber_vec") is not None and not all_fibers:
                    skipped_has_vector += 1
                    continue
                text = _fiber_text(row, contents)
                if not text:
                    skipped_no_text += 1
                    continue
                candidates.append((row["id"], text))
            if len(page) < _PAGE_SIZE:
                break

        print(
            f"fibers to embed: {len(candidates)} "
            f"(skipped: {skipped_has_vector} already vectored, {skipped_no_text} no text)"
        )
        if dry_run:
            print(f"[dry-run] provider={provider_name} model={model_name} — nothing written")
            return 0
        if not candidates:
            print("nothing to embed")
            return 0

        provider = _create_provider(brain_data.config, task_type="RETRIEVAL_DOCUMENT")
        embedded = 0
        failed = 0
        for start in range(0, len(candidates), _BATCH_SIZE):
            chunk = candidates[start : start + _BATCH_SIZE]
            texts = [t for _fid, t in chunk]
            try:
                vectors = await provider.embed_batch(texts)
            except Exception as exc:
                failed += len(chunk)
                print(
                    f"  batch {start}-{start + len(chunk)} embed failed: {exc!r}", file=sys.stderr
                )
                continue
            pairs = [(fid, vec) for (fid, _t), vec in zip(chunk, vectors, strict=True)]
            try:
                await storage.update_fiber_embeddings(pairs)
                embedded += len(pairs)
            except Exception as exc:
                failed += len(pairs)
                print(
                    f"  batch {start}-{start + len(chunk)} write failed: {exc!r}", file=sys.stderr
                )
                continue
            print(f"  embedded {min(start + _BATCH_SIZE, len(candidates))}/{len(candidates)}")

        print(f"done: embedded={embedded} failed={failed}")
        return 0 if failed == 0 else 2
    finally:
        await storage.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brain", required=True, help="Target brain name (no 'current brain' default)")
    ap.add_argument("--apply", action="store_true", help="Compute and write vectors")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Select every fiber, not just those missing fiber_vec (combine with --apply to "
        "re-embed everything; alone, reports how many that would be under --dry-run)",
    )
    args = ap.parse_args()
    return asyncio.run(_run(args.brain, dry_run=not args.apply, all_fibers=args.all))


if __name__ == "__main__":
    sys.exit(main())
