#!/usr/bin/env python
"""Measure and clean up the debt left by non-idempotent consolidation.

Before the idempotence gate landed, every consolidation run re-materialised each tag
cluster: a fresh concept neuron, a fresh summary fiber and up to ten RELATED_TO synapses,
each time. This script finds those duplicate groups and can remove all but the oldest
member of each.

Three modes, deliberately ordered by how much they can break:

    --measure   count only (default) — touches nothing
    --dry-run   list exactly what would be deleted — touches nothing
    --apply     delete the duplicates

``--brain`` is always required: there is no "current brain" default, because the one
thing this script must never do is clean a brain nobody asked it to.

IMPORTANT: run ``--apply`` only AFTER the fixed code is released and the smem services
have been restarted. pipx swaps files, not running processes, so a still-running old
smem-mcp / smem-web would recreate the duplicates at its next consolidation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from collections import defaultdict
from typing import Any

from surreal_memory.unified_config import get_shared_storage

SUMMARY_MARKER = "summary_fiber"


def _cluster_key(source_ids: list[str]) -> str:
    """Same identity the engine uses: a hash of the sorted source fiber ids."""
    joined = "|".join(sorted(source_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def _group_key(fiber: Any) -> str | None:
    """Cluster identity of a summary fiber, for old and new rows alike."""
    key = fiber.metadata.get("_cluster_key")
    if key:
        return str(key)
    sources = fiber.metadata.get("source_fibers")
    if isinstance(sources, list) and sources:
        return _cluster_key([str(s) for s in sources])
    return None


async def _load_summary_fibers(storage: Any) -> list[Any]:
    fibers = await storage.get_fibers(limit=100000)
    return [f for f in fibers if f.metadata.get("_consolidation") == SUMMARY_MARKER]


async def _find_orphaned_typed_memories(storage: Any) -> list[str]:
    """typed_memory rows whose fiber no longer exists.

    Record ids are stored with underscores while typed_memory.fiber_id keeps hyphens,
    so the two must be normalised before they can be compared at all.
    """
    rows = await storage._query("SELECT VALUE meta::id(id) FROM fiber")
    alive = {str(r).replace("_", "-") for r in rows}
    tms = await storage._query("SELECT VALUE fiber_id FROM typed_memory")
    return [str(t) for t in tms if str(t) not in alive]


def _duplicate_groups(summaries: list[Any]) -> dict[str, list[Any]]:
    """Summary fibers grouped by cluster identity, keeping only real duplicates."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for fiber in summaries:
        key = _group_key(fiber)
        if key:
            groups[key].append(fiber)
    return {k: v for k, v in groups.items() if len(v) > 1}


def _victims(group: list[Any]) -> list[Any]:
    """Everything except the oldest member — the original stays."""
    ordered = sorted(group, key=lambda f: f.created_at)
    return ordered[1:]


async def run(brain: str, mode: str) -> int:
    storage = await get_shared_storage()
    storage.set_brain(brain)

    summaries = await _load_summary_fibers(storage)
    groups = _duplicate_groups(summaries)
    orphans = await _find_orphaned_typed_memories(storage)

    total_victims = sum(len(_victims(g)) for g in groups.values())
    keyless = [f for f in summaries if _group_key(f) is None]

    print(f"brain:                      {brain}")
    print(f"summary fibers:             {len(summaries)}")
    print(f"duplicate cluster groups:   {len(groups)}")
    print(f"redundant summary fibers:   {total_victims}")
    print(f"summaries without identity: {len(keyless)} (left alone — cannot be grouped safely)")
    print(f"orphaned typed_memory rows: {len(orphans)}")

    if mode == "measure":
        return 0

    if mode == "dry-run":
        for key, group in sorted(groups.items()):
            victims = _victims(group)
            keeper = sorted(group, key=lambda f: f.created_at)[0]
            print(f"\ncluster {key}: {len(group)} copies")
            print(f"  KEEP   {keeper.id}  ({keeper.created_at})")
            for victim in victims:
                print(f"  DELETE {victim.id}  ({victim.created_at})")
        for fiber_id in orphans:
            print(f"DELETE orphaned typed_memory for missing fiber {fiber_id}")
        return 0

    # --apply
    deleted_fibers = 0
    deleted_neurons = 0
    failed = 0
    for group in groups.values():
        for victim in _victims(group):
            anchor_id = victim.anchor_neuron_id
            if await storage.delete_fiber(victim.id):
                deleted_fibers += 1
            else:
                failed += 1
                continue
            # The concept neuron exists only to head this summary, so it goes too.
            if anchor_id:
                try:
                    neuron = await storage.get_neuron(anchor_id)
                    if neuron is not None and neuron.metadata.get("_consolidation") == "summary":
                        await storage.delete_neuron(anchor_id)
                        deleted_neurons += 1
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"  could not remove concept neuron {anchor_id}: {exc}")

    deleted_typed = 0
    for fiber_id in orphans:
        if await storage.delete_typed_memory(fiber_id):
            deleted_typed += 1

    print(f"\ndeleted summary fibers:     {deleted_fibers}")
    print(f"deleted concept neurons:    {deleted_neurons}")
    print(f"deleted orphaned typed_mem: {deleted_typed}")
    if failed:
        print(f"FAILED deletes:             {failed}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", required=True, help="brain id to clean (no default, on purpose)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--measure", action="store_true", help="count only (default)")
    group.add_argument("--dry-run", action="store_true", help="list what would be deleted")
    group.add_argument("--apply", action="store_true", help="actually delete")
    args = parser.parse_args()

    mode = "measure"
    if args.dry_run:
        mode = "dry-run"
    elif args.apply:
        mode = "apply"

    return asyncio.run(run(args.brain, mode))


if __name__ == "__main__":
    sys.exit(main())
