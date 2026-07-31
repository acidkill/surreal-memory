"""Rehearsal-coverage overhead benchmark (RUN-010 section B / U2).

Measures the added cost of raising ``reinforcement_neuron_limit`` (10 -> 15,
see ``BrainConfig.reinforcement_neuron_limit``) on the reinforcement step
every recall runs after activation -- the mechanism this fix touches. Run
manually:

    .venv/bin/python benchmarks/rehearsal_coverage_overhead.py

Uses a real storage backend (not ``InMemoryStorage``) so the timing reflects
actual ``get_maturation``/``save_maturation`` round trips, since that IS the
added cost: each extra rehearsed fiber is one more read plus one more write.

Backend: SurrealDB when ``SURREALDB_URL`` is set (matches this repo's
live-test gating convention), falling back to a temporary SQLite file
otherwise. This matters: SQLite's ``find_fibers`` goes through an indexed
``fiber_neurons`` junction table, while the SurrealDB backend's equivalent
query is an unindexed array-containment scan over ``fiber.neuron_ids`` (no
index on that field, confirmed via an independent database-review pass on
this run) -- an earlier SQLite-only run of this benchmark was measuring a
different, faster query shape than what production actually pays. Prefer the
SurrealDB path whenever a live instance is available.

Intentionally not a CI assertion (microbenchmark timing is too noisy for a
hard threshold).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.lifecycle import ReinforcementManager

_NEURON_COUNT = 30
_REPS = 20


async def _seed(storage: Any, brain_id: str) -> list[str]:
    brain = Brain.create(name=f"bench-{brain_id}", brain_id=brain_id)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    neuron_ids = []
    for i in range(_NEURON_COUNT):
        nid = f"n-{i}"
        await storage.add_neuron(
            Neuron.create(type=NeuronType.ENTITY, content=f"topic-{i}", neuron_id=nid)
        )
        fiber = Fiber(
            id=f"f-{i}",
            neuron_ids={nid},
            synapse_ids=set(),
            anchor_neuron_id=nid,
            pathway=[nid],
        )
        await storage.add_fiber(fiber)
        neuron_ids.append(nid)
    return neuron_ids


async def _time_reinforce(storage: Any, neuron_ids: list[str], limit: int) -> float:
    mgr = ReinforcementManager(rehearsal_neuron_limit=limit)
    t0 = time.perf_counter()
    for _ in range(_REPS):
        await mgr.reinforce(storage, neuron_ids)
    return (time.perf_counter() - t0) / _REPS * 1000


async def _run_against(storage: Any, brain_id: str) -> tuple[float, float]:
    neuron_ids = await _seed(storage, brain_id)
    old_ms = await _time_reinforce(storage, neuron_ids, limit=10)  # old hardcoded default
    new_ms = await _time_reinforce(storage, neuron_ids, limit=15)  # shipped default
    return old_ms, new_ms


async def _main() -> None:
    surrealdb_url = os.environ.get("SURREALDB_URL")
    brain_id = f"bench-rehearsal-{uuid4().hex[:8]}"

    if surrealdb_url:
        backend = "SurrealDB"
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage(url=surrealdb_url)
        await storage.initialize()
        try:
            old_ms, new_ms = await _run_against(storage, brain_id)
        finally:
            # Cleanup discipline (project hard rule): delete only this run's
            # own brain, by its exact id, never `default`. Re-fetch the real
            # RecordID via SELECT rather than reconstructing one from the
            # string id -- a hand-built `type::record('brain', $bid)` looked
            # like it succeeded (no error) but silently matched zero rows,
            # confirmed live: an earlier version of this script left its test
            # brain orphaned this way even though the query "succeeded".
            await storage.clear(brain_id)
            rows = await storage._query(
                "SELECT id FROM brain WHERE id = type::record('brain', $bid)", bid=brain_id
            )
            for row in rows:
                await storage._query("DELETE $rid", rid=row["id"])
            await storage.close()
    else:
        backend = (
            "SQLite (SurrealDB not reachable -- set SURREALDB_URL for the production-accurate path)"
        )
        from surreal_memory.storage.sqlite_store import SQLiteStorage

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "bench.db"
            storage = SQLiteStorage(db_path)
            await storage.initialize()
            try:
                old_ms, new_ms = await _run_against(storage, brain_id)
            finally:
                await storage.close()

    print(f"backend:                                  {backend}")
    print(f"reinforce() at limit=10 (old default):   {old_ms:8.2f} ms/call")
    print(f"reinforce() at limit=15 (new default):   {new_ms:8.2f} ms/call")
    print(f"delta:                                    {new_ms - old_ms:+8.2f} ms/call")
    print(f"repetitions:                              {_REPS}")
    print(f"neurons seeded (1 fiber each):            {_NEURON_COUNT}")
    print(f"timestamp:                                {datetime.now(UTC).isoformat()}")


if __name__ == "__main__":
    asyncio.run(_main())
