"""Rehearsal-coverage overhead benchmark (RUN-010 section B / U2).

Measures the added cost of raising ``reinforcement_neuron_limit`` (10 -> 25,
see ``BrainConfig.reinforcement_neuron_limit``) on the reinforcement step
every recall runs after activation -- the mechanism this fix touches. Run
manually:

    .venv/bin/python benchmarks/rehearsal_coverage_overhead.py

Uses a real SQLiteStorage (not InMemoryStorage) so the timing reflects actual
get_maturation/save_maturation round trips, since that IS the added cost:
each extra rehearsed fiber is one more read plus one more write. Intentionally
not a CI assertion (microbenchmark timing is too noisy for a hard threshold).
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.lifecycle import ReinforcementManager
from surreal_memory.storage.sqlite_store import SQLiteStorage

_NEURON_COUNT = 30
_REPS = 20


async def _seed(storage: SQLiteStorage) -> list[str]:
    brain = Brain.create(name="bench", brain_id="bench-brain")
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


async def _time_reinforce(storage: SQLiteStorage, neuron_ids: list[str], limit: int) -> float:
    mgr = ReinforcementManager(rehearsal_neuron_limit=limit)
    t0 = time.perf_counter()
    for _ in range(_REPS):
        await mgr.reinforce(storage, neuron_ids)
    return (time.perf_counter() - t0) / _REPS * 1000


async def _main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bench.db"
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        neuron_ids = await _seed(storage)

        try:
            old_ms = await _time_reinforce(storage, neuron_ids, limit=10)
            new_ms = await _time_reinforce(storage, neuron_ids, limit=25)
        finally:
            await storage.close()

        print(f"reinforce() at limit=10 (old default):   {old_ms:8.2f} ms/call")
        print(f"reinforce() at limit=25 (new default):   {new_ms:8.2f} ms/call")
        print(f"delta:                                    {new_ms - old_ms:+8.2f} ms/call")
        print(f"repetitions:                              {_REPS}")
        print(f"neurons seeded (1 fiber each):            {_NEURON_COUNT}")


if __name__ == "__main__":
    asyncio.run(_main())
