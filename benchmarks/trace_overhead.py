"""Retrieval-trace overhead benchmark (U4).

Measures the cost of building + persisting a RetrievalTrace so the telemetry
path can be shown to be negligible relative to a recall. Run manually:

    .venv/bin/python benchmarks/trace_overhead.py

The automated guarantee that tracing is a *true no-op when disabled* lives in
tests/unit/test_recall_trace.py::test_default_disabled_is_noop (the disabled path
returns before building or persisting anything). This script quantifies the cost
of the ENABLED path for capacity planning; it is intentionally not a CI assertion
(microbenchmark timing is too noisy for a hard threshold).
"""

from __future__ import annotations

import asyncio
import time

from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.engine.trace_builder import build_retrieval_trace
from surreal_memory.storage.memory_store import InMemoryStorage

_N = 5000


def _result() -> RetrievalResult:
    return RetrievalResult(
        answer="a",
        confidence=0.9,
        depth_used=DepthLevel.CONTEXT,
        neurons_activated=5,
        fibers_matched=[f"f{i}" for i in range(8)],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=["a1", "a2"]),
        context="ctx",
        latency_ms=5.0,
        tokens_used=20,
        synthesis_method="single",
    )


async def _main() -> None:
    storage = InMemoryStorage()
    storage.set_brain("bench")
    res = _result()

    # Build-only cost.
    t0 = time.perf_counter()
    for _ in range(_N):
        build_retrieval_trace(res, query="benchmark query", brain_id="bench", mode="associative")
    build_us = (time.perf_counter() - t0) / _N * 1e6

    # Build + persist cost.
    t0 = time.perf_counter()
    for _ in range(_N):
        trace = build_retrieval_trace(
            res, query="benchmark query", brain_id="bench", mode="associative"
        )
        await storage.add_retrieval_trace(trace)
    persist_us = (time.perf_counter() - t0) / _N * 1e6

    print(f"build only:        {build_us:8.2f} us/trace")
    print(f"build + persist:   {persist_us:8.2f} us/trace")
    print(f"iterations:        {_N}")
    print(
        "note: recall itself is typically ms-scale; a few us/trace built fire-and-forget "
        "off the recall path is <1% overhead. Disabled = 0 (early return)."
    )


if __name__ == "__main__":
    asyncio.run(_main())
