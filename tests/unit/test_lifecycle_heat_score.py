"""Regression: LIFECYCLE heat score must read access_frequency from NeuronState.

Before the fix, `_lifecycle` read `access_frequency` and `last_accessed_at` from
`neuron.metadata` — neither field ever lives there. Both are columns on the
`neuron_state` table, prefetched the same way the dead-neuron / orphan pass at
`consolidation.py:_prune` already does via `get_all_neuron_states()`. Net
effect was that `access_score` (weight 0.4) and `recency_score` (weight 0.4)
were pinned to zero for every neuron; a neuron recalled a thousand times an
hour aged into COOL/COMPRESSED/ARCHIVED on the same schedule as one nobody
had ever touched.

The test spies on `calculate_heat_score` inside `_lifecycle` and asserts that
the argument `access_count` reflects the value stored on `NeuronState`, not
zero.
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

BRAIN_ID = "heat-score-brain"


@pytest.mark.asyncio
async def test_lifecycle_reads_access_frequency_from_neuron_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = InMemoryStorage()
    brain = Brain.create(name="heat_test", brain_id=BRAIN_ID)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    hot = Neuron.create(type=NeuronType.ENTITY, content="hot", neuron_id="n-hot")
    cold = Neuron.create(type=NeuronType.ENTITY, content="cold", neuron_id="n-cold")
    await storage.add_neuron(hot)
    await storage.add_neuron(cold)

    # The whole point: state is on NeuronState, not neuron.metadata.
    await storage.update_neuron_state(
        NeuronState(neuron_id=hot.id, access_frequency=42, last_activated=utcnow())
    )
    await storage.update_neuron_state(NeuronState(neuron_id=cold.id, access_frequency=0))

    seen: list[dict[str, Any]] = []
    # calculate_heat_score is imported inside `_lifecycle` from
    # surreal_memory.engine.compression — patch there so the spy actually
    # intercepts the call (the same module also owns determine_lifecycle_state,
    # which we do NOT patch — we want the real routing).
    import surreal_memory.engine.compression as _compression

    real_calculate = _compression.calculate_heat_score

    def spy_calculate_heat_score(**kwargs: Any) -> float:
        seen.append(dict(kwargs))
        return float(real_calculate(**kwargs))

    monkeypatch.setattr(_compression, "calculate_heat_score", spy_calculate_heat_score)

    engine = ConsolidationEngine(storage=storage)
    report = ConsolidationReport()
    # dry_run=True avoids invoking update_neuron_lifecycle (InMemoryStorage
    # inherits the base's NotImplementedError for that write).
    await engine._lifecycle(
        report,
        reference_time=utcnow(),
        dry_run=True,
    )

    counts = [call["access_count"] for call in seen]
    assert 42 in counts, (
        "expected access_count=42 to reach calculate_heat_score for the neuron "
        f"whose NeuronState.access_frequency was 42; got {counts}"
    )
    assert 0 in counts, (
        "expected access_count=0 for the neuron whose NeuronState.access_frequency "
        f"was 0 (control); got {counts}"
    )


@pytest.mark.asyncio
async def test_lifecycle_falls_back_to_one_batch_when_the_prefetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must still find the states, in one query rather than per neuron.

    ``get_all_neuron_states`` is the fast path; when it raises, the pass falls
    back to ``get_neuron_states_batch`` over every id it is about to score.
    Two things matter and neither is obvious from reading it: the fallback must
    produce the same answer as the prefetch, and it must ask once rather than
    once per neuron, which is what the cache around it is for.
    """
    storage = InMemoryStorage()
    brain = Brain.create(name="heat_fallback", brain_id="heat-fallback-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    hot = Neuron.create(type=NeuronType.ENTITY, content="hot", neuron_id="f-hot")
    cold = Neuron.create(type=NeuronType.ENTITY, content="cold", neuron_id="f-cold")
    await storage.add_neuron(hot)
    await storage.add_neuron(cold)
    await storage.update_neuron_state(
        NeuronState(neuron_id=hot.id, access_frequency=42, last_activated=utcnow())
    )
    await storage.update_neuron_state(NeuronState(neuron_id=cold.id, access_frequency=0))

    async def _raise() -> Any:
        raise RuntimeError("prefetch unavailable")

    storage.get_all_neuron_states = _raise  # type: ignore[method-assign]

    batch_calls: list[list[str]] = []
    real_batch = storage.get_neuron_states_batch

    async def _spy_batch(ids: list[str]) -> Any:
        batch_calls.append(list(ids))
        return await real_batch(ids)

    storage.get_neuron_states_batch = _spy_batch  # type: ignore[method-assign]

    seen: list[dict[str, Any]] = []
    import surreal_memory.engine.compression as _compression

    real_calculate = _compression.calculate_heat_score

    def spy_calculate_heat_score(**kwargs: Any) -> float:
        seen.append(dict(kwargs))
        return float(real_calculate(**kwargs))

    monkeypatch.setattr(_compression, "calculate_heat_score", spy_calculate_heat_score)

    engine = ConsolidationEngine(storage=storage)
    await engine._lifecycle(ConsolidationReport(), reference_time=utcnow(), dry_run=True)

    assert len(batch_calls) == 1, (
        "the fallback must ask once for the whole pass, not once per neuron; "
        f"got {len(batch_calls)} calls: {batch_calls}"
    )
    assert set(batch_calls[0]) == {hot.id, cold.id}, (
        f"the single batch must cover every neuron being scored; got {batch_calls[0]}"
    )
    counts = [call["access_count"] for call in seen]
    assert 42 in counts, f"the fallback must reach the same state as the prefetch; got {counts}"
    assert 0 in counts, f"expected the control neuron's 0 as well; got {counts}"
