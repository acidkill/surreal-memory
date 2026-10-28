"""A bounded validation chunk must not end an invocation with time remaining."""

from __future__ import annotations

import time

import pytest

from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationChunkCompleteError,
    ConsolidationLeaseLostError,
    ConsolidationPausedError,
)
from tests.unit.test_consolidation_mid_strategies_resume import _engine
from tests.unit.test_consolidation_progress import FakeProgressStorage
from tests.unit.test_semantic_discovery_cap_continuation import _semantic_link_fixture
from tests.unit.test_semantic_validation_integration import _FenceSession, _install_fence


@pytest.mark.asyncio
async def test_validation_chunk_continues_when_invocation_has_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, reference_time = _semantic_link_fixture(monkeypatch)
    session = _FenceSession("pending", "verified")
    reads, _ = _install_fence(storage, progress, session)
    engine = _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)
    engine._strategy_deadline = time.perf_counter() + 600
    engine._total_deadline = time.perf_counter() + 3600
    report = ConsolidationReport()

    await engine._run_strategy(ConsolidationStrategy.SEMANTIC_LINK, report, reference_time, False)

    assert len(session.restored) >= 2, "validation chunk ended invocation despite 600s remaining"
    assert "neuron-page" in reads
    assert progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete"
    assert report.semantic_synapses_created == len(storage.added_synapses) == 10
    assert len(set(storage.added_synapses)) == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["budget", "lease", "invalid-proof"])
async def test_next_validation_chunk_obeys_time_lease_and_proof_guards(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    storage, progress, reference_time = _semantic_link_fixture(monkeypatch)
    session = _FenceSession("pending", "invalid" if stop == "invalid-proof" else "verified")
    reads, writes = _install_fence(storage, progress, session)
    progress.brain_id = "brain-1"
    engine = _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)
    engine._strategy_deadline = time.perf_counter() + 600
    engine._total_deadline = time.perf_counter() + 3600

    def stop_after_chunk() -> None:
        if stop == "budget":
            engine._strategy_deadline = time.perf_counter() - 1
        elif stop == "lease":
            progress.lease_lost = True

    session.on_pending = stop_after_chunk
    error = {
        "budget": ConsolidationPausedError,
        "lease": ConsolidationLeaseLostError,
        "invalid-proof": RuntimeError,
    }[stop]
    with pytest.raises(error):
        await engine._run_strategy(
            ConsolidationStrategy.SEMANTIC_LINK, ConsolidationReport(), reference_time, False
        )
    assert reads == []
    assert writes == []
    assert len(session.restored) == (2 if stop == "invalid-proof" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["strategy", "total"])
async def test_public_run_keeps_one_deadline_across_chunks_and_pauses_at_budget(
    monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    from surreal_memory.engine.consolidation import ConsolidationConfig, ConsolidationEngine

    storage = FakeProgressStorage()
    config = (
        ConsolidationConfig()
        if budget == "strategy"
        else ConsolidationConfig(strategy_timeout_seconds=3600, total_timeout_seconds=600)
    )
    engine = ConsolidationEngine(storage, config)
    now = [100.0]
    monkeypatch.setattr("surreal_memory.engine.consolidation.time.perf_counter", lambda: now[0])
    deadlines: list[tuple[float | None, float | None]] = []

    async def chunk(report: ConsolidationReport, dry_run: bool) -> None:
        deadlines.append((engine._strategy_deadline, engine._total_deadline))
        await engine._checkpoint_progress(
            "semantic_link_discovery_batch", cursor=str(len(deadlines))
        )
        now[0] += 200
        raise ConsolidationChunkCompleteError("committed chunk")

    monkeypatch.setattr(engine, "_semantic_link", chunk)
    report = await engine.run([ConsolidationStrategy.SEMANTIC_LINK])

    assert len(deadlines) == 3
    assert len(set(deadlines)) == 1
    assert report.duration_ms == 600_000
    assert report.extra["consolidation_status"] == "paused"
    assert storage.progress is not None
    assert storage.progress["strategy_states"]["semantic_link"]["cursor"] == "3"
    assert storage.progress["completed_strategies"] == []
    assert storage.lease_owner is None
