"""Crash boundaries of durable per-fiber compression intents."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.compression import CompressionConfig, CompressionEngine, CompressionTier
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

ORIGINAL = (
    "The release contains detailed migration notes and recovery instructions. "
    "The first check is green. The second check verifies graph integrity. "
    "The third check confirms checkpoints. The fourth check reads the backup. "
    "The fifth check reviews the service state."
)


async def _store() -> tuple[InMemoryStorage, Fiber, Neuron]:
    store = InMemoryStorage()
    brain = Brain.create(name="intent-test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=ORIGINAL)
    await store.add_neuron(neuron)
    fiber = Fiber(
        id="f-intent",
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        compression_tier=0,
        created_at=utcnow() - timedelta(days=15),
    )
    await store.add_fiber(fiber)
    return store, fiber, neuron


async def _refresh(
    _store: InMemoryStorage, neuron: Neuron, content: str, *, brain: Brain | None = None
) -> Neuron:
    return replace(neuron, content=content)


@pytest.mark.parametrize(
    ("stage", "after"),
    [
        ("intent", False),
        ("intent", True),
        ("neuron", False),
        ("neuron", True),
        ("final", False),
        ("final", True),
    ],
)
@pytest.mark.asyncio
async def test_replay_after_each_write_boundary(
    monkeypatch: pytest.MonkeyPatch, stage: str, after: bool
) -> None:
    from surreal_memory.engine import compression

    store, fiber, neuron = await _store()
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    update_fiber = store.update_fiber
    update_neuron = store.update_neuron
    fired = False

    async def unreliable_fiber(updated: Fiber) -> None:
        nonlocal fired
        is_final = updated.compression_tier == 1
        should_fail = stage == ("final" if is_final else "intent") and not fired
        if should_fail:
            fired = True
            if after:
                await update_fiber(updated)
            raise RuntimeError("injected fiber write interruption")
        await update_fiber(updated)

    async def unreliable_neuron(updated: Neuron) -> None:
        nonlocal fired
        if stage == "neuron" and not fired:
            fired = True
            if after:
                await update_neuron(updated)
            raise RuntimeError("injected neuron write interruption")
        await update_neuron(updated)

    monkeypatch.setattr(store, "update_fiber", unreliable_fiber)
    monkeypatch.setattr(store, "update_neuron", unreliable_neuron)
    engine = CompressionEngine(store, CompressionConfig(tier1_max_sentences=1))
    with pytest.raises(RuntimeError, match="injected"):
        await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE, run_id="run-1")

    first_fiber = await store.get_fiber(fiber.id)
    assert first_fiber is not None
    if stage == "final" and after:
        assert first_fiber.compression_tier == 1
        assert first_fiber.metadata["_compression_receipt"]["run_id"] == "run-1"
    else:
        assert first_fiber.compression_tier == 0

    result = await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE, run_id="run-1")
    saved = await store.get_fiber(fiber.id)
    assert saved is not None and saved.compression_tier == 1
    assert "_compression_pending" not in saved.metadata
    assert (await store.get_neuron(neuron.id)).content != ORIGINAL
    backup = await store.get_compression_backup(fiber.id)
    assert backup is not None and backup["original_content"] == ORIGINAL
    if stage == "final" and after:
        assert result.skipped
    else:
        assert not result.skipped


@pytest.mark.asyncio
async def test_replay_rejects_concurrent_neuron_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    store, fiber, neuron = await _store()
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)
    write = store.update_neuron

    async def fail_before_write(_updated: Neuron) -> None:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(store, "update_neuron", fail_before_write)
    engine = CompressionEngine(store, CompressionConfig(tier1_max_sentences=1))
    with pytest.raises(RuntimeError, match="interrupted"):
        await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)
    monkeypatch.setattr(store, "update_neuron", write)
    await write(replace(neuron, content="a user changed the neuron"))
    with pytest.raises(RuntimeError, match="changed during compression replay"):
        await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)
    assert (await store.get_neuron(neuron.id)).content == "a user changed the neuron"


class _Progress:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {"run_id": "run-1", "strategy_states": {}}
        self.fail = True

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return self.state["strategy_states"].setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        if self.fail:
            self.fail = False
            raise RuntimeError("checkpoint interrupted")
        self.strategy_state(strategy).update(
            phase=phase, cursor=cursor, pending=pending, counters=dict(counters or {})
        )


@pytest.mark.asyncio
async def test_checkpoint_failure_replays_receipt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    store, fiber, neuron = await _store()
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)
    monkeypatch.setattr(
        CompressionEngine,
        "determine_target_tier",
        lambda _self, _fiber, _now, *, heat_score: CompressionTier.EXTRACTIVE,
    )
    progress = _Progress()
    engine = ConsolidationEngine(store, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.COMPRESS
    engine._progress_session = progress  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="checkpoint interrupted"):
        await engine._compress(ConsolidationReport(), utcnow(), dry_run=False)
    assert (await store.get_fiber(fiber.id)).metadata["_compression_receipt"]["run_id"] == "run-1"
    backup = await store.get_compression_backup(fiber.id)
    assert backup is not None and backup["original_content"] == ORIGINAL
    first_content = (await store.get_neuron(neuron.id)).content

    report = ConsolidationReport()
    await engine._compress(report, utcnow(), dry_run=False)
    assert report.fibers_compressed == 1
    assert progress.strategy_state("compress")["counters"]["fibers_compressed"] == 1
    assert (await store.get_neuron(neuron.id)).content == first_content

    # A new cycle has a distinct run ID; its old receipt cannot be counted again.
    progress.state["run_id"] = "run-2"
    progress.state["strategy_states"] = {}
    report_new = ConsolidationReport()
    await engine._compress(report_new, utcnow(), dry_run=False)
    assert report_new.fibers_compressed == 0


@pytest.mark.asyncio
async def test_graph_only_replay_keeps_all_original_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    store, fiber, first = await _store()
    second = Neuron.create(type=NeuronType.CONCEPT, content="A second original neuron.")
    await store.add_neuron(second)
    fiber = replace(fiber, neuron_ids={first.id, second.id})
    await store.update_fiber(fiber)

    async def refresh_batch(
        _store: InMemoryStorage, changes: list[tuple[Neuron, str]], *, brain: Brain | None = None
    ) -> list[Neuron]:
        return [replace(neuron, content=content) for neuron, content in changes]

    monkeypatch.setattr(compression, "contents_refreshed", refresh_batch)
    original_update = store.update_neuron
    calls = 0

    async def fail_second(updated: Neuron) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second neuron write interrupted")
        await original_update(updated)

    monkeypatch.setattr(store, "update_neuron", fail_second)
    engine = CompressionEngine(store)
    with pytest.raises(RuntimeError, match="second neuron"):
        await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY, run_id="run-1")

    assert (await store.get_fiber(fiber.id)).metadata.get("_compression_pending")
    monkeypatch.setattr(store, "update_neuron", original_update)
    result = await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY, run_id="run-1")
    assert not result.skipped
    assert (await store.get_fiber(fiber.id)).compression_tier == 4
    assert (await store.get_neuron(first.id)).content == "[graph-only]"
    assert (await store.get_neuron(second.id)).content == "[graph-only]"
    assert (await store.get_neuron_snapshot(first.id))["original_content"] == ORIGINAL
    assert (await store.get_neuron_snapshot(second.id))["original_content"] == (
        "A second original neuron."
    )


@pytest.mark.asyncio
async def test_compress_keyset_reaches_beyond_ten_thousand_fibers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStorage()
    brain = Brain.create(name="keyset-over-10k")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    created = utcnow() - timedelta(days=1)
    for index in range(10_050):
        await store.add_fiber(
            Fiber(
                id=f"f-{index:05d}",
                neuron_ids=set(),
                synapse_ids=set(),
                anchor_neuron_id="none",
                created_at=created,
            )
        )

    monkeypatch.setattr(
        CompressionEngine,
        "determine_target_tier",
        lambda _self, _fiber, _now, *, heat_score: CompressionTier.FULL,
    )
    progress = _Progress()
    progress.fail = False
    engine = ConsolidationEngine(store, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.COMPRESS
    engine._progress_session = progress  # type: ignore[assignment]
    await engine._compress(ConsolidationReport(), utcnow(), dry_run=False)

    state = progress.strategy_state("compress")
    assert state["cursor"] == "f-10049"
    assert state["counters"]["fibers_compressed"] == 0
