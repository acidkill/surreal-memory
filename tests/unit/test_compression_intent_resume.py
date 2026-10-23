"""Crash boundaries of durable per-fiber compression intents."""

from __future__ import annotations

import hashlib
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
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

ORIGINAL = (
    "The release contains detailed migration notes and recovery instructions. "
    "The first check is green. The second check verifies graph integrity. "
    "The third check confirms checkpoints. The fourth check reads the backup. "
    "The fifth check reviews the service state."
)


class _MergeMetadataStorage(InMemoryStorage):
    """Model SurrealDB MERGE retaining metadata keys omitted by an update."""

    async def update_fiber(self, fiber: Fiber) -> None:
        existing = await self.get_fiber(fiber.id)
        if existing is not None:
            fiber = replace(fiber, metadata={**existing.metadata, **fiber.metadata})
        await super().update_fiber(fiber)


async def _store(
    storage_type: type[InMemoryStorage] = InMemoryStorage,
) -> tuple[InMemoryStorage, Fiber, Neuron]:
    store = storage_type()
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
    assert saved.metadata.get("_compression_pending") is None
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


@pytest.mark.asyncio
async def test_completed_intent_is_not_replayed_after_later_neuron_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, fiber, anchor = await _store(_MergeMetadataStorage)
    other = Neuron.create(type=NeuronType.CONCEPT, content="An original secondary neuron.")
    await store.add_neuron(other)
    fiber = replace(fiber, neuron_ids={anchor.id, other.id})
    await store.update_fiber(fiber)

    run_id = "completed-run"
    compressed_content = "The completed prepared summary."
    source_hashes = [
        {
            "id": source.id,
            "hash": hashlib.sha256(source.content.encode("utf-8")).hexdigest(),
        }
        for source in (anchor, other)
    ]
    pending = {
        "version": 1,
        "run_id": run_id,
        "target_tier": int(CompressionTier.EXTRACTIVE),
        "original_tier": int(CompressionTier.FULL),
        "anchor_id": anchor.id,
        "compressed_content": compressed_content,
        "original_token_count": 20,
        "compressed_token_count": 5,
        "entities_preserved": 2,
        "backup_created": True,
        "source_hashes": source_hashes,
    }
    fiber = replace(
        fiber,
        compression_tier=int(CompressionTier.EXTRACTIVE),
        metadata={
            "_compression_pending": pending,
            "_compression_receipt": {
                "run_id": run_id,
                "target_tier": int(CompressionTier.EXTRACTIVE),
                "tokens_saved": 15,
            },
        },
    )
    await store.update_fiber(fiber)
    await store.update_neuron(replace(anchor, content=compressed_content))
    await store.update_neuron(replace(other, content="A later independent edit."))

    result = await CompressionEngine(store).compress_fiber(
        fiber, CompressionTier.EXTRACTIVE, run_id="new-run"
    )

    saved = await store.get_fiber(fiber.id)
    assert result.skipped
    assert saved is not None and saved.metadata["_compression_pending"] is None
    assert saved.metadata["_compression_receipt"]["run_id"] == run_id
    assert (await store.get_neuron(other.id)).content == "A later independent edit."


@pytest.mark.asyncio
async def test_successful_compression_tombstones_pending_for_merge_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    store, fiber, _neuron = await _store(_MergeMetadataStorage)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    result = await CompressionEngine(
        store, CompressionConfig(tier1_max_sentences=1)
    ).compress_fiber(fiber, CompressionTier.EXTRACTIVE, run_id="merge-run")

    saved = await store.get_fiber(fiber.id)
    assert not result.skipped
    assert saved is not None and saved.metadata["_compression_pending"] is None
    assert saved.metadata["_compression_receipt"]["run_id"] == "merge-run"


class _Progress:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {"run_id": "run-1", "strategy_states": {}}
        self.fail = True
        self.writes: list[tuple[str, str | None]] = []

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
        self.writes.append((phase, cursor))


async def _scan_store(count: int, *, varied_skips: bool = True) -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="compression-skip-scan")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    for index in range(count):
        await store.add_fiber(
            Fiber(
                id=f"f-{index:05d}",
                neuron_ids=set(),
                synapse_ids=set(),
                anchor_neuron_id="none",
                compression_tier=int(CompressionTier.GRAPH_ONLY),
                created_at=utcnow() - timedelta(days=15),
                pinned=varied_skips and index % 3 == 0,
                metadata={"_verbatim": varied_skips and index % 3 == 1},
            )
        )
    return store


def _compression_engine(store: InMemoryStorage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(store, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.COMPRESS
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest.mark.asyncio
async def test_compression_skips_checkpoint_once_per_keyset_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _scan_store(501)
    monkeypatch.setattr(
        CompressionEngine,
        "determine_target_tier",
        lambda _self, _fiber, _now, *, heat_score: CompressionTier.GRAPH_ONLY,
    )
    progress = _Progress()
    progress.fail = False

    await _compression_engine(store, progress)._compress(ConsolidationReport(), utcnow(), False)

    assert progress.writes == [
        ("compress_fibers", "f-00249"),
        ("compress_fibers", "f-00499"),
        ("compress_fibers", "f-00500"),
    ]
    assert progress.strategy_state("compress")["counters"]["fibers_compressed"] == 0


@pytest.mark.asyncio
async def test_compression_budget_pause_replays_from_last_page_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = await _scan_store(300, varied_skips=False)
    visited: list[str] = []

    def target(
        _self: CompressionEngine, fiber: Fiber, _now: Any, *, heat_score: float
    ) -> CompressionTier:
        visited.append(fiber.id)
        return CompressionTier.GRAPH_ONLY

    monkeypatch.setattr(CompressionEngine, "determine_target_tier", target)
    progress = _Progress()
    progress.fail = False
    engine = _compression_engine(store, progress)
    check_count = 0
    pause = True
    original_check = engine._check_progress_budget

    async def pause_midpage() -> None:
        nonlocal check_count
        check_count += 1
        if pause and check_count == 275:
            raise ConsolidationPausedError("simulated midpage budget interruption")
        await original_check()

    monkeypatch.setattr(engine, "_check_progress_budget", pause_midpage)
    with pytest.raises(ConsolidationPausedError, match="midpage"):
        await engine._compress(ConsolidationReport(), utcnow(), False)

    assert progress.strategy_state("compress")["cursor"] == "f-00249"
    assert visited[:250] == [f"f-{index:05d}" for index in range(250)]
    pause = False
    before_resume = len(visited)
    await engine._compress(ConsolidationReport(), utcnow(), False)

    assert visited[before_resume:] == [f"f-{index:05d}" for index in range(250, 300)]
    assert progress.strategy_state("compress")["cursor"] == "f-00299"


@pytest.mark.asyncio
async def test_skipped_result_defers_checkpoint_but_pending_and_receipt_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    store = await _scan_store(250)
    # Place three distinct outcomes after an initial span of ordinary no-ops.
    for index, pending, receipt in (
        (240, None, None),
        (241, {"target_tier": int(CompressionTier.EXTRACTIVE)}, None),
        (242, None, None),
        (243, None, {"run_id": "run-1", "target_tier": 0, "tokens_saved": 7}),
    ):
        fiber = await store.get_fiber(f"f-{index:05d}")
        assert fiber is not None
        metadata = dict(fiber.metadata)
        if pending is not None:
            metadata["_compression_pending"] = pending
        if receipt is not None:
            metadata["_compression_receipt"] = receipt
        await store.update_fiber(
            replace(fiber, compression_tier=0, pinned=False, metadata=metadata)
        )
    monkeypatch.setattr(
        CompressionEngine,
        "determine_target_tier",
        lambda _self, _fiber, _now, *, heat_score: CompressionTier.EXTRACTIVE,
    )

    async def compress(_self: Any, fiber: Fiber, _tier: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(skipped=fiber.id in {"f-00241", "f-00242"}, tokens_saved=0)

    # Pending cleanup and the concurrent no-intent skip both return skipped.
    monkeypatch.setattr(CompressionEngine, "compress_fiber", compress)
    progress = _Progress()
    progress.fail = False
    await _compression_engine(store, progress)._compress(ConsolidationReport(), utcnow(), False)

    assert progress.writes == [
        ("compress_fibers", "f-00240"),
        ("compress_fibers", "f-00241"),
        ("compress_fibers", "f-00243"),
        ("compress_fibers", "f-00249"),
    ]


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


async def _seed_legacy_destructive_intent(
    target_tier: CompressionTier,
) -> tuple[InMemoryStorage, Fiber, list[Neuron], dict[str, str], str]:
    store, fiber, anchor = await _store()
    second = Neuron.create(type=NeuronType.CONCEPT, content="A second source neuron.")
    await store.add_neuron(second)
    source_contents = {anchor.id: anchor.content, second.id: second.content}
    fiber = replace(
        fiber,
        neuron_ids=set(source_contents),
        compression_tier=int(CompressionTier.ENTITY_ONLY),
    )
    compressed_content = (
        "[graph-only]" if target_tier == CompressionTier.GRAPH_ONLY else "Prepared template."
    )
    intent = {
        "version": 1,
        "run_id": f"legacy-tier-{int(target_tier)}",
        "target_tier": int(target_tier),
        "original_tier": int(CompressionTier.ENTITY_ONLY),
        "anchor_id": anchor.id,
        "compressed_content": compressed_content,
        "original_token_count": 30,
        "compressed_token_count": 4,
        "entities_preserved": 2,
        "backup_created": False,
        "source_hashes": [
            {"id": neuron_id, "hash": hashlib.sha256(content.encode()).hexdigest()}
            for neuron_id, content in source_contents.items()
        ],
    }
    fiber = replace(fiber, metadata={"_compression_pending": intent})
    await store.update_fiber(fiber)
    for neuron_id, content in source_contents.items():
        await store.save_neuron_snapshot(
            neuron_id=neuron_id,
            brain_id=store.current_brain_id or "",
            original_content=content,
            compressed_at=utcnow().isoformat(),
            tier=int(target_tier),
        )
    neurons = [await store.get_neuron(neuron_id) for neuron_id in source_contents]
    return (
        store,
        fiber,
        [neuron for neuron in neurons if neuron is not None],
        source_contents,
        compressed_content,
    )


@pytest.mark.parametrize(
    ("target_tier", "partially_applied"),
    [
        (CompressionTier.TEMPLATE, False),
        (CompressionTier.TEMPLATE, True),
        (CompressionTier.GRAPH_ONLY, False),
        (CompressionTier.GRAPH_ONLY, True),
    ],
)
@pytest.mark.asyncio
async def test_legacy_destructive_intent_resumes_from_exact_source_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    target_tier: CompressionTier,
    partially_applied: bool,
) -> None:
    from surreal_memory.engine import compression

    (
        store,
        fiber,
        neurons,
        source_contents,
        compressed_content,
    ) = await _seed_legacy_destructive_intent(target_tier)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)

    async def refresh_batch(
        _store: InMemoryStorage, changes: list[tuple[Neuron, str]], *, brain: Brain | None = None
    ) -> list[Neuron]:
        return [replace(neuron, content=content) for neuron, content in changes]

    monkeypatch.setattr(compression, "contents_refreshed", refresh_batch)
    if partially_applied:
        partially_written = neurons[0]
        await store.update_neuron(replace(partially_written, content=compressed_content))

    result = await CompressionEngine(store).compress_fiber(fiber, target_tier)

    assert not result.skipped
    saved_fiber = await store.get_fiber(fiber.id)
    assert saved_fiber is not None and saved_fiber.compression_tier == int(target_tier)
    assert saved_fiber.metadata["_compression_pending"] is None
    receipt = saved_fiber.metadata["_compression_receipt"]
    assert receipt["lineage_version"] == 1
    assert set(receipt["snapshot_hashes"]) == set(source_contents)
    for neuron in neurons:
        saved_neuron = await store.get_neuron(neuron.id)
        assert saved_neuron is not None
        expected = (
            "[graph-only]"
            if target_tier == CompressionTier.GRAPH_ONLY
            else compressed_content
            if neuron.id == fiber.anchor_neuron_id
            else source_contents[neuron.id]
        )
        assert saved_neuron.content == expected


@pytest.mark.parametrize(
    ("target_tier", "snapshot_fault"),
    [
        (CompressionTier.TEMPLATE, "missing"),
        (CompressionTier.TEMPLATE, "changed"),
        (CompressionTier.GRAPH_ONLY, "missing"),
        (CompressionTier.GRAPH_ONLY, "changed"),
    ],
)
@pytest.mark.asyncio
async def test_legacy_destructive_replay_rejects_missing_or_changed_source_snapshot(
    target_tier: CompressionTier, snapshot_fault: str
) -> None:
    (
        store,
        fiber,
        neurons,
        _source_contents,
        _compressed_content,
    ) = await _seed_legacy_destructive_intent(target_tier)
    faulted_neuron = neurons[-1]
    if snapshot_fault == "missing":
        await store.delete_neuron_snapshot(faulted_neuron.id)
    else:
        await store.save_neuron_snapshot(
            neuron_id=faulted_neuron.id,
            brain_id=store.current_brain_id or "",
            original_content="unrelated changed snapshot",
            compressed_at=utcnow().isoformat(),
            tier=int(target_tier),
        )
    snapshots_before = {neuron.id: await store.get_neuron_snapshot(neuron.id) for neuron in neurons}

    with pytest.raises(RuntimeError, match="snapshot"):
        await CompressionEngine(store).compress_fiber(fiber, target_tier)

    saved_fiber = await store.get_fiber(fiber.id)
    assert saved_fiber == fiber
    assert saved_fiber is not None and saved_fiber.metadata["_compression_pending"]
    for neuron in neurons:
        assert await store.get_neuron(neuron.id) == neuron
        assert await store.get_neuron_snapshot(neuron.id) == snapshots_before[neuron.id]


@pytest.mark.asyncio
async def test_legacy_replay_rejects_malformed_source_hash_even_for_applied_output() -> None:
    (
        store,
        fiber,
        neurons,
        _source_contents,
        compressed_content,
    ) = await _seed_legacy_destructive_intent(CompressionTier.TEMPLATE)
    await store.update_neuron(replace(neurons[0], content=compressed_content))
    pending = dict(fiber.metadata["_compression_pending"])
    pending["source_hashes"] = [
        {"id": neurons[0].id, "hash": "not-a-sha256"},
        fiber.metadata["_compression_pending"]["source_hashes"][1],
    ]
    fiber = replace(fiber, metadata={"_compression_pending": pending})
    await store.update_fiber(fiber)
    neuron_contents = {neuron.id: (await store.get_neuron(neuron.id)).content for neuron in neurons}

    with pytest.raises(ValueError, match="Invalid compression intent") as raised:
        await CompressionEngine(store).compress_fiber(fiber, CompressionTier.TEMPLATE)
    assert str(raised.value.__cause__) == "invalid source hash"

    assert await store.get_fiber(fiber.id) == fiber
    for neuron in neurons:
        saved = await store.get_neuron(neuron.id)
        assert saved is not None and saved.content == neuron_contents[neuron.id]


@pytest.mark.asyncio
async def test_present_legacy_snapshot_hash_map_remains_exact_and_complete() -> None:
    (
        store,
        fiber,
        neurons,
        _source_contents,
        _compressed_content,
    ) = await _seed_legacy_destructive_intent(CompressionTier.TEMPLATE)
    pending = dict(fiber.metadata["_compression_pending"])
    pending["snapshot_hashes"] = {
        neurons[0].id: hashlib.sha256(neurons[0].content.encode()).hexdigest()
    }
    fiber = replace(fiber, metadata={"_compression_pending": pending})
    await store.update_fiber(fiber)

    with pytest.raises(RuntimeError, match="snapshots missing"):
        await CompressionEngine(store).compress_fiber(fiber, CompressionTier.TEMPLATE)

    assert await store.get_fiber(fiber.id) == fiber
    for neuron in neurons:
        assert await store.get_neuron(neuron.id) == neuron


def test_boolean_lineage_version_is_not_a_valid_receipt() -> None:
    fiber = Fiber(
        id="boolean-lineage-version",
        neuron_ids={"neuron-a"},
        synapse_ids=set(),
        anchor_neuron_id="neuron-a",
        compression_tier=int(CompressionTier.EXTRACTIVE),
    )

    assert not CompressionEngine._lineage_matches_live_state(
        fiber,
        {"lineage_version": True},
        {"neuron-a": "0" * 64},
        None,
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
