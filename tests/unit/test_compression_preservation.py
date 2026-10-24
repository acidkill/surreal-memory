"""Compression lineage prevents replacing the first recoverable content."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.compression import CompressionEngine, CompressionTier
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

ORIGINAL = (
    "A detailed synthetic original includes the release plan, validation sequence, "
    "recovery checks, dependency notes, and the exact review procedure."
)


class _CountingStorage(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.backup_writes = 0
        self.snapshot_writes = 0
        self.neuron_reads = 0
        self.synapse_reads = 0
        self.backup_reads = 0

    async def save_compression_backup(self, **kwargs: Any) -> None:
        self.backup_writes += 1
        await super().save_compression_backup(**kwargs)

    async def save_neuron_snapshot(self, **kwargs: Any) -> None:
        self.snapshot_writes += 1
        await super().save_neuron_snapshot(**kwargs)

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        self.neuron_reads += 1
        return await super().get_neurons_batch(neuron_ids)

    async def get_synapses_batch(self, synapse_ids: list[str]) -> dict[str, Any]:
        self.synapse_reads += 1
        return await super().get_synapses_batch(synapse_ids)

    async def get_compression_backup(self, fiber_id: str) -> dict[str, Any] | None:
        self.backup_reads += 1
        return await super().get_compression_backup(fiber_id)


async def _setup() -> tuple[_CountingStorage, Fiber, Neuron]:
    store = _CountingStorage()
    brain = Brain.create(name="compression-preservation-test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=ORIGINAL)
    await store.add_neuron(neuron)
    fiber = Fiber(
        id="preservation-fiber",
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        compression_tier=int(CompressionTier.FULL),
        created_at=utcnow(),
    )
    await store.add_fiber(fiber)
    return store, fiber, neuron


async def _refresh(
    _store: InMemoryStorage, neuron: Neuron, content: str, *, brain: Brain | None = None
) -> Neuron:
    return replace(neuron, content=content)


def _patch_compressors(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)
    monkeypatch.setattr(
        compression,
        "compress_tier1_extractive",
        lambda *_args: ("alpha beta gamma delta epsilon", 1),
    )
    monkeypatch.setattr(
        compression,
        "compress_tier2_entity_preserving",
        lambda *_args: ("alpha beta", 1),
    )
    monkeypatch.setattr(
        compression,
        "compress_tier3_template",
        lambda *_args: ("alpha", 1),
    )


@pytest.mark.asyncio
async def test_tier_advancement_keeps_first_backup_and_restores_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressors(monkeypatch)
    store, fiber, neuron = await _setup()
    engine = CompressionEngine(store)

    for tier in (
        CompressionTier.EXTRACTIVE,
        CompressionTier.ENTITY_ONLY,
        CompressionTier.TEMPLATE,
        CompressionTier.GRAPH_ONLY,
    ):
        result = await engine.compress_fiber(fiber, tier)
        assert not result.skipped

    backup = await store.get_compression_backup(fiber.id)
    assert backup is not None and backup["original_content"] == ORIGINAL
    assert store.backup_writes == 1
    assert store.snapshot_writes == 1
    snapshot = await store.get_neuron_snapshot(neuron.id)
    assert snapshot is not None and snapshot["original_content"] == "alpha beta"

    assert await engine.recover_fiber(fiber.id)
    restored_at_tier_two = await store.get_neuron(neuron.id)
    assert restored_at_tier_two is not None and restored_at_tier_two.content == "alpha beta"
    assert await engine.decompress_fiber(fiber.id)
    restored_original = await store.get_neuron(neuron.id)
    assert restored_original is not None and restored_original.content == ORIGINAL


@pytest.mark.asyncio
@pytest.mark.parametrize("edited_by", ["user", "shared_anchor"])
async def test_unproven_edit_skips_without_overwriting_backup(
    monkeypatch: pytest.MonkeyPatch, edited_by: str
) -> None:
    _patch_compressors(monkeypatch)
    store, fiber, neuron = await _setup()
    engine = CompressionEngine(store)
    assert not (await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)).skipped
    backup_before = await store.get_compression_backup(fiber.id)
    fiber_before = await store.get_fiber(fiber.id)
    assert backup_before is not None and fiber_before is not None

    if edited_by == "shared_anchor":
        from surreal_memory.engine import compression

        shared_fiber = replace(fiber, id="second-shared-fiber", metadata={})
        await store.add_fiber(shared_fiber)
        monkeypatch.setattr(
            compression,
            "compress_tier1_extractive",
            lambda *_args: ("shared anchor changed", 1),
        )
        shared_result = await engine.compress_fiber(shared_fiber, CompressionTier.EXTRACTIVE)
        assert not shared_result.skipped
        changed_content = "shared anchor changed"
    else:
        changed_content = "user changed this shared anchor"
        await store.update_neuron(replace(neuron, content=changed_content))

    result = await engine.compress_fiber(fiber, CompressionTier.ENTITY_ONLY)

    assert result.skipped and result.skip_reason == "preservation_conflict"
    assert store.backup_writes == (2 if edited_by == "shared_anchor" else 1)
    assert await store.get_compression_backup(fiber.id) == backup_before
    saved_fiber = await store.get_fiber(fiber.id)
    assert saved_fiber is not None and saved_fiber.compression_tier == 1
    saved_neuron = await store.get_neuron(neuron.id)
    assert saved_neuron is not None and saved_neuron.content == changed_content


@pytest.mark.asyncio
async def test_legacy_backup_without_lineage_skips_tier_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressors(monkeypatch)
    store, fiber, neuron = await _setup()
    await store.save_compression_backup(
        fiber_id=fiber.id,
        original_content=ORIGINAL,
        compression_tier=1,
        original_token_count=20,
        compressed_token_count=4,
    )
    await store.update_neuron(replace(neuron, content="legacy tier one output"))
    legacy_fiber = replace(fiber, compression_tier=int(CompressionTier.EXTRACTIVE))
    await store.update_fiber(legacy_fiber)
    backup_before = await store.get_compression_backup(fiber.id)

    result = await CompressionEngine(store).compress_fiber(fiber, CompressionTier.ENTITY_ONLY)

    assert result.skipped and result.skip_reason == "preservation_conflict"
    assert store.backup_writes == 1
    assert await store.get_compression_backup(fiber.id) == backup_before
    saved = await store.get_neuron(neuron.id)
    assert saved is not None and saved.content == "legacy tier one output"


@pytest.mark.asyncio
async def test_legacy_advanced_fiber_skips_before_reading_compression_sources() -> None:
    store, fiber, _neuron = await _setup()
    legacy_fiber = replace(fiber, compression_tier=int(CompressionTier.EXTRACTIVE))
    await store.update_fiber(legacy_fiber)

    result = await CompressionEngine(store).compress_fiber(
        legacy_fiber, CompressionTier.ENTITY_ONLY
    )

    assert result.skipped and result.skip_reason == "preservation_conflict"
    assert result.original_tier == result.new_tier == int(CompressionTier.EXTRACTIVE)
    assert result.original_token_count == result.compressed_token_count == 0
    assert not result.backup_created
    assert store.neuron_reads == store.synapse_reads == store.backup_reads == 0
    assert await store.get_fiber(fiber.id) == legacy_fiber
    assert store.backup_writes == store.snapshot_writes == 0


@pytest.mark.asyncio
async def test_backup_write_failure_prevents_neuron_and_intent_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressors(monkeypatch)
    store, fiber, neuron = await _setup()

    async def fail_backup(**_kwargs: Any) -> None:
        raise RuntimeError("synthetic backup failure")

    monkeypatch.setattr(store, "save_compression_backup", fail_backup)
    with pytest.raises(RuntimeError, match="synthetic backup failure"):
        await CompressionEngine(store).compress_fiber(fiber, CompressionTier.EXTRACTIVE)

    saved_neuron = await store.get_neuron(neuron.id)
    saved_fiber = await store.get_fiber(fiber.id)
    assert saved_neuron is not None and saved_neuron.content == ORIGINAL
    assert saved_fiber is not None and "_compression_pending" not in saved_fiber.metadata
    assert await store.get_compression_backup(fiber.id) is None


@pytest.mark.asyncio
async def test_decompress_retry_accepts_only_its_exact_restored_backup_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressors(monkeypatch)
    store, fiber, neuron = await _setup()
    engine = CompressionEngine(store)
    assert not (await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)).skipped
    update_fiber = store.update_fiber

    async def fail_fiber_write(_updated: Fiber) -> None:
        raise RuntimeError("synthetic fiber update failure")

    monkeypatch.setattr(store, "update_fiber", fail_fiber_write)
    assert not await engine.decompress_fiber(fiber.id)
    restored = await store.get_neuron(neuron.id)
    assert restored is not None and restored.content == ORIGINAL
    still_compressed = await store.get_fiber(fiber.id)
    assert still_compressed is not None and still_compressed.compression_tier == 1
    assert still_compressed.metadata.get("_compression_receipt", {}).get("lineage_version") == 1

    monkeypatch.setattr(store, "update_fiber", update_fiber)
    assert await engine.decompress_fiber(fiber.id)
    completed = await store.get_fiber(fiber.id)
    assert completed is not None and completed.compression_tier == 0
    assert completed.metadata.get("_compression_receipt") is None
    assert await store.get_compression_backup(fiber.id) is None


@pytest.mark.asyncio
async def test_destructive_snapshot_conflict_skips_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_compressors(monkeypatch)
    store, fiber, neuron = await _setup()
    await store.save_neuron_snapshot(
        neuron_id=neuron.id,
        brain_id=store.current_brain_id or "",
        original_content="unrelated older snapshot",
        compressed_at=utcnow().isoformat(),
        tier=int(CompressionTier.TEMPLATE),
    )
    before = await store.get_neuron_snapshot(neuron.id)

    result = await CompressionEngine(store).compress_fiber(fiber, CompressionTier.TEMPLATE)

    assert result.skipped and result.skip_reason == "preservation_conflict"
    assert await store.get_neuron(neuron.id) == neuron
    assert await store.get_fiber(fiber.id) == fiber
    assert await store.get_neuron_snapshot(neuron.id) == before
    assert await store.get_compression_backup(fiber.id) is None
