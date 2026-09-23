"""Legacy SurrealDB compression IDs and fail-closed writes."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.compression import CompressionConfig, CompressionEngine, CompressionTier
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.compression import SurrealDBCompressionMixin
from surreal_memory.utils.timeutils import utcnow

FIBER_ID = "21059641-8b94-4ae1-943f-1cdb2741bff1"
NEURON_ID = "05c557cb-2750-4088-a1a0-d578d8e06ee2"


def _legacy_store(rows: list[dict[str, object]]) -> tuple[SurrealDBCompressionMixin, MagicMock]:
    store = SurrealDBCompressionMixin()
    conn = MagicMock()
    conn.merge = AsyncMock()
    conn.insert = AsyncMock()
    conn.delete = AsyncMock()

    async def query(sql: str, **params: object) -> list[dict[str, object]]:
        table = "compression_backups" if "compression_backups" in sql else "neuron_snapshots"
        field = "fiber_id" if table == "compression_backups" else "neuron_id"
        ids = params["fiber_ids" if field == "fiber_id" else "neuron_ids"]
        assert isinstance(ids, list)
        return [
            {key: value for key, value in row.items() if key != "_table"}
            for row in rows
            if row["_table"] == table
            and row["brain_id"] == params["brain_id"]
            and row[field] in ids
        ][:1]

    store._get_brain_id = MagicMock(return_value="default")  # type: ignore[method-assign]
    store._ensure_conn = MagicMock(return_value=conn)  # type: ignore[method-assign]
    store._query = query  # type: ignore[method-assign]
    return store, conn


@pytest.mark.asyncio
async def test_legacy_fiber_backup_is_found_updated_and_deleted_without_insert() -> None:
    legacy_id = FIBER_ID.replace("-", "_")
    record_id = "compression_backups:default_" + legacy_id
    store, conn = _legacy_store(
        [
            {
                "_table": "compression_backups",
                "id": record_id,
                "brain_id": "default",
                "fiber_id": legacy_id,
                "original_content": "historical original",
                "compression_tier": 1,
                "compressed_at": utcnow().isoformat(),
                "original_token_count": 3,
                "compressed_token_count": 2,
            }
        ]
    )
    old = await store.get_compression_backup(FIBER_ID)
    assert old is not None and old["original_content"] == "historical original"

    await store.save_compression_backup(FIBER_ID, "new original", 2, 3, 1)
    conn.insert.assert_not_awaited()
    conn.merge.assert_awaited_once()
    assert conn.merge.await_args.args[0] == record_id
    assert conn.merge.await_args.args[1]["fiber_id"] == FIBER_ID
    assert await store.delete_compression_backup(FIBER_ID)
    conn.delete.assert_awaited_once_with(record_id)


@pytest.mark.asyncio
async def test_legacy_neuron_snapshot_is_found_updated_and_deleted_without_insert() -> None:
    legacy_id = NEURON_ID.replace("-", "_")
    record_id = "neuron_snapshots:default_" + legacy_id
    store, conn = _legacy_store(
        [
            {
                "_table": "neuron_snapshots",
                "id": record_id,
                "brain_id": "default",
                "neuron_id": legacy_id,
                "original_content": "historical original",
                "compressed_at": utcnow().isoformat(),
                "tier": 3,
            }
        ]
    )
    old = await store.get_neuron_snapshot(NEURON_ID)
    assert old is not None and old["original_content"] == "historical original"

    await store.save_neuron_snapshot(NEURON_ID, "default", "new original", utcnow().isoformat(), 4)
    conn.insert.assert_not_awaited()
    conn.merge.assert_awaited_once()
    assert conn.merge.await_args.args[0] == record_id
    assert conn.merge.await_args.args[1]["neuron_id"] == NEURON_ID
    assert await store.delete_neuron_snapshot(NEURON_ID)
    conn.delete.assert_awaited_once_with(record_id)


async def _seed_engine() -> tuple[InMemoryStorage, Fiber, Neuron]:
    store = InMemoryStorage()
    brain = Brain.create(name="compression-failure-test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="Original complete sentence with many details. Short summary. Another detail.",
    )
    await store.add_neuron(neuron)
    fiber = Fiber(
        id=FIBER_ID,
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        compression_tier=0,
        created_at=utcnow() - timedelta(days=30),
    )
    await store.add_fiber(fiber)
    return store, fiber, neuron


@pytest.mark.asyncio
async def test_backup_failure_prevents_reversible_compression() -> None:
    store, fiber, neuron = await _seed_engine()
    store.save_compression_backup = AsyncMock(side_effect=RuntimeError("backup refused"))  # type: ignore[method-assign]
    engine = CompressionEngine(store, CompressionConfig(tier1_max_sentences=1))
    with pytest.raises(RuntimeError, match="backup refused"):
        await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)
    assert (await store.get_neuron(neuron.id)).content == neuron.content
    assert (await store.get_fiber(fiber.id)).compression_tier == 0


@pytest.mark.asyncio
async def test_snapshot_failure_prevents_destructive_compression() -> None:
    store, fiber, neuron = await _seed_engine()
    store.save_neuron_snapshot = AsyncMock(side_effect=RuntimeError("snapshot refused"))  # type: ignore[method-assign]
    engine = CompressionEngine(store)
    with patch(
        "surreal_memory.engine.compression.compress_tier3_template", return_value=("short", 1)
    ):
        with pytest.raises(RuntimeError, match="snapshot refused"):
            await engine.compress_fiber(fiber, CompressionTier.TEMPLATE)
    assert (await store.get_neuron(neuron.id)).content == neuron.content
    assert (await store.get_fiber(fiber.id)).compression_tier == 0
