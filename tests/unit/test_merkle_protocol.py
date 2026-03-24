"""Tests for Merkle Phase 2 — protocol types, sync engine, and hub endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from neural_memory.sync.merkle import MerkleTreeBuilder
from neural_memory.sync.protocol import (
    ConflictStrategy,
    MerkleBucketDiff,
    MerkleSyncRequest,
    MerkleSyncResponse,
)

# ── Protocol dataclass tests ──────────────────────────────────────


class TestMerkleSyncRequest:
    def test_frozen(self) -> None:
        req = MerkleSyncRequest(
            device_id="abc123",
            brain_id="test",
            root_hash="deadbeef",
            buckets={},
        )
        with pytest.raises(AttributeError):
            req.root_hash = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        req = MerkleSyncRequest(
            device_id="abc123",
            brain_id="test",
            root_hash="deadbeef",
            buckets={"neuron": {"neurons/0a": "hash1"}},
        )
        assert req.strategy == ConflictStrategy.PREFER_RECENT
        assert req.buckets["neuron"]["neurons/0a"] == "hash1"


class TestMerkleSyncResponse:
    def test_in_sync(self) -> None:
        resp = MerkleSyncResponse(status="in_sync", hub_root_hash="abc")
        assert resp.status == "in_sync"
        assert resp.diffs == []
        assert resp.changed_prefixes == []

    def test_diff_with_buckets(self) -> None:
        diff = MerkleBucketDiff(
            entity_type="neuron",
            prefix="neurons/0a",
            entity_ids=["uuid1", "uuid2"],
            entities=[{"id": "uuid1", "content": "hello"}],
        )
        resp = MerkleSyncResponse(
            status="diff",
            changed_prefixes=["neurons/0a"],
            diffs=[diff],
            hub_sequence=42,
        )
        assert len(resp.diffs) == 1
        assert resp.diffs[0].entity_ids == ["uuid1", "uuid2"]
        assert resp.hub_sequence == 42


class TestMerkleBucketDiff:
    def test_frozen(self) -> None:
        diff = MerkleBucketDiff(entity_type="neuron", prefix="neurons/0a")
        with pytest.raises(AttributeError):
            diff.prefix = "changed"  # type: ignore[misc]

    def test_defaults(self) -> None:
        diff = MerkleBucketDiff(entity_type="synapse", prefix="synapses/ff")
        assert diff.entity_ids == []
        assert diff.entities == []


# ── SyncEngine Merkle methods ─────────────────────────────────────


class TestSyncEngineMerkle:
    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.current_brain_id = "test-brain"
        storage.compute_merkle_root = AsyncMock(return_value="roothash")
        storage.get_merkle_tree = AsyncMock(return_value={"neurons": "h1", "neurons/0a": "h2"})
        storage.get_merkle_root = AsyncMock(return_value="combinedroot")
        storage.get_change_log_stats = AsyncMock(return_value={"last_sequence": 10})
        storage.find_neurons = AsyncMock(return_value=[])
        storage.get_synapses = AsyncMock(return_value=[])
        storage.find_fibers = AsyncMock(return_value=[])
        storage.mark_synced = AsyncMock()
        storage.update_device_sync = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_prepare_merkle_request_free_returns_none(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        engine = SyncEngine(mock_storage, "device1")
        result = await engine.prepare_merkle_request("brain1", is_pro=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_prepare_merkle_request_pro(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        engine = SyncEngine(mock_storage, "device1")
        result = await engine.prepare_merkle_request("brain1", is_pro=True)
        assert result is not None
        assert result.device_id == "device1"
        assert result.brain_id == "brain1"
        assert result.root_hash == "combinedroot"
        # compute_merkle_root called for each entity type
        assert mock_storage.compute_merkle_root.call_count == 3

    @pytest.mark.asyncio
    async def test_prepare_merkle_request_no_root_returns_none(
        self, mock_storage: AsyncMock
    ) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        mock_storage.get_merkle_root = AsyncMock(return_value=None)
        engine = SyncEngine(mock_storage, "device1")
        result = await engine.prepare_merkle_request("brain1", is_pro=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_merkle_sync_not_pro(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        engine = SyncEngine(mock_storage, "hub")
        req = MerkleSyncRequest(device_id="dev1", brain_id="b1", root_hash="abc", buckets={})
        resp = await engine.handle_merkle_sync(req, is_pro=False)
        assert resp.status == "error"

    @pytest.mark.asyncio
    async def test_handle_merkle_sync_in_sync(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        engine = SyncEngine(mock_storage, "hub")
        req = MerkleSyncRequest(
            device_id="dev1", brain_id="b1", root_hash="combinedroot", buckets={}
        )
        resp = await engine.handle_merkle_sync(req, is_pro=True)
        assert resp.status == "in_sync"
        assert resp.hub_sequence == 10

    @pytest.mark.asyncio
    async def test_handle_merkle_sync_diff(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        # Make root hashes different so it enters diff mode
        mock_storage.get_merkle_root = AsyncMock(return_value="different_root")
        # Local has neurons/0a with hash "local_h", device sends "remote_h"
        mock_storage.get_merkle_tree = AsyncMock(
            return_value={"neurons": "type_root", "neurons/0a": "local_h"}
        )

        engine = SyncEngine(mock_storage, "hub")
        req = MerkleSyncRequest(
            device_id="dev1",
            brain_id="b1",
            root_hash="device_root",
            buckets={"neuron": {"neurons": "type_root", "neurons/0a": "remote_h"}},
        )
        resp = await engine.handle_merkle_sync(req, is_pro=True)
        assert resp.status == "diff"
        assert "neurons/0a" in resp.changed_prefixes

    @pytest.mark.asyncio
    async def test_process_merkle_response_in_sync(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        engine = SyncEngine(mock_storage, "device1")
        resp = MerkleSyncResponse(status="in_sync")
        result = await engine.process_merkle_response(resp, {})
        assert result["applied"] == 0
        assert result["status"] == "in_sync"

    @pytest.mark.asyncio
    async def test_process_merkle_response_with_inserts(self, mock_storage: AsyncMock) -> None:
        from neural_memory.sync.sync_engine import SyncEngine

        engine = SyncEngine(mock_storage, "device1")
        # Local has no entities in this bucket
        mock_storage.find_neurons = AsyncMock(return_value=[])

        diff = MerkleBucketDiff(
            entity_type="neuron",
            prefix="neurons/0a",
            entity_ids=["0abc-uuid1"],
            entities=[
                {
                    "id": "0abc-uuid1",
                    "type": "concept",
                    "content": "test content",
                    "content_hash": 0,
                    "created_at": "2026-01-01T00:00:00",
                    "metadata": {},
                }
            ],
        )
        resp = MerkleSyncResponse(
            status="diff",
            changed_prefixes=["neurons/0a"],
            diffs=[diff],
            hub_sequence=5,
        )

        with patch.object(engine, "_apply_remote_change", new_callable=AsyncMock):
            result = await engine.process_merkle_response(resp, {})
        assert result["applied"] >= 1
        assert result["hub_sequence"] == 5


# ── Integration: Merkle diff computation ──────────────────────────


class TestMerkleDiffIntegration:
    def test_identical_trees_no_diff(self) -> None:
        entities = [("uuid1", "2026-01-01", "hash1"), ("uuid2", "2026-01-02", "hash2")]
        tree_a = MerkleTreeBuilder.build_tree(entities, "neuron")
        tree_b = MerkleTreeBuilder.build_tree(entities, "neuron")
        diffs = MerkleTreeBuilder.compute_diff(tree_a, tree_b)
        assert diffs == []

    def test_one_entity_changed(self) -> None:
        local = [("uuid1", "2026-01-01", "hash1"), ("uuid2", "2026-01-02", "hash2")]
        remote = [("uuid1", "2026-01-01", "hash1"), ("uuid2", "2026-01-02", "hash_CHANGED")]
        tree_local = MerkleTreeBuilder.build_tree(local, "neuron")
        tree_remote = MerkleTreeBuilder.build_tree(remote, "neuron")
        diffs = MerkleTreeBuilder.compute_diff(tree_local, tree_remote)
        assert len(diffs) >= 1

    def test_full_tree_in_sync(self) -> None:
        neurons = [("n1", "2026-01-01", "h1")]
        synapses = [("s1", "2026-01-01", "h2")]
        fibers = [("f1", "2026-01-01", "h3")]
        tree_a = MerkleTreeBuilder.build_full_tree(neurons, synapses, fibers)
        tree_b = MerkleTreeBuilder.build_full_tree(neurons, synapses, fibers)
        assert MerkleTreeBuilder.compute_diff(tree_a, tree_b) == []
        assert tree_a.hash == tree_b.hash
