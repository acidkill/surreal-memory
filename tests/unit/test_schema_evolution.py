"""Tests for cognitive handler Phase 5: schema evolution (nmem_schema)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.mcp.cognitive_handler import CognitiveHandler

_ENCODER_PATCH = "neural_memory.engine.encoder.MemoryEncoder"


def _make_storage(brain_id: str | None = "test-brain") -> AsyncMock:
    """Build a mock NeuralStorage with explicit brain_id."""
    storage = AsyncMock()
    storage.current_brain_id = brain_id
    storage.list_cognitive_states = AsyncMock(return_value=[])
    storage.list_predictions = AsyncMock(return_value=[])
    storage.get_hot_index = AsyncMock(return_value=[])
    storage.list_knowledge_gaps = AsyncMock(return_value=[])
    storage.get_calibration_stats = AsyncMock(
        return_value={"correct_count": 0, "wrong_count": 0, "total_resolved": 0, "pending_count": 0}
    )
    return storage


def _make_handler(storage: AsyncMock) -> CognitiveHandler:
    handler = CognitiveHandler.__new__(CognitiveHandler)
    handler.config = MagicMock(
        encryption=MagicMock(enabled=False, auto_encrypt_sensitive=False),
        safety=MagicMock(auto_redact_min_severity=3),
    )
    handler.get_storage = AsyncMock(return_value=storage)
    return handler


# ──────────────────── nmem_schema evolve tests ────────────────────


class TestSchemaEvolve:
    """Tests for _schema action=evolve."""

    @pytest.mark.asyncio
    async def test_evolve_no_hypothesis_id(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._schema({"action": "evolve"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evolve_no_content(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._schema({"action": "evolve", "hypothesis_id": "h-1"})
        assert "error" in result
        assert "content" in result["error"]

    @pytest.mark.asyncio
    async def test_evolve_hypothesis_not_found(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(return_value=None)
        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "evolve", "hypothesis_id": "h-1", "content": "Updated hypothesis"}
        )
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_evolve_prediction_rejected(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "h-1",
                "confidence": 0.5,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "pending",
                "predicted_at": "2026-04-01",
                "schema_version": 1,
                "parent_schema_id": None,
            }
        )
        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "evolve", "hypothesis_id": "h-1", "content": "Updated"}
        )
        assert "error" in result
        assert "prediction" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_evolve_already_superseded(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "h-1",
                "confidence": 0.5,
                "evidence_for_count": 1,
                "evidence_against_count": 0,
                "status": "superseded",
                "predicted_at": None,
                "schema_version": 1,
                "parent_schema_id": None,
            }
        )
        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "evolve", "hypothesis_id": "h-1", "content": "Updated"}
        )
        assert "error" in result
        assert "superseded" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_evolve_success(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "h-1",
                "confidence": 0.6,
                "evidence_for_count": 2,
                "evidence_against_count": 1,
                "status": "active",
                "predicted_at": None,
                "schema_version": 1,
                "parent_schema_id": None,
            }
        )

        @dataclass(frozen=True)
        class FakeNeuron:
            id: str
            content: str

        @dataclass(frozen=True)
        class FakeFiber:
            id: str = "fiber-new"
            anchor_neuron_id: str = "h-2"

        @dataclass
        class FakeResult:
            fiber: FakeFiber
            neurons_created: list[Any]

        fake_result = FakeResult(fiber=FakeFiber(), neurons_created=[])
        storage.get_brain = AsyncMock(return_value=MagicMock(config={}))
        storage.get_neuron_state = AsyncMock(return_value=None)

        mock_encoder_cls = MagicMock()
        mock_encoder_cls.return_value.encode = AsyncMock(return_value=fake_result)

        handler = _make_handler(storage)

        import unittest.mock as um

        with um.patch(_ENCODER_PATCH, mock_encoder_cls):
            result = await handler._schema(
                {
                    "action": "evolve",
                    "hypothesis_id": "h-1",
                    "content": "Updated hypothesis content",
                    "reason": "New evidence suggests different mechanism",
                }
            )

        assert result["status"] == "evolved"
        assert result["new_hypothesis_id"] == "h-2"
        assert result["old_hypothesis_id"] == "h-1"
        assert result["schema_version"] == 2
        assert result["confidence"] == 0.6
        assert result["reason"] == "New evidence suggests different mechanism"

        # Verify old hypothesis marked as superseded
        storage.update_cognitive_evidence.assert_called_once()
        assert storage.update_cognitive_evidence.call_args.kwargs["status"] == "superseded"

        # Verify SUPERSEDES synapse created
        storage.add_synapse.assert_called_once()

    @pytest.mark.asyncio
    async def test_evolve_custom_confidence(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "h-1",
                "confidence": 0.6,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "active",
                "predicted_at": None,
                "schema_version": 1,
                "parent_schema_id": None,
            }
        )

        @dataclass(frozen=True)
        class FakeFiber:
            id: str = "fiber-new"
            anchor_neuron_id: str = "h-2"

        @dataclass
        class FakeResult:
            fiber: FakeFiber
            neurons_created: list[Any]

        fake_result = FakeResult(fiber=FakeFiber(), neurons_created=[])
        storage.get_brain = AsyncMock(return_value=MagicMock(config={}))
        storage.get_neuron_state = AsyncMock(return_value=None)

        mock_encoder_cls = MagicMock()
        mock_encoder_cls.return_value.encode = AsyncMock(return_value=fake_result)

        handler = _make_handler(storage)

        import unittest.mock as um

        with um.patch(_ENCODER_PATCH, mock_encoder_cls):
            result = await handler._schema(
                {
                    "action": "evolve",
                    "hypothesis_id": "h-1",
                    "content": "Updated",
                    "confidence": 0.8,
                }
            )

        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_evolve_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "evolve", "hypothesis_id": "h-1", "content": "Updated"}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evolve_content_too_long(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "h-1",
                "confidence": 0.5,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "active",
                "predicted_at": None,
                "schema_version": 1,
                "parent_schema_id": None,
            }
        )
        handler = _make_handler(storage)
        result = await handler._schema(
            {
                "action": "evolve",
                "hypothesis_id": "h-1",
                "content": "x" * 100_001,
            }
        )
        assert "error" in result
        assert "too long" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_action(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._schema({"action": "delete", "hypothesis_id": "h-1"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evolve_storage_failure(self) -> None:
        """Verify error returned and enable_auto_save called on storage failure."""
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "h-1",
                "confidence": 0.6,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "active",
                "predicted_at": None,
                "schema_version": 1,
                "parent_schema_id": None,
            }
        )
        storage.get_brain = AsyncMock(return_value=MagicMock(config={}))
        # Fail inside the try block (after disable_auto_save)
        storage.add_typed_memory = AsyncMock(side_effect=RuntimeError("DB error"))

        mock_encoder_cls = MagicMock()

        @dataclass(frozen=True)
        class FakeFiber:
            id: str = "fiber-new"
            anchor_neuron_id: str = "h-2"

        @dataclass
        class FakeResult:
            fiber: FakeFiber
            neurons_created: list[Any]

        mock_encoder_cls.return_value.encode = AsyncMock(
            return_value=FakeResult(fiber=FakeFiber(), neurons_created=[])
        )

        handler = _make_handler(storage)

        import unittest.mock as um

        with um.patch(_ENCODER_PATCH, mock_encoder_cls):
            result = await handler._schema(
                {"action": "evolve", "hypothesis_id": "h-1", "content": "Updated"}
            )
        assert "error" in result
        storage.enable_auto_save.assert_called_once()


# ──────────────────── nmem_schema history tests ────────────────────


class TestSchemaHistory:
    """Tests for _schema action=history."""

    @pytest.mark.asyncio
    async def test_history_no_id(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._schema({"action": "history"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_history_not_found(self) -> None:
        storage = _make_storage()
        storage.get_schema_history = AsyncMock(return_value=[])
        handler = _make_handler(storage)
        result = await handler._schema({"action": "history", "hypothesis_id": "h-1"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_history_success(self) -> None:
        storage = _make_storage()
        storage.get_schema_history = AsyncMock(
            return_value=[
                {
                    "neuron_id": "h-2",
                    "confidence": 0.8,
                    "status": "active",
                    "schema_version": 2,
                    "parent_schema_id": "h-1",
                    "created_at": "2026-03-06",
                },
                {
                    "neuron_id": "h-1",
                    "confidence": 0.5,
                    "status": "superseded",
                    "schema_version": 1,
                    "parent_schema_id": None,
                    "created_at": "2026-03-05",
                },
            ]
        )

        @dataclass(frozen=True)
        class FakeNeuron:
            id: str
            content: str

        storage.get_neuron = AsyncMock(
            side_effect=[
                FakeNeuron(id="h-2", content="Updated hypothesis about X"),
                FakeNeuron(id="h-1", content="Original hypothesis about X"),
            ]
        )

        handler = _make_handler(storage)
        result = await handler._schema({"action": "history", "hypothesis_id": "h-2"})

        assert result["version_count"] == 2
        assert result["versions"][0]["version"] == 2
        assert result["versions"][1]["version"] == 1

    @pytest.mark.asyncio
    async def test_history_storage_failure(self) -> None:
        storage = _make_storage()
        storage.get_schema_history = AsyncMock(side_effect=RuntimeError("DB error"))
        handler = _make_handler(storage)
        result = await handler._schema({"action": "history", "hypothesis_id": "h-1"})
        assert "error" in result


# ──────────────────── nmem_schema compare tests ────────────────────


class TestSchemaCompare:
    """Tests for _schema action=compare."""

    @pytest.mark.asyncio
    async def test_compare_missing_ids(self) -> None:
        storage = _make_storage()
        handler = _make_handler(storage)
        result = await handler._schema({"action": "compare", "hypothesis_id": "h-1"})
        assert "error" in result
        assert "other_id" in result["error"]

    @pytest.mark.asyncio
    async def test_compare_not_found(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(return_value=None)
        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "compare", "hypothesis_id": "h-1", "other_id": "h-2"}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_compare_success(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            side_effect=[
                {
                    "neuron_id": "h-1",
                    "confidence": 0.5,
                    "status": "superseded",
                    "evidence_for_count": 1,
                    "evidence_against_count": 2,
                    "schema_version": 1,
                },
                {
                    "neuron_id": "h-2",
                    "confidence": 0.8,
                    "status": "active",
                    "evidence_for_count": 3,
                    "evidence_against_count": 0,
                    "schema_version": 2,
                },
            ]
        )

        @dataclass(frozen=True)
        class FakeNeuron:
            id: str
            content: str

        storage.get_neuron = AsyncMock(
            side_effect=[
                FakeNeuron(id="h-1", content="Old hypothesis"),
                FakeNeuron(id="h-2", content="New hypothesis"),
            ]
        )

        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "compare", "hypothesis_id": "h-1", "other_id": "h-2"}
        )

        assert result["version_a"]["version"] == 1
        assert result["version_b"]["version"] == 2
        assert result["confidence_delta"] == 0.3

    @pytest.mark.asyncio
    async def test_compare_storage_failure(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(side_effect=RuntimeError("DB error"))
        handler = _make_handler(storage)
        result = await handler._schema(
            {"action": "compare", "hypothesis_id": "h-1", "other_id": "h-2"}
        )
        assert "error" in result


# ──────────────────── SQLite storage tests ────────────────────


class TestSQLiteSchemaHistory:
    """Integration tests for get_schema_history."""

    @pytest.mark.asyncio
    async def test_history_chain(self, tmp_path: Any) -> None:
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create a chain: h1 → h2 → h3 (newest)
        await storage.upsert_cognitive_state(
            "h-1",
            confidence=0.5,
            status="superseded",
            schema_version=1,
        )
        await storage.upsert_cognitive_state(
            "h-2",
            confidence=0.6,
            status="superseded",
            schema_version=2,
            parent_schema_id="h-1",
        )
        await storage.upsert_cognitive_state(
            "h-3",
            confidence=0.8,
            status="active",
            schema_version=3,
            parent_schema_id="h-2",
        )

        history = await storage.get_schema_history("h-3")
        assert len(history) == 3
        assert history[0]["neuron_id"] == "h-3"
        assert history[1]["neuron_id"] == "h-2"
        assert history[2]["neuron_id"] == "h-1"

        await storage.close()

    @pytest.mark.asyncio
    async def test_history_single(self, tmp_path: Any) -> None:
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        await storage.upsert_cognitive_state("h-1", confidence=0.5, status="active")

        history = await storage.get_schema_history("h-1")
        assert len(history) == 1
        assert history[0]["neuron_id"] == "h-1"

        await storage.close()

    @pytest.mark.asyncio
    async def test_history_not_found(self, tmp_path: Any) -> None:
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        history = await storage.get_schema_history("nonexistent")
        assert history == []

        await storage.close()

    @pytest.mark.asyncio
    async def test_history_cycle_guard(self, tmp_path: Any) -> None:
        """Cycle in parent_schema_id chain should not loop forever."""
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create a cycle: h-1 → h-2 → h-1
        await storage.upsert_cognitive_state(
            "h-1",
            confidence=0.5,
            status="superseded",
            parent_schema_id="h-2",
        )
        await storage.upsert_cognitive_state(
            "h-2",
            confidence=0.6,
            status="active",
            parent_schema_id="h-1",
        )

        history = await storage.get_schema_history("h-2")
        assert len(history) == 2  # Stops at cycle, doesn't loop

        await storage.close()

    @pytest.mark.asyncio
    async def test_history_max_depth(self, tmp_path: Any) -> None:
        """Chain deeper than max_depth should be capped."""
        from neural_memory.core.brain import Brain
        from neural_memory.storage.sqlite_store import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create a 5-deep chain, request max_depth=3
        for i in range(5):
            parent = f"h-{i}" if i > 0 else None
            await storage.upsert_cognitive_state(
                f"h-{i + 1}",
                confidence=0.5,
                status="active",
                schema_version=i + 1,
                parent_schema_id=parent,
            )

        history = await storage.get_schema_history("h-5", max_depth=3)
        assert len(history) == 3

        await storage.close()
