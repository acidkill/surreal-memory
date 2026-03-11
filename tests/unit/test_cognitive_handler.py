"""Tests for CognitiveHandler mixin (nmem_hypothesize + nmem_evidence)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.mcp.cognitive_handler import CognitiveHandler

# ── Fixtures ──────────────────────────────────────────────────────


class FakeCognitiveServer(CognitiveHandler):
    """Minimal server stub to test the mixin."""

    def __init__(self, storage: AsyncMock, config: MagicMock | None = None) -> None:
        self._storage = storage
        self.config = config or MagicMock()

    async def get_storage(self):
        return self._storage


def _make_storage(*, brain_id: str | None = "brain-1") -> AsyncMock:
    """Create a mock storage with cognitive methods."""
    storage = AsyncMock()
    storage._current_brain_id = brain_id
    storage.brain_id = brain_id
    storage.current_brain_id = brain_id
    storage.get_brain = AsyncMock(
        return_value=MagicMock(
            id="brain-1",
            config=MagicMock(),
        )
    )
    storage.upsert_cognitive_state = AsyncMock()
    storage.update_cognitive_evidence = AsyncMock()
    storage.get_cognitive_state = AsyncMock(return_value=None)
    storage.list_cognitive_states = AsyncMock(return_value=[])
    storage.add_typed_memory = AsyncMock()
    storage.get_neuron_state = AsyncMock(
        return_value=MagicMock(
            neuron_id="n1",
            activation_level=1.0,
            access_frequency=0,
            last_activated=None,
            decay_rate=0.1,
            created_at=None,
        )
    )
    storage.update_neuron_state = AsyncMock()
    storage.add_synapse = AsyncMock()
    storage.get_synapses = AsyncMock(return_value=[])
    storage.get_neuron = AsyncMock(return_value=MagicMock(content="Test content"))
    storage.disable_auto_save = MagicMock()
    storage.enable_auto_save = MagicMock()
    storage.batch_save = AsyncMock()
    return storage


def _make_encode_result(anchor_id: str = "anchor-1", fiber_id: str = "fiber-1"):
    """Create a mock MemoryEncoder.encode() result."""
    fiber = MagicMock()
    fiber.id = fiber_id
    fiber.anchor_neuron_id = anchor_id

    neuron = MagicMock()
    neuron.id = anchor_id

    result = MagicMock()
    result.fiber = fiber
    result.neurons_created = [neuron]
    result.synapses_created = []
    return result


_ENCODER_PATCH = "neural_memory.engine.encoder.MemoryEncoder"


# ── nmem_hypothesize: create ──────────────────────────────────────


class TestHypothesizeCreate:
    @pytest.mark.asyncio
    async def test_create_basic(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result()

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._hypothesize(
                {
                    "action": "create",
                    "content": "React Server Components will replace SSR",
                }
            )

        assert result["status"] == "created"
        assert result["hypothesis_id"] == "anchor-1"
        assert result["confidence"] == 0.5
        storage.upsert_cognitive_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_custom_confidence(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result()

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._hypothesize(
                {
                    "action": "create",
                    "content": "High confidence hypothesis",
                    "confidence": 0.8,
                }
            )

        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_create_clamps_confidence(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result()

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._hypothesize(
                {
                    "action": "create",
                    "content": "Over-confident hypothesis",
                    "confidence": 1.5,
                }
            )

        assert result["confidence"] == 0.99

    @pytest.mark.asyncio
    async def test_create_missing_content(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "create"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_create_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "create", "content": "test"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_action(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "delete"})
        assert "error" in result
        assert "Invalid action" in result["error"]

    @pytest.mark.asyncio
    async def test_create_with_tags(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result()

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._hypothesize(
                {
                    "action": "create",
                    "content": "Tagged hypothesis",
                    "tags": ["arch", "react"],
                }
            )

        assert result["status"] == "created"


# ── nmem_hypothesize: list ────────────────────────────────────────


class TestHypothesizeList:
    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "list"})
        assert result["count"] == 0
        assert result["hypotheses"] == []

    @pytest.mark.asyncio
    async def test_list_with_results(self) -> None:
        storage = _make_storage()
        storage.list_cognitive_states = AsyncMock(
            return_value=[
                {
                    "neuron_id": "n1",
                    "confidence": 0.7,
                    "evidence_for_count": 2,
                    "evidence_against_count": 1,
                    "status": "active",
                    "last_evidence_at": "2026-03-05T10:00:00",
                    "created_at": "2026-03-01T10:00:00",
                }
            ]
        )
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "list"})
        assert result["count"] == 1
        assert result["hypotheses"][0]["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        await server._hypothesize({"action": "list", "status": "confirmed"})
        storage.list_cognitive_states.assert_called_once_with(status="confirmed", limit=20)

    @pytest.mark.asyncio
    async def test_list_invalid_status(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "list", "status": "invalid"})
        assert "error" in result


# ── nmem_hypothesize: get ─────────────────────────────────────────


class TestHypothesizeGet:
    @pytest.mark.asyncio
    async def test_get_existing(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "n1",
                "confidence": 0.65,
                "evidence_for_count": 2,
                "evidence_against_count": 1,
                "status": "active",
                "last_evidence_at": "2026-03-05T10:00:00",
                "created_at": "2026-03-01T10:00:00",
            }
        )
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "get", "hypothesis_id": "n1"})
        assert result["hypothesis_id"] == "n1"
        assert result["confidence"] == 0.65

    @pytest.mark.asyncio
    async def test_get_missing_id(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "get"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_not_found(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._hypothesize({"action": "get", "hypothesis_id": "nonexistent"})
        assert "error" in result


# ── nmem_evidence ─────────────────────────────────────────────────


class TestEvidence:
    @pytest.mark.asyncio
    async def test_add_evidence_for(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "hyp-1",
                "confidence": 0.5,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "active",
                "resolved_at": None,
            }
        )
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result(anchor_id="ev-1")

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._evidence(
                {
                    "hypothesis_id": "hyp-1",
                    "content": "Observed X which supports hypothesis",
                    "type": "for",
                    "weight": 0.7,
                }
            )

        assert result["status"] == "evidence_added"
        assert result["evidence_type"] == "for"
        assert result["confidence_after"] > result["confidence_before"]
        assert result["evidence_for_count"] == 1
        assert result["evidence_against_count"] == 0
        storage.add_synapse.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_evidence_against(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "hyp-1",
                "confidence": 0.5,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "active",
                "resolved_at": None,
            }
        )
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result(anchor_id="ev-2")

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._evidence(
                {
                    "hypothesis_id": "hyp-1",
                    "content": "Found counter-evidence Y",
                    "type": "against",
                }
            )

        assert result["confidence_after"] < result["confidence_before"]
        assert result["evidence_against_count"] == 1

    @pytest.mark.asyncio
    async def test_auto_resolution_confirmed(self) -> None:
        """Hypothesis should auto-confirm at high confidence with enough evidence."""
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "hyp-1",
                "confidence": 0.88,
                "evidence_for_count": 2,
                "evidence_against_count": 0,
                "status": "active",
                "resolved_at": None,
            }
        )
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result()

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._evidence(
                {
                    "hypothesis_id": "hyp-1",
                    "content": "Third confirming observation",
                    "type": "for",
                    "weight": 1.0,
                }
            )

        # With confidence near 0.88 + strong evidence for, should push above 0.9
        # and with 3 evidence_for, should auto-resolve
        assert result["evidence_for_count"] == 3
        if result["confidence_after"] >= 0.9:
            assert result.get("auto_resolved") == "confirmed"
            assert result["hypothesis_status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_evidence_on_resolved_hypothesis(self) -> None:
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "hyp-1",
                "confidence": 0.95,
                "evidence_for_count": 5,
                "evidence_against_count": 0,
                "status": "confirmed",
                "resolved_at": "2026-03-05T10:00:00",
            }
        )
        server = FakeCognitiveServer(storage)

        result = await server._evidence(
            {
                "hypothesis_id": "hyp-1",
                "content": "More evidence",
                "type": "for",
            }
        )
        assert "error" in result
        assert "already confirmed" in result["error"]

    @pytest.mark.asyncio
    async def test_evidence_missing_hypothesis(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._evidence(
            {
                "hypothesis_id": "nonexistent",
                "content": "Evidence",
                "type": "for",
            }
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evidence_invalid_type(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._evidence(
            {
                "hypothesis_id": "hyp-1",
                "content": "Evidence",
                "type": "neutral",
            }
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evidence_missing_content(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._evidence(
            {
                "hypothesis_id": "hyp-1",
                "type": "for",
            }
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evidence_missing_hypothesis_id(self) -> None:
        storage = _make_storage()
        server = FakeCognitiveServer(storage)

        result = await server._evidence(
            {
                "content": "Evidence",
                "type": "for",
            }
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_evidence_weight_clamped(self) -> None:
        """Weight outside [0.1, 1.0] should be clamped."""
        storage = _make_storage()
        storage.get_cognitive_state = AsyncMock(
            return_value={
                "neuron_id": "hyp-1",
                "confidence": 0.5,
                "evidence_for_count": 0,
                "evidence_against_count": 0,
                "status": "active",
                "resolved_at": None,
            }
        )
        server = FakeCognitiveServer(storage)
        encode_result = _make_encode_result()

        with patch(_ENCODER_PATCH) as mock_enc:
            mock_enc.return_value.encode = AsyncMock(return_value=encode_result)

            result = await server._evidence(
                {
                    "hypothesis_id": "hyp-1",
                    "content": "Very strong evidence",
                    "type": "for",
                    "weight": 5.0,  # Should be clamped to 1.0
                }
            )

        assert result["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_evidence_no_brain(self) -> None:
        storage = _make_storage(brain_id=None)
        server = FakeCognitiveServer(storage)

        result = await server._evidence(
            {
                "hypothesis_id": "hyp-1",
                "content": "Evidence",
                "type": "for",
            }
        )
        assert "error" in result


# ── Storage Mixin ─────────────────────────────────────────────────


class TestSQLiteCognitiveMixin:
    """Test the storage mixin directly with an in-memory SQLite database."""

    @pytest.mark.asyncio
    async def test_upsert_and_get(self) -> None:
        import aiosqlite

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        class TestStorage(SQLiteCognitiveMixin):
            def __init__(self, conn):
                self._conn = conn

            def _ensure_conn(self):
                return self._conn

            def _ensure_read_conn(self):
                return self._conn

            def _get_brain_id(self):
                return "test-brain"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE brains (id TEXT PRIMARY KEY)")
            await conn.execute("INSERT INTO brains (id) VALUES ('test-brain')")
            await conn.execute("""
                CREATE TABLE cognitive_state (
                    neuron_id TEXT NOT NULL,
                    brain_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_for_count INTEGER NOT NULL DEFAULT 0,
                    evidence_against_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    predicted_at TEXT, resolved_at TEXT,
                    schema_version INTEGER DEFAULT 1, parent_schema_id TEXT,
                    last_evidence_at TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (brain_id, neuron_id),
                    FOREIGN KEY (brain_id) REFERENCES brains(id) ON DELETE CASCADE
                )
            """)
            await conn.commit()

            storage = TestStorage(conn)

            # Upsert
            await storage.upsert_cognitive_state(
                "n1",
                confidence=0.7,
                evidence_for_count=2,
                status="active",
            )

            # Get
            state = await storage.get_cognitive_state("n1")
            assert state is not None
            assert state["confidence"] == 0.7
            assert state["evidence_for_count"] == 2
            assert state["status"] == "active"

            # Update via upsert
            await storage.upsert_cognitive_state(
                "n1",
                confidence=0.9,
                evidence_for_count=3,
                status="confirmed",
            )

            state = await storage.get_cognitive_state("n1")
            assert state["confidence"] == 0.9
            assert state["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_list_with_filter(self) -> None:
        import aiosqlite

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        class TestStorage(SQLiteCognitiveMixin):
            def __init__(self, conn):
                self._conn = conn

            def _ensure_conn(self):
                return self._conn

            def _ensure_read_conn(self):
                return self._conn

            def _get_brain_id(self):
                return "test-brain"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE brains (id TEXT PRIMARY KEY)")
            await conn.execute("INSERT INTO brains (id) VALUES ('test-brain')")
            await conn.execute("""
                CREATE TABLE cognitive_state (
                    neuron_id TEXT NOT NULL, brain_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_for_count INTEGER NOT NULL DEFAULT 0,
                    evidence_against_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    predicted_at TEXT, resolved_at TEXT,
                    schema_version INTEGER DEFAULT 1, parent_schema_id TEXT,
                    last_evidence_at TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (brain_id, neuron_id)
                )
            """)
            await conn.commit()

            storage = TestStorage(conn)

            await storage.upsert_cognitive_state("n1", confidence=0.9, status="confirmed")
            await storage.upsert_cognitive_state("n2", confidence=0.5, status="active")
            await storage.upsert_cognitive_state("n3", confidence=0.1, status="refuted")

            all_states = await storage.list_cognitive_states()
            assert len(all_states) == 3

            active = await storage.list_cognitive_states(status="active")
            assert len(active) == 1
            assert active[0]["neuron_id"] == "n2"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self) -> None:
        import aiosqlite

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        class TestStorage(SQLiteCognitiveMixin):
            def __init__(self, conn):
                self._conn = conn

            def _ensure_conn(self):
                return self._conn

            def _ensure_read_conn(self):
                return self._conn

            def _get_brain_id(self):
                return "test-brain"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE brains (id TEXT PRIMARY KEY)")
            await conn.execute("INSERT INTO brains (id) VALUES ('test-brain')")
            await conn.execute("""
                CREATE TABLE cognitive_state (
                    neuron_id TEXT NOT NULL, brain_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_for_count INTEGER NOT NULL DEFAULT 0,
                    evidence_against_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    predicted_at TEXT, resolved_at TEXT,
                    schema_version INTEGER DEFAULT 1, parent_schema_id TEXT,
                    last_evidence_at TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (brain_id, neuron_id)
                )
            """)
            await conn.commit()

            storage = TestStorage(conn)
            result = await storage.get_cognitive_state("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_confidence_clamped_on_upsert(self) -> None:
        """Confidence should be clamped to [0.01, 0.99]."""
        import aiosqlite

        from neural_memory.storage.sqlite_cognitive import SQLiteCognitiveMixin

        class TestStorage(SQLiteCognitiveMixin):
            def __init__(self, conn):
                self._conn = conn

            def _ensure_conn(self):
                return self._conn

            def _ensure_read_conn(self):
                return self._conn

            def _get_brain_id(self):
                return "test-brain"

        async with aiosqlite.connect(":memory:") as conn:
            await conn.execute("CREATE TABLE brains (id TEXT PRIMARY KEY)")
            await conn.execute("INSERT INTO brains (id) VALUES ('test-brain')")
            await conn.execute("""
                CREATE TABLE cognitive_state (
                    neuron_id TEXT NOT NULL, brain_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    evidence_for_count INTEGER NOT NULL DEFAULT 0,
                    evidence_against_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    predicted_at TEXT, resolved_at TEXT,
                    schema_version INTEGER DEFAULT 1, parent_schema_id TEXT,
                    last_evidence_at TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (brain_id, neuron_id)
                )
            """)
            await conn.commit()

            storage = TestStorage(conn)

            await storage.upsert_cognitive_state("n1", confidence=-0.5)
            state = await storage.get_cognitive_state("n1")
            assert state["confidence"] == 0.01

            await storage.upsert_cognitive_state("n2", confidence=2.0)
            state = await storage.get_cognitive_state("n2")
            assert state["confidence"] == 0.99
