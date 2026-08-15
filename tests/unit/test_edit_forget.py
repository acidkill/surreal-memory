"""Tests for smem_edit and smem_forget MCP tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.mcp.server import MCPServer


def _make_server() -> MCPServer:
    """Create a test server with mocked config."""
    server = MCPServer.__new__(MCPServer)
    server._config = MagicMock()
    server._config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
    server._config.safety = MagicMock(auto_redact_min_severity=3)
    server._config.auto = MagicMock(enabled=False)
    server._config.dedup = MagicMock(enabled=False)
    server._config.tool_tier = MagicMock(tier="full")
    server._storage = None
    server._hooks = None
    server._eternal_trigger_count = 0
    return server


class TestNmemEdit:
    """Tests for the smem_edit handler."""

    @pytest.mark.asyncio
    async def test_edit_missing_memory_id(self) -> None:
        server = _make_server()
        result = await server.call_tool("smem_edit", {})
        assert "error" in result
        assert "memory_id" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_no_changes(self) -> None:
        server = _make_server()
        result = await server.call_tool("smem_edit", {"memory_id": "abc"})
        assert "error" in result
        assert "At least one" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_invalid_type(self) -> None:
        server = _make_server()
        result = await server.call_tool("smem_edit", {"memory_id": "abc", "type": "invalid_type"})
        assert "error" in result
        assert "Invalid memory type" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_content_too_long(self) -> None:
        server = _make_server()
        result = await server.call_tool("smem_edit", {"memory_id": "abc", "content": "x" * 200_000})
        assert "error" in result
        assert "too long" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_memory_not_found(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=None)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("smem_edit", {"memory_id": "nonexistent", "type": "fact"})
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_typed_memory_success(self) -> None:
        from surreal_memory.core.fiber import Fiber
        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.DECISION,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = Fiber.create(
            neuron_ids={"neuron-1"},
            synapse_ids=set(),
            anchor_neuron_id="neuron-1",
            fiber_id="fiber-1",
        )

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_typed_memory = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "smem_edit", {"memory_id": "fiber-1", "type": "fact", "priority": 8}
        )
        assert result["status"] == "edited"
        assert any("type:" in c for c in result["changes"])
        assert any("priority:" in c for c in result["changes"])
        storage.update_typed_memory.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_content_updates_anchor_neuron(self) -> None:
        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
        from surreal_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = MagicMock()
        fiber.anchor_neuron_id = "neuron-1"
        anchor = Neuron.create(type=NeuronType.CONCEPT, content="old content")

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=anchor)
        storage.update_neuron = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "smem_edit", {"memory_id": "fiber-1", "content": "new corrected content"}
        )
        assert result["status"] == "edited"
        assert any("content updated" in c for c in result["changes"])
        storage.update_neuron.assert_awaited_once()


class TestNmemForget:
    """Tests for the smem_forget handler."""

    @pytest.mark.asyncio
    async def test_forget_missing_memory_id(self) -> None:
        server = _make_server()
        result = await server.call_tool("smem_forget", {})
        assert "error" in result
        assert "memory_id" in result["error"]

    @pytest.mark.asyncio
    async def test_forget_memory_not_found(self) -> None:
        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=None)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("smem_forget", {"memory_id": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_soft_delete_success(self) -> None:
        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.TODO,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = MagicMock()

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.update_typed_memory = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool(
            "smem_forget", {"memory_id": "fiber-1", "reason": "completed"}
        )
        assert result["status"] == "soft_deleted"
        storage.update_typed_memory.assert_awaited_once()
        # Verify expires_at was set
        updated_tm = storage.update_typed_memory.call_args[0][0]
        assert updated_tm.expires_at is not None

    @pytest.mark.asyncio
    async def test_hard_delete_success(self) -> None:
        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.TODO,
            priority=Priority.NORMAL,
            source="test",
        )
        fiber = MagicMock()

        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.delete_typed_memory = AsyncMock()
        storage.delete_fiber = AsyncMock()
        storage.batch_save = AsyncMock()
        storage.disable_auto_save = MagicMock()
        storage.enable_auto_save = MagicMock()
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("smem_forget", {"memory_id": "fiber-1", "hard": True})
        assert result["status"] == "hard_deleted"
        storage.delete_typed_memory.assert_awaited_once()
        storage.delete_fiber.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hard_delete_neuron_only(self) -> None:
        """Hard delete on a neuron ID (not fiber) should work."""
        from surreal_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="orphan")
        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.delete_neuron = AsyncMock(return_value=True)
        server.get_storage = AsyncMock(return_value=storage)

        result = await server.call_tool("smem_forget", {"memory_id": neuron.id, "hard": True})
        assert result["status"] == "hard_deleted"
        storage.delete_neuron.assert_awaited_once()


class TestEditRefreshesDerivedFields:
    """A content edit must refresh the fields derived from the content.

    Without this, ``smem_edit`` re-saved the OLD SimHash and the OLD embedding
    vector alongside the NEW text — the memory stayed retrievable by what it
    used to say, and ``reindex --missing-only`` could not repair it because the
    vector field was never empty.
    """

    @pytest.mark.asyncio
    async def test_edit_recomputes_content_hash(self) -> None:
        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.utils.simhash import simhash

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        old = "the office is closed on fridays"
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=old)
        from dataclasses import replace as dc_replace

        neuron = dc_replace(neuron, content_hash=simhash(old))

        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.update_neuron = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        new = "the office is open on fridays again"
        result = await server.call_tool("smem_edit", {"memory_id": neuron.id, "content": new})
        assert result["status"] == "edited"

        saved = storage.update_neuron.await_args.args[0]
        assert saved.content == new
        assert saved.content_hash == simhash(new), (
            "content_hash must be the fingerprint of the NEW content — keeping the old "
            "one feeds near-duplicate detection the SimHash of text that no longer exists"
        )

    @pytest.mark.asyncio
    async def test_edit_reembeds_when_vector_present(self) -> None:
        from unittest.mock import patch

        from surreal_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        stale_vec = [0.1, 0.2, 0.3]
        anchor = Neuron.create(
            type=NeuronType.CONCEPT,
            content="old content",
            metadata={"_embedding": list(stale_vec)},
        )
        fiber = MagicMock()
        fiber.anchor_neuron_id = anchor.id

        from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory

        typed_mem = TypedMemory.create(
            fiber_id="fiber-1",
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            source="test",
        )
        storage.get_typed_memory = AsyncMock(return_value=typed_mem)
        storage.get_fiber = AsyncMock(return_value=fiber)
        storage.get_neuron = AsyncMock(return_value=anchor)
        storage.update_neuron = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        fresh_vec = [9.0, 8.0, 7.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            result = await server.call_tool(
                "smem_edit", {"memory_id": "fiber-1", "content": "corrected content"}
            )

        assert result["status"] == "edited"
        saved = storage.update_neuron.await_args.args[0]
        assert saved.metadata["_embedding"] == fresh_vec, (
            "the vector saved with the new content must describe the new content — "
            "re-saving the old one keeps the memory retrievable by what it used to say"
        )
        provider.embed_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_survives_provider_unavailable_but_warns(self, caplog) -> None:
        import logging
        from unittest.mock import patch

        from surreal_memory.core.neuron import Neuron, NeuronType

        server = _make_server()
        storage = AsyncMock()
        storage.current_brain_id = "brain-1"
        neuron = Neuron.create(
            type=NeuronType.CONCEPT,
            content="old content",
            metadata={"_embedding": [0.1, 0.2]},
        )
        storage.get_typed_memory = AsyncMock(return_value=None)
        storage.get_fiber = AsyncMock(return_value=None)
        storage.get_neuron = AsyncMock(return_value=neuron)
        storage.update_neuron = AsyncMock()
        server.get_storage = AsyncMock(return_value=storage)

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._create_provider",
                side_effect=RuntimeError("no provider"),
            ),
            caplog.at_level(logging.WARNING, logger="surreal_memory.mcp.lifecycle_handler"),
        ):
            result = await server.call_tool(
                "smem_edit", {"memory_id": neuron.id, "content": "new content"}
            )

        assert result["status"] == "edited", "edit must not depend on embedder availability"
        assert any("reindex" in r.message for r in caplog.records), (
            "a stale vector left behind must be reported loudly, with the remediation "
            "command — silence here is indistinguishable from a successful refresh"
        )
