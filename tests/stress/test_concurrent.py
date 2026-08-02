"""Stress tests — concurrent operations on real SQLite."""

from __future__ import annotations

import asyncio

import pytest

from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.mcp.maintenance_handler import MaintenanceHandler
from surreal_memory.storage.memory_store import InMemoryStorage

pytestmark = [pytest.mark.stress, pytest.mark.asyncio]


class TestConcurrentEncodes:
    """Verify multiple concurrent encode() calls don't corrupt data."""

    async def test_10_parallel_encodes(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        contents = [f"Concurrent memory #{i}: unique content about topic {i}" for i in range(10)]

        results = await asyncio.gather(
            *[encoder.encode(content) for content in contents],
            return_exceptions=True,
        )

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent encodes failed: {errors}"

        # All should produce distinct fibers
        fiber_ids = {r.fiber.id for r in results if not isinstance(r, Exception) and r.fiber}
        assert len(fiber_ids) == 10, f"Expected 10 unique fibers, got {len(fiber_ids)}"

        # Stats should reflect all 10
        stats = await sqlite_storage.get_stats(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert stats["fiber_count"] == 10


class TestConcurrentReadWrite:
    """Verify simultaneous encode + recall don't conflict."""

    async def test_encode_and_recall_simultaneously(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        # Pre-populate some data
        for i in range(5):
            await encoder.encode(f"Pre-existing memory #{i}: background data for queries")

        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        pipeline = ReflexPipeline(storage=sqlite_storage, config=brain.config)

        # Simultaneously encode new memory + recall existing
        async def do_encode() -> object:
            return await encoder.encode("New concurrent memory about databases")

        async def do_recall() -> object:
            return await pipeline.query("background data")

        results = await asyncio.gather(do_encode(), do_recall(), return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"Concurrent read/write failed: {errors}"


class TestMaintenanceCounter:
    """Verify maintenance op counter under concurrent access."""

    async def test_counter_increments_correctly(self) -> None:
        """In single-threaded asyncio, counter should be accurate."""

        # MaintenanceHandler uses class-level _op_count.
        # Create a minimal instance to test the counter.
        handler = MaintenanceHandler.__new__(MaintenanceHandler)
        handler._op_count = 0

        # 20 concurrent increment calls — since _increment_op_counter
        # is synchronous (no await), asyncio.gather runs them sequentially.
        async def increment() -> int:
            return handler._increment_op_counter()

        results = await asyncio.gather(*[increment() for _ in range(20)])

        # All increments should be sequential (1, 2, 3, ..., 20)
        assert handler._op_count == 20
        assert sorted(results) == list(range(1, 21))

    async def test_class_level_default_becomes_instance_on_write(self) -> None:
        """Document: _op_count default is class-level, but becomes instance on write.

        Python semantics: `self._op_count += 1` reads class default (0),
        then writes instance attribute (1). Other instances still see class default.
        This means counters are NOT shared — each MCPServer has its own.
        """
        original = MaintenanceHandler._op_count
        MaintenanceHandler._op_count = 0

        try:
            h1 = MaintenanceHandler.__new__(MaintenanceHandler)
            h2 = MaintenanceHandler.__new__(MaintenanceHandler)

            h1._increment_op_counter()
            h1._increment_op_counter()

            # h2 does NOT see h1's increments — each instance has its own counter
            assert h1._op_count == 2
            assert h2._op_count == 0  # Still reads class default
        finally:
            MaintenanceHandler._op_count = original
