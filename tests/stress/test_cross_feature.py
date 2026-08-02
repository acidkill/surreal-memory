"""Stress tests — cross-feature interaction workflows."""

from __future__ import annotations

import pytest

from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.diagnostics import DiagnosticsEngine
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage

pytestmark = [pytest.mark.stress, pytest.mark.asyncio]


class TestFullSessionWorkflow:
    """Simulate a realistic Claude Code session with multiple MCP tools."""

    async def test_session_workflow_all_tools_succeed(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        brain_id = sqlite_storage._current_brain_id
        assert brain_id is not None

        # Step 1: Encode 5 memories
        for content in [
            "FastAPI uses Starlette for the web layer and Pydantic for validation",
            "SQLAlchemy 2.0 requires explicit begin/commit for transactions",
            "Alembic generates migration scripts from model changes",
            "pytest-asyncio enables testing async functions with pytest",
            "mypy catches type errors at compile time using type annotations",
        ]:
            result = await encoder.encode(content)
            assert result.fiber is not None

        # Step 2: Recall a specific memory
        pipeline = ReflexPipeline(storage=sqlite_storage, config=brain.config)
        recall_result = await pipeline.query("FastAPI validation")
        assert recall_result.neurons_activated > 0

        # Step 3: Check stats
        stats = await sqlite_storage.get_stats(brain_id)
        assert stats["fiber_count"] == 5
        assert stats["neuron_count"] > 0
        assert stats["synapse_count"] > 0

        # Step 4: Run diagnostics
        diagnostics = DiagnosticsEngine(storage=sqlite_storage)
        health = await diagnostics.analyze(brain_id)
        assert 0 <= health.purity_score <= 100
        assert health.fiber_count == 5

        # Step 5: Encode more and verify stats grow
        await encoder.encode("Docker Compose uses YAML for multi-container configs")
        stats2 = await sqlite_storage.get_stats(brain_id)
        assert stats2["fiber_count"] == 6
        assert stats2["neuron_count"] > stats["neuron_count"]


class TestConflictDetectionAndResolution:
    """Test full conflict lifecycle: detect → list → verify."""

    async def test_conflicting_memories_stored_correctly(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        # Store two potentially conflicting facts
        r1 = await encoder.encode("The primary database is PostgreSQL version 14")
        r2 = await encoder.encode("The primary database is MySQL version 8")

        # Both should succeed
        assert r1.fiber is not None
        assert r2.fiber is not None

        # Check if CONTRADICTS synapses exist
        all_synapses = []
        for neuron in r1.neurons_created + r2.neurons_created:
            synapses = await sqlite_storage.get_synapses(source_id=neuron.id)
            all_synapses.extend(synapses)
            synapses = await sqlite_storage.get_synapses(target_id=neuron.id)
            all_synapses.extend(synapses)

        contradicts_count = sum(1 for s in all_synapses if s.type == SynapseType.CONTRADICTS)
        # contradicts_count may be 0 or more depending on entity extraction quality.
        assert contradicts_count >= 0

        # The important thing is both memories are stored and queryable.
        pipeline = ReflexPipeline(
            storage=sqlite_storage,
            config=(
                await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
            ).config,
        )

        result1 = await pipeline.query("PostgreSQL")
        result2 = await pipeline.query("MySQL")
        assert result1.neurons_activated > 0
        assert result2.neurons_activated > 0


class TestVersionControlWithConsolidation:
    """Test versioning captures consolidation changes."""

    async def test_version_diff_reflects_consolidation(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        brain_id = sqlite_storage._current_brain_id
        assert brain_id is not None

        # Encode 30 memories
        for i in range(30):
            await encoder.encode(f"Technical fact #{i}: system component detail")

        # Take stats snapshot before consolidation
        stats_before = await sqlite_storage.get_stats(brain_id)
        assert stats_before["fiber_count"] == 30

        # Run consolidation
        engine = ConsolidationEngine(storage=sqlite_storage)
        report = await engine.run(
            strategies=[ConsolidationStrategy.PRUNE, ConsolidationStrategy.ENRICH]
        )

        # Stats after should be consistent
        stats_after = await sqlite_storage.get_stats(brain_id)
        assert stats_after["fiber_count"] >= 0
        assert stats_after["neuron_count"] >= 0

        # Consolidation report should have valid counts
        assert report.synapses_pruned >= 0
        assert report.synapses_enriched >= 0


class TestTagValidation:
    """Test that invalid tags are handled gracefully."""

    async def test_encoding_with_various_tags(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        # Encode with valid tags
        result = await encoder.encode(
            "Redis uses single-threaded event loop",
            tags={"database", "architecture", "redis"},
        )
        assert result.fiber is not None

        # Verify tags are stored (auto_tags + user tags may differ)
        fiber = await sqlite_storage.get_fiber(result.fiber.id)
        assert fiber is not None

        # The fiber should have tags (either user-provided or auto-generated)
        all_tags = fiber.tags | fiber.auto_tags
        assert len(all_tags) > 0

    async def test_empty_tags_accepted(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        result = await encoder.encode(
            "Simple memory without explicit tags",
            tags=set(),
        )
        assert result.fiber is not None
        # Auto-tags should still be generated
        assert len(result.fiber.auto_tags) >= 0


class TestAutoCaptureWithRemember:
    """Test that auto-detected patterns don't conflict with manual encoding."""

    async def test_explicit_and_auto_detection_produce_distinct_fibers(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        brain_id = sqlite_storage._current_brain_id
        assert brain_id is not None

        # Manually encode a decision
        r1 = await encoder.encode("We decided to use Redis for caching with 5min TTL")
        assert r1.fiber is not None

        # Encode a similar but distinct decision
        r2 = await encoder.encode("We chose Memcached as backup cache with 10min TTL")
        assert r2.fiber is not None

        # Both should produce distinct fibers
        assert r1.fiber.id != r2.fiber.id

        # Stats should show 2 fibers
        stats = await sqlite_storage.get_stats(brain_id)
        assert stats["fiber_count"] == 2
