"""Stress tests — scale testing with 100+ memories in real SQLite."""

from __future__ import annotations

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.diagnostics import DiagnosticsEngine
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage

pytestmark = [pytest.mark.stress, pytest.mark.asyncio]


def _generate_memories(n: int) -> list[str]:
    """Generate N unique, diverse memory content strings."""
    topics = [
        "Python",
        "JavaScript",
        "Rust",
        "Go",
        "TypeScript",
        "PostgreSQL",
        "Redis",
        "MongoDB",
        "SQLite",
        "MySQL",
        "Docker",
        "Kubernetes",
        "Terraform",
        "Ansible",
        "Jenkins",
        "React",
        "Vue",
        "Angular",
        "Svelte",
        "Next.js",
        "FastAPI",
        "Django",
        "Flask",
        "Express",
        "Spring",
        "JWT",
        "OAuth",
        "CORS",
        "HTTPS",
        "WebSocket",
    ]
    actions = [
        "supports",
        "uses",
        "implements",
        "requires",
        "provides",
        "enables",
        "handles",
        "manages",
        "processes",
        "validates",
    ]
    features = [
        "concurrent request handling",
        "type-safe data validation",
        "automatic schema generation",
        "efficient memory management",
        "distributed caching layers",
        "real-time event streaming",
        "structured error handling",
        "automated test discovery",
        "incremental compilation",
        "hot module replacement",
        "connection pooling",
        "query optimization",
        "load balancing",
        "rate limiting",
        "health monitoring",
    ]
    memories = []
    for i in range(n):
        topic = topics[i % len(topics)]
        action = actions[i % len(actions)]
        feature = features[i % len(features)]
        memories.append(f"{topic} {action} {feature} (fact #{i})")
    return memories


class TestBulkEncode:
    """Verify encoding 150 memories succeeds without errors."""

    async def test_150_memories_no_errors(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        memories = _generate_memories(150)

        fiber_ids: set[str] = set()
        for content in memories:
            result = await encoder.encode(content)
            assert result.fiber is not None, f"Failed to encode: {content[:50]}"
            fiber_ids.add(result.fiber.id)

        # All fiber IDs should be unique
        assert len(fiber_ids) == 150, f"Expected 150 unique fibers, got {len(fiber_ids)}"

        # Verify stats match
        stats = await sqlite_storage.get_stats(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert stats["fiber_count"] == 150


class TestRecallPrecisionAtScale:
    """After 150 memories, verify targeted queries find correct content."""

    async def test_queries_find_targets(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        memories = _generate_memories(150)

        for content in memories:
            await encoder.encode(content)

        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        pipeline = ReflexPipeline(storage=sqlite_storage, config=brain.config)

        # Pick 10 specific memories and query for them
        test_indices = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
        found = 0

        for idx in test_indices:
            target = memories[idx]
            # Extract the topic keyword (first word)
            keyword = target.split()[0]
            result = await pipeline.query(keyword)

            if result.context and keyword.lower() in result.context.lower():
                found += 1

        # At least 5/10 should be found — activation in large graph is harder
        assert found >= 5, f"Only {found}/10 queries found their target"


class TestHealthAtScale:
    """Run diagnostics on a 150-memory brain."""

    async def test_health_metrics_sane(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        memories = _generate_memories(150)
        for content in memories:
            await encoder.encode(content)

        engine = DiagnosticsEngine(storage=sqlite_storage)
        brain_id = sqlite_storage._current_brain_id
        assert brain_id is not None

        report = await engine.analyze(brain_id)

        # Purity score should be a valid number
        assert 0 <= report.purity_score <= 100, f"Purity {report.purity_score} out of range"

        # Grade should be valid
        assert report.grade in ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F")

        # Component scores in [0, 1]
        for name in ("connectivity", "diversity", "freshness", "orphan_rate"):
            score = getattr(report, name)
            assert 0 <= score <= 1.0, f"{name}={score} out of [0,1]"

        # Counts should match stats
        stats = await sqlite_storage.get_stats(brain_id)
        assert report.fiber_count == stats["fiber_count"]

        # Freshly encoded brain should not have CRITICAL warnings
        critical = [w for w in report.warnings if w.severity.name == "CRITICAL"]
        assert len(critical) == 0, f"Unexpected CRITICAL warnings: {critical}"


class TestConsolidationAtScale:
    """Run full consolidation on a 200-memory brain."""

    async def test_consolidation_completes(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        memories = _generate_memories(200)
        for content in memories:
            await encoder.encode(content)

        engine = ConsolidationEngine(storage=sqlite_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.ALL])

        # Should complete within timeout (60s)
        assert report.duration_ms < 60_000, f"Consolidation took {report.duration_ms}ms"

        # Report counts should be non-negative
        assert report.synapses_pruned >= 0
        assert report.neurons_pruned >= 0
        assert report.fibers_merged >= 0
        assert report.synapses_enriched >= 0

        # Brain should still have most fibers
        stats = await sqlite_storage.get_stats(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert stats["fiber_count"] >= 100, "Consolidation removed too many fibers"


class TestStatsAccuracy:
    """Verify stats counts match actual SQL data."""

    async def test_stats_match_reality(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        for i in range(25):
            await encoder.encode(f"Memory number {i}: test content about topic {i}")

        brain_id = sqlite_storage._current_brain_id
        assert brain_id is not None
        stats = await sqlite_storage.get_stats(brain_id)

        # Fiber count should match number of encodes
        assert stats["fiber_count"] == 25

        # Neuron count should be positive (each encode creates multiple neurons)
        assert stats["neuron_count"] > 25

        # Synapse count should be positive
        assert stats["synapse_count"] > 0
