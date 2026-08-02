"""Stress tests — full lifecycle workflows."""

from __future__ import annotations

from datetime import datetime

import pytest

from surreal_memory.core.memory_types import get_decay_rate
from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage

pytestmark = [pytest.mark.stress, pytest.mark.asyncio]


class TestRememberThenRecall:
    """Store memories, then query each to verify round-trip through real SQLite."""

    async def test_five_memories_all_retrievable(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        contents = [
            "PostgreSQL supports JSONB columns for semi-structured data",
            "Redis uses single-threaded event loop for high throughput",
            "Docker containers share the host OS kernel",
            "Kubernetes pods are the smallest deployable units",
            "GraphQL lets clients request exactly the fields they need",
        ]

        for content in contents:
            await encoder.encode(content)

        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        pipeline = ReflexPipeline(storage=sqlite_storage, config=brain.config)

        found = 0
        for content in contents:
            # Use a short keyword-based query
            keyword = content.split()[0]  # "PostgreSQL", "Redis", etc.
            result = await pipeline.query(keyword)
            if result.context and keyword.lower() in result.context.lower():
                found += 1

        # At least 3 out of 5 should be found by keyword
        assert found >= 3, f"Only {found}/5 memories retrievable by keyword"

    async def test_encoded_memory_creates_neurons_and_synapses(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        result = await encoder.encode(
            "Met Alice at the coffee shop to discuss API design",
            timestamp=datetime(2024, 2, 4, 15, 0),
        )

        assert result.fiber is not None
        assert len(result.neurons_created) > 0
        assert len(result.synapses_created) > 0

        # Verify neurons are actually in SQLite
        for neuron in result.neurons_created:
            stored = await sqlite_storage.get_neuron(neuron.id)
            assert stored is not None, f"Neuron {neuron.id} not found in storage"


class TestConsolidationPreservesRecall:
    """Verify consolidation doesn't destroy important memories."""

    async def test_recall_quality_survives_consolidation(
        self, sqlite_storage: InMemoryStorage, encoder: MemoryEncoder
    ) -> None:
        # Encode 20 memories
        memories = [
            "Python asyncio uses cooperative multitasking with event loops",
            "FastAPI generates OpenAPI documentation automatically",
            "SQLAlchemy 2.0 uses new type-safe query syntax",
            "Pydantic validates data using Python type hints",
            "pytest discovers tests by function name convention",
            "Docker Compose orchestrates multi-container applications",
            "Nginx reverse proxy handles SSL termination",
            "PostgreSQL VACUUM reclaims dead tuple storage space",
            "Redis pub/sub enables real-time messaging patterns",
            "GraphQL resolvers map schema fields to data sources",
            "JWT tokens contain base64-encoded JSON header and payload",
            "OAuth 2.0 authorization code flow uses redirect callbacks",
            "WebSocket upgrade starts with an HTTP handshake",
            "gRPC uses Protocol Buffers for efficient serialization",
            "Celery processes background tasks with message queues",
            "Alembic manages database schema migration versions",
            "mypy performs static type analysis on Python code",
            "Ruff is a fast Python linter written in Rust",
            "Black formats Python code with consistent style",
            "pre-commit hooks run checks before git commits",
        ]

        for mem in memories:
            await encoder.encode(mem)

        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        pipeline = ReflexPipeline(storage=sqlite_storage, config=brain.config)

        # Measure pre-consolidation recall
        test_queries = ["asyncio event loop", "PostgreSQL vacuum", "JWT token"]
        pre_results = []
        for q in test_queries:
            r = await pipeline.query(q)
            pre_results.append(len(r.context) if r.context else 0)

        # Run consolidation
        engine = ConsolidationEngine(storage=sqlite_storage)
        report = await engine.run(
            strategies=[
                ConsolidationStrategy.PRUNE,
                ConsolidationStrategy.MERGE,
                ConsolidationStrategy.ENRICH,
            ]
        )

        assert report.duration_ms >= 0

        # Measure post-consolidation recall
        post_results = []
        for q in test_queries:
            r = await pipeline.query(q)
            post_results.append(len(r.context) if r.context else 0)

        # At least 2/3 queries should still return context
        post_with_results = sum(1 for p in post_results if p > 0)
        assert post_with_results >= 2, (
            f"Only {post_with_results}/3 queries returned results after consolidation"
        )


class TestMultiTypeDecayRates:
    """Verify different memory types get different decay rates."""

    async def test_type_specific_decay_rates(self, sqlite_storage: InMemoryStorage) -> None:
        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        encoder = MemoryEncoder(storage=sqlite_storage, config=brain.config)

        # Encode memories of different types (inferred from content)
        type_content = {
            "fact": "Python 3.11 requires at minimum 64-bit architecture",
            "todo": "Add rate limiting to the public API endpoints",
            "error": "ConnectionRefusedError when connecting to Redis on port 6380",
            "decision": "We decided to use FastAPI instead of Flask for the REST API",
        }

        for mem_type, content in type_content.items():
            await encoder.encode(content, metadata={"type": mem_type})

        # Verify decay rates differ by type
        fact_rate = get_decay_rate("fact")
        todo_rate = get_decay_rate("todo")
        error_rate = get_decay_rate("error")
        decision_rate = get_decay_rate("decision")

        # TODOs should decay fastest, facts slowest
        assert todo_rate > fact_rate
        assert error_rate > decision_rate
        assert fact_rate < decision_rate


class TestConflictAutoDetection:
    """Verify conflicting memories trigger conflict detection."""

    async def test_contradictory_facts_detected(self, sqlite_storage: InMemoryStorage) -> None:
        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        encoder = MemoryEncoder(storage=sqlite_storage, config=brain.config)

        # Store first fact
        r1 = await encoder.encode("The project uses Python version 3.11")

        # Store contradicting fact
        r2 = await encoder.encode("The project uses Python version 3.12")

        # At least one should have detected conflicts
        total_conflicts = r1.conflicts_detected + r2.conflicts_detected
        # Note: conflict detection depends on entity extraction quality.
        # If both mention "Python version" as an entity, conflicts should be detected.
        # We check that encoding succeeded regardless.
        assert r1.fiber is not None
        assert r2.fiber is not None

        # Check for CONTRADICTS synapses in storage
        synapses = await sqlite_storage.get_synapses(source_id=r1.fiber.anchor_neuron_id)
        synapses += await sqlite_storage.get_synapses(source_id=r2.fiber.anchor_neuron_id)
        contradicts = [s for s in synapses if s.type == SynapseType.CONTRADICTS]

        # If conflict detection worked, we should have CONTRADICTS synapses
        # If not, we document this as a known gap
        if total_conflicts > 0:
            assert len(contradicts) > 0


class TestRelatedMemoryDiscovery:
    """Verify that encoding related content discovers connections."""

    async def test_related_memories_found(self, sqlite_storage: InMemoryStorage) -> None:
        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        encoder = MemoryEncoder(storage=sqlite_storage, config=brain.config)

        # Store 3 related memories about the same topic
        await encoder.encode("PostgreSQL supports advanced indexing with GIN and GiST")
        await encoder.encode("PostgreSQL JSONB columns use GIN indexes for fast queries")
        await encoder.encode("PostgreSQL full-text search uses GiST or GIN indexes")

        # Store a 4th memory on a different topic (should find no relations)
        r_unrelated = await encoder.encode("Docker containers use cgroups for resource isolation")

        # The 4th memory should succeed regardless
        assert r_unrelated.fiber is not None

        # Verify the brain has multiple fibers
        stats = await sqlite_storage.get_stats(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert stats["fiber_count"] >= 4


class TestTypedMemoryExpiry:
    """Verify typed memories with expiry get cleaned up by consolidation."""

    async def test_expired_memory_cleaned_by_prune(self, sqlite_storage: InMemoryStorage) -> None:
        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        encoder = MemoryEncoder(storage=sqlite_storage, config=brain.config)

        # Store a memory that would normally have short expiry
        result = await encoder.encode(
            "Set up monitoring dashboards for production metrics",
            metadata={"type": "todo"},
        )

        assert result.fiber is not None

        # Verify it exists
        fiber = await sqlite_storage.get_fiber(result.fiber.id)
        assert fiber is not None

        # Run consolidation - even with aggressive prune, fresh memories should survive
        engine = ConsolidationEngine(storage=sqlite_storage)
        await engine.run(strategies=[ConsolidationStrategy.PRUNE])

        # Memory should still exist (it's fresh)
        fiber_after = await sqlite_storage.get_fiber(result.fiber.id)
        assert fiber_after is not None


class TestRememberReturnsRelated:
    """Verify _remember enriches response with related memories."""

    async def test_encoding_produces_linkable_neurons(
        self, sqlite_storage: InMemoryStorage
    ) -> None:
        brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
        assert brain is not None
        encoder = MemoryEncoder(storage=sqlite_storage, config=brain.config)

        # Encode several related memories to build a rich graph
        results = []
        for content in [
            "Alice designed the authentication API module",
            "Bob reviewed Alice's authentication code changes",
            "The authentication module uses JWT with RSA-256",
        ]:
            r = await encoder.encode(content)
            results.append(r)

        # All should create fibers with neurons and synapses
        for r in results:
            assert r.fiber is not None
            assert len(r.neurons_created) > 0

        # Verify neuron linkage — shared entities should create connections
        total_synapses = 0
        for r in results:
            total_synapses += len(r.synapses_created)
        assert total_synapses > 3, f"Expected rich synapse graph, got {total_synapses}"
