"""Tests for the drift-detection storage mixins (tag_cooccurrence + drift_clusters).

Exercised against InMemoryStorage — a synchronous, dependency-free stand-in for
the SurrealDB mixin (storage/surrealdb/drift.py). The two mixins are written to
share identical semantics (see memory_drift.py's docstring), so these tests
document the storage contract that test_surrealdb_drift_live.py verifies again
against the real backend.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def store() -> InMemoryStorage:
    """InMemoryStorage with a brain context set."""
    storage = InMemoryStorage()
    brain = Brain.create(name="drift-test", brain_id="drift-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


# ── record_tag_cooccurrence / get_tag_cooccurrence ──────────────────────────


@pytest.mark.asyncio
async def test_record_and_get_cooccurrence(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence({"react", "typescript", "frontend"})
    pairs = await store.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 3  # 3 choose 2 = 3 pairs


@pytest.mark.asyncio
async def test_cooccurrence_count_increments(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence({"a", "b"})
    await store.record_tag_cooccurrence({"a", "b"})
    pairs = await store.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 1
    assert pairs[0][2] == 2


@pytest.mark.asyncio
async def test_cooccurrence_canonical_order(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence({"z", "a"})
    pairs = await store.get_tag_cooccurrence(min_count=1)
    assert pairs[0][0] == "a"
    assert pairs[0][1] == "z"


@pytest.mark.asyncio
async def test_single_tag_no_cooccurrence(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence({"only_one"})
    pairs = await store.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 0


@pytest.mark.asyncio
async def test_empty_tags_no_cooccurrence(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence(set())
    pairs = await store.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 0


@pytest.mark.asyncio
async def test_min_count_filters_pairs(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence({"a", "b"})
    await store.record_tag_cooccurrence({"c", "d"})
    await store.record_tag_cooccurrence({"c", "d"})
    pairs = await store.get_tag_cooccurrence(min_count=2)
    assert len(pairs) == 1
    assert pairs[0][:2] == ("c", "d")


@pytest.mark.asyncio
async def test_cooccurrence_scoped_by_brain(store: InMemoryStorage) -> None:
    await store.record_tag_cooccurrence({"a", "b"})
    other = Brain.create(name="other", brain_id="other-brain")
    await store.save_brain(other)
    store.set_brain(other.id)
    assert await store.get_tag_cooccurrence(min_count=1) == []
    store.set_brain("drift-brain")
    assert len(await store.get_tag_cooccurrence(min_count=1)) == 1


# ── get_tag_fiber_counts ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_tag_fiber_counts(store: InMemoryStorage) -> None:
    f1 = Fiber.create(
        neuron_ids={"n1"},
        synapse_ids=set(),
        anchor_neuron_id="n1",
        auto_tags={"react", "typescript"},
    )
    f2 = Fiber.create(
        neuron_ids={"n2"},
        synapse_ids=set(),
        anchor_neuron_id="n2",
        auto_tags={"react", "python"},
        agent_tags={"api"},
    )
    await store.add_fiber(f1)
    await store.add_fiber(f2)

    counts = await store.get_tag_fiber_counts()
    assert counts["react"] == 2
    assert counts["typescript"] == 1
    assert counts["python"] == 1
    assert counts["api"] == 1


@pytest.mark.asyncio
async def test_get_tag_fiber_counts_empty_brain(store: InMemoryStorage) -> None:
    assert await store.get_tag_fiber_counts() == {}


def test_both_backends_cap_the_fiber_scan_at_the_same_bound() -> None:
    """The two mixins must truncate Jaccard's denominator identically.

    Regression guard: the in-memory mixin originally had NO cap while the
    SurrealDB one stopped at 10000, so every test written against the
    in-memory backend proved something untrue about production — precisely
    the divergence memory_drift.py's docstring promises does not exist.
    Past the cap the numerator (cumulative pair_count) and the denominator
    (a truncated fiber sample) describe different populations, so a mismatch
    here silently skews cluster confidences on exactly the large, long-lived
    brains where drift detection matters most.
    """
    from surreal_memory.storage.memory_drift import _MAX_FIBER_SCAN as MEMORY_CAP
    from surreal_memory.storage.surrealdb.drift import _MAX_FIBER_SCAN as SURREAL_CAP

    assert MEMORY_CAP == SURREAL_CAP


def test_both_backends_cap_pair_generation_at_the_same_bound() -> None:
    """Same argument for the O(n^2) tag-pair cap."""
    from surreal_memory.storage.memory_drift import _MAX_PAIRS_PER_CALL as MEMORY_CAP
    from surreal_memory.storage.surrealdb.drift import _MAX_PAIRS_PER_CALL as SURREAL_CAP

    assert MEMORY_CAP == SURREAL_CAP


@pytest.mark.asyncio
async def test_fiber_scan_truncation_is_logged_not_silent(
    store: InMemoryStorage, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Hitting the cap must leave a trace — the caller cannot detect truncation
    from the returned dict, so silence there means a large brain gets
    approximate confidences with nothing to explain why.
    """
    import logging

    from surreal_memory.storage import memory_drift

    monkeypatch.setattr(memory_drift, "_MAX_FIBER_SCAN", 2)
    for idx in range(3):
        await store.add_fiber(
            Fiber.create(
                neuron_ids={f"n{idx}"},
                synapse_ids=set(),
                anchor_neuron_id=f"n{idx}",
                auto_tags={f"tag{idx}"},
            )
        )

    with caplog.at_level(logging.WARNING, logger=memory_drift.__name__):
        counts = await store.get_tag_fiber_counts()

    assert len(counts) == 2, "the scan must actually stop at the cap"
    assert any("cap" in r.message for r in caplog.records), "truncation must be logged"


@pytest.mark.asyncio
async def test_fiber_scan_is_deterministic_across_calls(store: InMemoryStorage) -> None:
    """Two passes over an unchanged brain must return identical counts.

    Without a stable ordering the truncated sample varies between runs, so the
    same tag pair can be scored differently on consecutive consolidation passes
    with nothing about the brain having changed.
    """
    for idx in range(6):
        await store.add_fiber(
            Fiber.create(
                neuron_ids={f"n{idx}"},
                synapse_ids=set(),
                anchor_neuron_id=f"n{idx}",
                auto_tags={"shared", f"tag{idx}"},
            )
        )

    assert await store.get_tag_fiber_counts() == await store.get_tag_fiber_counts()


# ── save_drift_cluster / get_drift_clusters / resolve_drift_cluster ────────


@pytest.mark.asyncio
async def test_save_and_get_drift_cluster(store: InMemoryStorage) -> None:
    await store.save_drift_cluster(
        cluster_id="c1",
        canonical="react",
        members=["react", "reactjs"],
        confidence=0.85,
        status="detected",
    )
    clusters = await store.get_drift_clusters()
    assert len(clusters) == 1
    assert clusters[0]["canonical"] == "react"
    assert clusters[0]["confidence"] == 0.85
    assert clusters[0]["resolved_at"] is None


@pytest.mark.asyncio
async def test_get_clusters_filter_by_status(store: InMemoryStorage) -> None:
    await store.save_drift_cluster("c1", "a", ["a", "b"], 0.8, "detected")
    await store.save_drift_cluster("c2", "x", ["x", "y"], 0.6, "merged")
    detected = await store.get_drift_clusters(status="detected")
    merged = await store.get_drift_clusters(status="merged")
    assert len(detected) == 1
    assert len(merged) == 1


@pytest.mark.asyncio
async def test_resolve_drift_cluster(store: InMemoryStorage) -> None:
    await store.save_drift_cluster("c1", "a", ["a", "b"], 0.8, "detected")
    result = await store.resolve_drift_cluster("c1", "merged")
    assert result is True
    clusters = await store.get_drift_clusters(status="merged")
    assert len(clusters) == 1
    assert clusters[0]["resolved_at"] is not None


@pytest.mark.asyncio
async def test_resolve_nonexistent_cluster(store: InMemoryStorage) -> None:
    result = await store.resolve_drift_cluster("nonexistent", "merged")
    assert result is False


@pytest.mark.asyncio
async def test_upsert_drift_cluster_preserves_created_at(store: InMemoryStorage) -> None:
    await store.save_drift_cluster("c1", "a", ["a", "b"], 0.5, "detected")
    first = (await store.get_drift_clusters())[0]
    await store.save_drift_cluster("c1", "a", ["a", "b", "c"], 0.9, "detected")
    clusters = await store.get_drift_clusters()
    assert len(clusters) == 1
    assert clusters[0]["confidence"] == 0.9
    assert "c" in clusters[0]["members"]
    assert clusters[0]["created_at"] == first["created_at"]


@pytest.mark.asyncio
async def test_upsert_clears_resolved_at(store: InMemoryStorage) -> None:
    await store.save_drift_cluster("c1", "a", ["a", "b"], 0.5, "detected")
    await store.resolve_drift_cluster("c1", "merged")
    await store.save_drift_cluster("c1", "a", ["a", "b"], 0.9, "detected")
    clusters = await store.get_drift_clusters()
    assert clusters[0]["resolved_at"] is None
    assert clusters[0]["status"] == "detected"


# ── producer wiring: encode() -> BuildFiberStep -> record_tag_cooccurrence ─


@pytest.mark.asyncio
async def test_encode_records_tag_cooccurrence_for_merged_tags() -> None:
    """Regression guard for the write dropped in dd6f5a62 (#151) and restored
    in U7 once SurrealDB actually implements the storage side. Runs the real
    MemoryEncoder.encode() path (not a direct BuildFiberStep call) so a future
    refactor of the pipeline's step wiring is still caught.
    """
    storage = InMemoryStorage()
    brain = Brain.create(name="drift-producer-test", config=BrainConfig())
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    encoder = MemoryEncoder(storage, brain.config)

    result = await encoder.encode(
        "We decided to use Redis for caching",
        timestamp=datetime(2024, 2, 4, 15, 0),
        tags={"architecture-decision", "infrastructure"},
    )

    assert len(result.fiber.tags) >= 2
    pairs = await storage.get_tag_cooccurrence(min_count=1)
    recorded_tags = {t for pair in pairs for t in pair[:2]}
    assert recorded_tags & set(result.fiber.tags)


@pytest.mark.asyncio
async def test_encode_survives_a_failing_cooccurrence_write() -> None:
    """The producer is fail-soft: a raising record_tag_cooccurrence must not
    lose the fiber the user just asked to store. This is why the call sits
    AFTER add_fiber and inside a try/except that only logs at debug.
    """
    storage = InMemoryStorage()
    brain = Brain.create(name="drift-producer-failsoft-test", config=BrainConfig())
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    storage.record_tag_cooccurrence = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("storage is down")
    )
    encoder = MemoryEncoder(storage, brain.config)

    result = await encoder.encode(
        "We decided to use Redis for caching",
        timestamp=datetime(2024, 2, 4, 15, 0),
        tags={"architecture-decision", "infrastructure"},
    )

    storage.record_tag_cooccurrence.assert_awaited()
    assert await storage.get_fiber(result.fiber.id) is not None
