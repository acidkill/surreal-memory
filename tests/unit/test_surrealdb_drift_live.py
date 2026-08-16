"""Live-SurrealDB tests for the drift-detection storage mixin (U7).

Exercises storage/surrealdb/drift.py against a real server — the in-memory
mixin (test_drift_storage.py) proves the *contract*, this proves the actual
SurrealQL. Specifically covers the two failure classes this run hit while
building the feature:

- BUG-004-shaped record-id bugs: every id here goes through
  ``type::record('table', $var)`` with a bound parameter, never spliced into
  query text — tested here with tag/cluster-id content that starts with a
  digit, the exact shape that broke the un-fixed device registry.
- BUG-007-shaped UPSERT bugs: the multi-statement ``record_tag_cooccurrence``
  batch uses real per-index parameter names, not a literal ``{i}`` placeholder
  — tested here with >1 pair in a single call so a broken batch would fail on
  the second statement.

Skipped when SURREALDB_URL is unset so CI without docker still passes.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.drift_clusters import refresh_drift_clusters
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="drift-live-test-7c2e9a")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    try:
        await cleanup_live_brains(storage, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_record_and_get_cooccurrence_round_trip(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    await surrealdb_storage.record_tag_cooccurrence({"react", "typescript", "frontend"})
    pairs = await surrealdb_storage.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 3
    assert all(a < b for a, b, _count in pairs)


@pytest.mark.asyncio
async def test_cooccurrence_count_increments_on_repeat(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    await surrealdb_storage.record_tag_cooccurrence({"alpha", "beta"})
    await surrealdb_storage.record_tag_cooccurrence({"alpha", "beta"})
    pairs = await surrealdb_storage.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 1
    assert pairs[0] == ("alpha", "beta", 2)


@pytest.mark.asyncio
async def test_record_tag_cooccurrence_batch_survives_multiple_pairs(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """4 tags -> 6 pairs in one UPSERT batch; a literal (non-f-string) `{i}`
    placeholder bug (BUG-007's shape) would fail on the 2nd statement.
    """
    await surrealdb_storage.record_tag_cooccurrence({"w", "x", "y", "z"})
    pairs = await surrealdb_storage.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 6
    assert all(count == 1 for _a, _b, count in pairs)


@pytest.mark.asyncio
async def test_record_tag_cooccurrence_digit_leading_tags(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """Tag text starting with a digit must not trip record-id text parsing
    (the BUG-004 class of failure) — the pair id is a hashed digest, so this
    also guards against a future regression that stops hashing.
    """
    await surrealdb_storage.record_tag_cooccurrence({"3d-modeling", "9front"})
    pairs = await surrealdb_storage.get_tag_cooccurrence(min_count=1)
    assert len(pairs) == 1
    assert pairs[0][:2] == ("3d-modeling", "9front")


@pytest.mark.asyncio
async def test_get_tag_fiber_counts_scans_real_fibers(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    for idx, (auto, agent) in enumerate(
        [({"react", "typescript"}, set()), ({"react", "python"}, {"api"})]
    ):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"drift-live-{idx}")
        await surrealdb_storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            auto_tags=auto,
            agent_tags=agent,
        )
        await surrealdb_storage.add_fiber(fiber)

    counts = await surrealdb_storage.get_tag_fiber_counts()
    assert counts["react"] == 2
    assert counts["typescript"] == 1
    assert counts["python"] == 1
    assert counts["api"] == 1


@pytest.mark.asyncio
async def test_drift_cluster_round_trip(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    await surrealdb_storage.save_drift_cluster(
        cluster_id="c1",
        canonical="react",
        members=["react", "reactjs"],
        confidence=0.85,
        status="detected",
    )
    clusters = await surrealdb_storage.get_drift_clusters()
    assert len(clusters) == 1
    assert clusters[0]["canonical"] == "react"
    assert clusters[0]["confidence"] == pytest.approx(0.85)
    assert clusters[0]["resolved_at"] is None


@pytest.mark.asyncio
async def test_drift_cluster_id_starting_with_digit(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """cluster_id is caller-supplied (e.g. a sha256[:12] hex digest, which
    routinely starts with a digit) — must go through type::record(), never a
    raw f-string record-id.
    """
    await surrealdb_storage.save_drift_cluster(
        cluster_id="7f3a9c2b1e08",
        canonical="k8s",
        members=["k8s", "kubernetes"],
        confidence=0.6,
        status="detected",
    )
    clusters = await surrealdb_storage.get_drift_clusters()
    assert len(clusters) == 1
    assert clusters[0]["canonical"] == "k8s"


@pytest.mark.asyncio
async def test_resolve_drift_cluster_and_status_filter(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    await surrealdb_storage.save_drift_cluster("c1", "a", ["a", "b"], 0.8, "detected")
    await surrealdb_storage.save_drift_cluster("c2", "x", ["x", "y"], 0.6, "detected")

    resolved = await surrealdb_storage.resolve_drift_cluster("c1", "merged")
    assert resolved is True

    merged = await surrealdb_storage.get_drift_clusters(status="merged")
    detected = await surrealdb_storage.get_drift_clusters(status="detected")
    assert len(merged) == 1
    assert merged[0]["resolved_at"] is not None
    assert len(detected) == 1


@pytest.mark.asyncio
async def test_resolve_nonexistent_cluster_returns_false(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    assert await surrealdb_storage.resolve_drift_cluster("does-not-exist", "merged") is False


@pytest.mark.asyncio
async def test_upsert_clears_resolved_at_on_resave(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    await surrealdb_storage.save_drift_cluster("c1", "a", ["a", "b"], 0.5, "detected")
    await surrealdb_storage.resolve_drift_cluster("c1", "dismissed")
    await surrealdb_storage.save_drift_cluster("c1", "a", ["a", "b"], 0.9, "detected")

    clusters = await surrealdb_storage.get_drift_clusters()
    assert len(clusters) == 1
    assert clusters[0]["status"] == "detected"
    assert clusters[0]["resolved_at"] is None
    assert clusters[0]["confidence"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_refresh_drift_clusters_end_to_end(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """Full read (tag_cooccurrence + fiber tags) -> detect -> persist pass,
    exactly as the detect_drift consolidation strategy runs it.
    """
    for idx in range(3):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"drift-e2e-{idx}")
        await surrealdb_storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            auto_tags={"react", "reactjs"},
        )
        await surrealdb_storage.add_fiber(fiber)
        await surrealdb_storage.record_tag_cooccurrence({"react", "reactjs"})

    detected, saved = await refresh_drift_clusters(surrealdb_storage)

    assert detected == 1, "one cluster must be detected"
    assert saved == 1, "and it must actually persist — the two are reported separately"
    clusters = await surrealdb_storage.get_drift_clusters()
    assert len(clusters) == 1
    assert set(clusters[0]["members"]) == {"react", "reactjs"}
