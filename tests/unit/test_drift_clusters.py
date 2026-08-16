"""Tests for engine/drift_clusters.py — Jaccard tag-cluster detection.

Ported from the pre-3.0 test_drift_detection.py (removed alongside the SQLite
backend in 3524066d), scoped to the half of the old suite that still applies:
compute_jaccard, detect_clusters, the frozen data models, and UnionFind.
Deliberately NOT ported: temporal/activation (Wasserstein-1) drift and the old
MCP drift handler — see drift_clusters.py's module docstring for why.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.clustering import UnionFind
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.drift_clusters import (
    JACCARD_MERGE_THRESHOLD,
    MIN_COOCCURRENCE_COUNT,
    DriftReport,
    TagCluster,
    compute_jaccard,
    detect_clusters,
    refresh_drift_clusters,
)
from surreal_memory.storage.memory_store import InMemoryStorage

# ── compute_jaccard ─────────────────────────────────────────────────


class TestComputeJaccard:
    def test_perfect_overlap(self) -> None:
        assert compute_jaccard("a", "b", {"a": 10, "b": 10}, 10) == 1.0

    def test_no_overlap(self) -> None:
        assert compute_jaccard("a", "b", {"a": 10, "b": 10}, 0) == 0.0

    def test_partial_overlap(self) -> None:
        result = compute_jaccard("a", "b", {"a": 10, "b": 10}, 5)
        assert abs(result - 1 / 3) < 0.01

    def test_missing_tag_a(self) -> None:
        assert compute_jaccard("x", "b", {"b": 10}, 5) == 0.0

    def test_missing_tag_b(self) -> None:
        assert compute_jaccard("a", "x", {"a": 10}, 5) == 0.0

    def test_asymmetric_counts(self) -> None:
        assert compute_jaccard("a", "b", {"a": 20, "b": 5}, 5) == 0.25

    def test_zero_union(self) -> None:
        assert compute_jaccard("a", "b", {"a": 0, "b": 0}, 0) == 0.0

    def test_high_jaccard(self) -> None:
        result = compute_jaccard("a", "b", {"a": 10, "b": 10}, 8)
        assert abs(result - 8 / 12) < 0.001


# ── detect_clusters ─────────────────────────────────────────────────


class TestDetectClusters:
    def test_empty_cooccurrences(self) -> None:
        assert detect_clusters([], {}) == []

    def test_single_pair_below_threshold(self) -> None:
        cooccurrences = [("a", "b", 1)]
        counts = {"a": 10, "b": 10}
        assert detect_clusters(cooccurrences, counts) == []

    def test_single_pair_low_jaccard(self) -> None:
        cooccurrences = [("a", "b", MIN_COOCCURRENCE_COUNT)]
        counts = {"a": 100, "b": 100}
        assert detect_clusters(cooccurrences, counts) == []

    def test_merge_suggestion(self) -> None:
        cooccurrences = [("react", "reactjs", 10)]
        counts = {"react": 12, "reactjs": 11}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert reports[0].suggestion == "merge"
        assert reports[0].cluster.confidence >= JACCARD_MERGE_THRESHOLD

    def test_alias_suggestion(self) -> None:
        cooccurrences = [("auth", "authentication", 6)]
        counts = {"auth": 8, "authentication": 8}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert reports[0].suggestion == "alias"

    def test_canonical_is_most_used(self) -> None:
        cooccurrences = [("js", "javascript", 10)]
        counts = {"js": 5, "javascript": 20}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert reports[0].cluster.canonical == "javascript"

    def test_multiple_clusters(self) -> None:
        cooccurrences = [
            ("react", "reactjs", 10),
            ("vue", "vuejs", 8),
        ]
        counts = {"react": 12, "reactjs": 11, "vue": 10, "vuejs": 9}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 2

    def test_transitive_union(self) -> None:
        cooccurrences = [
            ("a", "b", 8),
            ("b", "c", 8),
        ]
        counts = {"a": 10, "b": 10, "c": 10}
        reports = detect_clusters(cooccurrences, counts)
        assert len(reports) == 1
        assert len(reports[0].cluster.members) == 3

    def test_tag_below_min_fibers_excluded(self) -> None:
        cooccurrences = [("rare", "common", 5)]
        counts = {"rare": 1, "common": 10}
        reports = detect_clusters(cooccurrences, counts)
        assert reports == []

    def test_cluster_id_is_stable(self) -> None:
        cooccurrences = [("x", "y", 10)]
        counts = {"x": 12, "y": 12}
        reports1 = detect_clusters(cooccurrences, counts)
        reports2 = detect_clusters(cooccurrences, counts)
        assert reports1[0].cluster_id == reports2[0].cluster_id

    def test_cluster_sorted_by_confidence(self) -> None:
        cooccurrences = [
            ("low_a", "low_b", 5),
            ("high_a", "high_b", 10),
        ]
        counts = {"low_a": 10, "low_b": 10, "high_a": 11, "high_b": 11}
        reports = detect_clusters(cooccurrences, counts)
        if len(reports) >= 2:
            assert reports[0].cluster.confidence >= reports[1].cluster.confidence


# ── TagCluster / DriftReport ────────────────────────────────────────


class TestDataModels:
    def test_tag_cluster_frozen(self) -> None:
        tc = TagCluster(canonical="react", members=frozenset({"react", "reactjs"}), confidence=0.8)
        with pytest.raises(AttributeError):
            tc.canonical = "vue"  # type: ignore[misc]

    def test_drift_report_frozen(self) -> None:
        tc = TagCluster(canonical="a", members=frozenset({"a", "b"}), confidence=0.5)
        dr = DriftReport(cluster=tc, suggestion="merge", cluster_id="abc123")
        assert dr.suggestion == "merge"
        with pytest.raises(AttributeError):
            dr.suggestion = "alias"  # type: ignore[misc]

    def test_tag_cluster_evidence(self) -> None:
        tc = TagCluster(
            canonical="react",
            members=frozenset({"react", "reactjs"}),
            confidence=0.8,
            evidence="Tags co-occur frequently",
        )
        assert "co-occur" in tc.evidence


# ── UnionFind ───────────────────────────────────────────────────────


class TestUnionFind:
    def test_basic_union(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)
        assert uf.find(2) != uf.find(0)

    def test_transitive_union(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_groups(self) -> None:
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 3

    def test_single_element_groups(self) -> None:
        uf = UnionFind(3)
        assert len(uf.groups()) == 3


# ── refresh_drift_clusters (orchestrator) ────────────────────────────


class TestRefreshDriftClusters:
    @pytest.mark.asyncio
    async def test_persists_and_counts_clusters(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(return_value=[("react", "reactjs", 10)])
        storage.get_tag_fiber_counts = AsyncMock(return_value={"react": 12, "reactjs": 11})
        storage.save_drift_cluster = AsyncMock()

        detected, saved = await refresh_drift_clusters(storage)

        assert saved == 1
        storage.save_drift_cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_brain_saves_nothing(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(return_value=[])
        storage.get_tag_fiber_counts = AsyncMock(return_value={})
        storage.save_drift_cluster = AsyncMock()

        detected, saved = await refresh_drift_clusters(storage)

        assert saved == 0
        storage.save_drift_cluster.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_soft_when_cooccurrence_read_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(side_effect=RuntimeError("boom"))
        storage.get_tag_fiber_counts = AsyncMock(return_value={})
        storage.save_drift_cluster = AsyncMock()

        with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.drift_clusters"):
            detected, saved = await refresh_drift_clusters(storage)

        assert saved == 0
        storage.save_drift_cluster.assert_not_called()
        # Degrading to 0 is fine; degrading SILENTLY is not — a bare 0 here is
        # indistinguishable from an honest "no drift found".
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_fails_soft_when_fiber_counts_read_raises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(return_value=[("a", "b", 10)])
        storage.get_tag_fiber_counts = AsyncMock(side_effect=RuntimeError("boom"))
        storage.save_drift_cluster = AsyncMock()

        with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.drift_clusters"):
            detected, saved = await refresh_drift_clusters(storage)

        assert saved == 0
        storage.save_drift_cluster.assert_not_called()
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    @pytest.mark.asyncio
    async def test_one_bad_persist_does_not_block_the_rest(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(
            return_value=[
                ("react", "reactjs", 10),
                ("vue", "vuejs", 8),
            ]
        )
        storage.get_tag_fiber_counts = AsyncMock(
            return_value={"react": 12, "reactjs": 11, "vue": 10, "vuejs": 9}
        )
        storage.save_drift_cluster = AsyncMock(side_effect=[RuntimeError("boom"), None])

        detected, saved = await refresh_drift_clusters(storage)

        assert detected == 2, "both clusters were detected"
        assert saved == 1, "only one persisted — the failure must stay visible"
        assert storage.save_drift_cluster.call_count == 2


# ── detect_drift as a consolidation strategy ─────────────────────────


@pytest_asyncio.fixture
async def drift_storage() -> InMemoryStorage:
    """A brain whose fibers + co-occurrence make exactly one cluster detectable."""
    store = InMemoryStorage()
    brain = Brain.create(name="detect-drift-test", brain_id="detect-drift-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    for idx in range(12):
        fiber = Fiber.create(
            neuron_ids={f"n{idx}"},
            synapse_ids=set(),
            anchor_neuron_id=f"n{idx}",
            auto_tags={"react", "reactjs"},
        )
        await store.add_fiber(fiber)
        await store.record_tag_cooccurrence({"react", "reactjs"})
    return store


class TestDetectDriftStrategy:
    @pytest.mark.asyncio
    async def test_detect_drift_persists_clusters(self, drift_storage: InMemoryStorage) -> None:
        engine = ConsolidationEngine(drift_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.DETECT_DRIFT])

        assert report.drift_clusters_found == 1
        assert len(await drift_storage.get_drift_clusters()) == 1

    @pytest.mark.asyncio
    async def test_detect_drift_dry_run_previews_without_writing(
        self, drift_storage: InMemoryStorage
    ) -> None:
        """A dry run must PREVIEW: report what it would save, persist nothing.

        Regression guard: the first implementation returned early on dry_run
        before detecting anything, so a preview of a drifting brain was
        indistinguishable from a preview of a clean one — a flat 0 either way.
        """
        engine = ConsolidationEngine(drift_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.DETECT_DRIFT], dry_run=True)

        assert report.dry_run is True
        assert report.drift_clusters_found == 1, "dry run must report the cluster it would save"
        assert await drift_storage.get_drift_clusters() == [], "dry run must persist nothing"

    @pytest.mark.asyncio
    async def test_dry_run_on_a_clean_brain_is_distinguishable_from_a_drifting_one(self) -> None:
        """The other half of the same guarantee: a clean brain previews as 0."""
        store = InMemoryStorage()
        brain = Brain.create(name="detect-drift-clean", brain_id="detect-drift-clean-brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        engine = ConsolidationEngine(store)
        report = await engine.run(strategies=[ConsolidationStrategy.DETECT_DRIFT], dry_run=True)

        assert report.drift_clusters_found == 0

    @pytest.mark.asyncio
    async def test_refresh_without_persist_reports_but_does_not_save(self) -> None:
        storage = MagicMock()
        storage.get_tag_cooccurrence = AsyncMock(return_value=[("react", "reactjs", 10)])
        storage.get_tag_fiber_counts = AsyncMock(return_value={"react": 12, "reactjs": 11})
        storage.save_drift_cluster = AsyncMock()

        detected, would_save = await refresh_drift_clusters(storage, persist=False)

        assert would_save == 1
        storage.save_drift_cluster.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_drift_runs_in_the_semantic_link_tier(self) -> None:
        tiers = [
            tier
            for tier in ConsolidationEngine.STRATEGY_TIERS
            if ConsolidationStrategy.DETECT_DRIFT in tier
        ]
        assert len(tiers) == 1
        assert ConsolidationStrategy.SEMANTIC_LINK in tiers[0]

    @pytest.mark.asyncio
    async def test_found_clusters_appear_in_the_report_users_actually_see(
        self, drift_storage: InMemoryStorage
    ) -> None:
        """summary() is the surface the CLI, the MCP handlers and the background
        daemon all print. A counter that is set but never rendered there — and
        never folded into the total_changes that suppresses the "Why nothing
        changed" hints — recreates, in the report layer, the same "found
        something but reported nothing" ambiguity U7 exists to remove.
        """
        engine = ConsolidationEngine(drift_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.DETECT_DRIFT])
        summary = report.summary()

        assert report.drift_clusters_found == 1
        assert "Drift clusters found: 1" in summary
        assert "Why nothing changed" not in summary

    @pytest.mark.asyncio
    async def test_empty_run_still_explains_itself(self) -> None:
        """The converse: with genuinely nothing found, the hints must still fire."""
        store = InMemoryStorage()
        brain = Brain.create(name="detect-drift-hints", brain_id="detect-drift-hints-brain")
        await store.save_brain(brain)
        store.set_brain(brain.id)

        report = await ConsolidationEngine(store).run(
            strategies=[ConsolidationStrategy.DETECT_DRIFT]
        )
        summary = report.summary()

        assert "Drift clusters found: 0" in summary
        assert "Why nothing changed" in summary

    @pytest.mark.asyncio
    async def test_detect_drift_survives_a_storage_failure(
        self, drift_storage: InMemoryStorage
    ) -> None:
        """A raising storage layer must degrade to 0, never abort the pass."""
        drift_storage.get_tag_cooccurrence = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("boom")
        )
        engine = ConsolidationEngine(drift_storage)
        report = await engine.run(strategies=[ConsolidationStrategy.DETECT_DRIFT])

        assert report.drift_clusters_found == 0
