"""Run-015 characterization + regression tests for consolidation integrity.

Each test here pins one registered defect (DEF-NN) from the consolidation-integrity plan.
A test marked ``xfail(strict=True)`` documents behaviour that is still broken; the marker
is removed in the same commit that fixes the defect.
"""

from __future__ import annotations

import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationReport,
)
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def summarize_storage() -> InMemoryStorage:
    """Storage holding one clusterable group of tag-overlapping fibers."""
    store = InMemoryStorage()
    brain = Brain.create(name="integrity_test", brain_id="integrity-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Four fibers sharing the same two tags -> one cluster above the default
    # min_cluster_size, so _summarize has exactly one group to work on.
    for idx in range(4):
        anchor = Neuron.create(
            type=NeuronType.ENTITY,
            content=f"anchor-{idx}",
            neuron_id=f"anchor-{idx}",
        )
        await store.add_neuron(anchor)
        fiber = Fiber.create(
            neuron_ids={anchor.id},
            synapse_ids=set(),
            anchor_neuron_id=anchor.id,
            summary=f"memory number {idx}",
            tags={"alpha", "beta"},
            fiber_id=f"src-fiber-{idx}",
        )
        await store.add_fiber(fiber)
    return store


async def _run_summarize(store: InMemoryStorage) -> ConsolidationReport:
    engine = ConsolidationEngine(store)
    report = ConsolidationReport()
    await engine._summarize(report, dry_run=False)
    return report


async def _summary_fiber_count(store: InMemoryStorage) -> int:
    fibers = await store.get_fibers(limit=10000)
    return sum(1 for f in fibers if f.metadata.get("_consolidation") == "summary_fiber")


class TestSummarizeIdempotence:
    """DEF-01 — _summarize must not recreate a summary it already produced."""

    async def test_first_pass_creates_a_summary(self, summarize_storage: InMemoryStorage) -> None:
        report = await _run_summarize(summarize_storage)
        assert report.summaries_created >= 1
        assert await _summary_fiber_count(summarize_storage) >= 1

    async def test_summarize_is_idempotent(self, summarize_storage: InMemoryStorage) -> None:
        """Second pass over unchanged input must create nothing (DEF-01)."""
        first = await _run_summarize(summarize_storage)
        count_after_first = await _summary_fiber_count(summarize_storage)

        second = await _run_summarize(summarize_storage)
        count_after_second = await _summary_fiber_count(summarize_storage)

        assert first.summaries_created >= 1, "precondition: first pass produced a summary"
        assert second.summaries_created == 0, (
            "second pass over unchanged input must create no new summaries"
        )
        assert count_after_second == count_after_first, (
            "no new summary fibers may be persisted on a repeat pass"
        )

    async def test_summarize_excludes_own_output_from_input(
        self, summarize_storage: InMemoryStorage
    ) -> None:
        """Summary fibers carry tags, so they must not feed the next clustering pass."""
        await _run_summarize(summarize_storage)

        engine = ConsolidationEngine(summarize_storage)
        fibers = await summarize_storage.get_fibers(limit=10000)
        eligible = engine._summarize_input_fibers(fibers)

        assert eligible, "clustering input must not be empty"
        assert all(f.metadata.get("_consolidation") != "summary_fiber" for f in eligible), (
            "summary fibers must be excluded from the clustering input"
        )


@pytest_asyncio.fixture
async def merge_storage() -> InMemoryStorage:
    """Two heavily-overlapping fibers, both carrying typed memory and rich metadata."""
    store = InMemoryStorage()
    brain = Brain.create(name="merge_test", brain_id="merge-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)

    shared = []
    for idx in range(6):
        neuron = Neuron.create(
            type=NeuronType.ENTITY, content=f"shared-{idx}", neuron_id=f"shared-{idx}"
        )
        await store.add_neuron(neuron)
        shared.append(neuron.id)

    for idx in range(2):
        fiber = Fiber.create(
            neuron_ids=set(shared),
            synapse_ids=set(),
            anchor_neuron_id=shared[0],
            summary=f"real content {idx}",
            tags={"merge"},
            fiber_id=f"merge-src-{idx}",
            metadata={"essence": f"essence-{idx}", "conductivity": 0.5},
        )
        await store.add_fiber(fiber)
        await store.add_typed_memory(
            TypedMemory(
                fiber_id=fiber.id,
                memory_type=MemoryType.FACT,
                priority=Priority.HIGH if idx == 0 else Priority.NORMAL,
                trust_score=0.9 if idx == 0 else 0.4,
            )
        )
    return store


async def _run_merge(store: InMemoryStorage) -> ConsolidationReport:
    engine = ConsolidationEngine(store)
    report = ConsolidationReport()
    await engine._merge(report, dry_run=False)
    return report


class TestMergeDataSafety:
    """DEF-02/03/04 — merging must not lose data."""

    async def test_merge_remaps_typed_memory(self, merge_storage: InMemoryStorage) -> None:
        """The typed layer must follow the surviving fiber, not be orphaned (DEF-03)."""
        report = await _run_merge(merge_storage)
        assert report.fibers_merged >= 2, "precondition: the two fibers must merge"

        surviving = [
            f for f in await merge_storage.get_fibers(limit=100) if f.metadata.get("merged_from")
        ]
        assert surviving, "a merged fiber must exist"
        merged_id = surviving[0].id

        merged_tm = await merge_storage.get_typed_memory(merged_id)
        assert merged_tm is not None, "merged fiber must carry the typed-memory layer"
        # Strongest priority and trust survive the merge.
        assert merged_tm.priority == Priority.HIGH
        assert merged_tm.trust_score == 0.9

        for src in ("merge-src-0", "merge-src-1"):
            assert await merge_storage.get_typed_memory(src) is None, (
                f"source typed memory {src} must not be left orphaned"
            )

    async def test_merge_preserves_source_metadata(self, merge_storage: InMemoryStorage) -> None:
        """essence/conductivity must survive; summary must not become a constant (DEF-04)."""
        await _run_merge(merge_storage)
        surviving = [
            f for f in await merge_storage.get_fibers(limit=100) if f.metadata.get("merged_from")
        ]
        assert surviving, "a merged fiber must exist"
        merged = surviving[0]

        assert "essence" in merged.metadata, "essence must survive the merge"
        assert "conductivity" in merged.metadata, "conductivity must survive the merge"
        assert merged.metadata.get("merged_from"), "merge provenance must be recorded"
        assert "real content" in merged.summary, (
            "merged summary must carry the sources' content, not a constant string"
        )

    async def test_fibers_removed_counts_only_confirmed_deletes(
        self, merge_storage: InMemoryStorage
    ) -> None:
        """A failing delete must not be counted as removed work (DEF-09)."""

        async def _always_fails(fiber_id: str) -> bool:
            return False

        merge_storage.delete_fiber = _always_fails  # type: ignore[method-assign]
        report = await _run_merge(merge_storage)

        assert report.fibers_merged >= 2, "precondition: merge still ran"
        assert report.fibers_removed == 0, (
            "fibers_removed must count confirmed deletes, not attempts"
        )
        assert report.extra.get("merge_delete_failures", 0) >= 1, (
            "failed deletes must be visible in the report"
        )


# Keys deliberately kept out of summary(), each with the reason it stays silent.
# Anything NOT listed here must be rendered — that is the contract.
_EXTRA_KEYS_NOT_RENDERED = {
    # Rendered indirectly: they feed the dedup census line, not a line of their own.
    "dedup_anchors_total": "feeds the dedup census line",
    "dedup_anchors_scanned": "feeds the dedup census line",
    "dedup_anchors_truncated": "feeds the dedup census line",
    "dedup_window_start": "diagnostic for the rotating dedup window",
    "alias_ledger_pairs": "ledger diagnostics, exported over MCP",
    "alias_ledger_complete": "ledger diagnostics, exported over MCP",
    "alias_ledger_load_failed": "ledger diagnostics, exported over MCP",
    "alias_checks_failed": "folded into the alias link line",
    "alias_writes_failed": "folded into the alias link line",
    "alias_pairs_skipped_invalid": "folded into the alias link line",
    "semantic_link_truncated": "folded into the semantic synapse line",
    "failed_strategies": "rendered by its own branch",
    "timed_out_strategies": "rendered by its own branch",
    "maturations_backfilled": "rendered by its own branch",
    "maturations_unreachable": "rendered by its own branch",
}


class TestCounterContract:
    """Anti-recurrence gate: a counter must not be able to go dark again.

    Two rounds of "consolidation honesty" fixes were shipped before and the same class
    of defect came back in neighbouring code, because each fix was local. This test is
    the systemic check: every key the engine writes into ``report.extra`` is either
    rendered by ``summary()`` or explicitly listed as intentionally silent.
    """

    def test_every_extra_key_is_rendered_or_explicitly_exempt(self) -> None:
        import re
        from pathlib import Path

        import surreal_memory.engine.consolidation as consolidation_module

        source = Path(consolidation_module.__file__).read_text(encoding="utf-8")
        written_keys = set(re.findall(r'report\.extra\[\s*"([a-z_]+)"\s*\]', source))
        assert written_keys, "the scan must find the keys the engine writes"

        rendered = set(re.findall(r'self\.extra\.get\(\s*"([a-z_]+)"', source))
        rendered |= set(re.findall(r'extra\.get\(\s*"([a-z_]+)"', source))

        undocumented = written_keys - rendered - set(_EXTRA_KEYS_NOT_RENDERED)
        assert not undocumented, (
            "these report.extra keys are written but never rendered and not listed as "
            f"intentionally silent: {sorted(undocumented)}. Either surface them in "
            "summary() or add them to _EXTRA_KEYS_NOT_RENDERED with a reason."
        )

    def test_drift_reports_detected_and_persisted_separately(self) -> None:
        """A failed persist must be visible, not silently lower the count (DEF-10)."""
        report = ConsolidationReport()
        report.drift_clusters_found = 5
        report.drift_clusters_persisted = 3

        text = report.summary()
        assert "Drift clusters found: 5" in text
        assert "FAILED to persist" in text, "the gap between detected and saved must show"

    def test_clean_drift_run_stays_quiet(self) -> None:
        report = ConsolidationReport()
        report.drift_clusters_found = 5
        report.drift_clusters_persisted = 5

        text = report.summary()
        assert "Drift clusters found: 5" in text
        assert "FAILED" not in text, "a healthy run must not add noise"

    def test_recorded_failures_reach_the_summary(self) -> None:
        """semantic_link_failures / compress_fibers_deferred were written, never shown."""
        report = ConsolidationReport()
        report.extra["semantic_link_failures"] = 7
        report.extra["compress_fibers_deferred"] = 4
        report.extra["summaries_skipped_existing"] = 25

        text = report.summary()
        assert "Semantic synapse writes FAILED: 7" in text
        assert "Fibers deferred (time budget): 4" in text
        assert "Summaries skipped (already exist): 25" in text
