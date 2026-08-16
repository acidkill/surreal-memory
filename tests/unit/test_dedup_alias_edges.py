"""Tests for idempotent dedup ALIAS edge creation.

Regression guard for the live-brain finding: 144,565 alias synapse rows
backing only 2,375 distinct (source, target) pairs, because the old
``except ValueError`` guard could never fire against a fresh UUID id.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
)
from surreal_memory.engine.dedup.alias_edges import (
    ALIAS_EDGE_WEIGHT,
    AliasEdgeLedger,
    AliasLinkOutcome,
    alias_edge_id,
    ensure_alias_edge,
)
from surreal_memory.engine.pipeline import PipelineContext
from surreal_memory.engine.pipeline_steps import CreateAnchorStep


class FakeStorage:
    """Minimal storage stub with the no-uniqueness semantics of both backends.

    Neither SQLite nor SurrealDB constrains ``(source, target, type)``; only
    the synapse primary key collides. This stub reproduces exactly that, so a
    test that passes here would also have caught the production bug.
    """

    def __init__(self, synapses: list[Synapse] | None = None) -> None:
        self.synapses: list[Synapse] = list(synapses or [])
        self.get_calls: list[dict] = []
        self.fail_reads = False
        self.fail_writes = False

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
        limit: int | None = None,
    ) -> list[Synapse]:
        self.get_calls.append({"source_id": source_id, "target_id": target_id, "type": type})
        if self.fail_reads:
            raise RuntimeError("storage unavailable")
        hits = [
            s
            for s in self.synapses
            if (source_id is None or s.source_id == source_id)
            and (target_id is None or s.target_id == target_id)
            and (type is None or s.type == type)
        ]
        return hits[:limit] if limit is not None else hits

    async def add_synapse(self, synapse: Synapse) -> str:
        if self.fail_writes:
            # Not a duplicate-key error: a dropped connection, a malformed row.
            raise RuntimeError("storage unavailable")
        if any(s.id == synapse.id for s in self.synapses):
            raise ValueError(f"Synapse {synapse.id} already exists")
        self.synapses.append(synapse)
        return synapse.id


def _alias(source_id: str, target_id: str) -> Synapse:
    return Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.ALIAS,
        weight=ALIAS_EDGE_WEIGHT,
        metadata={"_dedup": True},
    )


class TestAliasEdgeId:
    def test_id_is_deterministic_for_a_pair(self) -> None:
        assert alias_edge_id("dup-1", "canon-1") == alias_edge_id("dup-1", "canon-1")

    def test_id_is_direction_sensitive(self) -> None:
        # source -> target and target -> source are different claims.
        assert alias_edge_id("dup-1", "canon-1") != alias_edge_id("canon-1", "dup-1")

    def test_distinct_pairs_get_distinct_ids(self) -> None:
        assert alias_edge_id("dup-1", "canon-1") != alias_edge_id("dup-2", "canon-1")


class TestEnsureAliasEdgeSingleShot:
    @pytest.mark.asyncio
    async def test_creates_edge_when_absent(self) -> None:
        storage = FakeStorage()

        result = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert result.outcome is AliasLinkOutcome.CREATED
        assert result.created is True
        assert result.failed is False
        created = result.synapse
        assert created is not None
        assert created.type is SynapseType.ALIAS
        assert created.source_id == "dup-1"
        assert created.target_id == "canon-1"
        assert created.weight == ALIAS_EDGE_WEIGHT
        assert created.metadata == {"_dedup": True}
        assert len(storage.synapses) == 1

    @pytest.mark.asyncio
    async def test_second_call_for_same_pair_writes_nothing(self) -> None:
        storage = FakeStorage()

        first = await ensure_alias_edge(storage, "dup-1", "canon-1")
        second = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert first.outcome is AliasLinkOutcome.CREATED
        assert second.outcome is AliasLinkOutcome.ALREADY_EXISTS
        assert second.synapse is None
        assert len(storage.synapses) == 1

    @pytest.mark.asyncio
    async def test_repeated_runs_do_not_amplify(self) -> None:
        # The production failure mode: the same edge set re-created every
        # consolidation run. 100 runs must still leave exactly one row.
        storage = FakeStorage()

        for _ in range(100):
            await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert len(storage.synapses) == 1

    @pytest.mark.asyncio
    async def test_respects_preexisting_edge_created_elsewhere(self) -> None:
        # Rows already in the brain carry random UUID ids, so recognising them
        # has to work off the pair, not the id.
        storage = FakeStorage([_alias("dup-1", "canon-1")])

        result = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert result.outcome is AliasLinkOutcome.ALREADY_EXISTS
        assert result.synapse is None
        assert len(storage.synapses) == 1

    @pytest.mark.asyncio
    async def test_existence_check_is_type_scoped(self) -> None:
        # A SIMILAR_TO edge between the same anchors is a different claim and
        # must not suppress the alias edge.
        other = Synapse.create(
            source_id="dup-1",
            target_id="canon-1",
            type=SynapseType.SIMILAR_TO,
            weight=0.4,
        )
        storage = FakeStorage([other])

        result = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert result.outcome is AliasLinkOutcome.CREATED
        assert storage.get_calls[0]["type"] is SynapseType.ALIAS

    @pytest.mark.asyncio
    async def test_distinct_pairs_each_get_an_edge(self) -> None:
        storage = FakeStorage()

        await ensure_alias_edge(storage, "dup-1", "canon-1")
        await ensure_alias_edge(storage, "dup-2", "canon-1")
        await ensure_alias_edge(storage, "dup-3", "canon-2")

        assert len(storage.synapses) == 3


class TestEnsureAliasEdgeGuards:
    @pytest.mark.asyncio
    async def test_self_alias_is_rejected(self) -> None:
        storage = FakeStorage()

        result = await ensure_alias_edge(storage, "dup-1", "dup-1")

        assert result.outcome is AliasLinkOutcome.SKIPPED_INVALID
        assert result.synapse is None
        assert storage.synapses == []

    @pytest.mark.parametrize(("source", "target"), [("", "canon-1"), ("dup-1", "")])
    @pytest.mark.asyncio
    async def test_empty_endpoint_is_rejected(self, source: str, target: str) -> None:
        storage = FakeStorage()

        result = await ensure_alias_edge(storage, source, target)

        assert result.outcome is AliasLinkOutcome.SKIPPED_INVALID
        assert storage.synapses == []

    @pytest.mark.asyncio
    async def test_read_failure_skips_the_write(self) -> None:
        # Failing open would resurrect unbounded growth; a skipped edge is
        # recoverable on the next dedup pass, a duplicate row is not. What
        # changed is that the skip is now reported instead of looking like
        # "already linked".
        storage = FakeStorage()
        storage.fail_reads = True

        result = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert result.outcome is AliasLinkOutcome.CHECK_FAILED
        assert result.failed is True
        assert result.synapse is None
        assert storage.synapses == []

    @pytest.mark.asyncio
    async def test_partial_ledger_miss_that_cannot_be_verified_is_check_failed(self) -> None:
        # A partial ledger cannot prove absence, so the pair falls back to the
        # per-pair probe — and when that probe raises, the state is unknown.
        storage = FakeStorage()
        storage.fail_reads = True
        ledger = AliasEdgeLedger(complete=False)

        result = await ensure_alias_edge(storage, "dup-1", "canon-1", ledger=ledger)

        assert result.outcome is AliasLinkOutcome.CHECK_FAILED
        assert storage.synapses == []
        assert not ledger.has("dup-1", "canon-1")

    @pytest.mark.asyncio
    async def test_write_failure_is_reported_not_swallowed(self) -> None:
        # A dropped connection is not "already exists". Folding it into the
        # existing-edge count is the dishonesty this release removes.
        storage = FakeStorage()
        storage.fail_writes = True

        result = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert result.outcome is AliasLinkOutcome.WRITE_FAILED
        assert result.failed is True
        assert result.synapse is None
        assert storage.synapses == []

    @pytest.mark.asyncio
    async def test_failures_carry_the_exception_for_the_caller(self) -> None:
        # The helper logs failures at DEBUG and hands the exception up, so a
        # caller looping over thousands of pairs can report one traceback for
        # one root cause instead of one per pair.
        reads = FakeStorage()
        reads.fail_reads = True
        writes = FakeStorage()
        writes.fail_writes = True

        check_failed = await ensure_alias_edge(reads, "dup-1", "canon-1")
        write_failed = await ensure_alias_edge(writes, "dup-1", "canon-1")
        ok = await ensure_alias_edge(FakeStorage(), "dup-1", "canon-1")

        assert isinstance(check_failed.error, RuntimeError)
        assert isinstance(write_failed.error, RuntimeError)
        assert ok.error is None

    @pytest.mark.asyncio
    async def test_primary_key_collision_is_not_raised(self) -> None:
        # Backstop layer: a racing writer already inserted the deterministic
        # id, so add_synapse rejects. Losing that race is the correct outcome.
        preexisting = Synapse.create(
            source_id="dup-1",
            target_id="canon-1",
            type=SynapseType.ALIAS,
            synapse_id=alias_edge_id("dup-1", "canon-1"),
        )
        storage = FakeStorage()
        ledger = AliasEdgeLedger()
        storage.synapses.append(preexisting)

        result = await ensure_alias_edge(storage, "dup-1", "canon-1", ledger=ledger)

        assert result.outcome is AliasLinkOutcome.EXISTS_RACE
        assert result.failed is False
        assert result.synapse is None
        assert len(storage.synapses) == 1
        # The edge exists now, so the rest of this run must not probe for it.
        assert ledger.has("dup-1", "canon-1")


class TestAliasEdgeLedger:
    @pytest.mark.asyncio
    async def test_load_collects_existing_pairs(self) -> None:
        storage = FakeStorage([_alias("dup-1", "canon-1"), _alias("dup-2", "canon-1")])

        ledger = await AliasEdgeLedger.load(storage)

        assert len(ledger) == 2
        assert ledger.has("dup-1", "canon-1")
        assert ("dup-2", "canon-1") in ledger
        assert not ledger.has("dup-3", "canon-1")

    @pytest.mark.asyncio
    async def test_load_ignores_non_alias_edges(self) -> None:
        storage = FakeStorage(
            [
                _alias("dup-1", "canon-1"),
                Synapse.create(
                    source_id="a", target_id="b", type=SynapseType.RELATED_TO, weight=0.3
                ),
            ]
        )

        ledger = await AliasEdgeLedger.load(storage)

        assert len(ledger) == 1

    @pytest.mark.asyncio
    async def test_ledger_path_issues_no_per_pair_queries(self) -> None:
        # The consolidation pass compares up to 2000 anchors; the whole point
        # of the ledger is one query per run rather than one per pair.
        storage = FakeStorage()
        ledger = await AliasEdgeLedger.load(storage)
        queries_after_load = len(storage.get_calls)

        for i in range(50):
            await ensure_alias_edge(storage, f"dup-{i}", "canon-1", ledger=ledger)

        assert len(storage.get_calls) == queries_after_load
        assert len(storage.synapses) == 50

    @pytest.mark.asyncio
    async def test_ledger_suppresses_repeats_within_one_run(self) -> None:
        storage = FakeStorage()
        ledger = await AliasEdgeLedger.load(storage)

        first = await ensure_alias_edge(storage, "dup-1", "canon-1", ledger=ledger)
        second = await ensure_alias_edge(storage, "dup-1", "canon-1", ledger=ledger)

        assert first.outcome is AliasLinkOutcome.CREATED
        assert second.outcome is AliasLinkOutcome.ALREADY_EXISTS
        assert len(storage.synapses) == 1

    @pytest.mark.asyncio
    async def test_second_run_over_a_reloaded_ledger_is_a_no_op(self) -> None:
        # Full production shape: run dedup, reload, run it again.
        storage = FakeStorage()
        pairs = [(f"dup-{i}", "canon-1") for i in range(20)]

        run_one = await AliasEdgeLedger.load(storage)
        for src, tgt in pairs:
            await ensure_alias_edge(storage, src, tgt, ledger=run_one)
        assert len(storage.synapses) == 20

        run_two = await AliasEdgeLedger.load(storage)
        for src, tgt in pairs:
            await ensure_alias_edge(storage, src, tgt, ledger=run_two)

        assert len(storage.synapses) == 20

    @pytest.mark.asyncio
    async def test_load_propagates_read_failure(self) -> None:
        # Silently returning an empty ledger would mean "nothing exists" —
        # precisely the bug being fixed. Callers must fall back explicitly.
        storage = FakeStorage()
        storage.fail_reads = True

        with pytest.raises(RuntimeError):
            await AliasEdgeLedger.load(storage)


# ---------------------------------------------------------------------------
# Call sites
#
# The helper above was correct but unreferenced for one release: both dedup
# producers still minted their own fresh-UUID edge. These tests assert the
# wiring, not the helper — they fail if either producer goes back to calling
# ``Synapse.create`` directly.
# ---------------------------------------------------------------------------


ANCHOR_HASH = 0xDEADBEEFCAFEF00D


def _anchor(content: str, content_hash: int = ANCHOR_HASH) -> Neuron:
    return Neuron.create(
        type=NeuronType.CONCEPT,
        content=content,
        metadata={"is_anchor": True},
        content_hash=content_hash,
    )


class FakeAnchorStorage(FakeStorage):
    """FakeStorage plus the neuron read that ``_dedup`` paginates over."""

    def __init__(self, neurons: list[Neuron], synapses: list[Synapse] | None = None) -> None:
        super().__init__(synapses)
        self.neurons = list(neurons)
        self.current_brain_id = "brain-1"

    async def find_neurons(
        self,
        limit: int | None = None,
        offset: int = 0,
        **kwargs: object,
    ) -> list[Neuron]:
        window = self.neurons[offset:]
        return window[:limit] if limit is not None else window


class FakeEncodeStorage(FakeStorage):
    """FakeStorage plus the neuron write CreateAnchorStep performs."""

    def __init__(self) -> None:
        super().__init__()
        self.neurons: list[Neuron] = []

    async def add_neuron(self, neuron: Neuron) -> str:
        self.neurons.append(neuron)
        return neuron.id


class TestConsolidationDedupCallSite:
    @pytest.mark.asyncio
    async def test_repeated_runs_leave_one_row_per_pair(self) -> None:
        # The production shape exactly: consolidation re-derives the same
        # duplicate pair on every run. Three runs, one row.
        canonical = _anchor("deployment notes")
        duplicate = _anchor("deployment notes (copy)")
        storage = FakeAnchorStorage([canonical, duplicate])
        engine = ConsolidationEngine(storage)

        for _ in range(3):
            await engine._dedup(ConsolidationReport(), dry_run=False)

        alias_rows = [s for s in storage.synapses if s.type is SynapseType.ALIAS]
        assert len(alias_rows) == 1
        edge = alias_rows[0]
        assert (edge.source_id, edge.target_id) == (duplicate.id, canonical.id)
        assert edge.id == alias_edge_id(duplicate.id, canonical.id)
        assert edge.weight == ALIAS_EDGE_WEIGHT
        assert edge.metadata == {"_dedup": True}

    @pytest.mark.asyncio
    async def test_preexisting_random_id_row_is_not_duplicated(self) -> None:
        # Brains already hold alias rows with random UUID ids. Recognising them
        # is what stops the 61x backlog from growing further.
        canonical = _anchor("deployment notes")
        duplicate = _anchor("deployment notes (copy)")
        legacy = _alias(duplicate.id, canonical.id)
        storage = FakeAnchorStorage([canonical, duplicate], [legacy])

        await ConsolidationEngine(storage)._dedup(ConsolidationReport(), dry_run=False)

        assert storage.synapses == [legacy]

    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self) -> None:
        storage = FakeAnchorStorage([_anchor("a"), _anchor("b")])
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=True)

        assert storage.synapses == []
        assert report.duplicates_found == 1

    @pytest.mark.asyncio
    async def test_pass_scales_with_runs_not_pairs(self) -> None:
        # One ledger preload per run, no per-pair probe: 20 duplicates of the
        # same anchor must cost exactly one read.
        anchors = [_anchor(f"note {i}") for i in range(21)]
        storage = FakeAnchorStorage(anchors)

        await ConsolidationEngine(storage)._dedup(ConsolidationReport(), dry_run=False)

        assert len(storage.get_calls) == 1
        assert len(storage.synapses) == 20


class TestCreateAnchorStepCallSite:
    @staticmethod
    def _context() -> PipelineContext:
        return PipelineContext(
            content="deployment notes",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            metadata={},
            tags=set(),
            language="en",
        )

    @pytest.mark.asyncio
    async def test_alias_edge_id_is_derived_from_the_pair(self) -> None:
        anchor = _anchor("deployment notes")
        ctx = self._context()
        ctx.effective_metadata["_dedup_reused_anchor"] = anchor
        storage = FakeEncodeStorage()

        result = await CreateAnchorStep().execute(ctx, storage, None)

        assert result.anchor_neuron is not None
        assert len(storage.synapses) == 1
        edge = storage.synapses[0]
        assert edge.type is SynapseType.ALIAS
        assert edge.source_id == result.anchor_neuron.id
        assert edge.target_id == anchor.id
        assert edge.id == alias_edge_id(edge.source_id, edge.target_id)
        assert edge.weight == ALIAS_EDGE_WEIGHT
        assert edge.metadata == {"_dedup": True}
        assert ctx.synapses_created == [edge]

    @pytest.mark.asyncio
    async def test_encode_path_adds_no_read_round_trip(self) -> None:
        # The alias neuron is minted in this step, so no edge can already leave
        # it — probing for one would be a guaranteed-miss query on every dedup
        # hit during encode.
        ctx = self._context()
        ctx.effective_metadata["_dedup_reused_anchor"] = _anchor("deployment notes")
        storage = FakeEncodeStorage()

        await CreateAnchorStep().execute(ctx, storage, None)

        assert storage.get_calls == []

    @pytest.mark.asyncio
    async def test_failed_alias_write_is_logged_and_does_not_break_encode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Encoding must survive a failed alias edge — but not silently, or the
        # alias neuron is stranded with no link to its canonical anchor.
        ctx = self._context()
        ctx.effective_metadata["_dedup_reused_anchor"] = _anchor("deployment notes")
        storage = FakeEncodeStorage()
        storage.fail_writes = True

        with caplog.at_level(logging.WARNING):
            result = await CreateAnchorStep().execute(ctx, storage, None)

        assert result.anchor_neuron is not None
        assert ctx.synapses_created == []
        assert storage.synapses == []
        assert any("stranded" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Report accounting
#
# Until this release the dedup line printed a census and a created count, and
# every non-creation collapsed into the same silence: already linked, probe
# blew up, write blew up. These tests pin the distinction.
# ---------------------------------------------------------------------------


def _invariant_holds(report: ConsolidationReport) -> bool:
    """duplicates_found must be fully explained by the outcome counters."""
    return report.duplicates_found == (
        report.new_alias_links
        + report.alias_links_existing
        + int(report.extra.get("alias_checks_failed", 0))
        + int(report.extra.get("alias_writes_failed", 0))
        + int(report.extra.get("alias_pairs_skipped_invalid", 0))
    )


class NoLedgerStorage(FakeAnchorStorage):
    """Anchor storage whose alias slice always reads empty.

    Lets a test drive the write path with an empty-but-complete ledger, which
    is how a lost write race is reproduced: the pair looks absent, then
    ``add_synapse`` reports it is already there.
    """

    async def get_synapses(self, *args: object, **kwargs: object) -> list[Synapse]:
        await super().get_synapses(*args, **kwargs)  # type: ignore[arg-type]
        return []


class RaceLosingStorage(NoLedgerStorage):
    async def add_synapse(self, synapse: Synapse) -> str:
        raise ValueError(f"Synapse {synapse.id} already exists")


class TestDedupReportAccounting:
    @pytest.mark.asyncio
    async def test_a_new_duplicate_pair_is_reported_as_work_done(self) -> None:
        # The regression this whole unit exists for: a census of 1 with a link
        # actually written must NOT print "new alias links: 0".
        storage = FakeAnchorStorage([_anchor("deployment notes"), _anchor("deployment notes copy")])
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.duplicates_found == 1
        assert report.new_alias_links == 1
        assert report.alias_links_existing == 0
        assert "alias_checks_failed" not in report.extra
        assert "alias_writes_failed" not in report.extra
        assert _invariant_holds(report)

    @pytest.mark.asyncio
    async def test_an_already_linked_pair_counts_as_existing_not_as_work(self) -> None:
        canonical = _anchor("deployment notes")
        duplicate = _anchor("deployment notes copy")
        storage = FakeAnchorStorage([canonical, duplicate], [_alias(duplicate.id, canonical.id)])
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.duplicates_found == 1
        assert report.new_alias_links == 0
        assert report.alias_links_existing == 1
        assert _invariant_holds(report)

    @pytest.mark.asyncio
    async def test_a_lost_write_race_counts_as_existing_never_as_failure(self) -> None:
        storage = RaceLosingStorage([_anchor("a"), _anchor("b")])
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.duplicates_found == 1
        assert report.new_alias_links == 0
        assert report.alias_links_existing == 1
        assert "alias_writes_failed" not in report.extra
        assert _invariant_holds(report)

    @pytest.mark.asyncio
    async def test_unreadable_backend_is_counted_not_hidden(self) -> None:
        # Ledger preload fails, then every per-pair probe fails. Before this
        # release that produced "0 new alias links" and no other trace.
        storage = FakeAnchorStorage([_anchor("a"), _anchor("b"), _anchor("c")])
        storage.fail_reads = True
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.duplicates_found == 2
        assert report.new_alias_links == 0
        assert report.alias_links_existing == 0
        assert report.extra["alias_checks_failed"] == 2
        assert report.extra["alias_ledger_load_failed"] is True
        assert storage.synapses == []
        assert _invariant_holds(report)

    @pytest.mark.asyncio
    async def test_failed_writes_are_counted_not_hidden(self) -> None:
        storage = FakeAnchorStorage([_anchor("a"), _anchor("b")])
        storage.fail_writes = True
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.duplicates_found == 1
        assert report.new_alias_links == 0
        assert report.extra["alias_writes_failed"] == 1
        assert storage.synapses == []
        assert _invariant_holds(report)

    @pytest.mark.asyncio
    async def test_ledger_state_is_reported(self) -> None:
        storage = FakeAnchorStorage([_anchor("a"), _anchor("b")])
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.extra["alias_ledger_complete"] is True
        assert report.extra["alias_ledger_pairs"] == 0
        assert "alias_ledger_load_failed" not in report.extra

    @pytest.mark.asyncio
    async def test_anchor_census_reports_its_input(self) -> None:
        storage = FakeAnchorStorage([_anchor("a"), _anchor("b"), _anchor("c")])
        report = ConsolidationReport()

        await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.extra["dedup_anchors_total"] == 3
        assert report.extra["dedup_anchors_scanned"] == 3
        assert "dedup_anchors_truncated" not in report.extra

    @pytest.mark.asyncio
    async def test_truncated_census_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A silent cap turns "duplicates in my brain" into "duplicates among the
        # first N anchors in scan order" without telling anyone.
        storage = FakeAnchorStorage([_anchor(f"note {i}") for i in range(7)])
        report = ConsolidationReport()
        engine = ConsolidationEngine(storage, config=ConsolidationConfig(dedup_max_anchors=5))

        await engine._dedup(report, dry_run=False)

        assert report.extra["dedup_anchors_total"] == 7
        assert report.extra["dedup_anchors_scanned"] == 5
        assert report.extra["dedup_anchors_truncated"] is True
        assert report.duplicates_found == 4
        assert "census truncated at anchor cap" in report.summary()

    @pytest.mark.asyncio
    async def test_routine_truncation_does_not_raise_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Outgrowing the cap is a steady state for a big brain. Warning about it
        # every run trains operators to ignore dedup warnings, which buries the
        # failures this unit exists to surface.
        storage = FakeAnchorStorage([_anchor(f"note {i}") for i in range(7)])
        engine = ConsolidationEngine(storage, config=ConsolidationConfig(dedup_max_anchors=5))

        with caplog.at_level(logging.INFO):
            await engine._dedup(ConsolidationReport(), dry_run=False)

        assert any("census truncated" in r.message for r in caplog.records)
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_a_degraded_pass_warns_once_not_once_per_pair(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # One root cause (backend down) must not print one traceback per pair.
        storage = FakeAnchorStorage([_anchor(f"note {i}") for i in range(6)])
        storage.fail_reads = True
        report = ConsolidationReport()

        with caplog.at_level(logging.WARNING):
            await ConsolidationEngine(storage)._dedup(report, dry_run=False)

        assert report.extra["alias_checks_failed"] == 5
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 2  # ledger preload failed + one aggregate for the pass
        aggregate = [r for r in warnings if "Dedup alias linking degraded" in r.message]
        assert len(aggregate) == 1
        assert aggregate[0].exc_info is not None

    @pytest.mark.asyncio
    async def test_dry_run_counts_the_census_and_writes_nothing(self) -> None:
        storage = FakeAnchorStorage([_anchor("a"), _anchor("b")])
        report = ConsolidationReport(dry_run=True)

        await ConsolidationEngine(storage)._dedup(report, dry_run=True)

        assert report.duplicates_found == 1
        assert report.new_alias_links == 0
        assert report.alias_links_existing == 0
        assert storage.synapses == []
        assert "links not checked in dry run" in report.summary()


class TestAliasLinkSummaryLine:
    def test_healthy_run_shows_no_failure_text(self) -> None:
        report = ConsolidationReport(
            duplicates_found=955, new_alias_links=0, alias_links_existing=955
        )

        line = report._alias_link_line()

        assert line == "955 pairs (census), 0 new links, 955 already linked"
        assert "FAILED" not in line

    def test_failures_are_spelled_out(self) -> None:
        report = ConsolidationReport(duplicates_found=10, new_alias_links=4, alias_links_existing=3)
        report.extra["alias_checks_failed"] = 2
        report.extra["alias_writes_failed"] = 1

        line = report._alias_link_line()

        assert "4 new links, 3 already linked" in line
        assert "2 checks FAILED (state unknown)" in line
        assert "1 writes FAILED" in line

    def test_dry_run_does_not_claim_links_were_checked(self) -> None:
        report = ConsolidationReport(duplicates_found=12, dry_run=True)

        assert report._alias_link_line() == ("12 pairs (census only; links not checked in dry run)")
