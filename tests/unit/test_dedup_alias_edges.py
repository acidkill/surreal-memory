"""Tests for idempotent dedup ALIAS edge creation.

Regression guard for the live-brain finding: 144,565 alias synapse rows
backing only 2,375 distinct (source, target) pairs, because the old
``except ValueError`` guard could never fire against a fresh UUID id.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport
from surreal_memory.engine.dedup.alias_edges import (
    ALIAS_EDGE_WEIGHT,
    AliasEdgeLedger,
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

        created = await ensure_alias_edge(storage, "dup-1", "canon-1")

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

        assert first is not None
        assert second is None
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

        assert result is None
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

        created = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert created is not None
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

        assert result is None
        assert storage.synapses == []

    @pytest.mark.parametrize(("source", "target"), [("", "canon-1"), ("dup-1", "")])
    @pytest.mark.asyncio
    async def test_empty_endpoint_is_rejected(self, source: str, target: str) -> None:
        storage = FakeStorage()

        assert await ensure_alias_edge(storage, source, target) is None
        assert storage.synapses == []

    @pytest.mark.asyncio
    async def test_read_failure_skips_the_write(self) -> None:
        # Failing open would resurrect unbounded growth; a skipped edge is
        # recoverable on the next dedup pass, a duplicate row is not.
        storage = FakeStorage()
        storage.fail_reads = True

        result = await ensure_alias_edge(storage, "dup-1", "canon-1")

        assert result is None
        assert storage.synapses == []

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

        assert result is None
        assert len(storage.synapses) == 1


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

        assert first is not None
        assert second is None
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
