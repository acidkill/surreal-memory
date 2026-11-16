"""Legacy pending checkpoint chunking through the optimized prune engine."""

from __future__ import annotations

from copy import deepcopy

import pytest

from surreal_memory.engine.consolidation import ConsolidationReport, _encode_prune_synapse_cursor
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from tests.unit.test_consolidation_prune_resumable import (
    REFERENCE_TIME,
    _engine,
    _Progress,
    _PruneStorage,
    _weak_synapse,
)


class ChunkProgress(_Progress):
    def __init__(self, ids, *, pause=False):
        super().__init__()
        self.owner_token = "synthetic-owner"  # noqa: S105
        self.state.update(
            run_id="synthetic-run",
            completed_strategies=[
                "dedup",
                "learn_habits",
                "process_reasoning_traces",
                "process_tool_events",
            ],
        )
        self.cursor = _encode_prune_synapse_cursor(_weak_synapse())
        self.state["strategy_states"]["prune"] = {
            "phase": "synapse_pending",
            "cursor": self.cursor,
            "pending": list(ids),
            "counters": {
                "synapses_pruned": 17,
                "neurons_pruned": 3,
                "synapse_pages": 1,
                "neuron_pages": 0,
            },
        }
        self.pause = pause
        self.work = []

    async def checkpoint(
        self, strategy, phase, *, cursor=None, pending=None, counters=None, work=None
    ):
        self.work.append(deepcopy(work))
        await super().checkpoint(strategy, phase, cursor=cursor, pending=pending, counters=counters)
        if self.pause and phase == "synapse_pending" and len(pending or []) < 65:
            self.pause = False
            raise ConsolidationPausedError("synthetic pause after confirmed chunk")


class MissingStorage(_PruneStorage):
    def __init__(self):
        super().__init__()
        self.synapses = {}
        self.decisions = []
        self.repairs = []

    async def prune_synapses_if_unchanged(self, expected, *, pending_ids, operation_id, **kwargs):
        self.decisions.append((list(pending_ids), operation_id))
        return dict.fromkeys(pending_ids, "missing")

    async def repair_prune_synapse_refs_if_absent(self, ids, **kwargs):
        self.repairs.append(list(ids))
        return 0


async def test_legacy_pending_pause_persists_confirmed_chunk_before_resume():
    ids = [f"missing-{index:03}" for index in range(65)]
    storage, progress = MissingStorage(), ChunkProgress(ids, pause=True)
    with pytest.raises(ConsolidationPausedError):
        await _engine(storage, progress)._prune(ConsolidationReport(), REFERENCE_TIME, False)
    state = progress.strategy_state("prune")
    assert storage.repairs == [ids[:32]]
    assert state["phase"] == "synapse_pending" and state["pending"] == ids[32:]
    assert state["cursor"] == progress.cursor
    assert state["counters"]["synapses_pruned"] == 49
    assert state["counters"]["prune_legacy_missing"] == 32
    assert state["counters"]["synapse_pages"] == 1
    assert progress.state["completed_strategies"] == [
        "dedup",
        "learn_habits",
        "process_reasoning_traces",
        "process_tool_events",
    ]

    report = ConsolidationReport()
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    assert storage.repairs == [ids[:32], ids[32:64], ids[64:]]
    assert report.synapses_pruned == 82 and report.neurons_pruned == 3
    final = progress.strategy_state("prune")
    assert final["counters"]["prune_legacy_missing"] == 65
    assert final["counters"]["synapse_pages"] == 1
    assert progress.state["completed_strategies"] == [
        "dedup",
        "learn_habits",
        "process_reasoning_traces",
        "process_tool_events",
    ]
    uninterrupted_storage, uninterrupted_progress = MissingStorage(), ChunkProgress(ids)
    uninterrupted_report = ConsolidationReport()
    await _engine(uninterrupted_storage, uninterrupted_progress)._prune(
        uninterrupted_report, REFERENCE_TIME, False
    )
    assert uninterrupted_storage.repairs == storage.repairs
    assert uninterrupted_report.synapses_pruned == report.synapses_pruned
    assert uninterrupted_progress.strategy_state("prune")["counters"] == final["counters"]


async def test_journal_pending_keeps_single_full_operation_id():
    import hashlib
    import json

    ids = [f"missing-{index:03}" for index in range(65)]
    storage, progress = MissingStorage(), ChunkProgress(ids)
    progress.strategy_state("prune")["counters"]["prune_atomic_journal"] = 1
    report = ConsolidationReport()
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    expected_id = hashlib.sha256(
        json.dumps(["synthetic-run", progress.cursor, sorted(ids)]).encode()
    ).hexdigest()
    assert storage.decisions == [(ids, expected_id)]
    assert storage.repairs == [] and report.synapses_pruned == 17


async def test_protected_and_changed_decisions_advance_work_without_fake_deletions():
    class ProtectedStorage(MissingStorage):
        async def prune_synapses_if_unchanged(
            self, expected, *, pending_ids, operation_id, **kwargs
        ):
            self.decisions.append((list(pending_ids), operation_id))
            return {sid: ("protected" if i % 2 else "changed") for i, sid in enumerate(pending_ids)}

    ids = [f"candidate-{index:03}" for index in range(65)]
    storage, progress = ProtectedStorage(), ChunkProgress(ids, pause=True)
    with pytest.raises(ConsolidationPausedError):
        await _engine(storage, progress)._prune(ConsolidationReport(), REFERENCE_TIME, False)
    state = progress.strategy_state("prune")
    assert state["pending"] == ids[32:] and state["cursor"] == progress.cursor
    assert state["counters"]["synapses_pruned"] == 17
    assert state["counters"].get("prune_legacy_missing", 0) == 0
    assert progress.work[-1] is not None
    assert storage.repairs == []
    report = ConsolidationReport()
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    assert report.synapses_pruned == 17
    assert [chunk for chunk, _ in storage.decisions] == [ids[:32], ids[32:64], ids[64:]]
    assert all(not chunk or len(chunk) <= 32 for chunk, _ in storage.decisions)


async def test_lost_chunk_response_cannot_publish_or_replay_confirmed_prior_chunk():
    class InterruptedStorage(MissingStorage):
        def __init__(self):
            super().__init__()
            self.fail = True

        async def repair_prune_synapse_refs_if_absent(self, ids, **kwargs):
            self.repairs.append(list(ids))
            if len(self.repairs) == 2 and self.fail:
                self.fail = False
                raise ConnectionError("synthetic unconfirmed reply")
            return 0

    ids = [f"missing-{index:03}" for index in range(65)]
    storage, progress = InterruptedStorage(), ChunkProgress(ids)
    with pytest.raises(ConnectionError):
        await _engine(storage, progress)._prune(ConsolidationReport(), REFERENCE_TIME, False)
    state = deepcopy(progress.strategy_state("prune"))
    assert state["pending"] == ids[32:] and state["cursor"] == progress.cursor
    assert state["counters"]["synapses_pruned"] == 49
    assert state["counters"]["prune_legacy_missing"] == 32
    assert storage.repairs == [ids[:32], ids[32:64]]
    report = ConsolidationReport()
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    assert storage.repairs == [ids[:32], ids[32:64], ids[32:64], ids[64:]]
    assert report.synapses_pruned == 82


async def test_zero_deletion_chunks_advance_real_progress_watermark_after_restart(monkeypatch):
    from surreal_memory.engine.consolidation_progress import ConsolidationProgressSession
    from tests.unit.test_consolidation_progress import FakeProgressStorage

    class Storage(MissingStorage, FakeProgressStorage):
        def __init__(self):
            MissingStorage.__init__(self)
            FakeProgressStorage.__init__(self)

        async def prune_synapses_if_unchanged(
            self, expected, *, pending_ids, operation_id, **kwargs
        ):
            self.decisions.append((list(pending_ids), operation_id))
            return dict.fromkeys(pending_ids, "protected")

    ids = [f"candidate-{index:03}" for index in range(65)]
    storage = Storage()
    session = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert session is not None
    cursor = _encode_prune_synapse_cursor(_weak_synapse())
    await session.checkpoint(
        "prune",
        "synapse_pending",
        cursor=cursor,
        pending=ids,
        counters={"synapses_pruned": 17, "synapse_pages": 1},
    )
    before_seq = session.state.get("work_progress_seq", 0)
    original_checkpoint = session.checkpoint

    async def pause_after_chunk(strategy, phase, **kwargs):
        await original_checkpoint(strategy, phase, **kwargs)
        if phase == "synapse_pending" and len(kwargs.get("pending") or []) == 33:
            raise ConsolidationPausedError("pause after committed protected decisions")

    monkeypatch.setattr(session, "checkpoint", pause_after_chunk)
    with pytest.raises(ConsolidationPausedError):
        await _engine(storage, session)._prune(ConsolidationReport(), REFERENCE_TIME, False)
    first = deepcopy(session.strategy_state("prune"))
    assert first["pending"] == ids[32:] and first["counters"]["synapses_pruned"] == 17
    assert session.state["work_progress_seq"] > before_seq
    watermark = first["work_progress"]["prune_legacy_pending"]
    assert watermark["position"] == 32
    await session.close()
    resumed = await ConsolidationProgressSession.open(storage, ["prune"], "same", REFERENCE_TIME)
    assert resumed is not None
    try:
        report = ConsolidationReport()
        await _engine(storage, resumed)._prune(report, REFERENCE_TIME, False)
        assert report.synapses_pruned == 17
        assert (
            resumed.strategy_state("prune")["work_progress"]["prune_legacy_pending"]["position"]
            == 65
        )
        assert [chunk for chunk, _ in storage.decisions] == [ids[:32], ids[32:64], ids[64:]]
    finally:
        await resumed.close()
