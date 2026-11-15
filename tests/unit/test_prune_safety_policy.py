"""Conservative prune regressions through the real optimized engine."""

from dataclasses import replace
from datetime import timedelta

import pytest

from surreal_memory.engine.consolidation import ConsolidationReport
from tests.unit.test_consolidation_prune_resumable import (
    REFERENCE_TIME,
    _engine,
    _Progress,
    _PruneStorage,
    _weak_synapse,
)

_OWNER = "synthetic-prune-owner"


@pytest.mark.asyncio
@pytest.mark.parametrize("weight", [0.1, 0.16, 0.5])
async def test_stored_traversable_weight_survives_read_side_decay(weight):
    storage = _PruneStorage()
    edge = replace(
        _weak_synapse(),
        weight=weight,
        created_at=REFERENCE_TIME - timedelta(days=180),
        metadata={"_dream": True},
    )
    storage.synapses = {
        edge.id: edge,
        "other": replace(edge, id="other", target_id="other", weight=0.5),
    }
    await _engine(storage, _Progress())._prune(ConsolidationReport(), REFERENCE_TIME, False)
    assert edge.id in storage.synapses


@pytest.mark.asyncio
async def test_high_salience_target_is_protected():
    class Storage(_PruneStorage):
        async def get_fiber_neuron_ids_for(self, neuron_ids, *, min_salience=None):
            return {"target-1"} & set(neuron_ids)

    storage = Storage()
    await _engine(storage, _Progress())._prune(ConsolidationReport(), REFERENCE_TIME, False)
    assert "edge-1" in storage.synapses


@pytest.mark.asyncio
async def test_any_fiber_reference_protects_weak_edge():
    class Storage(_PruneStorage):
        async def get_fiber_synapse_ids_for(self, synapse_ids):
            return {"edge-1"} & set(synapse_ids)

    storage = Storage()
    await _engine(storage, _Progress())._prune(ConsolidationReport(), REFERENCE_TIME, False)
    assert "edge-1" in storage.synapses


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["response", "cancel"])
async def test_atomic_response_loss_replays_one_deletion_and_preserves_counters(failure):
    import asyncio

    class Storage(_PruneStorage):
        def __init__(self):
            super().__init__()
            self.receipts = {}
            self.lose_reply = True

        async def prune_synapses_if_unchanged(
            self, expected, *, pending_ids, policy, run_id, operation_id, lease_owner_token
        ):
            assert run_id == "old-run" and lease_owner_token == _OWNER
            key = (operation_id, tuple(pending_ids))
            if key not in self.receipts:
                self.receipts[key] = {
                    sid: "deleted" if self.synapses.pop(sid, None) else "missing"
                    for sid in pending_ids
                }
            if self.lose_reply:
                self.lose_reply = False
                if failure == "cancel":
                    raise asyncio.CancelledError
                raise ConnectionResetError("lost committed response")
            return self.receipts[key]

    storage, progress = Storage(), _Progress()
    progress.state["run_id"] = "old-run"
    progress.owner_token = _OWNER
    progress.state["strategy_states"]["prune"] = {
        "counters": {"synapses_pruned": 17, "neurons_pruned": 3}
    }
    engine = _engine(storage, progress)
    error = asyncio.CancelledError if failure == "cancel" else ConnectionResetError
    with pytest.raises(error):
        await engine._prune(ConsolidationReport(), REFERENCE_TIME, False)
    assert progress.strategy_state("prune")["phase"] == "synapse_pending"
    assert progress.strategy_state("prune")["counters"]["synapses_pruned"] == 17
    assert not storage.synapses
    report = ConsolidationReport()
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    assert report.synapses_pruned == 18
    assert report.neurons_pruned == 3
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    assert report.synapses_pruned == 18
    assert len(storage.receipts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("journal_marker", "expected_count"), [(False, 18), (True, 17)])
async def test_missing_pending_ids_keep_historical_accounting_without_new_fake_receipts(
    journal_marker, expected_count
):
    class Storage(_PruneStorage):
        async def repair_prune_synapse_refs_if_absent(self, ids, *, lease_owner_token):
            assert ids == ["historically-missing"]
            self.removed_fiber_refs.append(set(ids))
            return 1

        async def prune_synapses_if_unchanged(self, expected, *, pending_ids, **kwargs):
            assert expected == []
            return dict.fromkeys(pending_ids, "missing")

    storage, progress = Storage(), _Progress()
    storage.synapses = {}
    progress.state["run_id"] = "old-run"
    progress.owner_token = _OWNER
    counters = {"synapses_pruned": 17, "neurons_pruned": 3}
    if journal_marker:
        counters["prune_atomic_journal"] = 1
    progress.state["strategy_states"]["prune"] = {
        "phase": "synapse_pending",
        "cursor": None,
        "pending": ["historically-missing"],
        "counters": counters,
    }
    report = ConsolidationReport()
    await _engine(storage, progress)._prune(report, REFERENCE_TIME, False)
    assert report.synapses_pruned == expected_count
    assert report.neurons_pruned == 3
    assert storage.removed_fiber_refs == ([{"historically-missing"}] if not journal_marker else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_confirmed_conflict_rereads_and_rechecks_or_cooperatively_pauses(persistent):
    from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
    from surreal_memory.storage.errors import PruneTransactionConflictError

    class Storage(_PruneStorage):
        def __init__(self):
            super().__init__()
            self.attempts = 0
            self.operations = []

        async def prune_synapses_if_unchanged(self, expected, *, operation_id, **kwargs):
            self.attempts += 1
            self.operations.append(operation_id)
            if self.attempts == 1 or persistent:
                self.synapses["edge-1"] = self.synapses["edge-1"].reinforce(
                    delta=0.5, now=REFERENCE_TIME
                )
                raise PruneTransactionConflictError("confirmed server write conflict")
            assert expected[0].weight > 0.1
            assert not kwargs["policy"].eligible(expected[0])
            return {"edge-1": "protected"}

    storage, progress = Storage(), _Progress()
    progress.state["run_id"] = "run-conflict"
    progress.owner_token = _OWNER
    engine = _engine(storage, progress)
    report = ConsolidationReport()
    if persistent:
        with pytest.raises(ConsolidationPausedError, match="contention"):
            await engine._prune(report, REFERENCE_TIME, False)
        assert storage.attempts == 3
        assert progress.strategy_state("prune")["phase"] == "synapse_pending"
    else:
        await engine._prune(report, REFERENCE_TIME, False)
        assert storage.attempts == 2
        assert progress.strategy_state("prune")["phase"] == "synapse_scan"
    assert len(set(storage.operations)) == 1
    assert storage.synapses["edge-1"].weight > 0.1
    assert report.synapses_pruned == 0
