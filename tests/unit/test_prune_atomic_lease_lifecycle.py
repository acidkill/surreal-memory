"""Engine-to-atomic-SQL lease lifecycle with a deterministic transport seam."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.storage.surrealdb.prune import _PRUNE_TRANSACTION, SurrealDBPruneMixin
from tests.unit.test_consolidation_progress import FakeProgressStorage
from tests.unit.test_consolidation_prune_resumable import (
    REFERENCE_TIME,
    _PruneStorage,
    _weak_synapse,
)


class AtomicTransportStorage(SurrealDBPruneMixin, _PruneStorage, FakeProgressStorage):
    get_synapses_by_ids = _PruneStorage.get_synapses_by_ids

    def __init__(self):
        _PruneStorage.__init__(self)
        FakeProgressStorage.__init__(self)
        self.synapses = {
            f"edge-{i:03}": replace(_weak_synapse(), id=f"edge-{i:03}", target_id=f"target-{i:03}")
            for i in range(65)
        }
        self._change_seq = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.sql_batches = []

    def _get_brain_id(self):
        return self.current_brain_id

    def _ensure_conn(self):
        return self

    async def _ensure_conn_on_current_loop(self):
        pass

    async def _query(self, sql, **params):
        if "idx_fiber_synapses" in sql:
            return [[] for key in params if key.startswith("sid_")]
        assert "change_log" in sql
        return []

    async def query_raw(self, sql, params):
        assert sql == _PRUNE_TRANSACTION
        self.sql_batches.append([item["public_id"] for item in params["items"]])
        if len(self.sql_batches) == 1:
            self.entered.set()
            await self.release.wait()
        return {"result": [{"status": "OK", "result": []}]}


async def test_engine_atomic_transaction_excludes_own_renewal():
    storage = AtomicTransportStorage()
    engine = ConsolidationEngine(storage, ConsolidationConfig(prune_isolated_neurons=False))
    run = asyncio.create_task(
        engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    )
    renewal = None
    try:
        await asyncio.wait_for(storage.entered.wait(), 5)
        session = engine._progress_session
        assert session is not None
        renewal = asyncio.create_task(session._renew_once())
        await asyncio.sleep(0)
        assert not renewal.done(), "heartbeat renewal overlapped the atomic prune transaction"
    finally:
        storage.release.set()
        if renewal is not None:
            await asyncio.gather(renewal, return_exceptions=True)
        result = (await asyncio.gather(run, return_exceptions=True))[0]
    assert renewal is not None and renewal.result() is True
    assert not isinstance(result, BaseException)
    assert result.extra["consolidation_status"] == "completed"
    assert [len(batch) for batch in storage.sql_batches] == [32, 32, 1]
    assert storage.progress is not None
    assert storage.progress["status"] == "completed"


@pytest.mark.parametrize("status", ["failed", "lease_lost"])
async def test_unsafe_outcome_emitted_before_release_and_after_diagnostic_skipped(
    monkeypatch, status
):
    from surreal_memory.engine.consolidation_delta import run_with_delta
    from surreal_memory.engine.diagnostics import DiagnosticsEngine

    storage = AtomicTransportStorage()
    snapshots = []
    outcomes = []

    async def diagnostic(self, brain_id):
        snapshots.append(brain_id)
        raise RuntimeError("synthetic unavailable diagnostic")

    async def rejected_rpc(sql, params):
        assert sql == _PRUNE_TRANSACTION
        if status == "lease_lost":
            return {"result": [{"status": "ERR", "result": "prune lease lost"}]}
        raise RuntimeError("synthetic transport failure")

    def progress(message):
        if message.startswith("Consolidation engine outcome:"):
            outcomes.append((message, storage.lease_owner, storage.progress["status"]))

    monkeypatch.setattr(DiagnosticsEngine, "analyze", diagnostic)
    monkeypatch.setattr(storage, "query_raw", rejected_rpc)
    if status == "failed":

        async def failed_lookup(sql, **params):
            raise RuntimeError("synthetic lookup failure")

        monkeypatch.setattr(storage, "_query", failed_lookup)
    result = await run_with_delta(
        storage,
        "default",
        strategies=[ConsolidationStrategy.PRUNE],
        reference_time=REFERENCE_TIME,
        config=ConsolidationConfig(prune_isolated_neurons=False),
        on_progress=progress,
    )
    assert result.report.extra["consolidation_status"] == status
    assert snapshots == ["default"]
    assert len(outcomes) == 1 and status in outcomes[0][0]
    assert outcomes[0][1] is not None
    assert storage.lease_owner is None
    assert storage.progress is not None
    if status == "lease_lost":
        assert storage.progress["status"] == "running"
        pending = storage.progress["strategy_states"]["prune"]
        assert pending["phase"] == "synapse_pending" and len(pending["pending"]) == 65


async def test_legacy_pending_transitions_to_atomic_batches_in_one_engine_run(monkeypatch):
    storage = AtomicTransportStorage()
    legacy = [f"legacy-missing-{index:03}" for index in range(33)]
    events = []
    engine = ConsolidationEngine(storage, ConsolidationConfig(prune_isolated_neurons=False))
    original_strategy = engine._run_strategy

    async def seed_legacy(strategy, report, reference_time, dry_run):
        session = engine._progress_session
        assert session is not None
        await session.checkpoint(
            "prune",
            "synapse_pending",
            pending=legacy,
            counters={"synapses_pruned": 0, "synapse_pages": 1},
        )
        return await original_strategy(strategy, report, reference_time, dry_run)

    async def transport(sql, params):
        if "-- bounded legacy prune repair" in sql:
            events.append(("legacy", len(params["candidates"])))
            return {
                "result": [
                    {
                        "status": "OK",
                        "result": [
                            {
                                "updated_ids": [],
                                "has_more": False,
                                "next_cursor": None,
                                "scanned": 0,
                            }
                        ],
                    }
                ]
            }
        assert sql == _PRUNE_TRANSACTION
        events.append(("atomic", len(params["items"])))
        return {"result": [{"status": "OK", "result": []}]}

    monkeypatch.setattr(engine, "_run_strategy", seed_legacy)
    monkeypatch.setattr(storage, "query_raw", transport)
    report = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert report.extra["consolidation_status"] == "completed"
    assert report.synapses_pruned == 33
    assert events == [
        ("atomic", 32),
        ("legacy", 32),
        ("atomic", 1),
        ("legacy", 1),
        ("atomic", 32),
        ("atomic", 32),
        ("atomic", 1),
    ]
    assert storage.progress is not None and storage.progress["status"] == "completed"
    assert storage.lease_owner is None


async def test_real_semantic_cleanup_mutation_excludes_own_renewal(monkeypatch):
    from surreal_memory.storage.surrealdb.semantic_discovery_state import (
        SurrealDBSemanticDiscoveryStateMixin,
    )

    class CleanupStorage(SurrealDBSemanticDiscoveryStateMixin, FakeProgressStorage):
        def __init__(self):
            FakeProgressStorage.__init__(self)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        def _get_brain_id(self):
            return "default"

        async def get_consolidation_run_ids(self, **kwargs):
            return []

        async def _query(self, sql, **kwargs):
            if "UPDATE type::record('consolidation_lease'" in sql:
                self.entered.set()
                await self.release.wait()
                return [{"deleted": 0}]
            return []

    storage = CleanupStorage()
    engine = ConsolidationEngine(storage)

    async def strategy(*args):
        return None

    monkeypatch.setattr(engine, "_run_strategy", strategy)
    task = asyncio.create_task(engine.run([ConsolidationStrategy.SEMANTIC_LINK]))
    renewal = None
    try:
        await asyncio.wait_for(storage.entered.wait(), 5)
        session = engine._progress_session
        assert session is not None
        renewal = asyncio.create_task(session._renew_once())
        await asyncio.sleep(0)
        assert not renewal.done()
    finally:
        storage.release.set()
        if renewal is not None:
            await asyncio.gather(renewal, return_exceptions=True)
        result = (await asyncio.gather(task, return_exceptions=True))[0]
    assert renewal is not None and renewal.result() is True
    assert not isinstance(result, BaseException)
    assert result.extra["consolidation_status"] == "completed"
    assert "semantic_discovery_state_prune_error" not in result.extra


async def test_cleanup_uncertainty_preserves_completed_checkpoint_and_advisory(monkeypatch):
    from surreal_memory.storage.surrealdb.semantic_discovery_state import (
        SurrealDBSemanticDiscoveryStateMixin,
    )

    class CleanupStorage(SurrealDBSemanticDiscoveryStateMixin, FakeProgressStorage):
        def _get_brain_id(self):
            return "default"

        async def get_consolidation_run_ids(self, **kwargs):
            return []

        async def _query(self, sql, **kwargs):
            if "UPDATE type::record('consolidation_lease'" in sql:
                raise ConnectionError("synthetic unknown cleanup outcome")
            return []

    storage = CleanupStorage()
    engine = ConsolidationEngine(storage)

    async def strategy(*args):
        return None

    monkeypatch.setattr(engine, "_run_strategy", strategy)
    messages = []
    result = await engine.run([ConsolidationStrategy.SEMANTIC_LINK], on_progress=messages.append)
    assert result.extra["consolidation_status"] == "completed"
    assert storage.progress is not None and storage.progress["status"] == "completed"
    assert "semantic_discovery_state_prune_error" in result.extra
    assert not any(message.startswith("Consolidation engine outcome:") for message in messages)
    assert storage.lease_owner is None


async def test_atomic_partial_batch_pause_and_resume_replays_same_operation_once(monkeypatch):
    from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

    class JournalStorage(AtomicTransportStorage):
        def __init__(self):
            super().__init__()
            self.receipts = {}
            self.operations = []

        async def query_raw(self, sql, params):
            assert sql == _PRUNE_TRANSACTION
            self.operations.append(params["operation_id"])
            for item in params["items"]:
                sid = item["public_id"]
                if item["receipt_id"] not in self.receipts and sid in self.synapses:
                    self.synapses.pop(sid)
                    self.receipts[item["receipt_id"]] = {
                        "entity_id": sid,
                        "payload": {"_prune": {"synapse_id": sid, "outcome": "deleted"}},
                    }
            return {"result": [{"status": "OK", "result": []}]}

        async def _query(self, sql, **params):
            if "idx_fiber_synapses" in sql:
                return [[] for key in params if key.startswith("sid_")]
            assert "change_log" in sql
            return [
                self.receipts[value]
                for key, value in params.items()
                if key.startswith("receipt_") and value in self.receipts
            ]

    storage = JournalStorage()
    engine = ConsolidationEngine(storage, ConsolidationConfig(prune_isolated_neurons=False))
    original_budget = engine._check_progress_budget
    paused = False

    async def budget():
        nonlocal paused
        await original_budget()
        if storage.receipts and not paused:
            paused = True
            raise ConsolidationPausedError("synthetic pause between atomic batches")

    monkeypatch.setattr(engine, "_check_progress_budget", budget)
    first = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    assert first.extra["consolidation_status"] == "paused"
    assert len(storage.receipts) == 32
    assert storage.progress is not None
    run_id = storage.progress["run_id"]
    state = storage.progress["strategy_states"]["prune"]
    assert state["phase"] == "synapse_pending" and len(state["pending"]) == 65
    assert state["counters"].get("synapses_pruned", 0) == 0
    second = await engine.run([ConsolidationStrategy.PRUNE], resume_run_id=run_id)
    assert second.extra["consolidation_status"] == "completed"
    assert second.synapses_pruned == 65 and len(storage.receipts) == 65
    assert len(set(storage.operations)) == 1
    assert storage.progress["run_id"] == run_id and storage.progress["status"] == "completed"
    assert storage.lease_owner is None


@pytest.mark.parametrize("failure", ["unknown_commit", "cancel"])
async def test_atomic_uncertainty_never_advances_pending_checkpoint(monkeypatch, failure):
    storage = AtomicTransportStorage()
    calls = []

    async def uncertain(sql, params):
        assert sql == _PRUNE_TRANSACTION
        calls.append([item["public_id"] for item in params["items"]])
        if failure == "cancel":
            raise asyncio.CancelledError
        raise ConnectionError("synthetic unconfirmed commit")

    monkeypatch.setattr(storage, "query_raw", uncertain)
    engine = ConsolidationEngine(storage, ConsolidationConfig(prune_isolated_neurons=False))
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
    else:
        report = await engine.run([ConsolidationStrategy.PRUNE], reference_time=REFERENCE_TIME)
        assert report.extra["consolidation_status"] == "lease_lost"
    assert len(calls) == 1
    assert storage.progress is not None
    state = storage.progress["strategy_states"]["prune"]
    assert state["phase"] == "synapse_pending" and len(state["pending"]) == 65
    assert state["counters"].get("synapses_pruned", 0) == 0
    assert storage.lease_owner is None
