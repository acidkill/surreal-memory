"""Two-client prune contracts against explicit disposable loopback databases.

Run on SurrealDB 3.2.0 and 3.2.4; never inherits ordinary production settings.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationLeaseLostError
from surreal_memory.engine.prune_policy import PrunePolicy
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow
from tests._disposable_surrealdb import load_target
from tests.unit._surrealdb_live import ensure_real_surrealdb_sdk

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def clients(monkeypatch):
    target = load_target()
    if target is None:
        pytest.skip("requires registered disposable SurrealDB target")
    ensure_real_surrealdb_sdk()
    database = "prune_" + uuid4().hex
    options = {**target.child().storage_kwargs(), "database": database}
    first, second = SurrealDBStorage(**options), SurrealDBStorage(**options)
    try:
        await first.initialize()
        await second.initialize()
        brain = Brain.create(name="synthetic-prune")
        await first.save_brain(brain)
        first.set_brain(brain.id)
        second.set_brain(brain.id)
        now = utcnow()
        neurons = [
            Neuron.create(type=NeuronType.CONCEPT, content=name)
            for name in ("left", "right", "other")
        ]
        for neuron in neurons:
            await first.add_neuron(neuron)
        edge = Synapse.create(
            source_id=neurons[0].id,
            target_id=neurons[1].id,
            type=SynapseType.RELATED_TO,
            weight=0.001,
        )
        edge = replace(edge, created_at=now - timedelta(days=90))
        await first.add_synapse(edge)
        assert await first.acquire_consolidation_lease(brain.id, "owner", lease_seconds=30)
        # Read the DB representation: comparisons guard the exact persisted snapshot.
        edge = (await first.get_synapses_by_ids([edge.id]))[0]
        yield first, second, edge, PrunePolicy(now, 0.05, 7.0, 10.0)
    finally:
        try:
            await first._query_response("REMOVE DATABASE " + database)
        finally:
            await second.close()
            await first.close()


async def prune(first, edge, policy, *, operation="op", pending=None):
    return await first.prune_synapses_if_unchanged(
        [edge],
        pending_ids=pending or [edge.id],
        policy=policy,
        run_id="synthetic-run",
        operation_id=operation,
        lease_owner_token="owner",  # noqa: S106 - synthetic lease identity
    )


async def test_atomic_delete_receipt_and_replay(clients):
    first, _, edge, policy = clients
    assert await prune(first, edge, policy) == {edge.id: "deleted"}
    assert not await first.get_synapses_by_ids([edge.id])
    # Replayed lost response returns the prior decision, not an invented missing result.
    assert await prune(first, edge, policy) == {edge.id: "deleted"}
    receipts = await first._query(
        "SELECT * FROM change_log WHERE brain_id = $brain AND operation = 'delete'",
        brain=first.brain_id,
    )
    assert len(receipts) == 1
    assert receipts[0]["payload"]["_prune"]["decision"]["stored_weight"] == edge.weight
    changes = await first._query(
        "SELECT * FROM change_log WHERE brain_id = $brain AND entity_id = $edge AND operation = 'delete'",
        brain=first.brain_id,
        edge=edge.id,
    )
    assert len(changes) == 1


async def test_two_client_reinforcement_rejects_stale_snapshot_and_no_resurrection(clients):
    first, second, edge, policy = clients
    reinforced = edge.reinforce(delta=0.5, now=policy.reference_time)
    await second.update_synapse(reinforced)
    assert await prune(first, edge, policy) == {edge.id: "changed"}
    current = (await first.get_synapses_by_ids([edge.id]))[0]
    assert current.weight == reinforced.weight
    # A stale recall update after a real deletion must not recreate the relation.
    await second.update_synapse(edge)
    assert await prune(first, edge, policy, operation="delete") == {edge.id: "deleted"}
    await second.update_synapse(reinforced)
    await second.update_synapses_batch([reinforced])
    assert not await first.get_synapses_by_ids([edge.id])


@pytest.mark.parametrize(
    "protection", ["reference", "source_salience", "target_salience", "source_pin", "target_pin"]
)
async def test_fiber_changes_committed_before_snapshot_protect(clients, protection):
    first, second, edge, policy = clients
    endpoint = edge.target_id if protection.startswith("target") else edge.source_id
    fiber = Fiber.create(
        neuron_ids={endpoint},
        synapse_ids={edge.id} if protection == "reference" else set(),
        anchor_neuron_id=endpoint,
    )
    fiber = replace(
        fiber, pinned="pin" in protection, salience=0.9 if "salience" in protection else 0.1
    )
    await second.add_fiber(fiber)
    assert await prune(first, edge, policy) == {edge.id: "protected"}
    assert await first.get_synapses_by_ids([edge.id])
    assert await first.get_fiber_synapse_ids_for([edge.id]) == (
        {edge.id} if protection == "reference" else set()
    )


async def test_expired_lease_rolls_back_delete_and_journal(clients):
    first, _, edge, policy = clients
    await first._query(
        "UPDATE consolidation_lease SET expires_at = $expiry WHERE brain_id = $brain",
        expiry=utcnow() - timedelta(seconds=1),
        brain=first.brain_id,
    )
    with pytest.raises(ConsolidationLeaseLostError):
        await prune(first, edge, policy)
    assert await first.get_synapses_by_ids([edge.id])
    assert not await first._query(
        "SELECT * FROM change_log WHERE brain_id = $brain AND operation = 'delete'",
        brain=first.brain_id,
    )


async def test_lost_response_reconciles_without_duplicate_change(clients, monkeypatch):
    first, _, edge, policy = clients
    conn = first._ensure_conn()
    original = conn.query_raw

    async def lose_once(sql, params=None, **kwargs):
        result = await original(sql, params, **kwargs)
        if "BEGIN TRANSACTION" in sql:
            monkeypatch.setattr(conn, "query_raw", original)
            raise ConnectionResetError("synthetic response loss after execution")
        return result

    monkeypatch.setattr(conn, "query_raw", lose_once)
    with pytest.raises(ConnectionResetError):
        await prune(first, edge, policy)
    assert await prune(first, edge, policy) == {edge.id: "deleted"}
    changes = await first._query(
        "SELECT * FROM change_log WHERE brain_id = $brain AND entity_id = $edge AND operation = 'delete'",
        brain=first.brain_id,
        edge=edge.id,
    )
    assert len(changes) == 1


async def test_same_edge_write_during_transaction_never_deletes_reinforcement(clients, monkeypatch):
    first, second, edge, policy = clients
    conn = first._ensure_conn()
    original = conn.query_raw
    started = asyncio.Event()

    async def slow_delete(sql, params=None, **kwargs):
        if "BEGIN TRANSACTION" in sql:
            sql = sql.replace("IF $outcome = 'deleted' {", "IF $outcome = 'deleted' { SLEEP 500ms;")
            started.set()
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(conn, "query_raw", slow_delete)
    task = asyncio.create_task(prune(first, edge, policy))
    await started.wait()
    await asyncio.sleep(0.1)
    await second.update_synapse(edge.reinforce(delta=0.5, now=policy.reference_time))
    try:
        result = await task
        assert result[edge.id] in {"changed", "protected"}
    except RuntimeError as exc:
        assert "conflict" in str(exc).lower() or "transaction" in str(exc).lower()  # noqa: PT017 - race may also return changed
    current = (await first.get_synapses_by_ids([edge.id]))[0]
    assert current.weight > 0.1


async def test_fiber_reference_explain_uses_array_index(clients):
    first, second, edge, _ = clients
    await second.add_fiber(
        Fiber.create(
            neuron_ids={edge.source_id, edge.target_id},
            synapse_ids={edge.id},
            anchor_neuron_id=edge.source_id,
        )
    )
    plan = await first._query_response(
        "SELECT id FROM fiber WITH INDEX idx_fiber_synapses WHERE brain_id = $brain AND synapse_ids CONTAINS $edge LIMIT 1 EXPLAIN",
        brain=first.brain_id,
        edge=edge.id,
    )
    rendered = json.dumps(plan, default=str)
    assert "idx_fiber_synapses" in rendered
    assert "TableScan" not in rendered
    assert "IndexScan" in rendered


@pytest.mark.parametrize("interrupt", ["timeout", "cancel"])
async def test_interrupted_transaction_replay_has_one_deletion_receipt(
    clients, monkeypatch, interrupt
):
    first, _, edge, policy = clients
    conn = first._ensure_conn()
    original = conn.query_raw
    started = asyncio.Event()

    async def slow(sql, params=None, **kwargs):
        if "BEGIN TRANSACTION" in sql:
            sql = sql.replace("IF $outcome = 'deleted' {", "IF $outcome = 'deleted' { SLEEP 250ms;")
            started.set()
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(conn, "query_raw", slow)
    task = asyncio.create_task(prune(first, edge, policy))
    await started.wait()
    if interrupt == "timeout":
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, timeout=0.05)
    else:
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    # Client interruption does not promise server rollback. Reconcile either outcome.
    await asyncio.sleep(0.4)
    monkeypatch.setattr(conn, "query_raw", original)
    assert await prune(first, edge, policy) == {edge.id: "deleted"}
    assert not await first.get_synapses_by_ids([edge.id])
    changes = await first._query(
        "SELECT * FROM change_log WHERE brain_id = $brain AND entity_id = $edge AND operation = 'delete'",
        brain=first.brain_id,
        edge=edge.id,
    )
    assert len(changes) == 1


async def test_fiber_probe_mixed_and_multiple_references_keep_all_results(clients):
    first, second, edge, _ = clients
    other = replace(edge, id=str(uuid4()))
    await second.add_synapse(other)
    for selected in (edge, other):
        await second.add_fiber(
            Fiber.create(
                neuron_ids={selected.source_id, selected.target_id},
                synapse_ids={selected.id},
                anchor_neuron_id=selected.source_id,
            )
        )
    assert await first.get_fiber_synapse_ids_for(["absent", edge.id, other.id]) == {
        edge.id,
        other.id,
    }
    assert await first.get_fiber_synapse_ids_for(["absent-a", "absent-b"]) == set()
    assert await first.get_fiber_synapse_ids_for([edge.id, other.id]) == {edge.id, other.id}


async def test_legacy_absent_edge_repairs_refs_but_recreated_edge_keeps_them(clients):
    first, second, edge, _ = clients
    fiber = Fiber.create(
        neuron_ids={edge.source_id, edge.target_id},
        synapse_ids={edge.id},
        anchor_neuron_id=edge.source_id,
    )
    await second.add_fiber(fiber)
    # An old worker can have committed deletion before its reference cleanup.
    await second.delete_synapse(edge.id)
    await first.repair_prune_synapse_refs_if_absent([edge.id], lease_owner_token="owner")  # noqa: S106
    repaired = await first.get_fiber(fiber.id)
    assert repaired is not None and repaired.synapse_ids == set()
    # A same-ID edge recreated before the repair snapshot must retain its references.
    await second.add_synapse(edge)
    await second.update_fiber(fiber)
    await first.repair_prune_synapse_refs_if_absent([edge.id], lease_owner_token="owner")  # noqa: S106
    preserved = await first.get_fiber(fiber.id)
    assert preserved is not None and preserved.synapse_ids == {edge.id}
    assert await first.get_synapses_by_ids([edge.id])


async def test_real_conflict_rereads_reinforced_edge_and_completes_or_pauses(clients, monkeypatch):
    first, second, edge, policy = clients
    assert await first.release_consolidation_lease(first.brain_id, "owner")
    conn = first._ensure_conn()
    original = conn.query_raw
    started = asyncio.Event()
    slow_once = True

    async def overlap_once(sql, params=None, **kwargs):
        nonlocal slow_once
        if slow_once and "BEGIN TRANSACTION" in sql:
            slow_once = False
            sql = sql.replace("IF $outcome = 'deleted' {", "IF $outcome = 'deleted' { SLEEP 500ms;")
            started.set()
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(conn, "query_raw", overlap_once)
    engine = ConsolidationEngine(
        first,
        ConsolidationConfig(
            prune_isolated_neurons=False, strategy_timeout_seconds=30, total_timeout_seconds=60
        ),
    )
    task = asyncio.create_task(
        engine.run(strategies=[ConsolidationStrategy.PRUNE], reference_time=policy.reference_time)
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    await asyncio.sleep(0.1)
    await second.update_synapse(edge.reinforce(delta=0.5, now=policy.reference_time))
    report = await task
    assert report.extra["consolidation_status"] in {"completed", "paused"}
    assert not report.extra.get("failed_strategies")
    assert report.synapses_pruned == 0
    current = (await first.get_synapses_by_ids([edge.id]))[0]
    assert current.weight > 0.1
    if report.extra["consolidation_status"] == "paused":
        checkpoint = await first.get_consolidation_progress()
        assert checkpoint["strategy_states"]["prune"]["phase"] == "synapse_pending"
        assert checkpoint["strategy_states"]["prune"]["pending"] == [edge.id]
