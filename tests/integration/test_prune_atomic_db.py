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


async def test_atomic_delete_receipt_and_replay(clients, monkeypatch):
    first, _, edge, policy = clients
    # Probe the exact bound snapshot using only field-comparison booleans.
    from surreal_memory.storage.surrealdb.prune import _PRUNE_TRANSACTION, _checked_prune_results

    conn, original = first._ensure_conn(), first._ensure_conn().query_raw

    async def checked_snapshot(sql, params=None, **kwargs):
        if sql == _PRUNE_TRANSACTION:
            probe = """
                LET $item = $items[0];
                LET $rows = (SELECT * FROM type::record('synapse', $item.record_id) WHERE brain_id = $brain_id);
                LET $row = $rows[0];
                RETURN {
                    exists: array::len($rows) = 1,
                    snapshot_object: type::is_object($item.snapshot),
                    source: $row.in = type::record('neuron', $item.source_record),
                    target: $row.out = type::record('neuron', $item.target_record),
                    edge_type: $row.type = $item.snapshot.type,
                    direction: $row.direction = $item.snapshot.direction,
                    weight: $row.weight = $item.snapshot.weight,
                    created_at: $row.created_at = $item.snapshot.created_at,
                    created_type: type::is_datetime($item.snapshot.created_at),
                    last_activated: ($row.last_activated ?? NULL) = ($item.snapshot.last_activated ?? NULL),
                    last_none: ($row.last_activated ?? NULL) = (NONE ?? NULL),
                    last_null: ($row.last_activated ?? NULL) = (NULL ?? NULL),
                    reinforced: $row.reinforced_count = $item.snapshot.reinforced_count,
                    metadata: $row.metadata = $item.snapshot.metadata,
                    eligible: $item.eligible
                };
            """
            results = _checked_prune_results(await original(probe, params, **kwargs))
            flags = results[-1]["result"]
            assert isinstance(flags, dict) and all(value is True for value in flags.values()), flags
            monkeypatch.setattr(conn, "query_raw", original)
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(conn, "query_raw", checked_snapshot)
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
    # A separate never-activated weak edge supplies the deletion control.
    # Updating an old snapshot intentionally does not clear last_activated.
    fresh = replace(edge, id=str(uuid4()))
    await second.add_synapse(fresh)
    fresh = (await first.get_synapses_by_ids([fresh.id]))[0]
    assert await prune(first, fresh, policy, operation="delete") == {fresh.id: "deleted"}
    stale = fresh.reinforce(delta=0.5, now=policy.reference_time)
    await second.update_synapse(stale)
    await second.update_synapses_batch([stale])
    assert not await first.get_synapses_by_ids([fresh.id])
    preserved = (await first.get_synapses_by_ids([edge.id]))[0]
    assert preserved.weight == reinforced.weight


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


async def test_legacy_repair_pages_eight_fibers_once_with_id_only_replies(clients, monkeypatch):
    from surrealdb import RecordID

    from scripts.benchmark_semantic_discovery_prune import _repair_scan_evidence
    from surreal_memory.storage.surrealdb.prune import (
        _PRUNE_REPAIR_SELECTOR,
        _checked_prune_results,
    )

    first, second, edge, _ = clients
    owner = "owner"
    second_missing = str(uuid4())
    fibers = []
    numeric_record_id = None
    for index in range(17):
        fiber = Fiber.create(
            neuron_ids={edge.source_id, edge.target_id},
            synapse_ids={second_missing, "keep-synthetic"} | ({edge.id} if index < 16 else set()),
            anchor_neuron_id=edge.source_id,
            metadata={"synthetic_padding": "x" * 4096},
            fiber_id="1234567890" if index == 0 else None,
        )
        await second.add_fiber(fiber)
        fibers.append(fiber)
        if index == 0:
            # Bare numeric SurQL literals address integer IDs; this fixture stores a string ID.
            numeric_record_id = RecordID("fiber", fiber.id)
            before = await first._query(
                "SELECT id, synapse_ids, metadata FROM $record WHERE brain_id = $brain",
                record=numeric_record_id,
                brain=first.brain_id,
            )
            assert len(before) == 1, "numeric-looking string-ID fixture must exist before repair"
            assert before[0]["id"] == numeric_record_id
            assert set(before[0]["synapse_ids"]) == fiber.synapse_ids
            assert before[0]["metadata"] == fiber.metadata
    target_brain_id = second.brain_id
    foreign_brain = Brain.create(
        name="synthetic-foreign-prune", brain_id="00000000-0000-0000-0000-000000000000"
    )
    assert foreign_brain.id < target_brain_id
    await second.save_brain(foreign_brain)
    foreign_fibers = []
    second.set_brain(foreign_brain.id)
    try:
        for _ in range(10):
            foreign = Fiber.create(
                neuron_ids={edge.source_id, edge.target_id},
                synapse_ids={edge.id, second_missing, "keep-foreign"},
                anchor_neuron_id=edge.source_id,
                metadata={"scope": "foreign", "synthetic_padding": "y" * 4096},
            )
            await second.add_fiber(foreign)
            before_foreign = await second.get_fiber(foreign.id)
            assert before_foreign is not None
            assert before_foreign.synapse_ids == foreign.synapse_ids
            assert before_foreign.metadata == foreign.metadata
            foreign_fibers.append(foreign)
    finally:
        second.set_brain(target_brain_id)
    plan = await first._query_response(
        _PRUNE_REPAIR_SELECTOR + " EXPLAIN FULL", brain_id=first.brain_id, probe_sid=edge.id
    )
    evidence = _repair_scan_evidence(plan)
    assert evidence["indexed_scan_output_bound"], json.dumps(evidence, default=str)
    await second.delete_synapse(edge.id)
    conn, original = first._ensure_conn(), first._ensure_conn().query_raw
    page_counts = []

    async def observe(sql, params=None, **kwargs):
        raw = await original(sql, params, **kwargs)
        if "-- bounded legacy prune repair" in sql:
            results = _checked_prune_results(raw)
            pages = [
                row
                for statement in results
                for row in (
                    statement["result"] if isinstance(statement.get("result"), list) else []
                )
                if isinstance(row, dict) and set(row) == {"updated_ids", "has_more"}
            ]
            assert len(pages) == 1
            page_counts.append(len(pages[0]["updated_ids"]))
            assert page_counts[-1] <= 8
            assert (
                "RETURN AFTER;"
                not in sql.split("LET $updated", 1)[1].split("LET $final_lease", 1)[0]
            )
        return raw

    monkeypatch.setattr(conn, "query_raw", observe)
    budget_checks = 0

    async def budget():
        nonlocal budget_checks
        budget_checks += 1

    assert (
        await first.repair_prune_synapse_refs_if_absent(
            [edge.id, second_missing], lease_owner_token=owner, budget_check=budget
        )
        == 17
    )
    assert page_counts == [8, 8, 1]
    assert budget_checks == 3
    assert (
        await first.repair_prune_synapse_refs_if_absent(
            [edge.id, second_missing], lease_owner_token=owner
        )
        == 0
    )
    assert page_counts == [8, 8, 1, 0, 0]
    for index, fiber in enumerate(fibers):
        if index == 0:
            after = await first._query(
                "SELECT id, synapse_ids, metadata FROM $record WHERE brain_id = $brain",
                record=numeric_record_id,
                brain=first.brain_id,
            )
            assert len(after) == 1
            assert after[0]["id"] == numeric_record_id
            assert set(after[0]["synapse_ids"]) == {"keep-synthetic"}
            assert after[0]["metadata"] == fiber.metadata
        else:
            repaired = await first.get_fiber(fiber.id)
            assert repaired is not None and repaired.synapse_ids == {"keep-synthetic"}
            assert repaired.metadata == fiber.metadata
    changes = await first._query(
        "SELECT entity_id, payload FROM change_log WHERE brain_id = $brain AND entity_type = 'fiber' AND operation = 'update'",
        brain=first.brain_id,
    )
    assert len(changes) == 17
    assert {row["entity_id"] for row in changes} == {fiber.id for fiber in fibers}
    assert all(row.get("payload") is None for row in changes)
    second.set_brain(foreign_brain.id)
    try:
        for foreign in foreign_fibers:
            unchanged = await second.get_fiber(foreign.id)
            assert unchanged is not None
            assert unchanged.synapse_ids == foreign.synapse_ids
            assert unchanged.metadata == foreign.metadata
    finally:
        second.set_brain(target_brain_id)
    foreign_updates = await first._query(
        "SELECT entity_id FROM change_log WHERE brain_id = $brain "
        "AND entity_type = 'fiber' AND operation = 'update'",
        brain=foreign_brain.id,
    )
    assert foreign_updates == []
