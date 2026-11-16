"""Full engine and lease path against registered disposable SurrealDB."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from uuid import uuid4

import pytest

from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.storage.surrealdb.prune import _PRUNE_TRANSACTION
from tests.integration.test_prune_atomic_db import clients as clients

pytestmark = pytest.mark.integration


async def test_full_engine_legacy_then_atomic_own_renewal_waits(clients, monkeypatch):
    first, _, edge, policy = clients
    assert await first.release_consolidation_lease(first.brain_id, "owner")
    for _ in range(65):
        await first.add_synapse(replace(edge, id=str(uuid4())))
    legacy_ids = [str(uuid4()) for _ in range(33)]
    config = ConsolidationConfig(prune_isolated_neurons=False)
    engine = ConsolidationEngine(first, config)
    original_strategy = engine._run_strategy
    original_budget = engine._check_progress_budget
    paused = False
    operations = []
    owners = []

    async def pause_after_committed_batch():
        nonlocal paused
        await original_budget()
        if witness.get("response") is not None and not paused:
            paused = True
            raise ConsolidationPausedError("synthetic pause after confirmed atomic batch")

    original_raw = first._ensure_conn().query_raw
    submitted = asyncio.Event()
    witness = {}
    counts = {"legacy": 0, "atomic": 0}

    async def seed_legacy(strategy, report, reference_time, dry_run):
        session = engine._progress_session
        assert session is not None
        await session.checkpoint(
            "prune",
            "synapse_pending",
            pending=legacy_ids,
            counters={"synapses_pruned": 0, "synapse_pages": 1},
        )
        return await original_strategy(strategy, report, reference_time, dry_run)

    async def controlled_sql(sql, params=None, **kwargs):
        if "-- bounded legacy prune repair" in sql:
            counts["legacy"] += 1
        if sql == _PRUNE_TRANSACTION:
            counts["atomic"] += 1
            operations.append(params["operation_id"])
            owners.append(params["owner"])
            if counts["legacy"] and counts["atomic"] == 3:
                marker = "IF array::len($lease) != 1 { THROW 'prune lease lost'; };"
                assert sql.count(marker) == 1
                sql = sql.replace(marker, marker + "\nSLEEP 2s;", 1)
                started = time.monotonic()
                rpc = asyncio.create_task(original_raw(sql, params, **kwargs))
                try:
                    await asyncio.sleep(0.2)
                    assert not rpc.done(), "atomic server RPC ended before renewal"
                    submitted.set()
                    response = await rpc
                    witness["elapsed"] = time.monotonic() - started
                    witness["response"] = response
                    return response
                finally:
                    if not rpc.done():
                        rpc.cancel()
                        await asyncio.gather(rpc, return_exceptions=True)
        return await original_raw(sql, params, **kwargs)

    monkeypatch.setattr(engine, "_run_strategy", seed_legacy)
    monkeypatch.setattr(engine, "_check_progress_budget", pause_after_committed_batch)
    monkeypatch.setattr(first._ensure_conn(), "query_raw", controlled_sql)
    task = asyncio.create_task(
        engine.run([ConsolidationStrategy.PRUNE], reference_time=policy.reference_time)
    )
    renewal = None
    try:
        await asyncio.wait_for(submitted.wait(), 30)
        session = engine._progress_session
        assert session is not None
        renewal = asyncio.create_task(session._renew_once())
        await asyncio.sleep(0.2)
        assert not renewal.done(), "own renewal entered during atomic transaction"
        result = await asyncio.wait_for(task, 50)
        assert await asyncio.wait_for(renewal, 10)
    finally:
        for pending in (task, renewal):
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    assert result.extra["consolidation_status"] == "paused"
    assert witness["elapsed"] >= 1.5
    raw = witness["response"]
    rows = raw.get("result") if isinstance(raw, dict) else raw
    assert isinstance(rows, list) and rows and all(row.get("status") == "OK" for row in rows)
    checkpoint = await first.get_consolidation_progress(first.brain_id)
    assert checkpoint is not None and checkpoint["status"] == "paused"
    run_id = checkpoint["run_id"]
    state = checkpoint["strategy_states"]["prune"]
    assert state["phase"] == "synapse_pending" and len(state["pending"]) == 66
    assert state["counters"]["synapses_pruned"] == 33
    assert state["counters"]["prune_atomic_journal"] == 1
    query = (
        "SELECT entity_id, payload FROM change_log WHERE brain_id = $brain "
        "AND entity_type = 'synapse' AND operation = 'delete' "
        "AND payload._prune.run_id = $run"
    )
    first_receipts = await first._query(query, brain=first.brain_id, run=run_id)
    assert len(first_receipts) == 32
    first_operation = operations[2]
    assert {row["payload"]["_prune"]["operation_id"] for row in first_receipts} == {first_operation}
    resumed = await ConsolidationEngine(first, config).run(
        [ConsolidationStrategy.PRUNE], resume_run_id=run_id
    )
    assert resumed.extra["consolidation_status"] == "completed"
    assert resumed.synapses_pruned == 99
    assert counts["legacy"] >= 2 and counts["atomic"] >= 6
    assert len(set(operations[2:])) == 1 and operations[2] == first_operation
    assert owners[2] != owners[3]
    final_receipts = await first._query(query, brain=first.brain_id, run=run_id)
    assert len(final_receipts) == 66
    assert len({row["entity_id"] for row in final_receipts}) == 66
    final = await first.get_consolidation_progress(first.brain_id)
    assert final is not None and final["run_id"] == run_id and final["status"] == "completed"
    assert await first.get_synapses_by_ids([edge.id]) == []


async def test_real_cli_two_complete_consolidation_cycles(tmp_path):
    import subprocess
    import sys

    from surreal_memory.engine.consolidation import ConsolidationStrategy
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage
    from tests._disposable_surrealdb import load_target, subprocess_env
    from tests.unit._surrealdb_live import ensure_real_surrealdb_sdk

    target = load_target()
    if target is None:
        pytest.skip("requires registered disposable SurrealDB target")
    ensure_real_surrealdb_sdk()
    target = target.child()
    env = subprocess_env(target, tmp_path / "cli-cycle")

    def cli(*args):
        return subprocess.run(  # noqa: S603 - fixed test-only CLI inside verified target wrapper
            [sys.executable, "-m", "tests._disposable_surrealdb", "cli", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )

    seed = await asyncio.to_thread(
        cli, "remember", "Synthetic full cycle marker", "--type", "fact", "--json"
    )
    assert seed.returncode == 0, (seed.stdout, seed.stderr)
    storage = SurrealDBStorage(**target.storage_kwargs(), database="tests")
    await storage.initialize()
    try:
        run_ids = []
        expected = {s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL}
        assert len(expected) == 20
        for _ in range(2):
            result = await asyncio.to_thread(cli, "consolidate", "--brain", "default")
            assert result.returncode == 0, (result.stdout, result.stderr)
            checkpoint = await storage.get_consolidation_progress("default")
            assert checkpoint is not None and checkpoint["status"] == "completed"
            assert set(checkpoint["requested_strategies"]) == expected
            assert set(checkpoint["completed_strategies"]) == expected
            run_ids.append(checkpoint["run_id"])
        assert run_ids[0] != run_ids[1]
    finally:
        try:
            await storage._query_response("REMOVE DATABASE tests")
        finally:
            await storage.close()


async def test_engine_completed_semantic_cleanup_excludes_own_renewal(clients, monkeypatch):
    from datetime import timedelta

    from surreal_memory.engine.consolidation_progress import ConsolidationProgressSession
    from surreal_memory.utils.timeutils import utcnow

    first, _, _, _ = clients
    assert await first.release_consolidation_lease(first.brain_id, "owner")
    prior = await ConsolidationProgressSession.open(first, ["semantic_link"], "synthetic", utcnow())
    assert prior is not None
    prior_run = prior.state["run_id"]
    await prior.complete_strategy("semantic_link")
    await prior.complete()
    await prior.close()
    state_id = "synthetic-old-" + uuid4().hex
    await first.save_semantic_discovery_state(
        state_id,
        1,
        {
            "brain_id": first.brain_id,
            "run_id": prior_run,
            "owner_token": "synthetic-old-owner",
            "source_token": "synthetic-old-source",
        },
    )
    await first._query(
        "UPDATE semantic_discovery_state SET created_at = $created_at "
        "WHERE state_id = $state_id AND revision = 1",
        created_at=utcnow() - timedelta(days=30),
        state_id=state_id,
    )
    before = await first._query(
        "SELECT VALUE state_id FROM semantic_discovery_state WHERE state_id = $state_id",
        state_id=state_id,
    )
    assert before == [state_id]
    engine = ConsolidationEngine(first)
    original_query = first._query
    submitted = asyncio.Event()
    witness = {}

    async def controlled_query(sql, **params):
        marker = "IF array::len($lease) != 1 { THROW 'discovery prune lease lost'; }; "
        if marker not in sql:
            return await original_query(sql, **params)
        assert sql.count(marker) == 1
        sql = sql.replace(marker, marker + "SLEEP 2s; ", 1)
        started = time.monotonic()
        rpc = asyncio.create_task(original_query(sql, **params))
        try:
            await asyncio.sleep(0.2)
            assert not rpc.done()
            submitted.set()
            result = await rpc
            witness["elapsed"] = time.monotonic() - started
            return result
        finally:
            if not rpc.done():
                rpc.cancel()
                await asyncio.gather(rpc, return_exceptions=True)

    monkeypatch.setattr(first, "_query", controlled_query)
    task = asyncio.create_task(engine.run([ConsolidationStrategy.SEMANTIC_LINK]))
    renewal = None
    try:
        await asyncio.wait_for(submitted.wait(), 40)
        session = engine._progress_session
        assert session is not None
        renewal = asyncio.create_task(session._renew_once())
        await asyncio.sleep(0.2)
        assert not renewal.done(), "own renewal entered during cleanup mutation"
        result = await asyncio.wait_for(task, 50)
        assert await asyncio.wait_for(renewal, 10)
    finally:
        for pending in (task, renewal):
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    assert witness["elapsed"] >= 1.5
    assert result.extra["consolidation_status"] == "completed"
    assert result.semantic_discovery_states_pruned >= 1
    assert "semantic_discovery_state_prune_error" not in result.extra
    remaining = await first._query(
        "SELECT VALUE state_id FROM semantic_discovery_state WHERE state_id = $state_id",
        state_id=state_id,
    )
    assert remaining == []
