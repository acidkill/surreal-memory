"""Real two-client lease contention controls on registered disposable databases."""

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import pytest

from surreal_memory.storage.errors import PruneTransactionConflictError
from tests.integration.test_prune_atomic_db import clients as clients

pytestmark = pytest.mark.integration


async def test_unprotected_repair_page_and_lease_renewal_conflict(clients, monkeypatch, capsys):
    first, second, _, _ = clients
    owner = "owner"
    assert await second.renew_consolidation_lease(first.brain_id, owner, lease_seconds=180)
    missing = str(uuid4())
    # No-overlap control exercises the same actual repair operation first.
    assert await first.repair_prune_synapse_refs_if_absent([missing], lease_owner_token=owner) == 0

    conn = first._ensure_conn()
    original = conn.query_raw
    submitted = asyncio.Event()
    witness = {}

    async def delayed_page(sql, params=None, **kwargs):
        if "-- bounded legacy prune repair" not in sql:
            return await original(sql, params, **kwargs)
        marker = "IF array::len($lease) != 1 { THROW 'prune lease lost'; };"
        assert sql.count(marker) == 1
        sql = sql.replace(marker, marker + "\nSLEEP 3s;", 1)
        witness["started"] = time.monotonic()
        submitted.set()
        try:
            witness["raw"] = await original(sql, params, **kwargs)
            return witness["raw"]
        finally:
            witness["finished"] = time.monotonic()

    monkeypatch.setattr(conn, "query_raw", delayed_page)
    page = asyncio.create_task(
        first.repair_prune_synapse_refs_if_absent([missing], lease_owner_token=owner)
    )
    try:
        await asyncio.wait_for(submitted.wait(), 10)
        await asyncio.sleep(0.5)
        assert not page.done(), "repair RPC ended before renewal could overlap"
        assert await asyncio.wait_for(
            second.renew_consolidation_lease(first.brain_id, owner, lease_seconds=180), 10
        )
        witness["renewed"] = time.monotonic()
        assert not page.done(), "renewal did not finish while repair RPC was pending"
        with pytest.raises(PruneTransactionConflictError):
            await asyncio.wait_for(page, 10)
    finally:
        if not page.done():
            page.cancel()
            await asyncio.gather(page, return_exceptions=True)

    raw = witness["raw"]
    rows = raw.get("result") if isinstance(raw, dict) else raw
    assert isinstance(rows, list) and rows
    errors = [row for row in rows if isinstance(row, dict) and row.get("status") == "ERR"]
    assert errors and any(
        term in str(row.get("result", "")).lower()
        for row in errors
        for term in ("transaction conflict", "write conflict")
    ), errors
    assert witness["started"] < witness["renewed"] < witness["finished"]
    assert witness["finished"] - witness["started"] >= 2.5, witness
    with capsys.disabled():
        print(
            "SERVER_CONFLICT_WITNESS "
            + json.dumps(
                {
                    "errors": errors,
                    "elapsed": witness["finished"] - witness["started"],
                    "renewed_after": witness["renewed"] - witness["started"],
                },
                default=str,
            )
        )


async def test_guarded_pages_queue_own_renewal_and_resume_between_pages(clients, monkeypatch):
    from surreal_memory.core.fiber import Fiber
    from surreal_memory.engine.consolidation_progress import ConsolidationProgressSession

    first, _, edge, _ = clients
    session = ConsolidationProgressSession(first, first.brain_id, "owner", {}, lease_seconds=180)
    missing = str(uuid4())
    for index in range(17):
        await first.add_fiber(
            Fiber.create(
                neuron_ids={edge.source_id},
                synapse_ids={missing},
                anchor_neuron_id=edge.source_id,
                fiber_id=f"bbbbbbbb-bbbb-4bbb-8bbb-{index:012d}",
            )
        )
    original = first._ensure_conn().query_raw
    submitted = asyncio.Event()
    pages = []

    async def delayed_page(sql, params=None, **kwargs):
        if "-- bounded legacy prune repair" in sql:
            pages.append(time.monotonic())
            if len(pages) == 1:
                marker = "IF array::len($lease) != 1 { THROW 'prune lease lost'; };"
                assert sql.count(marker) == 1
                sql = sql.replace(marker, marker + "\nSLEEP 3s;", 1)
                submitted.set()
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(first._ensure_conn(), "query_raw", delayed_page)
    repair = asyncio.create_task(
        first.repair_prune_synapse_refs_if_absent(
            [missing], lease_owner_token=session.owner_token, page_guard=session.lease_fenced_page
        )
    )
    renewal = None
    try:
        await asyncio.wait_for(submitted.wait(), 10)
        renewal = asyncio.create_task(session._renew_once())
        await asyncio.sleep(0.5)
        assert not repair.done() and not renewal.done()
        assert await asyncio.wait_for(repair, 15) == 17
        assert await asyncio.wait_for(renewal, 10)
    finally:
        for task in (repair, renewal):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert len(pages) >= 3
    assert not session._renew_lock.locked()
    assert not session.lease_lost


async def test_guarded_lost_reply_fences_session_without_replay(clients, monkeypatch):
    from surreal_memory.engine.consolidation_progress import (
        ConsolidationCheckpointUnconfirmedError,
        ConsolidationLeaseLostError,
        ConsolidationProgressSession,
    )

    first, _, _, _ = clients
    session = ConsolidationProgressSession(first, first.brain_id, "owner", {}, lease_seconds=180)
    original = first._ensure_conn().query_raw
    requests = 0

    async def lost_reply(sql, params=None, **kwargs):
        nonlocal requests
        result = await original(sql, params, **kwargs)
        if "-- bounded legacy prune repair" in sql:
            requests += 1
            raise ConnectionError("synthetic lost response after server completion")
        return result

    monkeypatch.setattr(first._ensure_conn(), "query_raw", lost_reply)
    missing = str(uuid4())
    with pytest.raises(ConsolidationCheckpointUnconfirmedError):
        await first.repair_prune_synapse_refs_if_absent(
            [missing], lease_owner_token=session.owner_token, page_guard=session.lease_fenced_page
        )
    assert requests == 1 and session.lease_lost and session._checkpoint_unconfirmed
    assert not session._renew_lock.locked()
    with pytest.raises(ConsolidationLeaseLostError):
        await first.repair_prune_synapse_refs_if_absent(
            [missing], lease_owner_token=session.owner_token, page_guard=session.lease_fenced_page
        )
    assert requests == 1


async def test_guarded_cancellation_blocks_replay_and_checkpoint(clients, monkeypatch):
    from surreal_memory.engine.consolidation_progress import (
        ConsolidationLeaseLostError,
        ConsolidationProgressSession,
    )

    first, second, _, _ = clients
    session = ConsolidationProgressSession(
        first, first.brain_id, "owner", {"marker": "unchanged"}, lease_seconds=180
    )
    checkpoint_before = await second.get_consolidation_progress(first.brain_id)
    original = first._ensure_conn().query_raw
    submitted = asyncio.Event()
    calls = 0

    async def delayed_page(sql, params=None, **kwargs):
        nonlocal calls
        if "-- bounded legacy prune repair" in sql:
            calls += 1
            marker = "IF array::len($lease) != 1 { THROW 'prune lease lost'; };"
            assert sql.count(marker) == 1
            sql = sql.replace(marker, marker + "\nSLEEP 3s;", 1)
            submitted.set()
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(first._ensure_conn(), "query_raw", delayed_page)
    missing = str(uuid4())
    task = asyncio.create_task(
        first.repair_prune_synapse_refs_if_absent(
            [missing], lease_owner_token=session.owner_token, page_guard=session.lease_fenced_page
        )
    )
    try:
        await asyncio.wait_for(submitted.wait(), 10)
        await asyncio.sleep(0.5)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert session.state == {"marker": "unchanged"}
    with pytest.raises(ConsolidationLeaseLostError):
        await session.checkpoint("prune", "synapse_pending", cursor="changed")
    assert await second.get_consolidation_progress(first.brain_id) == checkpoint_before
    assert session.lease_lost and session._checkpoint_unconfirmed
    assert not session._renew_lock.locked()
    with pytest.raises(ConsolidationLeaseLostError):
        await first.repair_prune_synapse_refs_if_absent(
            [missing], lease_owner_token=session.owner_token, page_guard=session.lease_fenced_page
        )
    assert calls == 1


async def test_guard_does_not_hide_independent_fiber_writer(clients, monkeypatch):
    from dataclasses import replace

    from surreal_memory.core.fiber import Fiber
    from surreal_memory.engine.consolidation_progress import ConsolidationProgressSession

    first, second, edge, _ = clients
    missing = str(uuid4())
    fiber = Fiber.create(
        neuron_ids={edge.source_id}, synapse_ids={missing}, anchor_neuron_id=edge.source_id
    )
    await first.add_fiber(fiber)
    session = ConsolidationProgressSession(first, first.brain_id, "owner", {}, lease_seconds=180)
    original = first._ensure_conn().query_raw
    submitted = asyncio.Event()
    raw_responses = []

    async def delayed_page(sql, params=None, **kwargs):
        if "-- bounded legacy prune repair" in sql:
            marker = "IF array::len($lease) != 1 { THROW 'prune lease lost'; };"
            assert sql.count(marker) == 1
            sql = sql.replace(marker, marker + "\nSLEEP 3s;", 1)
            submitted.set()
            result = await original(sql, params, **kwargs)
            raw_responses.append(result)
            return result
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(first._ensure_conn(), "query_raw", delayed_page)
    task = asyncio.create_task(
        first.repair_prune_synapse_refs_if_absent(
            [missing], lease_owner_token=session.owner_token, page_guard=session.lease_fenced_page
        )
    )
    try:
        await asyncio.wait_for(submitted.wait(), 10)
        await asyncio.sleep(0.5)
        await asyncio.wait_for(second.update_fiber(replace(fiber, pinned=True)), 10)
        assert not task.done(), "independent write did not overlap page"
        with pytest.raises(PruneTransactionConflictError):
            await asyncio.wait_for(task, 10)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert raw_responses and not session.lease_lost
    saved = await first.get_fiber(fiber.id)
    assert saved is not None and saved.pinned and missing in saved.synapse_ids
    assert not session._renew_lock.locked()
