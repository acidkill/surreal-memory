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


async def test_unprotected_repair_page_and_lease_renewal_conflict(clients, monkeypatch):
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
