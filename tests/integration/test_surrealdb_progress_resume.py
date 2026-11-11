"""A repaired resume must not retain an old error after an ordinary pause."""

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from datetime import timedelta
from urllib.parse import urlsplit

import pytest

from surreal_memory.engine.consolidation_progress import ConsolidationProgressSession
from surreal_memory.storage.surrealdb.schema import ensure_schema
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

TEST_AUTH = ("root", "root")  # Fixed disposable-server credentials, never inherited secrets.


@pytest.mark.integration
async def test_failed_resume_clears_error_under_lease_and_preserves_saved_work() -> None:
    url = os.getenv("SMEM_TEST_SURREALDB_URL")
    parsed = urlsplit(url or "")
    if (
        parsed.scheme not in {"http", "https", "ws", "wss"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is None
    ):
        pytest.skip("requires an explicit loopback SMEM_TEST_SURREALDB_URL with port")
    namespace = "smem_resume_" + uuid.uuid4().hex
    store = SurrealDBStorage(
        url=url,
        namespace=namespace,
        database="synthetic",
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
    )
    sessions: list[ConsolidationProgressSession] = []
    await store.initialize()
    store.set_brain("synthetic")
    try:
        reference = utcnow() - timedelta(days=3)
        first = await ConsolidationProgressSession.open(
            store, ["prune", "merge"], "same", reference
        )
        assert first is not None
        sessions.append(first)
        legacy = deepcopy(await store.get_consolidation_progress("synthetic"))
        assert legacy is not None
        assert legacy.get("work_progress_seq") is None
        assert legacy.get("last_work_progress_at") is None
        await store._query("REMOVE FIELD work_progress_seq ON consolidation_progress")
        await store._query("REMOVE FIELD last_work_progress_at ON consolidation_progress")
        await ensure_schema(store._conn)
        await ensure_schema(store._conn)
        assert await store.get_consolidation_progress("synthetic") == legacy
        await first.checkpoint("prune", "done", counters={"synapses_pruned": 17})
        await first.complete_strategy("prune")
        await first.checkpoint(
            "merge", "fiber_census", cursor="synthetic", pending=["synthetic-page"]
        )
        await first.fail("prior attempt failed")
        await first.pause()
        await first.close()
        before = deepcopy(await store.get_consolidation_progress("synthetic"))
        assert before is not None and before["last_error"] == "prior attempt failed"

        resumed = await ConsolidationProgressSession.open(
            store, ["prune", "merge"], "same", utcnow(), resume_run_id=before["run_id"]
        )
        assert resumed is not None
        sessions.append(resumed)
        running = await store.get_consolidation_progress("synthetic")
        assert running is not None and running["status"] == "running"
        assert running.get("last_error") is None
        for key in (
            "run_id",
            "reference_time",
            "requested_strategies",
            "options_fingerprint",
            "completed_strategies",
            "strategy_states",
            "phase",
            "cursor",
            "counters",
            "work_progress_seq",
            "last_work_progress_at",
        ):
            assert running[key] == before[key]
        await resumed.pause()
        paused = await store.get_consolidation_progress("synthetic")
        assert paused is not None and paused["status"] == "paused"
        assert paused.get("last_error") is None
        await resumed.fail("new attempt failed")
        failed = await store.get_consolidation_progress("synthetic")
        assert failed is not None and failed["last_error"] == "new attempt failed"
        assert failed["work_progress_seq"] == before["work_progress_seq"]
        await resumed.checkpoint("merge", "fiber_census", cursor="synthetic-next")
        advanced = await store.get_consolidation_progress("synthetic")
        assert advanced is not None
        assert advanced["work_progress_seq"] == before["work_progress_seq"] + 1
        assert advanced["last_work_progress_at"] >= before["last_work_progress_at"]
    finally:
        for session in sessions:
            await session.close()
        await store._query(f"REMOVE NAMESPACE {namespace}")
        await store.close()
