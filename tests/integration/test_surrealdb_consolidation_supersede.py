"""Disposable local-DB integration test for superseding a failed semantic_link run.

Real lease, real progress record, real UPSERT. Loopback only - never run this against a shared or
production endpoint.
"""

from __future__ import annotations

import ipaddress
import os
import uuid
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressSession,
    ConsolidationResumeMismatchError,
)
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_fence_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        if hostname == "localhost":
            return True
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires explicit loopback SURREALDB_URL for a disposable SurrealDB >= 3.2",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="supersede_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-supersede-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _failed_semantic_link_run(store) -> str:
    session = await ConsolidationProgressSession.open(store, ["semantic_link"], "fp-it", utcnow())
    assert session is not None
    await session.checkpoint(
        "semantic_link", "semantic_link_discovery_similarity", cursor="7", pending=["{}"]
    )
    await session.fail("SemanticSourceChangedError: neuron changed since the token was captured")
    run_id = str(session.state["run_id"])
    await session.close()
    return run_id


@pytest.mark.asyncio
async def test_a_failed_semantic_link_run_blocks_merge_until_it_is_superseded(store) -> None:
    old_run = await _failed_semantic_link_run(store)

    with pytest.raises(ConsolidationResumeMismatchError, match="requested strategies differ"):
        await ConsolidationProgressSession.open(store, ["merge"], "fp-it", utcnow())

    session = await ConsolidationProgressSession.open(
        store, ["merge"], "fp-it", utcnow(), supersede_failed_semantic_link=True
    )
    assert session is not None and not session.resumed
    await session.checkpoint("merge", "merge_scan", cursor="1")
    await session.complete_strategy("merge")
    await session.complete()
    new_run = str(session.state["run_id"])
    await session.close()

    record = await store.get_consolidation_progress(store.current_brain_id)
    assert record is not None
    assert record["run_id"] == new_run != old_run
    assert record["status"] == "completed"
    audit = record["strategy_states"]["_superseded_runs"]
    assert [item["run_id"] for item in audit] == [old_run]
    assert audit[0]["status"] == "failed"
    assert audit[0]["last_error"].startswith("SemanticSourceChangedError")
    # the lease the takeover held was released: the next run can start at once
    again = await ConsolidationProgressSession.open(store, ["merge"], "fp-it", utcnow())
    assert again is not None
    await again.close()
