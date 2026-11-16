"""Offline contracts for page guard boundaries; database races run separately."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.engine.consolidation_progress import (
    ConsolidationCheckpointUnconfirmedError,
    ConsolidationProgressSession,
)
from tests.unit.test_prune_legacy_repair_bounds import RepairDB


async def test_page_guard_limits_lock_to_raw_page_and_validation():
    db = RepairDB()
    db.renew_consolidation_lease = AsyncMock(return_value=True)
    session = ConsolidationProgressSession(db, "synthetic", "synthetic", {}, lease_seconds=180)
    events = []

    async def budget():
        assert not session._renew_lock.locked()
        events.append("budget")

    async def before_raw():
        assert session._renew_lock.locked()
        events.append("raw")

    original_journal = db._record_changes_bulk

    async def journal(*args):
        assert not session._renew_lock.locked()
        events.append("journal")
        await original_journal(*args)

    db.before_raw = before_raw
    db._record_changes_bulk = journal
    assert (
        await db.repair_prune_synapse_refs_if_absent(
            ["edge-a"],
            lease_owner_token=session.owner_token,
            budget_check=budget,
            page_guard=session.lease_fenced_page,
        )
        == 17
    )
    assert events == ["budget", "raw", "journal"] * 3
    assert db.renew_consolidation_lease.await_count == 3
    assert not session._renew_lock.locked()


async def test_malformed_raw_page_fences_session_before_journal_or_replay():
    db = RepairDB()
    db.renew_consolidation_lease = AsyncMock(return_value=True)
    session = ConsolidationProgressSession(db, "synthetic", "synthetic", {}, lease_seconds=180)
    db.page_override = lambda page: {**page, "scanned": 99}
    with pytest.raises(ConsolidationCheckpointUnconfirmedError):
        await db.repair_prune_synapse_refs_if_absent(
            ["edge-a"],
            lease_owner_token=session.owner_token,
            page_guard=session.lease_fenced_page,
        )
    assert len(db.responses) == 1
    assert not db.journal_ids
    assert session.lease_lost and session._checkpoint_unconfirmed
    assert not session._renew_lock.locked()
