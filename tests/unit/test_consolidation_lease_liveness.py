"""The consolidation lease must stay alive while a CPU-bound discovery scan runs.

The heartbeat task shares the event loop with the scan; a scan that never awaits something
that really suspends starves it, the lease (180 s) expires and the run stops with
``ConsolidationLeaseLostError`` even though nothing else touched the brain. The fix has two
layers: the scans hand control back to the loop, and every budget check renews the lease
opportunistically when the heartbeat has not done so for a third of the lease.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.engine.consolidation import ConsolidationConfig, ConsolidationEngine
from surreal_memory.engine.consolidation_progress import (
    ConsolidationLeaseLostError,
    ConsolidationProgressSession,
)


def _session(
    *, renew_result: bool = True, lease_seconds: int = 180
) -> ConsolidationProgressSession:
    storage = SimpleNamespace(renew_consolidation_lease=AsyncMock(return_value=renew_result))
    return ConsolidationProgressSession(
        storage, "brain-1", "owner-1", {"strategy_states": {}}, lease_seconds=lease_seconds
    )


@pytest.mark.asyncio
async def test_renew_if_due_renews_after_a_third_of_the_lease_and_records_the_gap() -> None:
    session = _session()
    session._last_renewed = time.monotonic() - 100.0  # heartbeat starved for 100 s
    await session.renew_if_due()
    session.storage.renew_consolidation_lease.assert_awaited_once()
    assert session.max_renew_gap_s >= 100.0
    assert time.monotonic() - session._last_renewed < 5.0


@pytest.mark.asyncio
async def test_renew_if_due_does_nothing_when_the_lease_was_renewed_recently() -> None:
    session = _session()
    session._last_renewed = time.monotonic() - 5.0  # null control: no starvation
    await session.renew_if_due()
    session.storage.renew_consolidation_lease.assert_not_awaited()
    assert session.max_renew_gap_s == 0.0


@pytest.mark.asyncio
async def test_renew_if_due_raises_lease_lost_when_the_store_refuses_the_renewal() -> None:
    session = _session(renew_result=False)
    session._last_renewed = time.monotonic() - 100.0
    with pytest.raises(ConsolidationLeaseLostError):
        await session.renew_if_due()
    assert session.lease_lost is True


@pytest.mark.asyncio
async def test_budget_check_fences_work_when_lease_renewal_raises() -> None:
    session = _session()
    session._last_renewed = time.monotonic() - 100.0
    session.storage.renew_consolidation_lease.side_effect = OSError("fixture unavailable")
    engine = ConsolidationEngine(SimpleNamespace(), ConsolidationConfig())
    engine._progress_session = session

    with pytest.raises(ConsolidationLeaseLostError, match="could not be renewed"):
        await engine._check_progress_budget()

    assert session.lease_lost is True
    session.storage.renew_consolidation_lease.assert_awaited_once()


@pytest.mark.asyncio
async def test_budget_check_renews_the_lease_when_the_session_supports_it() -> None:
    engine = ConsolidationEngine(SimpleNamespace(), ConsolidationConfig())
    engine._progress_session = SimpleNamespace(  # type: ignore[assignment]
        lease_lost=False, brain_id="brain-1", renew_if_due=AsyncMock()
    )
    await engine._check_progress_budget()
    engine._progress_session.renew_if_due.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_budget_check_tolerates_sessions_without_renew_if_due() -> None:
    engine = ConsolidationEngine(SimpleNamespace(), ConsolidationConfig())
    engine._progress_session = SimpleNamespace(lease_lost=False, brain_id="brain-1")  # type: ignore[assignment]
    await engine._check_progress_budget()  # must not raise


def test_report_renders_the_lease_gap_only_when_it_was_measured() -> None:
    from surreal_memory.engine.consolidation import ConsolidationReport

    quiet = ConsolidationReport()
    assert "Lease max renew gap" not in quiet.summary()
    measured = ConsolidationReport()
    measured.extra["lease_max_renew_gap_s"] = 104.2
    assert "Lease max renew gap: 104.2s" in measured.summary()


def _unused(_: Any) -> None:  # keeps ``Any`` import honest for mypy strictness in this module
    return None
