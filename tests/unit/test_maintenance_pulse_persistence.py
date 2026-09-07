"""The health-pulse op counter persists per-brain across processes (#222).

The counter used to be a class-level in-process attribute reset on every
start, so pulsing depended on how work was distributed across processes: the
reporter's arithmetic — 100 sessions x 24 operations (2400 total, interval 25)
— produced ZERO pulses, while one 2400-op session produced 96. The fix seeds
each process's counter from the brain's metadata and flushes the total back
(pulse boundary + every 25 ops), fail-soft to the old per-process behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.mcp.maintenance_handler import MaintenanceHandler


class _BrainStore:
    """Fake storage persisting brain metadata like the real backends do."""

    def __init__(self) -> None:
        self.brain = SimpleNamespace(id="b1", config=MagicMock(), metadata={})
        self.brain_id = "b1"

    # MaintenanceHandler surface
    async def get_brain(self, _id: str) -> Any:
        return self.brain

    async def save_brain(self, brain: Any) -> None:
        self.brain = brain


def _handler(store: _BrainStore, interval: int) -> MaintenanceHandler:
    h = MaintenanceHandler.__new__(MaintenanceHandler)
    h._op_count = 0
    h._op_base = None
    h._last_pulse = None
    h._last_consolidation_at = None
    h._last_dream_at = None
    h._effective_check_interval = None
    h._consolidation_task = None
    h.config = MagicMock()
    h.config.maintenance.enabled = True
    h.config.maintenance.check_interval = interval
    h.get_storage = AsyncMock(return_value=store)
    return h


@pytest.mark.asyncio
async def test_short_sessions_pulse_on_total_work() -> None:
    """The reporter's arithmetic: 100 x 24 ops must not yield zero pulses."""
    store = _BrainStore()
    pulses = 0
    for _session in range(100):  # a fresh handler per short-lived session
        h = _handler(store, interval=25)
        h._health_pulse = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(hints=[], should_consolidate=False)
        )
        h._fire_health_trigger = MagicMock()  # type: ignore[method-assign]
        h._maybe_auto_consolidate = AsyncMock()  # type: ignore[method-assign]
        h._maybe_run_expiry_cleanup = AsyncMock()  # type: ignore[method-assign]
        h._create_alerts_from_pulse = AsyncMock()  # type: ignore[method-assign]
        h._auto_resolve_cleared = AsyncMock()  # type: ignore[method-assign]
        for _op in range(24):
            pulse = await h._check_maintenance()
            if pulse is not None:
                pulses += 1
    assert pulses == 96, (
        f"2400 total ops at interval 25 must pulse 96 times across sessions; got {pulses}"
    )


@pytest.mark.asyncio
async def test_storage_failure_degrades_to_per_process() -> None:
    h = MaintenanceHandler.__new__(MaintenanceHandler)
    h._op_count = 0
    h._op_base = None
    h._last_pulse = None
    h._last_consolidation_at = None
    h._last_dream_at = None
    h._effective_check_interval = None
    h._consolidation_task = None
    h.config = MagicMock()
    h.config.maintenance.enabled = True
    h.config.maintenance.check_interval = 25

    async def _boom() -> None:
        raise RuntimeError("storage down")

    h.get_storage = AsyncMock(side_effect=RuntimeError("storage down"))  # type: ignore[method-assign]
    total = await h._increment_op_counter()
    assert total == 1, "fail-soft: the counter still works per-process"
    assert h._should_check_health(total) is False
    h._op_count = 25
    assert h._should_check_health() is True, "in-process modulo semantics unchanged"
