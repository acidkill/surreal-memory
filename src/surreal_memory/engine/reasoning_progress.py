"""Live progress reporting for a reasoning-mining run.

A ``MiningProgress`` snapshot is emitted through a ``ProgressCallback`` as a
mining run moves through its phases (``scanning`` the transcript corpus ->
``ingesting`` traces per file -> ``distilling`` patterns per model -> ``done``).
The callback is invoked ONLY from the event-loop thread: blocking file work runs
in ``asyncio.to_thread`` and returns its data to the caller, which then emits the
progress update. Consumers (the dashboard mining-state, the CLI progress printer,
the MCP response) read these snapshots; they never mutate a live object because
each emit passes an immutable copy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Ordered mining phases. ``idle`` is the resting state before a run starts.
PHASE_IDLE = "idle"
PHASE_SCANNING = "scanning"
PHASE_INGESTING = "ingesting"
PHASE_DISTILLING = "distilling"
PHASE_DONE = "done"


@dataclass
class MiningProgress:
    """A point-in-time snapshot of a reasoning-mining run.

    Additive by design: scanning/ingesting populate the file + trace counters;
    distillation (a later stage) populates ``current_model`` and the model /
    pattern counters. All fields default so a partially-populated snapshot is
    always valid.
    """

    phase: str = PHASE_IDLE
    files_total: int = 0
    files_scanned: int = 0
    traces_found: int = 0
    traces_ingested: int = 0
    traces_processed: int = 0
    patterns_learned: int = 0
    current_model: str | None = None
    models_done: int = 0
    models_total: int = 0


# A progress sink. Invoked from the event-loop thread with an immutable snapshot.
ProgressCallback = Callable[[MiningProgress], None]
