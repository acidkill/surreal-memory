"""Tool memory processing engine.

Reads raw tool events from the staging table (or JSONL buffer),
detects usage patterns, and promotes them to neurons and synapses:
- USED_WITH: Tools frequently used together within a time window.
- EFFECTIVE_FOR: Tools that succeed in the context of a specific task.

Processing is designed to be idempotent and batched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Direction, Synapse, SynapseType

if TYPE_CHECKING:
    # Storage is typed as Any because tool_memory functions accept concrete
    # backend classes with tool-event methods — neither backend declares
    # them on the base NeuralStorage protocol.
    from typing import Protocol

    from surreal_memory.unified_config import ToolMemoryConfig

    class _ToolEventStorage(Protocol):
        async def insert_tool_events(self, brain_id: str, events: list[dict[str, Any]]) -> int: ...
        async def get_unprocessed_events(
            self, brain_id: str, limit: int = ...
        ) -> list[dict[str, Any]]: ...
        async def mark_events_processed(self, brain_id: str, event_ids: list[Any]) -> None: ...
        async def find_neurons(self, *, content_exact: str, limit: int = ...) -> list[Neuron]: ...
        async def add_neuron(self, neuron: Any) -> None: ...
        async def get_synapses(
            self,
            *,
            source_id: str | None = ...,
            target_id: str | None = ...,
            type: Any | None = ...,
        ) -> list[Any]: ...
        async def add_synapse(self, synapse: Any) -> None: ...
        async def update_synapse(self, synapse: Any) -> None: ...
        def set_brain(self, brain_id: str) -> None: ...


logger = logging.getLogger(__name__)

# Max characters for args_summary in JSONL buffer
_MAX_ARGS_SUMMARY = 200


@dataclass(frozen=True)
class IngestResult:
    """Result of ingesting JSONL buffer into tool_events table."""

    events_ingested: int
    events_skipped: int


@dataclass(frozen=True)
class ProcessResult:
    """Result of processing one committed tool-event batch."""

    neurons_created: int
    synapses_created: int
    synapses_reinforced: int
    events_processed: int
    last_event_id: str | None = None


def _parse_buffer_line(line: str) -> dict[str, Any] | None:
    """Parse a single JSONL line into an event dict.

    Returns None if the line is malformed.
    """
    try:
        data: dict[str, Any] = json.loads(line)
        if not isinstance(data, dict) or "tool_name" not in data:
            return None
        return data
    except (json.JSONDecodeError, TypeError):
        return None


@contextmanager
def _buffer_lock(buffer_path: Path) -> Iterator[None]:
    """Coordinate buffer reads/acks with the hook's stable sidecar lock."""
    lock_path = buffer_path.with_name(f"{buffer_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback is best-effort
            yield
            return
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read_complete_buffer_prefix(buffer_path: Path, max_lines: int) -> tuple[bytes, list[bytes]]:
    """Snapshot complete oldest lines without racing hook appends or rotation."""
    if max_lines <= 0:
        return b"", []
    with _buffer_lock(buffer_path):
        try:
            with buffer_path.open("rb") as stream:
                lines: list[bytes] = []
                for _ in range(max_lines):
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    lines.append(line)
        except OSError:
            logger.debug("Failed to read tool events buffer", exc_info=True)
            return b"", []
    return b"".join(lines), lines


def _ack_buffer_prefix(buffer_path: Path, acknowledged_prefix: bytes) -> bool:
    """Atomically remove only the DB-acknowledged prefix, preserving new appends."""
    if not acknowledged_prefix:
        return True
    temporary_path: str | None = None
    with _buffer_lock(buffer_path):
        try:
            current = buffer_path.read_bytes()
            if not current.startswith(acknowledged_prefix):
                return False
            file_mode = buffer_path.stat().st_mode & 0o777
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=buffer_path.parent,
                prefix=f".{buffer_path.name}.",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                temporary.write(current[len(acknowledged_prefix) :])
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_path, file_mode)
            os.replace(temporary_path, buffer_path)
            temporary_path = None
            return True
        except OSError:
            logger.debug("Failed to acknowledge tool events buffer prefix", exc_info=True)
            return False
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    logger.debug("Failed to remove temporary tool event buffer", exc_info=True)


async def ingest_buffer(
    storage: _ToolEventStorage,
    brain_id: str,
    buffer_path: Path,
    max_lines: int = 10000,
) -> IngestResult:
    """Read a bounded JSONL prefix, insert idempotently, then acknowledge it.

    Stable event IDs make a database commit followed by a process crash safe to
    retry. The source prefix remains in place until storage acknowledges the
    insert; concurrent hook appends are preserved when that exact prefix is
    removed.
    """
    if not buffer_path.exists() or max_lines <= 0:
        return IngestResult(events_ingested=0, events_skipped=0)

    prefix, lines = _read_complete_buffer_prefix(buffer_path, max_lines)
    if not lines:
        return IngestResult(events_ingested=0, events_skipped=0)

    events: list[dict[str, Any]] = []
    skipped = 0
    for index, raw_line in enumerate(lines):
        try:
            line = raw_line.rstrip(b"\r\n").decode("utf-8")
        except UnicodeDecodeError:
            skipped += 1
            continue
        parsed = _parse_buffer_line(line)
        if parsed is None:
            skipped += 1
            continue
        event_id = parsed.get("event_id") or parsed.get("id")
        if not event_id:
            legacy_identity = b"\0".join(
                (brain_id.encode("utf-8"), str(index).encode("ascii"), raw_line)
            )
            event_id = hashlib.sha256(legacy_identity).hexdigest()
        parsed["event_id"] = str(event_id)
        events.append(parsed)

    inserted = 0
    if events:
        # If this raises after some rows committed, don't acknowledge the source
        # bytes. Stable IDs make the next attempt skip those existing DB rows.
        inserted = await storage.insert_tool_events(brain_id, events)

    if not _ack_buffer_prefix(buffer_path, prefix):
        logger.warning(
            "Tool-event buffer changed before acknowledgement; leaving source lines for retry"
        )

    return IngestResult(events_ingested=inserted, events_skipped=skipped)


def _tool_neuron_content(tool_name: str) -> str:
    """Canonical content string for a tool neuron."""
    return f"tool:{tool_name}"


async def _find_or_create_tool_neuron(
    storage: _ToolEventStorage,
    tool_name: str,
    server_name: str,
) -> tuple[str, bool]:
    """Find existing tool neuron or create a new one.

    Storage must have brain context set via set_brain() before calling.
    Returns (neuron_id, was_created).
    """
    content = _tool_neuron_content(tool_name)

    # Search for existing neuron with this content
    existing = await storage.find_neurons(content_exact=content, limit=1)
    if existing:
        return existing[0].id, False

    # Create new tool neuron
    neuron = Neuron.create(
        content=content,
        type=NeuronType.ENTITY,
        metadata={"tool_server": server_name, "tool_type": "mcp"},
    )
    await storage.add_neuron(neuron)
    return neuron.id, True


async def _find_or_create_concept_neuron(
    storage: _ToolEventStorage,
    concept: str,
) -> tuple[str, bool]:
    """Find existing concept neuron or create a new one for task context.

    Returns (neuron_id, was_created).
    """
    existing = await storage.find_neurons(content_exact=concept, limit=1)
    if existing:
        return existing[0].id, False

    neuron = Neuron.create(
        content=concept,
        type=NeuronType.CONCEPT,
    )
    await storage.add_neuron(neuron)
    return neuron.id, True


async def _find_synapse_between(
    storage: _ToolEventStorage,
    source_id: str,
    target_id: str,
    synapse_type: SynapseType,
) -> Synapse | None:
    """Find an existing synapse between two neurons."""
    synapses = await storage.get_synapses(
        source_id=source_id,
        target_id=target_id,
        type=synapse_type,
    )
    return synapses[0] if synapses else None


async def process_events(
    storage: _ToolEventStorage,
    brain_id: str,
    config: ToolMemoryConfig,
) -> ProcessResult:
    """Process unprocessed tool events into neurons and synapses.

    Pattern detection:
    1. USED_WITH: Tools used within cooccurrence_window_s in same session.
    2. EFFECTIVE_FOR: Successful tool used with a task_context.

    Requires storage.set_brain(brain_id) to have been called before.

    Args:
        storage: Storage backend.
        brain_id: Brain context (used for tool_events table queries).
        config: Tool memory configuration.

    Returns:
        ProcessResult with counts.
    """
    events = await storage.get_unprocessed_events(brain_id, config.process_batch_size)
    if not events:
        return ProcessResult(
            neurons_created=0,
            synapses_created=0,
            synapses_reinforced=0,
            events_processed=0,
        )

    neurons_created = 0
    synapses_created = 0
    synapses_reinforced = 0

    # Count tool frequency across all events
    tool_freq: dict[str, int] = defaultdict(int)
    tool_server: dict[str, str] = {}
    for ev in events:
        tool_freq[ev["tool_name"]] += 1
        tool_server[ev["tool_name"]] = ev.get("server_name", "")

    # Only process tools that meet frequency threshold
    frequent_tools = {t for t, c in tool_freq.items() if c >= config.min_frequency}

    # Group events by session for co-occurrence detection. Tool events commonly
    # carry no session_id (background/subagent tool calls) — those still form a
    # valid, shared time-ordered stream rather than being dropped entirely, the
    # same treatment `learn_tool_habits` (sequence_mining.py) already gives the
    # session-less tool_events buffer. The sliding-window check below already
    # bounds pairing to events within `cooccurrence_window_s`, so folding all
    # session-less events into one bucket doesn't change its complexity.
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        sid = ev.get("session_id") or "_no_session"
        sessions[sid].append(ev)

    # USED_WITH detection: sliding window within each session
    seen_pairs: set[tuple[str, str]] = set()
    for session_events in sessions.values():
        session_events.sort(key=lambda e: e["created_at"])
        for i, ev_a in enumerate(session_events):
            if ev_a["tool_name"] not in frequent_tools:
                continue
            for j in range(i + 1, len(session_events)):
                ev_b = session_events[j]
                if ev_b["tool_name"] not in frequent_tools:
                    continue
                if ev_a["tool_name"] == ev_b["tool_name"]:
                    continue

                # Check time window
                try:
                    ts_a = datetime.fromisoformat(ev_a["created_at"])
                    ts_b = datetime.fromisoformat(ev_b["created_at"])
                    delta = abs((ts_b - ts_a).total_seconds())
                except (ValueError, TypeError):
                    continue

                if delta > config.cooccurrence_window_s:
                    break  # Sorted by time — no more within window

                # Canonical pair (alphabetical order for dedup)
                pair = tuple(sorted([ev_a["tool_name"], ev_b["tool_name"]]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Find or create neurons for both tools
                nid_a, created_a = await _find_or_create_tool_neuron(
                    storage, pair[0], tool_server.get(pair[0], "")
                )
                nid_b, created_b = await _find_or_create_tool_neuron(
                    storage, pair[1], tool_server.get(pair[1], "")
                )
                neurons_created += int(created_a) + int(created_b)

                # Find or create USED_WITH synapse (check both directions)
                existing_syn = await _find_synapse_between(
                    storage, nid_a, nid_b, SynapseType.USED_WITH
                )
                if existing_syn is None:
                    existing_syn = await _find_synapse_between(
                        storage, nid_b, nid_a, SynapseType.USED_WITH
                    )

                if existing_syn is None:
                    # Keep retries idempotent: if a crash happens before the event
                    # marker is committed, replay must not reinforce this edge again.
                    synapse = Synapse.create(
                        source_id=nid_a,
                        target_id=nid_b,
                        type=SynapseType.USED_WITH,
                        weight=0.3,
                        direction=Direction.BIDIRECTIONAL,
                    )
                    await storage.add_synapse(synapse)
                    synapses_created += 1

    # EFFECTIVE_FOR detection: tools with task_context
    task_tool_pairs: set[tuple[str, str]] = set()  # (tool_name, task_context)
    for ev in events:
        if ev["tool_name"] not in frequent_tools:
            continue
        task = ev.get("task_context", "").strip()
        if not task:
            continue
        if not ev.get("success", True):
            continue

        pair_key = (ev["tool_name"], task)
        if pair_key in task_tool_pairs:
            continue
        task_tool_pairs.add(pair_key)

        tool_nid, tool_created = await _find_or_create_tool_neuron(
            storage, ev["tool_name"], ev.get("server_name", "")
        )
        task_nid, task_created = await _find_or_create_concept_neuron(storage, task)
        neurons_created += int(tool_created) + int(task_created)

        existing_syn = await _find_synapse_between(
            storage, tool_nid, task_nid, SynapseType.EFFECTIVE_FOR
        )
        if existing_syn is None:
            # Existing edges are intentionally not reinforced here: event markers
            # and this graph write cannot share a transaction, so a retry must be
            # a no-op for an edge already created by an interrupted attempt.
            synapse = Synapse.create(
                source_id=tool_nid,
                target_id=task_nid,
                type=SynapseType.EFFECTIVE_FOR,
                weight=0.4,
                direction=Direction.UNIDIRECTIONAL,
            )
            await storage.add_synapse(synapse)
            synapses_created += 1

    # Mark all events as processed
    event_ids = [ev["id"] for ev in events]
    await storage.mark_events_processed(brain_id, event_ids)

    return ProcessResult(
        neurons_created=neurons_created,
        synapses_created=synapses_created,
        synapses_reinforced=synapses_reinforced,
        events_processed=len(events),
        last_event_id=str(event_ids[-1]) if event_ids else None,
    )
