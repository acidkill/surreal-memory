"""SurrealDB integration coverage for tool-event replay after a process crash."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.unified_config import ToolMemoryConfig, UnifiedConfig

SURREALDB_TEST_URL = os.getenv("SURREALDB_TEST_URL")
TEST_AUTH = ("root", "root")


def _is_loopback_test_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_test_url(SURREALDB_TEST_URL),
        reason="requires an explicit loopback SURREALDB_TEST_URL with port",
    ),
]


def _new_store(database: str) -> SurrealDBStorage:
    return SurrealDBStorage(
        url=SURREALDB_TEST_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace="smem_tool_resume_it",
        database=database,
    )


async def _snapshot(store: SurrealDBStorage, brain_id: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "events": await store._query(
            "SELECT event_id, processed FROM tool_events WHERE brain_id = $bid ORDER BY event_id",
            bid=brain_id,
        ),
        "neurons": await store._query(
            "SELECT id, content FROM neuron WHERE brain_id = $bid ORDER BY content",
            bid=brain_id,
        ),
        "synapses": await store._query(
            "SELECT id, type, weight, source_id, target_id FROM synapse "
            "WHERE brain_id = $bid ORDER BY id",
            bid=brain_id,
        ),
    }


@pytest.mark.asyncio
async def test_process_tool_events_replays_after_restart_without_duplicate_graph_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only the explicit loopback test URL is honored. No inherited production
    # endpoint, credentials, or application configuration participates in this test.
    for name in tuple(os.environ):
        if name.startswith(("SURREALDB_", "SMEM_")) and name != "SURREALDB_TEST_URL":
            monkeypatch.delenv(name, raising=False)

    database = "it_" + uuid.uuid4().hex[:12]
    brain = Brain.create(name="tool-events-resume-it")
    tool_config = ToolMemoryConfig(
        enabled=True,
        min_frequency=1,
        process_batch_size=10,
        cooccurrence_window_s=60,
    )
    monkeypatch.setattr(
        UnifiedConfig,
        "load",
        classmethod(lambda cls: SimpleNamespace(data_dir=tmp_path, tool_memory=tool_config)),
    )

    first = _new_store(database)
    await first.initialize()
    await first.save_brain(brain)
    first.set_brain(brain.id)
    timestamp = datetime.now(UTC).isoformat()
    await first.insert_tool_events(
        brain.id,
        [
            {
                "event_id": "resume-event-a",
                "tool_name": "Read",
                "server_name": "fs",
                "session_id": "resume-session",
                "success": True,
                "created_at": timestamp,
            },
            {
                "event_id": "resume-event-b",
                "tool_name": "Grep",
                "server_name": "fs",
                "session_id": "resume-session",
                "success": True,
                "created_at": timestamp,
            },
        ],
    )

    async def crash_before_marking(brain_id: str, event_ids: list[str]) -> None:
        raise RuntimeError("simulated process crash after graph writes")

    monkeypatch.setattr(first, "mark_events_processed", crash_before_marking)
    try:
        with pytest.raises(RuntimeError, match="after graph writes"):
            await ConsolidationEngine(first)._process_tool_events(
                ConsolidationReport(), dry_run=False
            )
        after_crash = await _snapshot(first, brain.id)
    finally:
        await first.close()

    assert len(after_crash["neurons"]) == 2
    assert len(after_crash["synapses"]) == 1
    assert [row["processed"] for row in after_crash["events"]] == [False, False]
    assert after_crash["synapses"][0]["weight"] == pytest.approx(0.3)

    resumed = _new_store(database)
    await resumed.initialize()
    resumed.set_brain(brain.id)
    try:
        report = ConsolidationReport()
        await ConsolidationEngine(resumed)._process_tool_events(report, dry_run=False)
        after_resume = await _snapshot(resumed, brain.id)

        assert after_resume["events"] == [
            {"event_id": "resume-event-a", "processed": True},
            {"event_id": "resume-event-b", "processed": True},
        ]
        assert after_resume["neurons"] == after_crash["neurons"]
        assert after_resume["synapses"] == after_crash["synapses"]
        assert len(after_resume["neurons"]) == 2
        assert {row["content"] for row in after_resume["neurons"]} == {"tool:Read", "tool:Grep"}
        assert len(after_resume["synapses"]) == 1
        assert after_resume["synapses"][0]["weight"] == pytest.approx(0.3)
        assert report.extra["tool_events_processed"] == 2
    finally:
        await resumed.close()
