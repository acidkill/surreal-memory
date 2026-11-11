"""Live census test for SurrealDB's omitted-null round trip behavior."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlsplit

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationProgressSession
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

TEST_SURREALDB_URL = os.getenv("SMEM_TEST_SURREALDB_URL")
TEST_AUTH = ("root", "root")
_NULLABLE_FIBER_FIELDS = (
    "last_conducted",
    "time_start",
    "time_end",
    "summary",
    "essence",
    "last_ghost_shown_at",
)


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
        not _is_loopback_test_url(TEST_SURREALDB_URL),
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL with port",
    ),
]


@pytest.mark.asyncio
async def test_census_round_trip_restores_fields_surrealdb_omits() -> None:
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace=os.environ["SURREALDB_NS"],
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="census-null-roundtrip-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    reference_time = datetime(2026, 10, 12, 12, 0)
    fiber = Fiber(
        id="null-roundtrip-" + uuid.uuid4().hex,
        neuron_ids={"synthetic-neuron"},
        synapse_ids=set(),
        anchor_neuron_id="synthetic-neuron",
        conductivity=0.75,
        coherence=0.4,
        salience=0.6,
        frequency=3,
        metadata={"fixture": "synthetic"},
        created_at=reference_time - timedelta(minutes=1),
    )
    session: ConsolidationProgressSession | None = None
    try:
        await storage.add_fiber(fiber)
        session = await ConsolidationProgressSession.open(
            storage,
            [ConsolidationStrategy.MERGE.value],
            "null-roundtrip-integration",
            reference_time,
        )
        assert session is not None

        engine = ConsolidationEngine(storage, ConsolidationConfig())
        engine._progress_session = session
        engine._active_strategy = ConsolidationStrategy.MERGE
        observed = await engine._all_fibers_paged(created_before=reference_time)

        assert [item.id for item in observed] == [fiber.id]
        assert observed[0].conductivity == 0.75
        assert observed[0].metadata == {"fixture": "synthetic"}
        for field in _NULLABLE_FIBER_FIELDS:
            assert getattr(observed[0], field) is None

        rows = await storage._query(
            "SELECT * FROM consolidation_fiber_census WHERE run_id = $run_id "
            "AND strategy = $strategy ORDER BY page_index ASC",
            run_id=session.state["run_id"],
            strategy=ConsolidationStrategy.MERGE.value,
        )
        data_pages = [row for row in rows if row.get("complete") is not True]
        markers = [row for row in rows if row.get("complete") is True]
        assert len(data_pages) == 1
        assert len(markers) == 1
        stored_fiber = data_pages[0]["fibers"][0]
        assert all(field not in stored_fiber for field in _NULLABLE_FIBER_FIELDS)
        assert "first_fiber_id" not in markers[0]
        assert markers[0]["last_fiber_id"] == fiber.id

        reread = await engine._all_fibers_paged(created_before=reference_time)
        assert reread == observed
    finally:
        if session is not None:
            await session.close()
        await storage.close()
