from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from surreal_memory.server.routes.consolidation import get_storage, router


class _Storage:
    async def get_brain(self, brain_id: str) -> object:
        assert brain_id == "default"
        return object()

    def set_brain(self, brain_id: str) -> None:
        assert brain_id == "default"


def test_consolidate_response_exposes_paused_progress() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage] = _Storage

    report = SimpleNamespace(
        started_at=datetime.now(UTC),
        duration_ms=12.5,
        synapses_pruned=0,
        neurons_pruned=0,
        fibers_merged=0,
        fibers_removed=0,
        fibers_created=0,
        summaries_created=0,
        drift_clusters_found=0,
        drift_clusters_persisted=0,
        semantic_synapses_created=0,
        semantic_synapses_skipped=0,
        duplicates_found=0,
        new_alias_links=0,
        merge_details=[],
        dry_run=False,
        extra={
            "consolidation_status": "paused",
            "consolidation_progress_messages": ["resuming prune at synapse_scan"],
            "last_checkpoint": "prune phase=synapse_scan cursor=synapse-42",
            "failed_strategies": [],
        },
    )
    engine = SimpleNamespace(run=AsyncMock(return_value=report))

    with patch("surreal_memory.engine.consolidation.ConsolidationEngine", return_value=engine):
        response = TestClient(app).post(
            "/brain/default/consolidate",
            json={"strategies": ["prune"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["consolidation_status"] == "paused"
    assert data["consolidation_progress_messages"] == ["resuming prune at synapse_scan"]
    assert data["last_checkpoint"] == "prune phase=synapse_scan cursor=synapse-42"
    assert data["failed_strategies"] == []
