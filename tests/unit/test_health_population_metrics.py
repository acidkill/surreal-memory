"""Health metrics must use the complete organic fiber cohort, across all surfaces."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.engine.diagnostics import (
    BrainHealthReport,
    DiagnosticsEngine,
    _build_dynamic_action,
)
from surreal_memory.engine.memory_stages import MemoryStage
from surreal_memory.server.routes import dashboard_api
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


@pytest.mark.asyncio
async def test_fiber_health_counts_use_full_population_and_matching_semantic_membership() -> None:
    cutoff = datetime(2026, 11, 1)
    fibers = [
        SimpleNamespace(
            id="old", tags=set(), created_at=cutoff - timedelta(days=1), last_conducted=None
        ),
        SimpleNamespace(id="fresh", tags=set(), created_at=cutoff, last_conducted=None),
        SimpleNamespace(id="indexed", tags={"code_index"}, created_at=cutoff, last_conducted=None),
    ]
    storage = SimpleNamespace(
        brain_id="brain",
        get_stats=AsyncMock(return_value={"fiber_count": 3}),
        get_fibers=AsyncMock(return_value=fibers),
        find_maturations=AsyncMock(
            return_value=[SimpleNamespace(fiber_id=i) for i in ("old", "indexed", "missing")]
        ),
    )
    counts = await NeuralStorage.get_fiber_health_counts(storage, "brain", cutoff)
    assert counts == {
        "organic_fiber_count": 2,
        "fresh_organic_fiber_count": 1,
        "semantic_organic_fiber_count": 1,
    }
    storage.get_fibers.assert_awaited_once_with(limit=3)
    storage.find_maturations.assert_awaited_once_with(stage=MemoryStage.SEMANTIC)


@pytest.mark.asyncio
@pytest.mark.parametrize("populated", [True, False])
async def test_analyze_uses_population_counts_instead_of_newest_10000_sample(
    populated: bool,
) -> None:
    storage = SimpleNamespace(
        set_brain=lambda _: None,
        get_enhanced_stats=AsyncMock(
            return_value={
                "neuron_count": 100,
                "synapse_count": 300,
                "fiber_count": 13740,
                "synapse_stats": {
                    "by_type": {"related_to": {"count": 300}},
                    "organic_synapse_count": 300,
                    "avg_weight": 0.5,
                },
            }
        ),
        get_fibers=AsyncMock(return_value=[]),
        get_fiber_health_counts=AsyncMock(
            return_value={
                "organic_fiber_count": 13008 if populated else 0,
                "fresh_organic_fiber_count": 2611 if populated else 0,
                "semantic_organic_fiber_count": 1503 if populated else 0,
            }
        ),
        count_activated_neuron_states=AsyncMock(return_value=100),
        get_connected_neuron_ids=AsyncMock(return_value={str(i) for i in range(100)}),
        get_fiber_stage_counts=AsyncMock(return_value={"semantic": 1591}),
        find_maturations=AsyncMock(return_value=[]),
    )
    report = await DiagnosticsEngine(storage).analyze("brain")
    assert report.consolidation_ratio == (round(1503 / 13008, 4) if populated else 0.0)
    assert report.freshness == (round(2611 / 13008, 4) if populated else 0.0)
    assert report.organic_fiber_count == (13008 if populated else 0)
    assert report.fresh_organic_fiber_count == (2611 if populated else 0)
    assert report.semantic_organic_fiber_count == (1503 if populated else 0)
    assert report.orphan_rate_sampled is True
    storage.get_fiber_health_counts.assert_awaited_once()


@pytest.mark.asyncio
async def test_surreal_health_counts_are_brain_scoped_aggregates_without_fiber_content() -> None:
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    queries: list[tuple[str, dict[str, Any]]] = []

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        queries.append((sql, params))
        return [{"c": 1 if "FROM maturation" in sql or "last_conducted" in sql else 2}]

    store._query = query  # type: ignore[assignment,method-assign]
    cutoff = datetime(2026, 11, 1)
    counts = await store.get_fiber_health_counts("brain", cutoff)
    assert counts == {
        "organic_fiber_count": 2,
        "fresh_organic_fiber_count": 1,
        "semantic_organic_fiber_count": 1,
    }
    assert len(queries) == 3
    assert all(
        "SELECT count()" in sql
        and "LIMIT" not in sql
        and "auto_tags" in sql
        and "agent_tags" in sql
        for sql, _ in queries
    )
    assert all("brain_id" in sql for sql, _ in queries)
    assert any(params.get("cutoff") == cutoff for _, params in queries)
    assert any("type::record('fiber'" in sql and "string::replace" in sql for sql, _ in queries)


@pytest.mark.asyncio
async def test_dashboard_health_exposes_existing_gate_and_cohort_explanations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = dict.fromkeys(
        (
            "connectivity",
            "diversity",
            "freshness",
            "consolidation_ratio",
            "orphan_rate",
            "activation_efficiency",
            "recall_confidence",
        ),
        0.5,
    )
    report = BrainHealthReport(
        purity_score=50,
        grade="D",
        neuron_count=20,
        synapse_count=50,
        fiber_count=10,
        warnings=(),
        recommendations=(),
        structural_neuron_count=5,
        connectivity_neuron_count=15,
        organic_fiber_count=9,
        fresh_organic_fiber_count=3,
        semantic_organic_fiber_count=2,
        orphan_rate_sampled=True,
        stage_distribution={"episodic": 8, "semantic": 2},
        semantic_gate_blockers={"ready": 2, "spacing_gate": 6, "time_gate": 0},
        **components,
    )
    monkeypatch.setattr(dashboard_api, "_cached_health_report", AsyncMock(return_value=report))
    response = await dashboard_api.get_health(SimpleNamespace())
    payload = response.model_dump()
    assert payload["stage_distribution"] == report.stage_distribution
    assert payload["semantic_gate_blockers"] == report.semantic_gate_blockers
    assert payload["structural_neuron_count"] == 5
    assert payload["connectivity_neuron_count"] == 15
    assert payload["organic_fiber_count"] == 9
    assert payload["fresh_organic_fiber_count"] == 3
    assert payload["semantic_organic_fiber_count"] == 2
    assert payload["orphan_rate_sampled"] is True


def test_freshness_action_reports_measured_organic_count() -> None:
    action = _build_dynamic_action(
        "freshness",
        "fallback",
        {
            "fiber_count": 13740,
            "organic_fiber_count": 13008,
            "fresh_organic_fiber_count": 2611,
            "freshness": 2611 / 13008,
        },
    )
    assert "2611 active in last 7 days" in action
    assert "1300+ memories" in action


@pytest.mark.asyncio
async def test_fallback_rejects_a_different_active_brain() -> None:
    storage = SimpleNamespace(brain_id="other", get_stats=AsyncMock())
    with pytest.raises(ValueError, match="requested brain to be active"):
        await NeuralStorage.get_fiber_health_counts(storage, "brain", datetime(2026, 11, 1))
    storage.get_stats.assert_not_awaited()


@pytest.mark.asyncio
async def test_in_memory_counts_more_than_10000_without_fetching_fiber_content() -> None:
    cutoff = datetime(2026, 11, 1)
    store = InMemoryStorage()
    store.set_brain("brain")
    store._fibers["brain"] = {
        str(i): SimpleNamespace(
            id=str(i), tags=set(), created_at=cutoff - timedelta(days=1), last_conducted=None
        )
        for i in range(10001)
    }
    store._fibers["brain"]["0"].last_conducted = cutoff
    store._fibers["brain"]["1"].tags = {"code_index"}
    store.find_maturations = AsyncMock(
        return_value=[SimpleNamespace(fiber_id=i) for i in ("0", "1", "missing")]
    )
    store.get_fibers = AsyncMock(side_effect=AssertionError("must count existing objects directly"))
    counts = await store.get_fiber_health_counts("brain", cutoff)
    assert counts == {
        "organic_fiber_count": 10000,
        "fresh_organic_fiber_count": 1,
        "semantic_organic_fiber_count": 1,
    }
    store.get_fibers.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,returned,error", [(10001, 0, NotImplementedError), (3, 2, ValueError)]
)
async def test_fallback_refuses_incomplete_or_unbounded_populations(
    size: int, returned: int, error: type[Exception]
) -> None:
    storage = SimpleNamespace(
        brain_id="brain",
        get_stats=AsyncMock(return_value={"fiber_count": size}),
        get_fibers=AsyncMock(return_value=[None] * returned),
    )
    with pytest.raises(error, match=r"complete|incomplete"):
        await NeuralStorage.get_fiber_health_counts(storage, "brain", datetime(2026, 11, 1))
    if size > 10000:
        storage.get_fibers.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_fallback_does_not_request_a_zero_limit() -> None:
    storage = SimpleNamespace(
        brain_id="brain",
        get_stats=AsyncMock(return_value={"fiber_count": 0}),
        get_fibers=AsyncMock(),
        find_maturations=AsyncMock(return_value=[]),
    )
    counts = await NeuralStorage.get_fiber_health_counts(storage, "brain", datetime(2026, 11, 1))
    assert set(counts.values()) == {0}
    storage.get_fibers.assert_not_awaited()
