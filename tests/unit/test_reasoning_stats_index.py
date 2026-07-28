"""Regression guard: get_reasoning_stats must not reintroduce the full backward scan.

``get_reasoning_stats`` issues one "latest trace" query per model. Without an
index hint the SurrealDB 3.2.x planner satisfies ``ORDER BY created_at DESC``
by walking ``idx_rtr_time`` (brain_id, created_at) BACKWARD over the whole brain
scope and filtering ``model`` afterwards, so the cost tracks the brain's total
trace count rather than the model's. Measured on a 10.5k-row brain that was
92.6s for a model owning 57 rows -- past the SDK's 30s client timeout, which
surfaced as a TimeoutError and an HTTP 500 from the reasoning status endpoint.
Pinning ``WITH INDEX idx_rtr_model`` brings it to 2.2ms.

The hint is invisible to every functional assertion (results are identical
either way), so nothing else in the suite would notice if a refactor dropped it
and silently restored a 90-second endpoint. These tests exist purely to make
that removal loud.
"""

from __future__ import annotations

from typing import Any

from surreal_memory.storage.surrealdb.reasoning_traces import SurrealDBReasoningTracesMixin

# The composite (brain_id, model) index the per-model query must be pinned to.
# Defined in schema.py as: DEFINE INDEX idx_rtr_model ON reasoning_traces
# FIELDS brain_id, model.
_HINT = "WITH INDEX idx_rtr_model"


class _StatsStore(SurrealDBReasoningTracesMixin):
    """Records every SurrealQL statement get_reasoning_stats issues."""

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return "default"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        c = self.cfg
        if "AND processed = false GROUP BY model" in sql:
            return c.get("unproc_groups", [])
        if "GROUP BY category" in sql:
            return c.get("cat_groups", [])
        if "GROUP BY model" in sql:
            return c.get("model_groups", [])
        if "ORDER BY created_at DESC LIMIT 1" in sql:
            return c.get("latest", {}).get(params.get("model"), [])
        return []

    def latest_queries(self) -> list[tuple[str, dict[str, Any]]]:
        """The per-model "most recent trace" statements."""
        return [(s, p) for s, p in self.queries if "ORDER BY created_at DESC" in s]


def _store(**over: Any) -> _StatsStore:
    cfg: dict[str, Any] = {
        "model_groups": [
            {"model": "claude-fable-5", "cnt": 2},
            {"model": "claude-sonnet-5", "cnt": 1},
        ],
        "unproc_groups": [{"model": "claude-fable-5", "cnt": 2}],
        "latest": {
            "claude-fable-5": [{"created_at": "2026-03-05T00:00:00"}],
            "claude-sonnet-5": [{"created_at": "2026-03-02T00:00:00"}],
        },
        "cat_groups": [{"category": "debugging", "cnt": 1}],
    }
    cfg.update(over)
    return _StatsStore(**cfg)


async def test_per_model_latest_query_carries_index_hint() -> None:
    store = _store()
    await store.get_reasoning_stats("default")

    latest = store.latest_queries()
    assert latest, "expected a per-model 'latest trace' query"
    for sql, _ in latest:
        assert _HINT in sql, f"index hint dropped -> full backward scan restored: {sql}"


async def test_no_unhinted_descending_scan_is_ever_issued() -> None:
    """Catches a *new* unhinted ORDER BY ... DESC query, not just an edited one."""
    store = _store()
    await store.get_reasoning_stats("default")

    unhinted = [s for s, _ in store.queries if "ORDER BY created_at DESC" in s and _HINT not in s]
    assert unhinted == [], f"unhinted descending scan(s) on reasoning_traces: {unhinted}"


async def test_hint_is_placed_between_table_and_where() -> None:
    """SurrealQL only accepts the hint directly after the target table."""
    store = _store()
    await store.get_reasoning_stats("default")

    for sql, _ in store.latest_queries():
        assert f"FROM reasoning_traces {_HINT} WHERE" in sql, (
            f"hint must sit between table and WHERE to parse: {sql}"
        )


async def test_hinted_query_keeps_both_predicates_and_ordering() -> None:
    """The hint narrows the plan; it must not replace the filters it relies on.

    idx_rtr_model is a composite (brain_id, model) index, so dropping either
    predicate would both widen the scan and leak traces across brains.
    """
    store = _store()
    await store.get_reasoning_stats("default")

    for sql, params in store.latest_queries():
        assert "brain_id = $bid" in sql
        assert "model = $model" in sql
        assert "ORDER BY created_at DESC LIMIT 1" in sql
        # Bound params, never interpolated: brain_id/model reach the engine as
        # values, so a hostile model name cannot break out of the statement.
        assert params["bid"] == "default"
        assert params["model"] in {"claude-fable-5", "claude-sonnet-5"}


async def test_one_hinted_query_per_model() -> None:
    store = _store()
    await store.get_reasoning_stats("default")

    models = [p["model"] for _, p in store.latest_queries()]
    assert sorted(models) == ["claude-fable-5", "claude-sonnet-5"]


async def test_hint_does_not_change_returned_stats() -> None:
    """The optimisation must be invisible in the result shape callers consume."""
    store = _store()
    stats = await store.get_reasoning_stats("default")

    assert stats["total"] == 3
    assert stats["unprocessed"] == 2
    assert stats["by_model"]["claude-fable-5"] == {
        "trace_count": 2,
        "unprocessed": 2,
        "last_trace_at": "2026-03-05T00:00:00",
    }
    assert stats["by_model"]["claude-sonnet-5"] == {
        "trace_count": 1,
        "unprocessed": 0,
        "last_trace_at": "2026-03-02T00:00:00",
    }
    assert stats["by_category"] == {"debugging": 1}


async def test_missing_latest_row_yields_empty_timestamp() -> None:
    """A model with no matching row must not raise (index may be mid-build)."""
    store = _store(latest={})
    stats = await store.get_reasoning_stats("default")

    assert stats["by_model"]["claude-fable-5"]["last_trace_at"] == ""
    assert stats["by_model"]["claude-fable-5"]["trace_count"] == 2


async def test_no_models_issues_no_latest_queries() -> None:
    store = _store(model_groups=[], unproc_groups=[], latest={}, cat_groups=[])
    stats = await store.get_reasoning_stats("default")

    assert store.latest_queries() == []
    assert stats == {"by_model": {}, "by_category": {}, "total": 0, "unprocessed": 0}
