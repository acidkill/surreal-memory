"""Unit tests for the SurrealDB reasoning-traces mixin (no live DB required).

Mirrors test_surrealdb_tool_events.py: a fake store routes ``_query`` by SQL
fragment and records inserts/updates/deletes, so the mixin's SurrealQL and
mapping logic are verified without a running SurrealDB. Live-engine behaviour is
covered separately by the real-db test runner.
"""

from __future__ import annotations

from typing import Any

from surreal_memory.storage.surrealdb.reasoning_traces import SurrealDBReasoningTracesMixin


class _RTStore(SurrealDBReasoningTracesMixin):
    """Routes _query by SQL fragment; records inserts and mutating queries."""

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        self.inserts: list[tuple[str, dict[str, Any]]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return "default"

    def _ensure_conn(self) -> Any:
        store = self

        class _Conn:
            async def insert(self, table: str, data: dict[str, Any]) -> None:
                store.inserts.append((table, data))

        return _Conn()

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        c = self.cfg
        if sql.startswith("UPDATE reasoning_traces SET processed = true"):
            return []
        if sql.startswith("UPDATE reasoning_traces SET processed = false"):
            return []
        if sql.startswith("SELECT count() AS c") and sql.endswith(
            # Reset's count-select, both shapes. Checked before prune's, which
            # shares the prefix but carries a created_at cutoff.
            ("AND processed = true GROUP ALL", "AND model IN $models GROUP ALL")
        ):
            return [{"c": c.get("reset_count", 0)}]
        if sql.startswith("UPDATE reasoning_traces SET category"):
            return []
        if sql.startswith("DELETE reasoning_traces"):
            return []
        if (
            sql.startswith("SELECT trace_hash FROM reasoning_traces")
            and "trace_hash IN $hashes" in sql
        ):
            return [{"trace_hash": h} for h in c.get("existing", [])]
        if "processed = true" in sql and "ORDER BY created_at ASC" in sql:
            return [{"trace_hash": h} for h in c.get("cap_victims", [])]
        if "processed = false" in sql and "ORDER BY created_at ASC" in sql:
            rows = list(c.get("unprocessed", []))
            if params.get("model"):
                rows = [r for r in rows if r.get("model") == params["model"]]
            return rows
        if "AND model = $model GROUP ALL" in sql:
            return [{"c": c.get("del_count", 0)}]
        if "processed = true AND created_at < $cutoff GROUP ALL" in sql:
            return [{"c": c.get("prune_count", 0)}]
        if "count() AS c FROM reasoning_traces WHERE brain_id = $bid GROUP ALL" in sql:
            return [{"c": c.get("total", 0)}]
        if "AND processed = false GROUP BY model" in sql:
            return c.get("unproc_groups", [])
        if "count() AS cnt FROM reasoning_traces WHERE brain_id = $bid GROUP BY model" in sql:
            return c.get("model_groups", [])
        if "ORDER BY created_at DESC LIMIT 1" in sql:
            return c.get("latest", {}).get(params.get("model"), [])
        if "GROUP BY category" in sql:
            return c.get("cat_groups", [])
        if sql.startswith(
            "SELECT model FROM reasoning_traces WHERE brain_id = $bid GROUP BY model"
        ):
            return c.get("distinct_models", [])
        return []


def _tr(trace_hash: str, model: str = "claude-fable-5", **kw: Any) -> dict[str, Any]:
    base = {
        "trace_hash": trace_hash,
        "model": model,
        "session_id": "s1",
        "project": "proj",
        "task_context": "ctx",
        "content": "restate goal then verify",
        "created_at": "2026-03-01T10:00:00",
    }
    base.update(kw)
    return base


async def test_insert_skips_existing_and_batch_dupes() -> None:
    store = _RTStore(existing=["h2"])
    n = await store.insert_reasoning_traces(
        "default", [_tr("h1"), _tr("h2"), _tr("h1", content="dup"), _tr("h3")]
    )
    # h2 already exists; h1 duplicated in batch (inserted once); h3 new.
    assert n == 2
    inserted_hashes = {data["trace_hash"] for _, data in store.inserts}
    assert inserted_hashes == {"h1", "h3"}
    for table, data in store.inserts:
        assert table == "reasoning_traces"
        assert data["processed"] is False
        assert data["brain_id"] == "default"


async def test_insert_empty_returns_zero() -> None:
    store = _RTStore()
    assert await store.insert_reasoning_traces("default", []) == 0
    assert store.inserts == []


async def test_delete_by_model_counts_and_deletes() -> None:
    store = _RTStore(del_count=3)
    n = await store.delete_reasoning_traces_by_model("default", "claude-fable-5")
    assert n == 3
    # A parameterized DELETE was issued for that model (no string interpolation).
    assert any(
        sql.startswith("DELETE reasoning_traces") and p.get("model") == "claude-fable-5"
        for sql, p in store.queries
    )


async def test_delete_by_model_empty_is_noop() -> None:
    store = _RTStore(del_count=99)
    assert await store.delete_reasoning_traces_by_model("default", "") == 0
    assert store.queries == []  # short-circuits before issuing any query


async def test_insert_truncates_task_context_and_computes_chars() -> None:
    store = _RTStore()
    await store.insert_reasoning_traces(
        "default", [_tr("h1", task_context="x" * 600, content="abcde")]
    )
    data = store.inserts[0][1]
    assert len(data["task_context"]) == 500
    assert data["content_chars"] == 5


async def test_get_unprocessed_maps_rows_and_filters_model() -> None:
    unprocessed = [
        {"trace_hash": "h1", "model": "claude-fable-5", "content": "c1", "created_at": "t1"},
        {"trace_hash": "h2", "model": "claude-sonnet-5", "content": "c2", "created_at": "t2"},
    ]
    store = _RTStore(unprocessed=unprocessed)
    rows = await store.get_unprocessed_reasoning_traces("default", model="claude-fable-5")
    assert len(rows) == 1
    assert rows[0]["id"] == "h1"
    assert rows[0]["trace_hash"] == "h1"
    # limit is interpolated into the SQL
    assert any("LIMIT" in sql for sql, _ in store.queries)


async def test_mark_processed_issues_update_by_trace_hash() -> None:
    store = _RTStore()
    await store.mark_reasoning_traces_processed("default", ["h1", "h2"])
    upd = [
        q for q in store.queries if q[0].startswith("UPDATE reasoning_traces SET processed = true")
    ]
    assert upd and upd[0][1]["ids"] == ["h1", "h2"]


async def test_mark_processed_empty_noop() -> None:
    store = _RTStore()
    await store.mark_reasoning_traces_processed("default", [])
    assert store.queries == []


async def test_set_categories_updates_per_hash() -> None:
    store = _RTStore()
    await store.set_trace_categories("default", {"h1": "debugging", "h2": "planning"})
    cat_updates = [
        q for q in store.queries if q[0].startswith("UPDATE reasoning_traces SET category")
    ]
    assert len(cat_updates) == 2
    seen = {(q[1]["id"], q[1]["cat"]) for q in cat_updates}
    assert seen == {("h1", "debugging"), ("h2", "planning")}


async def test_prune_returns_count_and_deletes() -> None:
    store = _RTStore(prune_count=3)
    deleted = await store.prune_reasoning_traces("default", keep_days=90)
    assert deleted == 3
    assert any(q[0].startswith("DELETE reasoning_traces") for q in store.queries)


async def test_cap_deletes_oldest_excess() -> None:
    store = _RTStore(total=5, cap_victims=["h1", "h2", "h3"])
    deleted = await store.cap_reasoning_traces("default", max_total=2)
    assert deleted == 3
    dels = [q for q in store.queries if q[0].startswith("DELETE reasoning_traces")]
    assert dels and dels[0][1]["ids"] == ["h1", "h2", "h3"]


async def test_cap_noop_when_under_limit() -> None:
    store = _RTStore(total=2)
    assert await store.cap_reasoning_traces("default", max_total=5) == 0
    assert not any(q[0].startswith("DELETE reasoning_traces") for q in store.queries)


async def test_get_reasoning_stats_assembles_all_sections() -> None:
    store = _RTStore(
        model_groups=[
            {"model": "claude-fable-5", "cnt": 2},
            {"model": "claude-sonnet-5", "cnt": 1},
        ],
        unproc_groups=[{"model": "claude-fable-5", "cnt": 2}],
        latest={
            "claude-fable-5": [{"created_at": "2026-03-05T00:00:00"}],
            "claude-sonnet-5": [{"created_at": "2026-03-02T00:00:00"}],
        },
        cat_groups=[{"category": "debugging", "cnt": 1}, {"category": "planning", "cnt": 2}],
    )
    stats = await store.get_reasoning_stats("default")
    assert stats["total"] == 3
    assert stats["unprocessed"] == 2
    assert stats["by_model"]["claude-fable-5"]["trace_count"] == 2
    assert stats["by_model"]["claude-fable-5"]["unprocessed"] == 2
    assert stats["by_model"]["claude-fable-5"]["last_trace_at"] == "2026-03-05T00:00:00"
    assert stats["by_model"]["claude-sonnet-5"]["unprocessed"] == 0
    assert stats["by_category"] == {"debugging": 1, "planning": 2}


async def test_get_reasoning_trace_models_distinct_sorted_nonempty() -> None:
    store = _RTStore(
        distinct_models=[
            {"model": "claude-sonnet-5"},
            {"model": "claude-fable-5"},
            {"model": ""},
        ]
    )
    assert await store.get_reasoning_trace_models("default") == [
        "claude-fable-5",
        "claude-sonnet-5",
    ]


async def test_insert_preserves_explicit_content_chars_zero() -> None:
    store = _RTStore()
    await store.insert_reasoning_traces("default", [_tr("h1", content="abcde", content_chars=0)])
    assert store.inserts[0][1]["content_chars"] == 0


async def test_insert_skips_empty_trace_hash() -> None:
    store = _RTStore()
    n = await store.insert_reasoning_traces("default", [_tr(""), _tr("h1")])
    assert n == 1
    assert {data["trace_hash"] for _, data in store.inserts} == {"h1"}


# ── reset_reasoning_traces_processed ──────────────────────────────────────────


async def test_reset_processed_counts_then_updates() -> None:
    store = _RTStore(reset_count=7)
    assert await store.reset_reasoning_traces_processed("default") == 7

    counted, updated = store.queries
    assert counted[0].startswith("SELECT count() AS c FROM reasoning_traces")
    assert updated[0] == (
        "UPDATE reasoning_traces SET processed = false WHERE brain_id = $bid AND processed = true"
    )
    assert counted[1] == updated[1] == {"bid": "default"}


async def test_reset_processed_scopes_to_models() -> None:
    store = _RTStore(reset_count=2)
    assert await store.reset_reasoning_traces_processed("default", ["claude-opus-5"]) == 2

    sql, params = store.queries[-1]
    assert "AND model IN $models" in sql
    assert params == {"bid": "default", "models": ["claude-opus-5"]}


async def test_reset_processed_empty_model_list_never_queries() -> None:
    # An empty resolved filter must not fall through to a blanket reset.
    store = _RTStore()
    assert await store.reset_reasoning_traces_processed("default", []) == 0
    assert store.queries == []
