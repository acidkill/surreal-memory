"""Tests for the SQLite reasoning_traces staging storage (run 007, schema v40).

Mirrors test_tool_memory.py: exercises a real in-process SQLiteStorage against a
temporary DB (no mocks), covering insert/dedup, unprocessed retrieval + model
filter, mark-processed, category tagging, prune/cap retention, and stats.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from surreal_memory.storage.sqlite_store import SQLiteStorage
from surreal_memory.utils.timeutils import utcnow

BRAIN = "test-brain"


@pytest.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    """Initialized SQLiteStorage (schema v40) with a test brain."""
    store = SQLiteStorage(tmp_path / "test.db")
    await store.initialize()
    await store._ensure_conn().execute(
        "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (BRAIN, "test", "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    await store._ensure_conn().commit()
    store.set_brain(BRAIN)
    return store


def _trace(
    trace_hash: str = "h1",
    model: str = "claude-fable-5",
    *,
    session_id: str = "s1",
    project: str = "proj",
    task_context: str = "ctx",
    content: str = "restate goal then verify",
    content_chars: int | None = None,
    category: str = "",
    created_at: str = "2026-03-01T10:00:00",
) -> dict:
    d = {
        "trace_hash": trace_hash,
        "model": model,
        "session_id": session_id,
        "project": project,
        "task_context": task_context,
        "content": content,
        "category": category,
        "created_at": created_at,
    }
    if content_chars is not None:
        d["content_chars"] = content_chars
    return d


async def test_migration_created_reasoning_traces_table(storage: SQLiteStorage) -> None:
    """The v40 migration/SCHEMA creates the reasoning_traces table."""
    async with storage._ensure_read_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='reasoning_traces'"
    ) as cur:
        row = await cur.fetchone()
    assert row is not None


async def test_insert_returns_count_and_dedups_within_batch(storage: SQLiteStorage) -> None:
    n = await storage.insert_reasoning_traces(
        BRAIN,
        [_trace("h1"), _trace("h2"), _trace("h1", content="dup")],
    )
    assert n == 2  # h1 inserted once (OR IGNORE on the duplicate), h2 once


async def test_insert_is_idempotent_across_calls(storage: SQLiteStorage) -> None:
    assert await storage.insert_reasoning_traces(BRAIN, [_trace("h1")]) == 1
    assert await storage.insert_reasoning_traces(BRAIN, [_trace("h1")]) == 0


async def test_empty_insert_returns_zero(storage: SQLiteStorage) -> None:
    assert await storage.insert_reasoning_traces(BRAIN, []) == 0


async def test_get_unprocessed_orders_oldest_first_and_limits(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("h3", created_at="2026-03-03T00:00:00"),
            _trace("h1", created_at="2026-03-01T00:00:00"),
            _trace("h2", created_at="2026-03-02T00:00:00"),
        ],
    )
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN, limit=2)
    assert [r["trace_hash"] for r in rows] == ["h1", "h2"]


async def test_get_unprocessed_model_filter(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("h1", model="claude-fable-5"),
            _trace("h2", model="claude-sonnet-5"),
            _trace("h3", model="claude-fable-5"),
        ],
    )
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert {r["trace_hash"] for r in rows} == {"h1", "h3"}


async def test_mark_processed_excludes_from_unprocessed(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1"), _trace("h2")])
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    ids = [r["id"] for r in rows]
    await storage.mark_reasoning_traces_processed(BRAIN, [ids[0]])
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN)
    assert len(remaining) == 1
    assert remaining[0]["id"] == ids[1]


async def test_set_trace_categories(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1"), _trace("h2")])
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    ids = {r["trace_hash"]: r["id"] for r in rows}
    await storage.set_trace_categories(BRAIN, {ids["h1"]: "debugging", ids["h2"]: "planning"})
    stats = await storage.get_reasoning_stats(BRAIN)
    assert stats["by_category"].get("debugging") == 1
    assert stats["by_category"].get("planning") == 1


async def test_task_context_truncated_to_500(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1", task_context="x" * 600)])
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    assert len(rows[0]["task_context"]) == 500


async def test_content_chars_defaults_to_len(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN, [_trace("h1", content="abcde", content_chars=None)]
    )
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    assert rows[0]["content_chars"] == 5


async def test_prune_deletes_only_old_processed(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("old_proc", created_at="2000-01-01T00:00:00"),
            _trace("old_unproc", created_at="2000-01-01T00:00:00"),
            _trace("recent_proc", created_at=(utcnow() - timedelta(days=1)).isoformat()),
        ],
    )
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    by_hash = {r["trace_hash"]: r["id"] for r in rows}
    await storage.mark_reasoning_traces_processed(
        BRAIN, [by_hash["old_proc"], by_hash["recent_proc"]]
    )
    deleted = await storage.prune_reasoning_traces(BRAIN, keep_days=90)
    assert deleted == 1  # only old_proc (processed AND old)
    stats = await storage.get_reasoning_stats(BRAIN)
    assert stats["total"] == 2


async def test_cap_deletes_oldest_processed_beyond_max(storage: SQLiteStorage) -> None:
    traces = [_trace(f"h{i}", created_at=f"2026-03-{i:02d}T00:00:00") for i in range(1, 6)]
    await storage.insert_reasoning_traces(BRAIN, traces)
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    await storage.mark_reasoning_traces_processed(BRAIN, [r["id"] for r in rows])
    deleted = await storage.cap_reasoning_traces(BRAIN, max_total=2)
    assert deleted == 3
    stats = await storage.get_reasoning_stats(BRAIN)
    assert stats["total"] == 2  # the 2 newest survive


async def test_get_reasoning_stats_shapes(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("h1", model="claude-fable-5", created_at="2026-03-01T00:00:00"),
            _trace("h2", model="claude-fable-5", created_at="2026-03-05T00:00:00"),
            _trace("h3", model="claude-sonnet-5", created_at="2026-03-02T00:00:00"),
        ],
    )
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-sonnet-5")
    await storage.mark_reasoning_traces_processed(BRAIN, [rows[0]["id"]])
    stats = await storage.get_reasoning_stats(BRAIN)
    assert stats["total"] == 3
    assert stats["unprocessed"] == 2
    assert stats["by_model"]["claude-fable-5"]["trace_count"] == 2
    assert stats["by_model"]["claude-fable-5"]["unprocessed"] == 2
    assert stats["by_model"]["claude-fable-5"]["last_trace_at"] == "2026-03-05T00:00:00"
    assert stats["by_model"]["claude-sonnet-5"]["unprocessed"] == 0


async def test_get_reasoning_trace_models_distinct_sorted(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("h1", model="claude-sonnet-5"),
            _trace("h2", model="claude-fable-5"),
            _trace("h3", model="claude-fable-5"),
        ],
    )
    assert await storage.get_reasoning_trace_models(BRAIN) == [
        "claude-fable-5",
        "claude-sonnet-5",
    ]


async def test_delete_reasoning_traces_by_model(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("h1", model="claude-fable-5"),
            _trace("h2", model="claude-fable-5"),
            _trace("h3", model="claude-sonnet-5"),
        ],
    )
    deleted = await storage.delete_reasoning_traces_by_model(BRAIN, "claude-fable-5")
    assert deleted == 2
    assert await storage.get_reasoning_trace_models(BRAIN) == ["claude-sonnet-5"]
    # Empty model is a no-op (never a blanket wipe).
    assert await storage.delete_reasoning_traces_by_model(BRAIN, "") == 0


async def test_delete_reasoning_traces_by_model_in_memory() -> None:
    # The in-memory backend is a real, selectable production backend — cover it too.
    from surreal_memory.storage.memory_store import InMemoryStorage

    store = InMemoryStorage()
    store.set_brain(BRAIN)
    await store.insert_reasoning_traces(
        BRAIN,
        [_trace("h1", model="claude-fable-5"), _trace("h2", model="claude-sonnet-5")],
    )
    assert await store.delete_reasoning_traces_by_model(BRAIN, "claude-fable-5") == 1
    assert await store.get_reasoning_trace_models(BRAIN) == ["claude-sonnet-5"]
    assert await store.delete_reasoning_traces_by_model(BRAIN, "") == 0


async def test_traces_are_brain_scoped(storage: SQLiteStorage) -> None:
    await storage._ensure_conn().execute(
        "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("other", "other", "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    await storage._ensure_conn().commit()
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1")])
    await storage.insert_reasoning_traces("other", [_trace("h1")])
    assert (await storage.get_reasoning_stats(BRAIN))["total"] == 1
    assert (await storage.get_reasoning_stats("other"))["total"] == 1


async def test_content_chars_explicit_zero_preserved(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1", content="abcde", content_chars=0)])
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    assert rows[0]["content_chars"] == 0


async def test_empty_trace_hash_is_skipped(storage: SQLiteStorage) -> None:
    n = await storage.insert_reasoning_traces(BRAIN, [_trace(""), _trace("h1")])
    assert n == 1
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    assert [r["trace_hash"] for r in rows] == ["h1"]


# ── reset_reasoning_traces_processed ──────────────────────────────────────────


async def _process_all(storage: SQLiteStorage, brain: str = BRAIN) -> list:
    rows = await storage.get_unprocessed_reasoning_traces(brain)
    await storage.mark_reasoning_traces_processed(brain, [r["id"] for r in rows])
    return rows


async def test_reset_processed_reopens_every_model(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            _trace("h1", model="claude-fable-5"),
            _trace("h2", model="claude-opus-5"),
            _trace("h3", model="claude-opus-5"),
        ],
    )
    await _process_all(storage)
    assert await storage.get_unprocessed_reasoning_traces(BRAIN) == []

    assert await storage.reset_reasoning_traces_processed(BRAIN) == 3
    assert len(await storage.get_unprocessed_reasoning_traces(BRAIN)) == 3


async def test_reset_processed_honors_model_filter(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(
        BRAIN,
        [_trace("h1", model="claude-fable-5"), _trace("h2", model="claude-opus-5")],
    )
    await _process_all(storage)

    assert await storage.reset_reasoning_traces_processed(BRAIN, ["claude-opus-5"]) == 1
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN)
    assert [r["model"] for r in remaining] == ["claude-opus-5"]


async def test_reset_processed_counts_only_flipped_rows(storage: SQLiteStorage) -> None:
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1"), _trace("h2")])
    rows = await storage.get_unprocessed_reasoning_traces(BRAIN)
    await storage.mark_reasoning_traces_processed(BRAIN, [rows[0]["id"]])
    # Only the one processed row is a reset; the already-unprocessed one is not.
    assert await storage.reset_reasoning_traces_processed(BRAIN) == 1


async def test_reset_processed_empty_model_list_is_a_noop(storage: SQLiteStorage) -> None:
    # A resolved model filter that matched nothing must never widen into a
    # blanket reset — that would re-open every other model's backlog.
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1")])
    await _process_all(storage)
    assert await storage.reset_reasoning_traces_processed(BRAIN, []) == 0
    assert await storage.get_unprocessed_reasoning_traces(BRAIN) == []


async def test_reset_processed_is_brain_scoped(storage: SQLiteStorage) -> None:
    await storage._ensure_conn().execute(
        "INSERT OR IGNORE INTO brains (id, name, config, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("other", "other", "{}", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    await storage._ensure_conn().commit()
    await storage.insert_reasoning_traces(BRAIN, [_trace("h1")])
    await storage.insert_reasoning_traces("other", [_trace("h1")])
    await _process_all(storage)
    await _process_all(storage, "other")

    assert await storage.reset_reasoning_traces_processed(BRAIN) == 1
    assert await storage.get_unprocessed_reasoning_traces("other") == []


async def test_reset_processed_in_memory() -> None:
    # The in-memory backend is a real, selectable production backend — cover it too.
    from surreal_memory.storage.memory_store import InMemoryStorage

    store = InMemoryStorage()
    store.set_brain(BRAIN)
    await store.insert_reasoning_traces(
        BRAIN,
        [_trace("h1", model="claude-fable-5"), _trace("h2", model="claude-opus-5")],
    )
    rows = await store.get_unprocessed_reasoning_traces(BRAIN)
    await store.mark_reasoning_traces_processed(BRAIN, [r["id"] for r in rows])

    assert await store.reset_reasoning_traces_processed(BRAIN, []) == 0
    assert await store.reset_reasoning_traces_processed(BRAIN, ["claude-fable-5"]) == 1
    assert [r["model"] for r in await store.get_unprocessed_reasoning_traces(BRAIN)] == [
        "claude-fable-5"
    ]
    assert await store.reset_reasoning_traces_processed(BRAIN) == 1
    assert len(await store.get_unprocessed_reasoning_traces(BRAIN)) == 2
