"""The maturation scan keeps a durable record-ID cursor, including renamed rows."""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.engine.memory_stages import MemoryStage
from surreal_memory.storage.surrealdb.maturation import SurrealDBMaturationMixin


class _PagedMaturations(SurrealDBMaturationMixin):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return "default"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, params))
        assert "START" not in sql
        assert "ORDER BY id ASC LIMIT" in sql
        rows = [row for row in self.rows if row["brain_id"] == params["brain_id"]]
        if "cursor_id" in params:
            rows = [row for row in rows if row["id"] > f"maturation:{params['cursor_id']}"]
        if "stage" in params:
            rows = [row for row in rows if row["stage"] == params["stage"]]
        if "min_rc" in params:
            rows = [row for row in rows if row["rehearsal_count"] >= params["min_rc"]]
        return rows[: int(sql.rsplit(" LIMIT ", 1)[1])]


def _row(record_id: str, *, stage: str = "stm", count: int = 0) -> dict[str, Any]:
    return {
        "id": f"maturation:{record_id}",
        "brain_id": "default",
        "fiber_id": record_id.split("_", 1)[-1],
        "stage": stage,
        "stage_entered_at": "2026-01-01T00:00:00Z",
        "rehearsal_count": count,
        "reinforcement_timestamps": [],
    }


@pytest.mark.asyncio
async def test_keyset_pages_include_renamed_brain_prefix_without_skips() -> None:
    storage = _PagedMaturations([_row("old_a"), _row("old_b"), _row("new_c")])
    storage.rows.sort(key=lambda row: row["id"])
    cursor = None
    collected: list[str] = []
    while page := await storage.find_maturations_after_id(cursor, limit=2):
        collected.extend(record_id for record_id, _record in page)
        cursor = page[-1][0]
    assert collected == sorted(row["id"] for row in storage.rows)
    assert len(collected) == len(set(collected)) == 3
    assert storage.calls[1][1]["cursor_id"] == collected[1].split(":", 1)[1]


@pytest.mark.asyncio
async def test_keyset_filters_and_canonicalises_fiber_id() -> None:
    storage = _PagedMaturations(
        [_row("default_0019e251_94df_4a6b_b446_72c76a083635", stage="semantic", count=4)]
    )
    page = await storage.find_maturations_after_id(
        None, limit=50, stage=MemoryStage.SEMANTIC, min_rehearsal_count=3
    )
    assert len(page) == 1
    assert page[0][1].fiber_id == "0019e251-94df-4a6b-b446-72c76a083635"
    assert storage.calls[0][1]["stage"] == "semantic"
    assert storage.calls[0][1]["min_rc"] == 3
    with pytest.raises(ValueError, match="record ID"):
        await storage.find_maturations_after_id("fiber:not-a-maturation")
