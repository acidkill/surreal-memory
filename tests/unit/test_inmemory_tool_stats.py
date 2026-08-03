"""Tests for InMemoryStorage's tool-event statistics (#154 finding 2).

`get_tool_stats` / `get_tool_stats_by_period` used to exist only on the
SurrealDB mixin, called through `# type: ignore[attr-defined]` with no
declaration on `NeuralStorage` and no InMemoryStorage implementation -- an
AttributeError on any other backend. These mirror the SurrealDB mixin's own
test shapes (test_surrealdb_tool_events.py) against the real in-memory
buffer instead of a query-routing fake.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

BRAIN = "default"


@pytest.fixture
def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    store.set_brain(BRAIN)
    return store


async def _seed(
    storage: InMemoryStorage,
    tool_name: str,
    *,
    server_name: str = "",
    success: bool = True,
    duration_ms: float = 10.0,
    age_days: float = 0.0,
) -> None:
    await storage.insert_tool_events(
        BRAIN,
        [
            {
                "tool_name": tool_name,
                "server_name": server_name,
                "success": success,
                "duration_ms": duration_ms,
                "created_at": utcnow() - timedelta(days=age_days),
            }
        ],
    )


class TestGetToolStats:
    async def test_empty_brain_returns_zeros(self, storage: InMemoryStorage) -> None:
        stats = await storage.get_tool_stats(BRAIN)

        assert stats == {"total_events": 0, "success_rate": 0, "top_tools": []}

    async def test_computes_rate_and_top_tools(self, storage: InMemoryStorage) -> None:
        await _seed(storage, "Read", success=True, duration_ms=10.0)
        await _seed(storage, "Read", success=True, duration_ms=20.0)
        await _seed(storage, "Read", success=False, duration_ms=30.0)
        await _seed(storage, "Bash", success=True, duration_ms=100.0)

        stats = await storage.get_tool_stats(BRAIN)

        assert stats["total_events"] == 4
        assert stats["success_rate"] == 0.75
        read = next(t for t in stats["top_tools"] if t["tool_name"] == "Read")
        assert read["count"] == 3
        assert read["success_rate"] == round(2 / 3, 2)
        assert read["avg_duration_ms"] == 20  # mean(10, 20, 30)
        # Most-used tool sorts first.
        assert stats["top_tools"][0]["tool_name"] == "Read"

    async def test_zero_successes_is_zero_not_nan(self, storage: InMemoryStorage) -> None:
        await _seed(storage, "Bash", success=False)

        stats = await storage.get_tool_stats(BRAIN)

        tool = stats["top_tools"][0]
        assert tool["success_rate"] == 0.0
        assert isinstance(tool["success_rate"], float)

    async def test_days_filters_the_summary_not_just_the_daily_series(
        self, storage: InMemoryStorage
    ) -> None:
        """The exact #154 finding: days must filter the summary too."""
        await _seed(storage, "Read", age_days=1)
        await _seed(storage, "Read", age_days=100)  # outside a 7-day window

        recent = await storage.get_tool_stats(BRAIN, days=7)
        everything = await storage.get_tool_stats(BRAIN, days=365)

        assert recent["total_events"] == 1
        assert everything["total_events"] == 2

    async def test_default_days_is_30(self, storage: InMemoryStorage) -> None:
        await _seed(storage, "Read", age_days=45)

        stats = await storage.get_tool_stats(BRAIN)

        assert stats["total_events"] == 0

    async def test_events_without_tool_name_are_excluded(self, storage: InMemoryStorage) -> None:
        """`_action_events` is shared with plain (non-tool) action events."""
        await storage.insert_tool_events(BRAIN, [{"created_at": utcnow()}])  # no tool_name
        await _seed(storage, "Read")

        stats = await storage.get_tool_stats(BRAIN)

        assert stats["total_events"] == 1


class TestGetToolStatsByPeriod:
    async def test_empty_brain_returns_empty_list(self, storage: InMemoryStorage) -> None:
        assert await storage.get_tool_stats_by_period(BRAIN) == []

    async def test_groups_by_day_and_tool(self, storage: InMemoryStorage) -> None:
        await _seed(storage, "Read", success=True)
        await _seed(storage, "Read", success=False)
        await _seed(storage, "Bash", success=True)

        daily = await storage.get_tool_stats_by_period(BRAIN, days=30)

        by_tool = {row["tool_name"]: row for row in daily}
        assert by_tool["Read"]["count"] == 2
        assert by_tool["Read"]["success_rate"] == 0.5
        assert by_tool["Bash"]["count"] == 1

    async def test_days_excludes_old_events(self, storage: InMemoryStorage) -> None:
        await _seed(storage, "Read", age_days=1)
        await _seed(storage, "Read", age_days=100)

        daily = await storage.get_tool_stats_by_period(BRAIN, days=7)

        assert sum(row["count"] for row in daily) == 1

    async def test_limit_caps_result_rows(self, storage: InMemoryStorage) -> None:
        for i in range(5):
            await _seed(storage, f"Tool{i}")

        daily = await storage.get_tool_stats_by_period(BRAIN, limit=2)

        assert len(daily) == 2
