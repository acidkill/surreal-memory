"""Tests for event_at original timestamp feature."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from neural_memory.core.brain import Brain
from neural_memory.engine.encoder import MemoryEncoder
from neural_memory.storage.sqlite_store import SQLiteStorage
from neural_memory.utils.timeutils import utcnow


class TestEventTimestampMCP:
    """Test event_at parameter in encoder (simulating MCP handler logic)."""

    async def test_encode_with_custom_timestamp(self, tmp_path: Path) -> None:
        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        encoder = MemoryEncoder(storage, brain.config)
        custom_time = datetime(2026, 3, 2, 8, 0, 0)

        result = await encoder.encode(
            content="Morning meeting with Alice",
            timestamp=custom_time,
        )

        # Fiber should exist
        assert result.fiber is not None
        # Time neurons should use the custom timestamp
        time_neurons = [n for n in result.neurons_created if n.type.value == "time"]
        assert len(time_neurons) > 0

        # At least one time neuron should reference the custom date
        found_custom_time = False
        for tn in time_neurons:
            meta = tn.metadata or {}
            abs_start = meta.get("absolute_start", "")
            if "2026-03-02" in abs_start:
                found_custom_time = True
                break
        assert found_custom_time, "Custom timestamp should propagate to time neurons"

        await storage.close()

    async def test_encode_without_timestamp_uses_now(self, tmp_path: Path) -> None:
        storage = SQLiteStorage(tmp_path / "test.db")
        await storage.initialize()

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        encoder = MemoryEncoder(storage, brain.config)
        now_before = utcnow()

        result = await encoder.encode(content="Quick note")

        # Fiber created_at should be approximately now
        assert result.fiber.created_at >= now_before

        await storage.close()


class TestEventTimestampParsing:
    """Test ISO datetime parsing logic (mirrors handler code)."""

    def test_parse_valid_iso(self) -> None:
        raw = "2026-03-02T08:00:00"
        dt = datetime.fromisoformat(raw)
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.hour == 8

    def test_parse_with_timezone_strips_tzinfo(self) -> None:
        raw = "2026-03-02T08:00:00+07:00"
        dt = datetime.fromisoformat(raw)
        assert dt.tzinfo is not None
        # Handler logic: strip timezone
        dt_naive = dt.replace(tzinfo=None)
        assert dt_naive.tzinfo is None
        assert dt_naive.hour == 8

    def test_parse_date_only(self) -> None:
        raw = "2026-03-02"
        dt = datetime.fromisoformat(raw)
        assert dt.hour == 0
        assert dt.minute == 0

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            datetime.fromisoformat("not-a-date")

    def test_parse_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            datetime.fromisoformat("")
