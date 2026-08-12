"""Tests for Baby Mi feedback features (v2.28.0).

Covers:
1. SEMANTIC alternative path (rehearsal count + distinct windows)
2. Bulk remember batch
3. Trust score field
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.memory_types import (
    MemoryType,
    TypedMemory,
    _cap_trust_score,
)
from surreal_memory.engine.memory_stages import (
    _MIN_DISTINCT_WINDOWS,
    _MIN_REHEARSAL_COUNT,
    MaturationRecord,
    MemoryStage,
    compute_stage_transition,
)
from surreal_memory.mcp.constants import MAX_BATCH_SIZE, MAX_BATCH_TOTAL_CHARS
from surreal_memory.utils.timeutils import utcnow

# ─────────────────── #2: SEMANTIC Alternative Path ───────────────────


class TestSemanticAlternativePath:
    """EPISODIC→SEMANTIC via rehearsal count + distinct windows."""

    def test_constants(self):
        assert _MIN_REHEARSAL_COUNT == 15
        assert _MIN_DISTINCT_WINDOWS == 5

    def test_classic_path_still_works(self):
        """3 distinct days + 7 days elapsed → SEMANTIC."""
        now = utcnow()
        entered = now - timedelta(days=8)
        # 3 distinct days of reinforcement
        timestamps = tuple((entered + timedelta(days=d)).isoformat() for d in [1, 3, 5])
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=3,
            reinforcement_timestamps=timestamps,
        )
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.SEMANTIC

    def test_agent_path_high_rehearsals_with_spread(self):
        """15+ rehearsals, 5+ distinct 2h windows, 7+ days → SEMANTIC."""
        now = utcnow()
        entered = now - timedelta(days=8)
        # Generate 15 timestamps spread across different 2h windows on same day
        base = entered + timedelta(days=1)
        timestamps = tuple(
            (base + timedelta(hours=i * 2, minutes=10)).isoformat() for i in range(15)
        )
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=15,
            reinforcement_timestamps=timestamps,
        )
        # Should have >= 5 distinct 2h windows
        assert record.distinct_reinforcement_windows >= 5
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.SEMANTIC

    def test_agent_path_high_rehearsals_no_spread_stays_episodic(self):
        """15 rehearsals but all in same 2h window → still EPISODIC."""
        now = datetime(2026, 3, 10, 12, 0, 0)
        entered = datetime(2026, 3, 1, 12, 0, 0)
        # All timestamps in hour 10:00-11:14 on same day → bucket 10//2=5
        base = datetime(2026, 3, 2, 10, 0, 0)
        timestamps = tuple((base + timedelta(minutes=i * 5)).isoformat() for i in range(15))
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=15,
            reinforcement_timestamps=timestamps,
        )
        assert record.distinct_reinforcement_windows == 1
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.EPISODIC

    def test_agent_path_not_enough_rehearsals(self):
        """10 rehearsals (< 15) with spread → still EPISODIC."""
        now = utcnow()
        entered = now - timedelta(days=8)
        base = entered + timedelta(days=1)
        timestamps = tuple((base + timedelta(hours=i * 2)).isoformat() for i in range(10))
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=10,
            reinforcement_timestamps=timestamps,
        )
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.EPISODIC

    def test_time_gate_enforced(self):
        """15 rehearsals + spread but only 3 days elapsed → EPISODIC."""
        now = utcnow()
        entered = now - timedelta(days=3)  # Only 3 days, not 7
        base = entered + timedelta(hours=1)
        timestamps = tuple((base + timedelta(hours=i * 2)).isoformat() for i in range(15))
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=entered,
            rehearsal_count=15,
            reinforcement_timestamps=timestamps,
        )
        result = compute_stage_transition(record, now=now)
        assert result.stage == MemoryStage.EPISODIC

    def test_distinct_reinforcement_windows_property(self):
        """Test the window bucketing logic."""
        ts = [
            "2026-03-01T08:30:00",  # bucket 4 (8/2=4)
            "2026-03-01T09:30:00",  # bucket 4 (9/2=4) — same
            "2026-03-01T10:30:00",  # bucket 5 (10/2=5)
            "2026-03-01T14:30:00",  # bucket 7 (14/2=7)
            "2026-03-02T08:30:00",  # bucket 4 on different day
        ]
        record = MaturationRecord(
            fiber_id="f1",
            brain_id="b1",
            stage=MemoryStage.EPISODIC,
            stage_entered_at=utcnow(),
            reinforcement_timestamps=tuple(ts),
        )
        # day1:4, day1:5, day1:7, day2:4 = 4 distinct windows
        assert record.distinct_reinforcement_windows == 4


# ─────────────────── #3: Bulk Remember ───────────────────


class TestBulkRemember:
    """smem_remember_batch tool."""

    def test_batch_constants(self):
        assert MAX_BATCH_SIZE == 20
        assert MAX_BATCH_TOTAL_CHARS == 500_000

    @pytest.mark.asyncio
    async def test_batch_empty_array_error(self):
        """Empty memories array should return error."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        result = await handler._remember_batch({"memories": []})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_batch_too_many_items_error(self):
        """More than 20 items should return error."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        memories = [{"content": f"memory {i}"} for i in range(25)]
        result = await handler._remember_batch({"memories": memories})
        assert "error" in result
        assert "25" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_total_chars_limit(self):
        """Total content exceeding 500K should return error."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        memories = [{"content": "x" * 100_000} for _ in range(6)]  # 600K
        result = await handler._remember_batch({"memories": memories})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_batch_partial_success(self):
        """Some items succeed, some fail — partial success."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        # _remember returns success for first, error for second
        async def mock_remember(args):
            if args.get("content") == "good":
                return {"success": True, "fiber_id": "f1", "memory_type": "fact"}
            return {"error": "bad content"}

        handler._remember = AsyncMock(side_effect=mock_remember)

        result = await handler._remember_batch(
            {
                "memories": [
                    {"content": "good"},
                    {"content": "bad"},
                    {"content": "good"},
                ]
            }
        )
        assert result["saved"] == 2
        assert result["failed"] == 1
        assert result["total"] == 3
        assert len(result["results"]) == 3
        assert result["results"][0]["status"] == "ok"
        assert result["results"][1]["status"] == "error"
        assert result["results"][2]["status"] == "ok"
        # Partial success stays a plain success — no "error" key.
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_batch_all_failed_reports_error(self):
        """0/N saved must be distinguishable from success — the accounting bug
        this run measured live: server.py only marks a tool_events row failed
        via result.get("error"), so a 0/20 batch without this key logged as a
        silent success."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)
        handler._remember = AsyncMock(return_value={"error": "bad content"})

        result = await handler._remember_batch(
            {"memories": [{"content": "a"}, {"content": "b"}, {"content": "c"}]}
        )

        assert result["saved"] == 0
        assert result["failed"] == 3
        assert result["success"] is False
        assert "error" in result
        assert "3" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_pass_through_fields_reach_remember(self):
        """trust_score/source_id/context were declared in the tool schema but
        dropped by the allow-list; tier was passed through but undeclared in
        the schema. Both directions fixed."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)
        handler._remember = AsyncMock(
            return_value={"success": True, "fiber_id": "f1", "memory_type": "fact"}
        )

        item = {
            "content": "note",
            "trust_score": 0.7,
            "source_id": "src-1",
            "context": {"reason": "because"},
            "tier": "hot",
        }
        await handler._remember_batch({"memories": [item]})

        single_args = handler._remember.call_args.args[0]
        assert single_args["trust_score"] == 0.7
        assert single_args["source_id"] == "src-1"
        assert single_args["context"] == {"reason": "because"}
        assert single_args["tier"] == "hot"

    @pytest.mark.asyncio
    async def test_batch_non_string_content_is_a_per_item_error(self):
        """A non-str content value used to reach an unguarded len() in the
        total_chars sum and abort the WHOLE call with an uncaught TypeError —
        one bad item taking down every other item in the batch."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)
        handler._remember = AsyncMock(
            return_value={"success": True, "fiber_id": "f1", "memory_type": "fact"}
        )

        result = await handler._remember_batch(
            {"memories": [{"content": "good"}, {"content": 12345}, {"content": "also good"}]}
        )

        assert result["total"] == 3
        assert result["saved"] == 2
        assert result["failed"] == 1
        assert result["results"][1]["status"] == "error"
        assert "string" in result["results"][1]["reason"]
        # The two valid items were still attempted.
        assert handler._remember.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_total_chars_counts_context_not_just_content(self):
        """security-reviewer finding (U4): context is merged into stored
        content server-side, so a tiny "content" with a huge "context" dict
        must not bypass the total_chars guard — context pass-through would
        otherwise amplify a pre-existing gap 20x (MAX_BATCH_SIZE) via batch."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)

        big_context = {"note": "x" * 600_000}
        result = await handler._remember_batch(
            {"memories": [{"content": "tiny", "context": big_context}]}
        )

        assert "error" in result
        assert "too large" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_item_exception_carries_error_type_not_message(self):
        """The pinned literal ("failed to store") and the ban on str(e) stay;
        error_type is an additive, safe (class-name-only) diagnostic."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        handler = MagicMock(spec=ToolHandler)
        handler._remember_batch = ToolHandler._remember_batch.__get__(handler)
        handler._remember = AsyncMock(side_effect=ConnectionError("db unreachable: secret-ish"))

        result = await handler._remember_batch({"memories": [{"content": "a"}]})

        entry = result["results"][0]
        assert entry["reason"] == "failed to store"
        assert entry["error_type"] == "ConnectionError"
        assert "secret-ish" not in str(entry)


# ─────────────────── #5: Trust Score ───────────────────


class TestTrustScore:
    """Trust score field on TypedMemory."""

    def test_trust_score_field_on_typed_memory(self):
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            trust_score=0.8,
        )
        assert tm.trust_score is not None
        assert tm.trust_score <= 0.9  # Capped by user_input ceiling

    def test_trust_score_none_by_default(self):
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
        )
        assert tm.trust_score is None

    def test_cap_trust_score_user_input(self):
        assert _cap_trust_score(1.0, "user_input") == 0.9

    def test_cap_trust_score_verified(self):
        assert _cap_trust_score(1.0, "verified") == 1.0

    def test_cap_trust_score_ai_inference(self):
        assert _cap_trust_score(0.9, "ai_inference") == 0.7

    def test_cap_trust_score_auto_capture(self):
        assert _cap_trust_score(0.8, "auto_capture") == 0.5

    def test_cap_trust_score_none_passthrough(self):
        assert _cap_trust_score(None, "user_input") is None

    def test_cap_trust_score_clamps_negative(self):
        result = _cap_trust_score(-0.5, "verified")
        assert result == 0.0

    def test_cap_trust_score_clamps_over_one(self):
        result = _cap_trust_score(1.5, "verified")
        assert result == 1.0

    def test_cap_trust_score_mcp_source(self):
        """mcp:claude_code → mcp_tool ceiling 0.8."""
        result = _cap_trust_score(0.95, "mcp:claude_code")
        assert result == 0.8

    def test_typed_memory_source_field(self):
        tm = TypedMemory.create(
            fiber_id="f1",
            memory_type=MemoryType.FACT,
            source="user_input",
        )
        assert tm.source == "user_input"
