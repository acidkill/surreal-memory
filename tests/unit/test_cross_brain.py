"""Tests for cross-brain recall engine."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.engine.cross_brain import (
    CrossBrainFiber,
    CrossBrainResult,
    _dedup_fibers,
    _query_single_brain,
    cross_brain_recall,
)
from surreal_memory.engine.retrieval_types import DepthLevel


class TestCrossBrainFiber:
    """Tests for CrossBrainFiber dataclass."""

    def test_defaults(self) -> None:
        f = CrossBrainFiber(
            fiber_id="f1",
            source_brain="test",
            summary="hello",
            confidence=0.8,
        )
        assert f.content_hash == 0
        assert f.source_brain == "test"

    def test_frozen(self) -> None:
        f = CrossBrainFiber(
            fiber_id="f1",
            source_brain="test",
            summary="hello",
            confidence=0.8,
        )
        with pytest.raises(AttributeError):
            f.summary = "changed"  # type: ignore[misc]


class TestCrossBrainResult:
    """Tests for CrossBrainResult dataclass."""

    def test_defaults(self) -> None:
        r = CrossBrainResult(
            query="test",
            brains_queried=["a"],
            fibers=[],
        )
        assert r.total_neurons_activated == 0
        assert r.merged_context == ""
        assert r.errors == {}

    def test_frozen(self) -> None:
        r = CrossBrainResult(query="q", brains_queried=[], fibers=[])
        with pytest.raises(AttributeError):
            r.query = "changed"  # type: ignore[misc]


class TestDedupFibers:
    """Tests for fiber deduplication."""

    def test_no_duplicates(self) -> None:
        fibers = [
            CrossBrainFiber("f1", "brain1", "alpha", 0.9, content_hash=0),
            CrossBrainFiber("f2", "brain2", "beta", 0.8, content_hash=0),
        ]
        result = _dedup_fibers(fibers)
        assert len(result) == 2

    def test_dedup_keeps_higher_confidence(self) -> None:
        """When two fibers have near-duplicate hashes, keep higher confidence."""
        # Use identical hashes to simulate near-duplicates
        fibers = [
            CrossBrainFiber("f1", "brain1", "alpha", 0.7, content_hash=12345),
            CrossBrainFiber("f2", "brain2", "alpha copy", 0.9, content_hash=12345),
        ]
        result = _dedup_fibers(fibers)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_dedup_zero_hash_not_deduped(self) -> None:
        """Fibers with content_hash=0 should not be deduplicated."""
        fibers = [
            CrossBrainFiber("f1", "brain1", "alpha", 0.9, content_hash=0),
            CrossBrainFiber("f2", "brain2", "beta", 0.8, content_hash=0),
        ]
        result = _dedup_fibers(fibers)
        assert len(result) == 2

    def test_dedup_different_hashes_kept(self) -> None:
        """Fibers with very different hashes should be kept."""
        # Hashes must differ by > 10 bits (DEFAULT_THRESHOLD)
        # 0xAAAAAAAAAAAAAAAA and 0x5555555555555555 differ in all 64 bits
        fibers = [
            CrossBrainFiber("f1", "brain1", "alpha", 0.9, content_hash=0xAAAAAAAAAAAAAAAA),
            CrossBrainFiber("f2", "brain2", "beta", 0.8, content_hash=0x5555555555555555),
        ]
        result = _dedup_fibers(fibers)
        assert len(result) == 2


class TestQuerySingleBrain:
    """Tests for _query_single_brain — the backend-aware single-brain query."""

    async def test_uses_backend_aware_shared_storage_not_sqlite_file(self) -> None:
        """Should route through get_shared_storage (backend-aware) instead of
        unconditionally opening a throwaway SQLiteStorage file, and should
        return the real fibers/context the pipeline produced."""
        mock_brain = MagicMock()
        mock_brain.name = "brain1"
        mock_brain.config = MagicMock()

        mock_storage = MagicMock()
        mock_storage.find_brain_by_name = AsyncMock(return_value=mock_brain)
        mock_storage.set_brain = MagicMock()
        mock_storage.get_fiber = AsyncMock(
            return_value=MagicMock(summary="real memory content", content_hash=0)
        )

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.fibers_matched = ["f1"]
        mock_pipeline_result.confidence = 0.85
        mock_pipeline_result.neurons_activated = 7
        mock_pipeline_result.context = "actual retrieved context"

        with (
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ) as mock_get_shared_storage,
            patch("surreal_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline_cls.return_value.query = AsyncMock(return_value=mock_pipeline_result)

            brain_name, fibers, neurons, context, error = await _query_single_brain(
                "brain1", "find real content", DepthLevel.CONTEXT, 500
            )

        mock_get_shared_storage.assert_awaited_once_with("brain1")
        assert brain_name == "brain1"
        assert len(fibers) == 1
        assert fibers[0].summary == "real memory content"
        assert neurons == 7
        assert context == "actual retrieved context"
        assert error is None
        # The shared/cached storage instance must not be torn down here — its
        # lifecycle belongs to the shared-storage cache, not this one query.
        mock_storage.close.assert_not_called()

    async def test_brain_not_found_returns_empty_without_error(self) -> None:
        """A brain absent from the resolved storage is a clean empty result,
        not an error — distinct from an actual exception."""
        mock_storage = MagicMock()
        mock_storage.find_brain_by_name = AsyncMock(return_value=None)

        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            new_callable=AsyncMock,
            return_value=mock_storage,
        ):
            brain_name, fibers, neurons, context, error = await _query_single_brain(
                "ghost", "q", DepthLevel.CONTEXT, 500
            )

        assert (brain_name, fibers, neurons, context, error) == ("ghost", [], 0, "", None)

    async def test_exception_returns_error_message_and_logs_at_warning(self, caplog) -> None:
        """A real failure must surface a short error message (for
        CrossBrainResult.errors) and log at WARNING, not DEBUG — DEBUG is
        invisible by default and was hiding this exact class of bug."""
        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.cross_brain"):
                brain_name, fibers, neurons, context, error = await _query_single_brain(
                    "brain1", "q", DepthLevel.CONTEXT, 500
                )

        assert brain_name == "brain1"
        assert fibers == []
        assert neurons == 0
        assert context == ""
        assert error == "boom"
        matching = [r for r in caplog.records if "Cross-brain query failed" in r.message]
        assert matching, "expected a 'Cross-brain query failed' log record"
        assert all(r.levelno >= logging.WARNING for r in matching)


class TestCrossBrainRecall:
    """Tests for the cross_brain_recall function."""

    async def test_no_valid_brains(self) -> None:
        """Empty brains list returns empty result."""
        config = MagicMock()

        with patch(
            "surreal_memory.unified_config.list_available_brains",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=["nonexistent"],
                query="test query",
            )
        assert result.brains_queried == []
        assert result.fibers == []
        assert "No valid brains" in result.merged_context

    async def test_caps_at_five_brains(self) -> None:
        """Should cap brain names at MAX_CROSS_BRAINS (5)."""
        config = MagicMock()

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=[f"brain{i}" for i in range(10)],
            ),
            patch(
                "surreal_memory.engine.cross_brain._query_single_brain",
                new_callable=AsyncMock,
                return_value=("brainX", [], 0, "", None),
            ) as mock_query,
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=[f"brain{i}" for i in range(10)],
                query="test",
            )
        # Even though 10 names were passed and all 10 are "available", only 5
        # (MAX_CROSS_BRAINS) should ever be queried.
        assert mock_query.await_count <= 5
        assert len(result.brains_queried) <= 5

    async def test_skips_nonexistent_brains_without_disk_check(self) -> None:
        """Should skip brains not returned by list_available_brains, without
        ever consulting a per-brain on-disk path (SurrealDB brains have
        none)."""
        config = MagicMock()
        config.get_brain_db_path = MagicMock(
            side_effect=AssertionError("must not check a per-brain disk path")
        )
        config.list_brains = MagicMock(
            side_effect=AssertionError("must not use the disk-file brain listing directly")
        )

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=["exists"],
            ),
            patch(
                "surreal_memory.engine.cross_brain._query_single_brain",
                new_callable=AsyncMock,
                return_value=("exists", [], 5, "some context", None),
            ) as mock_query,
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=["exists", "nonexistent"],
                query="test",
            )
        assert "exists" in result.brains_queried
        assert "nonexistent" not in result.brains_queried
        mock_query.assert_awaited_once()
        config.get_brain_db_path.assert_not_called()
        config.list_brains.assert_not_called()

    async def test_merges_results_from_multiple_brains(self) -> None:
        """Results from multiple brains should be merged."""
        config = MagicMock()

        fiber1 = CrossBrainFiber("f1", "brain1", "memory from brain1", 0.9)
        fiber2 = CrossBrainFiber("f2", "brain2", "memory from brain2", 0.7)

        async def mock_query(name, query, depth, max_tokens, tags=None, near=None):
            if name == "brain1":
                return ("brain1", [fiber1], 10, "[brain1] context", None)
            return ("brain2", [fiber2], 5, "[brain2] context", None)

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=["brain1", "brain2"],
            ),
            patch(
                "surreal_memory.engine.cross_brain._query_single_brain",
                side_effect=mock_query,
            ),
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=["brain1", "brain2"],
                query="test",
            )

        assert len(result.brains_queried) == 2
        assert len(result.fibers) == 2
        assert result.total_neurons_activated == 15
        # Fibers should be sorted by confidence (0.9 first)
        assert result.fibers[0].confidence >= result.fibers[1].confidence
        assert result.errors == {}

    async def test_invalid_depth_defaults_to_context(self) -> None:
        """Invalid depth should default to CONTEXT."""
        config = MagicMock()

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=["brain1"],
            ),
            patch(
                "surreal_memory.engine.cross_brain._query_single_brain",
                new_callable=AsyncMock,
                return_value=("brain1", [], 0, "", None),
            ),
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=["brain1"],
                query="test",
                depth=99,  # Invalid
            )
        assert result.brains_queried == ["brain1"]

    async def test_handles_brain_query_failure(self) -> None:
        """An internally-handled brain-query failure surfaces in
        CrossBrainResult.errors rather than raising or silently
        disappearing into an empty result indistinguishable from success."""
        config = MagicMock()

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=["brain1"],
            ),
            patch(
                "surreal_memory.engine.cross_brain._query_single_brain",
                new_callable=AsyncMock,
                return_value=("brain1", [], 0, "", "DB corrupted"),
            ),
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=["brain1"],
                query="test",
            )
        assert result.brains_queried == ["brain1"]
        assert result.fibers == []
        assert result.errors == {"brain1": "DB corrupted"}

    async def test_partial_failure_reports_error_for_failed_brain_only(self) -> None:
        """When one brain fails and another succeeds, the successful brain's
        results still come back, and only the failed brain appears in
        errors."""
        config = MagicMock()
        fiber_ok = CrossBrainFiber("f1", "brain_ok", "good memory", 0.8)

        async def mock_query(name, query, depth, max_tokens, tags=None, near=None):
            if name == "brain_ok":
                return ("brain_ok", [fiber_ok], 4, "[brain_ok] context", None)
            return ("brain_bad", [], 0, "", "connection refused")

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=["brain_ok", "brain_bad"],
            ),
            patch(
                "surreal_memory.engine.cross_brain._query_single_brain",
                side_effect=mock_query,
            ),
        ):
            result = await cross_brain_recall(
                config=config,
                brain_names=["brain_ok", "brain_bad"],
                query="test",
            )

        assert len(result.fibers) == 1
        assert result.fibers[0].fiber_id == "f1"
        assert result.errors == {"brain_bad": "connection refused"}
        assert "brain_ok" not in result.errors

    async def test_backend_aware_recall_returns_real_results_not_empty(self) -> None:
        """End-to-end through cross_brain_recall with storage_backend !=
        sqlite: a brain with real matching content must return actual
        fibers/context, not an empty 'no relevant memories' result."""
        config = MagicMock()
        config.storage_backend = "surrealdb"

        mock_brain = MagicMock()
        mock_brain.name = "brain1"
        mock_brain.config = MagicMock()

        mock_storage = MagicMock()
        mock_storage.find_brain_by_name = AsyncMock(return_value=mock_brain)
        mock_storage.set_brain = MagicMock()
        mock_storage.get_fiber = AsyncMock(
            return_value=MagicMock(summary="real memory content", content_hash=0)
        )

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.fibers_matched = ["f1"]
        mock_pipeline_result.confidence = 0.85
        mock_pipeline_result.neurons_activated = 7
        mock_pipeline_result.context = "actual retrieved context"

        with (
            patch(
                "surreal_memory.unified_config.list_available_brains",
                new_callable=AsyncMock,
                return_value=["brain1"],
            ),
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                new_callable=AsyncMock,
                return_value=mock_storage,
            ),
            patch("surreal_memory.engine.retrieval.ReflexPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline_cls.return_value.query = AsyncMock(return_value=mock_pipeline_result)

            result = await cross_brain_recall(
                config=config,
                brain_names=["brain1"],
                query="find real content",
            )

        assert result.brains_queried == ["brain1"]
        assert len(result.fibers) == 1
        assert result.fibers[0].summary == "real memory content"
        assert "actual retrieved context" in result.merged_context
        assert result.merged_context != "No relevant memories found."
        assert result.errors == {}


class TestCrossBrainRecallHandler:
    """Tests for the _cross_brain_recall method in ToolHandler."""

    async def test_recall_with_brains_param_triggers_cross_brain(self) -> None:
        """_recall with brains param should call _cross_brain_recall."""
        from unittest.mock import MagicMock

        from surreal_memory.mcp.tool_handlers import ToolHandler

        class MockServer(ToolHandler):
            def __init__(self):
                self.config = MagicMock()
                self.hooks = MagicMock()
                self.hooks.emit = AsyncMock()

            async def get_storage(self):
                return MagicMock()

        server = MockServer()
        with patch.object(
            server,
            "_cross_brain_recall",
            new_callable=AsyncMock,
            return_value={"answer": "cross-brain result", "cross_brain": True},
        ) as mock_cross:
            result = await server._recall(
                {
                    "query": "test query",
                    "brains": ["brain1", "brain2"],
                }
            )
            mock_cross.assert_called_once()
            assert result["cross_brain"] is True

    async def test_recall_without_brains_uses_normal_path(self) -> None:
        """_recall without brains param should use normal single-brain path."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        class MockServer(ToolHandler):
            def __init__(self):
                self.config = MagicMock()
                self.hooks = MagicMock()
                self.hooks.emit = AsyncMock()

            async def get_storage(self):
                storage = MagicMock()
                storage._current_brain_id = None
                storage.get_brain = AsyncMock(return_value=None)
                return storage

        server = MockServer()
        # Without brains param, should hit normal path and return error (no brain)
        result = await server._recall({"query": "test"})
        assert result == {"error": "No brain configured"}

    async def test_cross_brain_recall_includes_errors_key_when_present(self) -> None:
        """When CrossBrainResult.errors is non-empty, the response dict must
        surface it under 'errors' so a real failure isn't indistinguishable
        from a genuine zero-match result."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        class MockServer(ToolHandler):
            def __init__(self):
                self.config = MagicMock()
                self.hooks = MagicMock()
                self.hooks.emit = AsyncMock()

        server = MockServer()
        fake_result = CrossBrainResult(
            query="q",
            brains_queried=["ok", "bad"],
            fibers=[],
            errors={"bad": "boom"},
        )
        with patch(
            "surreal_memory.engine.cross_brain.cross_brain_recall",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            response = await server._cross_brain_recall({"query": "q"}, ["ok", "bad"])

        assert response["errors"] == {"bad": "boom"}

    async def test_cross_brain_recall_omits_errors_key_when_empty(self) -> None:
        """When no brain query failed, the 'errors' key must be absent
        entirely — not clutter every response with an empty dict."""
        from surreal_memory.mcp.tool_handlers import ToolHandler

        class MockServer(ToolHandler):
            def __init__(self):
                self.config = MagicMock()
                self.hooks = MagicMock()
                self.hooks.emit = AsyncMock()

        server = MockServer()
        fake_result = CrossBrainResult(
            query="q",
            brains_queried=["ok"],
            fibers=[],
            errors={},
        )
        with patch(
            "surreal_memory.engine.cross_brain.cross_brain_recall",
            new_callable=AsyncMock,
            return_value=fake_result,
        ):
            response = await server._cross_brain_recall({"query": "q"}, ["ok"])

        assert "errors" not in response
