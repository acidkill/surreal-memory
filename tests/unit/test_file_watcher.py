"""Tests for file watcher — watch_state + file_watcher + watch_handler."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from neural_memory.engine.watch_state import WatchStateTracker

# ── WatchStateTracker tests ──────────────────────────


class TestWatchStateTracker:
    @pytest.fixture
    async def tracker(self) -> Any:
        db = await aiosqlite.connect(":memory:")
        tracker = WatchStateTracker(db)
        await tracker.initialize()
        yield tracker
        await db.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_table(self, tracker: WatchStateTracker) -> None:
        cursor = await tracker._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='watch_state'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_should_process_new_file(self, tracker: WatchStateTracker) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            assert await tracker.should_process(path) is True
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_should_process_unchanged(self, tracker: WatchStateTracker) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            mtime = path.stat().st_mtime
            await tracker.mark_processed(path, mtime, 12345, 3)
            assert await tracker.should_process(path) is False
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_mark_processed(self, tracker: WatchStateTracker) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test")
            path = Path(f.name)

        try:
            await tracker.mark_processed(path, 1000.0, 99999, 5)
            files = await tracker.list_watched_files()
            assert len(files) == 1
            assert files[0].neuron_count == 5
            assert files[0].simhash == 99999
            assert files[0].status == "active"
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_mark_deleted(self, tracker: WatchStateTracker) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test")
            path = Path(f.name)

        try:
            await tracker.mark_processed(path, 1000.0, 99999, 5)
            await tracker.mark_deleted(path)
            files = await tracker.list_watched_files(status="deleted")
            assert len(files) == 1
            assert files[0].status == "deleted"
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_list_by_status(self, tracker: WatchStateTracker) -> None:
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
                f.write(f"content {i}".encode())
                path = Path(f.name)
            await tracker.mark_processed(path, 1000.0 + i, i, 1)

        all_files = await tracker.list_watched_files()
        assert len(all_files) == 3

        active = await tracker.list_watched_files(status="active")
        assert len(active) == 3

    @pytest.mark.asyncio
    async def test_get_stats(self, tracker: WatchStateTracker) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test")
            path = Path(f.name)

        try:
            await tracker.mark_processed(path, 1000.0, 0, 10)
            stats = await tracker.get_stats()
            assert stats["total_files"] == 1
            assert stats["total_neurons"] == 10
        finally:
            path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_upsert_on_reprocess(self, tracker: WatchStateTracker) -> None:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test")
            path = Path(f.name)

        try:
            await tracker.mark_processed(path, 1000.0, 111, 5)
            await tracker.mark_processed(path, 2000.0, 222, 8)
            files = await tracker.list_watched_files()
            assert len(files) == 1
            assert files[0].neuron_count == 8
            assert files[0].simhash == 222
        finally:
            path.unlink(missing_ok=True)


# ── FileWatcher tests ────────────────────────────────


class TestFileWatcher:
    @pytest.fixture
    def watcher(self) -> Any:
        from neural_memory.engine.file_watcher import FileWatcher, WatchConfig

        trainer = AsyncMock()
        trainer.train_file = AsyncMock(return_value=MagicMock(neurons_created=3, chunks_encoded=2))
        state = AsyncMock(spec=WatchStateTracker)
        state.should_process_with_simhash = AsyncMock(return_value=True)
        state.mark_processed = AsyncMock()
        state.initialize = AsyncMock()

        return FileWatcher(trainer, state, WatchConfig())

    def test_validate_valid_dir(self, watcher: Any) -> None:
        with tempfile.TemporaryDirectory() as d:
            assert watcher.validate_path(Path(d)) is None

    def test_validate_nonexistent(self, watcher: Any) -> None:
        result = watcher.validate_path(Path("/nonexistent/path/xyz"))
        assert result is not None
        assert "not a directory" in result

    def test_validate_file_not_dir(self, watcher: Any) -> None:
        with tempfile.NamedTemporaryFile() as f:
            result = watcher.validate_path(Path(f.name))
            assert result is not None

    def test_should_ignore(self, watcher: Any) -> None:
        assert watcher._should_ignore(Path("/project/node_modules/foo.js")) is True
        assert watcher._should_ignore(Path("/project/.git/config")) is True
        assert watcher._should_ignore(Path("/project/src/main.py")) is False

    def test_extension_filter(self, watcher: Any) -> None:

        assert ".md" in watcher._config.extensions
        assert ".py" not in watcher._config.extensions

    @pytest.mark.asyncio
    async def test_process_path_empty_dir(self, watcher: Any) -> None:
        with tempfile.TemporaryDirectory() as d:
            results = await watcher.process_path(Path(d))
            assert results == []

    @pytest.mark.asyncio
    async def test_process_path_with_files(self, watcher: Any) -> None:
        with tempfile.TemporaryDirectory() as d:
            md_file = Path(d) / "test.md"
            md_file.write_text("# Hello\nSome content here")
            py_file = Path(d) / "test.py"
            py_file.write_text("print('hello')")

            results = await watcher.process_path(Path(d))
            # Only .md should be processed, not .py
            assert len(results) == 1
            assert results[0].success is True

    @pytest.mark.asyncio
    async def test_process_skips_unchanged(self, watcher: Any) -> None:
        watcher._state.should_process_with_simhash = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as d:
            md_file = Path(d) / "test.md"
            md_file.write_text("# Test")

            results = await watcher.process_path(Path(d))
            assert len(results) == 1
            assert results[0].skipped is True

    @pytest.mark.asyncio
    async def test_process_large_file_rejected(self, watcher: Any) -> None:
        from neural_memory.engine.file_watcher import WatchConfig

        watcher._config = WatchConfig(max_file_size_mb=0)  # 0 MB = reject all

        with tempfile.TemporaryDirectory() as d:
            md_file = Path(d) / "test.md"
            md_file.write_text("x" * 100)

            results = await watcher.process_path(Path(d))
            assert len(results) == 1
            assert results[0].success is False
            assert "too large" in results[0].error.lower()

    def test_queue_event_filters_extension(self, watcher: Any) -> None:
        watcher._queue_event(Path("/test/file.py"), "created")
        assert len(watcher._pending) == 0

        watcher._queue_event(Path("/test/file.md"), "created")
        assert len(watcher._pending) == 1

    def test_queue_event_filters_ignored(self, watcher: Any) -> None:
        watcher._queue_event(Path("/test/node_modules/readme.md"), "created")
        assert len(watcher._pending) == 0


# ── WatchHandler tests ───────────────────────────────


class TestWatchHandler:
    @pytest.fixture
    def handler(self) -> Any:
        from neural_memory.mcp.watch_handler import WatchHandler

        class TestHandler(WatchHandler):
            def __init__(self) -> None:
                self._storage = AsyncMock()
                self.config = MagicMock()
                self._file_watcher = None

            async def get_storage(self) -> Any:
                return self._storage

        h = TestHandler()
        h._storage.get_brain = AsyncMock(return_value=MagicMock(id="b1"))
        return h

    @pytest.mark.asyncio
    async def test_missing_action(self, handler: Any) -> None:
        result = await handler._watch({"action": "bogus"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_status_action(self, handler: Any) -> None:
        mock_watcher = MagicMock()
        mock_watcher.is_running = False
        mock_watcher._state = AsyncMock()
        mock_watcher._state.get_stats = AsyncMock(
            return_value={"total_files": 0, "total_neurons": 0, "by_status": {}}
        )
        mock_watcher.get_recent_results = MagicMock(return_value=[])
        handler._file_watcher = mock_watcher

        result = await handler._watch({"action": "status"})
        assert result["action"] == "status"
        assert result["running"] is False

    @pytest.mark.asyncio
    async def test_stop_not_running(self, handler: Any) -> None:
        result = await handler._watch({"action": "stop"})
        assert result["status"] == "not_running"

    @pytest.mark.asyncio
    async def test_scan_missing_directory(self, handler: Any) -> None:
        mock_watcher = MagicMock()
        mock_watcher.validate_path = MagicMock(return_value=None)
        mock_watcher.process_path = AsyncMock(return_value=[])
        mock_watcher._state = AsyncMock()
        mock_watcher._state.initialize = AsyncMock()
        handler._file_watcher = mock_watcher

        result = await handler._watch({"action": "scan"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_start_missing_directories(self, handler: Any) -> None:
        result = await handler._watch({"action": "start"})
        assert "error" in result
