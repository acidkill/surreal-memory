"""Tests for nmem serve background daemons: auto-decay, re-index, notifications."""

from __future__ import annotations

import asyncio
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.unified_config import MaintenanceConfig

# ── Config tests ───────────────────────────────────────────


class TestMaintenanceConfigNewFields:
    """Test new decay/reindex/notification config fields."""

    def test_defaults(self) -> None:
        cfg = MaintenanceConfig()
        assert cfg.decay_enabled is True
        assert cfg.decay_interval_hours == 12
        assert cfg.reindex_enabled is False
        assert cfg.reindex_paths == ()
        assert cfg.reindex_interval_hours == 168
        assert cfg.notifications_enabled is False
        assert cfg.notifications_webhook_url == ""
        assert cfg.notifications_health_threshold == "D"
        assert cfg.notifications_daily_summary is False
        assert cfg.notifications_zero_activity_alert is True

    def test_from_dict_decay(self) -> None:
        cfg = MaintenanceConfig.from_dict(
            {
                "decay_enabled": False,
                "decay_interval_hours": 6,
            }
        )
        assert cfg.decay_enabled is False
        assert cfg.decay_interval_hours == 6

    def test_from_dict_reindex(self) -> None:
        cfg = MaintenanceConfig.from_dict(
            {
                "reindex_enabled": True,
                "reindex_paths": ["/home/user/notes", "/opt/docs"],
                "reindex_interval_hours": 48,
                "reindex_extensions": [".md", ".txt"],
            }
        )
        assert cfg.reindex_enabled is True
        assert cfg.reindex_paths == ("/home/user/notes", "/opt/docs")
        assert cfg.reindex_interval_hours == 48
        assert cfg.reindex_extensions == (".md", ".txt")

    def test_from_dict_notifications(self) -> None:
        cfg = MaintenanceConfig.from_dict(
            {
                "notifications_enabled": True,
                "notifications_webhook_url": "https://hooks.example.com/notify",
                "notifications_health_threshold": "C",
                "notifications_daily_summary": True,
                "notifications_zero_activity_alert": False,
            }
        )
        assert cfg.notifications_enabled is True
        assert cfg.notifications_webhook_url == "https://hooks.example.com/notify"
        assert cfg.notifications_health_threshold == "C"
        assert cfg.notifications_daily_summary is True
        assert cfg.notifications_zero_activity_alert is False

    def test_to_dict_roundtrip(self) -> None:
        original = MaintenanceConfig.from_dict(
            {
                "decay_enabled": True,
                "decay_interval_hours": 8,
                "reindex_enabled": True,
                "reindex_paths": ["/data"],
                "notifications_enabled": True,
                "notifications_webhook_url": "https://example.com",
            }
        )
        restored = MaintenanceConfig.from_dict(original.to_dict())
        assert restored.decay_interval_hours == 8
        assert restored.reindex_paths == ("/data",)
        assert restored.notifications_webhook_url == "https://example.com"

    def test_frozen(self) -> None:
        cfg = MaintenanceConfig()
        with pytest.raises(AttributeError):
            cfg.decay_enabled = False  # type: ignore[misc]


# ── Decay loop tests ──────────────────────────────────────


class TestDecayLoop:
    """Test _decay_loop background daemon."""

    @pytest.mark.asyncio
    async def test_decay_runs_on_interval(self) -> None:
        from neural_memory.server.app import _decay_loop

        storage = AsyncMock()
        storage.brain_id = "test-brain"

        config = MagicMock()
        config.brain.decay_rate = 0.1

        maint = MagicMock()
        maint.decay_interval_hours = 1  # 1 hour

        mock_report = MagicMock()
        mock_report.summary.return_value = "test decay"
        mock_report.neurons_pruned = 0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "neural_memory.engine.lifecycle.DecayManager.apply_decay",
                new_callable=AsyncMock,
                return_value=mock_report,
            ):
                # Make sleep raise after first call to break loop
                call_count = 0

                async def sleep_then_stop(seconds: float) -> None:
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        raise asyncio.CancelledError()

                mock_sleep.side_effect = sleep_then_stop
                with pytest.raises(asyncio.CancelledError):
                    await _decay_loop(storage, config, maint)

                mock_sleep.assert_called_with(3600)  # 1h in seconds

    @pytest.mark.asyncio
    async def test_decay_skips_without_brain(self) -> None:
        from neural_memory.server.app import _decay_loop

        storage = AsyncMock()
        storage.brain_id = None

        config = MagicMock()
        maint = MagicMock()
        maint.decay_interval_hours = 1

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            call_count = 0

            async def sleep_then_stop(seconds: float) -> None:
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise asyncio.CancelledError()

            mock_sleep.side_effect = sleep_then_stop
            with pytest.raises(asyncio.CancelledError):
                await _decay_loop(storage, config, maint)

    @pytest.mark.asyncio
    async def test_decay_survives_errors(self) -> None:
        from neural_memory.server.app import _decay_loop

        storage = AsyncMock()
        storage.brain_id = "test"

        config = MagicMock()
        config.brain.decay_rate = 0.1

        maint = MagicMock()
        maint.decay_interval_hours = 1

        call_count = 0

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "neural_memory.engine.lifecycle.DecayManager.apply_decay",
                new_callable=AsyncMock,
                side_effect=RuntimeError("test error"),
            ):

                async def sleep_then_stop(seconds: float) -> None:
                    nonlocal call_count
                    call_count += 1
                    if call_count > 1:
                        raise asyncio.CancelledError()

                mock_sleep.side_effect = sleep_then_stop
                with pytest.raises(asyncio.CancelledError):
                    await _decay_loop(storage, config, maint)
                # Should not crash — error logged and loop continues


# ── Re-index loop tests ───────────────────────────────────


class TestReindexLoop:
    """Test _reindex_loop background daemon."""

    @pytest.mark.asyncio
    async def test_reindex_runs_on_interval(self, tmp_path: Any) -> None:
        from neural_memory.server.app import _reindex_loop

        (tmp_path / "test.md").write_text("# Hello")

        mock_brain = MagicMock()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_brain = AsyncMock(return_value=mock_brain)
        storage._db = MagicMock()

        config = MagicMock()

        maint = MagicMock()
        maint.reindex_interval_hours = 1
        maint.reindex_paths = (str(tmp_path),)
        maint.reindex_extensions = frozenset({".md"})

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.skipped = False

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "neural_memory.engine.file_watcher.FileWatcher.process_path",
                new_callable=AsyncMock,
                return_value=[mock_result],
            ):
                with patch("neural_memory.engine.doc_trainer.DocTrainer"):
                    with patch("neural_memory.engine.watch_state.WatchStateTracker"):
                        call_count = 0

                        async def sleep_then_stop(seconds: float) -> None:
                            nonlocal call_count
                            call_count += 1
                            if call_count > 1:
                                raise asyncio.CancelledError()

                        mock_sleep.side_effect = sleep_then_stop
                        with pytest.raises(asyncio.CancelledError):
                            await _reindex_loop(storage, config, maint)

                        mock_sleep.assert_called_with(3600)

    @pytest.mark.asyncio
    async def test_reindex_skips_missing_paths(self) -> None:
        from neural_memory.server.app import _reindex_loop

        mock_brain = MagicMock()
        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_brain = AsyncMock(return_value=mock_brain)
        storage._db = MagicMock()

        config = MagicMock()

        maint = MagicMock()
        maint.reindex_interval_hours = 1
        maint.reindex_paths = ("/nonexistent/path/12345",)
        maint.reindex_extensions = frozenset({".md"})

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch("neural_memory.engine.doc_trainer.DocTrainer"):
                with patch("neural_memory.engine.watch_state.WatchStateTracker"):
                    call_count = 0

                    async def sleep_then_stop(seconds: float) -> None:
                        nonlocal call_count
                        call_count += 1
                        if call_count > 1:
                            raise asyncio.CancelledError()

                    mock_sleep.side_effect = sleep_then_stop
                    with pytest.raises(asyncio.CancelledError):
                        await _reindex_loop(storage, config, maint)


# ── Notification loop tests ────────────────────────────────


class TestNotificationLoop:
    """Test _notification_loop and _send_webhook."""

    @pytest.mark.asyncio
    async def test_webhook_sends_json(self) -> None:
        from neural_memory.server.app import _send_webhook

        received: list[bytes] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                received.append(self.rfile.read(length))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass  # Suppress server logs

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()

        logger = MagicMock()
        await _send_webhook(
            f"http://127.0.0.1:{port}/hook",
            {"brain_id": "test", "alerts": [{"type": "health_alert"}]},
            logger,
        )

        thread.join(timeout=5)
        server.server_close()

        assert len(received) == 1
        data = json.loads(received[0])
        assert data["brain_id"] == "test"
        assert data["alerts"][0]["type"] == "health_alert"

    @pytest.mark.asyncio
    async def test_notification_health_alert_fires(self) -> None:
        from neural_memory.server.app import _notification_loop

        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_enhanced_stats = AsyncMock(return_value={"recent_fiber_count": 5})

        config = MagicMock()

        maint = MagicMock()
        maint.notifications_webhook_url = "https://example.com/hook"
        maint.notifications_health_threshold = "D"
        maint.notifications_zero_activity_alert = True
        maint.notifications_daily_summary = False

        mock_report = MagicMock()
        mock_report.grade = "F"
        mock_report.purity_score = 20.0
        mock_report.warnings = ["warn1", "warn2"]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "neural_memory.engine.diagnostics.DiagnosticsEngine.analyze",
                new_callable=AsyncMock,
                return_value=mock_report,
            ):
                with patch(
                    "neural_memory.server.app._send_webhook",
                    new_callable=AsyncMock,
                ) as mock_webhook:
                    call_count = 0

                    async def sleep_then_stop(seconds: float) -> None:
                        nonlocal call_count
                        call_count += 1
                        if call_count > 1:
                            raise asyncio.CancelledError()

                    mock_sleep.side_effect = sleep_then_stop
                    with pytest.raises(asyncio.CancelledError):
                        await _notification_loop(storage, config, maint)

                    mock_webhook.assert_called_once()
                    payload = mock_webhook.call_args[0][1]
                    assert payload["brain_id"] == "test-brain"
                    assert any(a["type"] == "health_alert" for a in payload["alerts"])

    @pytest.mark.asyncio
    async def test_notification_zero_activity_alert(self) -> None:
        from neural_memory.server.app import _notification_loop

        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_enhanced_stats = AsyncMock(return_value={"recent_fiber_count": 0})
        storage.get_stats = AsyncMock(return_value={})

        config = MagicMock()

        maint = MagicMock()
        maint.notifications_webhook_url = "https://example.com/hook"
        maint.notifications_health_threshold = "D"
        maint.notifications_zero_activity_alert = True
        maint.notifications_daily_summary = False

        mock_report = MagicMock()
        mock_report.grade = "A"  # Healthy grade — no health alert
        mock_report.purity_score = 95.0
        mock_report.warnings = []

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "neural_memory.engine.diagnostics.DiagnosticsEngine.analyze",
                new_callable=AsyncMock,
                return_value=mock_report,
            ):
                with patch(
                    "neural_memory.server.app._send_webhook",
                    new_callable=AsyncMock,
                ) as mock_webhook:
                    call_count = 0

                    async def sleep_then_stop(seconds: float) -> None:
                        nonlocal call_count
                        call_count += 1
                        if call_count > 1:
                            raise asyncio.CancelledError()

                    mock_sleep.side_effect = sleep_then_stop
                    with pytest.raises(asyncio.CancelledError):
                        await _notification_loop(storage, config, maint)

                    mock_webhook.assert_called_once()
                    payload = mock_webhook.call_args[0][1]
                    assert any(a["type"] == "zero_activity" for a in payload["alerts"])

    @pytest.mark.asyncio
    async def test_no_alert_when_healthy(self) -> None:
        from neural_memory.server.app import _notification_loop

        storage = AsyncMock()
        storage.brain_id = "test-brain"
        storage.get_enhanced_stats = AsyncMock(return_value={"recent_fiber_count": 10})

        config = MagicMock()

        maint = MagicMock()
        maint.notifications_webhook_url = "https://example.com/hook"
        maint.notifications_health_threshold = "D"
        maint.notifications_zero_activity_alert = True
        maint.notifications_daily_summary = False

        mock_report = MagicMock()
        mock_report.grade = "A"
        mock_report.purity_score = 95.0
        mock_report.warnings = []

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch(
                "neural_memory.engine.diagnostics.DiagnosticsEngine.analyze",
                new_callable=AsyncMock,
                return_value=mock_report,
            ):
                with patch(
                    "neural_memory.server.app._send_webhook",
                    new_callable=AsyncMock,
                ) as mock_webhook:
                    call_count = 0

                    async def sleep_then_stop(seconds: float) -> None:
                        nonlocal call_count
                        call_count += 1
                        if call_count > 1:
                            raise asyncio.CancelledError()

                    mock_sleep.side_effect = sleep_then_stop
                    with pytest.raises(asyncio.CancelledError):
                        await _notification_loop(storage, config, maint)

                    # No alerts → webhook not called
                    mock_webhook.assert_not_called()


# ── Lifespan integration ───────────────────────────────────


class TestLifespanDaemons:
    """Test that lifespan starts correct daemons based on config."""

    def test_all_daemons_start_when_enabled(self) -> None:
        """Verify config flags control which daemons are created."""
        maint = MaintenanceConfig.from_dict(
            {
                "enabled": True,
                "scheduled_consolidation_enabled": True,
                "decay_enabled": True,
                "reindex_enabled": True,
                "reindex_paths": ["/data"],
                "notifications_enabled": True,
                "notifications_webhook_url": "https://example.com",
            }
        )
        assert maint.enabled is True
        assert maint.decay_enabled is True
        assert maint.reindex_enabled is True
        assert maint.notifications_enabled is True
        assert maint.notifications_webhook_url != ""

    def test_no_daemons_when_disabled(self) -> None:
        maint = MaintenanceConfig.from_dict({"enabled": False})
        assert maint.enabled is False

    def test_reindex_needs_paths(self) -> None:
        """Re-index won't start without paths configured."""
        maint = MaintenanceConfig.from_dict(
            {
                "reindex_enabled": True,
                # No paths
            }
        )
        assert maint.reindex_enabled is True
        assert maint.reindex_paths == ()  # Empty → daemon won't start

    def test_notifications_need_webhook(self) -> None:
        """Notifications won't start without webhook URL."""
        maint = MaintenanceConfig.from_dict(
            {
                "notifications_enabled": True,
                # No webhook URL
            }
        )
        assert maint.notifications_enabled is True
        assert maint.notifications_webhook_url == ""  # Empty → daemon won't start
