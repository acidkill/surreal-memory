"""Unit tests for TraceConfig (schema v9, U1)."""

from __future__ import annotations

from surreal_memory.unified_config import TraceConfig


class TestTraceConfig:
    def test_neutral_defaults(self) -> None:
        c = TraceConfig()
        assert c.enabled is False
        assert c.sample_rate == 1.0
        assert c.retention_days == 30
        assert c.max_traces == 5000

    def test_round_trip(self) -> None:
        c = TraceConfig(enabled=True, sample_rate=0.5, retention_days=7, max_traces=100)
        assert TraceConfig.from_dict(c.to_dict()) == c

    def test_from_dict_defaults_on_missing_keys(self) -> None:
        c = TraceConfig.from_dict({})
        assert c == TraceConfig()


class TestUnifiedConfigTraceWiring:
    """U4: UnifiedConfig exposes + persists the [trace] section."""

    def test_default_trace_is_off(self) -> None:
        from surreal_memory.unified_config import UnifiedConfig

        assert UnifiedConfig().trace.enabled is False

    def test_save_load_round_trip_preserves_trace(self, tmp_path: object) -> None:
        from pathlib import Path

        from surreal_memory.unified_config import UnifiedConfig

        data_dir = Path(str(tmp_path))
        cfg = UnifiedConfig(data_dir=data_dir)
        cfg.trace = TraceConfig(enabled=True, sample_rate=0.25, retention_days=14, max_traces=99)
        cfg.save()

        loaded = UnifiedConfig.load(config_path=data_dir / "config.toml")
        assert loaded.trace.enabled is True
        assert loaded.trace.sample_rate == 0.25
        assert loaded.trace.retention_days == 14
        assert loaded.trace.max_traces == 99
