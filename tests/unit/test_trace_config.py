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
