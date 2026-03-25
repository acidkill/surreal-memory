"""Tests for B8: Adaptive Synapse Decay — reinforcement-modulated half-life."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.utils.timeutils import ensure_naive_utc, utcnow


def _make_synapse(
    weight: float = 0.8,
    reinforced_count: int = 0,
    hours_ago: float = 0,
) -> Synapse:
    """Create a synapse for testing with controlled age."""
    now = utcnow()
    created = now - timedelta(hours=hours_ago)
    s = Synapse.create(
        source_id="src",
        target_id="tgt",
        type=SynapseType.RELATED_TO,
        weight=weight,
    )
    return replace(s, reinforced_count=reinforced_count, last_activated=created, created_at=created)


class TestAdaptiveDecay:
    """Test reinforcement-modulated time_decay()."""

    def test_unreinforced_same_as_before(self) -> None:
        """Unreinforced synapse (count=0) decays at original rate."""
        s = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=1440)  # 60 days
        decayed = s.time_decay()
        # At center of sigmoid (60 days), factor ~0.5
        assert 0.4 < decayed.weight < 0.6

    def test_reinforced_decays_slower(self) -> None:
        """Reinforced synapse decays slower than unreinforced at same age."""
        s_unreinforced = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=1440)
        s_reinforced = _make_synapse(weight=1.0, reinforced_count=5, hours_ago=1440)

        d_unreinforced = s_unreinforced.time_decay()
        d_reinforced = s_reinforced.time_decay()

        # Reinforced should retain more weight
        assert d_reinforced.weight > d_unreinforced.weight

    def test_heavily_reinforced_nearly_permanent(self) -> None:
        """Synapse reinforced 10x retains most weight even at 60 days."""
        s = _make_synapse(weight=1.0, reinforced_count=10, hours_ago=1440)
        decayed = s.time_decay()
        # With reinforcement_factor = 1 + 10*0.5 = 6.0
        # effective_half_life = 1440 * 6 = 8640h (~360 days), spread = 4320
        # At 1440h: sigmoid = 1/(1+exp((1440-8640)/4320)) ~ 0.84
        assert decayed.weight > 0.80

    def test_floor_unreinforced(self) -> None:
        """Unreinforced synapse has floor at 0.3 (original behavior)."""
        s = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=50000)  # very old
        decayed = s.time_decay()
        assert decayed.weight >= 0.3 * 0.99  # allow tiny float tolerance

    def test_floor_reinforced_higher(self) -> None:
        """Reinforced synapse has higher floor than unreinforced."""
        s_unreinforced = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=50000)
        s_reinforced = _make_synapse(weight=1.0, reinforced_count=6, hours_ago=50000)

        d_unreinforced = s_unreinforced.time_decay()
        d_reinforced = s_reinforced.time_decay()

        # Reinforced floor: 0.3 + min(0.5, 6*0.05) = 0.3 + 0.3 = 0.6
        assert d_reinforced.weight > d_unreinforced.weight

    def test_floor_capped_at_0_8(self) -> None:
        """Floor caps at 0.8 even with extreme reinforcement."""
        s = _make_synapse(weight=1.0, reinforced_count=100, hours_ago=50000)
        decayed = s.time_decay()
        # Floor = 0.3 + min(0.5, 100*0.05) = 0.3 + 0.5 = 0.8
        assert decayed.weight >= 0.8 * 0.99
        # But sigmoid could be above floor anyway
        assert decayed.weight <= 1.0

    def test_recent_synapse_barely_decays(self) -> None:
        """Recent synapses barely decay regardless of reinforcement."""
        s0 = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=24)  # 1 day
        s5 = _make_synapse(weight=1.0, reinforced_count=5, hours_ago=24)

        d0 = s0.time_decay()
        d5 = s5.time_decay()

        # Both should retain most weight at 1 day
        # Sigmoid at 24h with half-life 1440, spread 720: ~0.88
        assert d0.weight > 0.85
        assert d5.weight > 0.85

    def test_reinforcement_factor_formula(self) -> None:
        """Verify reinforcement_factor = 1 + count * 0.5."""
        # reinforced 2x: factor = 1 + 2*0.5 = 2.0, half-life = 2880h (120 days)
        s = _make_synapse(weight=1.0, reinforced_count=2, hours_ago=2880)
        decayed = s.time_decay()
        # At effective half-life (2880h), factor should be ~0.5
        assert 0.4 < decayed.weight < 0.6

    def test_floor_formula(self) -> None:
        """Verify floor = 0.3 + min(0.5, count * 0.05)."""
        # reinforced 4x: floor = 0.3 + min(0.5, 4*0.05) = 0.3 + 0.2 = 0.5
        s = _make_synapse(weight=1.0, reinforced_count=4, hours_ago=100000)
        decayed = s.time_decay()
        assert decayed.weight >= 0.5 * 0.99

    def test_weight_preserved_not_amplified(self) -> None:
        """Decay never increases weight above original."""
        s = _make_synapse(weight=0.6, reinforced_count=10, hours_ago=1)
        decayed = s.time_decay()
        assert decayed.weight <= 0.6

    def test_zero_weight_stays_zero(self) -> None:
        """Zero weight synapse stays at zero after decay."""
        s = _make_synapse(weight=0.0, reinforced_count=5, hours_ago=500)
        decayed = s.time_decay()
        assert decayed.weight == 0.0

    def test_timezone_aware_reference_time_no_crash(self) -> None:
        """Issue #113: aware reference_time must not crash with naive stored datetimes."""
        s = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=24)
        aware_ref = datetime.now(UTC)
        decayed = s.time_decay(reference_time=aware_ref)
        assert decayed.weight > 0.0

    def test_timezone_aware_non_utc_reference_time(self) -> None:
        """Aware reference_time in non-UTC timezone should be converted to naive UTC."""
        s = _make_synapse(weight=1.0, reinforced_count=0, hours_ago=24)
        # UTC+7
        tz7 = timezone(timedelta(hours=7))
        aware_ref = datetime.now(tz7)
        decayed = s.time_decay(reference_time=aware_ref)
        assert decayed.weight > 0.0


class TestEnsureNaiveUtc:
    """Tests for ensure_naive_utc helper."""

    def test_naive_passthrough(self) -> None:
        dt = utcnow()
        assert ensure_naive_utc(dt) is dt

    def test_aware_utc_stripped(self) -> None:
        dt = datetime.now(UTC)
        result = ensure_naive_utc(dt)
        assert result.tzinfo is None
        assert abs((dt.replace(tzinfo=None) - result).total_seconds()) < 1

    def test_aware_non_utc_converted(self) -> None:
        tz7 = timezone(timedelta(hours=7))
        dt = datetime(2026, 3, 25, 14, 0, 0, tzinfo=tz7)  # 14:00 UTC+7 = 07:00 UTC
        result = ensure_naive_utc(dt)
        assert result.tzinfo is None
        assert result.hour == 7
