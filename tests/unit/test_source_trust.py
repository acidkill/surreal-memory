"""Unit tests for Source.trust (schema v9, U1)."""

from __future__ import annotations

import pytest

from surreal_memory.core.source import Source


def _make(**kwargs: object) -> Source:
    return Source.create(brain_id="b1", name="doc.pdf", **kwargs)  # type: ignore[arg-type]


class TestSourceTrust:
    def test_default_trust_is_none(self) -> None:
        assert _make().trust is None

    def test_with_trust_sets_value_and_returns_new_object(self) -> None:
        s = _make()
        s2 = s.with_trust(0.5)
        assert s.trust is None  # original immutable
        assert s2.trust == 0.5
        assert s2.id == s.id

    def test_with_trust_updates_timestamp(self) -> None:
        s = _make()
        s2 = s.with_trust(0.9)
        assert s2.updated_at >= s.updated_at

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, -5.0])
    def test_out_of_range_trust_rejected(self, value: float) -> None:
        with pytest.raises(ValueError):
            _make().with_trust(value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_boundary_values_accepted(self, value: float) -> None:
        assert _make().with_trust(value).trust == value

    def test_create_accepts_trust(self) -> None:
        assert _make(trust=0.7).trust == 0.7

    def test_create_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            _make(trust=1.5)

    def test_with_trust_can_clear_to_none(self) -> None:
        s = _make(trust=0.7).with_trust(None)
        assert s.trust is None
