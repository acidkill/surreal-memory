"""Unit tests for TypedMemory validity/supersession helpers (schema v9, U1).

Covers valid_from/valid_until/superseded_by fields, the with_validity helper,
is_superseded / is_valid_at, and that every copy method threads the new fields.
"""

from __future__ import annotations

from datetime import timedelta

from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.utils.timeutils import utcnow


def _make(**kwargs: object) -> TypedMemory:
    return TypedMemory.create(fiber_id="f1", memory_type=MemoryType.FACT, **kwargs)  # type: ignore[arg-type]


class TestValidityDefaults:
    def test_new_typed_memory_has_no_validity(self) -> None:
        tm = _make()
        assert tm.valid_from is None
        assert tm.valid_until is None
        assert tm.superseded_by is None
        assert tm.is_superseded is False


class TestWithValidity:
    def test_sets_fields_and_returns_new_object(self) -> None:
        tm = _make()
        now = utcnow()
        tm2 = tm.with_validity(valid_until=now, superseded_by="f2")
        # immutability: original untouched
        assert tm.valid_until is None
        assert tm.superseded_by is None
        # new object updated
        assert tm2.valid_until == now
        assert tm2.superseded_by == "f2"
        assert tm2.is_superseded is True
        # unrelated fields preserved
        assert tm2.fiber_id == tm.fiber_id
        assert tm2.memory_type == tm.memory_type
        assert tm2.created_at == tm.created_at
        assert tm2.trust_score == tm.trust_score

    def test_partial_update_leaves_other_validity_fields(self) -> None:
        now = utcnow()
        tm = _make().with_validity(valid_from=now)
        tm2 = tm.with_validity(superseded_by="f9")
        assert tm2.valid_from == now  # untouched by second call
        assert tm2.superseded_by == "f9"

    def test_can_clear_field_explicitly(self) -> None:
        tm = _make().with_validity(superseded_by="f2")
        tm2 = tm.with_validity(superseded_by=None)
        assert tm2.superseded_by is None
        assert tm2.is_superseded is False


class TestIsValidAt:
    def test_open_ended_after_created_at(self) -> None:
        tm = _make()  # valid_from None -> fallback created_at; valid_until None
        assert tm.is_valid_at(tm.created_at) is True
        assert tm.is_valid_at(tm.created_at + timedelta(days=1)) is True

    def test_before_created_at_is_invalid_when_no_valid_from(self) -> None:
        tm = _make()
        assert tm.is_valid_at(tm.created_at - timedelta(seconds=1)) is False

    def test_respects_explicit_valid_from(self) -> None:
        start = utcnow()
        tm = _make().with_validity(valid_from=start)
        assert tm.is_valid_at(start) is True  # boundary inclusive
        assert tm.is_valid_at(start - timedelta(seconds=1)) is False

    def test_valid_until_is_exclusive_upper_bound(self) -> None:
        start = utcnow()
        end = start + timedelta(days=2)
        tm = _make().with_validity(valid_from=start, valid_until=end)
        assert tm.is_valid_at(start) is True
        assert tm.is_valid_at(start + timedelta(days=1)) is True
        assert tm.is_valid_at(end) is False  # boundary exclusive
        assert tm.is_valid_at(end + timedelta(seconds=1)) is False


class TestCopyMethodsPreserveValidity:
    def test_with_priority_preserves_validity(self) -> None:
        now = utcnow()
        tm = _make().with_validity(valid_until=now, superseded_by="f2")
        tm2 = tm.with_priority(9)
        assert tm2.valid_until == now
        assert tm2.superseded_by == "f2"

    def test_with_tier_preserves_validity(self) -> None:
        now = utcnow()
        tm = _make().with_validity(valid_until=now, superseded_by="f2")
        tm2 = tm.with_tier("cold")
        assert tm2.valid_until == now
        assert tm2.superseded_by == "f2"

    def test_verify_preserves_validity(self) -> None:
        now = utcnow()
        tm = _make().with_validity(valid_until=now, superseded_by="f2")
        tm2 = tm.verify()
        assert tm2.valid_until == now
        assert tm2.superseded_by == "f2"

    def test_extend_expiry_preserves_validity(self) -> None:
        now = utcnow()
        tm = _make().with_validity(valid_until=now, superseded_by="f2")
        tm2 = tm.extend_expiry(30)
        assert tm2.valid_until == now
        assert tm2.superseded_by == "f2"
