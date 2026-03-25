"""Timezone-safe datetime utilities.

Provides ``utcnow()`` as a drop-in replacement for the deprecated
``datetime.utcnow()`` (scheduled for removal in Python 3.14+).

Returns **naive** UTC datetimes to maintain backward compatibility with
existing SQLite storage (isoformat without ``+00:00`` suffix).  A future
major version may switch to timezone-aware datetimes.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime.

    Equivalent to the deprecated ``datetime.utcnow()`` but without
    triggering a ``DeprecationWarning``.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def ensure_naive_utc(dt: datetime) -> datetime:
    """Convert a datetime to naive UTC.

    If the datetime is timezone-aware, convert to UTC and strip tzinfo.
    If already naive, return as-is (assumed to be UTC per project convention).
    """
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt
