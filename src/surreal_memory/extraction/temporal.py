"""Temporal extraction for English time expressions."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class TimeGranularity(StrEnum):
    """Granularity level of a time reference."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass(frozen=True)
class TimeHint:
    """
    A parsed time reference from text.

    Attributes:
        original: The original text that was matched
        absolute_start: Resolved start datetime
        absolute_end: Resolved end datetime
        granularity: How precise this time reference is
        is_fuzzy: Whether this is an approximate time
    """

    original: str
    absolute_start: datetime
    absolute_end: datetime
    granularity: TimeGranularity
    is_fuzzy: bool = True

    @property
    def midpoint(self) -> datetime:
        """Get the midpoint of this time range."""
        delta = (self.absolute_end - self.absolute_start) / 2
        return self.absolute_start + delta


# Type alias for time resolver functions
# Some resolvers take (ref) and some take (ref, match) — arity is checked at runtime
TimeResolver = Callable[..., tuple[datetime, datetime]]


def _start_of_day(dt: datetime) -> datetime:
    """Get start of day (00:00:00)."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    """Get end of day (23:59:59)."""
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


class TemporalExtractor:
    """
    English temporal expression extractor.
    """

    # English time patterns
    EN_PATTERNS: dict[str, TimeResolver] = {
        # Relative days
        r"today": lambda ref: (_start_of_day(ref), _end_of_day(ref)),
        r"yesterday": lambda ref: (
            _start_of_day(ref - timedelta(days=1)),
            _end_of_day(ref - timedelta(days=1)),
        ),
        r"day before yesterday": lambda ref: (
            _start_of_day(ref - timedelta(days=2)),
            _end_of_day(ref - timedelta(days=2)),
        ),
        r"tomorrow": lambda ref: (
            _start_of_day(ref + timedelta(days=1)),
            _end_of_day(ref + timedelta(days=1)),
        ),
        # Parts of day (today)
        r"this morning": lambda ref: (
            ref.replace(hour=6, minute=0, second=0, microsecond=0),
            ref.replace(hour=12, minute=0, second=0, microsecond=0),
        ),
        r"this afternoon": lambda ref: (
            ref.replace(hour=12, minute=0, second=0, microsecond=0),
            ref.replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        r"this evening": lambda ref: (
            ref.replace(hour=18, minute=0, second=0, microsecond=0),
            ref.replace(hour=22, minute=0, second=0, microsecond=0),
        ),
        r"tonight": lambda ref: (
            ref.replace(hour=20, minute=0, second=0, microsecond=0),
            (ref + timedelta(days=1)).replace(hour=4, minute=0, second=0, microsecond=0),
        ),
        # Parts of day (yesterday)
        r"yesterday morning": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0),
        ),
        r"yesterday afternoon": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
        ),
        r"yesterday evening|last night": lambda ref: (
            (ref - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0),
            (ref - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0),
        ),
        # Relative weeks
        r"this week": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday())),
            _end_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=6)),
        ),
        r"last week": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday() + 7)),
            _end_of_day(ref - timedelta(days=ref.weekday() + 1)),
        ),
        r"next week": lambda ref: (
            _start_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=7)),
            _end_of_day(ref - timedelta(days=ref.weekday()) + timedelta(days=13)),
        ),
        # Relative months
        r"this month": lambda ref: (
            ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            _end_of_day(
                (ref.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            ),
        ),
        r"last month": lambda ref: (
            (ref.replace(day=1) - timedelta(days=1)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ),
            _end_of_day(ref.replace(day=1) - timedelta(days=1)),
        ),
        # Recent time
        r"just now|just|recently": lambda ref: (
            ref - timedelta(hours=2),
            ref,
        ),
        r"earlier|earlier today": lambda ref: (
            _start_of_day(ref),
            ref - timedelta(hours=1),
        ),
        # Ago patterns
        r"(\d+)\s*minutes?\s*ago": lambda ref, m: (
            ref - timedelta(minutes=int(m.group(1)) + 5),
            ref - timedelta(minutes=max(0, int(m.group(1)) - 5)),
        ),
        r"(\d+)\s*hours?\s*ago": lambda ref, m: (
            ref - timedelta(hours=int(m.group(1)) + 0.5),
            ref - timedelta(hours=max(0, int(m.group(1)) - 0.5)),
        ),
        r"(\d+)\s*days?\s*ago": lambda ref, m: (
            _start_of_day(ref - timedelta(days=int(m.group(1)))),
            _end_of_day(ref - timedelta(days=int(m.group(1)))),
        ),
    }

    def __init__(self) -> None:
        """Initialize the extractor."""
        # Compile regex patterns and cache resolver arity
        self._en_compiled = [
            (re.compile(p, re.IGNORECASE), r, len(inspect.signature(r).parameters))
            for p, r in self.EN_PATTERNS.items()
        ]

    def extract(
        self,
        text: str,
        reference_time: datetime | None = None,
        language: str = "auto",
    ) -> list[TimeHint]:
        """
        Extract time references from text.

        Args:
            text: The text to extract from
            reference_time: Reference point for relative times (default: now)
            language: Accepted for caller compatibility; ignored (English-only).

        Returns:
            List of TimeHint objects for found time references
        """
        del language  # English-only; retained in signature for caller compatibility
        if reference_time is None:
            reference_time = utcnow()

        results: list[TimeHint] = []

        patterns = self._en_compiled

        # Try each pattern
        for pattern, resolver, arity in patterns:
            for match in pattern.finditer(text):
                try:
                    if arity == 2:
                        start, end = resolver(reference_time, match)
                    else:
                        start, end = resolver(reference_time)

                    results.append(
                        TimeHint(
                            original=match.group(0),
                            absolute_start=start,
                            absolute_end=end,
                            granularity=_infer_granularity(start, end),
                            is_fuzzy=True,
                        )
                    )
                except (ValueError, TypeError, OverflowError) as e:
                    logger.debug("Time pattern failed to resolve '%s': %s", match.group(0), e)
                    continue

        # Remove duplicates (same time range)
        seen: set[tuple[datetime, datetime]] = set()
        unique_results: list[TimeHint] = []
        for hint in results:
            key = (hint.absolute_start, hint.absolute_end)
            if key not in seen:
                seen.add(key)
                unique_results.append(hint)

        return unique_results


def _infer_granularity(start: datetime, end: datetime) -> TimeGranularity:
    """Infer granularity from time range."""
    delta = end - start
    seconds = delta.total_seconds()

    if seconds <= 3600:  # 1 hour
        return TimeGranularity.MINUTE
    elif seconds <= 86400:  # 1 day
        return TimeGranularity.HOUR
    elif seconds <= 604800:  # 1 week
        return TimeGranularity.DAY
    elif seconds <= 2678400:  # ~1 month
        return TimeGranularity.WEEK
    elif seconds <= 31536000:  # 1 year
        return TimeGranularity.MONTH
    else:
        return TimeGranularity.YEAR
