"""Spaced repetition review schedule — Leitner box system."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from neural_memory.utils.timeutils import utcnow

# Leitner box intervals (in days): box 1 = 1d, box 2 = 3d, box 3 = 7d, box 4 = 14d, box 5 = 30d
LEITNER_INTERVALS: dict[int, int] = {
    1: 1,
    2: 3,
    3: 7,
    4: 14,
    5: 30,
}

MAX_BOX = 5
MIN_BOX = 1


@dataclass(frozen=True)
class ReviewSchedule:
    """A spaced repetition schedule for a fiber.

    Attributes:
        fiber_id: The fiber being reviewed
        brain_id: Brain owning the fiber
        box: Current Leitner box (1-5)
        next_review: When this fiber is next due for review
        last_reviewed: When last reviewed (None if never)
        review_count: Total number of reviews
        streak: Consecutive successful reviews
        created_at: When this schedule was created
    """

    fiber_id: str
    brain_id: str
    box: int = MIN_BOX
    next_review: datetime | None = None
    last_reviewed: datetime | None = None
    review_count: int = 0
    streak: int = 0
    created_at: datetime | None = None

    @classmethod
    def create(cls, fiber_id: str, brain_id: str) -> ReviewSchedule:
        """Create a new schedule starting at box 1, due immediately."""
        now = utcnow()
        return cls(
            fiber_id=fiber_id,
            brain_id=brain_id,
            box=MIN_BOX,
            next_review=now,
            last_reviewed=None,
            review_count=0,
            streak=0,
            created_at=now,
        )

    def advance(self, success: bool) -> ReviewSchedule:
        """Return a new schedule after a review.

        Args:
            success: True if recall was successful, False otherwise

        Returns:
            New ReviewSchedule with updated box, streak, and next_review
        """
        now = utcnow()
        if success:
            new_box = min(self.box + 1, MAX_BOX)
            new_streak = self.streak + 1
        else:
            new_box = MIN_BOX
            new_streak = 0

        interval_days = LEITNER_INTERVALS[new_box]
        new_next_review = now + timedelta(days=interval_days)

        return replace(
            self,
            box=new_box,
            next_review=new_next_review,
            last_reviewed=now,
            review_count=self.review_count + 1,
            streak=new_streak,
        )
