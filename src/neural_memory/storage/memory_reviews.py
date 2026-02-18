"""In-memory review schedule storage mixin for testing."""

from __future__ import annotations

from neural_memory.core.review_schedule import ReviewSchedule
from neural_memory.utils.timeutils import utcnow


class InMemoryReviewsMixin:
    """Mixin providing review schedule CRUD for InMemoryStorage."""

    # Declared in InMemoryStorage.__init__
    _review_schedules: dict[str, dict[str, ReviewSchedule]]

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def add_review_schedule(self, schedule: ReviewSchedule) -> str:
        """Insert or update a review schedule (upsert)."""
        brain_id = self._get_brain_id()
        if brain_id not in self._review_schedules:
            self._review_schedules[brain_id] = {}
        self._review_schedules[brain_id][schedule.fiber_id] = schedule
        return schedule.fiber_id

    async def get_review_schedule(self, fiber_id: str) -> ReviewSchedule | None:
        """Get a review schedule by fiber ID."""
        brain_id = self._get_brain_id()
        return self._review_schedules.get(brain_id, {}).get(fiber_id)

    async def get_due_reviews(self, limit: int = 20) -> list[ReviewSchedule]:
        """Get review schedules that are due (next_review <= now)."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 100)
        now = utcnow()

        due: list[ReviewSchedule] = []
        for schedule in self._review_schedules.get(brain_id, {}).values():
            if schedule.next_review is not None and schedule.next_review <= now:
                due.append(schedule)

        due.sort(key=lambda s: s.next_review or now)
        return due[:safe_limit]

    async def delete_review_schedule(self, fiber_id: str) -> bool:
        """Delete a review schedule. Returns True if deleted."""
        brain_id = self._get_brain_id()
        schedules = self._review_schedules.get(brain_id, {})
        if fiber_id in schedules:
            del schedules[fiber_id]
            return True
        return False

    async def get_review_stats(self) -> dict[str, int]:
        """Get review statistics for the current brain."""
        brain_id = self._get_brain_id()
        schedules = list(self._review_schedules.get(brain_id, {}).values())
        now = utcnow()

        stats = {"total": 0, "due": 0, "box_1": 0, "box_2": 0, "box_3": 0, "box_4": 0, "box_5": 0}
        for s in schedules:
            stats["total"] += 1
            if s.next_review is not None and s.next_review <= now:
                stats["due"] += 1
            box_key = f"box_{s.box}"
            if box_key in stats:
                stats[box_key] += 1

        return stats
