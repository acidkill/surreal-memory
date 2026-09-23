"""Recency anchor for fibers that were never recalled (`last_conducted is None`).

Through the REAL pipeline, not a re-implementation of the formula. The sibling file
``test_fiber_scoring.py`` scores fibers with a local copy of ``_fiber_score``; a copy
cannot fail when the engine changes, so it cannot protect this behaviour. These tests
drive ``ReflexPipeline.query`` against ``InMemoryStorage`` (a real storage backend) and
assert the ORDER of ``fibers_matched`` — the same list ``smem recall --json`` returns.

The defect: ``recency`` fell back to a flat ``0.5`` whenever ``last_conducted`` was
unset, so a memory written a minute ago started *below* one that was recalled a day
ago (recency ≈ 0.85 at the 168 h half-life). New knowledge therefore lost to older
knowledge that happened to be popular — the ranking rewarded rehearsal, never age.
With ``recency_from_created`` the anchor falls back to ``created_at``, so a fresh
memory is scored as fresh instead of as half-forgotten.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

_QUERY = "widget alpha protocol handshake"


@pytest.fixture
async def storage() -> AsyncIterator[InMemoryStorage]:
    s = InMemoryStorage()
    brain = Brain.create(name="recency_from_created_test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
    yield s
    await s.close()


async def _add_anchor_neuron(storage: InMemoryStorage, content: str) -> str:
    """One neuron shared by every fiber in a test.

    Both competing fibers reference the SAME neuron on purpose: ``activation_signal``
    is computed from the activated neurons of a fiber (max/mean level and coverage), so
    sharing the neuron makes that factor identical for both and leaves ``recency`` as
    the only difference. Two separate neurons with identical text do NOT give identical
    activation — spreading activation depends on graph position — and the resulting
    test would measure noise instead of the fix (measured: it did).
    """
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
    await storage.add_neuron(neuron)
    return neuron.id


async def _add_fiber(
    storage: InMemoryStorage,
    neuron_id: str,
    *,
    created_ago_hours: float | None,
    conducted_ago_hours: float | None,
    salience: float = 0.6,
) -> Fiber:
    """Add one fiber over the shared neuron, with explicit timestamps.

    ``None`` means "no timestamp at all" — the legacy shape this fix is about.
    """
    now = utcnow()
    fiber = Fiber.create(
        neuron_ids={neuron_id},
        synapse_ids=set(),
        anchor_neuron_id=neuron_id,
        summary=_QUERY,
    )
    fiber = replace(
        fiber,
        salience=salience,
        conductivity=1.0,
        created_at=(
            None if created_ago_hours is None else now - timedelta(hours=created_ago_hours)
        ),
        last_conducted=(
            None if conducted_ago_hours is None else now - timedelta(hours=conducted_ago_hours)
        ),
    )
    await storage.add_fiber(fiber)
    return fiber


def _order(fibers_matched: list[str], *fiber_ids: str) -> list[int]:
    """Positions of the given fibers in the ranked list; -1 when absent."""
    return [fibers_matched.index(f) if f in fibers_matched else -1 for f in fiber_ids]


class TestRecencyAnchorForNeverConductedFibers:
    async def test_fresh_never_recalled_fiber_outranks_older_recently_recalled_one(
        self, storage: InMemoryStorage
    ) -> None:
        """The regression this program exists for.

        ``fresh`` was written minutes ago and never recalled; ``popular`` is 200 days
        old but was recalled a day ago. Everything else about them is identical.
        Before the fix ``fresh`` scored 0.5 against ``popular``'s ≈0.85 and lost.
        """
        anchor = await _add_anchor_neuron(storage, _QUERY)
        fresh = await _add_fiber(storage, anchor, created_ago_hours=0.1, conducted_ago_hours=None)
        popular = await _add_fiber(
            storage, anchor, created_ago_hours=24 * 200, conducted_ago_hours=24
        )

        pipeline = ReflexPipeline(storage, BrainConfig(recency_from_created=True))
        result = await pipeline.query(_QUERY)

        pos_fresh, pos_popular = _order(result.fibers_matched, fresh.id, popular.id)
        assert pos_fresh >= 0 and pos_popular >= 0, "both fibers must be retrieved at all"
        assert pos_fresh < pos_popular, (
            "a memory written minutes ago must not rank below a 200-day-old one "
            "whose only advantage is having been recalled yesterday"
        )

    async def test_flag_off_keeps_the_legacy_flat_fallback(self, storage: InMemoryStorage) -> None:
        """With the flag off the old behaviour must be reproduced exactly.

        This is the distinguishability control: if this test and the one above both
        pass, the flag really is what changes the outcome — not some unrelated drift.
        """
        anchor = await _add_anchor_neuron(storage, _QUERY)
        fresh = await _add_fiber(storage, anchor, created_ago_hours=0.1, conducted_ago_hours=None)
        popular = await _add_fiber(
            storage, anchor, created_ago_hours=24 * 200, conducted_ago_hours=24
        )

        pipeline = ReflexPipeline(storage, BrainConfig(recency_from_created=False))
        result = await pipeline.query(_QUERY)

        pos_fresh, pos_popular = _order(result.fibers_matched, fresh.id, popular.id)
        assert pos_fresh >= 0 and pos_popular >= 0
        assert pos_popular < pos_fresh, "flag off must keep the recalled-yesterday fiber on top"

    async def test_newer_creation_beats_older_creation_when_neither_was_recalled(
        self, storage: InMemoryStorage
    ) -> None:
        """Among never-recalled fibers the anchor must actually order them by age."""
        anchor = await _add_anchor_neuron(storage, _QUERY)
        newer = await _add_fiber(storage, anchor, created_ago_hours=1, conducted_ago_hours=None)
        older = await _add_fiber(
            storage, anchor, created_ago_hours=24 * 365, conducted_ago_hours=None
        )

        pipeline = ReflexPipeline(storage, BrainConfig(recency_from_created=True))
        result = await pipeline.query(_QUERY)

        pos_newer, pos_older = _order(result.fibers_matched, newer.id, older.id)
        assert pos_newer >= 0 and pos_older >= 0
        assert pos_newer < pos_older

    async def test_fiber_without_any_timestamp_still_scores(self, storage: InMemoryStorage) -> None:
        """Backward compatibility: a legacy row with neither timestamp must not crash.

        It keeps the historical flat 0.5 anchor, so it still competes — it simply has
        no age signal to offer.
        """
        anchor = await _add_anchor_neuron(storage, _QUERY)
        legacy = await _add_fiber(storage, anchor, created_ago_hours=None, conducted_ago_hours=None)

        pipeline = ReflexPipeline(storage, BrainConfig(recency_from_created=True))
        result = await pipeline.query(_QUERY)

        assert legacy.id in result.fibers_matched


class TestRecencyFromCreatedConfigSurface:
    def test_default_is_on_in_this_fork(self) -> None:
        """Default carries the fix; an installation can still opt out in config.toml."""
        assert BrainConfig().recency_from_created is True

    def test_with_updates_toggles_it(self) -> None:
        cfg = BrainConfig()
        assert cfg.with_updates(recency_from_created=False).recency_from_created is False
        assert cfg.recency_from_created is True  # original untouched

    def test_reaches_a_stored_brain_through_config_toml_extras(self) -> None:
        """The knob must be migratable onto an already-stored brain.

        ``BrainSettings`` carries a handful of *explicit* keys (``freshness_weight``
        among them) that ``runtime_overrides()`` deliberately does NOT migrate, so a
        stored brain keeps whatever it was created with (issue #168). Any new knob has
        to travel through ``extras`` instead — this test pins that it does, because a
        flag that cannot reach the production brain is a flag that does nothing.
        """
        from surreal_memory.unified_config import BrainSettings

        settings = BrainSettings.from_dict({"recency_from_created": False})
        assert settings.extras["recency_from_created"] is False
        assert settings.runtime_overrides()["recency_from_created"] is False
