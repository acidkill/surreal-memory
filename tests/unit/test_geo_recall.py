"""U8: geospatial recall (`near`) through the REAL pipeline.

Mirrors tests/unit/test_valid_at_pipeline.py. The `near` filter lives in exactly the
same place as `_fiber_valid_at` (ReflexPipeline._find_matching_fibers, right after the
valid_at filter), so — per the valid_at lesson (U3's mocked-pipeline tests hid the
`_fiber_valid_at` bug) — it MUST be exercised end-to-end against the real pipeline, not
a mock. A hard filter: fibers outside the radius (or without a location) are dropped;
`near=None` is a strict no-op (golden recall stays green).
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline, _fiber_near
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.sqlite_store import SQLiteStorage
from surreal_memory.utils.geo import GeoFilter, GeoPoint

_OSLO = {"lat": 59.9139, "lon": 10.7522, "label": "Oslo"}
_BERGEN = {"lat": 60.3913, "lon": 5.3221, "label": "Bergen"}
_OSLO_CENTER = GeoPoint(59.9139, 10.7522)


class TestFiberNearUnit:
    def _fiber(self, location: dict[str, float] | None) -> Fiber:
        return Fiber.create(
            neuron_ids={"n"},
            synapse_ids=set(),
            anchor_neuron_id="n",
            summary="x",
            metadata={"location": location} if location is not None else None,
        )

    def test_inside_radius(self) -> None:
        near = GeoFilter(GeoPoint(59.95, 10.80), 50_000)  # a few km from Oslo
        assert _fiber_near(self._fiber(_OSLO), near) is True

    def test_outside_radius(self) -> None:
        near = GeoFilter(_OSLO_CENTER, 50_000)  # 50 km around Oslo
        assert _fiber_near(self._fiber(_BERGEN), near) is False  # Bergen ~305 km away

    def test_no_location_is_excluded(self) -> None:
        near = GeoFilter(_OSLO_CENTER, 1_000)
        assert _fiber_near(self._fiber(None), near) is False

    def test_garbage_location_is_excluded(self) -> None:
        near = GeoFilter(_OSLO_CENTER, 20_015_087.0)  # whole globe — only garbage fails
        assert _fiber_near(self._fiber({"lat": 999.0, "lon": 0.0}), near) is False


@pytest.fixture
async def storage() -> AsyncIterator[SQLiteStorage]:
    with tempfile.TemporaryDirectory() as tmpdir:
        s = SQLiteStorage(Path(tmpdir) / "test.db")
        await s.initialize()
        brain = Brain.create(name="geo_test")
        await s.save_brain(brain)
        s.set_brain(brain.id)
        yield s
        await s.close()


async def _add_located_fiber(
    storage: SQLiteStorage, content: str, location: dict[str, float] | None
) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=content,
        metadata={"location": location} if location is not None else None,
    )
    await storage.add_fiber(fiber)
    return fiber


class TestNearThroughPipeline:
    async def test_near_keeps_only_fibers_in_radius(self, storage: SQLiteStorage) -> None:
        oslo = await _add_located_fiber(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        bergen = await _add_located_fiber(storage, "Cafe Mocca is in Bergen Norway", _BERGEN)

        pipeline = ReflexPipeline(storage, BrainConfig())
        result = await pipeline.query("Cafe Mocca Norway", near=GeoFilter(_OSLO_CENTER, 50_000))

        assert oslo.id in result.fibers_matched
        assert bergen.id not in result.fibers_matched  # ~305 km away, filtered out

    async def test_wider_radius_keeps_both(self, storage: SQLiteStorage) -> None:
        oslo = await _add_located_fiber(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        bergen = await _add_located_fiber(storage, "Cafe Mocca is in Bergen Norway", _BERGEN)

        pipeline = ReflexPipeline(storage, BrainConfig())
        result = await pipeline.query("Cafe Mocca Norway", near=GeoFilter(_OSLO_CENTER, 400_000))

        assert oslo.id in result.fibers_matched
        assert bergen.id in result.fibers_matched  # 400 km covers Bergen

    async def test_no_near_is_a_noop(self, storage: SQLiteStorage) -> None:
        # Golden invariant: near=None leaves recall unchanged (both returned).
        oslo = await _add_located_fiber(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        bergen = await _add_located_fiber(storage, "Cafe Mocca is in Bergen Norway", _BERGEN)

        pipeline = ReflexPipeline(storage, BrainConfig())
        result = await pipeline.query("Cafe Mocca Norway")

        assert oslo.id in result.fibers_matched
        assert bergen.id in result.fibers_matched

    async def test_fiber_without_location_excluded_when_near_set(
        self, storage: SQLiteStorage
    ) -> None:
        located = await _add_located_fiber(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        unlocated = await _add_located_fiber(storage, "Cafe Mocca is in Oslo Norway too", None)

        pipeline = ReflexPipeline(storage, BrainConfig())
        result = await pipeline.query("Cafe Mocca Norway", near=GeoFilter(_OSLO_CENTER, 50_000))

        assert located.id in result.fibers_matched
        assert unlocated.id not in result.fibers_matched  # no location → excluded


@pytest.fixture(params=["memory", "sqlite"])
async def browse_storage(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    """Same geo browse contract on both in-process backends (SurrealDB: *_live test)."""
    if request.param == "memory":
        s: object = InMemoryStorage()
        brain = Brain.create(name="geo_browse")
        await s.save_brain(brain)  # type: ignore[attr-defined]
        s.set_brain(brain.id)  # type: ignore[attr-defined]
        yield s
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            sq = SQLiteStorage(Path(tmpdir) / "test.db")
            await sq.initialize()
            brain = Brain.create(name="geo_browse")
            await sq.save_brain(brain)
            sq.set_brain(brain.id)
            yield sq
            await sq.close()


async def _browse_add(storage: object, content: str, location: dict[str, float] | None) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
    await storage.add_neuron(neuron)  # type: ignore[attr-defined]
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=content,
        metadata={"location": location} if location is not None else None,
    )
    await storage.add_fiber(fiber)  # type: ignore[attr-defined]
    return fiber


class TestFindFibersNearBrowse:
    """Browse pushdown `find_fibers(near=)` — identical hard-filter across backends."""

    async def test_filters_by_radius_and_excludes_locationless(
        self, browse_storage: object
    ) -> None:
        oslo = await _browse_add(browse_storage, "in oslo", _OSLO)
        bergen = await _browse_add(browse_storage, "in bergen", _BERGEN)
        nowhere = await _browse_add(browse_storage, "no location", None)

        found = await browse_storage.find_fibers(near=GeoFilter(_OSLO_CENTER, 50_000))  # type: ignore[attr-defined]
        ids = {f.id for f in found}
        assert oslo.id in ids
        assert bergen.id not in ids  # ~305 km away
        assert nowhere.id not in ids  # no location → excluded

    async def test_wider_radius_keeps_bergen(self, browse_storage: object) -> None:
        oslo = await _browse_add(browse_storage, "in oslo", _OSLO)
        bergen = await _browse_add(browse_storage, "in bergen", _BERGEN)

        found = await browse_storage.find_fibers(near=GeoFilter(_OSLO_CENTER, 400_000))  # type: ignore[attr-defined]
        ids = {f.id for f in found}
        assert oslo.id in ids and bergen.id in ids

    async def test_antimeridian_not_wrongly_excluded(self, browse_storage: object) -> None:
        # Across the antimeridian the exact haversine must still see ~22 km (a naive
        # lon-bbox would wrap and wrongly drop this fiber — which is why there is none).
        f_near = await _browse_add(browse_storage, "east of the line", {"lat": 0.0, "lon": 179.9})
        found = await browse_storage.find_fibers(near=GeoFilter(GeoPoint(0.0, -179.9), 30_000))  # type: ignore[attr-defined]
        assert f_near.id in {f.id for f in found}  # ~22 km across the antimeridian

    async def test_edge_of_radius_included(self, browse_storage: object) -> None:
        # A fiber ~9.9 km from center must survive a 10 km browse filter (no bbox
        # pre-filter can shave the edge; exact haversine is the only bound).
        import math

        edge = {"lat": math.degrees(9_900.0 / 6_371_008.8), "lon": 0.0}
        f_edge = await _browse_add(browse_storage, "near the edge", edge)
        found = await browse_storage.find_fibers(near=GeoFilter(GeoPoint(0.0, 0.0), 10_000))  # type: ignore[attr-defined]
        assert f_edge.id in {f.id for f in found}

    async def test_no_near_returns_all(self, browse_storage: object) -> None:
        located = await _browse_add(browse_storage, "in oslo", _OSLO)
        nowhere = await _browse_add(browse_storage, "no location", None)

        found = await browse_storage.find_fibers()  # type: ignore[attr-defined]
        ids = {f.id for f in found}
        assert located.id in ids and nowhere.id in ids
