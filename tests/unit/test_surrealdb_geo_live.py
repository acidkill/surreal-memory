"""Live-DB integration for U8 geospatial recall on SurrealDB.

Pins the two SurrealDB-specific risks from the plan: (1) ``metadata.location`` is a
FLEXIBLE object, so the ``metadata.location != NONE`` server-side pre-filter must
traverse it without erroring, and (2) a ``location`` dict must round-trip through the
store so the exact haversine post-filter can read it back. Also drives the REAL
ReflexPipeline with ``near=`` end-to-end (the valid_at lesson: never mock the pipeline).
Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.utils.geo import GeoFilter, GeoPoint

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)

_OSLO = {"lat": 59.9139, "lon": 10.7522, "label": "Oslo"}
_BERGEN = {"lat": 60.3913, "lon": 5.3221, "label": "Bergen"}
_OSLO_CENTER = GeoPoint(59.9139, 10.7522)


def _norm(fiber_id: str) -> str:
    """Canonical dash form. SurrealDB's find_fibers/get_fiber/fibers_matched return the
    underscore-sanitized Fiber.id round-trip (the documented "Bug C" deferred id form),
    so id comparisons must be form-agnostic — same pattern as the supersession live test.
    """
    return fiber_id.replace("_", "-")


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name="u8-geo-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    try:
        await store.close()
    except Exception:
        pass


async def _add(storage, content, location):  # type: ignore[no-untyped-def]
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


class TestSurrealGeoBrowse:
    async def test_location_roundtrips(self, storage) -> None:  # type: ignore[no-untyped-def]
        fiber = await _add(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        loaded = await storage.get_fiber(fiber.id)
        assert loaded is not None
        assert loaded.metadata.get("location") == _OSLO

    async def test_find_fibers_near_filters_on_real_db(self, storage) -> None:  # type: ignore[no-untyped-def]
        oslo = await _add(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        bergen = await _add(storage, "Cafe Mocca is in Bergen Norway", _BERGEN)
        nowhere = await _add(storage, "Cafe Mocca has no location", None)

        found = await storage.find_fibers(near=GeoFilter(_OSLO_CENTER, 50_000), limit=100)
        ids = {_norm(f.id) for f in found}
        assert _norm(oslo.id) in ids
        assert _norm(bergen.id) not in ids  # ~305 km away
        assert _norm(nowhere.id) not in ids  # metadata.location != NONE pre-filter drops it

    async def test_wider_radius_keeps_bergen(self, storage) -> None:  # type: ignore[no-untyped-def]
        oslo = await _add(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        bergen = await _add(storage, "Cafe Mocca is in Bergen Norway", _BERGEN)

        found = await storage.find_fibers(near=GeoFilter(_OSLO_CENTER, 400_000), limit=100)
        ids = {_norm(f.id) for f in found}
        assert _norm(oslo.id) in ids and _norm(bergen.id) in ids


class TestSurrealGeoRecallPipeline:
    async def test_near_through_real_pipeline(self, storage) -> None:  # type: ignore[no-untyped-def]
        oslo = await _add(storage, "Cafe Mocca is in Oslo Norway", _OSLO)
        bergen = await _add(storage, "Cafe Mocca is in Bergen Norway", _BERGEN)

        pipeline = ReflexPipeline(storage, BrainConfig())
        result = await pipeline.query("Cafe Mocca Norway", near=GeoFilter(_OSLO_CENTER, 50_000))

        matched = {_norm(x) for x in result.fibers_matched}
        assert _norm(oslo.id) in matched
        assert _norm(bergen.id) not in matched
