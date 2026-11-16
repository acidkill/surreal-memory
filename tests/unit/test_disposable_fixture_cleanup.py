"""Disposable integration fixtures remove only their own database, even on failures."""

from __future__ import annotations

import importlib
import re

import pytest

FIXTURES = [
    ("test_census_stored_form_db", "store"),
    ("test_semantic_link_resume_db", "store"),
    ("test_surrealdb_entity_ref_record_ids", "store"),
    ("test_surrealdb_fiber_id_round_trip", "store"),
    ("test_surrealdb_fiber_snapshot_round_trip", "store"),
    ("test_surrealdb_fiber_snapshot_round_trip", "target"),
    ("test_surrealdb_fiber_vector_anchors", "store"),
    ("test_surrealdb_knn_anchors", "store"),
]


@pytest.mark.parametrize("module_name,fixture_name", FIXTURES)
@pytest.mark.parametrize("failure", ["body", "initialize", "seed", "remove"])
async def test_fixture_cleans_only_unique_database_and_always_closes(
    module_name, fixture_name, failure, monkeypatch
):
    module = importlib.import_module("tests.integration." + module_name)
    events = []
    databases = {"control"}

    class FakeStorage:
        def __init__(self, **kwargs):
            self.database = kwargs["database"]
            databases.add(self.database)

        async def initialize(self):
            if failure == "initialize" or (failure == "seed" and fixture_name == "target"):
                raise RuntimeError("synthetic initialization failure")

        async def save_brain(self, brain):
            if failure == "seed":
                raise RuntimeError("synthetic seeding failure")

        def set_brain(self, brain_id):
            pass

        async def _query_response(self, query):
            events.append(("remove", self.database, query))
            if failure == "remove":
                raise RuntimeError("synthetic removal failure")
            databases.remove(self.database)

        async def close(self):
            events.append(("close", self.database))

    monkeypatch.setattr(module, "SurrealDBStorage", FakeStorage)
    monkeypatch.setenv("SURREALDB_NS", "synthetic-disposable")
    if hasattr(module, "TEST_SURREALDB_URL"):
        monkeypatch.setattr(module, "TEST_SURREALDB_URL", "http://127.0.0.1:1")
    fixture = getattr(module, fixture_name).__wrapped__()
    if failure in {"initialize", "seed"}:
        with pytest.raises(RuntimeError, match=r"initialization|seeding"):
            await anext(fixture)
    else:
        store = await anext(fixture)
        assert store.database != "control"
        if failure == "body":
            with pytest.raises(RuntimeError, match="synthetic body"):
                await fixture.athrow(RuntimeError("synthetic body"))
        else:
            with pytest.raises(RuntimeError, match="removal"):
                await fixture.aclose()
    assert [event[0] for event in events] == ["remove", "close"]
    database = events[0][1]
    assert re.fullmatch(r"it_[0-9a-f]{12}", database)
    assert events[0][2] == f"REMOVE DATABASE `{database}`"
    assert "control" in databases
    assert (database in databases) == (failure == "remove")


async def test_census_standalone_cleans_after_initialization_failure(monkeypatch):
    from tests.integration import test_consolidation_census_null_roundtrip as module

    events = []

    class FakeStorage:
        def __init__(self, **kwargs):
            self.database = kwargs["database"]

        async def initialize(self):
            raise RuntimeError("synthetic initialization failure")

        async def _query_response(self, query):
            events.append(("remove", self.database, query))

        async def close(self):
            events.append(("close", self.database))

    monkeypatch.setattr(module, "SurrealDBStorage", FakeStorage)
    monkeypatch.setenv("SURREALDB_NS", "synthetic-disposable")
    with pytest.raises(RuntimeError, match="initialization"):
        await module.test_census_round_trip_restores_fields_surrealdb_omits()
    assert [event[0] for event in events] == ["remove", "close"]
    assert re.fullmatch(r"it_[0-9a-f]{12}", events[0][1])
    assert events[0][2] == f"REMOVE DATABASE `{events[0][1]}`"
