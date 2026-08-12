"""Regression: GET /api/dashboard/brains AND /api/dashboard/stats must not
repoint the shared storage.

Both `list_brains_api` and `get_stats` run per-brain diagnostics in a loop.
They used to obtain each brain's storage with `get_shared_storage(brain_name=
name)`, which on the SurrealDB backend returns the process-wide singleton
*after* calling `set_brain(name)` on it. The loop never snapshotted or
restored the original pointer, so a read-only GET left the singleton bound to
whichever brain came last in the iteration order. Everything that later reads
that same instance — the other dashboard endpoints, and the consolidation
(24h) / decay (12h) background daemons — then silently operated against the
wrong brain.

`/brains` was fixed first; `/stats` (the endpoint backing the dashboard's
DEFAULT landing page, so hit at least as often) carried the identical defect
in its own `_analyze_brain` closure and was found still leaking during this
same fix's own independent verification — fixed in the same pass once found.

The fix routes each iteration through `storage_for_scope`, the helper added for
the identical defect in the hub router's read paths: it reuses the shared
instance only when it is already bound to the requested scope, and otherwise
opens an isolated instance that is closed afterwards.

Two layers of coverage per endpoint, mirroring test_route_hub.py:

* mock-based, always runs — asserts the shared storage is never `set_brain`ed
  and that its pointer survives the request;
* live, skipped without SURREALDB_URL — the singleton behaviour that makes the
  bug real only exists on the SurrealDB backend, so the pointer is checked
  across a real request with a real second brain present.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from surreal_memory.server.dependencies import get_storage
from surreal_memory.server.routes.dashboard_api import router
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

BOUND_BRAIN = "default"
OTHER_BRAIN = "other-brain"

#: Sorts after "default", so a leak leaves the pointer on it rather than
#: coincidentally landing back on the active brain.
LIVE_THROWAWAY_BRAIN = "zz-dashboard-brains-scope-live"

#: Separate name (not reused) so the /brains and /stats live tests never race
#: on the same brain row under pytest-xdist.
LIVE_THROWAWAY_BRAIN_STATS = "zz-dashboard-stats-scope-live"

SURREALDB_URL = os.getenv("SURREALDB_URL")


def _make_client(storage: Any) -> httpx.AsyncClient:
    """An ASGI client for a bare app exposing only the dashboard router."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage] = lambda: storage
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")


class TestListBrainsDoesNotLeakBrainState:
    """The read-only listing must never mutate the shared storage's brain."""

    @pytest.fixture(autouse=True)
    def _no_real_isolated_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail loudly if the handler opens a real storage instead of a mock."""

        async def _refuse(brain_name: str | None = None) -> None:
            raise AssertionError(
                f"create_isolated_storage({brain_name!r}) reached the real backend; "
                "patch it in the test"
            )

        monkeypatch.setattr("surreal_memory.unified_config.create_isolated_storage", _refuse)

    @pytest.fixture
    def shared_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.brain_id = BOUND_BRAIN
        # set_brain is synchronous on every real storage; left as an AsyncMock
        # child it would only run its side effect when awaited, and the pointer
        # would never actually move here.
        storage.set_brain = MagicMock(side_effect=lambda name: setattr(storage, "brain_id", name))
        storage.get_stats = AsyncMock(
            return_value={"neuron_count": 1, "synapse_count": 2, "fiber_count": 3}
        )
        return storage

    @pytest.fixture
    def scoped_storage(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """The isolated instance storage_for_scope opens for OTHER_BRAIN."""
        scoped = AsyncMock()
        scoped.brain_id = OTHER_BRAIN
        scoped.get_stats = AsyncMock(
            return_value={"neuron_count": 10, "synapse_count": 20, "fiber_count": 30}
        )
        scoped.close = AsyncMock()
        monkeypatch.setattr(
            "surreal_memory.unified_config.create_isolated_storage",
            AsyncMock(return_value=scoped),
        )
        return scoped

    @pytest.fixture(autouse=True)
    def _patched_env(self, monkeypatch: pytest.MonkeyPatch, shared_storage: AsyncMock) -> None:
        cfg = MagicMock()
        cfg.current_brain = BOUND_BRAIN
        cfg.storage_backend = "surrealdb"
        monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: cfg)

        # Stand in for the real SurrealDB singleton getter, faithfully: it hands
        # back THE shared instance after repointing it. Without this the pre-fix
        # code would blow up inside the real get_shared_storage before it could
        # move the pointer, and these assertions would pass vacuously against the
        # very bug they exist to catch.
        async def _fake_get_shared_storage(brain_name: str | None = None) -> AsyncMock:
            shared_storage.set_brain(brain_name or BOUND_BRAIN)
            return shared_storage

        monkeypatch.setattr(
            "surreal_memory.unified_config.get_shared_storage",
            AsyncMock(side_effect=_fake_get_shared_storage),
        )
        monkeypatch.setattr(
            "surreal_memory.unified_config.list_available_brains",
            AsyncMock(return_value=[BOUND_BRAIN, OTHER_BRAIN]),
        )
        engine = MagicMock()
        engine.return_value.analyze = AsyncMock(return_value=MagicMock(grade="B", purity_score=0.8))
        monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine", engine)

    async def test_listing_does_not_call_set_brain_on_shared_storage(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/brains")

        assert resp.status_code == 200, resp.text
        shared_storage.set_brain.assert_not_called()

    async def test_listing_leaves_shared_storage_brain_id_unchanged(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        """The exact bug: after listing, the singleton was bound to the LAST
        brain in the iteration order instead of the active one, so the
        background consolidation/decay passes ran against the wrong brain."""
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/brains")

        assert resp.status_code == 200, resp.text
        assert shared_storage.brain_id == BOUND_BRAIN

    async def test_other_brain_is_read_through_an_isolated_storage(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/brains")

        assert resp.status_code == 200, resp.text
        # The bound brain reuses the shared instance; the other one does not.
        shared_storage.get_stats.assert_awaited_once_with(BOUND_BRAIN)
        scoped_storage.get_stats.assert_awaited_once_with(OTHER_BRAIN)
        scoped_storage.close.assert_awaited_once()

    async def test_summaries_carry_per_brain_stats_and_grade(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        """Guards against "fixed the leak, broke the payload"."""
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/brains")

        by_name = {row["name"]: row for row in resp.json()}
        assert by_name[BOUND_BRAIN]["neuron_count"] == 1
        assert by_name[BOUND_BRAIN]["is_active"] is True
        assert by_name[OTHER_BRAIN]["neuron_count"] == 10
        assert by_name[OTHER_BRAIN]["is_active"] is False
        assert by_name[OTHER_BRAIN]["grade"] == "B"


class TestGetStatsDoesNotLeakBrainState:
    """The overview endpoint's read-only GET must never mutate the shared storage."""

    @pytest.fixture(autouse=True)
    def _no_real_isolated_storage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _refuse(brain_name: str | None = None) -> None:
            raise AssertionError(
                f"create_isolated_storage({brain_name!r}) reached the real backend; "
                "patch it in the test"
            )

        monkeypatch.setattr("surreal_memory.unified_config.create_isolated_storage", _refuse)

    @pytest.fixture
    def shared_storage(self) -> AsyncMock:
        storage = AsyncMock()
        storage.brain_id = BOUND_BRAIN
        storage.set_brain = MagicMock(side_effect=lambda name: setattr(storage, "brain_id", name))
        storage.get_stats = AsyncMock(
            return_value={"neuron_count": 1, "synapse_count": 2, "fiber_count": 3}
        )
        return storage

    @pytest.fixture
    def scoped_storage(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        scoped = AsyncMock()
        scoped.brain_id = OTHER_BRAIN
        scoped.get_stats = AsyncMock(
            return_value={"neuron_count": 10, "synapse_count": 20, "fiber_count": 30}
        )
        scoped.close = AsyncMock()
        monkeypatch.setattr(
            "surreal_memory.unified_config.create_isolated_storage",
            AsyncMock(return_value=scoped),
        )
        return scoped

    @pytest.fixture(autouse=True)
    def _patched_env(self, monkeypatch: pytest.MonkeyPatch, shared_storage: AsyncMock) -> None:
        cfg = MagicMock()
        cfg.current_brain = BOUND_BRAIN
        cfg.storage_backend = "surrealdb"
        monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda: cfg)
        monkeypatch.setattr(
            "surreal_memory.unified_config.list_available_brains",
            AsyncMock(return_value=[BOUND_BRAIN, OTHER_BRAIN]),
        )
        # get_stats reaches diagnostics through _cached_grade_purity (a TTL
        # cache wrapping DiagnosticsEngine), not DiagnosticsEngine directly —
        # bypass the cache entirely so this test exercises the scope-leak fix,
        # not cache timing.
        monkeypatch.setattr(
            "surreal_memory.server.routes.dashboard_api._cached_grade_purity",
            AsyncMock(return_value=("B", 0.8)),
        )

    async def test_stats_does_not_call_set_brain_on_shared_storage(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/stats")

        assert resp.status_code == 200, resp.text
        shared_storage.set_brain.assert_not_called()

    async def test_stats_leaves_shared_storage_brain_id_unchanged(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        """The exact bug: after the overview loaded, the singleton was bound to
        the LAST brain in the iteration order instead of the active one, so the
        background consolidation/decay passes ran against the wrong brain."""
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/stats")

        assert resp.status_code == 200, resp.text
        assert shared_storage.brain_id == BOUND_BRAIN

    async def test_stats_other_brain_is_read_through_an_isolated_storage(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/stats")

        assert resp.status_code == 200, resp.text
        shared_storage.get_stats.assert_awaited_once_with(BOUND_BRAIN)
        scoped_storage.get_stats.assert_awaited_once_with(OTHER_BRAIN)
        scoped_storage.close.assert_awaited_once()

    async def test_stats_totals_and_active_grade_are_correct(
        self, shared_storage: AsyncMock, scoped_storage: AsyncMock
    ) -> None:
        """Guards against "fixed the leak, broke the payload". Only the active
        brain runs diagnostics (get_stats' own cost-saving design, unrelated to
        this fix) — the non-active brain must still show a real count with a
        placeholder grade, not be silently dropped from the totals."""
        async with _make_client(shared_storage) as client:
            resp = await client.get("/api/dashboard/stats")

        body = resp.json()
        assert body["active_brain"] == BOUND_BRAIN
        assert body["total_brains"] == 2
        assert body["total_neurons"] == 1 + 10
        assert body["health_grade"] == "B"

        by_name = {row["name"]: row for row in body["brains"]}
        assert by_name[BOUND_BRAIN]["is_active"] is True
        assert by_name[BOUND_BRAIN]["grade"] == "B"
        assert by_name[OTHER_BRAIN]["is_active"] is False
        assert by_name[OTHER_BRAIN]["neuron_count"] == 10
        # Non-active brains skip diagnostics by design; must not silently
        # inherit the active brain's cached grade.
        assert by_name[OTHER_BRAIN]["grade"] == "—"


@pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)
class TestListBrainsScopeLive:
    """The singleton pointer must survive a real request against a real store."""

    async def test_request_leaves_the_shared_singleton_on_the_active_brain(self) -> None:
        ensure_real_surrealdb_sdk()

        from surreal_memory.core.brain import Brain
        from surreal_memory.unified_config import get_shared_storage, list_available_brains

        storage = await get_shared_storage()
        active = storage.current_brain_id
        assert active is not None

        # Create the second brain directly: get_shared_storage(brain_name=...)
        # would itself repoint the singleton and mask the assertion.
        if await storage.get_brain(LIVE_THROWAWAY_BRAIN) is None:
            await storage.save_brain(
                Brain.create(LIVE_THROWAWAY_BRAIN, brain_id=LIVE_THROWAWAY_BRAIN)
            )
        storage.set_brain(active)

        try:
            names = await list_available_brains()
            assert LIVE_THROWAWAY_BRAIN in names
            assert len(names) > 1, "the leak needs at least two brains to be observable"
            # The pointer only ends up somewhere wrong if the active brain is
            # not the last one visited.
            assert names[-1] != active

            storage.set_brain(active)
            async with _make_client(storage) as client:
                resp = await client.get("/api/dashboard/brains")

            assert resp.status_code == 200, resp.text
            assert storage.current_brain_id == active
        finally:
            storage.set_brain(active)
            try:
                await cleanup_live_brains(storage, own_brain_id=LIVE_THROWAWAY_BRAIN)
            finally:
                storage.set_brain(active)


@pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)
class TestGetStatsScopeLive:
    """The singleton pointer must survive a real overview request against a
    real store — this is the endpoint that was found still leaking after
    /brains had already been fixed, so it gets the same live proof.
    """

    async def test_request_leaves_the_shared_singleton_on_the_active_brain(self) -> None:
        ensure_real_surrealdb_sdk()

        from surreal_memory.core.brain import Brain
        from surreal_memory.unified_config import get_shared_storage, list_available_brains

        storage = await get_shared_storage()
        active = storage.current_brain_id
        assert active is not None

        if await storage.get_brain(LIVE_THROWAWAY_BRAIN_STATS) is None:
            await storage.save_brain(
                Brain.create(LIVE_THROWAWAY_BRAIN_STATS, brain_id=LIVE_THROWAWAY_BRAIN_STATS)
            )
        storage.set_brain(active)

        try:
            names = await list_available_brains()
            assert LIVE_THROWAWAY_BRAIN_STATS in names
            assert len(names) > 1, "the leak needs at least two brains to be observable"
            assert names[-1] != active

            storage.set_brain(active)
            async with _make_client(storage) as client:
                resp = await client.get("/api/dashboard/stats")

            assert resp.status_code == 200, resp.text
            assert storage.current_brain_id == active
        finally:
            storage.set_brain(active)
            try:
                await cleanup_live_brains(storage, own_brain_id=LIVE_THROWAWAY_BRAIN_STATS)
            finally:
                storage.set_brain(active)
