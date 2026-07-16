"""Cross-backend parity test for get_project_memories.

Locks SQLite/SurrealDB equivalence: both backends, seeded with the same
6 TypedMemory rows, must return the same fiber_id set when filtered by
project_id. Prevents future drift between the two NeuralStorage
implementations.

The SurrealDB half of the test is skipped automatically when
SURREALDB_URL is not set, so this file is safe to run in CI without
docker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.project import Project
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.sqlite_store import SQLiteStorage
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")
_skip_surrealdb = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)


@dataclass(frozen=True)
class SeedLayout:
    """Project-id-keyed fiber sets returned by _seed_six_memories."""

    alpha_id: str
    beta_id: str
    fibers_by_project: dict[str | None, set[str]]


# ---- Seed helper ---------------------------------------------------------


async def _seed_six_memories(
    storage: NeuralStorage,
) -> SeedLayout:
    """Seed the storage with 2 projects + 6 fibers + 6 TypedMemory rows.

    One of the alpha rows is expired. Returns the project IDs so tests
    can query the correct partition.
    """
    alpha = Project.create(name="alpha")
    beta = Project.create(name="beta")
    await storage.add_project(alpha)
    await storage.add_project(beta)

    layout = [
        (alpha.id, False),
        (alpha.id, True),  # expired alpha row
        (beta.id, False),
        (beta.id, False),
        (None, False),
        (None, False),
    ]

    by_project: dict[str | None, set[str]] = {alpha.id: set(), beta.id: set(), None: set()}
    for idx, (project_id, expired) in enumerate(layout):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"seed-{idx}")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary=f"fiber-{idx}",
        )
        await storage.add_fiber(fiber)

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
            project_id=project_id,
            expires_in_days=-1 if expired else None,
        )
        await storage.add_typed_memory(typed_mem)
        by_project[project_id].add(fiber.id)

    return SeedLayout(alpha_id=alpha.id, beta_id=beta.id, fibers_by_project=by_project)


# ---- Fixtures ------------------------------------------------------------


@pytest.fixture
async def sqlite_storage(tmp_path: Path) -> SQLiteStorage:
    storage = SQLiteStorage(tmp_path / "parity.db")
    await storage.initialize()
    brain = Brain.create(name="parity-test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    if not SURREALDB_URL:
        pytest.skip("SURREALDB_URL not set")
    # Skips when the SDK is absent, and heals sys.modules when another test
    # module stubbed `surrealdb` (importorskip would accept the stub).
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="parity-test-surreal")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    # Best-effort cleanup: drop this test's brain (and stale leftovers) so
    # `smem brain list` doesn't accumulate test brains on the shared DB.
    try:
        await cleanup_live_brains(storage, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


# ---- Tests ---------------------------------------------------------------


class TestGetProjectMemoriesParity:
    @pytest.mark.asyncio
    async def test_sqlite_filters_by_project_id_excludes_expired(
        self, sqlite_storage: SQLiteStorage
    ) -> None:
        seed = await _seed_six_memories(sqlite_storage)
        rows = await sqlite_storage.get_project_memories(seed.alpha_id)
        # 2 alpha rows seeded, 1 expired → expect 1 returned
        assert len(rows) == 1
        assert rows[0].fiber_id in seed.fibers_by_project[seed.alpha_id]

    @pytest.mark.asyncio
    async def test_sqlite_include_expired_returns_both(self, sqlite_storage: SQLiteStorage) -> None:
        seed = await _seed_six_memories(sqlite_storage)
        rows = await sqlite_storage.get_project_memories(seed.alpha_id, include_expired=True)
        assert {r.fiber_id for r in rows} == seed.fibers_by_project[seed.alpha_id]
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_sqlite_other_projects_not_returned(self, sqlite_storage: SQLiteStorage) -> None:
        seed = await _seed_six_memories(sqlite_storage)
        rows = await sqlite_storage.get_project_memories(seed.beta_id)
        assert {r.fiber_id for r in rows} == seed.fibers_by_project[seed.beta_id]
        assert len(rows) == 2

        rows_none = await sqlite_storage.get_project_memories("nonexistent-project-id")
        assert rows_none == []

    @pytest.mark.asyncio
    @_skip_surrealdb
    async def test_surrealdb_matches_sqlite_partitioning(
        self, sqlite_storage: SQLiteStorage, surrealdb_storage: NeuralStorage
    ) -> None:
        """Cross-backend parity: same partition shape on both backends.

        Fiber IDs differ between backends (UUID-based, not deterministic),
        so we compare counts and the per-backend self-consistency, not raw
        fiber_id sets across backends.
        """
        sqlite_seed = await _seed_six_memories(sqlite_storage)
        surreal_seed = await _seed_six_memories(surrealdb_storage)

        # Default (exclude expired) → both backends return 1 alpha row
        sqlite_alpha = await sqlite_storage.get_project_memories(sqlite_seed.alpha_id)
        surreal_alpha = await surrealdb_storage.get_project_memories(surreal_seed.alpha_id)
        assert len(sqlite_alpha) == len(surreal_alpha) == 1

        # include_expired=True → both backends return 2 alpha rows
        sqlite_all = await sqlite_storage.get_project_memories(
            sqlite_seed.alpha_id, include_expired=True
        )
        surreal_all = await surrealdb_storage.get_project_memories(
            surreal_seed.alpha_id, include_expired=True
        )
        assert len(sqlite_all) == 2
        assert len(surreal_all) == 2

        # Each backend returns exactly the rows it seeded for that project
        assert {m.fiber_id for m in sqlite_all} == sqlite_seed.fibers_by_project[
            sqlite_seed.alpha_id
        ]
        assert {m.fiber_id for m in surreal_all} == surreal_seed.fibers_by_project[
            surreal_seed.alpha_id
        ]
