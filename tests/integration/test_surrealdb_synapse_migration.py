"""Live integration tests for the synapse->RELATE migration (RUN-005 U6).

These run against a REAL SurrealDB >= 3.2.0 (skipped unless SURREALDB_URL is set).
Each test seeds a v7 flat ``synapse`` table (source_id/target_id string columns)
directly, then drives the real SurrealDBStorage.initialize() upgrade path and
asserts the migration is faithful: edge count, every id resolvable, endpoints
(incl. self-loop / orphan / dashed neuron id) preserved, fiber.synapse_ids still
resolve, export deep-equal, Merkle root identical, get_neighbors/get_path parity,
second initialize() no-op, and two concurrent migrations run exactly once.

Run (real-db-test-runner drives this):
    SURREALDB_URL=http://localhost:<port> SURREALDB_USER=root SURREALDB_PASS=... \
        uv run --extra dev --extra server --extra surrealdb \
        pytest tests/integration/test_surrealdb_synapse_migration.py -m integration -q
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, Iterator

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb import migrations as M  # noqa: N812
from surreal_memory.storage.surrealdb.store import SurrealDBStorage, _to_surreal_id

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
# The live station database — throwaway ``it_<hex>`` test brains must NEVER be
# created here. This suite used to read the generic ``SURREALDB_NS``, so a run
# that inherited the station's ``SURREALDB_NS=surreal_memory`` leaked residue
# ``it_<hex>`` databases straight into the live station DB (the same class of
# test-isolation bug fixed in PR #47). The tests now run in a dedicated,
# ephemeral namespace that can never be the station namespace and is cleaned up
# in teardown. A dedicated QA container may pin the namespace via
# ``SURREALDB_IT_NS``; otherwise we mint a unique one per session and drop it
# wholesale when the session ends.
_STATION_NS = "surreal_memory"
_PINNED_NS = os.getenv("SURREALDB_IT_NS")
SURREALDB_NS = _PINNED_NS or f"smem_it_{uuid.uuid4().hex[:8]}"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
    pytest.mark.skipif(
        SURREALDB_NS == _STATION_NS,
        reason=(
            f"refusing to run against the station namespace {_STATION_NS!r}: "
            "unset SURREALDB_IT_NS or point it at an isolated namespace"
        ),
    ),
]


@pytest_asyncio.fixture
async def fresh_db() -> AsyncGenerator[str, None]:
    """Yield a throwaway ``it_<hex>`` database name in the isolated test
    namespace and drop it (``REMOVE DATABASE``) on teardown — even if the test
    body raises — so no residue brains survive in any shared or station DB.
    """
    database = "it_" + uuid.uuid4().hex[:12]
    try:
        yield database
    finally:
        conn = await _raw_conn(database)
        try:
            await conn.query(f"REMOVE DATABASE IF EXISTS `{database}`")
        finally:
            close = getattr(conn, "close", None)
            if close is not None:
                await close()


@pytest.fixture(scope="session", autouse=True)
def _reap_ephemeral_namespace() -> Iterator[None]:
    """Best-effort sweep: when the namespace was auto-minted for this run, drop
    it wholesale after the session so empty ``smem_it_*`` namespaces don't
    accumulate on the server. Skipped when ``SURREALDB_IT_NS`` pins a
    caller-owned namespace (e.g. a dedicated QA container) — ``fresh_db`` already
    drops each database there, so the namespace itself is left intact. Never
    fails an otherwise-green run.
    """
    yield
    if not SURREALDB_URL or _PINNED_NS or SURREALDB_NS == _STATION_NS:
        return

    async def _reap() -> None:
        from surrealdb import AsyncSurreal

        conn = AsyncSurreal(SURREALDB_URL)
        await conn.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
        try:
            await conn.use(SURREALDB_NS, "reap")
            await conn.query(f"REMOVE NAMESPACE IF EXISTS `{SURREALDB_NS}`")
        finally:
            close = getattr(conn, "close", None)
            if close is not None:
                await close()

    try:
        asyncio.run(_reap())
    except Exception:  # cleanup must never fail an otherwise-green run
        pass


async def _raw_conn(database: str):
    from surrealdb import AsyncSurreal

    conn = AsyncSurreal(SURREALDB_URL)
    await conn.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
    await conn.use(SURREALDB_NS, database)
    return conn


def _store(database: str) -> SurrealDBStorage:
    return SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database=database,
    )


async def _seed_v7(conn, seeder: SurrealDBStorage, brain_id: str) -> dict[str, dict]:
    """Create neurons, a fiber, and a flat v7 ``synapse`` table + rows via a raw conn.

    Returns the expected synapse specs keyed by synapse id (dashed form).
    """
    # neurons (via the store so the neuron rows carry every field _row_to_neuron reads)
    neurons: dict[str, Neuron] = {}
    for name in ("a", "b", "c"):
        n = Neuron.create(type=NeuronType.CONCEPT, content=name)
        await seeder.add_neuron(n)
        neurons[name] = n
    # a neuron whose id contains dashes (UUID) — its sanitized endpoint must round-trip
    dashed = Neuron.create(type=NeuronType.CONCEPT, content="dashed")
    await seeder.add_neuron(dashed)
    neurons["dashed"] = dashed

    a, b, c, d = (neurons[k].id for k in ("a", "b", "c", "dashed"))

    # flat v7 synapse rows (raw CREATE auto-creates a TYPE NORMAL table)
    specs = {
        "e1": (a, b, "related_to", 1.0),
        "e2": (b, c, "related_to", 1.0),
        "e_self": (a, a, "related_to", 1.0),  # self-loop
        "e_orphan": (a, "missingneuron", "related_to", 1.0),  # orphan endpoint
        "e_dash": (a, d, "related_to", 2.0),  # dashed neuron id endpoint
        "e_meta": (b, c, "alias", 1.0),  # non-empty NESTED metadata (regression guard)
    }
    # Nested metadata keys (e.g. {"_dedup": true}) are the regression guard for the
    # v2.6.1 FLEXIBLE fix: a SCHEMAFULL RELATION table with a plain `TYPE object`
    # metadata field rejected undefined nested keys, so v2.6.0 silently SKIPPED every
    # synapse with non-empty metadata. This row must survive with its metadata intact.
    metas: dict[str, dict] = {"e_meta": {"_dedup": True, "note": {"nested": "keeps"}}}
    expected: dict[str, dict] = {}
    for sid, (src, tgt, styp, weight) in specs.items():
        ss = _to_surreal_id(src)
        st = _to_surreal_id(tgt)
        meta = metas.get(sid, {})
        await conn.query(
            f"CREATE synapse:{sid} SET brain_id=$b, type=$t, source_id=$s, target_id=$g, "
            "weight=$w, direction='uni', created_at=time::now(), reinforced_count=0, "
            "metadata=$m",
            {"b": brain_id, "t": styp, "s": ss, "g": st, "w": weight, "m": meta},
        )
        expected[sid] = {
            "id": sid.replace("_", "-"),
            "source_id": ss.replace("_", "-"),
            "target_id": st.replace("_", "-"),
            "type": styp,
            "weight": weight,
            "metadata": meta,
        }

    # a fiber referencing the seeded synapse ids
    fiber = Fiber.create(
        neuron_ids={a, b, c},
        synapse_ids={s.replace("_", "-") for s in specs},
        anchor_neuron_id=a,
        summary="seed-fiber",
    )
    await seeder.add_fiber(fiber)
    return expected


def _syn_key(d: dict) -> str:
    return str(d["id"])


@pytest.mark.asyncio
async def test_v7_upgrade_preserves_count_ids_endpoints_export_and_merkle(
    fresh_db: str,
) -> None:
    db = fresh_db
    conn = await _raw_conn(db)
    brain = Brain.create(name="upgrade-brain")

    # seeder store shares the raw conn (no initialize -> no migration yet)
    seeder = _store(db)
    seeder._conn = conn
    seeder._current_brain_id = brain.id
    await seeder.save_brain(brain)
    expected = await _seed_v7(conn, seeder, brain.id)

    # pre-migration snapshot (reads flat rows via the _row_to_synapse fallback)
    pre_export = await seeder.export_brain(brain.id)
    pre_merkle = await seeder.get_merkle_root(is_pro=True)
    pre_synapses = sorted(pre_export.synapses, key=_syn_key)
    assert len(pre_synapses) == len(expected)

    # upgrade: the real initialize() path (version gate -> ensure_schema -> migrate)
    store = _store(db)
    await store.initialize()
    store.set_brain(brain.id)

    # synapse table is now a native RELATION
    info = await conn.query("INFO FOR DB")
    sdef = (info.get("tables", {}) or {}).get("synapse", "")
    assert "TYPE RELATION" in str(sdef)

    # edge count preserved
    edges = await store.get_all_synapses()
    assert len(edges) == len(expected)

    by_id = {e.id: e for e in edges}
    for exp in expected.values():
        eid = exp["id"]
        assert eid in by_id, f"synapse {eid} lost in migration"
        got = by_id[eid]
        assert got.source_id == exp["source_id"]
        assert got.target_id == exp["target_id"]
        assert got.type.value == exp["type"]
        assert got.weight == exp["weight"]
        assert got.metadata == exp["metadata"], f"metadata lost for {eid}"

    # regression (v2.6.1 FLEXIBLE fix): a synapse with NESTED metadata survives with
    # its keys intact — on v2.6.0 this row was silently skipped (count would drop).
    assert by_id["e-meta"].metadata == {"_dedup": True, "note": {"nested": "keeps"}}

    # self-loop / orphan specifically survive
    assert by_id["e-self"].source_id == by_id["e-self"].target_id
    assert by_id["e-orphan"].target_id == "missingneuron"

    # fiber.synapse_ids still resolve to real edges
    fibers = await store.get_fibers(limit=100)
    assert fibers
    fib = fibers[0]
    for sid in fib.synapse_ids:
        assert await store.get_synapse(sid) is not None, f"fiber synapse {sid} unresolved"

    # export deep-equal (byte-stable wire format) + Merkle root identical
    post_export = await store.export_brain(brain.id)
    post_synapses = sorted(post_export.synapses, key=_syn_key)
    assert post_synapses == pre_synapses
    post_merkle = await store.get_merkle_root(is_pro=True)
    assert post_merkle == pre_merkle

    # schema_meta stamped + backup retained
    assert await M._read_stamped_version(conn) == 8
    assert await M._count(conn, M.BACKUP_TABLE) == len(expected)

    # get_neighbors + get_path parity via native edges
    neigh = await store.get_neighbors(neuron_id=_neuron_id(pre_export, "a"), direction="out")
    assert any(n.content == "b" for n, _ in neigh)
    path = await store.get_path(
        _neuron_id(pre_export, "a"), _neuron_id(pre_export, "c"), max_hops=4
    )
    assert path is not None and path[-1][0].id == _neuron_id(pre_export, "c")


def _neuron_id(export, content: str) -> str:
    for n in export.neurons:
        if n["content"] == content:
            return str(n["id"])
    raise AssertionError(f"neuron with content {content!r} not found")


@pytest.mark.asyncio
async def test_second_initialize_is_noop(fresh_db: str) -> None:
    db = fresh_db
    conn = await _raw_conn(db)
    brain = Brain.create(name="noop-brain")
    seeder = _store(db)
    seeder._conn = conn
    seeder._current_brain_id = brain.id
    await seeder.save_brain(brain)
    expected = await _seed_v7(conn, seeder, brain.id)

    store1 = _store(db)
    await store1.initialize()
    store1.set_brain(brain.id)
    count1 = len(await store1.get_all_synapses())
    assert count1 == len(expected)

    # second connect must not re-migrate or duplicate
    store2 = _store(db)
    await store2.initialize()
    store2.set_brain(brain.id)
    assert await M._read_stamped_version(conn) == 8
    assert len(await store2.get_all_synapses()) == len(expected)


@pytest.mark.asyncio
async def test_two_parallel_apply_migrations_run_exactly_once(fresh_db: str) -> None:
    db = fresh_db
    conn = await _raw_conn(db)
    brain = Brain.create(name="parallel-brain")
    seeder = _store(db)
    seeder._conn = conn
    seeder._current_brain_id = brain.id
    await seeder.save_brain(brain)
    expected = await _seed_v7(conn, seeder, brain.id)

    # ensure the v8 schema exists (as first connect would) before the race
    from surreal_memory.storage.surrealdb.schema import ensure_schema

    await ensure_schema(conn)

    conn2 = await _raw_conn(db)
    results = await asyncio.gather(
        M.apply_migrations(conn),
        M.apply_migrations(conn2),
        return_exceptions=True,
    )
    # both calls return 8 (one migrates, the other observes the completed migration);
    # any MigrationLockError is acceptable, but no duplication may occur.
    for r in results:
        assert r == 8 or isinstance(r, M.MigrationError), f"unexpected result {r!r}"

    assert await M._read_stamped_version(conn) == 8
    assert await M._count(conn, "synapse") == len(expected)  # migrated exactly once
    assert await M._count(conn, M.BACKUP_TABLE) == len(expected)


@pytest.mark.asyncio
async def test_fresh_db_is_v8_directly_no_backup(fresh_db: str) -> None:
    db = fresh_db
    conn = await _raw_conn(db)
    brain = Brain.create(name="fresh-brain")

    store = _store(db)
    await store.initialize()  # fresh DB -> v8 directly, no migration
    store.set_brain(brain.id)
    await store.save_brain(brain)

    info = await conn.query("INFO FOR DB")
    sdef = (info.get("tables", {}) or {}).get("synapse", "")
    assert "TYPE RELATION" in str(sdef)
    assert await M._read_stamped_version(conn) == 8
    # no migration ran -> no backup table rows
    assert await M._count(conn, M.BACKUP_TABLE) == 0
