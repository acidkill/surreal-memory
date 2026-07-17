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

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb import migrations as M  # noqa: N812
from surreal_memory.storage.surrealdb.store import SurrealDBStorage, _to_surreal_id

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]


def _fresh_db() -> str:
    return "it_" + uuid.uuid4().hex[:12]


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
async def test_v7_upgrade_preserves_count_ids_endpoints_export_and_merkle() -> None:
    db = _fresh_db()
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
    assert await M._read_stamped_version(conn) == 9
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
async def test_second_initialize_is_noop() -> None:
    db = _fresh_db()
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
    assert await M._read_stamped_version(conn) == 9
    assert len(await store2.get_all_synapses()) == len(expected)


@pytest.mark.asyncio
async def test_two_parallel_apply_migrations_run_exactly_once() -> None:
    db = _fresh_db()
    conn = await _raw_conn(db)
    brain = Brain.create(name="parallel-brain")
    seeder = _store(db)
    seeder._conn = conn
    seeder._current_brain_id = brain.id
    await seeder.save_brain(brain)
    expected = await _seed_v7(conn, seeder, brain.id)

    # ensure the current schema exists (as first connect would) before the race
    from surreal_memory.storage.surrealdb.schema import ensure_schema

    await ensure_schema(conn)

    conn2 = await _raw_conn(db)
    results = await asyncio.gather(
        M.apply_migrations(conn),
        M.apply_migrations(conn2),
        return_exceptions=True,
    )

    # Acceptable per-call outcomes under the race: 9 = TARGET_VERSION (this call drove the
    # 7->8->9 migration or observed it already complete), a MigrationError/MigrationLockError
    # (lost the lock), or a *retryable* SurrealDB write-conflict — two live connections
    # stamping/DDL-ing concurrently can collide, and the server explicitly flags these
    # "This transaction can be retried". None of these may break the exactly-once guarantee,
    # which the state assertions below verify independently (they fail if any transient left
    # the migration incomplete), so a retryable transient is only tolerated when the end
    # state is still correct.
    def _acceptable(r: object) -> bool:
        if r == 9 or isinstance(r, M.MigrationError):
            return True
        return isinstance(r, Exception) and "can be retried" in str(r).lower()

    for r in results:
        assert _acceptable(r), f"unexpected result {r!r}"

    # Regardless of which call won the race, the migration must have completed exactly once.
    assert await M._read_stamped_version(conn) == 9
    assert await M._count(conn, "synapse") == len(expected)  # migrated exactly once
    assert await M._count(conn, M.BACKUP_TABLE) == len(expected)


@pytest.mark.asyncio
async def test_fresh_db_is_target_version_directly_no_backup() -> None:
    db = _fresh_db()
    conn = await _raw_conn(db)
    brain = Brain.create(name="fresh-brain")

    store = _store(db)
    await store.initialize()  # fresh DB -> v9 (target) directly, no migration
    store.set_brain(brain.id)
    await store.save_brain(brain)

    info = await conn.query("INFO FOR DB")
    sdef = (info.get("tables", {}) or {}).get("synapse", "")
    assert "TYPE RELATION" in str(sdef)
    assert await M._read_stamped_version(conn) == 9
    # no migration ran -> no backup table rows
    assert await M._count(conn, M.BACKUP_TABLE) == 0
