"""Live-DB pinning tests: the MIXED neuron_state lookup — per-id below the threshold,
set-based (and not driven by the brain index) above it.

``get_neuron_states_batch`` runs about 14 times per recall. Upstream #239 reads small
batches one record at a time (``neuron_state:state_{sid}``); our measurement showed the
set-based query stops being the slow side much earlier than upstream's 256 once it is
hinted past the brain index, so the two are combined with a MEASURED threshold
(``_DIRECT_STATE_FETCH_LIMIT``; the timing table sits next to the constant).

Why the set path needs the hint: ``idx_state_neuron`` is the composite UNIQUE
``(brain_id, neuron_id)``; for ``neuron_id IN $ids`` the planner can use only the
``brain_id`` prefix, which on a single-brain database selects every row and evaluates the
IN list after decoding each one (85 ms vs 9.7 ms measured on a copy of production).

The PLAN is pinned, not the clock — a timing assertion measures the machine.
Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.storage.surrealdb.store import _DIRECT_STATE_FETCH_LIMIT, SurrealDBStorage

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


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="neuron-state-plan-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed(store: SurrealDBStorage, n: int) -> list[str]:
    """Neurons with a state row each; returns the ids we will look up."""
    ids: list[str] = []
    for i in range(n):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"state anchor {i}")
        await store.add_neuron(neuron)
        await store.update_neuron_state(NeuronState(neuron_id=neuron.id, activation_level=0.5))
        ids.append(neuron.id)
    return ids


def _capture_sql(store: SurrealDBStorage) -> list[str]:
    zebrane: list[str] = []
    oryg = store._query

    async def obs(sql: str, **params):
        zebrane.append(sql)
        return await oryg(sql, **params)

    store._query = obs  # type: ignore[method-assign]
    return zebrane


async def _plan(store: SurrealDBStorage, sql: str, **params) -> str:
    plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN", **params)
    return json.dumps(plan, ensure_ascii=False, default=str)


async def test_the_set_path_above_the_threshold_does_not_use_the_brain_index(
    store: SurrealDBStorage,
) -> None:
    ids = await _seed(store, _DIRECT_STATE_FETCH_LIMIT + 6)
    zebrane = _capture_sql(store)

    stany = await store.get_neuron_states_batch(ids)
    assert len(stany) == len(ids), "the set path must return every state it was asked for"

    (sql,) = [s for s in zebrane if "FROM neuron_state" in s and "IN $ids" in s]
    plan = await _plan(store, sql, brain_id=store._get_brain_id(), ids=ids)

    assert "idx_state_neuron" not in plan, (
        "the batched neuron_state lookup is driven by idx_state_neuron's brain_id prefix, "
        f"which selects every row of a single-brain database. plan={plan[:400]}"
    )
    assert '"pre_decode_filter": "yes"' in plan, (
        f"the IN list is not evaluated before decoding the row. plan={plan[:400]}"
    )


async def test_below_the_threshold_the_lookup_is_a_record_read(
    store: SurrealDBStorage,
) -> None:
    """Small batches take upstream's per-id route — one record read per id, no IN list."""
    ids = await _seed(store, 5)
    zebrane = _capture_sql(store)

    stany = await store.get_neuron_states_batch(ids)

    assert len(stany) == 5
    assert not [s for s in zebrane if "IN $ids" in s], (
        f"a 5-id batch must not go through the set query: {zebrane}"
    )
    punktowe = [s for s in zebrane if "FROM neuron_state:state_" in s]
    assert len(punktowe) == 5, f"expected one record read per id, got {punktowe}"


@pytest.mark.parametrize("fold", [False, True], ids=["dashed-ids", "underscored-ids"])
async def test_both_id_spellings_return_the_same_states_on_both_sides_of_the_threshold(
    store: SurrealDBStorage, fold: bool
) -> None:
    """C5: the record path folds ``-`` to ``_``, the column stores the dashed spelling.

    Measured on a copy of production before this was fixed: an underscored id list
    returned 8 of 8 through the per-id path and 0 of 8 through the set path — so the
    same call would answer correctly below the threshold and silently empty above it.
    """
    ids = await _seed(store, _DIRECT_STATE_FETCH_LIMIT + 6)
    pytany = [i.replace("-", "_") for i in ids] if fold else list(ids)

    male = await store.get_neuron_states_batch(pytany[:5])
    duze = await store.get_neuron_states_batch(pytany)

    assert len(male) == 5, f"per-id path lost states for {'underscored' if fold else 'dashed'} ids"
    assert len(duze) == len(ids), (
        f"set path lost states for {'underscored' if fold else 'dashed'} ids: "
        f"{len(duze)} of {len(ids)}"
    )
