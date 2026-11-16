"""Indexed atomic-prune plans and eligible 32-edge pages on disposable SurrealDB."""

from __future__ import annotations

import time
from dataclasses import replace
from uuid import uuid4

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.consolidation_progress import (
    ConsolidationCheckpointUnconfirmedError,
    ConsolidationProgressSession,
)
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.storage.surrealdb.prune import _PRUNE_TRANSACTION
from tests.integration.test_prune_atomic_db import clients as clients

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("lost_response", [False, True])
async def test_atomic_eligible_32_with_high_degree_and_foreign_fibers(
    clients, monkeypatch, lost_response
):
    first, _, edge, policy = clients
    brain_id = first.brain_id
    assert await first.release_consolidation_lease(brain_id, "owner")
    # Distinct targets keep the group-all target count above one even after each deletion.
    edges = [replace(edge, id=str(uuid4()), target_id=str(uuid4()), weight=0.04) for _ in range(32)]
    keepers = [
        replace(edge, id=str(uuid4()), target_id=str(uuid4()), weight=0.9) for _ in range(1024)
    ]
    for item in edges:
        await first.add_synapse(item)
    for start in range(0, len(keepers), 128):
        await first.add_synapses_batch(keepers[start : start + 128], record_change=False)
    degree = await first._query(
        "SELECT count() AS count FROM synapse WHERE brain_id = $brain "
        "AND in = type::record('neuron', $source) GROUP ALL",
        brain=brain_id,
        source=_to_surreal_id(edge.source_id),
    )
    assert degree and degree[0]["count"] >= 1056
    foreign = Brain.create(name="synthetic-foreign")
    await first.save_brain(foreign)
    first.set_brain(foreign.id)
    for i in range(128):
        fiber = Fiber.create(
            neuron_ids={edge.source_id},
            synapse_ids={edges[i % len(edges)].id},
            anchor_neuron_id=edge.source_id,
        )
        await first.add_fiber(replace(fiber, pinned=True, salience=0.9))
    first.set_brain(brain_id)
    edges = await first.get_synapses_by_ids([item.id for item in edges])
    assert len(edges) == 32 and all(item.weight >= 0.02 and policy.eligible(item) for item in edges)
    assert await first.acquire_consolidation_lease(brain_id, "owner", lease_seconds=120)
    session = ConsolidationProgressSession(first, brain_id, "owner", {})
    original = first._ensure_conn().query_raw
    batches = []

    async def capture(sql, params=None, **kwargs):
        if sql == _PRUNE_TRANSACTION:
            batches.append(len(params["items"]))
            result = await original(sql, params, **kwargs)
            if lost_response and len(batches) == 1:
                raise ConnectionError("synthetic loss after commit")
            return result
        return await original(sql, params, **kwargs)

    monkeypatch.setattr(first._ensure_conn(), "query_raw", capture)
    start = time.monotonic()

    async def run_page(owner, progress):
        return await first.prune_synapses_if_unchanged(
            edges,
            pending_ids=[item.id for item in edges],
            policy=policy,
            run_id="synthetic-indexed-run",
            operation_id="synthetic-indexed-operation",
            lease_owner_token=owner,
            page_guard=progress.lease_fenced_page,
        )

    if lost_response:
        with pytest.raises(ConsolidationCheckpointUnconfirmedError):
            await run_page("owner", session)
        assert session.lease_lost and session._checkpoint_unconfirmed
    else:
        assert await run_page("owner", session) == {item.id: "deleted" for item in edges}
    elapsed = time.monotonic() - start
    assert batches == [32] and elapsed < 60
    assert await first.get_synapses_by_ids([item.id for item in edges]) == []
    assert len(await first.get_synapses_by_ids([keepers[0].id])) == 1
    rows = await first._query(
        "SELECT entity_id FROM change_log WHERE brain_id = $brain AND entity_type = 'synapse' "
        "AND operation = 'delete' AND payload._prune.run_id = $run",
        brain=brain_id,
        run="synthetic-indexed-run",
    )
    assert {row["entity_id"] for row in rows} == {item.id for item in edges}
    if lost_response:
        assert await first.release_consolidation_lease(brain_id, "owner")
        assert await first.acquire_consolidation_lease(brain_id, "next-owner", lease_seconds=120)
        session = ConsolidationProgressSession(first, brain_id, "next-owner", {})
    replay = await run_page("next-owner" if lost_response else "owner", session)
    assert replay == {item.id: "deleted" for item in edges}
    again = await first._query(
        "SELECT entity_id FROM change_log WHERE brain_id = $brain AND entity_type = 'synapse' "
        "AND operation = 'delete' AND payload._prune.run_id = $run",
        brain=brain_id,
        run="synthetic-indexed-run",
    )
    assert len(again) == 32


def _plan_has_operator(plan, operator):
    if isinstance(plan, list):
        return any(_plan_has_operator(item, operator) for item in plan)
    if not isinstance(plan, dict):
        return False
    aliases = {"IndexScan": "Iterate Index", "TableScan": "Iterate Table", "Filter": "Filter"}
    return (
        plan.get("operator") == operator
        or plan.get("operation") == aliases[operator]
        or any(_plan_has_operator(value, operator) for value in plan.values())
    )


@pytest.mark.parametrize(
    "name,index,scalar,old_expression",
    [
        ("refs", "idx_fiber_synapses", "$public_id", "$item.public_id"),
        ("source_protection", "idx_fiber_neurons", "$source_id", "$item.source_id"),
        ("target_protection", "idx_fiber_neurons", "$target_id", "$item.target_id"),
        ("targets", "idx_synapse_in", "$source_record", "$row.in"),
    ],
)
async def test_actual_for_selector_plan_uses_index(clients, name, index, scalar, old_expression):
    import re

    first, _, edge, _ = clients
    # Pull the actual production selector and scalar declarations out of the transaction.
    # A separate negative control replaces just its bound scalar with the old nested expression.
    match = re.search(
        rf"LET \${name} = IF .*?\{{\s*\((SELECT .*?)\)\s*\}} ELSE",
        _PRUNE_TRANSACTION,
        flags=re.DOTALL,
    )
    assert match, name
    selector = match.group(1)
    assert index in selector and scalar in selector
    declarations = re.findall(
        r"LET \$(?:public_id|source_id|target_id|source_record)\s*=\s*[^;]+;",
        _PRUNE_TRANSACTION,
    )
    assert len(declarations) == 4
    declarations = "\n".join(declarations)
    item = {"public_id": edge.id, "source_id": edge.source_id, "target_id": edge.target_id}
    for label, query in (("new", selector), ("old", selector.replace(scalar, old_expression))):
        sql = f"""
            FOR $item IN $items {{
                LET $row = {{in: type::record('neuron', $source_record)}};
                {declarations}
                RETURN ({query} EXPLAIN);
            }};
        """
        from surreal_memory.storage.surrealdb.prune import _checked_prune_results

        raw = await first._ensure_conn().query_raw(
            sql,
            {
                "items": [item],
                "source_record": _to_surreal_id(edge.source_id),
                "brain_id": first.brain_id,
            },
        )
        plans = _checked_prune_results(raw)
        assert len(plans) == 1
        plan = plans[0]["result"]
        if label == "new":
            assert _plan_has_operator(plan, "IndexScan") and index in str(plan), plan
            assert not _plan_has_operator(plan, "TableScan"), plan
        elif name == "targets":
            assert _plan_has_operator(plan, "IndexScan") and index in str(plan), plan
            assert _plan_has_operator(plan, "Filter"), plan
            assert _to_surreal_id(edge.source_id) not in str(plan), plan
        else:
            assert _plan_has_operator(plan, "TableScan"), plan
        if label == "new" and name == "targets":
            assert str(first.brain_id) in str(plan) and _to_surreal_id(edge.source_id) in str(
                plan
            ), plan
            assert not _plan_has_operator(plan, "Filter"), plan
