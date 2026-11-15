"""Activation CAS against an explicitly registered disposable SurrealDB target."""

from __future__ import annotations

from copy import deepcopy

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb.activation_migration import ActivationMigrationError
from tests.integration.test_prune_atomic_db import clients as clients

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def default_clients(clients):
    first, second, _, _ = clients
    brain = Brain.create(name="default", metadata={"marker": "synthetic-migration"})
    await first.save_brain(brain)
    await first._query(
        "UPDATE type::record('brain', $id) SET config.future_knob = "
        "{ nullable: NULL, nested: [1, NULL, NONE, { name: 'synthetic' }] }",
        id=brain.id,
    )
    first.set_brain("default")
    second.set_brain("default")
    return first, second, brain.id


async def migrate(store, snapshot, source="classic", target="hybrid"):
    return await store.set_activation_strategy(
        brain_id=snapshot["brain_id"],
        expected_config=snapshot["config"],
        expected_config_sha256=snapshot["config_sha256"],
        from_strategy=source,
        to_strategy=target,
    )


async def test_real_record_id_and_forward_reverse_preserve_full_config(default_clients):
    first, _, brain_id = default_clients
    rows = await first._query("SELECT * FROM brain WHERE name = $name", name="default")
    assert len(rows) == 1
    record_id = rows[0]["id"]
    assert record_id.table_name == "brain"
    assert str(record_id.id) == brain_id
    typed_before = (
        await first._query(
            "SELECT type::string(config.future_knob) AS typed FROM brain WHERE name = $name",
            name="default",
        )
    )[0]["typed"]
    assert "NULL" in typed_before and "NONE" in typed_before
    before = await first.get_activation_snapshot()
    assert before["brain_id"] == brain_id
    assert before["config"] == rows[0]["config"]
    assert before["config"]["activation_strategy"] == "classic"
    assert before["config"]["future_knob"] == {
        "nullable": None,
        "nested": [1, None, None, {"name": "synthetic"}],
    }
    assert rows[0]["metadata"]["marker"] == "synthetic-migration"
    assert (await migrate(first, before))["confirmed"] is True
    after = await first.get_activation_snapshot()
    expected = deepcopy(before["config"])
    expected["activation_strategy"] = "hybrid"
    assert after["config"] == expected
    typed_after = (
        await first._query(
            "SELECT type::string(config.future_knob) AS typed FROM brain WHERE name = $name",
            name="default",
        )
    )[0]["typed"]
    assert typed_after == typed_before
    full_expected = deepcopy(rows[0])
    full_expected["config"]["activation_strategy"] = "hybrid"
    assert (await first._query("SELECT * FROM brain WHERE name = $name", name="default"))[
        0
    ] == full_expected
    assert (await migrate(first, after, "hybrid", "classic"))["confirmed"] is True
    assert (await first.get_activation_snapshot())["config"] == before["config"]
    assert (await first._query("SELECT * FROM brain WHERE name = $name", name="default"))[
        0
    ] == rows[0]


async def test_active_default_scope_lease_blocks_transition(default_clients):
    first, second, _ = default_clients
    before = await first.get_activation_snapshot()
    assert await second.acquire_consolidation_lease(
        "default", "migration-test-owner", lease_seconds=30
    )
    try:
        with pytest.raises(ActivationMigrationError, match="lease"):
            await migrate(first, before)
        assert await first.get_activation_snapshot() == before
    finally:
        assert await second.release_consolidation_lease("default", "migration-test-owner")


async def test_full_config_drift_rejected(default_clients):
    first, second, brain_id = default_clients
    before = await first.get_activation_snapshot()
    await second._query(
        "UPDATE type::record('brain', $id) SET config.max_context_tokens = $value",
        id=brain_id,
        value=before["config"]["max_context_tokens"] + 17,
    )
    drifted = await second.get_activation_snapshot()
    assert drifted["config"] != before["config"]
    with pytest.raises(ActivationMigrationError, match="changed"):
        await migrate(first, before)
    assert await first.get_activation_snapshot() == drifted


async def test_lost_update_reply_reconciles_from_real_readback(default_clients, monkeypatch):
    first, _, _ = default_clients
    before = await first.get_activation_snapshot()
    original = first._run_query
    calls = 0

    async def lose_update_reply(sql, params):
        nonlocal calls
        result = await original(sql, params)
        if sql.startswith("UPDATE type::record('brain'"):
            calls += 1
            raise ConnectionResetError("synthetic lost response after update")
        return result

    monkeypatch.setattr(first, "_run_query", lose_update_reply)
    receipt = await migrate(first, before)
    assert calls == 1
    assert receipt["confirmed"] is True and receipt["reconciled"] is True
    expected = deepcopy(before["config"])
    expected["activation_strategy"] = "hybrid"
    assert (await first.get_activation_snapshot())["config"] == expected


async def test_null_none_only_drift_rejected_by_server_signature(default_clients):
    first, second, brain_id = default_clients
    before = await first.get_activation_snapshot()
    await second._query(
        "UPDATE type::record('brain', $id) SET config.future_knob.nested = "
        "[1, NONE, NULL, { name: 'synthetic' }]",
        id=brain_id,
    )
    drifted = await second.get_activation_snapshot()
    assert drifted["config"] == before["config"]
    assert drifted["config_sha256"] != before["config_sha256"]
    with pytest.raises(ActivationMigrationError):
        await migrate(first, before)
    assert await first.get_activation_snapshot() == drifted
