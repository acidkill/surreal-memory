"""Live SurrealDB lifecycle contract, restricted to a loopback test server.

An inherited production SURREALDB_URL must never make this test write to it;
only the explicit SMEM_TEST_SURREALDB_URL opt-in enables it.
The database is unique per test and the namespace and credentials are fixed.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import timedelta
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

TEST_SURREALDB_URL = os.getenv("SMEM_TEST_SURREALDB_URL")
TEST_AUTH = ("root", "root")  # Docker/CI's disposable server, never inherited credentials.


def _is_loopback_test_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_test_url(TEST_SURREALDB_URL),
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL with port",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace="smem_ci",
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="lifecycle-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def test_lifecycle_write_survives_adapter_reads_and_repeat_scan_is_noop(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_time = utcnow()
    neuron = replace(
        Neuron.create(
            type=NeuronType.CONCEPT,
            content="synthetic lifecycle fact",
            metadata={"priority": 0, "lifecycle_state": "active", "source": "integration"},
        ),
        created_at=reference_time - timedelta(days=200),
    )
    await store.add_neuron(neuron)

    await store.update_neuron_lifecycle(neuron.id, "archived")
    rows = await store._query(
        "SELECT lifecycle_state, metadata FROM neuron WHERE brain_id = $brain_id",
        brain_id=store.brain_id,
    )
    assert len(rows) == 1
    assert rows[0]["lifecycle_state"] == "archived"
    assert rows[0]["metadata"] == neuron.metadata  # No metadata shadow write.

    single = await store.get_neuron(neuron.id)
    page = await store.find_neurons_after_id(
        None, limit=10, created_before=reference_time, ephemeral=False
    )
    batch = await store.find_neurons_by_ids([neuron.id])
    assert single is not None
    assert [item.id for item in page] == [neuron.id]
    assert [item.id for item in batch] == [neuron.id]
    for reread in (single, page[0], batch[0]):
        assert reread.metadata == {
            "priority": 0,
            "lifecycle_state": "archived",
            "source": "integration",
        }

    async def unexpected_write(neuron_id: str, lifecycle_state: str) -> None:
        pytest.fail(f"repeat lifecycle scan rewrote {neuron_id} to {lifecycle_state}")

    monkeypatch.setattr(store, "update_neuron_lifecycle", unexpected_write)
    report = ConsolidationReport()
    await ConsolidationEngine(store)._lifecycle(report, reference_time, dry_run=False)
    assert report.extra.get("lifecycle_states_updated", 0) == 0


async def test_generic_update_does_not_persist_stale_lifecycle_mirror(
    store: SurrealDBStorage,
) -> None:
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="synthetic stale lifecycle snapshot",
        metadata={"priority": 17, "lifecycle_state": "warm", "source": "integration"},
    )
    await store.add_neuron(neuron)
    await store.update_neuron_lifecycle(neuron.id, "warm")

    stale_snapshot = await store.get_neuron(neuron.id)
    assert stale_snapshot is not None
    assert stale_snapshot.metadata["lifecycle_state"] == "warm"

    await store.update_neuron_lifecycle(neuron.id, "cool")
    await store.update_neuron(replace(stale_snapshot, content="synthetic edited content"))

    rows = await store._query(
        "SELECT lifecycle_state, metadata, content FROM neuron "
        "WHERE brain_id = $brain_id AND id = type::record('neuron', $neuron_id)",
        brain_id=store.brain_id,
        neuron_id=neuron.id.replace("-", "_"),
    )
    assert len(rows) == 1
    assert rows[0]["lifecycle_state"] == "cool"
    assert rows[0]["metadata"] == {"priority": 17, "source": "integration"}
    assert rows[0]["content"] == "synthetic edited content"

    reread = await store.get_neuron(neuron.id)
    assert reread is not None
    assert reread.metadata == {
        "priority": 17,
        "lifecycle_state": "cool",
        "source": "integration",
    }

    changes = await store._query(
        "SELECT sequence, operation, payload FROM change_log "
        "WHERE brain_id = $brain_id AND entity_type = 'neuron' "
        "AND entity_id = $entity_id ORDER BY sequence DESC LIMIT 1",
        brain_id=store.brain_id,
        entity_id=neuron.id,
    )
    assert len(changes) == 1
    assert changes[0]["operation"] == "update"
    assert changes[0]["payload"]["content"] == "synthetic edited content"
    assert changes[0]["payload"]["metadata"] == {
        "priority": 17,
        "lifecycle_state": "cool",
        "source": "integration",
    }


async def test_missing_or_wrong_brain_row_is_not_reported_as_updated(
    store: SurrealDBStorage,
) -> None:
    with pytest.raises(LookupError, match="not found in brain"):
        await store.update_neuron_lifecycle(str(uuid.uuid4()), "archived")

    neuron = Neuron.create(type=NeuronType.CONCEPT, content="other brain")
    await store.add_neuron(neuron)
    other = Brain.create(name="other-lifecycle-it")
    await store.save_brain(other)
    store.set_brain(other.id)
    with pytest.raises(LookupError, match="not found in brain"):
        await store.update_neuron_lifecycle(neuron.id, "archived")


async def test_database_failure_is_not_reported_as_updated(store: SurrealDBStorage) -> None:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content="unavailable transport")
    await store.add_neuron(neuron)
    await store.close()
    with pytest.raises(RuntimeError, match="SurrealDB not initialized"):
        await store.update_neuron_lifecycle(neuron.id, "archived")


async def test_generic_update_targets_all_digit_string_id_and_is_brain_scoped(
    store: SurrealDBStorage,
) -> None:
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="synthetic numeric string record",
        metadata={"priority": 23, "source": "integration"},
        neuron_id="1234567890",
    )
    await store.add_neuron(neuron)
    await store.update_neuron_lifecycle(neuron.id, "warm")
    stale_snapshot = await store.get_neuron(neuron.id)
    assert stale_snapshot is not None
    assert stale_snapshot.id == "1234567890"
    assert stale_snapshot.metadata["lifecycle_state"] == "warm"

    await store.update_neuron_lifecycle(neuron.id, "cool")
    await store.update_neuron(replace(stale_snapshot, content="synthetic numeric-id edit"))
    rows = await store._query(
        "SELECT id, lifecycle_state, metadata, content FROM neuron "
        "WHERE brain_id = $brain_id AND id = type::record('neuron', $record_id)",
        brain_id=store.brain_id,
        record_id="1234567890",
    )
    assert len(rows) == 1
    assert getattr(rows[0]["id"], "id", None) == "1234567890"
    assert rows[0]["lifecycle_state"] == "cool"
    assert rows[0]["metadata"] == {"priority": 23, "source": "integration"}
    assert rows[0]["content"] == "synthetic numeric-id edit"

    original_brain_id = store.brain_id
    other = Brain.create(name="other-numeric-id-it")
    await store.save_brain(other)
    store.set_brain(other.id)
    assert await store.get_neuron(neuron.id) is None
    with pytest.raises(LookupError, match="not found in brain"):
        await store.update_neuron_lifecycle(neuron.id, "archived")
    with pytest.raises(LookupError, match="not found in brain"):
        await store.update_neuron(replace(stale_snapshot, content="must not cross brains"))

    store.set_brain(original_brain_id)
    verify = await store._query(
        "SELECT lifecycle_state, metadata, content FROM neuron "
        "WHERE brain_id = $brain_id AND id = type::record('neuron', $record_id)",
        brain_id=original_brain_id,
        record_id="1234567890",
    )
    assert len(verify) == 1
    assert verify[0]["lifecycle_state"] == "cool"
    assert verify[0]["metadata"] == {"priority": 23, "source": "integration"}
    assert verify[0]["content"] == "synthetic numeric-id edit"
