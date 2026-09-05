"""Live-DB regression: change_log must actually store the payload it is given.

``change_log`` is SCHEMAFULL and its ``payload`` field was declared
``option<object>`` without ``FLEXIBLE``. SurrealDB rejects a nested object
against such a field outright — "Found field 'payload.content', but no such
field exists for table 'change_log'" — and both writers swallowed the error, so
every neuron, synapse and fiber write produced no change-log row at all and
sync had nothing to replay. Skipped unless SURREALDB_URL points at a running
SurrealDB.
"""

from __future__ import annotations

import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name="change-log-payload-live")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await store.close()
    except Exception:
        pass


class TestChangeLogPayload:
    async def test_add_neuron_writes_a_change_log_row_with_its_payload(self, storage) -> None:  # type: ignore[no-untyped-def]
        """One neuron write must leave one replayable change-log row behind."""
        neuron = Neuron.create(type=NeuronType.CONCEPT, content="change log payload content")
        await storage.add_neuron(neuron)

        rows = await storage._query(
            "SELECT entity_type, entity_id, operation, payload FROM change_log "
            "WHERE brain_id = $bid AND entity_id = $eid",
            bid=storage.current_brain_id,
            eid=neuron.id,
        )

        assert rows, (
            "add_neuron left no change_log row at all; a SCHEMAFULL payload field "
            "without FLEXIBLE rejects the nested payload and both writers swallow it"
        )
        row = rows[0]
        assert row["entity_type"] == "neuron"
        assert row["operation"] == "insert"
        payload = row.get("payload")
        assert payload, f"the change_log row landed but carries no payload: {row}"
        assert isinstance(payload, dict), f"payload should be an object, got {type(payload)}"
        assert payload.get("content") == "change log payload content", (
            f"the payload must survive the round trip intact; got {payload}"
        )
