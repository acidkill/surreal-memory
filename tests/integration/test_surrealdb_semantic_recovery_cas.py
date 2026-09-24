"""Real SurrealDB 3.2 CAS fence for the operator semantic recovery path."""

from __future__ import annotations

import ipaddress
import json
import os
import uuid
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.engine.consolidation import ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressSession,
    recover_legacy_semantic_link_discovery,
)
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow


def _loopback(url: str | None) -> bool:
    if not url:
        return False
    hostname = urlparse(url).hostname
    try:
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return hostname == "localhost"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _loopback(os.getenv("SURREALDB_URL")),
        reason="requires explicit loopback SurrealDB URL",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=os.environ["SURREALDB_URL"],
        user=os.getenv("SURREALDB_USER", "root"),
        password=os.getenv("SURREALDB_PASS", "root"),
        namespace="smem_recovery_it",
        database="recovery_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-recovery-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_recovery_cas_requires_live_lease_and_exact_snapshot(store) -> None:
    brain_id = store._get_brain_id()
    before = await store.start_consolidation_progress(
        {
            "brain_id": brain_id,
            "run_id": "run-recovery-test",
            "schema_version": 12,
            "engine_version": "3.11.0:checkpoint-v1",
            "format_version": 1,
            "requested_strategies": ["semantic_link"],
            "completed_strategies": [],
            "options_fingerprint": "options-test",
            "reference_time": utcnow(),
            "owner_token": "previous-owner",
            "status": "paused",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_discovery_neurons",
            "cursor": "neuron:123",
            "strategy_states": {"semantic_link": {"phase": "semantic_link_discovery_neurons"}},
            "counters": {"semantic_link": {}},
            "started_at": utcnow(),
            "updated_at": utcnow(),
            "last_error": None,
        }
    )
    parameters = {
        "brain_id": brain_id,
        "expected_run_id": "run-recovery-test",
        "expected_owner_token": "previous-owner",
        "expected_options_fingerprint": "options-test",
        "expected_status": "paused",
        "expected_phase": "semantic_link_discovery_neurons",
        "expected_updated_at": before["updated_at"],
        "expected_reference_time": before["reference_time"],
        "lease_owner_token": "recovery-owner",
        "new_owner_token": "recovery-owner",
        "strategy_states": {
            "semantic_link": {"phase": "semantic_link_restart_pending", "counters": {}}
        },
        "counters": {"semantic_link": {}},
        "updated_at": utcnow(),
    }
    assert await store.compare_and_swap_semantic_link_recovery(**parameters) is None
    assert await store.acquire_consolidation_lease(brain_id, "recovery-owner", lease_seconds=30)
    saved = await store.compare_and_swap_semantic_link_recovery(**parameters)
    assert saved is not None
    assert saved["phase"] == "semantic_link_restart_pending"
    assert saved["owner_token"] == parameters["new_owner_token"]
    assert saved["reference_time"] == before["reference_time"]
    assert await store.compare_and_swap_semantic_link_recovery(**parameters) is None
    assert await store.release_consolidation_lease(brain_id, "recovery-owner")


@pytest.mark.asyncio
@pytest.mark.parametrize("readonly_field_present", [False, True])
async def test_legacy_discovery_recovery_preserves_completed_run_and_reopens(
    store, readonly_field_present: bool
) -> None:
    brain_id = store._get_brain_id()
    run_id = "run-legacy-test"
    reference_time = utcnow()
    strategies = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    completed = sorted(strategies - {"semantic_link"})
    token_fields = {
        "version": 1,
        "brain_id": brain_id,
        "versionstamp": 0,
        "captured_at": utcnow().isoformat(),
    }
    if readonly_field_present:
        token_fields["created_at_readonly"] = None
    token = json.dumps(token_fields, sort_keys=True)
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": {
            "state_id": "stage-legacy-test",
            "revision": 1,
            "state_revision": 1,
            "source_token": token,
            "brain_id": brain_id,
            "run_id": run_id,
        },
    }
    nested = {
        "phase": "semantic_link_discovery_neurons",
        "cursor": "neuron:123",
        "pending": [json.dumps([json.dumps(manifest)])],
        "counters": {"semantic_synapses_created": 0},
    }
    before = await store.start_consolidation_progress(
        {
            "brain_id": brain_id,
            "run_id": run_id,
            "schema_version": 12,
            "engine_version": "3.11.0:checkpoint-v1",
            "format_version": 1,
            "requested_strategies": sorted(strategies),
            "completed_strategies": completed,
            "options_fingerprint": "options-legacy-test",
            "reference_time": reference_time,
            "owner_token": "previous-owner",
            "status": "paused",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_discovery_neurons",
            "cursor": "neuron:123",
            "strategy_states": {"prune": {"phase": "completed"}, "semantic_link": nested},
            "counters": {"prune": {"synapses_pruned": 4}, "semantic_link": nested["counters"]},
            "started_at": reference_time,
            "updated_at": utcnow(),
            "last_error": None,
        }
    )
    args = {
        "run_id": run_id,
        "expected_options_fingerprint": "options-legacy-test",
        "expected_status": "paused",
        "confirmation_run_id": run_id,
    }
    preview = await recover_legacy_semantic_link_discovery(store, **args)
    assert preview["dry_run"] is True
    assert (await store.get_consolidation_progress(brain_id))["phase"] == before["phase"]

    applied = await recover_legacy_semantic_link_discovery(store, **args, dry_run=False)
    assert applied["already_recovered"] is False
    after = await store.get_consolidation_progress(brain_id)
    assert after is not None
    assert after["reference_time"] == before["reference_time"]
    assert after["completed_strategies"] == completed
    assert after["strategy_states"]["prune"] == {"phase": "completed"}
    assert after["counters"]["prune"] == {"synapses_pruned": 4}
    assert after["phase"] == "semantic_link_restart_pending"
    assert "pending" not in after["strategy_states"]["semantic_link"]

    resumed = await ConsolidationProgressSession.open(
        store, sorted(strategies), "options-legacy-test", utcnow()
    )
    assert resumed is not None
    try:
        assert resumed.resumed
        assert resumed.completed_strategies == set(completed)
        assert not resumed.strategy_state("semantic_link").get("pending")
    finally:
        await resumed.close()
