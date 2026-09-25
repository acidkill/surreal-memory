"""Real SurrealDB 3.2 CAS fence for the operator semantic recovery path."""

from __future__ import annotations

import ipaddress
import json
import os
import uuid
from datetime import timedelta
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation import ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressSession,
    ConsolidationResumeMismatchError,
    recover_legacy_semantic_link_discovery,
    recover_stale_semantic_link_discovery,
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


@pytest.mark.asyncio
async def test_stale_discovery_recovery_requires_verified_pre_reference_source_change(
    store,
) -> None:
    brain_id = store._get_brain_id()
    run_id = "run-stale-source-test"
    strategies = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    completed = sorted(strategies - {"semantic_link"})
    token = await store.capture_semantic_source_token()
    assert json.loads(token)["created_at_readonly"] is True

    # Keep created_at before the run reference time while writing after token
    # capture, matching the stale-token case this operator path is for.
    neuron = Neuron.create(NeuronType.CONCEPT, "stale-source-recovery-fixture")
    neuron = Neuron(
        id=neuron.id,
        type=neuron.type,
        content=neuron.content,
        metadata=neuron.metadata,
        content_hash=neuron.content_hash,
        created_at=utcnow() - timedelta(seconds=1),
    )
    await store.add_neuron(neuron)
    reference_time = utcnow()

    legacy_recovery = {
        "kind": "legacy_semantic_discovery_restart",
        "version": 1,
        "run_id": run_id,
        "options_fingerprint": "options-stale-test",
        "expected_status": "paused",
        "previous_phase": "semantic_link_discovery_neurons",
        "recovered_at": utcnow(),
    }
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": {
            "state_id": "stage-stale-test",
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
        "counters": {},
        "recovery": legacy_recovery,
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
            "options_fingerprint": "options-stale-test",
            "reference_time": reference_time,
            "owner_token": "previous-owner",
            "status": "running",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_discovery_neurons",
            "cursor": "neuron:123",
            "strategy_states": {"prune": {"phase": "completed"}, "semantic_link": nested},
            "counters": {"prune": {"synapses_pruned": 4}, "semantic_link": {}},
            "started_at": reference_time,
            "updated_at": utcnow(),
            "last_error": None,
        }
    )
    args = {
        "run_id": run_id,
        "expected_options_fingerprint": "options-stale-test",
        "expected_status": "running",
        "confirmation_run_id": run_id,
    }
    preview = await recover_stale_semantic_link_discovery(store, **args)
    assert preview["dry_run"] is True
    assert (await store.get_consolidation_progress(brain_id))["updated_at"] == before["updated_at"]

    applied = await recover_stale_semantic_link_discovery(store, **args, dry_run=False)
    assert applied["already_recovered"] is False
    after = await store.get_consolidation_progress(brain_id)
    assert after is not None
    assert after["reference_time"] == before["reference_time"]
    assert after["completed_strategies"] == completed
    assert after["strategy_states"]["prune"] == {"phase": "completed"}
    assert after["counters"]["prune"] == {"synapses_pruned": 4}
    semantic_after = after["strategy_states"]["semantic_link"]
    assert semantic_after["phase"] == "semantic_link_restart_pending"
    assert "pending" not in semantic_after
    assert semantic_after["recovery_history"] == [
        before["strategy_states"]["semantic_link"]["recovery"],
        semantic_after["recovery"],
    ]
    assert semantic_after["recovery"]["source_changed_verified"] is True

    retried = await recover_stale_semantic_link_discovery(store, **args, dry_run=False)
    assert retried["already_recovered"] is True


@pytest.mark.asyncio
async def test_stale_discovery_recovery_refuses_when_change_is_not_proven(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_id = store._get_brain_id()
    run_id = "run-stale-source-unchanged-test"
    strategies = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    completed = sorted(strategies - {"semantic_link"})
    token = await store.capture_semantic_source_token()
    reference_time = utcnow()
    legacy_recovery = {
        "kind": "legacy_semantic_discovery_restart",
        "version": 1,
        "run_id": run_id,
        "options_fingerprint": "options-stale-unchanged-test",
        "expected_status": "failed",
        "previous_phase": "semantic_link_discovery_neurons",
    }
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": {
            "state_id": "stage-stale-unchanged-test",
            "revision": 1,
            "source_token": token,
            "brain_id": brain_id,
            "run_id": run_id,
        },
    }
    nested = {
        "phase": "semantic_link_discovery_neurons",
        "cursor": "neuron:123",
        "pending": [json.dumps([json.dumps(manifest)])],
        "counters": {},
        "recovery": legacy_recovery,
    }
    await store.start_consolidation_progress(
        {
            "brain_id": brain_id,
            "run_id": run_id,
            "schema_version": 12,
            "engine_version": "3.11.0:checkpoint-v1",
            "format_version": 1,
            "requested_strategies": sorted(strategies),
            "completed_strategies": completed,
            "options_fingerprint": "options-stale-unchanged-test",
            "reference_time": reference_time,
            "owner_token": "previous-owner",
            "status": "running",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_discovery_neurons",
            "cursor": "neuron:123",
            "strategy_states": {"semantic_link": nested},
            "counters": {"semantic_link": {}},
            "started_at": reference_time,
            "updated_at": utcnow(),
            "last_error": None,
        }
    )
    with pytest.raises(ConsolidationResumeMismatchError, match=r"change .* not proven"):
        await recover_stale_semantic_link_discovery(
            store,
            run_id=run_id,
            expected_options_fingerprint="options-stale-unchanged-test",
            expected_status="running",
            confirmation_run_id=run_id,
        )

    from surreal_memory.storage.surrealdb.semantic_source_revision import (
        SemanticSourceFenceExpiredError,
    )

    async def expired_source_fence(*args, **kwargs) -> None:
        raise SemanticSourceFenceExpiredError("changefeed expired")

    monkeypatch.setattr(store, "assert_semantic_source_unchanged", expired_source_fence)
    with pytest.raises(ConsolidationResumeMismatchError, match="could not be verified"):
        await recover_stale_semantic_link_discovery(
            store,
            run_id=run_id,
            expected_options_fingerprint="options-stale-unchanged-test",
            expected_status="running",
            confirmation_run_id=run_id,
        )


@pytest.mark.asyncio
async def test_stale_recovery_repeats_from_failed_checkpoint_and_preserves_audit(store) -> None:
    brain_id = store._get_brain_id()
    run_id = "run-stale-source-repeat-test"
    fingerprint = "options-stale-repeat-test"
    strategies = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    completed = sorted(strategies - {"semantic_link"})
    source_token = await store.capture_semantic_source_token()
    neuron = Neuron.create(NeuronType.CONCEPT, "repeat-stale-source-recovery-fixture")
    neuron = Neuron(
        id=neuron.id,
        type=neuron.type,
        content=neuron.content,
        metadata=neuron.metadata,
        content_hash=neuron.content_hash,
        created_at=utcnow() - timedelta(seconds=1),
    )
    await store.add_neuron(neuron)
    reference_time = utcnow()

    legacy_marker = {
        "kind": "legacy_semantic_discovery_restart",
        "version": 1,
        "run_id": run_id,
        "options_fingerprint": fingerprint,
        "expected_status": "paused",
        "previous_phase": "semantic_link_discovery_neurons",
        "recovered_at": utcnow(),
    }
    prior_stale_marker = {
        "kind": "stale_semantic_source_restart",
        "version": 1,
        "run_id": run_id,
        "options_fingerprint": fingerprint,
        "expected_status": "running",
        "previous_phase": "semantic_link_discovery_neurons",
        "manifest_sha256": "a" * 64,
        "source_token_sha256": "b" * 64,
        "prior_recovery_sha256": "c" * 64,
        "source_changed_verified": True,
        "owner_token": "first-recovery-owner",
        "recovered_at": utcnow() - timedelta(minutes=1),
    }
    manifest = {
        "kind": "semantic_link_discovery",
        "version": 3,
        "stage": "neurons",
        "source_state_ref": {
            "state_id": "stage-repeat-test",
            "revision": 1,
            "state_revision": 1,
            "source_token": source_token,
            "brain_id": brain_id,
            "run_id": run_id,
        },
    }
    semantic_state = {
        "phase": "semantic_link_discovery_neurons",
        "cursor": "neuron:123",
        "pending": [json.dumps([json.dumps(manifest)])],
        "counters": {"semantic_synapses_created": 0},
        "recovery": prior_stale_marker,
        "recovery_history": [legacy_marker, prior_stale_marker],
        "updated_at": utcnow(),
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
            "options_fingerprint": fingerprint,
            "reference_time": reference_time,
            "owner_token": "failed-worker-owner",
            "status": "failed",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_discovery_neurons",
            "cursor": "neuron:123",
            "strategy_states": {"semantic_link": semantic_state},
            "counters": {
                "prune": {"synapses_pruned": 3},
                "semantic_link": semantic_state["counters"],
            },
            "started_at": reference_time,
            "updated_at": utcnow(),
            "last_error": "semantic source changed",
        }
    )
    args = {
        "run_id": run_id,
        "expected_options_fingerprint": fingerprint,
        "expected_status": "failed",
        "confirmation_run_id": run_id,
    }

    preview = await recover_stale_semantic_link_discovery(store, **args)
    assert preview["dry_run"] is True
    assert (await store.get_consolidation_progress(brain_id))["updated_at"] == before["updated_at"]

    applied = await recover_stale_semantic_link_discovery(store, **args, dry_run=False)
    assert applied["already_recovered"] is False
    after = await store.get_consolidation_progress(brain_id)
    assert after is not None
    assert after["status"] == "paused"
    assert after["reference_time"] == before["reference_time"]
    assert after["completed_strategies"] == completed
    assert after["counters"]["prune"] == {"synapses_pruned": 3}
    recovered = after["strategy_states"]["semantic_link"]
    assert recovered["phase"] == "semantic_link_restart_pending"
    assert recovered["recovery_history"][:2] == [legacy_marker, prior_stale_marker]
    assert recovered["recovery_history"][-1] == recovered["recovery"]
    assert recovered["recovery"]["source_changed_verified"] is True

    retried = await recover_stale_semantic_link_discovery(store, **args, dry_run=False)
    assert retried["already_recovered"] is True
