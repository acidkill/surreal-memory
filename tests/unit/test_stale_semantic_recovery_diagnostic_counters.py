"""Recovery of an unapplied stale window after earlier committed batches."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressError,
    recover_stale_semantic_link_discovery,
)
from tests.unit.test_stale_semantic_recovery_repeat import (
    BRAIN_ID,
    REFERENCE_TIME,
    RUN_ID,
    StaleRecoveryStorage,
    _arguments,
)


class PartialRecoveryStorage(StaleRecoveryStorage):
    def __init__(self, *, cas_race: bool = False) -> None:
        super().__init__(cas_race=cas_race)
        semantic = self.progress["strategy_states"]["semantic_link"]
        semantic.pop("recovery")
        semantic.pop("recovery_history")
        old = json.loads(json.loads(semantic["pending"][0])[0])
        source_ref = old["source_state_ref"]
        source_ref.update(revision=2121, state_revision=2121)
        token = source_ref["source_token"]
        inner = {
            "kind": "semantic_link_discovery",
            "version": 5,
            "stage": "similarity",
            "cursor": "25",
            "source_state_ref": source_ref,
        }
        wrapper = {
            "kind": "semantic_link_discovery",
            "version": 5,
            "stage": "validation",
            "cursor": "25",
            "discovery": inner,
            "source_validation_ref": {
                "state_id": "validation-old",
                "revision": 1,
                "state_revision": 1,
                "source_token": token,
                "sha256": "a" * 64,
            },
            "validation_cursor": 123,
        }
        counters = {
            "semantic_synapses_created": 7625,
            "semantic_synapses_skipped": 77982,
            "semantic_link_failures": 0,
            "eligible_neurons": 500,
            "existing_pairs": 20,
            "validation_cursor": 123,
        }
        semantic.update(
            phase="semantic_link_discovery_validation",
            cursor="25",
            pending=[json.dumps(wrapper)],
            counters=deepcopy(counters),
            last_error="semantic_link (progress coordination: ConsolidationProgressError)",
        )
        self.progress.update(
            phase=semantic["phase"], cursor=semantic["cursor"], last_error=semantic["last_error"]
        )
        self.progress["counters"]["semantic_link"] = deepcopy(counters)
        self.states: dict[tuple[str, int], dict[str, Any]] = {
            (source_ref["state_id"], 2121): {
                "state_id": source_ref["state_id"],
                "revision": 2121,
                "state_revision": 2121,
                "brain_id": BRAIN_ID,
                "run_id": RUN_ID,
                "reference_time": REFERENCE_TIME.isoformat(),
                "source_token": token,
                "batch_index": 4,
                "coverage_offset": 40,
                "source_complete": True,
                "candidates": ["stale-neuron"],
            }
        }
        self.new_token = json.dumps({**json.loads(token), "versionstamp": 13}, sort_keys=True)

    async def load_semantic_discovery_state(
        self,
        state_id: str,
        revision: int,
        *,
        recovery_owner_token: str | None = None,
        expected_progress_owner_token: str | None = None,
    ) -> dict[str, Any] | None:
        if recovery_owner_token is not None:
            assert recovery_owner_token == self.lease_owner
            assert expected_progress_owner_token == self.progress["owner_token"]
        return deepcopy(self.states.get((state_id, revision)))

    async def save_semantic_discovery_state(
        self, state_id: str, revision: int, payload: dict[str, Any]
    ) -> None:
        self.states[(state_id, revision)] = deepcopy(payload)

    async def capture_semantic_source_token(self) -> str:
        return self.new_token

    async def compare_and_swap_semantic_link_recovery(self, **kwargs: Any) -> dict[str, Any] | None:
        expected_error = kwargs.get("resolved_source_error")
        if expected_error is not None and self.progress.get("last_error") != expected_error:
            return None
        result = await super().compare_and_swap_semantic_link_recovery(**kwargs)
        if result is not None and expected_error is not None:
            self.progress["last_error"] = None
            result["last_error"] = None
        return result


@pytest.mark.asyncio
async def test_stale_validation_rebases_without_losing_committed_batches() -> None:
    storage = PartialRecoveryStorage()
    before = deepcopy(storage.progress)
    old_states = deepcopy(storage.states)
    preview = await recover_stale_semantic_link_discovery(storage, **_arguments())
    assert preview["dry_run"] is True
    assert storage.progress == before
    assert storage.cas_calls == []
    result = await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert result["already_recovered"] is False
    after = storage.progress
    for key in (
        "run_id",
        "reference_time",
        "options_fingerprint",
        "completed_strategies",
        "counters",
    ):
        assert after[key] == before[key]
    assert all(storage.states[key] == value for key, value in old_states.items())
    semantic = after["strategy_states"]["semantic_link"]
    manifest = json.loads(semantic["pending"][0])
    assert manifest["stage"] == "neurons"
    assert manifest["cursor"] is None
    ref = manifest["source_state_ref"]
    fresh = storage.states[(ref["state_id"], ref["revision"])]
    assert fresh["source_token"] == storage.new_token
    assert fresh["coverage_offset"] == 0
    assert fresh["batch_index"] >= 1
    assert not fresh["candidates"]
    assert semantic["recovery"]["prior_manifest"]["stage"] == "validation"
    assert semantic["recovery"]["prior_counters"] == before["counters"]["semantic_link"]
    assert len(storage.cas_calls) == 1
    retry = await recover_stale_semantic_link_discovery(
        storage, **{**_arguments(), "expected_status": "paused"}, dry_run=False
    )
    assert retry["already_recovered"] is True
    assert len(storage.cas_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "counter_mismatch",
        "unknown_error",
        "nested_error",
        "apply_phase",
        "lease_busy",
        "cas_race",
        "wrong_scope",
        "wrong_token",
        "wrong_revision",
        "wrong_cursor",
    ],
)
async def test_partial_recovery_refuses_unsafe_evidence(failure: str) -> None:
    storage = PartialRecoveryStorage(cas_race=failure == "cas_race")
    semantic = storage.progress["strategy_states"]["semantic_link"]
    if failure == "counter_mismatch":
        semantic["counters"]["semantic_synapses_created"] += 1
    elif failure == "unknown_error":
        semantic["last_error"] = "unrelated apply failure"
    elif failure == "nested_error":
        semantic["last_error"] = "unrelated nested failure"
    elif failure in {"wrong_scope", "wrong_token", "wrong_revision", "wrong_cursor"}:
        wrapper = json.loads(semantic["pending"][0])
        if failure == "wrong_scope":
            next(iter(storage.states.values()))["brain_id"] = "other-brain"
        elif failure == "wrong_token":
            wrapper["discovery"]["source_state_ref"]["source_token"] = storage.new_token
        elif failure == "wrong_revision":
            wrapper["discovery"]["source_state_ref"]["state_revision"] = 2120
        else:
            wrapper["cursor"] = "different"
        semantic["pending"] = [json.dumps(wrapper)]
    elif failure == "apply_phase":
        storage.progress["phase"] = semantic["phase"] = "semantic_link_apply"
    elif failure == "lease_busy":
        storage.lease_owner = "another-owner"
    before = deepcopy(storage.progress)
    with pytest.raises(ConsolidationProgressError):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert storage.progress == before
    if failure != "cas_race":
        assert storage.cas_calls == []


@pytest.mark.asyncio
async def test_generation_audit_rejects_tampering() -> None:
    from surreal_memory.engine.consolidation_progress import (
        validate_semantic_source_generation_audit,
    )

    storage = PartialRecoveryStorage()
    await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    marker = storage.progress["strategy_states"]["semantic_link"]["recovery"]
    identity = {
        "brain_id": BRAIN_ID,
        "run_id": RUN_ID,
        "options_fingerprint": _arguments()["expected_options_fingerprint"],
        "reference_time": REFERENCE_TIME,
    }
    assert validate_semantic_source_generation_audit(marker, **identity)
    for field, value in (
        ("generation", 0),
        ("resolved_error", "unrelated failure"),
        ("run_id", "other"),
        ("unexpected", True),
    ):
        altered = {**marker, field: value}
        assert not validate_semantic_source_generation_audit(altered, **identity)
    altered = deepcopy(marker)
    altered["prior_counters"]["semantic_synapses_created"] += 1
    assert not validate_semantic_source_generation_audit(altered, **identity)


@pytest.mark.asyncio
async def test_recovered_bootstrap_scans_live_population_and_skips_live_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from dataclasses import replace
    from datetime import timedelta

    from surreal_memory.engine import semantic_discovery as discovery
    from tests.unit.test_semantic_batch_cursor_review import (
        _config,
        _fixture,
        _FrozenStorage,
        _unused_checkpoint,
    )

    recovery = PartialRecoveryStorage()
    await recover_stale_semantic_link_discovery(recovery, **_arguments(), dry_run=False)
    manifest = json.loads(recovery.progress["strategy_states"]["semantic_link"]["pending"][0])
    neurons, edges = _fixture()
    neurons = [replace(item, created_at=REFERENCE_TIME - timedelta(days=1)) for item in neurons]
    deleted_id = neurons.pop().id
    edges = [replace(item, created_at=REFERENCE_TIME + timedelta(seconds=1)) for item in edges]
    storage = _FrozenStorage(neurons, edges)
    storage.current_brain_id = BRAIN_ID
    storage.states = deepcopy(recovery.states)
    lookups: list[Any] = []
    original_lookup = storage.find_existing_synapse_pairs

    async def unchanged(token: str, *, created_before: Any = None) -> None:
        assert token == recovery.new_token
        assert created_before == REFERENCE_TIME

    async def lookup(pairs: Any, *, created_before: Any = None) -> Any:
        lookups.append(created_before)
        return await original_lookup(pairs, created_before=created_before)

    monkeypatch.setattr(storage, "assert_semantic_source_unchanged", unchanged)
    monkeypatch.setattr(storage, "find_existing_synapse_pairs", lookup)
    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setattr(discovery, "_effective_embedding", lambda _config: (True, "", ""))
    result = await discovery.discover_semantic_synapses(
        storage,
        _config(100),
        checkpoint=_unused_checkpoint,
        resume_state=manifest,
        run_id=RUN_ID,
        reference_time=REFERENCE_TIME,
    )
    assert lookups and all(value is None for value in lookups)
    assert all(deleted_id not in (edge.source_id, edge.target_id) for edge in result.synapses)
    existing = frozenset((edges[0].source_id, edges[0].target_id))
    assert all(frozenset((edge.source_id, edge.target_id)) != existing for edge in result.synapses)


@pytest.mark.asyncio
async def test_recovery_continues_restored_validation_past_first_pending_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.storage.surrealdb.semantic_source_revision import SemanticSourceChangedError

    storage = PartialRecoveryStorage()
    original = deepcopy(storage.progress)
    visited: list[tuple[int, str]] = []

    target = "fixed-target"

    class Session:
        watermark = 100
        target_token = target

        async def verify(self, *, budget_check: Any) -> str:
            visited.append((self.watermark, self.target_token))
            assert await budget_check()
            if len(visited) == 1:
                self.watermark = 200
                return "pending"
            raise SemanticSourceChangedError("semantic source changed: deleted record")

    class Validation:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.session = Session()
            assert kwargs["reference"]["state_id"] == "validation-old"
            assert kwargs["expected_progress_owner_token"] == original["owner_token"]

        async def restore(self) -> None:
            pass

    monkeypatch.setattr(
        storage, "_create_semantic_source_fence_session", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_validation.SemanticSourceValidation", Validation
    )
    result = await recover_stale_semantic_link_discovery(storage, **_arguments())
    assert result["dry_run"] is True
    assert visited == [(100, "fixed-target"), (200, "fixed-target")]
    assert storage.source_checks == []
    assert storage.progress == original
    assert storage.cas_calls == []


@pytest.mark.asyncio
async def test_unconfirmed_generation_save_does_not_change_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = PartialRecoveryStorage()
    original = deepcopy(storage.progress)

    async def unconfirmed(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("synthetic save response lost")

    monkeypatch.setattr(storage, "save_semantic_discovery_state", unconfirmed)
    with pytest.raises(RuntimeError, match="save response lost"):
        await recover_stale_semantic_link_discovery(storage, **_arguments(), dry_run=False)
    assert storage.progress == original
    assert storage.cas_calls == []
    assert storage.lease_owner is None
