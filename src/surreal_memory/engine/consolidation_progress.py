"""Durable, brain-scoped state for resumable consolidation runs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from surreal_memory import __version__
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow

logger = logging.getLogger(__name__)

PROGRESS_FORMAT_VERSION = 1
CONSOLIDATION_ENGINE_VERSION = f"{__version__}:checkpoint-v1"
LEASE_SECONDS = 180
INCOMPLETE_STATUSES = {"queued", "running", "paused", "failed"}


class ConsolidationProgressError(RuntimeError):
    """Base error for durable consolidation coordination."""


class ConsolidationPausedError(ConsolidationProgressError):
    """The strategy reached its budget at a committed work-unit boundary."""


class ConsolidationLeaseBusyError(ConsolidationProgressError):
    """Another worker currently owns this brain's consolidation lease."""


class ConsolidationResumeMismatchError(ConsolidationProgressError):
    """An unfinished run cannot safely resume with the requested parameters."""


class ConsolidationLeaseLostError(ConsolidationProgressError):
    """The worker lost its lease and must stop before starting another work unit."""


def options_fingerprint(options: Mapping[str, Any]) -> str:
    """Return a stable digest for all behavior-affecting run parameters."""
    encoded = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _parse_reference_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_naive_utc(value)
    if isinstance(value, str):
        return ensure_naive_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise ConsolidationResumeMismatchError("unfinished consolidation has no valid reference_time")


_LEGACY_DISCOVERY_PHASES = {
    "semantic_link_discovery_neurons",
    "semantic_link_discovery_selection",
    "semantic_link_discovery_synapses",
    "semantic_link_discovery_similarity",
}
_LEGACY_PENDING_MAX_BYTES = 512 * 1024


def _decode_legacy_discovery_pending(pending: Any) -> tuple[dict[str, Any], str]:
    """Decode only the bounded single-item legacy semantic discovery envelope."""
    if not isinstance(pending, (list, tuple)) or len(pending) != 1:
        raise ConsolidationResumeMismatchError(
            "legacy semantic discovery checkpoint is missing or ambiguous"
        )
    original = pending[0]
    value: Any = original
    for _ in range(5):
        if isinstance(value, Mapping):
            manifest = dict(value)
            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            if len(canonical.encode("utf-8")) > _LEGACY_PENDING_MAX_BYTES:
                raise ConsolidationResumeMismatchError(
                    "legacy semantic discovery checkpoint exceeds the recovery limit"
                )
            return manifest, canonical
        if isinstance(value, str):
            if len(value.encode("utf-8")) > _LEGACY_PENDING_MAX_BYTES:
                raise ConsolidationResumeMismatchError(
                    "legacy semantic discovery checkpoint exceeds the recovery limit"
                )
            try:
                value = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ConsolidationResumeMismatchError(
                    "legacy semantic discovery checkpoint is malformed"
                ) from exc
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        break
    raise ConsolidationResumeMismatchError(
        "legacy semantic discovery checkpoint has an unsupported envelope"
    )


def _legacy_invalid_source_token(manifest: Mapping[str, Any], brain_id: str, run_id: str) -> str:
    """Return the proven-invalid legacy v1 token or refuse recovery."""
    candidates: list[str] = []
    for key in ("source_token",):
        value = manifest.get(key)
        if isinstance(value, str):
            candidates.append(value)
    source_ref = manifest.get("source_state_ref")
    if source_ref is not None:
        if not isinstance(source_ref, Mapping):
            raise ConsolidationResumeMismatchError(
                "legacy semantic discovery source reference is malformed"
            )
        state_id = source_ref.get("state_id")
        revision = source_ref.get("revision", source_ref.get("state_revision"))
        state_revision = source_ref.get("state_revision", revision)
        if (
            not isinstance(state_id, str)
            or not state_id
            or type(revision) is not int
            or revision < 1
            or type(state_revision) is not int
            or state_revision != revision
            or source_ref.get("brain_id", brain_id) != brain_id
            or source_ref.get("run_id", run_id) != run_id
        ):
            raise ConsolidationResumeMismatchError(
                "legacy semantic discovery source reference is incomplete"
            )
        value = source_ref.get("source_token")
        if not isinstance(value, str) or not value:
            raise ConsolidationResumeMismatchError(
                "legacy semantic discovery source reference has no token"
            )
        candidates.append(value)
    source_rebuild = manifest.get("source_rebuild")
    if source_rebuild is not None:
        if not isinstance(source_rebuild, Mapping):
            raise ConsolidationResumeMismatchError(
                "legacy semantic discovery inline source state is malformed"
            )
        state_id = source_rebuild.get("state_id")
        state_revision = source_rebuild.get("state_revision")
        if (
            not isinstance(state_id, str)
            or not state_id
            or type(state_revision) is not int
            or state_revision < 1
            or source_rebuild.get("brain_id", brain_id) != brain_id
            or source_rebuild.get("run_id", run_id) != run_id
        ):
            raise ConsolidationResumeMismatchError(
                "legacy semantic discovery inline source reference is incomplete"
            )
        value = source_rebuild.get("source_token")
        if not isinstance(value, str) or not value:
            raise ConsolidationResumeMismatchError(
                "legacy semantic discovery inline source state has no token"
            )
        candidates.append(value)
        legacy = source_rebuild.get("legacy_checkpoint")
        if isinstance(legacy, Mapping) and isinstance(legacy.get("source_token"), str):
            candidates.append(str(legacy["source_token"]))
    if not candidates or any(token != candidates[0] for token in candidates):
        raise ConsolidationResumeMismatchError(
            "legacy semantic discovery source token evidence is missing or inconsistent"
        )

    token = candidates[0]
    if len(token.encode("utf-8")) > 4096:
        raise ConsolidationResumeMismatchError("legacy semantic source token exceeds the limit")
    try:
        decoded = json.loads(token)
        captured_at = decoded.get("captured_at") if isinstance(decoded, Mapping) else None
        datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConsolidationResumeMismatchError("legacy semantic source token is malformed") from exc
    if (
        not isinstance(decoded, Mapping)
        or type(decoded.get("version")) is not int
        or decoded.get("version") != 1
        or decoded.get("brain_id") != brain_id
        or type(decoded.get("versionstamp")) is not int
        or decoded["versionstamp"] < 0
        or not isinstance(captured_at, str)
        or decoded.get("created_at_readonly") is not None
    ):
        raise ConsolidationResumeMismatchError(
            "semantic source token is not the invalid legacy v1 token eligible for recovery"
        )
    return token


async def recover_legacy_semantic_link_discovery(
    storage: Any,
    *,
    run_id: str,
    expected_options_fingerprint: str,
    expected_status: str,
    confirmation_run_id: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Explicitly abandon only a proven-unapplied legacy semantic discovery checkpoint."""
    if not isinstance(dry_run, bool):
        raise ConsolidationResumeMismatchError("dry_run must be an explicit boolean")
    if not run_id or confirmation_run_id != run_id:
        raise ConsolidationResumeMismatchError("typed run-id confirmation does not match")
    if not expected_options_fingerprint:
        raise ConsolidationResumeMismatchError("expected options fingerprint is required")
    if expected_status not in {"paused", "failed"}:
        raise ConsolidationResumeMismatchError("only paused or failed runs can be recovered")
    brain_id = str(getattr(storage, "current_brain_id", "") or "")
    required_methods = (
        "get_consolidation_progress",
        "acquire_consolidation_lease",
        "renew_consolidation_lease",
        "release_consolidation_lease",
        "compare_and_swap_semantic_link_recovery",
    )
    if not brain_id or not supports_storage_methods(storage, required_methods):
        raise ConsolidationResumeMismatchError(
            "storage does not support fenced semantic discovery recovery"
        )

    owner_token = str(uuid4())
    if not await storage.acquire_consolidation_lease(
        brain_id, owner_token, lease_seconds=LEASE_SECONDS
    ):
        raise ConsolidationLeaseBusyError(
            f"brain {brain_id!r} already has an active consolidation worker"
        )

    try:
        state = await storage.get_consolidation_progress(brain_id)
        if not isinstance(state, Mapping) or state.get("brain_id") != brain_id:
            raise ConsolidationResumeMismatchError("durable run evidence is missing")
        if (
            state.get("run_id") != run_id
            or state.get("options_fingerprint") != expected_options_fingerprint
            or state.get("status") != expected_status
            or state.get("current_strategy") != "semantic_link"
            or not isinstance(state.get("owner_token"), str)
            or not state.get("owner_token")
            or state.get("updated_at") is None
        ):
            raise ConsolidationResumeMismatchError(
                "run, status, owner, or options changed; recovery refused"
            )

        from surreal_memory.engine.consolidation import ConsolidationStrategy

        all_strategies = {
            strategy.value
            for strategy in ConsolidationStrategy
            if strategy is not ConsolidationStrategy.ALL
        }
        requested = state.get("requested_strategies")
        completed = state.get("completed_strategies")
        if (
            not isinstance(requested, (list, tuple))
            or any(not isinstance(item, str) for item in requested)
            or len(requested) != len(all_strategies)
            or set(requested) != all_strategies
            or not isinstance(completed, (list, tuple))
            or any(not isinstance(item, str) for item in completed)
            or len(completed) != len(all_strategies) - 1
            or set(completed) != all_strategies - {"semantic_link"}
        ):
            raise ConsolidationResumeMismatchError(
                "run is not the expected all-strategies run with exactly 19 completed"
            )

        reference_time = _parse_reference_time(state.get("reference_time"))
        phase = state.get("phase")
        states = state.get("strategy_states")
        if not isinstance(states, Mapping):
            raise ConsolidationResumeMismatchError("strategy progress evidence is missing")
        semantic_state = states.get("semantic_link")
        if not isinstance(semantic_state, Mapping):
            raise ConsolidationResumeMismatchError("semantic_link progress evidence is missing")
        top_counters = state.get("counters")
        nested_counters = semantic_state.get("counters")
        aggregate_semantic_counters = (
            top_counters.get("semantic_link") if isinstance(top_counters, Mapping) else None
        )
        recovery_marker = semantic_state.get("recovery")
        if recovery_marker is not None:
            if (
                isinstance(recovery_marker, Mapping)
                and recovery_marker.get("kind") == "legacy_semantic_discovery_restart"
                and recovery_marker.get("run_id") == run_id
                and recovery_marker.get("options_fingerprint") == expected_options_fingerprint
                and recovery_marker.get("expected_status") == expected_status
                and phase == "semantic_link_restart_pending"
                and semantic_state.get("phase") == "semantic_link_restart_pending"
                and state.get("cursor") is None
                and semantic_state.get("pending") in (None, [])
                and semantic_state.get("cursor") is None
                and isinstance(top_counters, Mapping)
                and isinstance(nested_counters, Mapping)
                and isinstance(aggregate_semantic_counters, Mapping)
                and dict(nested_counters) == {}
                and dict(aggregate_semantic_counters) == {}
            ):
                return {
                    "run_id": run_id,
                    "reference_time": reference_time,
                    "already_recovered": True,
                    "dry_run": dry_run,
                    "previous_phase": recovery_marker.get("previous_phase"),
                }
            raise ConsolidationResumeMismatchError(
                "recovery marker exists but the run has progressed; refusing another reset"
            )

        nested_phase = semantic_state.get("phase")
        if (
            phase not in _LEGACY_DISCOVERY_PHASES
            or nested_phase != phase
            or state.get("cursor") != semantic_state.get("cursor")
        ):
            raise ConsolidationResumeMismatchError(
                "run is not at a durable semantic_link discovery-only phase"
            )
        if (
            not isinstance(top_counters, Mapping)
            or not isinstance(nested_counters, Mapping)
            or not isinstance(aggregate_semantic_counters, Mapping)
            or dict(nested_counters) != dict(aggregate_semantic_counters)
        ):
            raise ConsolidationResumeMismatchError(
                "durable semantic_link counter evidence is missing or inconsistent"
            )
        for counters in (nested_counters, aggregate_semantic_counters):
            for key in (
                "semantic_synapses_created",
                "semantic_synapses_skipped",
                "semantic_link_failures",
            ):
                value = counters.get(key, 0)
                if type(value) not in {int, float} or value != 0:
                    raise ConsolidationResumeMismatchError(
                        "semantic_link apply counters show possible graph effects"
                    )

        manifest, canonical_manifest = _decode_legacy_discovery_pending(
            semantic_state.get("pending")
        )
        if manifest.get("kind") != "semantic_link_discovery" or manifest.get("version") not in (
            1,
            2,
            3,
        ):
            raise ConsolidationResumeMismatchError(
                "semantic_link pending state is not a supported discovery manifest"
            )
        manifest_version = manifest.get("version")
        stage = manifest.get("stage")
        phase_stage = str(phase).removeprefix("semantic_link_discovery_")
        if (
            stage not in {"neurons", "selection", "synapses", "similarity"}
            or stage != phase_stage
            or (manifest_version == 2 and stage != "similarity")
            or (manifest_version == 3 and stage not in {"neurons", "synapses", "similarity"})
            or (
                manifest_version == 3
                and manifest.get("source_state_ref") is None
                and manifest.get("source_rebuild") is None
            )
        ):
            raise ConsolidationResumeMismatchError(
                "discovery manifest stage does not match its durable phase"
            )
        source_token = _legacy_invalid_source_token(manifest, brain_id, run_id)

        if dry_run:
            return {
                "run_id": run_id,
                "reference_time": reference_time,
                "already_recovered": False,
                "dry_run": True,
                "previous_phase": str(phase),
            }

        old_owner_token = str(state["owner_token"])
        updated_at = utcnow()
        next_states = dict(states)
        next_semantic_state = dict(semantic_state)
        next_semantic_state["phase"] = "semantic_link_restart_pending"
        next_semantic_state["cursor"] = None
        next_semantic_state.pop("pending", None)
        next_semantic_state["counters"] = {}
        next_semantic_state["updated_at"] = updated_at
        next_semantic_state["recovery"] = {
            "kind": "legacy_semantic_discovery_restart",
            "version": 1,
            "run_id": run_id,
            "options_fingerprint": expected_options_fingerprint,
            "expected_status": expected_status,
            "previous_phase": phase,
            "manifest_sha256": sha256(canonical_manifest.encode("utf-8")).hexdigest(),
            "source_token_sha256": sha256(source_token.encode("utf-8")).hexdigest(),
            "recovered_at": updated_at,
        }
        next_states["semantic_link"] = next_semantic_state
        next_counters = dict(top_counters)
        next_counters["semantic_link"] = {}

        if not await storage.renew_consolidation_lease(
            brain_id, owner_token, lease_seconds=LEASE_SECONDS
        ):
            raise ConsolidationLeaseBusyError("brain lease was lost before recovery commit")
        saved = await storage.compare_and_swap_semantic_link_recovery(
            brain_id=brain_id,
            expected_run_id=run_id,
            expected_owner_token=old_owner_token,
            expected_options_fingerprint=expected_options_fingerprint,
            expected_status=expected_status,
            expected_phase=str(phase),
            expected_updated_at=state["updated_at"],
            expected_reference_time=state["reference_time"],
            lease_owner_token=owner_token,
            new_owner_token=owner_token,
            strategy_states=next_states,
            counters=next_counters,
            updated_at=updated_at,
        )
        if saved is None:
            raise ConsolidationResumeMismatchError(
                "durable run changed during recovery; conditional update was not applied"
            )
        return {
            "run_id": run_id,
            "reference_time": reference_time,
            "already_recovered": False,
            "dry_run": False,
            "previous_phase": str(phase),
        }
    finally:
        try:
            await storage.release_consolidation_lease(brain_id, owner_token)
        except Exception:
            logger.warning("Could not release semantic recovery lease for brain %r", brain_id)


async def recover_stale_semantic_link_discovery(
    storage: Any,
    *,
    run_id: str,
    expected_options_fingerprint: str,
    expected_status: str,
    confirmation_run_id: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Explicitly restart discovery only after proving its source token stale."""
    if not isinstance(dry_run, bool):
        raise ConsolidationResumeMismatchError("dry_run must be an explicit boolean")
    if not run_id or confirmation_run_id != run_id:
        raise ConsolidationResumeMismatchError("typed run-id confirmation does not match")
    if not expected_options_fingerprint:
        raise ConsolidationResumeMismatchError("expected options fingerprint is required")
    if expected_status not in {"running", "failed"}:
        raise ConsolidationResumeMismatchError(
            "stale-source recovery requires a running or failed run"
        )

    brain_id = str(getattr(storage, "current_brain_id", "") or "")
    required_methods = (
        "get_consolidation_progress",
        "acquire_consolidation_lease",
        "renew_consolidation_lease",
        "release_consolidation_lease",
        "compare_and_swap_semantic_link_recovery",
        "assert_semantic_source_unchanged",
    )
    if not brain_id or not supports_storage_methods(storage, required_methods):
        raise ConsolidationResumeMismatchError(
            "storage does not support fenced stale-source semantic recovery"
        )

    owner_token = str(uuid4())
    if not await storage.acquire_consolidation_lease(
        brain_id, owner_token, lease_seconds=LEASE_SECONDS
    ):
        raise ConsolidationLeaseBusyError(
            f"brain {brain_id!r} already has an active consolidation worker"
        )

    try:
        state = await storage.get_consolidation_progress(brain_id)
        if not isinstance(state, Mapping) or state.get("brain_id") != brain_id:
            raise ConsolidationResumeMismatchError("durable run evidence is missing")
        if (
            state.get("run_id") != run_id
            or state.get("options_fingerprint") != expected_options_fingerprint
            or not (
                state.get("status") == expected_status
                or (
                    expected_status == "failed"
                    and state.get("status") == "paused"
                    and state.get("phase") == "semantic_link_restart_pending"
                )
            )
            or state.get("current_strategy") != "semantic_link"
            or not isinstance(state.get("owner_token"), str)
            or not state.get("owner_token")
            or state.get("updated_at") is None
        ):
            raise ConsolidationResumeMismatchError(
                "run, status, owner, or options changed; recovery refused"
            )

        from surreal_memory.engine.consolidation import ConsolidationStrategy

        all_strategies = {
            strategy.value
            for strategy in ConsolidationStrategy
            if strategy is not ConsolidationStrategy.ALL
        }
        requested = state.get("requested_strategies")
        completed = state.get("completed_strategies")
        if (
            not isinstance(requested, (list, tuple))
            or any(not isinstance(item, str) for item in requested)
            or len(requested) != len(all_strategies)
            or set(requested) != all_strategies
            or not isinstance(completed, (list, tuple))
            or any(not isinstance(item, str) for item in completed)
            or len(completed) != len(all_strategies) - 1
            or set(completed) != all_strategies - {"semantic_link"}
        ):
            raise ConsolidationResumeMismatchError(
                "run is not the expected all-strategies run with exactly 19 completed"
            )

        reference_time = _parse_reference_time(state.get("reference_time"))
        phase = state.get("phase")
        states = state.get("strategy_states")
        if not isinstance(states, Mapping):
            raise ConsolidationResumeMismatchError("strategy progress evidence is missing")
        semantic_state = states.get("semantic_link")
        if not isinstance(semantic_state, Mapping):
            raise ConsolidationResumeMismatchError("semantic_link progress evidence is missing")
        nested_phase = semantic_state.get("phase")
        if (
            (phase not in _LEGACY_DISCOVERY_PHASES and phase != "semantic_link_restart_pending")
            or nested_phase != phase
            or state.get("cursor") != semantic_state.get("cursor")
        ):
            raise ConsolidationResumeMismatchError(
                "run is not at an unapplied semantic_link discovery phase"
            )

        top_counters = state.get("counters")
        nested_counters = semantic_state.get("counters")
        aggregate_semantic_counters = (
            top_counters.get("semantic_link") if isinstance(top_counters, Mapping) else None
        )
        if (
            not isinstance(top_counters, Mapping)
            or not isinstance(nested_counters, Mapping)
            or not isinstance(aggregate_semantic_counters, Mapping)
            or dict(nested_counters) != dict(aggregate_semantic_counters)
            or any(
                type(value) not in {int, float} or value != 0 for value in nested_counters.values()
            )
        ):
            raise ConsolidationResumeMismatchError(
                "durable semantic_link graph-effect counters are missing or nonzero"
            )

        recovery_history_raw = semantic_state.get("recovery_history", [])
        if not isinstance(recovery_history_raw, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in recovery_history_raw
        ):
            raise ConsolidationResumeMismatchError("semantic recovery history is malformed")

        # Re-running an already committed recovery is safe only when its exact
        # marker and cleared discovery state are still the live checkpoint.
        if phase == "semantic_link_restart_pending":
            latest_recovery = semantic_state.get("recovery")
            history = list(recovery_history_raw)
            prior = history[-2] if len(history) >= 2 else None
            if (
                isinstance(latest_recovery, Mapping)
                and latest_recovery.get("kind") == "stale_semantic_source_restart"
                and latest_recovery.get("run_id") == run_id
                and latest_recovery.get("options_fingerprint") == expected_options_fingerprint
                and latest_recovery.get("expected_status") == expected_status
                and latest_recovery.get("previous_phase") in _LEGACY_DISCOVERY_PHASES
                and latest_recovery.get("source_changed_verified") is True
                and latest_recovery.get("owner_token") == state.get("owner_token")
                and latest_recovery.get("recovered_at") == state.get("updated_at")
                and latest_recovery.get("recovered_at") == semantic_state.get("updated_at")
                and isinstance(prior, Mapping)
                and prior.get("kind")
                in {"legacy_semantic_discovery_restart", "stale_semantic_source_restart"}
                and (
                    prior.get("kind") != "stale_semantic_source_restart"
                    or prior.get("source_changed_verified") is True
                )
                and prior.get("run_id") == run_id
                and prior.get("options_fingerprint") == expected_options_fingerprint
                and state.get("cursor") is None
                and semantic_state.get("cursor") is None
                and semantic_state.get("pending") in (None, [])
                and all(value == 0 for value in nested_counters.values())
                and dict(history[-1]) == dict(latest_recovery)
            ):
                return {
                    "run_id": run_id,
                    "reference_time": reference_time,
                    "already_recovered": True,
                    "dry_run": dry_run,
                    "previous_phase": latest_recovery.get("previous_phase"),
                }
            raise ConsolidationResumeMismatchError(
                "restart phase lacks the exact stale-source recovery audit; refusing retry"
            )

        if phase not in _LEGACY_DISCOVERY_PHASES:
            raise ConsolidationResumeMismatchError(
                "run is not at an unapplied semantic_link discovery phase"
            )

        previous_recovery = semantic_state.get("recovery")
        if not isinstance(previous_recovery, Mapping):
            raise ConsolidationResumeMismatchError(
                "the prior semantic recovery audit marker is missing or unrecognized"
            )
        previous_kind = previous_recovery.get("kind")
        if previous_kind == "legacy_semantic_discovery_restart":
            if (
                previous_recovery.get("run_id") != run_id
                or previous_recovery.get("options_fingerprint") != expected_options_fingerprint
                or previous_recovery.get("expected_status") not in {"paused", "failed"}
            ):
                raise ConsolidationResumeMismatchError(
                    "the prior legacy recovery audit marker is missing or unrecognized"
                )
        elif previous_kind == "stale_semantic_source_restart":
            history = list(recovery_history_raw)
            legacy_marker = history[0] if history else None
            prior_marker = history[-1] if history else None
            if (
                not isinstance(legacy_marker, Mapping)
                or legacy_marker.get("kind") != "legacy_semantic_discovery_restart"
                or legacy_marker.get("run_id") != run_id
                or legacy_marker.get("options_fingerprint") != expected_options_fingerprint
                or not isinstance(prior_marker, Mapping)
                or dict(prior_marker) != dict(previous_recovery)
                or previous_recovery.get("run_id") != run_id
                or previous_recovery.get("options_fingerprint") != expected_options_fingerprint
                or previous_recovery.get("expected_status") not in {"running", "failed"}
                or previous_recovery.get("previous_phase") not in _LEGACY_DISCOVERY_PHASES
                or previous_recovery.get("source_changed_verified") is not True
                or not previous_recovery.get("manifest_sha256")
                or not previous_recovery.get("source_token_sha256")
                or not previous_recovery.get("prior_recovery_sha256")
                or not isinstance(previous_recovery.get("owner_token"), str)
                or not previous_recovery.get("owner_token")
                or previous_recovery.get("recovered_at") is None
            ):
                raise ConsolidationResumeMismatchError(
                    "the prior stale-source recovery audit marker is missing or unrecognized"
                )
        else:
            raise ConsolidationResumeMismatchError(
                "the prior semantic recovery audit marker is missing or unrecognized"
            )
        manifest, canonical_manifest = _decode_legacy_discovery_pending(
            semantic_state.get("pending")
        )
        if manifest.get("kind") != "semantic_link_discovery" or manifest.get("version") != 3:
            raise ConsolidationResumeMismatchError(
                "stale-source recovery requires a version 3 discovery manifest"
            )
        stage = manifest.get("stage")
        if stage not in {"neurons", "synapses", "similarity"} or stage != str(phase).removeprefix(
            "semantic_link_discovery_"
        ):
            raise ConsolidationResumeMismatchError(
                "discovery manifest stage does not match its durable phase"
            )

        # Validate every stored source reference and require one unambiguous,
        # readonly-attested v1 token before asking storage to inspect its feed.
        token_candidates: list[Any] = []
        source_ref = manifest.get("source_state_ref")
        if source_ref is not None:
            if not isinstance(source_ref, Mapping):
                raise ConsolidationResumeMismatchError("semantic source reference is malformed")
            revision = source_ref.get("revision", source_ref.get("state_revision"))
            if (
                not isinstance(source_ref.get("state_id"), str)
                or not source_ref.get("state_id")
                or type(revision) is not int
                or revision < 1
                or source_ref.get("state_revision", revision) != revision
                or source_ref.get("brain_id", brain_id) != brain_id
                or source_ref.get("run_id", run_id) != run_id
            ):
                raise ConsolidationResumeMismatchError("semantic source reference is incomplete")
            token_candidates.append(source_ref.get("source_token"))
        source_rebuild = manifest.get("source_rebuild")
        if source_rebuild is not None:
            if not isinstance(source_rebuild, Mapping):
                raise ConsolidationResumeMismatchError("semantic source rebuild is malformed")
            revision = source_rebuild.get("state_revision")
            if (
                not isinstance(source_rebuild.get("state_id"), str)
                or not source_rebuild.get("state_id")
                or type(revision) is not int
                or revision < 1
                or source_rebuild.get("brain_id", brain_id) != brain_id
                or source_rebuild.get("run_id", run_id) != run_id
            ):
                raise ConsolidationResumeMismatchError("semantic source rebuild is incomplete")
            token_candidates.append(source_rebuild.get("source_token"))
        top_token = manifest.get("source_token")
        if top_token is not None:
            token_candidates.append(top_token)
        if (
            not token_candidates
            or any(not isinstance(token, str) or not token for token in token_candidates)
            or any(token != token_candidates[0] for token in token_candidates)
            or len(token_candidates[0].encode("utf-8")) > 4096
        ):
            raise ConsolidationResumeMismatchError(
                "semantic source token evidence is missing or inconsistent"
            )
        source_token_candidate = token_candidates[0]
        if not isinstance(source_token_candidate, str):
            raise ConsolidationResumeMismatchError("semantic source token is malformed")
        source_token = source_token_candidate
        try:
            decoded_token = json.loads(source_token)
            if not isinstance(decoded_token, Mapping):
                raise ConsolidationResumeMismatchError("semantic source token is malformed")
            captured_at = decoded_token.get("captured_at")
            datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConsolidationResumeMismatchError("semantic source token is malformed") from exc
        versionstamp = decoded_token.get("versionstamp")
        if (
            not isinstance(decoded_token, Mapping)
            or type(decoded_token.get("version")) is not int
            or decoded_token.get("version") != 1
            or decoded_token.get("brain_id") != brain_id
            or type(versionstamp) is not int
            or versionstamp < 0
            or not isinstance(captured_at, str)
            or decoded_token.get("created_at_readonly") is not True
        ):
            raise ConsolidationResumeMismatchError(
                "source token is not a valid readonly-attested v1 token"
            )

        from surreal_memory.storage.surrealdb.semantic_source_revision import (
            SemanticSourceChangedError,
            SemanticSourceFenceError,
        )

        try:
            await storage.assert_semantic_source_unchanged(
                source_token, created_before=reference_time
            )
        except SemanticSourceChangedError:
            pass
        except SemanticSourceFenceError as exc:
            raise ConsolidationResumeMismatchError(
                "semantic source change could not be verified; recovery refused"
            ) from exc
        else:
            raise ConsolidationResumeMismatchError(
                "semantic source change before reference_time was not proven"
            )

        if dry_run:
            return {
                "run_id": run_id,
                "reference_time": reference_time,
                "already_recovered": False,
                "dry_run": True,
                "previous_phase": str(phase),
            }

        def audit_json_default(value: Any) -> str:
            if isinstance(value, datetime):
                return value.isoformat()
            raise TypeError(f"unsupported semantic recovery audit value: {type(value).__name__}")

        updated_at = utcnow()
        prior_marker = dict(previous_recovery)
        history = [dict(item) for item in recovery_history_raw]
        if not history or history[-1] != prior_marker:
            history.append(prior_marker)
        recovery_marker = {
            "kind": "stale_semantic_source_restart",
            "version": 1,
            "run_id": run_id,
            "options_fingerprint": expected_options_fingerprint,
            "expected_status": expected_status,
            "previous_phase": str(phase),
            "manifest_sha256": sha256(canonical_manifest.encode("utf-8")).hexdigest(),
            "source_token_sha256": sha256(source_token.encode("utf-8")).hexdigest(),
            "prior_recovery_sha256": sha256(
                json.dumps(
                    prior_marker,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=audit_json_default,
                ).encode("utf-8")
            ).hexdigest(),
            "source_changed_verified": True,
            "owner_token": owner_token,
            "recovered_at": updated_at,
        }
        history.append(recovery_marker)

        old_owner_token = str(state["owner_token"])
        next_states = dict(states)
        next_semantic_state = dict(semantic_state)
        next_semantic_state["phase"] = "semantic_link_restart_pending"
        next_semantic_state["cursor"] = None
        next_semantic_state.pop("pending", None)
        next_semantic_state["recovery"] = recovery_marker
        next_semantic_state["recovery_history"] = history
        next_semantic_state["updated_at"] = updated_at
        next_states["semantic_link"] = next_semantic_state

        if not await storage.renew_consolidation_lease(
            brain_id, owner_token, lease_seconds=LEASE_SECONDS
        ):
            raise ConsolidationLeaseBusyError("brain lease was lost before recovery commit")
        saved = await storage.compare_and_swap_semantic_link_recovery(
            brain_id=brain_id,
            expected_run_id=run_id,
            expected_owner_token=old_owner_token,
            expected_options_fingerprint=expected_options_fingerprint,
            expected_status=expected_status,
            expected_phase=str(phase),
            expected_updated_at=state["updated_at"],
            expected_reference_time=state["reference_time"],
            lease_owner_token=owner_token,
            new_owner_token=owner_token,
            strategy_states=next_states,
            counters=dict(top_counters),
            updated_at=updated_at,
            new_status="paused" if expected_status == "failed" else expected_status,
        )
        if saved is None:
            raise ConsolidationResumeMismatchError(
                "durable run changed during recovery; conditional update was not applied"
            )
        return {
            "run_id": run_id,
            "reference_time": reference_time,
            "already_recovered": False,
            "dry_run": False,
            "previous_phase": str(phase),
        }
    finally:
        try:
            await storage.release_consolidation_lease(brain_id, owner_token)
        except Exception:
            logger.warning("Could not release stale-source recovery lease for brain %r", brain_id)


def supports_persistent_progress(storage: Any) -> bool:
    """Whether this backend implements the shared SurrealDB progress contract."""
    required = (
        "get_consolidation_progress",
        "start_consolidation_progress",
        "claim_consolidation_progress",
        "save_consolidation_progress",
        "acquire_consolidation_lease",
        "renew_consolidation_lease",
        "release_consolidation_lease",
    )
    return bool(getattr(storage, "current_brain_id", None)) and supports_storage_methods(
        storage, required
    )


def supports_storage_methods(storage: Any, method_names: Sequence[str]) -> bool:
    """Return true only when the concrete backend overrides the optional API.

    Static class inspection avoids treating dynamically-created mock attributes
    or the base class's NotImplemented stubs as backend capabilities.
    """
    storage_type = type(storage)
    for name in method_names:
        implementation = inspect.getattr_static(storage_type, name, None)
        base_implementation = inspect.getattr_static(NeuralStorage, name, None)
        if not callable(implementation) or implementation is base_implementation:
            return False
    return True


class ConsolidationProgressSession:
    """Own a distributed lease and persist committed strategy checkpoints."""

    def __init__(
        self,
        storage: Any,
        brain_id: str,
        owner_token: str,
        state: dict[str, Any],
        *,
        lease_seconds: int = LEASE_SECONDS,
    ) -> None:
        self.storage = storage
        self.brain_id = brain_id
        self.owner_token = owner_token
        self.state = state
        self.lease_seconds = lease_seconds
        self.resumed = False
        self._lease_lost = False
        self._closed = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(
        cls,
        storage: Any,
        requested_strategies: Sequence[str],
        fingerprint: str,
        reference_time: datetime,
        *,
        explicit_reference_time: datetime | None = None,
    ) -> ConsolidationProgressSession | None:
        """Acquire the brain lease, then resume or initialize its active record."""
        if not supports_persistent_progress(storage):
            return None

        brain_id = str(storage.current_brain_id)
        owner_token = str(uuid4())
        acquired = await storage.acquire_consolidation_lease(
            brain_id, owner_token, lease_seconds=LEASE_SECONDS
        )
        if not acquired:
            raise ConsolidationLeaseBusyError(
                f"brain {brain_id!r} already has an active consolidation worker"
            )

        try:
            requested = sorted(set(requested_strategies))
            existing = await storage.get_consolidation_progress(brain_id)
            if existing and str(existing.get("status")) in INCOMPLETE_STATUSES:
                if existing.get("engine_version") != CONSOLIDATION_ENGINE_VERSION:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation uses a different code version; "
                        "its checkpoint was left untouched"
                    )
                if existing.get("format_version") != PROGRESS_FORMAT_VERSION:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation uses an incompatible progress format; "
                        "its checkpoint was left untouched"
                    )
                if existing.get("schema_version") != SCHEMA_VERSION:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation uses a different database schema version; "
                        "its checkpoint was left untouched"
                    )
                if existing.get("options_fingerprint") != fingerprint:
                    raise ConsolidationResumeMismatchError(
                        "unfinished consolidation has different strategy/configuration "
                        "parameters; its checkpoint was left untouched"
                    )
                if sorted(existing.get("requested_strategies") or []) != requested:
                    raise ConsolidationResumeMismatchError(
                        "requested strategies differ from the unfinished run; "
                        "its checkpoint was left untouched"
                    )
                previous_reference_time = _parse_reference_time(existing.get("reference_time"))
                if (
                    explicit_reference_time is not None
                    and ensure_naive_utc(explicit_reference_time) != previous_reference_time
                ):
                    raise ConsolidationResumeMismatchError(
                        "explicit reference_time differs from the unfinished run; "
                        "its checkpoint was left untouched"
                    )
                claimed = await storage.claim_consolidation_progress(brain_id, owner_token)
                if claimed is None:
                    raise ConsolidationLeaseBusyError(
                        "unfinished consolidation changed while its lease was acquired"
                    )
                session = cls(storage, brain_id, owner_token, dict(claimed))
                session.resumed = True
            else:
                state: dict[str, Any] = {
                    "brain_id": brain_id,
                    "run_id": str(uuid4()),
                    "schema_version": SCHEMA_VERSION,
                    "format_version": PROGRESS_FORMAT_VERSION,
                    "engine_version": CONSOLIDATION_ENGINE_VERSION,
                    "requested_strategies": requested,
                    "completed_strategies": [],
                    "options_fingerprint": fingerprint,
                    "reference_time": ensure_naive_utc(reference_time),
                    "status": "running",
                    "current_strategy": None,
                    "phase": "starting",
                    "cursor": None,
                    "strategy_states": {},
                    "counters": {},
                    "owner_token": owner_token,
                    "started_at": utcnow(),
                    "updated_at": utcnow(),
                    "last_error": None,
                }
                record = await storage.start_consolidation_progress(state)
                session = cls(storage, brain_id, owner_token, dict(record))
            session._heartbeat_task = asyncio.create_task(
                session._renew_lease(), name=f"consolidation-lease:{brain_id}"
            )
            return session
        except BaseException:
            await storage.release_consolidation_lease(brain_id, owner_token)
            raise

    @property
    def lease_lost(self) -> bool:
        """Whether the background heartbeat has fenced this worker out."""
        return self._lease_lost

    @property
    def reference_time(self) -> datetime:
        return _parse_reference_time(self.state.get("reference_time"))

    @property
    def completed_strategies(self) -> set[str]:
        return set(self.state.get("completed_strategies") or [])

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        states = self.state.get("strategy_states") or {}
        current = states.get(strategy, {})
        return dict(current) if isinstance(current, Mapping) else {}

    @property
    def last_checkpoint(self) -> str:
        strategy = self.state.get("current_strategy") or "none"
        per_strategy = self.strategy_state(str(strategy))
        phase = per_strategy.get("phase") or self.state.get("phase") or "unknown"
        cursor = per_strategy.get("cursor") or self.state.get("cursor") or "start"
        return f"{strategy} phase={phase} cursor={cursor}"

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: Sequence[str] | None = None,
        counters: Mapping[str, int | float] | None = None,
    ) -> None:
        """Persist the latest committed unit boundary for one strategy."""
        if self._lease_lost:
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} was lost; stopping before more work"
            )
        self.state["status"] = "running"
        self.state["current_strategy"] = strategy
        self.state["phase"] = phase
        self.state["cursor"] = cursor
        states = dict(self.state.get("strategy_states") or {})
        strategy_state = dict(states.get(strategy) or {})
        strategy_state.update(phase=phase, cursor=cursor, updated_at=utcnow())
        if pending is not None:
            strategy_state["pending"] = list(pending)
        elif "pending" in strategy_state:
            strategy_state.pop("pending")
        if counters is not None:
            strategy_state["counters"] = dict(counters)
            all_counters = dict(self.state.get("counters") or {})
            all_counters[strategy] = dict(counters)
            self.state["counters"] = all_counters
        states[strategy] = strategy_state
        self.state["strategy_states"] = states
        await self._save()

    async def complete_strategy(self, strategy: str) -> None:
        completed = list(self.state.get("completed_strategies") or [])
        if strategy not in completed:
            completed.append(strategy)
        self.state["completed_strategies"] = completed
        self.state["current_strategy"] = None
        self.state["phase"] = "strategy_completed"
        self.state["cursor"] = None
        states = dict(self.state.get("strategy_states") or {})
        current = dict(states.get(strategy) or {})
        current.update(phase="completed", cursor=None, updated_at=utcnow())
        current.pop("pending", None)
        states[strategy] = current
        self.state["strategy_states"] = states
        await self._save()

    async def pause(self) -> None:
        self.state["status"] = "paused"
        await self._save()

    async def fail(self, error: str) -> None:
        self.state["status"] = "failed"
        self.state["last_error"] = error[:2000]
        await self._save()

    async def complete(self) -> None:
        self.state["status"] = "completed"
        self.state["current_strategy"] = None
        self.state["phase"] = "completed"
        self.state["cursor"] = None
        self.state["last_error"] = None
        await self._save()

    async def _save(self) -> None:
        if self._lease_lost:
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} was lost; checkpoint was not saved"
            )
        saved = await self.storage.save_consolidation_progress(
            self.brain_id, self.owner_token, self.state
        )
        if saved is None:
            self._lease_lost = True
            raise ConsolidationLeaseLostError(
                f"lease fence for brain {self.brain_id!r} rejected the checkpoint"
            )
        self.state = dict(saved)

    async def _renew_lease(self) -> None:
        interval = max(10, self.lease_seconds // 3)
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                if not await self.storage.renew_consolidation_lease(
                    self.brain_id,
                    self.owner_token,
                    lease_seconds=self.lease_seconds,
                ):
                    self._lease_lost = True
                    logger.error(
                        "Consolidation lease lost for brain %s; the active strategy "
                        "will stop at its next committed checkpoint",
                        self.brain_id,
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._lease_lost = True
            logger.exception("Consolidation lease renewal failed for brain %s", self.brain_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        try:
            await self.storage.release_consolidation_lease(self.brain_id, self.owner_token)
        except Exception:
            logger.exception("Failed to release consolidation lease for brain %s", self.brain_id)
