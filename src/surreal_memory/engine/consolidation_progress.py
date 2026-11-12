"""Durable, brain-scoped state for resumable consolidation runs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION
from surreal_memory.utils.timeutils import ensure_naive_utc, utcnow

logger = logging.getLogger(__name__)

PROGRESS_FORMAT_VERSION = 1
# Keep this compatibility epoch stable across package releases; change it only
# when checkpoint semantics become incompatible.
CONSOLIDATION_ENGINE_VERSION = "3.11.0:checkpoint-v1"
LEASE_SECONDS = 180
INCOMPLETE_STATUSES = {"queued", "running", "paused", "failed"}


class ConsolidationProgressError(RuntimeError):
    """Base error for durable consolidation coordination."""


class ConsolidationCompletionUnconfirmedError(ConsolidationProgressError):
    """The terminal checkpoint could not be confirmed without replaying writes."""


class ConsolidationPausedError(ConsolidationProgressError):
    """The strategy reached its budget at a committed work-unit boundary."""


class ConsolidationChunkCompleteError(ConsolidationPausedError):
    """A committed chunk can continue within the current invocation's budget."""


class ConsolidationLeaseBusyError(ConsolidationProgressError):
    """Another worker currently owns this brain's consolidation lease."""


class ConsolidationResumeMismatchError(ConsolidationProgressError):
    """An unfinished run cannot safely resume with the requested parameters."""


class ConsolidationLeaseLostError(ConsolidationProgressError):
    """The worker lost its lease and must stop before starting another work unit."""


class ConsolidationCheckpointUnconfirmedError(ConsolidationLeaseLostError):
    """An uncertain checkpoint publication fences this session against stale writes."""


def options_fingerprint(options: Mapping[str, Any]) -> str:
    """Return a stable digest for all behavior-affecting run parameters."""
    encoded = json.dumps(options, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def stored_form(value: Any) -> Any:
    """Return ``value`` as SurrealDB stores it.

    The Python SDK sends ``None`` as NONE, and SurrealDB keeps no object key whose
    value is NONE, at any depth; array elements are kept. Census rows are
    fingerprinted and compared in this form, so the row read back is the row written.
    """
    if isinstance(value, dict):
        return {key: stored_form(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [stored_form(item) for item in value]
    return value


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


def validate_semantic_source_generation_audit(
    marker: Any,
    *,
    brain_id: str,
    run_id: str,
    options_fingerprint: str,
    reference_time: Any,
) -> bool:
    """Recognize historical source failures only with their generation evidence."""
    if not isinstance(marker, Mapping):
        return False
    expected_fields = {
        "kind",
        "version",
        "brain_id",
        "run_id",
        "options_fingerprint",
        "reference_time",
        "expected_status",
        "previous_phase",
        "generation",
        "semantics",
        "source_changed_verified",
        "source_change",
        "prior_manifest",
        "manifest_sha256",
        "prior_counters",
        "counters_sha256",
        "retired_source_state_ref",
        "retired_cursor",
        "source_token_sha256",
        "new_source_token_sha256",
        "source_state_ref",
        "bootstrap_sha256",
        "resolved_error",
        "resolved_nested_error",
        "owner_token",
        "recovered_at",
    }
    optional_fields = {"retired_cursor", "resolved_error", "resolved_nested_error"}
    if set(marker) - expected_fields or expected_fields - optional_fields - set(marker):
        return False

    def digest(value: Any) -> str:
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def safe_metadata(value: Any, depth: int = 0) -> bool:
        if depth > 64:
            return False
        if isinstance(value, Mapping):
            return all(
                (
                    not any(word in str(key).lower() for word in ("error", "fail"))
                    or (key == "semantic_link_failures" and item == 0)
                )
                and safe_metadata(item, depth + 1)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return all(safe_metadata(item, depth + 1) for item in value)
        return True

    try:
        prior = marker["prior_manifest"]
        inner = prior["discovery"] if prior.get("stage") == "validation" else prior
        old_ref, new_ref = marker["retired_source_state_ref"], marker["source_state_ref"]
        old_token, new_token = old_ref["source_token"], new_ref["source_token"]
        old_decoded, new_decoded = json.loads(old_token), json.loads(new_token)
        bootstrap = {
            "kind": "semantic_link_discovery",
            "version": 5,
            "stage": "neurons",
            "cursor": None,
            "source_state_ref": new_ref,
        }
        return (
            marker.get("kind") == "stale_semantic_source_restart"
            and marker.get("version") == 2
            and marker.get("brain_id") == brain_id
            and marker.get("run_id") == run_id
            and marker.get("options_fingerprint") == options_fingerprint
            and _parse_reference_time(marker.get("reference_time"))
            == _parse_reference_time(reference_time)
            and type(marker.get("generation")) is int
            and marker["generation"] > 0
            and marker.get("semantics") == "cumulative_write_events"
            and marker.get("source_changed_verified") is True
            and all(
                safe_metadata(marker[key])
                for key in (
                    "prior_manifest",
                    "prior_counters",
                    "retired_source_state_ref",
                    "source_state_ref",
                )
            )
            and marker.get("expected_status") in {"running", "failed"}
            and marker.get("previous_phase") == "semantic_link_discovery_" + str(prior.get("stage"))
            and marker.get("retired_cursor") == prior.get("cursor")
            and isinstance(marker.get("source_change"), str)
            and 0 < len(marker["source_change"]) <= 2000
            and _parse_reference_time(marker["recovered_at"])
            >= _parse_reference_time(reference_time)
            and new_ref["state_id"] != old_ref["state_id"]
            and type(old_decoded.get("versionstamp")) is int
            and old_decoded["versionstamp"] >= 0
            and type(new_decoded.get("versionstamp")) is int
            and new_decoded["versionstamp"] >= old_decoded["versionstamp"]
            and all(
                decoded.get("version") == 1
                and decoded.get("created_at_readonly") is True
                and _parse_reference_time(decoded["captured_at"]) is not None
                for decoded in (old_decoded, new_decoded)
            )
            and inner.get("kind") == "semantic_link_discovery"
            and inner.get("source_state_ref") == old_ref
            and marker.get("manifest_sha256") == digest(prior)
            and isinstance(marker.get("prior_counters"), Mapping)
            and all(
                type(value) is int and value >= 0 for value in marker["prior_counters"].values()
            )
            and marker["prior_counters"].get("semantic_link_failures", 0) == 0
            and marker.get("counters_sha256") == digest(marker["prior_counters"])
            and marker.get("source_token_sha256") == sha256(old_token.encode()).hexdigest()
            and marker.get("new_source_token_sha256") == sha256(new_token.encode()).hexdigest()
            and marker.get("bootstrap_sha256") == digest(bootstrap)
            and all(
                isinstance(ref.get("state_id"), str)
                and bool(ref["state_id"])
                and type(ref.get("revision")) is int
                and ref["revision"] > 0
                and ref.get("state_revision", ref["revision"]) == ref["revision"]
                and ref.get("brain_id", brain_id) == brain_id
                and ref.get("run_id", run_id) == run_id
                and json.loads(ref["source_token"]).get("brain_id") == brain_id
                for ref in (old_ref, new_ref)
            )
            and isinstance(marker.get("owner_token"), str)
            and bool(marker["owner_token"])
            and marker.get("recovered_at") is not None
            and all(
                _known_source_error(marker.get(key))
                for key in ("resolved_error", "resolved_nested_error")
            )
        )
    except (KeyError, AttributeError, TypeError, ValueError, ConsolidationProgressError):
        return False


def _known_source_error(value: Any) -> bool:
    return value in (
        None,
        "",
        "semantic_link (progress coordination: ConsolidationProgressError)",
    ) or (
        isinstance(value, str)
        and value.startswith("semantic_link source changed; discovery checkpoint preserved:")
    )


async def _restart_semantic_source_generation(
    storage: Any,
    state: Mapping[str, Any],
    owner_token: str,
    *,
    dry_run: bool = True,
    source_changed: bool = False,
    source_change_reason: str | None = None,
    budget_check: Callable[[], Awaitable[None]] | None = None,
    progress_session: ConsolidationProgressSession | None = None,
) -> dict[str, Any]:
    """Retire an unapplied proposal; committed counters remain write-event totals."""
    from surreal_memory.engine.consolidation import ConsolidationConfig, ConsolidationStrategy
    from surreal_memory.engine.semantic_discovery import _load_source_state
    from surreal_memory.storage.surrealdb.semantic_source_revision import SemanticSourceChangedError

    loop = asyncio.get_running_loop()
    deadline = loop.time() + ConsolidationConfig().strategy_timeout_seconds
    next_renewal = loop.time() + LEASE_SECONDS / 3

    async def within_budget() -> bool:
        nonlocal next_renewal
        if budget_check is not None:
            await budget_check()
        if loop.time() >= deadline:
            return False
        if loop.time() >= next_renewal:
            if not await storage.renew_consolidation_lease(
                str(storage.current_brain_id), owner_token, lease_seconds=LEASE_SECONDS
            ):
                raise ConsolidationLeaseBusyError("recovery validation lease was lost")
            next_renewal = loop.time() + LEASE_SECONDS / 3
        return True

    brain_id = str(storage.current_brain_id)
    run_id = str(state["run_id"])
    semantic = state.get("strategy_states", {}).get("semantic_link")
    if not isinstance(semantic, Mapping):
        raise ConsolidationResumeMismatchError("semantic discovery state is missing")
    phase = state.get("phase")
    counters = semantic.get("counters")
    allowed_counters = {
        "semantic_synapses_created",
        "semantic_synapses_skipped",
        "semantic_link_failures",
        "eligible_neurons",
        "existing_pairs",
        "synapses_created",
        "skipped_existing",
        "pairs_evaluated",
        "validation_database_events",
        "validation_neuron_events",
        "validation_synapse_events",
        "validation_cursor",
    }
    strategies = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    if (
        state.get("brain_id") != brain_id
        or state.get("current_strategy") != "semantic_link"
        or state.get("status") not in {"running", "failed", "paused"}
        or len(state.get("requested_strategies", [])) != len(strategies)
        or set(state.get("requested_strategies", [])) != strategies
        or len(state.get("completed_strategies", [])) != len(strategies) - 1
        or set(state.get("completed_strategies", [])) != strategies - {"semantic_link"}
        or not isinstance(counters, Mapping)
        or dict(counters) != state.get("counters", {}).get("semantic_link")
        or set(counters) - allowed_counters
        or any(type(value) is not int or value < 0 for value in counters.values())
        or counters.get("semantic_link_failures", 0) != 0
        or semantic.get("phase") != phase
        or semantic.get("cursor") != state.get("cursor")
        or not _known_source_error(state.get("last_error"))
        or not _known_source_error(semantic.get("last_error"))
    ):
        raise ConsolidationResumeMismatchError(
            "source generation evidence or active error is unsafe"
        )
    reference_time = _parse_reference_time(state.get("reference_time"))
    old_marker = semantic.get("recovery")
    history_raw = semantic.get("recovery_history", [])
    if (
        not isinstance(history_raw, (list, tuple))
        or any(not isinstance(item, Mapping) for item in history_raw)
        or (old_marker is not None and not isinstance(old_marker, Mapping))
        or (history_raw and history_raw[-1] != old_marker)
        or (
            isinstance(old_marker, Mapping)
            and old_marker.get("version") == 2
            and not validate_semantic_source_generation_audit(
                old_marker,
                brain_id=brain_id,
                run_id=run_id,
                options_fingerprint=str(state.get("options_fingerprint")),
                reference_time=reference_time,
            )
        )
    ):
        raise ConsolidationResumeMismatchError("source generation history is malformed")
    if phase == "semantic_link_restart_pending":
        manifest, _ = _decode_legacy_discovery_pending(semantic.get("pending"))
        if not isinstance(old_marker, Mapping):
            raise ConsolidationResumeMismatchError("source generation audit is missing")
        if (
            not validate_semantic_source_generation_audit(
                old_marker,
                brain_id=brain_id,
                run_id=run_id,
                options_fingerprint=str(state.get("options_fingerprint")),
                reference_time=reference_time,
            )
            or old_marker.get("bootstrap_sha256") != options_fingerprint(manifest)
            or old_marker.get("prior_counters") != dict(counters)
            or old_marker.get("owner_token") != state.get("owner_token")
            or _parse_reference_time(old_marker.get("recovered_at"))
            != _parse_reference_time(state.get("updated_at"))
            or _parse_reference_time(old_marker.get("recovered_at"))
            != _parse_reference_time(semantic.get("updated_at"))
            or not semantic.get("recovery_history")
            or semantic["recovery_history"][-1] != old_marker
        ):
            raise ConsolidationResumeMismatchError("source generation retry evidence changed")
        return {
            "run_id": run_id,
            "reference_time": reference_time,
            "already_recovered": True,
            "dry_run": dry_run,
            "previous_phase": old_marker["previous_phase"],
        }

    if state.get("status") == "paused":
        raise ConsolidationResumeMismatchError(
            "paused source generation recovery only verifies an exact restart"
        )
    wrapper, _ = _decode_legacy_discovery_pending(semantic.get("pending"))
    wrapper = stored_form(wrapper)
    canonical = json.dumps(wrapper, sort_keys=True, separators=(",", ":"))
    manifest = wrapper
    if wrapper.get("stage") == "validation":
        inner = wrapper.get("discovery")
        if (
            phase != "semantic_link_discovery_validation"
            or wrapper.get("kind") != "semantic_link_discovery"
            or wrapper.get("version") != 5
            or not isinstance(inner, Mapping)
            or wrapper.get("cursor") != inner.get("cursor")
            or not isinstance(wrapper.get("source_validation_ref"), Mapping)
        ):
            raise ConsolidationResumeMismatchError("invalid discovery validation wrapper")
        manifest = dict(inner)
    elif phase != "semantic_link_discovery_" + str(wrapper.get("stage")):
        raise ConsolidationResumeMismatchError("source generation requires unapplied discovery")
    if (
        manifest.get("kind") != "semantic_link_discovery"
        or manifest.get("version") not in {4, 5}
        or manifest.get("stage") not in {"neurons", "synapses", "similarity"}
        or wrapper.get("cursor") != state.get("cursor")
    ):
        raise ConsolidationResumeMismatchError("invalid source generation discovery manifest")
    source_ref = manifest.get("source_state_ref")
    try:
        source, _, revision = await _load_source_state(
            storage,
            source_ref,
            brain_id=brain_id,
            run_id=run_id,
            recovery_owner_token=owner_token,
            expected_progress_owner_token=state["owner_token"],
        )
    except RuntimeError as exc:
        raise ConsolidationResumeMismatchError(str(exc)) from exc
    assert isinstance(source_ref, Mapping)
    token = source.get("source_token")
    batch_index = source.get("batch_index")
    if (
        source_ref.get("state_revision", revision) != revision
        or source_ref.get("brain_id", brain_id) != brain_id
        or source_ref.get("run_id", run_id) != run_id
        or source.get("state_revision", revision) != revision
        or not isinstance(token, str)
        or source_ref.get("source_token") != token
        or _parse_reference_time(source.get("reference_time")) != reference_time
        or type(batch_index) is not int
        or batch_index < 0
        or (counters.get("semantic_synapses_created", 0) > 0 and batch_index < 1)
    ):
        raise ConsolidationResumeMismatchError("staged source generation scope changed")
    decoded = json.loads(token)
    if (
        decoded.get("version") != 1
        or decoded.get("brain_id") != brain_id
        or decoded.get("created_at_readonly") is not True
        or type(decoded.get("versionstamp")) is not int
    ):
        raise ConsolidationResumeMismatchError("source generation token is not readonly-attested")
    proof_ref = wrapper.get("source_validation_ref", manifest.get("source_validation_ref"))
    if proof_ref is not None:
        if not isinstance(proof_ref, Mapping) or proof_ref.get("source_token") != token:
            raise ConsolidationResumeMismatchError("source generation validation scope changed")
        if callable(getattr(storage, "_create_semantic_source_fence_session", None)):
            from surreal_memory.engine.semantic_validation import SemanticSourceValidation

            validation = SemanticSourceValidation(
                storage,
                token,
                reference_time=reference_time,
                run_id=run_id,
                owner_token=owner_token,
                reference=proof_ref,
                recovery_owner_token=owner_token,
                expected_progress_owner_token=state["owner_token"],
            )
            try:
                await validation.restore()
            except RuntimeError as exc:
                raise ConsolidationResumeMismatchError(str(exc)) from exc
            if not source_changed:
                while True:
                    previous_watermark = validation.session.watermark
                    previous_target = validation.session.target_token
                    try:
                        status = await validation.session.verify(budget_check=within_budget)
                    except SemanticSourceChangedError as exc:
                        source_changed = True
                        source_change_reason = str(exc)
                        break
                    except RuntimeError as exc:
                        raise ConsolidationPausedError(
                            f"source generation validation unavailable; checkpoint unchanged: {exc}"
                        ) from exc
                    if status == "verified":
                        raise ConsolidationResumeMismatchError(
                            "semantic source change before reference_time was not proven"
                        )
                    if status != "pending" or (
                        previous_target is not None
                        and validation.session.target_token != previous_target
                    ):
                        raise ConsolidationResumeMismatchError("source validation target changed")
                    if (
                        validation.session.watermark <= previous_watermark
                        or not await within_budget()
                    ):
                        raise ConsolidationPausedError(
                            "source generation validation pending; checkpoint unchanged"
                        )
    if not source_changed:
        try:
            await storage.assert_semantic_source_unchanged(token, created_before=reference_time)
        except SemanticSourceChangedError as exc:
            source_changed = True
            source_change_reason = str(exc)
    if not source_changed:
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

    new_token = await storage.capture_semantic_source_token()
    new_decoded = json.loads(new_token)
    if (
        new_decoded.get("brain_id") != brain_id
        or new_decoded.get("version") != 1
        or new_decoded.get("created_at_readonly") is not True
        or type(new_decoded.get("versionstamp")) is not int
        or new_decoded["versionstamp"] < decoded["versionstamp"]
    ):
        raise ConsolidationResumeMismatchError("new source generation token is invalid")
    state_id = "generation-" + str(uuid4())
    fresh = {
        "state_id": state_id,
        "revision": 1,
        "state_revision": 1,
        "brain_id": brain_id,
        "run_id": run_id,
        "owner_token": owner_token,
        "source_token": new_token,
        "reference_time": reference_time.isoformat(timespec="microseconds"),
        "batch_index": max(1, batch_index),
        "coverage_offset": 0,
        "coverage_source_fingerprint": None,
        "legacy_checkpoint": None,
        "neuron_cursor": None,
        "edge_cursor": None,
        "neuron_complete": False,
        "edge_complete": False,
        "source_complete": False,
        "similarity_rebuild": None,
        "eligible_total": 0,
        "type_ranks": {"concept": 0, "entity": 0},
        "candidates": [],
        "bounded": False,
        "degree_mode": False,
    }
    await storage.save_semantic_discovery_state(state_id, 1, fresh)
    new_ref = {"state_id": state_id, "revision": 1, "state_revision": 1, "source_token": new_token}
    bootstrap = {
        "kind": "semantic_link_discovery",
        "version": 5,
        "stage": "neurons",
        "cursor": None,
        "source_state_ref": new_ref,
    }
    updated_at = utcnow()
    history = list(semantic.get("recovery_history", []))
    if old_marker is not None and (not history or history[-1] != old_marker):
        history.append(dict(old_marker))
    marker = {
        "kind": "stale_semantic_source_restart",
        "version": 2,
        "brain_id": brain_id,
        "run_id": run_id,
        "options_fingerprint": state["options_fingerprint"],
        "reference_time": reference_time,
        "expected_status": state["status"],
        "previous_phase": phase,
        "generation": 1 + max((item.get("generation", 0) for item in history), default=0),
        "semantics": "cumulative_write_events",
        "source_changed_verified": True,
        "source_change": str(source_change_reason or "semantic source changed")[:2000],
        "prior_manifest": wrapper,
        "manifest_sha256": sha256(canonical.encode()).hexdigest(),
        "prior_counters": dict(counters),
        "counters_sha256": options_fingerprint(counters),
        "retired_source_state_ref": dict(source_ref),
        "retired_cursor": state.get("cursor"),
        "source_token_sha256": sha256(token.encode()).hexdigest(),
        "new_source_token_sha256": sha256(new_token.encode()).hexdigest(),
        "source_state_ref": new_ref,
        "bootstrap_sha256": options_fingerprint(bootstrap),
        "resolved_error": state.get("last_error"),
        "resolved_nested_error": semantic.get("last_error"),
        "owner_token": owner_token,
        "recovered_at": updated_at,
    }
    history.append(marker)
    next_semantic = dict(semantic)
    next_semantic.update(
        phase="semantic_link_restart_pending",
        cursor=None,
        pending=[json.dumps(bootstrap, sort_keys=True, separators=(",", ":"))],
        recovery=marker,
        recovery_history=history,
        updated_at=updated_at,
    )
    next_semantic.pop("last_error", None)
    states = dict(state["strategy_states"])
    states["semantic_link"] = next_semantic
    if not await storage.renew_consolidation_lease(
        brain_id, owner_token, lease_seconds=LEASE_SECONDS
    ):
        raise ConsolidationLeaseBusyError("brain lease was lost before generation commit")

    def confirmed(candidate: Any) -> bool:
        if not isinstance(candidate, Mapping):
            return False
        try:
            current_semantic = candidate["strategy_states"]["semantic_link"]
            current_marker = dict(current_semantic["recovery"])
            intended_marker = dict(marker)
            for key in ("reference_time", "recovered_at"):
                current_marker[key] = _parse_reference_time(current_marker[key])
                intended_marker[key] = _parse_reference_time(intended_marker[key])
            pending_manifest, _ = _decode_legacy_discovery_pending(current_semantic.get("pending"))
            return (
                candidate.get("brain_id") == brain_id
                and candidate.get("owner_token") == owner_token
                and candidate.get("status")
                == ("paused" if state["status"] == "failed" else state["status"])
                and candidate.get("phase") == "semantic_link_restart_pending"
                and candidate.get("current_strategy") == "semantic_link"
                and candidate.get("cursor") is None
                and candidate.get("last_error") in (None, "")
                and _parse_reference_time(candidate.get("reference_time")) == reference_time
                and _parse_reference_time(candidate.get("updated_at")) == updated_at
                and all(
                    candidate.get(key) == state.get(key)
                    for key in (
                        "run_id",
                        "options_fingerprint",
                        "requested_strategies",
                        "completed_strategies",
                        "counters",
                    )
                )
                and stored_form(current_marker) == stored_form(intended_marker)
                and current_semantic.get("phase") == "semantic_link_restart_pending"
                and current_semantic.get("cursor") is None
                and current_semantic.get("counters") == counters
                and _parse_reference_time(current_semantic.get("updated_at")) == updated_at
                and options_fingerprint(pending_manifest) == marker["bootstrap_sha256"]
                and stored_form(current_semantic["recovery_history"][:-1])
                == stored_form(history[:-1])
                and current_semantic["recovery_history"][-1] == current_semantic["recovery"]
                and all(
                    stored_form(candidate["strategy_states"].get(key)) == stored_form(value)
                    for key, value in state["strategy_states"].items()
                    if key != "semantic_link"
                )
            )
        except (KeyError, IndexError, TypeError, ValueError, ConsolidationProgressError):
            return False

    if progress_session is not None:
        progress_session._checkpoint_unconfirmed = True
    publication_error: BaseException | None = None
    try:
        saved = await storage.compare_and_swap_semantic_link_recovery(
            brain_id=brain_id,
            expected_run_id=run_id,
            expected_owner_token=state["owner_token"],
            expected_options_fingerprint=state["options_fingerprint"],
            expected_status=state["status"],
            expected_phase=str(phase),
            expected_updated_at=state["updated_at"],
            expected_reference_time=state["reference_time"],
            lease_owner_token=owner_token,
            new_owner_token=owner_token,
            strategy_states=states,
            counters=dict(state["counters"]),
            updated_at=updated_at,
            new_status="paused" if state["status"] == "failed" else state["status"],
            resolved_source_error=state.get("last_error"),
        )
    except (Exception, asyncio.CancelledError) as exc:
        publication_error = exc
        saved = None
    if not confirmed(saved):
        try:
            saved = await asyncio.wait_for(
                asyncio.shield(storage.get_consolidation_progress(brain_id)), timeout=10.0
            )
        except (Exception, asyncio.CancelledError):
            saved = None
        if not confirmed(saved):
            raise ConsolidationCheckpointUnconfirmedError(
                "source generation publication is unconfirmed; no stale failure or pause checkpoint was written; "
                "inspect the durable run before resuming"
            ) from publication_error
    if progress_session is not None:
        progress_session.state = dict(saved)
        progress_session._checkpoint_unconfirmed = False
    if isinstance(publication_error, asyncio.CancelledError):
        raise publication_error
    return {
        "run_id": run_id,
        "reference_time": reference_time,
        "already_recovered": False,
        "dry_run": False,
        "previous_phase": str(phase),
        "generation": marker["generation"],
        "state": saved,
    }


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
    if expected_status not in {"running", "failed", "paused"}:
        raise ConsolidationResumeMismatchError(
            "stale-source recovery requires a running or failed run, or an exact paused retry"
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

        semantic = state.get("strategy_states", {}).get("semantic_link", {})
        marker = semantic.get("recovery") if isinstance(semantic, Mapping) else None
        if state.get("phase") == "semantic_link_discovery_validation" or (
            isinstance(marker, Mapping) and marker.get("version") == 2
        ):
            return await _restart_semantic_source_generation(
                storage, state, owner_token, dry_run=dry_run
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
        if expected_status == "paused" and phase != "semantic_link_restart_pending":
            raise ConsolidationResumeMismatchError(
                "paused stale-source recovery only verifies an already committed failed-run restart"
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
                and latest_recovery.get("expected_status")
                == ("failed" if expected_status == "paused" else expected_status)
                and latest_recovery.get("previous_phase") in _LEGACY_DISCOVERY_PHASES
                and latest_recovery.get("source_changed_verified") is True
                and latest_recovery.get("owner_token") == state.get("owner_token")
                and latest_recovery.get("recovered_at") == state.get("updated_at")
                and latest_recovery.get("recovered_at") == semantic_state.get("updated_at")
                and (
                    (
                        len(history) == 1
                        and latest_recovery.get("prior_recovery_sha256")
                        == sha256(b"null").hexdigest()
                    )
                    or (
                        isinstance(prior, Mapping)
                        and prior.get("kind")
                        in {"legacy_semantic_discovery_restart", "stale_semantic_source_restart"}
                        and (
                            prior.get("kind") != "stale_semantic_source_restart"
                            or prior.get("source_changed_verified") is True
                        )
                        and prior.get("run_id") == run_id
                        and prior.get("options_fingerprint") == expected_options_fingerprint
                    )
                )
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
        first_recovery = "recovery" not in semantic_state and not recovery_history_raw
        if not isinstance(previous_recovery, Mapping):
            if not first_recovery:
                raise ConsolidationResumeMismatchError(
                    "the prior semantic recovery audit marker is missing or unrecognized"
                )
            previous_recovery = {}
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
            first_marker = history[0] if history else None
            prior_marker = history[-1] if history else None
            if (
                not isinstance(first_marker, Mapping)
                or not (
                    first_marker.get("kind") == "legacy_semantic_discovery_restart"
                    or (
                        first_marker.get("kind") == "stale_semantic_source_restart"
                        and first_marker.get("source_changed_verified") is True
                        and first_marker.get("prior_recovery_sha256") == sha256(b"null").hexdigest()
                    )
                )
                or first_marker.get("run_id") != run_id
                or first_marker.get("options_fingerprint") != expected_options_fingerprint
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
        elif not first_recovery:
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
        prior_marker = None if first_recovery else dict(previous_recovery)
        history = [dict(item) for item in recovery_history_raw]
        if prior_marker is not None and (not history or history[-1] != prior_marker):
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


# Only these existing committed unit contracts can establish work from ordinary checkpoints.
_WORK_CURSOR_PHASES = frozenset(
    [
        "synapse_scan",
        "neuron_scan",
        "retention_change_log",
        "fiber_census",
        "dedup_census",
        "dedup_pairs",
        "merge_plan_stage",
        "merge_plan_pairs",
        "merge_plan_members",
        "merge_plan_units",
        "summarize_plan_stage",
        "summarize_plan_pairs",
        "summarize_plan_members",
        "summarize_plan_units",
        "summarize_pairs",
        "summarize_cluster",
        "mature_promotion",
        "mature_stage",
        "mature_pattern",
        "essence_scan",
        "enrich_apply",
        "dream_plan",
        "dream_apply",
        "replay_apply",
        "schema_apply",
        "interference_scan",
        "interference_aggregate",
        "drift_clusters",
        "scan",
        "batch_committed",
        "reasoning_ingest",
        "reasoning_distill",
        "infer_prune",
        "infer_add",
        "infer_reinforce",
        "infer_counts",
        "infer_classify",
        "infer_select",
        "infer_tags_existing_scan",
        "infer_tags",
    ]
)
_WORK_COUNTERS = frozenset(
    [
        "synapses_pruned",
        "neurons_pruned",
        "synapse_pages",
        "neuron_pages",
        "action_events_pruned",
        "change_log_collapsed",
        "co_activations_pruned",
        "decay_passes_pruned",
        "retrieval_traces_pruned",
        "pages",
        "fibers",
        "neurons",
        "anchors",
        "merge_plan_candidates",
        "merge_pairs_examined",
        "merge_plan_members",
        "summarize_plan_candidates",
        "summarize_pairs_examined",
        "summarize_plan_members",
        "summaries_created",
        "summaries_skipped_existing",
        "memories_promoted",
        "patterns_extracted",
        "essences_generated",
        "synapses_enriched",
        "synapses_inferred",
        "dream_synapses_created",
        "replay_episodes",
        "replay_ltd",
        "replay_ltp",
        "schemas_created",
        "interference_fan_effects",
        "habits_learned",
        "action_habits_learned",
        "tool_habits_learned",
        "query_patterns_learned",
        "duplicates_found",
        "new_alias_links",
        "alias_links_existing",
        "fibers_compressed",
        "fibers_preserved",
        "tokens_saved",
        "persisted",
        "lifecycle_states_updated",
        "events_processed",
        "events_ingested",
        "files_scanned",
        "traces_scanned",
        "traces_ingested",
        "models_done",
        "traces_processed",
        "patterns_learned",
    ]
)


def _cursor_work_units(cursor: str | None) -> tuple[str, dict[str, int | str]]:
    """Extract ordered unit positions, never treating a serialized hash as work."""
    if not cursor or cursor == "completed":
        return "", {}
    try:
        decoded = json.loads(cursor)
    except (ValueError, TypeError):
        prefix, separator, suffix = cursor.rpartition(":")
        if separator and suffix.isdigit():
            return prefix, {"cursor": int(suffix)}
        return "", {"cursor": cursor}
    if type(decoded) is int:
        return "", {"cursor": decoded}
    if not isinstance(decoded, dict):
        return "", {}
    scope = options_fingerprint(
        {key: decoded[key] for key in ("kind", "plan_id", "fingerprint") if key in decoded}
    )
    if type(decoded.get("window_start")) is int and type(decoded.get("next_i")) is int:
        return scope, {"cursor": decoded["window_start"] + decoded["next_i"]}
    return scope, {
        key: value
        for key, value in decoded.items()
        if (key.startswith("after_") or key in {"next_sequence", "next_i", "window_start"})
        and (type(value) is int or isinstance(value, str))
    }


def _checkpoint_work_units(
    strategy: str, phase: str, cursor: str | None, counters: Mapping[str, Any]
) -> dict[str, tuple[str, int | str]]:
    """Read committed coverage/effects, excluding failure and coordination counters."""
    if strategy == "semantic_link":
        return {}
    scope, positions = _cursor_work_units(cursor) if phase in _WORK_CURSOR_PHASES else ("", {})
    units = {f"{phase}:{key}": (scope, value) for key, value in positions.items()}
    for key, value in counters.items():
        if key in _WORK_COUNTERS and type(value) is int and value >= 0:
            units[f"counter:{key}"] = ("", value)
    return units


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
        self._checkpoint_unconfirmed = False
        self._closed = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        # The lease is fresh when the session is created (acquired just before). Every successful
        # renewal, from the heartbeat or from ``renew_if_due``, moves the mark and feeds the
        # longest-gap metric that the report and the weekly ledger expose.
        self._last_renewed = time.monotonic()
        self._max_renew_gap_s = 0.0
        self._renew_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        storage: Any,
        requested_strategies: Sequence[str],
        fingerprint: str,
        reference_time: datetime,
        *,
        explicit_reference_time: datetime | None = None,
        resume_run_id: str | None = None,
    ) -> ConsolidationProgressSession | None:
        """Acquire the brain lease, then resume or initialize its active record."""
        if not supports_persistent_progress(storage):
            if resume_run_id is not None:
                raise ConsolidationResumeMismatchError(
                    "explicit resume requires durable consolidation progress"
                )
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
            if resume_run_id is not None and (
                not existing
                or existing.get("run_id") != resume_run_id
                or str(existing.get("status")) not in INCOMPLETE_STATUSES
            ):
                raise ConsolidationResumeMismatchError(
                    f"resume checkpoint {resume_run_id!r} changed or is no longer unfinished; "
                    "its checkpoint was left untouched"
                )
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
                        "parameters; its checkpoint was left untouched. "
                        f"Run: {existing.get('run_id')}; "
                        f"saved strategies: {sorted(existing.get('requested_strategies') or [])}; "
                        f"requested strategies: {requested}. "
                        "Use smem consolidate --resume with the original configuration "
                        "to finish the saved run before starting another strategy set"
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
                # This lease holder starts a new attempt at the same saved work.
                # A later ordinary budget pause must not retain the prior
                # attempt's failure. Persist through the existing owner fence
                # before any strategy executes; a new failure is recorded anew.
                session.state.update(status="running", last_error=None)
                await session._save()
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
    def max_renew_gap_s(self) -> float:
        """Longest observed gap between two successful lease renewals, in seconds."""
        return self._max_renew_gap_s

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
        work: Mapping[str, tuple[str, int | str, int | str]] | None = None,
    ) -> None:
        """Persist the latest committed unit boundary for one strategy."""
        if self._lease_lost:
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} was lost; stopping before more work"
            )
        previous = self.strategy_state(strategy)
        if work is None:
            prior_units = _checkpoint_work_units(
                strategy,
                str(previous.get("phase") or ""),
                previous.get("cursor"),
                previous.get("counters") or {},
            )
            next_units = _checkpoint_work_units(strategy, phase, cursor, counters or {})
            work = {}
            for key, (scope, position) in next_units.items():
                old_scope, old_position = prior_units.get(
                    key, (scope, 0 if type(position) is int else "")
                )
                if (
                    old_scope != scope
                    or (key.startswith(f"{phase}:") and cursor == previous.get("cursor"))
                    or (key.startswith(f"{phase}:") and phase.endswith("_pending"))
                ):
                    old_position = position
                work[key] = (scope, old_position, position)
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
        watermarks = dict(strategy_state.get("work_progress") or {})
        advanced = False
        for key, (scope, before, after) in work.items():
            saved = watermarks.get(key, {})
            saved_position = saved.get("position") if saved.get("scope") == scope else None
            if isinstance(after, int) and isinstance(before, int):
                before = max(before, saved_position) if type(saved_position) is int else before
                advanced |= after > before
                watermark: int | str = max(before, after)
            elif isinstance(after, str) and isinstance(before, str):
                before = max(before, saved_position) if isinstance(saved_position, str) else before
                advanced |= after > before
                watermark = max(before, after)
            else:
                raise ValueError("work positions must have matching ordered types")
            watermarks[key] = {"scope": scope, "position": watermark}
        if watermarks:
            strategy_state["work_progress"] = watermarks
        states[strategy] = strategy_state
        self.state["strategy_states"] = states
        if advanced:
            self._advance_work_marker()
        await self._save()

    async def complete_strategy(self, strategy: str) -> None:
        completed = list(self.state.get("completed_strategies") or [])
        if strategy not in completed:
            completed.append(strategy)
            self._advance_work_marker()
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

    def _advance_work_marker(self) -> None:
        self.state["work_progress_seq"] = int(self.state.get("work_progress_seq") or 0) + 1
        self.state["last_work_progress_at"] = utcnow()

    async def pause(self) -> None:
        self.state["status"] = "paused"
        await self._save()

    async def fail(self, error: str) -> None:
        self.state["status"] = "failed"
        self.state["last_error"] = error[:2000]
        await self._save()

    async def finalization_checkpoint(self, phase: str) -> None:
        """Persist a finalization boundary without inventing a strategy."""
        self.state["phase"] = phase
        await self._save()

    async def complete(self) -> None:
        self.state["status"] = "completed"
        self.state["current_strategy"] = None
        self.state["phase"] = "completed"
        self.state["cursor"] = None
        self.state["last_error"] = None
        try:
            await self._save()
        except ConsolidationLeaseLostError:
            raise
        except asyncio.CancelledError:
            self.state["status"] = "running"
            self.state["phase"] = "completion_unconfirmed"
            raise
        except Exception as exc:
            # A lost acknowledgement can follow a committed write; never replay it.
            try:
                persisted = await self.storage.get_consolidation_progress(self.brain_id)
            except Exception:
                persisted = None
            if (
                persisted
                and persisted.get("run_id") == self.state.get("run_id")
                and persisted.get("options_fingerprint") == self.state.get("options_fingerprint")
                and persisted.get("brain_id") == self.brain_id
                and persisted.get("status") == "completed"
                and persisted.get("phase") == "completed"
                and bool(self.state.get("requested_strategies"))
                and persisted.get("requested_strategies") == self.state.get("requested_strategies")
                and persisted.get("completed_strategies") == self.state.get("completed_strategies")
                and set(self.state["requested_strategies"]).issubset(
                    self.state["completed_strategies"]
                )
            ):
                self.state = dict(persisted)
                return
            # Local intent is not durable evidence of completion.
            self.state["status"] = "running"
            self.state["phase"] = "completion_unconfirmed"
            raise ConsolidationCompletionUnconfirmedError(
                "terminal checkpoint was not confirmed; inspect the saved run and use "
                "--resume if it remains unfinished"
            ) from exc

    async def _save(self) -> None:
        if self._checkpoint_unconfirmed:
            raise ConsolidationCheckpointUnconfirmedError(
                "checkpoint publication is unconfirmed; stale checkpoint writes are fenced; "
                "inspect the durable run before resuming"
            )
        if self._lease_lost:
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} was lost; checkpoint was not saved"
            )
        try:
            saved = await self.storage.save_consolidation_progress(
                self.brain_id, self.owner_token, self.state
            )
        except ConsolidationPausedError:
            # Budget pauses signal a committed boundary, not a transport failure.
            raise
        except BaseException:
            # A committed write may have lost its acknowledgement; only a new owner may retry.
            self._checkpoint_unconfirmed = True
            raise
        if saved is None:
            self._lease_lost = True
            raise ConsolidationLeaseLostError(
                f"lease fence for brain {self.brain_id!r} rejected the checkpoint"
            )
        self.state = dict(saved)

    async def _renew_once(self) -> bool:
        """Renew the lease once, serialised with other renewals, and record the gap."""
        async with self._renew_lock:
            renewed = await self.storage.renew_consolidation_lease(
                self.brain_id,
                self.owner_token,
                lease_seconds=self.lease_seconds,
            )
            if renewed:
                now = time.monotonic()
                self._max_renew_gap_s = max(self._max_renew_gap_s, now - self._last_renewed)
                self._last_renewed = now
            return bool(renewed)

    async def renew_if_due(self) -> None:
        """Renew the lease when the heartbeat has not done so for a third of its length.

        The heartbeat is an asyncio task on the same loop as the work. A scan that holds the loop
        (pure-Python hashing of every candidate vector) starves it, so budget checks renew the
        lease themselves. Raises ``ConsolidationLeaseLostError`` when the renewal is refused.
        """
        if self._closed or self._lease_lost:
            return
        if time.monotonic() - self._last_renewed < max(10, self.lease_seconds // 3):
            return
        try:
            renewed = await self._renew_once()
        except Exception:
            self._lease_lost = True
            logger.exception("Consolidation lease renewal failed for brain %s", self.brain_id)
            raise ConsolidationLeaseLostError(
                f"lease for brain {self.brain_id!r} could not be renewed"
            ) from None
        if not renewed:
            self._lease_lost = True
            logger.error("Consolidation lease lost for brain %s", self.brain_id)
            raise ConsolidationLeaseLostError(f"lease for brain {self.brain_id!r} was lost")

    async def _renew_lease(self) -> None:
        interval = max(10, self.lease_seconds // 3)
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                if not await self._renew_once():
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
