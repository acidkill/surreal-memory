"""Pure receipt validation shared by the deployed consolidation runner and observers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from typing import Any

MARKER = "HOURLY_RECEIPT "
STATE = "smem-consolidate-hourly-state"


class BlockedError(RuntimeError):
    """Evidence does not authorize another consolidation attempt."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def checkpoint_hash(cp: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(cp).encode()).hexdigest()


def _native_normalize(value: Any, key: str = "") -> Any:
    from surreal_memory.engine.consolidation_progress import stored_form

    value = stored_form(value)
    if isinstance(value, datetime):
        return (
            value.replace(tzinfo=UTC).isoformat()
            if value.tzinfo is None
            else value.astimezone(UTC).isoformat()
        )
    if isinstance(value, str) and key in {
        "reference_time",
        "started_at",
        "updated_at",
        "last_work_progress_at",
        "captured_at",
        "observed_at",
        "resolved_at",
    }:
        parsed = timestamp(value)
        if parsed is not None:
            return parsed.isoformat()
    if isinstance(value, dict):
        return {name: _native_normalize(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_native_normalize(item) for item in value]
    return value


def _native_digest(value: Any) -> str:
    encoded = json.dumps(
        _native_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def native_payload_hash(cp: dict[str, Any]) -> str:
    """Bind work and owner state, excluding proof and storage-owned transport fields."""
    return _native_digest(
        {
            key: value
            for key, value in cp.items()
            if key not in {"id", "updated_at", "native_handoff"}
        }
    )


def native_handoff_valid(cp: Any) -> bool:
    """Validate bounded cooperative-engine evidence, including its exact current payload."""
    if not isinstance(cp, dict):
        return False
    proof = cp.get("native_handoff")
    if not isinstance(proof, dict) or set(proof) != {
        "version",
        "lineage_root_sha256",
        "attempt_seq",
        "run_seq",
        "payload_sha256",
    }:
        return False
    if type(proof.get("version")) is not int or proof["version"] != 1:
        return False
    if any(type(proof.get(key)) is not int or proof[key] < 0 for key in ("attempt_seq", "run_seq")):
        return False
    if proof["attempt_seq"] < 1 or proof["run_seq"] >= proof["attempt_seq"]:
        return False
    if any(
        not isinstance(proof.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", proof[key]) is None
        for key in ("lineage_root_sha256", "payload_sha256")
    ):
        return False
    try:
        return bool(proof["payload_sha256"] == native_payload_hash(cp))
    except (TypeError, ValueError, OverflowError):
        return False


def _native_scope(cp: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cp.get(key)
        for key in (
            "brain_id",
            "options_fingerprint",
            "requested_strategies",
            "engine_version",
            "schema_version",
            "format_version",
        )
    }


def prepare_native_handoff(
    existing: dict[str, Any] | None, state: dict[str, Any], *, fresh: bool
) -> dict[str, Any]:
    """Carry lineage only through a verified resume or completed same-scope new run."""
    prior = None
    same_scope = existing is not None and _native_digest(_native_scope(existing)) == _native_digest(
        _native_scope(state)
    )
    if existing is not None:
        if "native_handoff" in existing:
            if not native_handoff_valid(existing):
                raise BlockedError("native handoff payload is stale or unsupported")
            prior = existing["native_handoff"]
        if fresh:
            if (
                existing.get("status") != "completed"
                or not existing.get("requested_strategies")
                or set(existing["requested_strategies"])
                != set(existing.get("completed_strategies", []))
                or checkpoint_errors(existing)
            ):
                raise BlockedError(
                    "new native lineage run requires a verified completed predecessor"
                )
            if existing.get("run_id") == state.get("run_id"):
                raise BlockedError("new native lineage run reused its predecessor identity")
        elif not same_scope or existing.get("run_id") != state.get("run_id"):
            raise BlockedError("native resume scope or identity changed")
    carry = existing is not None and same_scope
    proof: dict[str, Any] = {
        "version": 1,
        "lineage_root_sha256": prior["lineage_root_sha256"]
        if prior is not None and carry
        else checkpoint_hash(existing if existing is not None and carry else state),
        "attempt_seq": prior["attempt_seq"] + 1 if prior is not None and carry else 1,
        "run_seq": prior["run_seq"] + int(fresh)
        if prior is not None and carry
        else int(fresh and carry),
        "payload_sha256": native_payload_hash(state),
    }
    # Genesis can start after a legacy completed run, so count that predecessor attempt.
    if proof["run_seq"] >= proof["attempt_seq"]:
        proof["attempt_seq"] = proof["run_seq"] + 1
    return proof


def native_handoff_anchor(cp: Any) -> dict[str, Any] | None:
    """Keep a bounded managed anchor without copying manifests or inventing CLI results."""
    if not native_handoff_valid(cp):
        return None
    proof = cp["native_handoff"]
    states = cp.get("strategy_states") or {}
    return {
        **proof,
        "run_id": cp.get("run_id"),
        "scope_sha256": _native_digest(_native_scope(cp)),
        "reference_time": _native_normalize(cp.get("reference_time"), "reference_time"),
        "completed_states": {
            name: _native_digest(states.get(name)) for name in cp.get("completed_strategies", [])
        },
        "counters": _native_normalize(cp.get("counters") or {}),
    }


def native_handoff_matches(receipt: Any, cp: Any) -> bool:
    """Accept supported native continuation while retaining the actual managed result."""
    if not isinstance(receipt, dict) or not native_handoff_valid(cp):
        return False
    if (
        receipt.get("phase") != "terminal"
        or type(receipt.get("cli_rc")) is not int
        or receipt["cli_rc"] != (0 if receipt.get("status") == "completed" else 1)
        or receipt.get("status") not in {"paused", "completed"}
    ):
        return False
    try:
        clean_checkpoint(cp)
        proof = cp["native_handoff"]
        anchor = receipt.get("native_handoff_anchor")
        if anchor is None:
            return bool(
                proof["lineage_root_sha256"] == receipt.get("checkpoint_sha256")
                and (
                    (proof["run_seq"] == 0 and cp["run_id"] == receipt.get("run_id"))
                    or (proof["run_seq"] > 0 and cp["run_id"] != receipt.get("run_id"))
                )
            )
        if (
            not isinstance(anchor, dict)
            or anchor.get("version") != 1
            or proof["lineage_root_sha256"] != anchor.get("lineage_root_sha256")
            or type(anchor.get("attempt_seq")) is not int
            or type(anchor.get("run_seq")) is not int
            or proof["attempt_seq"] <= anchor["attempt_seq"]
            or proof["run_seq"] < anchor["run_seq"]
            or _native_digest(_native_scope(cp)) != anchor.get("scope_sha256")
        ):
            return False
        if proof["run_seq"] > anchor["run_seq"]:
            return bool(cp["run_id"] != anchor.get("run_id"))
        if cp["run_id"] != anchor.get("run_id") or _native_normalize(
            cp.get("reference_time"), "reference_time"
        ) != anchor.get("reference_time"):
            return False
        states = cp.get("strategy_states") or {}
        completed = set(cp.get("completed_strategies") or [])
        if not isinstance(anchor.get("completed_states"), dict) or any(
            name not in completed or _native_digest(states.get(name)) != digest
            for name, digest in anchor["completed_states"].items()
        ):
            return False
        old_counters = anchor.get("counters")
        counters = cp.get("counters") or {}
        if not isinstance(old_counters, dict):
            return False
        for strategy, old in old_counters.items():
            current = counters.get(strategy, {})
            if strategy in anchor["completed_states"] and _native_digest(current) != _native_digest(
                old
            ):
                return False
            if strategy == "semantic_link":
                for key in ("semantic_synapses_created", "semantic_synapses_skipped"):
                    if key in old and (
                        type(current.get(key)) is not int or current[key] < old[key]
                    ):
                        return False
        return True
    except (BlockedError, TypeError, ValueError, KeyError, OverflowError):
        return False


def checkpoint_errors(value: Any) -> bool:
    cp = value if isinstance(value, dict) else {}
    audit_paths = {
        ("strategy_states", "semantic_link", "recovery"),
        ("strategy_states", "semantic_link", "recovery_history", "[]"),
    }

    def active(item: Any, path: tuple[str, ...]) -> bool:
        if path == ("strategy_states", "semantic_link", "recovery") and item is None:
            return False
        if path in audit_paths:
            from surreal_memory.engine import consolidation_progress

            validate = getattr(
                consolidation_progress, "validate_semantic_source_generation_audit", None
            )
            return not (
                isinstance(item, dict)
                and type(item.get("version")) is int
                and item["version"] in {2, 3}
                and all(
                    isinstance(cp.get(key), str) and bool(cp[key])
                    for key in ("brain_id", "run_id", "options_fingerprint")
                )
                and callable(validate)
                and validate(
                    item,
                    brain_id=cp.get("brain_id"),
                    run_id=cp.get("run_id"),
                    options_fingerprint=cp.get("options_fingerprint"),
                    reference_time=cp.get("reference_time"),
                )
            )
        if path == ("strategy_states", "semantic_link", "recovery_history") and not isinstance(
            item, list
        ):
            return True
        if isinstance(item, dict):
            return any(
                (bool(child) and re.search("error|fail", str(key), re.I) is not None)
                or active(child, (*path, str(key)))
                for key, child in item.items()
            )
        if isinstance(item, list):
            return any(active(child, (*path, "[]")) for child in item)
        return False

    return active(value, ())


def clean_checkpoint(cp: Any) -> None:
    if not isinstance(cp, dict) or cp.get("brain_id") != "default" or not cp.get("run_id"):
        raise BlockedError("missing or wrong-brain checkpoint")
    if not cp.get("options_fingerprint"):
        raise BlockedError("checkpoint options evidence missing")
    if checkpoint_errors(cp):
        raise BlockedError("checkpoint contains error evidence")
    if cp.get("status") not in {"completed", "paused"}:
        raise BlockedError("checkpoint is not safely terminal")
    if cp["status"] == "completed" and set(cp.get("requested_strategies", [])) != set(
        cp.get("completed_strategies", [])
    ):
        raise BlockedError("completed checkpoint has unfinished strategies")


def _operator_recovery_seed(receipt: dict[str, Any], cp: dict[str, Any]) -> bool:
    if receipt.get("source") != "operator_recovery":
        return False
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    states = cp.get("strategy_states")
    semantic = states.get("semantic_link") if isinstance(states, dict) else None
    audit = semantic.get("recovery") if isinstance(semantic, dict) else None
    if not isinstance(semantic, dict) or not isinstance(audit, dict):
        raise BlockedError("operator recovery audit missing")
    pending = semantic.get("pending")
    if (
        not isinstance(pending, list)
        or len(pending) != 1
        or not isinstance(pending[0], str)
        or len(pending[0].encode()) > 512 * 1024
    ):
        raise BlockedError("operator recovery bootstrap manifest missing or oversized")
    try:
        manifest = json.loads(pending[0])
    except ValueError as exc:
        raise BlockedError("operator recovery bootstrap manifest malformed") from exc
    expected = {s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL}
    requested, completed = cp.get("requested_strategies"), cp.get("completed_strategies")
    counters, aggregate = semantic.get("counters"), cp.get("counters")
    history = semantic.get("recovery_history")
    if not (
        cp["status"] == "paused"
        and receipt.get("cli_rc") is None
        and receipt.get("next_action") == "resume"
        and receipt.get("run_id") == cp["run_id"]
        and receipt.get("options_fingerprint") == cp["options_fingerprint"]
        and timestamp(receipt.get("reference_time")) == timestamp(cp.get("reference_time"))
        and receipt.get("no_active_worker_verified") is True
        and receipt.get("no_active_lease_verified") is True
        and isinstance(receipt.get("resource_baseline"), list)
        and bool(receipt["resource_baseline"])
        and audit.get("kind") == "stale_semantic_source_restart"
        and audit.get("version") in {2, 3}
        and type(receipt.get("source_generation")) is int
        and receipt["source_generation"] == audit.get("generation")
        and receipt.get("recovery_audit_sha256") == checkpoint_hash(audit)
        and cp.get("current_strategy") == "semantic_link"
        and isinstance(requested, list)
        and isinstance(completed, list)
        and all(isinstance(s, str) for s in requested + completed)
        and len(requested) == len(expected)
        and len(completed) == len(expected) - 1
        and set(requested) == expected
        and set(completed) == expected - {"semantic_link"}
        and cp.get("phase") == semantic.get("phase") == "semantic_link_restart_pending"
        and cp.get("cursor") is None
        and semantic.get("cursor") is None
        and cp.get("owner_token") == audit.get("owner_token")
        and timestamp(cp.get("updated_at"))
        == timestamp(semantic.get("updated_at"))
        == timestamp(audit.get("recovered_at"))
        and isinstance(history, list)
        and bool(history)
        and canonical(history[-1]) == canonical(audit)
        and isinstance(counters, dict)
        and all(type(value) is int and value >= 0 for value in counters.values())
        and type(counters.get("semantic_link_failures")) is int
        and counters["semantic_link_failures"] == 0
        and isinstance(aggregate, dict)
        and aggregate.get("semantic_link") == counters == audit.get("prior_counters")
        and isinstance(manifest, dict)
        and manifest.get("kind") == "semantic_link_discovery"
        and type(manifest.get("version")) is int
        and manifest["version"] == 5
        and manifest.get("stage") == "neurons"
        and manifest.get("cursor") is None
        and checkpoint_hash(manifest) == audit.get("bootstrap_sha256")
        and manifest.get("source_state_ref") == audit.get("source_state_ref")
    ):
        raise BlockedError("operator recovery evidence does not match the live checkpoint")
    return True


def resource_rebaseline_seed(
    previous: dict[str, Any],
    cp: dict[str, Any],
    log: str,
    baseline: Any,
    previous_limit: int,
    memory_limit: int,
    approved_at: str,
) -> dict[str, Any]:
    """Authorize only the unchanged paused run after an explicit resource replacement."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    clean_checkpoint(cp)
    if not isinstance(log, str) or len(log.encode()) > 128 * 1024:
        raise BlockedError("resource rebaseline predecessor log missing or oversized")
    if (
        not isinstance(previous, dict)
        or previous.get("phase") != "terminal"
        or previous.get("run_id") != cp["run_id"]
        or cp["status"] != "paused"
    ):
        raise BlockedError("resource rebaseline requires a clean paused terminal predecessor")
    select_command(previous, cp, log)
    terminal(
        cp, 1, cp, True, "\n".join(line for line in log.splitlines() if not line.startswith(MARKER))
    )
    approved, finished = timestamp(approved_at), timestamp(previous.get("finished_at"))
    if approved is None or finished is None or approved < finished or approved > datetime.now(UTC):
        raise BlockedError("resource rebaseline approval time is invalid")
    expected = {s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL}
    requested, completed = cp.get("requested_strategies"), cp.get("completed_strategies")
    states = cp.get("strategy_states")
    semantic = states.get("semantic_link") if isinstance(states, dict) else None
    counters = semantic.get("counters") if isinstance(semantic, dict) else None
    if not (
        isinstance(requested, list)
        and isinstance(completed, list)
        and all(isinstance(s, str) for s in requested + completed)
        and len(requested) == len(expected)
        and set(requested) == expected
        and len(completed) == len(expected) - 1
        and set(completed) == expected - {"semantic_link"}
        and cp.get("current_strategy") == "semantic_link"
        and isinstance(states, dict)
        and expected - {"semantic_link"} <= set(states)
        and isinstance(counters, dict)
        and all(type(v) is int and v >= 0 for v in counters.values())
        and counters.get("semantic_link_failures") == 0
        and isinstance(cp.get("counters"), dict)
        and cp["counters"].get("semantic_link") == counters
        and timestamp(cp.get("reference_time")) is not None
    ):
        raise BlockedError("resource rebaseline checkpoint scope or retained effects are invalid")
    gib = 1024**3
    if not (
        type(previous_limit) is int
        and type(memory_limit) is int
        and previous_limit > 3 * gib
        and memory_limit > 3 * gib
        and previous_limit % gib == memory_limit % gib == 0
    ):
        raise BlockedError("resource rebaseline requires approved whole-GiB limits above reserve")
    previous_policy = previous.get("resource_policy_sha256")
    if previous_policy is not None and previous_policy != checkpoint_hash(
        {"memory_limit_bytes": previous_limit}
    ):
        raise BlockedError("resource rebaseline previous limit conflicts with terminal policy")

    def identity(value: Any, fresh: bool) -> tuple[str, list[list[Any]]]:
        if not (
            isinstance(value, list)
            and len(value) == 3
            and isinstance(value[0], list)
            and len(value[0]) == 2
            and isinstance(value[0][0], str)
            and value[0][0]
            and isinstance(value[0][1], str)
            and all(type(v) in {int, float} and v == 0 for v in value[1:])
        ):
            raise BlockedError("resource rebaseline identity or OOM evidence is invalid")
        try:
            containers = json.loads(value[0][1])
        except ValueError as exc:
            raise BlockedError("resource rebaseline container evidence is malformed") from exc
        if not (
            isinstance(containers, list)
            and containers
            and all(
                isinstance(c, list)
                and len(c) == 3
                and all(isinstance(v, str) and v for v in c[:2])
                and type(c[2]) is int
                and c[2] >= 0
                and (not fresh or c[2] == 0)
                for c in containers
            )
            and len({c[0] for c in containers}) == len(containers)
        ):
            raise BlockedError("resource rebaseline container restart evidence is invalid")
        return value[0][0], containers

    old_baseline = previous.get("resource_baseline")
    old_uid, old_containers = identity(old_baseline, False)
    new_baseline = json.loads(canonical(baseline))
    new_uid, new_containers = identity(new_baseline, True)
    if (
        old_uid == new_uid
        or {c[0] for c in old_containers} != {c[0] for c in new_containers}
        or {c[1] for c in old_containers} & {c[1] for c in new_containers}
    ):
        raise BlockedError("resource rebaseline requires a distinct replacement database")
    audit = {
        "kind": "planned_database_resource_restart",
        "version": 1,
        "previous_receipt": previous,
        "previous_log": log,
        "previous_receipt_sha256": checkpoint_hash(previous),
        "previous_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
        "previous_memory_limit_bytes": previous_limit,
        "memory_limit_bytes": memory_limit,
        "approved_at": approved_at,
        "retained_states_sha256": checkpoint_hash(states),
        "retained_counters_sha256": checkpoint_hash(cp["counters"]),
    }
    return {
        "phase": "bootstrap",
        "source": "operator_resource_rebaseline",
        "approved": True,
        "status": "paused",
        "cli_rc": None,
        "next_action": "resume",
        "observed_at": approved_at,
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp["reference_time"],
        "checkpoint_sha256": checkpoint_hash(cp),
        "resource_baseline": new_baseline,
        "resource_policy_sha256": checkpoint_hash({"memory_limit_bytes": memory_limit}),
        "no_active_worker_verified": True,
        "no_active_lease_verified": True,
        "resource_rebaseline": audit,
    }


def _operator_resource_rebaseline_seed(receipt: dict[str, Any], cp: dict[str, Any]) -> bool:
    if receipt.get("source") != "operator_resource_rebaseline":
        return False
    audit = receipt.get("resource_rebaseline")
    if not isinstance(audit, dict):
        raise BlockedError("resource rebaseline audit missing")
    try:
        expected = resource_rebaseline_seed(
            audit["previous_receipt"],
            cp,
            audit["previous_log"],
            receipt["resource_baseline"],
            audit["previous_memory_limit_bytes"],
            audit["memory_limit_bytes"],
            audit["approved_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BlockedError("resource rebaseline audit malformed") from exc
    if canonical(receipt) != canonical(expected):
        raise BlockedError("resource rebaseline audit or retained checkpoint diverged")
    return True


def resource_rebaseline_matches(receipt: Any, cp: Any) -> bool:
    """Let read-only observers validate the same distinct operator bootstrap contract."""
    if not isinstance(receipt, dict) or not isinstance(cp, dict):
        return False
    try:
        return receipt.get("phase") == "bootstrap" and _operator_resource_rebaseline_seed(
            receipt, cp
        )
    except (BlockedError, TypeError, ValueError, KeyError):
        return False


def native_handoff_bridge_seed(
    previous: dict[str, Any],
    previous_log: str,
    first_blocked: dict[str, Any],
    cp: dict[str, Any],
    *,
    approved_at: str,
    approved_checkpoint_sha256: str,
    approved_other_states_sha256: str,
) -> dict[str, Any]:
    """Bind a reviewed legacy gap once; never infer external CLI results or alter work."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    clean_checkpoint(cp)
    if (
        cp["status"] != "paused"
        or previous.get("phase") != "terminal"
        or previous.get("status") != "paused"
        or type(previous.get("cli_rc")) is not int
        or previous["cli_rc"] != 1
        or previous.get("run_id") != cp["run_id"]
        or checkpoint_hash(cp) != approved_checkpoint_sha256
        or first_blocked.get("phase") != "blocked"
        or first_blocked.get("reason") != "live checkpoint diverged from receipt"
        or {
            key: value
            for key, value in first_blocked.items()
            if key not in {"phase", "reason", "blocked_at"}
        }
        != {key: value for key, value in previous.items() if key != "phase"}
    ):
        raise BlockedError("native handoff bridge predecessor or approved snapshot differs")
    if not isinstance(previous_log, str) or len(previous_log.encode()) > 128 * 1024:
        raise BlockedError("native handoff bridge terminal log missing or oversized")
    markers = [line for line in previous_log.splitlines() if line.startswith(MARKER)]
    if markers != [MARKER + canonical(previous)] or cli_log_has_errors(
        "\n".join(line for line in previous_log.splitlines() if not line.startswith(MARKER)), cp
    ):
        raise BlockedError("native handoff bridge terminal log is not verified")
    approved, blocked, finished = (
        timestamp(approved_at),
        timestamp(first_blocked.get("blocked_at")),
        timestamp(previous.get("finished_at")),
    )
    if (
        approved is None
        or blocked is None
        or finished is None
        or not finished <= blocked <= approved <= datetime.now(UTC)
    ):
        raise BlockedError("native handoff bridge approval time is invalid")
    expected = {
        strategy.value
        for strategy in ConsolidationStrategy
        if strategy is not ConsolidationStrategy.ALL
    }
    states, counters = cp.get("strategy_states"), cp.get("counters")
    if (
        not isinstance(states, dict)
        or not isinstance(counters, dict)
        or set(cp.get("requested_strategies", [])) != expected
        or set(cp.get("completed_strategies", [])) != expected - {"semantic_link"}
        or cp.get("current_strategy") != "semantic_link"
        or checkpoint_hash({key: value for key, value in states.items() if key != "semantic_link"})
        != approved_other_states_sha256
        or counters.get("semantic_link", {}).get("semantic_link_failures") != 0
        or not previous.get("resource_baseline")
        or not isinstance(previous.get("resource_policy_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", previous["resource_policy_sha256"]) is None
    ):
        raise BlockedError("native handoff bridge retained work or resource evidence differs")
    audit = {
        "version": 1,
        "previous_receipt": previous,
        "previous_log": previous_log,
        "first_blocked_receipt": first_blocked,
        "previous_receipt_sha256": checkpoint_hash(previous),
        "previous_log_sha256": hashlib.sha256(previous_log.encode()).hexdigest(),
        "first_blocked_receipt_sha256": checkpoint_hash(first_blocked),
        "other_states_sha256": approved_other_states_sha256,
        "requested_strategies_sha256": checkpoint_hash(cp["requested_strategies"]),
        "completed_strategies_sha256": checkpoint_hash(cp["completed_strategies"]),
        "counters_sha256": checkpoint_hash(counters),
    }
    return {
        "phase": "bootstrap",
        "source": "operator_native_handoff_bridge",
        "approved": True,
        "status": "paused",
        "cli_rc": None,
        "next_action": "resume",
        "observed_at": approved_at,
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp.get("reference_time"),
        "checkpoint_sha256": approved_checkpoint_sha256,
        "resource_baseline": previous["resource_baseline"],
        "resource_policy_sha256": previous["resource_policy_sha256"],
        "native_handoff_bridge": audit,
    }


def _operator_native_handoff_bridge_seed(receipt: dict[str, Any], cp: dict[str, Any]) -> bool:
    if receipt.get("source") != "operator_native_handoff_bridge":
        return False
    audit = receipt.get("native_handoff_bridge")
    if not isinstance(audit, dict):
        raise BlockedError("native handoff bridge audit missing")
    try:
        expected = native_handoff_bridge_seed(
            audit["previous_receipt"],
            audit["previous_log"],
            audit["first_blocked_receipt"],
            cp,
            approved_at=receipt["observed_at"],
            approved_checkpoint_sha256=receipt["checkpoint_sha256"],
            approved_other_states_sha256=audit["other_states_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BlockedError("native handoff bridge audit malformed") from exc
    if canonical(expected) != canonical(receipt):
        raise BlockedError("native handoff bridge audit or retained checkpoint diverged")
    return True


def native_handoff_bridge_matches(receipt: Any, cp: Any) -> bool:
    """Let observers apply the same exact reviewed legacy bridge contract."""
    if not isinstance(receipt, dict) or not isinstance(cp, dict):
        return False
    try:
        return receipt.get("phase") == "bootstrap" and _operator_native_handoff_bridge_seed(
            receipt, cp
        )
    except (BlockedError, KeyError, TypeError, ValueError):
        return False


def _failed_runtime_source_ref(cp: dict[str, Any]) -> dict[str, Any]:
    pending = cp["strategy_states"]["semantic_link"].get("pending")
    if not isinstance(pending, list) or len(pending) != 1 or not isinstance(pending[0], str):
        raise BlockedError("failed runtime resume manifest missing")
    if len(pending[0].encode()) > 512 * 1024:
        raise BlockedError("failed runtime resume manifest oversized")
    manifest = json.loads(pending[0])
    if not isinstance(manifest, dict):
        raise BlockedError("failed runtime resume manifest malformed")
    if manifest.get("stage") == "validation":
        manifest = manifest.get("discovery")
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "semantic_link_discovery"
        or type(manifest.get("version")) is not int
        or manifest["version"] != 5
        or manifest.get("stage") != "similarity"
    ):
        raise BlockedError("failed runtime resume manifest malformed")
    ref = manifest.get("source_state_ref")
    if (
        not isinstance(ref, dict)
        or not isinstance(ref.get("state_id"), str)
        or not ref["state_id"]
        or type(ref.get("revision")) is not int
        or ref["revision"] < 1
    ):
        raise BlockedError("failed runtime resume source reference missing")
    return ref


def failed_runtime_resume_seed(
    previous: dict[str, Any],
    previous_log: str,
    first_blocked: dict[str, Any],
    cp: dict[str, Any],
    *,
    approval: dict[str, Any],
) -> dict[str, Any]:
    """Bind one reviewed backend-mismatch retry; retain the failed checkpoint verbatim."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    if not all(
        isinstance(item, dict) for item in (previous, first_blocked, cp, approval)
    ) or not isinstance(previous_log, str):
        raise BlockedError("failed runtime resume evidence malformed")
    expected = {s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL}
    if (
        cp.get("status") != "failed"
        or cp.get("last_error")
        != "semantic_link (progress coordination: ConsolidationProgressError)"
        or cp.get("phase") != "semantic_link_discovery_validation"
        or cp.get("current_strategy") != "semantic_link"
        or set(cp.get("requested_strategies", [])) != expected
        or set(cp.get("completed_strategies", [])) != expected - {"semantic_link"}
        or not native_handoff_valid(cp)
        or timestamp(cp.get("reference_time")) is None
    ):
        raise BlockedError("failed runtime resume checkpoint scope differs")
    # Classify the single permitted active error without changing the actual checkpoint.
    clean_checkpoint({**cp, "status": "paused", "last_error": None})
    states = cp["strategy_states"]
    counters = cp.get("counters")
    if (
        set(states) != expected
        or states["semantic_link"].get("phase") != cp["phase"]
        or any(states[name].get("phase") != "completed" for name in expected - {"semantic_link"})
        or not isinstance(counters, dict)
        or counters.get("semantic_link") != states["semantic_link"].get("counters")
        or counters.get("semantic_link", {}).get("semantic_link_failures") != 0
    ):
        raise BlockedError("failed runtime resume retained work differs")
    bindings = {
        "checkpoint_sha256": checkpoint_hash(cp),
        "previous_receipt_sha256": checkpoint_hash(previous),
        "previous_log_sha256": hashlib.sha256(previous_log.encode()).hexdigest(),
        "first_blocked_sha256": checkpoint_hash(first_blocked),
        "other_states_sha256": checkpoint_hash(
            {key: value for key, value in states.items() if key != "semantic_link"}
        ),
        "counters_sha256": checkpoint_hash(counters),
        "source_state_ref": _failed_runtime_source_ref(cp),
    }
    if any(approval.get(key) != value for key, value in bindings.items()):
        raise BlockedError("failed runtime resume approved evidence changed")
    failure = "semantic_link similarity backend changed after its checkpoint"
    if (
        not isinstance(previous_log, str)
        or len(previous_log.encode()) > 128 * 1024
        or "RuntimeError: " + failure not in previous_log.splitlines()
        or not previous_log.rstrip().endswith("HOURLY_BLOCKED live checkpoint reports failure")
        or MARKER in previous_log
        or re.search(r"(?i)(OOMKilled|lease.lost|timed.out)", previous_log)
        or any(
            re.match(
                r"^(?:[a-zA-Z_][a-zA-Z0-9_]*\.)*(?:[a-zA-Z0-9_]*Error|[a-zA-Z0-9_]*Exception):",
                line,
            )
            and line
            not in {
                "RuntimeError: " + failure,
                "surreal_memory.engine.consolidation_progress.ConsolidationProgressError: "
                "semantic_link discovery requires optimization or a safe restart: " + failure,
            }
            for line in previous_log.splitlines()
        )
        or previous != first_blocked
        or previous.get("phase") != "blocked"
        or previous.get("reason") != "live checkpoint reports failure"
        or previous.get("before_run_id") != cp["run_id"]
        or type(previous.get("cli_rc")) is not int
        or previous["cli_rc"] != 130
    ):
        raise BlockedError("failed runtime resume is not the reviewed backend failure")
    pod = approval.get("failed_pod")
    if (
        not isinstance(pod, dict)
        or set(pod) != {"pod_uid", "job_uid", "exit_code"}
        or type(pod["exit_code"]) is not int
        or pod["exit_code"] != 2
        or any(not pod.get(key) or pod[key] != previous.get(key) for key in ("pod_uid", "job_uid"))
    ):
        raise BlockedError("failed runtime resume failed Pod provenance differs")
    approved, blocked = (
        timestamp(approval.get("approved_at")),
        timestamp(previous.get("blocked_at")),
    )
    if approved is None or blocked is None or not blocked <= approved <= datetime.now(UTC):
        raise BlockedError("failed runtime resume approval time is invalid")
    runtime = approval.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("numpy_version") != "2.5.4"
        or runtime.get("selected_backend") != "numpy_float32:2.5.4"
        or approval.get("source_backend") != runtime["selected_backend"]
        or not isinstance(runtime.get("package_version"), str)
        or not runtime["package_version"]
        or re.fullmatch(r"3\.(?:1[2-9]|[2-9][0-9])\.\d+", str(runtime.get("python_version")))
        is None
        or re.fullmatch(r"[0-9a-f]{40}", str(runtime.get("source_sha"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(runtime.get("image_digest"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(approval.get("source_state_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(approval.get("resource_policy_sha256"))) is None
        or not isinstance(approval.get("source_token"), str)
        or not approval["source_token"]
        or approval["source_state_ref"].get("source_token") != approval["source_token"]
        or approval.get("no_active_worker_verified") is not True
        or approval.get("no_active_lease_verified") is not True
    ):
        raise BlockedError("failed runtime resume compatible runtime or safety proof missing")
    baseline = approval.get("resource_baseline")
    if (
        not isinstance(baseline, list)
        or len(baseline) != 3
        or not isinstance(baseline[0], list)
        or len(baseline[0]) != 2
        or not all(isinstance(item, str) and item for item in baseline[0])
        or any(type(item) not in {int, float} or item != 0 for item in baseline[1:])
    ):
        raise BlockedError("failed runtime resume fresh resource identity missing")
    containers = json.loads(baseline[0][1])
    if (
        not isinstance(containers, list)
        or not containers
        or not all(
            isinstance(item, list)
            and len(item) == 3
            and all(isinstance(value, str) and value for value in item[:2])
            and type(item[2]) is int
            and item[2] >= 0
            for item in containers
        )
    ):
        raise BlockedError("failed runtime resume container identity missing")
    return {
        "phase": "bootstrap",
        "source": "operator_failed_runtime_resume",
        "approved": True,
        "status": "failed",
        "cli_rc": None,
        "next_action": "resume",
        "observed_at": approval["approved_at"],
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp["reference_time"],
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "resource_baseline": baseline,
        "resource_policy_sha256": approval["resource_policy_sha256"],
        "failed_runtime_resume": {
            "version": 1,
            "previous_receipt": previous,
            "previous_log": previous_log,
            "first_blocked_receipt": first_blocked,
            "approval": approval,
        },
    }


def failed_runtime_resume_matches(receipt: Any, cp: Any) -> bool:
    """Share exact approval validation without treating failed work as healthy."""
    if not isinstance(receipt, dict) or not isinstance(cp, dict):
        return False
    if receipt.get("source") != "operator_failed_runtime_resume":
        return False
    try:
        audit = receipt["failed_runtime_resume"]
        if type(audit["version"]) is not int or audit["version"] != 1:
            return False
        expected = failed_runtime_resume_seed(
            audit["previous_receipt"],
            audit["previous_log"],
            audit["first_blocked_receipt"],
            cp,
            approval=audit["approval"],
        )
        return canonical(receipt) == canonical(expected)
    except (BlockedError, KeyError, TypeError, ValueError, AttributeError):
        return False


_CAPACITY_DATABASE_FIELDS = {
    "namespace",
    "pod_name",
    "pod_uid",
    "node_name",
    "image_id",
    "pod_spec_sha256",
    "replica_set_name",
    "replica_set_uid",
    "replica_set_spec_sha256",
    "deployment_name",
    "deployment_uid",
    "deployment_spec_sha256",
    "pvc_name",
    "pvc_uid",
    "pvc_spec_sha256",
    "pv_name",
    "pv_uid",
    "pv_spec_sha256",
}


def _capacity_resource_identity(value: Any, *, fresh: bool) -> tuple[str, list[Any]]:
    if not (
        isinstance(value, list)
        and len(value) == 3
        and isinstance(value[0], list)
        and len(value[0]) == 2
        and all(isinstance(v, str) and v for v in value[0])
        and all(type(v) in {int, float} and v == 0 for v in value[1:])
    ):
        raise BlockedError("capacity rebind resource identity invalid")
    try:
        containers = json.loads(value[0][1])
    except ValueError as exc:
        raise BlockedError("capacity rebind container proof malformed") from exc
    if not (
        isinstance(containers, list)
        and containers
        and all(
            isinstance(c, list)
            and len(c) == 3
            and all(isinstance(v, str) and v for v in c[:2])
            and type(c[2]) is int
            and c[2] >= 0
            and (not fresh or c[2] == 0)
            for c in containers
        )
        and len({c[0] for c in containers}) == len(containers)
    ):
        raise BlockedError("capacity rebind container identity invalid")
    return value[0][0], containers


def _capacity_rebind_evidence(previous: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    """Validate explicit reviewed archives; hashes bind inputs, not their capture authenticity."""
    rebind = approval.get("incarnation_rebind")
    if not isinstance(rebind, dict):
        raise BlockedError("capacity incarnation rebind proof missing")
    if canonical(rebind.get("previous_resource_baseline")) != canonical(
        previous.get("before_resources")
    ):
        raise BlockedError("capacity historical baseline changed")
    old_uid, old_containers = _capacity_resource_identity(
        previous.get("before_resources"), fresh=False
    )
    new_uid, new_containers = _capacity_resource_identity(
        approval.get("resource_baseline"), fresh=True
    )
    if (
        old_uid == new_uid
        or {c[0] for c in old_containers} != {c[0] for c in new_containers}
        or {c[1] for c in old_containers} & {c[1] for c in new_containers}
    ):
        raise BlockedError("capacity rebind requires a distinct database incarnation")
    current = rebind.get("database")
    historical = rebind.get("previous_database")
    if not isinstance(current, dict) or set(current) != _CAPACITY_DATABASE_FIELDS:
        raise BlockedError("capacity database binding missing or malformed")
    for key, value in current.items():
        if not isinstance(value, str) or not value:
            raise BlockedError("capacity database binding value invalid")
        if key.endswith("_sha256") and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise BlockedError("capacity database specification digest invalid")
        if key.endswith("_name") or key == "namespace":
            if len(value) > 253 or re.fullmatch(r"[a-z0-9][a-z0-9.-]*", value) is None:
                raise BlockedError("capacity database object name invalid")
    expected_historical = {
        "proof_kind",
        "observation_raw",
        "observation_sha256",
        "captured_at",
        "safety_raw",
        "safety_sha256",
        "unavailable_owner_mount_spec_acknowledged",
    }
    if (
        not isinstance(historical, dict)
        or set(historical) != expected_historical
        or historical.get("proof_kind") != "reviewed_observed_database_continuity"
        or historical.get("unavailable_owner_mount_spec_acknowledged") is not True
    ):
        raise BlockedError(
            "capacity observed database continuity requires explicit historical-gap review"
        )
    for kind in ("observation", "safety"):
        raw = historical[kind + "_raw"]
        if (
            not isinstance(raw, str)
            or len(raw.encode()) > 64 * 1024
            or hashlib.sha256(raw.encode()).hexdigest() != historical[kind + "_sha256"]
        ):
            raise BlockedError("capacity historical database observation bytes differ")
    observations = json.loads(historical["observation_raw"])
    safety = json.loads(historical["safety_raw"])
    if not isinstance(observations, list) or not 0 < len(observations) <= 100:
        raise BlockedError("capacity historical database observation list invalid")
    matches = [
        item for item in observations if isinstance(item, dict) and item.get("uid") == old_uid
    ]
    if len(matches) != 1:
        raise BlockedError("capacity historical database UID missing or ambiguous")
    observed = matches[0]
    if (
        set(observed) != {"name", "uid", "ready", "restarts", "imageIDs", "oom"}
        or current["pod_uid"] != new_uid
        or len(old_containers) != 1
        or old_containers[0][0] != "surrealdb"
        or not isinstance(observed.get("name"), str)
        or not observed["name"].startswith(current["replica_set_name"] + "-")
        or observed["name"] == current["replica_set_name"] + "-"
        or not current["replica_set_name"].startswith(current["deployment_name"] + "-")
        or not isinstance(observed.get("ready"), list)
        or len(observed["ready"]) != 1
        or observed["ready"][0] is not True
        or not isinstance(observed.get("restarts"), list)
        or len(observed["restarts"]) != 1
        or type(observed["restarts"][0]) is not int
        or observed["restarts"][0] != old_containers[0][2]
        or observed.get("imageIDs") != [current["image_id"]]
        or not isinstance(observed.get("oom"), list)
        or len(observed["oom"]) != 1
        or not isinstance(observed["oom"][0], str)
        or observed["oom"][0] == "OOMKilled"
    ):
        raise BlockedError("capacity historical observed database identity differs")
    # The legacy 'oom' observation is a termination reason, not a counter.
    # Counter/CID continuity comes from the separately captured Safety baseline.
    if (
        not isinstance(safety, dict)
        or safety.get("status") != "passed"
        or type(safety.get("memory_limit_bytes")) is not int
        or safety["memory_limit_bytes"] != approval.get("memory_limit_bytes")
        or canonical(safety.get("baseline")) != canonical(previous.get("before_resources"))
        or type(safety.get("active_jobs")) is not int
        or safety["active_jobs"] != 0
        or safety.get("cron_suspended") is not True
        or type(safety.get("at_epoch")) not in {int, float}
        or not math.isfinite(safety["at_epoch"])
        or not 0 < safety["at_epoch"] <= datetime.now(UTC).timestamp()
    ):
        raise BlockedError("capacity historical Safety proof differs")
    observed_at = timestamp(historical.get("captured_at"))
    historical_safety_at = datetime.fromtimestamp(safety["at_epoch"], UTC)
    blocked_at = timestamp(previous.get("blocked_at"))
    if (
        observed_at is None
        or blocked_at is None
        or not blocked_at <= observed_at <= historical_safety_at
    ):
        raise BlockedError("capacity historical observation chronology differs")
    brain = rebind.get("brain")
    if (
        not isinstance(brain, dict)
        or set(brain) != {"brain_id", "config_sha256"}
        or not isinstance(brain.get("brain_id"), str)
        or not brain["brain_id"]
        or re.fullmatch(r"[0-9a-f]{64}", str(brain.get("config_sha256"))) is None
    ):
        raise BlockedError("capacity default brain proof invalid")
    archive = rebind.get("archived_predecessor")
    if not isinstance(archive, dict):
        raise BlockedError("capacity archived predecessor missing")
    for key in ("pod", "job", "log"):
        raw = archive.get(key + "_raw") if key != "log" else archive.get("log")
        if (
            not isinstance(raw, str)
            or len(raw.encode()) > 64 * 1024
            or hashlib.sha256(raw.encode()).hexdigest() != archive.get(key + "_sha256")
        ):
            raise BlockedError("capacity archived predecessor bytes changed or oversized")
    try:
        pod, job = json.loads(archive["pod_raw"]), json.loads(archive["job_raw"])
    except ValueError as exc:
        raise BlockedError("capacity archived predecessor malformed") from exc
    if not isinstance(pod, dict) or not isinstance(job, dict):
        raise BlockedError("capacity archived predecessor object invalid")
    metadata, job_metadata = pod.get("metadata", {}), job.get("metadata", {})
    namespace = current["namespace"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(job_metadata, dict)
        or any(metadata.get(key) != previous.get("pod_" + key) for key in ("name", "uid"))
        or metadata.get("namespace") != namespace
        or job_metadata.get("namespace") != namespace
        or any(job_metadata.get(key) != previous.get("job_" + key) for key in ("name", "uid"))
        or not any(
            isinstance(o, dict)
            and o.get("kind") == "Job"
            and o.get("name") == previous["job_name"]
            and o.get("uid") == previous["job_uid"]
            for o in metadata.get("ownerReferences", [])
        )
    ):
        raise BlockedError("capacity archived predecessor owner identity differs")
    status = pod.get("status", {})
    job_status = job.get("status", {})
    containers = status.get("containerStatuses", []) if isinstance(status, dict) else []
    terminated = (
        containers[0].get("state", {}).get("terminated", {})
        if len(containers) == 1 and isinstance(containers[0], dict)
        else {}
    )
    if (
        status.get("phase") != "Failed"
        or len(containers) != 1
        or type(terminated.get("exitCode")) is not int
        or terminated["exitCode"] != 2
        or terminated.get("reason") == "OOMKilled"
        or type(containers[0].get("restartCount")) is not int
        or containers[0]["restartCount"] != 0
        or job_status.get("failed") != 1
        or job_status.get("active", 0)
        or job_status.get("succeeded", 0)
        or not any(
            c.get("type") == "Failed" and c.get("status") == "True"
            for c in job_status.get("conditions", [])
        )
    ):
        raise BlockedError("capacity archived predecessor is not failed wrapper2")
    incident = rebind.get("incident")
    if not isinstance(incident, dict) or checkpoint_hash(incident) != rebind.get("incident_sha256"):
        raise BlockedError("capacity reviewed incident proof differs")
    event = incident.get("eviction_event", {})
    ref = event.get("involvedObject", {}) if isinstance(event, dict) else {}
    captured, not_ready, evicted, approved, finished = (
        timestamp(archive.get("captured_at")),
        timestamp(incident.get("node_not_ready_at")),
        timestamp(event.get("firstTimestamp")),
        timestamp(approval.get("approved_at")),
        timestamp(terminated.get("finishedAt")),
    )
    job_started = timestamp(job_status.get("startTime"))
    if job_started is None or (finished is not None and job_started > finished):
        raise BlockedError("capacity historical failed Job start time invalid")
    if not_ready is None or historical_safety_at >= not_ready:
        raise BlockedError("capacity historical Safety proof is not before the incident")
    if any(value is None for value in (captured, not_ready, evicted, approved, finished)):
        raise BlockedError("capacity incident timestamps missing")
    assert captured is not None
    assert not_ready is not None
    assert evicted is not None
    assert approved is not None
    assert finished is not None
    if (
        not isinstance(event, dict)
        or event.get("reason") != "TaintManagerEviction"
        or not event.get("metadata", {}).get("uid")
        or event.get("source", {}).get("component") != "taint-eviction-controller"
        or ref.get("kind") != "Pod"
        or ref.get("namespace") != namespace
        or ref.get("name") != previous["pod_name"]
        or ref.get("uid") not in (None, previous["pod_uid"])
        or incident.get("node_name") != pod.get("spec", {}).get("nodeName")
        or incident.get("node_name") != current["node_name"]
        or not (finished <= captured < not_ready <= evicted <= approved)
    ):
        raise BlockedError("capacity archive capture or incident corroboration invalid")
    return rebind


def capacity_rebind_evidence(previous: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on malformed reviewed input without exposing archived contents."""
    try:
        return _capacity_rebind_evidence(previous, approval)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise BlockedError("capacity incarnation evidence malformed") from exc


def capacity_resume_seed(
    previous_raw: str,
    previous_log: str,
    cp: dict[str, Any],
    *,
    approval: dict[str, Any],
    observation_raw: str | None = None,
    observation_log: str | None = None,
) -> dict[str, Any]:
    """Bind one explicit reviewed retry of an actual blocked capacity/CLI130 stop."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    if (
        not isinstance(previous_raw, str)
        or not isinstance(previous_log, str)
        or len(previous_raw.encode()) > 128 * 1024
        or len(previous_log.encode()) > 128 * 1024
        or not isinstance(approval, dict)
    ):
        raise BlockedError("capacity resume evidence malformed or oversized")
    try:
        previous = json.loads(previous_raw)
    except ValueError as exc:
        raise BlockedError("capacity resume predecessor is malformed") from exc
    version = approval.get("capacity_resume_version", 1)
    if (
        type(version) is not int
        or version not in {1, 2}
        or (version == 1 and "incarnation_rebind" in approval)
    ):
        raise BlockedError("capacity admission version is unsupported")
    clean_checkpoint(cp)
    completed = {"dedup", "learn_habits", "process_reasoning_traces", "process_tool_events"}
    expected = {s.value for s in ConsolidationStrategy if s is not ConsolidationStrategy.ALL}
    states = cp.get("strategy_states")
    prune = states.get("prune") if isinstance(states, dict) else None
    if (
        cp.get("status") != "paused"
        or cp.get("current_strategy") != "prune"
        or cp.get("phase") != "synapse_pending"
        or set(cp.get("requested_strategies", [])) != expected
        or len(cp.get("requested_strategies", [])) != len(expected)
        or set(cp.get("completed_strategies", [])) != completed
        or len(cp.get("completed_strategies", [])) != len(completed)
        or not native_handoff_valid(cp)
        or timestamp(cp.get("reference_time")) is None
        or not isinstance(cp.get("owner_token"), str)
        or not cp["owner_token"]
        or not isinstance(states, dict)
        or set(states) != completed | {"prune"}
        or any(
            not isinstance(states[name], dict) or states[name].get("phase") != "completed"
            for name in completed
        )
        or not isinstance(prune, dict)
        or prune.get("phase") != "synapse_pending"
        or not isinstance(prune.get("pending"), list)
        or not 0 < len(prune["pending"]) <= 2000
        or not all(isinstance(sid, str) and sid for sid in prune["pending"])
        or not isinstance(cp.get("counters"), dict)
        or cp["counters"].get("prune") != prune.get("counters")
    ):
        raise BlockedError("capacity resume retained prune checkpoint differs")
    capacity = previous.get("capacity_stop_reason") if isinstance(previous, dict) else None
    disagreement = "CLI exit disagrees with terminal checkpoint"
    if (
        not isinstance(previous, dict)
        or previous.get("phase") != "blocked"
        or type(previous.get("cli_rc")) is not int
        or previous["cli_rc"] != 130
        or previous.get("before_run_id") != cp["run_id"]
        or capacity
        not in {"database memory exceeds safety threshold", "node free memory below safety floor"}
        or previous.get("reason") not in {disagreement, capacity}
        or (previous.get("reason") == capacity and previous.get("secondary_reason") != disagreement)
        or (previous.get("reason") == disagreement and previous.get("secondary_reason") is not None)
        or MARKER in previous_log
    ):
        raise BlockedError("capacity resume is not the reviewed blocked CLI130 stop")
    suffix = "HOURLY_BLOCKED " + previous["reason"] + "\n"
    if previous.get("secondary_reason"):
        suffix = "HOURLY_SECONDARY_BLOCKED " + disagreement + "\n" + suffix
    if not previous_log.endswith(suffix):
        raise BlockedError("capacity resume predecessor decision log differs")
    cli_output = previous_log[: -len(suffix)]
    if (
        hashlib.sha256(cli_output.encode()).hexdigest() != previous.get("cli_output_sha256")
        or previous.get("cli_output_tail") != cli_output[-4096:]
        or cli_log_has_errors(cli_output, cp)
    ):
        raise BlockedError("capacity resume CLI evidence differs or reports another failure")
    bindings = {
        "checkpoint_sha256": checkpoint_hash(cp),
        "previous_receipt_raw_sha256": hashlib.sha256(previous_raw.encode()).hexdigest(),
        "previous_log_sha256": hashlib.sha256(previous_log.encode()).hexdigest(),
        "first_blocked_sha256": checkpoint_hash(previous),
        "observation_raw_sha256": (
            hashlib.sha256(observation_raw.encode()).hexdigest()
            if isinstance(observation_raw, str)
            else None
        ),
        "observation_log_sha256": (
            hashlib.sha256(observation_log.encode()).hexdigest()
            if isinstance(observation_log, str)
            else None
        ),
        "completed_states_sha256": checkpoint_hash({name: states[name] for name in completed}),
        "prune_state_sha256": checkpoint_hash(prune),
        "counters_sha256": checkpoint_hash(cp["counters"]),
        "native_handoff_sha256": checkpoint_hash(cp["native_handoff"]),
        **{
            key: cp[key]
            for key in ("run_id", "owner_token", "options_fingerprint", "reference_time")
        },
    }
    if any(canonical(approval.get(key)) != canonical(value) for key, value in bindings.items()):
        raise BlockedError("capacity resume approved evidence changed")
    identity = approval.get("failed_pod")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"pod_name", "pod_uid", "job_name", "job_uid", "exit_code"}
        or type(identity.get("exit_code")) is not int
        or identity["exit_code"] != 2
        or any(
            not identity.get(key) or identity[key] != previous.get(key)
            for key in ("pod_name", "pod_uid", "job_name", "job_uid")
        )
    ):
        raise BlockedError("capacity resume failed Pod provenance differs")
    approved, blocked = (
        timestamp(approval.get("approved_at")),
        timestamp(previous.get("blocked_at")),
    )
    if approved is None or blocked is None or not blocked <= approved <= datetime.now(UTC):
        raise BlockedError("capacity resume approval time is invalid")
    runtime = approval.get("runtime")
    if (
        not isinstance(runtime, dict)
        or re.fullmatch(r"[0-9a-f]{40}", str(runtime.get("source_sha"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(runtime.get("image_digest"))) is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(runtime.get(key))) is None
            for key in ("runner_sha256", "receipts_sha256")
        )
        or any(
            not isinstance(runtime.get(key), str) or not runtime[key]
            for key in ("python_version", "numpy_version", "package_version")
        )
        or approval.get("reviewed_source_sha") != runtime["source_sha"]
        or any(
            approval.get(key) is not True
            for key in (
                "source_fix_reviewed",
                "resource_root_cause_addressed",
                "no_active_worker_verified",
                "no_active_lease_verified",
            )
        )
    ):
        raise BlockedError("capacity resume reviewed runtime or safety proof missing")
    limit = approval.get("memory_limit_bytes")
    policy = checkpoint_hash({"memory_limit_bytes": limit})
    baseline = approval.get("resource_baseline")
    if (
        type(limit) is not int
        or limit <= 3 * 1024**3
        or limit % 1024**3
        or approval.get("resource_policy_sha256") != policy
        or approval.get("previous_resource_policy_sha256") != policy
        or previous.get("resource_policy_sha256", policy) != policy
        or (version == 1 and canonical(baseline) != canonical(previous.get("before_resources")))
        or not isinstance(baseline, list)
        or len(baseline) != 3
        or not isinstance(baseline[0], list)
        or len(baseline[0]) != 2
        or not all(isinstance(value, str) and value for value in baseline[0])
        or any(type(value) not in {int, float} or value != 0 for value in baseline[1:])
    ):
        raise BlockedError("capacity resume database incarnation or resource policy differs")
    try:
        containers = json.loads(baseline[0][1])
    except ValueError as exc:
        raise BlockedError("capacity resume database container proof malformed") from exc
    if (
        not isinstance(containers, list)
        or not containers
        or not all(
            isinstance(item, list)
            and len(item) == 3
            and all(isinstance(value, str) and value for value in item[:2])
            and type(item[2]) is int
            and item[2] >= 0
            for item in containers
        )
        or len({item[0] for item in containers}) != len(containers)
    ):
        raise BlockedError("capacity resume database container proof differs")
    if version == 2:
        rebind = capacity_rebind_evidence(previous, approval)
        if rebind["archived_predecessor"]["log"] != previous_log:
            raise BlockedError("capacity archived log differs from predecessor")
    if observation_raw is not None:
        try:
            observation = json.loads(observation_raw)
        except (TypeError, ValueError) as exc:
            raise BlockedError("capacity resume observation malformed") from exc
        if (
            not isinstance(observation, dict)
            or not isinstance(observation_log, str)
            or len(observation_log.encode()) > 128 * 1024
            or observation.get("outcome") != "blocked"
            or observation.get("reason") != previous["reason"]
            or "HOURLY_OBSERVATION " + canonical(observation) not in observation_log.splitlines()
            or MARKER + canonical(previous) not in observation_log.splitlines()
        ):
            raise BlockedError("capacity resume observation proof differs")
    elif observation_log is not None:
        raise BlockedError("capacity resume observation is missing")
    return {
        "phase": "bootstrap",
        "source": "operator_capacity_resume",
        "approved": True,
        "status": "paused",
        "cli_rc": None,
        "next_action": "resume",
        "observed_at": approval["approved_at"],
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp["reference_time"],
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "resource_baseline": baseline,
        "resource_policy_sha256": policy,
        "capacity_resume": {
            "version": version,
            "previous_receipt_raw": previous_raw,
            "previous_log": previous_log,
            "approval": approval,
            "observation_raw": observation_raw,
            "observation_log": observation_log,
        },
    }


def capacity_resume_matches(receipt: Any, cp: Any) -> bool:
    """Validate capacity admission without treating the historical failure as success."""
    if not isinstance(receipt, dict) or receipt.get("source") != "operator_capacity_resume":
        return False
    try:
        audit = receipt["capacity_resume"]
        if (
            type(audit["version"]) is not int
            or audit["version"] not in {1, 2}
            or audit["version"] != audit["approval"].get("capacity_resume_version", 1)
        ):
            return False
        expected = capacity_resume_seed(
            audit["previous_receipt_raw"],
            audit["previous_log"],
            cp,
            approval=audit["approval"],
            observation_raw=audit["observation_raw"],
            observation_log=audit["observation_log"],
        )
        return canonical(expected) == canonical(receipt)
    except (BlockedError, KeyError, TypeError, ValueError, AttributeError):
        return False


def lease_loss_resume_seed(
    previous_raw: str,
    previous_log: str,
    cp: dict[str, Any],
    *,
    approval: dict[str, Any],
    previous_intent_raw: str | None = None,
) -> dict[str, Any]:
    """Bind one reviewed interrupted atomic-prune attempt without rewriting its outcome."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    if (
        not isinstance(previous_raw, str)
        or not isinstance(previous_log, str)
        or max(len(previous_raw.encode()), len(previous_log.encode())) > 128 * 1024
        or not isinstance(approval, dict)
        or type(approval.get("lease_loss_resume_version")) is not int
        or approval["lease_loss_resume_version"] not in {1, 2}
    ):
        raise BlockedError("lease loss resume evidence malformed or oversized")
    version = approval["lease_loss_resume_version"]
    if version == 1 and previous_intent_raw is not None:
        raise BlockedError("lease loss resume v1 does not accept a prior intent")
    try:
        previous = json.loads(previous_raw)
    except ValueError as exc:
        raise BlockedError("lease loss resume predecessor malformed") from exc
    if not isinstance(cp, dict) or cp.get("status") != "running":
        raise BlockedError("lease loss resume requires a retained running checkpoint")
    # This validates a projection only; the actual checkpoint is never changed.
    clean_checkpoint({**cp, "status": "paused"})
    expected = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    requested, completed = cp.get("requested_strategies"), cp.get("completed_strategies")
    states = cp.get("strategy_states")
    prune = states.get("prune") if isinstance(states, dict) else None
    if (
        not isinstance(requested, list)
        or len(requested) != len(expected)
        or not all(isinstance(name, str) and name in expected for name in requested)
        or set(requested) != expected
        or not isinstance(completed, list)
        or not all(isinstance(name, str) and name in expected for name in completed)
        or len(set(completed)) != len(completed)
        or "prune" in completed
        or cp.get("current_strategy") != "prune"
        or cp.get("phase") != "synapse_pending"
        or not native_handoff_valid(cp)
        or timestamp(cp.get("reference_time")) is None
        or not isinstance(cp.get("owner_token"), str)
        or not cp["owner_token"]
        or not isinstance(states, dict)
        or set(states) != set(completed) | {"prune"}
        or any(
            not isinstance(states[name], dict) or states[name].get("phase") != "completed"
            for name in completed
        )
        or not isinstance(prune, dict)
        or prune.get("phase") != "synapse_pending"
        or not isinstance(prune.get("pending"), list)
        or not 0 < len(prune["pending"]) <= 2000
        or not all(isinstance(sid, str) and sid for sid in prune["pending"])
        or not isinstance(prune.get("counters"), dict)
        or type(prune["counters"].get("prune_atomic_journal")) is not int
        or prune["counters"]["prune_atomic_journal"] != 1
        or not isinstance(cp.get("counters"), dict)
        or cp["counters"].get("prune") != prune["counters"]
    ):
        raise BlockedError("lease loss resume retained atomic prune scope differs")
    reason = (
        "engine lease absent or lost"
        if version == 1
        else "failed runtime resume native claim differs"
    )
    expected_cli_rc = 130 if version == 1 else 1
    suffix = "HOURLY_BLOCKED " + reason + "\n"
    if (
        not isinstance(previous, dict)
        or previous.get("phase") != "blocked"
        or previous.get("reason") != reason
        or previous.get("secondary_reason") is not None
        or previous.get("capacity_stop_reason") is not None
        or type(previous.get("cli_rc")) is not int
        or previous["cli_rc"] != expected_cli_rc
        or previous.get("before_run_id") != cp["run_id"]
        or MARKER in previous_log
        or not previous_log.endswith(suffix)
    ):
        raise BlockedError(
            f"lease loss resume is not the reviewed blocked CLI{expected_cli_rc} attempt"
        )
    output = previous_log[: -len(suffix)]
    if hashlib.sha256(output.encode()).hexdigest() != previous.get("cli_output_sha256") or output[
        -4096:
    ] != previous.get("cli_output_tail"):
        raise BlockedError("lease loss resume predecessor CLI evidence differs")
    if version == 1:
        if cli_log_has_errors(output, cp):
            raise BlockedError("lease loss resume predecessor CLI evidence differs")
    else:
        message = (
            "Consolidation stopped safely in prune: lease-fenced page outcome is unconfirmed; "
            "this session cannot replay or checkpoint"
        )
        required_lines = (
            "Consolidation engine outcome: lease_lost; " + message,
            "  " + message,
            "  Consolidation status: lease_lost",
            "  Stages failed: prune (progress coordination: ConsolidationCheckpointUnconfirmedError)",
        )
        diagnostic_line = "  Health delta unavailable: diagnostic snapshot failed"
        lines = output.splitlines()
        if (
            any(lines.count(line) != 1 for line in required_lines)
            or any(
                sum(line.lstrip().startswith(prefix) for line in lines) != 1
                for prefix in ("Consolidation engine outcome:", "Consolidation status:")
            )
            or lines.count(diagnostic_line) > 1
            or cli_log_has_errors(
                "\n".join(
                    line for line in lines if line not in required_lines and line != diagnostic_line
                ),
                cp,
            )
            or approval.get("known_engine_status") != "lease_lost"
            or approval.get("known_engine_error") != "ConsolidationCheckpointUnconfirmedError"
            or "original_engine_outcome_unknown" in approval
        ):
            raise BlockedError("lease loss resume v2 native failure evidence differs")
        if (
            not isinstance(previous_intent_raw, str)
            or len(previous_intent_raw.encode()) > 128 * 1024
            or re.fullmatch(r"[0-9a-f]{64}", str(approval.get("prior_admission_sha256"))) is None
        ):
            raise BlockedError("lease loss resume v2 prior intent is missing or malformed")
        try:
            prior = json.loads(previous_intent_raw)
        except ValueError as exc:
            raise BlockedError("lease loss resume v2 prior intent is malformed") from exc
        prior_audit = prior.get("lease_loss_resume") if isinstance(prior, dict) else None
        prior_approval = prior_audit.get("approval") if isinstance(prior_audit, dict) else None
        if (
            not isinstance(prior, dict)
            or previous_intent_raw != canonical(prior)
            or checkpoint_hash(prior) != approval["prior_admission_sha256"]
            or previous.get("lease_loss_resume_sha256") != approval["prior_admission_sha256"]
            or any(
                previous.get(key) is not None
                for key in ("failed_runtime_resume_sha256", "capacity_resume_sha256")
            )
            or prior.get("phase") != "bootstrap"
            or prior.get("source") != "operator_lease_loss_resume"
            or prior.get("approved") is not True
            or prior.get("status") != "running"
            or prior.get("cli_rc") is not None
            or prior.get("next_action") != "resume"
            or any(prior.get(key) != cp[key] for key in ("run_id", "options_fingerprint"))
            or timestamp(prior.get("reference_time")) != timestamp(cp["reference_time"])
            or re.fullmatch(r"[0-9a-f]{64}", str(prior.get("checkpoint_sha256"))) is None
            or prior.get("checkpoint_sha256") != previous.get("before_sha256")
            or prior.get("resource_policy_sha256") != approval.get("resource_policy_sha256")
            or canonical(prior.get("resource_baseline"))
            != canonical(approval.get("resource_baseline"))
            or not isinstance(prior_audit, dict)
            or type(prior_audit.get("version")) is not int
            or prior_audit["version"] != 1
            or not isinstance(prior_approval, dict)
            or type(prior_approval.get("lease_loss_resume_version")) is not int
            or prior_approval["lease_loss_resume_version"] != 1
            or prior_approval.get("original_engine_outcome_unknown") is not True
            or prior_approval.get("source_fix_reviewed") is not True
            or prior_approval.get("checkpoint_sha256") != prior["checkpoint_sha256"]
            or prior_approval.get("approved_at") != prior.get("observed_at")
            or any(
                prior_approval.get(key) != approval.get(key)
                for key in ("brain_id", "brain_config_sha256")
            )
        ):
            raise BlockedError("lease loss resume v2 prior approved intent chain differs")
    bindings = {
        "checkpoint_sha256": checkpoint_hash(cp),
        "previous_receipt_raw_sha256": hashlib.sha256(previous_raw.encode()).hexdigest(),
        "previous_log_sha256": hashlib.sha256(previous_log.encode()).hexdigest(),
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
    }
    if any(approval.get(key) != value for key, value in bindings.items()):
        raise BlockedError("lease loss resume approved evidence changed")
    pod = approval.get("failed_pod")
    if (
        not isinstance(pod, dict)
        or set(pod) != {"pod_name", "pod_uid", "job_name", "job_uid", "exit_code"}
        or type(pod.get("exit_code")) is not int
        or pod["exit_code"] != 2
        or any(
            not pod.get(key) or pod[key] != previous.get(key)
            for key in ("pod_name", "pod_uid", "job_name", "job_uid")
        )
    ):
        raise BlockedError("lease loss resume predecessor identity differs")
    approved, blocked = (
        timestamp(approval.get("approved_at")),
        timestamp(previous.get("blocked_at")),
    )
    if approved is None or blocked is None or not blocked <= approved <= datetime.now(UTC):
        raise BlockedError("lease loss resume approval time is invalid")
    runtime = approval.get("runtime")
    if (
        not isinstance(runtime, dict)
        or re.fullmatch(r"[0-9a-f]{40}", str(runtime.get("source_sha"))) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(runtime.get("image_digest"))) is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(runtime.get(key))) is None
            for key in ("runner_sha256", "receipts_sha256")
        )
        or any(
            not isinstance(runtime.get(key), str) or not runtime[key]
            for key in ("python_version", "numpy_version", "package_version")
        )
        or approval.get("reviewed_source_sha") != runtime["source_sha"]
        or any(
            approval.get(key) is not True
            for key in (
                "source_fix_reviewed",
                *(
                    ("original_engine_outcome_unknown",)
                    if version == 1
                    else ("transaction_commit_outcome_unknown", "original_cause_unknown")
                ),
                "no_active_worker_verified",
                "no_active_lease_verified",
            )
        )
        or not isinstance(approval.get("brain_id"), str)
        or not approval["brain_id"]
        or re.fullmatch(r"[0-9a-f]{64}", str(approval.get("brain_config_sha256"))) is None
    ):
        raise BlockedError("lease loss resume reviewed runtime proof missing")
    limit = approval.get("memory_limit_bytes")
    baseline = approval.get("resource_baseline")
    policy = checkpoint_hash({"memory_limit_bytes": limit})
    _capacity_resource_identity(baseline, fresh=False)
    if (
        type(limit) is not int
        or limit <= 3 * 1024**3
        or limit % 1024**3
        or approval.get("resource_policy_sha256") != policy
        or ("resource_policy_sha256" in previous and previous["resource_policy_sha256"] != policy)
        or canonical(baseline) != canonical(previous.get("before_resources"))
    ):
        raise BlockedError("lease loss resume resource incarnation or policy differs")
    return {
        "phase": "bootstrap",
        "source": "operator_lease_loss_resume",
        "approved": True,
        "status": "running",
        "cli_rc": None,
        "next_action": "resume",
        "observed_at": approval["approved_at"],
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp["reference_time"],
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "resource_baseline": baseline,
        "resource_policy_sha256": policy,
        "lease_loss_resume": {
            "version": version,
            **({"previous_intent_raw": previous_intent_raw} if version == 2 else {}),
            "previous_receipt_raw": previous_raw,
            "previous_log": previous_log,
            "approval": approval,
        },
    }


def lease_loss_resume_matches(receipt: Any, cp: Any) -> bool:
    if not isinstance(receipt, dict) or receipt.get("source") != "operator_lease_loss_resume":
        return False
    try:
        audit = receipt["lease_loss_resume"]
        if type(audit["version"]) is not int or audit["version"] not in {1, 2}:
            return False
        expected = lease_loss_resume_seed(
            audit["previous_receipt_raw"],
            audit["previous_log"],
            cp,
            approval=audit["approval"],
            previous_intent_raw=audit.get("previous_intent_raw"),
        )
        return canonical(receipt) == canonical(expected)
    except (BlockedError, KeyError, TypeError, ValueError, AttributeError):
        return False


def hourly_options_fingerprint() -> str:
    """Fingerprint the unchanged default ALL CLI configuration, excluding runtime limits."""
    from dataclasses import asdict

    from surreal_memory.engine.consolidation import ConsolidationConfig, ConsolidationStrategy
    from surreal_memory.engine.consolidation_progress import options_fingerprint

    return options_fingerprint(
        {
            "requested_strategies": sorted(
                item.value
                for item in ConsolidationStrategy
                if item is not ConsolidationStrategy.ALL
            ),
            "config": asdict(ConsolidationConfig()),
            "dream_decay_multiplier": 10.0,
            "tier_config": None,
        }
    )


def supervision_timeout_resume_seed(
    previous_raw: str,
    previous_log: str,
    cp: dict[str, Any],
    *,
    approval: dict[str, Any],
    observation_raw: str | None = None,
) -> dict[str, Any]:
    """Approve one exact paused ALL after a proved DB-Pod supervision read stop."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    if any(
        not isinstance(raw, str) or len(raw.encode()) > 128 * 1024
        for raw in (previous_raw, previous_log)
    ) or not isinstance(approval, dict):
        raise BlockedError("supervision timeout evidence malformed or oversized")
    if observation_raw is not None and (
        not isinstance(observation_raw, str) or len(observation_raw.encode()) > 128 * 1024
    ):
        raise BlockedError("supervision timeout observation malformed or oversized")
    try:
        previous = json.loads(previous_raw)
        if observation_raw is not None:
            if not isinstance(json.loads(observation_raw), dict):
                raise ValueError("observation is not an object")
    except (ValueError, TypeError) as exc:
        raise BlockedError("supervision timeout evidence is malformed") from exc
    clean_checkpoint(cp)
    expected = sorted(
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    )
    completed, states, current = (
        cp.get("completed_strategies"),
        cp.get("strategy_states"),
        cp.get("current_strategy"),
    )
    if (
        cp.get("status") != "paused"
        or cp.get("requested_strategies") != expected
        or not isinstance(completed, list)
        or any(not isinstance(name, str) or name not in expected for name in completed)
        or len(set(completed)) != len(completed)
        or current not in expected
        or current in completed
        or not isinstance(states, dict)
        or set(states) != set(completed) | {current}
        or any(
            not isinstance(states[name], dict) or states[name].get("phase") != "completed"
            for name in completed
        )
        or not isinstance(states.get(current), dict)
        or states[current].get("phase") != cp.get("phase")
        or not isinstance(cp.get("phase"), str)
        or not cp["phase"]
        or not isinstance(cp.get("counters"), dict)
        or not native_handoff_valid(cp)
        or timestamp(cp.get("reference_time")) is None
        or type(cp.get("work_progress_seq")) is not int
        or cp["work_progress_seq"] < 1
    ):
        raise BlockedError("supervision timeout requires the exact clean paused ALL checkpoint")
    if (
        not isinstance(previous, dict)
        or previous.get("phase") != "blocked"
        or type(previous.get("cli_rc")) is not int
        or previous["cli_rc"] != 130
    ):
        raise BlockedError("supervision timeout requires the retained blocked CLI130")
    reason = previous.get("reason")
    typed_reason = "database pod supervision read timed out"
    if reason not in {"TimeoutError", typed_reason}:
        raise BlockedError("supervision timeout stop classification differs")
    suffix = "HOURLY_BLOCKED " + reason + "\n"
    if not previous_log.endswith(suffix) or MARKER in previous_log:
        raise BlockedError("supervision timeout decision log differs")
    output = previous_log[: -len(suffix)]
    if (
        hashlib.sha256(output.encode()).hexdigest() != previous.get("cli_output_sha256")
        or previous.get("cli_output_tail") != output[-4096:]
        or cli_log_has_errors(output, cp)
    ):
        raise BlockedError("supervision timeout CLI evidence differs")
    baseline = approval.get("resource_baseline")
    try:
        db_uid, _ = _capacity_resource_identity(baseline, fresh=True)
    except BlockedError as exc:
        raise BlockedError("supervision timeout resource identity invalid") from exc
    if canonical(previous.get("before_resources")) != canonical(baseline):
        raise BlockedError("supervision timeout database incarnation or OOM changed")
    incident = approval.get("incident_evidence")
    if not isinstance(incident, dict):
        raise BlockedError("supervision timeout incident evidence missing")
    path = incident.get("path")
    if (
        incident.get("version") != 1
        or type(incident.get("version")) is not int
        or incident.get("guard") != "database_pod"
        or incident.get("method") != "GET"
        or type(incident.get("timeout_seconds")) is not int
        or incident["timeout_seconds"] != 15
        or not isinstance(path, str)
        or re.fullmatch(r"/api/v1/namespaces/[a-z0-9][-a-z0-9]*/pods/[a-z0-9][-a-z0-9.]*", path)
        is None
        or incident.get("db_uid") != db_uid
        or any(
            not isinstance(previous.get(key), str) or not previous[key]
            for key in ("job_uid", "pod_uid", "job_name", "pod_name")
        )
        or any(incident.get(key) != previous.get(key) for key in ("job_uid", "pod_uid"))
    ):
        raise BlockedError("supervision timeout endpoint or source binding differs")
    observed, blocked, approved = (
        timestamp(incident.get("observed_at")),
        timestamp(previous.get("blocked_at")),
        timestamp(approval.get("approved_at")),
    )
    if (
        observed is None
        or blocked is None
        or approved is None
        or not observed <= blocked <= approved <= datetime.now(UTC)
    ):
        raise BlockedError("supervision timeout approval chronology differs")
    if reason == typed_reason:
        proof = previous.get("supervision_timeout")
        common = ("version", "guard", "method", "path", "timeout_seconds", "observed_at", "db_uid")
        if (
            not isinstance(proof, dict)
            or any(proof.get(key) != incident.get(key) for key in common)
            or proof.get("child_was_running") is not True
            or proof.get("stop_graceful") is not True
            or type(proof.get("actual_child_rc")) is not int
            or proof["actual_child_rc"] != 130
        ):
            raise BlockedError("supervision timeout typed interruption proof differs")
    else:
        fault = incident.get("legacy_fault_record")
        if (
            previous.get("supervision_timeout") is not None
            or not isinstance(fault, dict)
            or fault.get("method") != "GET"
            or fault.get("path") != path
            or not isinstance(fault.get("raw_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", fault["raw_sha256"]) is None
            or incident.get("legacy_fault_record_sha256") != checkpoint_hash(fault)
            or incident.get("previous_runner_sha256")
            != "7d66fb20ad6b935cf848c2819a4313896c3eeeb37045586b9759481e5bd3b47e"
            or incident.get("previous_getter_sha256")
            != "d1b0de663e958624fbaf86fdfd3b1e388a776e1b70c4df89abc05d3ff48af8f9"
        ):
            raise BlockedError("legacy supervision timeout requires reviewed fault evidence")
        started, finished = timestamp(fault.get("started_at")), timestamp(fault.get("finished_at"))
        invocation_started = timestamp(previous.get("started_at"))
        if (
            started is None
            or finished is None
            or invocation_started is None
            or not invocation_started <= started
            or not 15 <= (finished - started).total_seconds() <= 60
            or not finished <= observed
            or not 0 <= (blocked - observed).total_seconds() <= 20
        ):
            raise BlockedError("legacy supervision timeout fault chronology differs")

    def raw_hash(raw: str | None) -> str | None:
        return hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None

    bindings = {
        "previous_receipt_raw_sha256": raw_hash(previous_raw),
        "previous_log_sha256": raw_hash(previous_log),
        "observation_raw_sha256": raw_hash(observation_raw),
        "checkpoint_sha256": checkpoint_hash(cp),
        "native_checkpoint_sha256": _native_digest(cp),
        "identity_sha256": _native_digest(_native_scope(cp)),
        "states_sha256": checkpoint_hash(states),
        "counters_sha256": checkpoint_hash(cp["counters"]),
        "incident_evidence_sha256": checkpoint_hash(incident),
    }
    limit, runtime = approval.get("memory_limit_bytes"), approval.get("runtime")
    if (
        approval.get("version") != 1
        or type(approval.get("version")) is not int
        or approval.get("kind") != "supervision_timeout_resume"
        or any(approval.get(key) != value for key, value in bindings.items())
        or type(limit) is not int
        or limit <= 3 * 1024**3
        or limit % (1024**3)
        or not isinstance(runtime, dict)
        or any(
            not isinstance(runtime.get(key), str) or not runtime[key]
            for key in ("python_version", "numpy_version", "package_version")
        )
        or any(
            not isinstance(runtime.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", runtime[key]) is None
            for key in ("runner_sha256", "receipts_sha256")
        )
        or not isinstance(runtime.get("image_digest"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["image_digest"]) is None
        or not approval.get("brain_uuid")
        or not isinstance(approval.get("brain_config_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", approval["brain_config_sha256"]) is None
        or not isinstance(approval.get("previous_image_digest"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", approval["previous_image_digest"]) is None
    ):
        raise BlockedError("supervision timeout approval or runtime binding differs")
    parents = [
        (key, prefix)
        for key, prefix in (
            ("completed_semantic_handoff_sha256", "completed_semantic_handoff_"),
            ("supervision_timeout_resume_sha256", "supervision_timeout_resume_"),
        )
        if key in previous
    ]
    terminal_parent = previous.get("terminal_predecessor")
    parent_archive = None
    if (
        not isinstance(previous.get("before_run_id"), str)
        or not previous["before_run_id"]
        or not isinstance(previous.get("before_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", previous["before_sha256"]) is None
    ):
        raise BlockedError("supervision timeout predecessor checkpoint binding missing")
    if terminal_parent is not None:
        if (
            reason != typed_reason
            or parents
            or not isinstance(terminal_parent, dict)
            or type(terminal_parent.get("version")) is not int
            or terminal_parent["version"] != 1
            or terminal_parent.get("run_id") != previous["before_run_id"]
            or terminal_parent.get("before_sha256") != previous["before_sha256"]
            or terminal_parent.get("options_fingerprint") != cp["options_fingerprint"]
            or terminal_parent.get("scope_sha256") != checkpoint_hash(_native_scope(cp))
            or terminal_parent.get("status") not in {"paused", "completed"}
            or type(terminal_parent.get("cli_rc")) is not int
            or terminal_parent["cli_rc"] != (0 if terminal_parent["status"] == "completed" else 1)
            or any(
                not isinstance(terminal_parent.get(key), str) or not terminal_parent[key]
                for key in ("job_name", "job_uid", "pod_name", "pod_uid")
            )
            or any(
                not isinstance(terminal_parent.get(key), str)
                or re.fullmatch(r"[0-9a-f]{64}", terminal_parent[key]) is None
                for key in ("receipt_sha256", "checkpoint_sha256")
            )
        ):
            raise BlockedError("supervision timeout terminal predecessor differs")
        parent_ref, current_ref = (
            timestamp(terminal_parent.get("reference_time")),
            timestamp(cp.get("reference_time")),
        )
        finished, started = (
            timestamp(terminal_parent.get("finished_at")),
            timestamp(previous.get("started_at")),
        )
        if (
            parent_ref is None
            or current_ref is None
            or finished is None
            or started is None
            or not finished <= started <= blocked
            or (
                terminal_parent["status"] == "paused"
                and (terminal_parent["run_id"] != cp["run_id"] or parent_ref != current_ref)
            )
            or (
                terminal_parent["status"] == "completed"
                and (terminal_parent["run_id"] == cp["run_id"] or parent_ref >= current_ref)
            )
        ):
            raise BlockedError("supervision timeout terminal predecessor lineage differs")
    else:
        if len(parents) != 1:
            raise BlockedError("supervision timeout predecessor archive binding missing")
        parent_key, prefix = parents[0]
        if (
            not isinstance(previous[parent_key], str)
            or re.fullmatch(r"[0-9a-f]{64}", previous[parent_key]) is None
        ):
            raise BlockedError("supervision timeout predecessor archive digest invalid")
        parent_archive = prefix + previous[parent_key] + ".json"
    return {
        "phase": "bootstrap",
        "source": "operator_supervision_timeout_resume",
        "approved": True,
        "status": "paused",
        "cli_rc": None,
        "next_action": "resume",
        "observed_at": approval["approved_at"],
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp["reference_time"],
        "checkpoint_sha256": checkpoint_hash(cp),
        "resource_baseline": baseline,
        "resource_policy_sha256": checkpoint_hash({"memory_limit_bytes": limit}),
        "supervision_timeout_resume": {
            "previous_raw": previous_raw,
            "previous_log": previous_log,
            "observation_raw": observation_raw,
            "predecessor_archive_key": parent_archive,
            "terminal_predecessor": terminal_parent,
            "approval": approval,
        },
    }


def supervision_timeout_resume_matches(receipt: Any, cp: Any) -> bool:
    if (
        not isinstance(receipt, dict)
        or receipt.get("source") != "operator_supervision_timeout_resume"
    ):
        return False
    try:
        audit = receipt["supervision_timeout_resume"]
        expected = supervision_timeout_resume_seed(
            audit["previous_raw"],
            audit["previous_log"],
            cp,
            approval=audit["approval"],
            observation_raw=audit.get("observation_raw"),
        )
        return canonical(receipt) == canonical(expected)
    except (BlockedError, KeyError, ValueError, TypeError, AttributeError):
        return False


def _completed_semantic_identity(cp: dict[str, Any]) -> dict[str, Any]:
    clean_checkpoint(cp)
    if (
        cp["status"] != "completed"
        or cp.get("requested_strategies") != ["semantic_link"]
        or cp.get("completed_strategies") != ["semantic_link"]
        or not native_handoff_valid(cp)
        or timestamp(cp.get("reference_time")) is None
    ):
        raise BlockedError("handoff requires a proved completed semantic-only checkpoint")
    return {
        key: cp.get(key)
        for key in (
            "brain_id",
            "run_id",
            "options_fingerprint",
            "reference_time",
            "schema_version",
            "engine_version",
            "format_version",
        )
    }


def _semantic_final_output(last: dict[str, Any], log: str) -> str:
    """Recover one exact final CLI output by its digest, not earlier attempt logs."""
    tail, digest = last.get("output_tail"), last.get("output_sha256")
    if (
        not isinstance(tail, str)
        or not tail
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise BlockedError("semantic final CLI output evidence is missing")
    position = log.rfind(tail)
    if position < 0:
        raise BlockedError("semantic final CLI tail is absent from retained log")
    end = position + len(tail)
    starts = [0, *(match.end() for match in re.finditer("\n", log[:position]))]
    if len(starts) > 4096:
        raise BlockedError("semantic final CLI log has too many candidate boundaries")
    for start in reversed(starts):
        output = log[start:end]
        if hashlib.sha256(output.encode()).hexdigest() == digest:
            return output
    raise BlockedError("semantic final CLI output digest does not match retained log")


def completed_semantic_handoff_seed(
    previous_raw: str,
    cp: dict[str, Any],
    *,
    approval: dict[str, Any],
    semantic_receipt_raw: str,
    semantic_log: str,
    observation_raw: str | None = None,
) -> dict[str, Any]:
    """Archive a reviewed closed semantic run as predecessor, never as an ALL success."""
    for raw in (previous_raw, semantic_receipt_raw, semantic_log, observation_raw):
        if raw is not None and (not isinstance(raw, str) or len(raw.encode()) > 128 * 1024):
            raise BlockedError("completed semantic handoff evidence missing or oversized")
    try:
        previous = json.loads(previous_raw)
        semantic = json.loads(semantic_receipt_raw)
    except (ValueError, TypeError) as exc:
        raise BlockedError("completed semantic handoff evidence is malformed") from exc
    identity = _completed_semantic_identity(cp)
    cp_hash = _native_digest(cp)

    def raw_hash(raw: str | None) -> str | None:
        return hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None

    if not isinstance(approval, dict) or (
        type(approval.get("version")) is not int
        or approval["version"] != 1
        or approval.get("kind") != "completed_semantic_to_fresh_all"
        or approval.get("previous_receipt_sha256") != raw_hash(previous_raw)
        or approval.get("observation_sha256") != raw_hash(observation_raw)
        or approval.get("checkpoint_sha256") != cp_hash
        or approval.get("identity_sha256") != _native_digest(identity)
        or approval.get("semantic_receipt_sha256") != raw_hash(semantic_receipt_raw)
        or approval.get("semantic_log_sha256") != raw_hash(semantic_log)
    ):
        raise BlockedError("completed semantic handoff approval or evidence changed")
    if not isinstance(previous, dict) or previous.get("phase") not in {"blocked", "terminal"}:
        raise BlockedError("hourly predecessor is not a reviewed historical receipt")
    if not isinstance(semantic, dict):
        raise BlockedError("semantic completion receipt is malformed")
    attempts = semantic.get("attempts")
    last = attempts[-1] if isinstance(attempts, list) and attempts else None
    if (
        semantic.get("mode") != "semantic_link"
        or semantic.get("status") != "completed"
        or semantic.get("phase") != "terminal"
        or semantic.get("checkpoint_sha256") != cp_hash
        or _native_digest(semantic.get("identity")) != _native_digest(identity)
        or not isinstance(last, dict)
        or type(last.get("cli_rc")) is not int
        or last["cli_rc"] != 0
        or last.get("status") != "completed"
        or last.get("checkpoint_sha256") != cp_hash
        or not isinstance(last.get("output_tail"), str)
        or not last["output_tail"]
        or last["output_tail"] not in semantic_log
        or "[frozen candidate coverage complete]" not in semantic_log
    ):
        raise BlockedError("semantic CLI-zero or full-coverage evidence is missing")
    final_output = _semantic_final_output(last, semantic_log)
    if "[frozen candidate coverage complete]" not in final_output or cli_log_has_errors(
        final_output, cp
    ):
        raise BlockedError("semantic final CLI coverage or outcome evidence is invalid")
    approved, finished = (
        timestamp(approval.get("approved_at")),
        timestamp(semantic.get("finished_at")),
    )
    if approved is None or finished is None or not finished <= approved <= datetime.now(UTC):
        raise BlockedError("completed semantic handoff approval time is invalid")
    limit = approval.get("memory_limit_bytes")
    baseline = approval.get("resource_baseline")
    source = approval.get("semantic_source")
    required_source = {
        "state_name",
        "state_uid",
        "job_name",
        "job_uid",
        "pod_name",
        "pod_uid",
        "image_digest",
    }
    if (
        type(limit) is not int
        or limit <= 3 * 1024**3
        or limit % (1024**3)
        or not isinstance(baseline, list)
        or not baseline
        or not isinstance(approval.get("runtime"), dict)
        or not isinstance(source, dict)
        or set(source) != required_source
        or any(not isinstance(value, str) or not value for value in source.values())
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source["image_digest"]) is None
        or not approval.get("brain_uuid")
        or not approval.get("state_resource_version")
        or not isinstance(approval.get("brain_config_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", approval["brain_config_sha256"]) is None
    ):
        raise BlockedError("completed semantic handoff runtime/resource binding is missing")
    return {
        "phase": "bootstrap",
        "source": "operator_completed_semantic_handoff",
        "approved": True,
        "status": "completed",
        "cli_rc": None,
        "next_action": "fresh_all",
        "observed_at": approval["approved_at"],
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "reference_time": cp["reference_time"],
        "checkpoint_sha256": checkpoint_hash(cp),
        "expected_all_options_fingerprint": hourly_options_fingerprint(),
        "resource_baseline": baseline,
        "resource_policy_sha256": checkpoint_hash({"memory_limit_bytes": limit}),
        "completed_semantic_handoff": {
            "previous_raw": previous_raw,
            "observation_raw": observation_raw,
            "semantic_receipt_raw": semantic_receipt_raw,
            "semantic_log": semantic_log,
            "approval": approval,
        },
    }


def completed_semantic_handoff_matches(receipt: Any, cp: Any) -> bool:
    if not isinstance(receipt, dict) or not isinstance(cp, dict):
        return False
    try:
        audit = receipt["completed_semantic_handoff"]
        expected = completed_semantic_handoff_seed(
            audit["previous_raw"],
            cp,
            approval=audit["approval"],
            semantic_receipt_raw=audit["semantic_receipt_raw"],
            semantic_log=audit["semantic_log"],
            observation_raw=audit["observation_raw"],
        )
        return canonical(receipt) == canonical(expected)
    except (BlockedError, KeyError, TypeError, ValueError):
        return False


def select_command(receipt: dict[str, Any], cp: dict[str, Any], log: str | None) -> list[str]:
    if receipt.get("source") == "operator_supervision_timeout_resume":
        if not supervision_timeout_resume_matches(receipt, cp):
            raise BlockedError("supervision timeout approval or checkpoint diverged")
        return ["smem", "consolidate", "--brain", "default", "--resume"]
    if receipt.get("source") == "operator_completed_semantic_handoff":
        if not completed_semantic_handoff_matches(receipt, cp):
            raise BlockedError("completed semantic handoff approval or checkpoint diverged")
        return ["smem", "consolidate", "--brain", "default"]
    if receipt.get("source") == "operator_lease_loss_resume":
        if not lease_loss_resume_matches(receipt, cp):
            raise BlockedError("lease loss resume approval or checkpoint diverged")
        return ["smem", "consolidate", "--brain", "default", "--resume"]
    if receipt.get("source") == "operator_capacity_resume":
        if not capacity_resume_matches(receipt, cp):
            raise BlockedError("capacity resume approval or checkpoint diverged")
        return ["smem", "consolidate", "--brain", "default", "--resume"]
    if receipt.get("source") == "operator_failed_runtime_resume":
        if not failed_runtime_resume_matches(receipt, cp):
            raise BlockedError("failed runtime resume approval or checkpoint diverged")
        return ["smem", "consolidate", "--brain", "default", "--resume"]
    clean_checkpoint(cp)
    if receipt.get("phase") not in {"bootstrap", "terminal"}:
        raise BlockedError("previous attempt is interrupted, blocked, or unknown")
    advanced = receipt.get("checkpoint_sha256") != checkpoint_hash(cp)
    if advanced and not native_handoff_matches(receipt, cp):
        raise BlockedError("live checkpoint diverged from receipt")
    recovered = False
    if receipt.get("phase") == "bootstrap":
        recovered = (
            _operator_recovery_seed(receipt, cp)
            or _operator_resource_rebaseline_seed(receipt, cp)
            or _operator_native_handoff_bridge_seed(receipt, cp)
        )
        interrupted = (
            receipt.get("source") == "user_interrupted"
            and cp["status"] == "paused"
            and receipt.get("cli_rc") is None
            and receipt.get("next_action") == "resume"
        )
        if receipt.get("approved") is not True or not (
            cp["status"] == "completed" or interrupted or recovered
        ):
            raise BlockedError("bootstrap needs approved terminal or verified paused checkpoint")
    else:
        if not log:
            raise BlockedError("previous job logs missing")
        markers = [line[len(MARKER) :] for line in log.splitlines() if line.startswith(MARKER)]
        if len(markers) != 1 or json.loads(markers[0]) != receipt:
            raise BlockedError("previous terminal log does not match receipt")
    status = cp["status"]
    interrupted_seed = (
        receipt.get("phase") == "bootstrap" and receipt.get("source") == "user_interrupted"
    )
    if not advanced and (
        receipt.get("status") != status
        or (
            not (interrupted_seed or recovered)
            and (
                type(receipt.get("cli_rc")) is not int
                or receipt["cli_rc"] != (0 if status == "completed" else 1)
            )
        )
    ):
        raise BlockedError("exit status and checkpoint disagree")
    command = ["smem", "consolidate", "--brain", "default"]
    if status == "paused":
        command.append("--resume")
    return command


def cli_log_has_errors(output: str, cp: dict[str, Any]) -> bool:
    """Allow the CLI's bounded-stage report only for a verified cooperative pause."""
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    incomplete = set(cp.get("requested_strategies", [])) - set(cp.get("completed_strategies", []))
    known = {item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL}
    lines = output.splitlines()
    advisory = "Health delta unavailable: diagnostic snapshot failed"
    if (
        cp.get("status") == "completed"
        and cp.get("requested_strategies")
        and not incomplete
        and sum(line.strip() == advisory for line in lines) == 1
    ):
        lines = [line for line in lines if line.strip() != advisory]
        output = "\n".join(lines)
    if (
        cp.get("status") == "paused"
        and lines.count("  Consolidation status: paused") == 1
        and sum(line.startswith("  Stages timed out: ") for line in lines) == 1
    ):
        retained = []
        for line in lines:
            match = re.fullmatch(r"  Stages timed out: ([a-z_]+(?:, [a-z_]+)*)", line)
            if match:
                stages = match[1].split(", ")
                if len(stages) == len(set(stages)) and set(stages).issubset(incomplete & known):
                    continue
            retained.append(line)
        output = "\n".join(retained)
    return (
        re.search(
            r"(?im)(traceback|\berror\b|\bfailed\b|lease.lost|timed.out|TimeoutError|OOMKilled)",
            output,
        )
        is not None
    )


def terminal(
    cp: dict[str, Any],
    rc: int,
    before: dict[str, Any],
    resume: bool,
    output: str,
    *,
    fresh_all_options_fingerprint: str | None = None,
) -> None:
    clean_checkpoint(cp)
    if type(rc) is not int or rc != (0 if cp["status"] == "completed" else 1):
        raise BlockedError("CLI exit disagrees with terminal checkpoint")
    if resume and cp["run_id"] != before["run_id"]:
        raise BlockedError("resume changed run identity")
    if not resume and cp["run_id"] == before["run_id"]:
        raise BlockedError("ordinary command did not create a fresh run")
    if fresh_all_options_fingerprint is not None:
        from surreal_memory.engine.consolidation import ConsolidationStrategy

        _completed_semantic_identity(before)
        requested = cp.get("requested_strategies")
        completed = cp.get("completed_strategies")
        before_reference = timestamp(before.get("reference_time"))
        after_reference = timestamp(cp.get("reference_time"))
        expected = {
            item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
        }
        if (
            resume
            or fresh_all_options_fingerprint != hourly_options_fingerprint()
            or cp.get("options_fingerprint") != fresh_all_options_fingerprint
            or not isinstance(requested, list)
            or len(requested) != len(expected)
            or not all(isinstance(item, str) for item in requested)
            or set(requested) != expected
            or not isinstance(completed, list)
            or not all(isinstance(item, str) for item in completed)
            or len(completed) != len(set(completed))
            or not set(completed) <= expected
            or not native_handoff_valid(cp)
            or before_reference is None
            or after_reference is None
            or after_reference <= before_reference
            or any(
                cp.get(key) != before.get(key)
                for key in ("brain_id", "schema_version", "engine_version", "format_version")
            )
        ):
            raise BlockedError("fresh ALL handoff scope or expected configuration changed")
    elif cp.get("options_fingerprint") != before.get("options_fingerprint"):
        raise BlockedError("options changed or auto-tier ambiguity")
    if resume and (
        _native_digest(_native_scope(cp)) != _native_digest(_native_scope(before))
        or _native_normalize(cp.get("reference_time"), "reference_time")
        != _native_normalize(before.get("reference_time"), "reference_time")
    ):
        raise BlockedError("resume scope or reference changed")
    if cli_log_has_errors(output, cp):
        raise BlockedError("CLI log contains error evidence")


def timestamp(value: Any) -> datetime | None:
    """Interpret legacy naive timestamps as UTC, independent of the observer's timezone."""
    if not isinstance(value, (str, datetime)):
        return None
    try:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    except ValueError:
        return None


def terminal_receipt_matches(receipt: Any, cp: Any) -> bool:
    """Validate existing runner evidence without inventing a missing CLI result or log."""
    if not isinstance(receipt, dict) or not isinstance(cp, dict):
        return False
    try:
        clean_checkpoint(cp)
    except (BlockedError, TypeError, ValueError):
        return False
    return (
        receipt.get("phase") == "terminal"
        and receipt.get("status") == cp.get("status")
        and type(receipt.get("cli_rc")) is int
        and receipt["cli_rc"] == (0 if cp["status"] == "completed" else 1)
        and receipt.get("run_id") == cp.get("run_id")
        and receipt.get("checkpoint_sha256") == checkpoint_hash(cp)
        and timestamp(receipt.get("finished_at")) is not None
    )


def all_strategies_completed(cp: Any) -> bool:
    from surreal_memory.engine.consolidation import ConsolidationStrategy

    expected = {
        item.value for item in ConsolidationStrategy if item is not ConsolidationStrategy.ALL
    }
    requested = cp.get("requested_strategies") if isinstance(cp, dict) else None
    completed = cp.get("completed_strategies") if isinstance(cp, dict) else None
    return (
        isinstance(requested, list)
        and isinstance(completed, list)
        and all(isinstance(item, str) for item in requested + completed)
        and len(requested) == len(completed) == len(expected)
        and set(requested) == set(completed) == expected
    )


def confirmed_success(receipt: Any, cp: Any) -> dict[str, Any] | None:
    """Produce a historical success proof only from a matching actual CLI-zero terminal."""
    if not terminal_receipt_matches(receipt, cp) or cp["status"] != "completed":
        return None
    if not all_strategies_completed(cp):
        return None
    return {
        "version": 1,
        "cli_rc": receipt["cli_rc"],
        "run_id": cp["run_id"],
        "options_fingerprint": cp["options_fingerprint"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "requested_strategies": list(cp["requested_strategies"]),
        "completed_strategies": list(cp["completed_strategies"]),
        "finished_at": receipt["finished_at"],
    }


def success_timestamp(proof: Any) -> datetime | None:
    """Read only complete runner-owned proofs; legacy absence stays unknown."""
    if not isinstance(proof, dict) or not all_strategies_completed(proof):
        return None
    if (
        type(proof.get("version")) is not int
        or proof["version"] != 1
        or type(proof.get("cli_rc")) is not int
        or proof["cli_rc"] != 0
        or not isinstance(proof.get("run_id"), str)
        or not proof["run_id"]
        or not isinstance(proof.get("options_fingerprint"), str)
        or not proof["options_fingerprint"]
        or not isinstance(proof.get("checkpoint_sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", proof["checkpoint_sha256"]) is None
    ):
        return None
    return timestamp(proof.get("finished_at"))
