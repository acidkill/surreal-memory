"""Pure receipt validation shared by the deployed consolidation runner and observers."""

from __future__ import annotations

import hashlib
import json
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
    if markers != [MARKER + canonical(previous)] or re.search(
        r"(?im)(traceback|\berror\b|\bfailed\b|lease.lost|timed.out)",
        "\n".join(line for line in previous_log.splitlines() if not line.startswith(MARKER)),
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


def select_command(receipt: dict[str, Any], cp: dict[str, Any], log: str | None) -> list[str]:
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


def terminal(
    cp: dict[str, Any], rc: int, before: dict[str, Any], resume: bool, output: str
) -> None:
    clean_checkpoint(cp)
    if type(rc) is not int or rc != (0 if cp["status"] == "completed" else 1):
        raise BlockedError("CLI exit disagrees with terminal checkpoint")
    if resume and cp["run_id"] != before["run_id"]:
        raise BlockedError("resume changed run identity")
    if not resume and cp["run_id"] == before["run_id"]:
        raise BlockedError("ordinary command did not create a fresh run")
    if cp.get("options_fingerprint") != before.get("options_fingerprint"):
        raise BlockedError("options changed or auto-tier ambiguity")
    if re.search(r"(?im)(traceback|\berror\b|\bfailed\b|lease.lost|timed.out)", output):
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
