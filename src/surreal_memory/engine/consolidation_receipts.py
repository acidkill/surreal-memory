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


def checkpoint_errors(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (bool(child) and re.search("error|fail", str(key), re.I) is not None)
            or checkpoint_errors(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(checkpoint_errors(child) for child in value)
    return False


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


def select_command(receipt: dict[str, Any], cp: dict[str, Any], log: str | None) -> list[str]:
    clean_checkpoint(cp)
    if receipt.get("phase") not in {"bootstrap", "terminal"}:
        raise BlockedError("previous attempt is interrupted, blocked, or unknown")
    if receipt.get("checkpoint_sha256") != checkpoint_hash(cp):
        raise BlockedError("live checkpoint diverged from receipt")
    if receipt.get("phase") == "bootstrap":
        interrupted = (
            receipt.get("source") == "user_interrupted"
            and cp["status"] == "paused"
            and receipt.get("cli_rc") is None
            and receipt.get("next_action") == "resume"
        )
        if receipt.get("approved") is not True or not (cp["status"] == "completed" or interrupted):
            raise BlockedError(
                "bootstrap needs approved terminal or user-interrupted paused checkpoint"
            )
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
    if receipt.get("status") != status or (
        not interrupted_seed and receipt.get("cli_rc") != (0 if status == "completed" else 1)
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
    if rc != (0 if cp["status"] == "completed" else 1):
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
