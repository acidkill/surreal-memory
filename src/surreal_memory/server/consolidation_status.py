"""Read-only, bounded consolidation observation and Prometheus exposition."""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from surreal_memory.engine.consolidation_receipts import (
    STATE,
    checkpoint_errors,
    checkpoint_hash,
    confirmed_success,
    success_timestamp,
    terminal_receipt_matches,
    timestamp,
)

STATES = (
    "blocked",
    "failed",
    "running",
    "paused",
    "completed",
    "deferred_resource",
    "deferred_lease",
    "unknown",
    "idle",
)
# The hourly observer is stale after two missed scheduler observations.
OBSERVATION_MAX_AGE_SECONDS = 7200
READ_TIMEOUT_SECONDS = 3.0


class ConsolidationStatus(BaseModel):
    brain: Literal["default"] = "default"
    state: Literal[
        "blocked",
        "failed",
        "running",
        "paused",
        "completed",
        "deferred_resource",
        "deferred_lease",
        "unknown",
        "idle",
    ] = "unknown"
    checkpoint_available: bool = False
    observation_available: bool = False
    observation_stale: bool | None = None
    run_id: str | None = None
    checkpoint_status: str | None = None
    current_strategy: str | None = None
    phase: str | None = None
    completed_strategies: int | None = None
    requested_strategies: int | None = None
    progress_known: bool = False
    work_progress_seq: int | None = None
    last_work_progress_at: datetime | None = None
    last_observed_at: datetime | None = None
    last_confirmed_success_at: datetime | None = None
    eligible_work: bool | None = None
    reason: str | None = None


def _time(value: Any, now: datetime) -> datetime | None:
    parsed = timestamp(value)
    return parsed if parsed is not None and parsed <= now + timedelta(seconds=60) else None


def _object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or len(value.encode()) > 1_048_576:
        return None
    try:
        decoded = json.loads(value)
    except (ValueError, RecursionError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[\w.:-]{1,128}", value) else None


def summarize_consolidation(
    cp: Any,
    cm: Any,
    *,
    checkpoint_available: bool,
    now: datetime | None = None,
) -> ConsolidationStatus:
    """Summarize saved evidence; HTTP reads never become scheduler observations."""
    now = now or datetime.now(UTC)
    status = ConsolidationStatus(checkpoint_available=checkpoint_available)
    valid_cp = isinstance(cp, dict) and cp.get("brain_id") == "default"
    if valid_cp:
        status.run_id = _identifier(cp.get("run_id"))
        status.checkpoint_status = (
            cp.get("status")
            if isinstance(cp.get("status"), str)
            and cp.get("status")
            in {
                "queued",
                "running",
                "paused",
                "failed",
                "completed",
            }
            else None
        )
        status.current_strategy = _identifier(cp.get("current_strategy"))
        status.phase = _identifier(cp.get("phase"))
        requested, completed = cp.get("requested_strategies"), cp.get("completed_strategies")
        if (
            isinstance(requested, list)
            and isinstance(completed, list)
            and all(isinstance(item, str) for item in requested + completed)
            and len(set(requested)) == len(requested)
            and len(set(completed)) == len(completed)
            and set(completed).issubset(requested)
            and requested
        ):
            status.requested_strategies, status.completed_strategies = (
                len(requested),
                len(completed),
            )
            if status.checkpoint_status is not None:
                status.eligible_work = cp["status"] != "completed" or len(completed) < len(
                    requested
                )
        seq = cp.get("work_progress_seq")
        work_at = _time(cp.get("last_work_progress_at"), now)
        if type(seq) is int and seq >= 0 and (work_at is not None or seq == 0):
            status.progress_known = True
            status.work_progress_seq, status.last_work_progress_at = seq, work_at
    elif checkpoint_available and cp is None:
        status.eligible_work = False

    data = cm.get("data") if isinstance(cm, dict) else None
    metadata = cm.get("metadata") if isinstance(cm, dict) else None
    if not (
        isinstance(data, dict)
        and isinstance(metadata, dict)
        and metadata.get("name") == STATE
        and isinstance(metadata.get("resourceVersion"), str)
        and metadata.get("resourceVersion")
    ):
        data = {}
    receipt = _object(data.get("receipt.json"))
    observation = _object(data.get("last_observation.json"))
    status.observation_available = (
        receipt is not None
        and isinstance(receipt.get("phase"), str)
        and receipt.get("phase")
        in {
            "bootstrap",
            "started",
            "terminal",
            "blocked",
            "awaiting_manual",
        }
    )
    if "last_observation.json" in data and (
        observation is None
        or observation.get("outcome")
        not in ("blocked", "deferred_capacity", "deferred_active_lease")
        or _time(observation.get("observed_at"), now) is None
    ):
        status.observation_available = False
    if status.observation_available:
        assert receipt is not None
        times = [
            _time(receipt.get(key), now)
            for key in ("observed_at", "started_at", "finished_at", "blocked_at")
        ]
        if observation is not None:
            times.append(_time(observation.get("observed_at"), now))
        status.last_observed_at = max((item for item in times if item is not None), default=None)
        if status.last_observed_at is not None:
            status.observation_stale = (
                now - status.last_observed_at
            ).total_seconds() > OBSERVATION_MAX_AGE_SECONDS
        proof = _object(data.get("last_confirmed_success.json"))
        current_proof = confirmed_success(receipt, cp) if valid_cp else None
        success_times = [
            _time(success_timestamp(proof), now),
            _time(success_timestamp(current_proof), now),
        ]
        status.last_confirmed_success_at = max(
            (item for item in success_times if item is not None),
            default=None,
        )

    if (
        valid_cp
        and cp.get("status") == "completed"
        and not (status.observation_available and confirmed_success(receipt, cp) is not None)
    ):
        status.eligible_work = None

    if receipt is not None and receipt.get("phase") == "blocked" and status.observation_available:
        status.state, status.reason = "blocked", "scheduler_blocked"
    elif valid_cp and (cp.get("status") == "failed" or checkpoint_errors(cp)):
        status.state, status.reason = "failed", "checkpoint_failed"
    elif not status.observation_available:
        status.reason = "scheduler_unavailable"
    elif status.last_observed_at is None or status.observation_stale:
        status.reason = "observation_stale"
    elif not checkpoint_available:
        status.reason = "checkpoint_unavailable"
    elif cp is None:
        status.state = "idle"
    elif not valid_cp or status.run_id is None:
        status.reason = "checkpoint_unavailable"
    elif observation is not None and observation.get("outcome") == "deferred_capacity":
        status.state = "deferred_resource"
    elif observation is not None and observation.get("outcome") == "deferred_active_lease":
        status.state = "deferred_lease"
    elif cp.get("status") == "running" and receipt and receipt.get("phase") == "started":
        status.state = "running"
    elif terminal_receipt_matches(receipt, cp):
        if cp["status"] == "completed" and confirmed_success(receipt, cp) is not None:
            status.state = "completed"
        elif cp["status"] == "paused":
            status.state = "paused"
        else:
            status.reason = "incomplete_strategy_evidence"
    elif (
        cp.get("status") == "paused"
        and receipt
        and (
            receipt.get("phase") == "bootstrap"
            and receipt.get("approved") is True
            and receipt.get("source") == "user_interrupted"
            and receipt.get("cli_rc") is None
            and receipt.get("checkpoint_sha256") == checkpoint_hash(cp)
        )
    ):
        status.state = "paused"
    else:
        status.reason = "receipt_mismatch"
    return status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def read_scheduler_configmap() -> dict[str, Any] | None:
    """GET one existing in-cluster ConfigMap; never list, watch, write or fall back remotely."""
    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    if not host:
        return None
    root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    namespace = os.environ.get("POD_NAMESPACE") or (root / "namespace").read_text().strip()
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", namespace) is None:
        raise ValueError("invalid namespace")
    if re.fullmatch(r"[\w.:-]+", host) is None:
        raise ValueError("invalid Kubernetes service host")
    port = int(os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid Kubernetes port")
    host = f"[{host}]" if ":" in host else host
    url = f"https://{host}:{port}/api/v1/namespaces/{namespace}/configmaps/{STATE}"
    token = (root / "token").read_text().strip()
    context = ssl.create_default_context(cafile=str(root / "ca.crt"))
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})  # noqa: S310 - fixed HTTPS API path.
    # Service-account credentials must never follow redirects or inherited proxies.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
        _NoRedirect(),
    )
    with opener.open(request, timeout=READ_TIMEOUT_SECONDS) as response:
        content = response.read(1_048_577)
    if len(content) > 1_048_576:
        raise ValueError("ConfigMap response exceeds limit")
    result = json.loads(content)
    if not isinstance(result, dict) or result.get("metadata", {}).get("namespace") != namespace:
        raise ValueError("ConfigMap namespace mismatch")
    return result


async def observe_consolidation(storage: Any) -> ConsolidationStatus:
    async def checkpoint() -> tuple[Any, bool]:
        read = getattr(storage, "get_consolidation_progress", None)
        if not callable(read):
            return None, False
        try:
            return await asyncio.wait_for(read("default"), timeout=READ_TIMEOUT_SECONDS), True
        except Exception:
            return None, False

    async def scheduler() -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(read_scheduler_configmap),
                timeout=READ_TIMEOUT_SECONDS + 0.5,
            )
        except Exception:
            return None

    (cp, available), cm = await asyncio.gather(checkpoint(), scheduler())
    return summarize_consolidation(cp, cm, checkpoint_available=available)


def render_metrics(status: ConsolidationStatus) -> str:
    lines = ["# TYPE smem_consolidation_state gauge"] + [
        f'smem_consolidation_state{{brain="default",state="{state}"}} {int(status.state == state)}'
        for state in STATES
    ]
    values: dict[str, int | float | None] = {
        "observation_available": int(status.observation_available),
        "progress_known": int(status.progress_known),
        "eligible_work": int(status.eligible_work) if status.eligible_work is not None else None,
        "work_progress_seq": status.work_progress_seq,
    }
    for name, value in (
        ("last_observed_timestamp_seconds", status.last_observed_at),
        ("last_work_progress_timestamp_seconds", status.last_work_progress_at),
        ("last_confirmed_success_timestamp_seconds", status.last_confirmed_success_at),
    ):
        values[name] = value.timestamp() if value is not None else None
    lines.extend(f"# TYPE smem_consolidation_{name} gauge" for name in values)
    lines.extend(
        f'smem_consolidation_{name}{{brain="default"}} {value}'
        for name, value in values.items()
        if value is not None
    )
    return "\n".join(lines) + "\n"
