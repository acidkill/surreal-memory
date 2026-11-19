#!/usr/bin/env python3
"""Fail-closed hourly supervisor with bounded, resumable CLI invocations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from surreal_memory.engine.consolidation_receipts import (
    MARKER,
    STATE,
    BlockedError,
    canonical,
    capacity_rebind_evidence,
    capacity_resume_matches,
    capacity_resume_seed,
    checkpoint_hash,
    completed_semantic_handoff_matches,
    completed_semantic_handoff_seed,
    confirmed_success,
    failed_runtime_resume_matches,
    forced_supervision_timeout_resume_matches,
    forced_supervision_timeout_resume_seed,
    lease_loss_resume_matches,
    lease_loss_resume_seed,
    native_handoff_anchor,
    resource_rebaseline_seed,
    select_command,
    success_timestamp,
    supervision_timeout_resume_matches,
    supervision_timeout_resume_seed,
    terminal,
    timestamp,
)
from surreal_memory.engine.consolidation_receipts import (
    checkpoint_errors as checkpoint_errors,
)
from surreal_memory.engine.consolidation_receipts import (
    clean_checkpoint as clean_checkpoint,
)

GIB = 1024**3


class DeferredError(BlockedError):
    """Temporary capacity shortage before any mutation."""


class CapacityError(BlockedError):
    """Runtime capacity requires a graceful, verified child stop."""


class SupervisionTimeoutError(BlockedError):
    """The bounded DB-Pod identity read failed; this is not a capacity finding."""

    def __init__(self, path: str, db_uid: str | None) -> None:
        super().__init__("database pod supervision read timed out")
        self.provenance = {
            "version": 1,
            "guard": "database_pod",
            "method": "GET",
            "path": path,
            "timeout_seconds": 15,
            "observed_at": utcnow(),
            "db_uid": db_uid,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Kubernetes:
    def __init__(self) -> None:
        root = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        self.namespace = os.environ["POD_NAMESPACE"]
        self.base = (
            "https://"
            + os.environ["KUBERNETES_SERVICE_HOST"]
            + ":"
            + os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        )
        self.token = (root / "token").read_text().strip()
        self.context = ssl.create_default_context(cafile=str(root / "ca.crt"))

    def request(self, path: str, body: Any = None, raw: bool = False) -> Any:
        data = None if body is None else canonical(body).encode()
        request = urllib.request.Request(  # noqa: S310 - base is fixed Kubernetes HTTPS.
            self.base + path,
            data=data,
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
            method="GET" if body is None else "PUT",
        )
        with urllib.request.urlopen(request, context=self.context, timeout=15) as response:  # noqa: S310
            content = response.read().decode()
        return content if raw else json.loads(content)

    def path(self, resource: str) -> str:
        return f"/api/v1/namespaces/{self.namespace}/{resource}"

    def state(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self.request(self.path("configmaps/" + STATE)))

    def cas(
        self, cm: dict[str, Any], receipt: dict[str, Any], observation: Any = None
    ) -> dict[str, Any]:
        # The original resourceVersion is mandatory: a duplicate pod must lose before spawning.
        if not cm["metadata"].get("resourceVersion"):
            raise BlockedError("state lacks resourceVersion")
        data = dict(cm.get("data", {}), **{"receipt.json": canonical(receipt)})
        proof = receipt.get("confirmed_success")
        if (
            receipt.get("phase") == "terminal"
            and receipt.get("status") == "completed"
            and type(receipt.get("cli_rc")) is int
            and receipt["cli_rc"] == 0
            and isinstance(proof, dict)
            and success_timestamp(proof) is not None
            and all(
                proof.get(key) == receipt.get(key)
                for key in (
                    "run_id",
                    "checkpoint_sha256",
                    "finished_at",
                )
            )
        ):
            data["last_confirmed_success.json"] = canonical(proof)
        if observation is not None:
            data["last_observation.json"] = canonical(observation)
        elif receipt.get("phase") in {"started", "terminal"}:
            data.pop("last_observation.json", None)
        cm = dict(cm, data=data)
        return cast("dict[str, Any]", self.request(self.path("configmaps/" + STATE), cm))

    def previous_log(
        self, receipt: dict[str, Any], *, expected_exit_code: int | None = None
    ) -> str:
        pod = self.request(self.path("pods/" + receipt["pod_name"]))
        if pod["metadata"]["uid"] != receipt["pod_uid"]:
            raise BlockedError("previous pod incarnation changed")
        owners = pod["metadata"].get("ownerReferences", [])
        if not any(o.get("kind") == "Job" and o["uid"] == receipt["job_uid"] for o in owners):
            raise BlockedError("previous pod is not owned by recorded job")
        job = self.request(f"/apis/batch/v1/namespaces/{self.namespace}/jobs/{receipt['job_name']}")
        if job["metadata"]["uid"] != receipt["job_uid"]:
            raise BlockedError("previous job incarnation changed")
        status = job.get("status", {})
        if expected_exit_code is None:
            if status.get("active", 0) or status.get("failed", 0) or status.get("succeeded") != 1:
                raise BlockedError("previous job has not succeeded")
            if pod.get("status", {}).get("phase") != "Succeeded":
                raise BlockedError("previous pod has not succeeded")
        else:
            containers = pod.get("status", {}).get("containerStatuses", [])
            if (
                expected_exit_code not in {2, 255}
                or status.get("active", 0)
                or status.get("failed") != 1
                or status.get("succeeded", 0)
                or pod.get("status", {}).get("phase") != "Failed"
                or len(containers) != 1
                or containers[0].get("state", {}).get("terminated", {}).get("reason") == "OOMKilled"
                or type(containers[0].get("state", {}).get("terminated", {}).get("exitCode"))
                is not int
                or containers[0]["state"]["terminated"]["exitCode"] != expected_exit_code
                or (
                    expected_exit_code == 255
                    and (
                        containers[0]["state"]["terminated"].get("reason") != "Unknown"
                        or containers[0].get("restartCount") != 0
                    )
                )
            ):
                raise BlockedError(
                    "forced supervision predecessor lacks terminal255/Unknown"
                    if expected_exit_code == 255
                    else "capacity resume failed predecessor has not exited with wrapper2"
                )
        return cast(
            "str", self.request(self.path("pods/" + receipt["pod_name"] + "/log"), raw=True)
        )

    def identity(self) -> dict[str, str]:
        name = os.environ["POD_NAME"]
        pod = self.request(self.path("pods/" + name))
        owners = [o for o in pod["metadata"].get("ownerReferences", []) if o.get("kind") == "Job"]
        if len(owners) != 1:
            raise BlockedError("runner pod needs exactly one Job owner")
        return {
            "pod_name": name,
            "pod_uid": pod["metadata"]["uid"],
            "job_name": owners[0]["name"],
            "job_uid": owners[0]["uid"],
        }

    def no_other_jobs(self, own_uid: str) -> None:
        result = self.request(f"/apis/batch/v1/namespaces/{self.namespace}/jobs")
        jobs = result.get("items") if isinstance(result, dict) else None
        if not isinstance(jobs, list) or not own_uid:
            raise BlockedError("resource rebaseline Job inventory unavailable")
        own = 0
        for job in jobs:
            uid = job.get("metadata", {}).get("uid")
            status = job.get("status", {})
            if not isinstance(uid, str) or not uid or not isinstance(status, dict):
                raise BlockedError("resource rebaseline Job identity unknown")
            if uid == own_uid:
                own += 1
                continue
            active = status.get("active", 0)
            terminal_job = any(
                condition.get("type") in {"Complete", "Failed"}
                and condition.get("status") == "True"
                for condition in status.get("conditions", [])
            )
            if type(active) is not int or active != 0 or not terminal_job:
                raise BlockedError("resource rebaseline requires zero other active or unknown Jobs")
        if own != 1:
            raise BlockedError("resource rebaseline operator Job UID missing or ambiguous")


def configured_memory_limit() -> int:
    path = os.environ.get("SURREAL_MEMORY_MAINTENANCE_CONFIG")
    if path is None:
        return 16 * GIB

    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate maintenance config field")
        return dict(pairs)

    try:
        with Path(path).open("rb") as source:
            raw = source.read(4097)
        config = json.loads(raw, object_pairs_hook=unique_fields) if len(raw) <= 4096 else None
    except (OSError, ValueError) as exc:
        raise BlockedError("maintenance instance config is unavailable or malformed") from exc
    if not isinstance(config, dict) or set(config) != {"memory_limit_bytes"}:
        raise BlockedError("maintenance instance config fields are invalid")
    limit = config["memory_limit_bytes"]
    if type(limit) is not int or limit <= 3 * GIB or limit % GIB:
        raise BlockedError("approved memory limit must be whole GiB above the resident reserve")
    return limit


class Safety:
    def __init__(self, api: Kubernetes) -> None:
        self.api = api
        self.baseline: Any = None
        self.memory_limit_bytes = configured_memory_limit()
        self.policy_sha256 = checkpoint_hash({"memory_limit_bytes": self.memory_limit_bytes})

    def sample(self, initial: bool = False) -> None:
        path = self.api.path("pods/" + os.environ["DB_POD"])
        try:
            pod = self.api.request(path)
        except TimeoutError as exc:
            db_uid = self.baseline[0][0] if self.baseline is not None else None
            raise SupervisionTimeoutError(path, db_uid) from exc
        containers = pod.get("spec", {}).get("containers", [])
        limits = [
            c.get("resources", {}).get("limits", {}).get("memory")
            for c in containers
            if c.get("name") == "surrealdb"
        ]
        expected_limit = f"{self.memory_limit_bytes // GIB}Gi"
        if limits != [expected_limit]:
            raise BlockedError(f"database must have the approved {expected_limit} pod limit")
        statuses = pod.get("status", {}).get("containerStatuses", [])
        if not statuses or pod.get("status", {}).get("phase") != "Running":
            raise BlockedError("database pod unavailable")
        identity = (
            pod["metadata"]["uid"],
            canonical([(s["name"], s.get("containerID"), s["restartCount"]) for s in statuses]),
        )
        if not all(s.get("ready") for s in statuses):
            raise BlockedError("database container not ready")
        values = {}
        for key in ("current", "working", "available", "oom", "failcnt"):
            query = os.environ["PROM_QUERY_" + key.upper()]
            url = (
                os.environ["PROMETHEUS_URL"].rstrip("/")
                + "/api/v1/query?"
                + urllib.parse.urlencode({"query": query})
            )
            if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
                raise BlockedError("Prometheus URL must use HTTP(S)")
            with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - scheme checked.
                result = json.load(response)
            vectors = result.get("data", {}).get("result", [])
            if result.get("status") != "success" or len(vectors) != 1:
                raise BlockedError("resource metric missing or ambiguous")
            stamp, value = vectors[0]["value"]
            value = float(value)
            if abs(time.time() - float(stamp)) > 60 or not math.isfinite(value) or value < 0:
                raise BlockedError("resource metric stale or invalid")
            stamp_url = (
                os.environ["PROMETHEUS_URL"].rstrip("/")
                + "/api/v1/query?"
                + urllib.parse.urlencode({"query": "timestamp(" + query + ")"})
            )
            with urllib.request.urlopen(stamp_url, timeout=15) as response:  # noqa: S310 - same checked origin.
                stamped = json.load(response)
            samples = stamped.get("data", {}).get("result", [])
            if (
                stamped.get("status") != "success"
                or len(samples) != 1
                or abs(time.time() - float(samples[0]["value"][1])) > 60
            ):
                raise BlockedError("underlying resource scrape stale or missing")
            values[key] = value
        if values["working"] > values["current"]:
            raise BlockedError("working-set metric exceeds current memory")
        baseline = (identity, values["oom"], values["failcnt"])
        if self.baseline is not None and baseline != self.baseline:
            raise BlockedError("database restarted, changed incarnation, or new OOM")
        self.baseline = baseline
        if values["available"] <= 4 * GIB:
            error = DeferredError if initial else CapacityError
            raise error("node free memory below safety floor")
        startup_resident = self.memory_limit_bytes - 3 * GIB
        if initial:
            unsafe = values["working"] >= startup_resident
        else:
            unsafe = values["working"] >= self.memory_limit_bytes - 1.5 * GIB or (
                values["current"] >= self.memory_limit_bytes - 0.75 * GIB
                and values["working"] >= startup_resident
            )
        unsafe = unsafe or values["current"] > self.memory_limit_bytes
        if unsafe:
            error = DeferredError if initial else CapacityError
            raise error("database memory exceeds safety threshold")


async def live(storage: Any) -> tuple[dict[str, Any], list[Any]]:
    cp = await storage.get_consolidation_progress("default")
    query = getattr(storage, "_query_response", None) or storage._query
    leases = await query(
        "SELECT * FROM consolidation_lease WHERE brain_id = $brain_id AND expires_at > time::now()",
        brain_id="default",
    )
    if not isinstance(leases, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("owner_token"), str)
        or not row["owner_token"]
        for row in leases
    ):
        raise BlockedError("native active-lease response is ambiguous")
    return cp, leases


async def rebaseline_resources(
    api: Kubernetes, storage: Any, safety: Safety, args: argparse.Namespace
) -> int:
    """One audited ConfigMap CAS; never mutate an engine checkpoint or start a child."""
    cm = api.state()
    raw = cm["data"]["receipt.json"]
    if hashlib.sha256(raw.encode()).hexdigest() != args.previous_receipt_sha256:
        raise BlockedError("resource rebaseline predecessor bytes changed")
    previous = json.loads(raw)
    cp, leases = await live(storage)
    if (
        leases
        or cp.get("run_id") != args.confirm_run_id
        or checkpoint_hash(cp) != args.checkpoint_sha256
    ):
        raise BlockedError("resource rebaseline checkpoint or lease proof changed")
    identity = api.identity()
    api.no_other_jobs(identity["job_uid"])
    log = api.previous_log(previous)
    if len(log.encode()) > 128 * 1024:
        raise BlockedError("resource rebaseline predecessor log oversized")
    observation_raw = cm["data"].get("last_observation.json")
    observation_log = None
    if observation_raw is not None:
        observation = json.loads(observation_raw)
        observation_log = api.previous_log(observation)
        if (
            len(observation_log.encode()) > 128 * 1024
            or "HOURLY_OBSERVATION " + canonical(observation) not in observation_log.splitlines()
            or MARKER + canonical(previous) not in observation_log.splitlines()
        ):
            raise BlockedError("resource rebaseline latest observation proof missing")
    safety.sample(initial=True)
    seed = resource_rebaseline_seed(
        previous,
        cp,
        log,
        safety.baseline,
        args.previous_memory_limit_bytes,
        safety.memory_limit_bytes,
        utcnow(),
    )
    archive = {
        "receipt_raw": raw,
        "receipt_log": log,
        "observation_raw": observation_raw,
        "observation_log": observation_log,
        "operator": identity,
        "seed_sha256": checkpoint_hash(seed),
    }
    key = "resource_rebaseline_" + checkpoint_hash(archive) + ".json"
    if key in cm["data"]:
        raise BlockedError("resource rebaseline archive already exists")
    data = dict(cm["data"])
    data[key] = canonical(archive)
    data.pop("last_observation.json", None)
    prepared = dict(cm, data=data)
    safety.sample(initial=True)
    latest, leases = await live(storage)
    api.no_other_jobs(identity["job_uid"])
    if leases or checkpoint_hash(latest) != args.checkpoint_sha256:
        raise BlockedError("resource rebaseline pre-CAS checkpoint or lease changed")
    expected_data = dict(data, **{"receipt.json": canonical(seed)})
    if len(canonical(expected_data).encode()) > 900 * 1024:
        raise BlockedError("resource rebaseline archive exceeds bounded state capacity")
    try:
        api.cas(prepared, seed)
        readback = api.state()
        latest, leases = await live(storage)
        safety.sample(initial=True)
        api.no_other_jobs(identity["job_uid"])
        if (
            readback.get("data") != expected_data
            or leases
            or checkpoint_hash(latest) != args.checkpoint_sha256
        ):
            raise BlockedError("resource rebaseline readback or checkpoint diverged")
    except Exception as exc:
        raise BlockedError(
            "resource rebaseline outcome uncertain; inspect state without retry"
        ) from exc
    print(
        "HOURLY_RESOURCE_REBASELINE "
        + canonical(
            {
                "checkpoint_sha256": args.checkpoint_sha256,
                "archive_key": key,
                "seed_sha256": checkpoint_hash(seed),
                "resource_policy_sha256": safety.policy_sha256,
            }
        ),
        flush=True,
    )
    return 0


def capacity_predecessor_log(
    api: Kubernetes, previous: dict[str, Any], approval: dict[str, Any]
) -> str:
    """Use reviewed archives only after typed 404 for the exact historical Pod."""
    if approval.get("capacity_resume_version", 1) != 2:
        return api.previous_log(previous, expected_exit_code=2)
    rebind = capacity_rebind_evidence(previous, approval)
    namespace = rebind["database"]["namespace"]
    if api.namespace != namespace:
        raise BlockedError("capacity archive namespace differs")
    try:
        api.request(api.path("pods/" + previous["pod_name"]))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        raise BlockedError("capacity archived predecessor requires the recorded Pod to be absent")
    job = api.request(f"/apis/batch/v1/namespaces/{namespace}/jobs/{previous['job_name']}")
    meta, status = job.get("metadata", {}), job.get("status", {})
    if (
        meta.get("uid") != previous["job_uid"]
        or meta.get("name") != previous["job_name"]
        or meta.get("namespace") != namespace
        or meta.get("deletionTimestamp")
        or status.get("active", 0)
        or status.get("failed") != 1
        or status.get("succeeded", 0)
        or not any(
            c.get("type") == "Failed" and c.get("status") == "True"
            for c in status.get("conditions", [])
        )
    ):
        raise BlockedError("capacity archived predecessor retained Job differs")
    return cast("str", rebind["archived_predecessor"]["log"])


def _capacity_database_binding(
    api: Kubernetes,
    binding: dict[str, Any],
    *,
    historical_started_at: datetime | None = None,
    incident_at: datetime | None = None,
) -> dict[str, Any]:
    """Read exact current objects; pin stable identity/spec, recheck live readiness."""
    namespace = binding["namespace"]
    if api.namespace != namespace or os.environ.get("DB_POD") != binding["pod_name"]:
        raise BlockedError("capacity replacement database target differs")
    pod = api.request(api.path("pods/" + binding["pod_name"]))
    meta, spec, status = pod.get("metadata", {}), pod.get("spec", {}), pod.get("status", {})
    statuses = status.get("containerStatuses", [])
    if (
        meta.get("deletionTimestamp")
        or status.get("phase") != "Running"
        or len(statuses) != 1
        or statuses[0].get("name") != "surrealdb"
        or statuses[0].get("ready") is not True
        or type(statuses[0].get("restartCount")) is not int
        or statuses[0]["restartCount"] != 0
        or statuses[0].get("state", {}).get("terminated")
        or statuses[0].get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled"
    ):
        raise BlockedError("capacity replacement database is unready or restarted")

    def owner(obj: dict[str, Any], kind: str) -> dict[str, Any]:
        owners = [
            o
            for o in obj.get("metadata", {}).get("ownerReferences", [])
            if o.get("kind") == kind and o.get("controller") is True
        ]
        if len(owners) != 1:
            raise BlockedError("capacity database controller chain invalid")
        return cast("dict[str, Any]", owners[0])

    rs_owner = owner(pod, "ReplicaSet")
    rs = api.request(
        f"/apis/apps/v1/namespaces/{namespace}/replicasets/{binding['replica_set_name']}"
    )
    dep_owner = owner(rs, "Deployment")
    deployment = api.request(
        f"/apis/apps/v1/namespaces/{namespace}/deployments/{binding['deployment_name']}"
    )
    claims = [
        v["persistentVolumeClaim"]["claimName"]
        for v in spec.get("volumes", [])
        if "persistentVolumeClaim" in v
    ]
    if claims != [binding["pvc_name"]]:
        raise BlockedError("capacity database persistent claim differs")
    pvc = api.request(api.path("persistentvolumeclaims/" + binding["pvc_name"]))
    pv = api.request("/api/v1/persistentvolumes/" + binding["pv_name"])
    node = api.request("/api/v1/nodes/" + binding["node_name"])
    if (
        node.get("metadata", {}).get("name") != spec.get("nodeName")
        or node.get("metadata", {}).get("deletionTimestamp")
        or not any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in node.get("status", {}).get("conditions", [])
        )
    ):
        raise BlockedError("capacity database node is not ready")
    if any(
        obj.get("metadata", {}).get("namespace") != namespace for obj in (pod, rs, deployment, pvc)
    ):
        raise BlockedError("capacity database dependency namespace differs")
    for obj in (rs, deployment, pvc, pv):
        if obj.get("metadata", {}).get("deletionTimestamp"):
            raise BlockedError("capacity database dependency is deleting")
    if (
        rs_owner.get("name") != rs.get("metadata", {}).get("name")
        or rs_owner.get("uid") != rs.get("metadata", {}).get("uid")
        or dep_owner.get("name") != deployment.get("metadata", {}).get("name")
        or dep_owner.get("uid") != deployment.get("metadata", {}).get("uid")
        or pvc.get("status", {}).get("phase") != "Bound"
        or pv.get("status", {}).get("phase") != "Bound"
        or pvc.get("spec", {}).get("volumeName") != pv.get("metadata", {}).get("name")
        or pv.get("spec", {}).get("claimRef", {}).get("uid") != pvc.get("metadata", {}).get("uid")
        or pv.get("spec", {}).get("claimRef", {}).get("name") != pvc.get("metadata", {}).get("name")
        or pv.get("spec", {}).get("claimRef", {}).get("namespace") != namespace
    ):
        raise BlockedError("capacity database controller or bound volume identity differs")
    created = timestamp(meta.get("creationTimestamp"))
    if created is None or (incident_at is not None and created < incident_at):
        raise BlockedError("capacity replacement database predates approved incident")
    if historical_started_at is not None and any(
        timestamp(obj.get("metadata", {}).get("creationTimestamp")) is None
        or timestamp(obj["metadata"]["creationTimestamp"]) > historical_started_at
        for obj in (rs, deployment, pvc, pv)
    ):
        raise BlockedError("capacity persistent objects do not predate the historical failed Job")
    return {
        "namespace": namespace,
        "pod_name": meta["name"],
        "pod_uid": meta["uid"],
        "node_name": spec["nodeName"],
        "image_id": statuses[0]["imageID"],
        "pod_spec_sha256": checkpoint_hash(spec),
        "replica_set_name": rs["metadata"]["name"],
        "replica_set_uid": rs["metadata"]["uid"],
        "replica_set_spec_sha256": checkpoint_hash(rs["spec"]),
        "deployment_name": deployment["metadata"]["name"],
        "deployment_uid": deployment["metadata"]["uid"],
        "deployment_spec_sha256": checkpoint_hash(deployment["spec"]),
        "pvc_name": pvc["metadata"]["name"],
        "pvc_uid": pvc["metadata"]["uid"],
        "pvc_spec_sha256": checkpoint_hash(pvc["spec"]),
        "pv_name": pv["metadata"]["name"],
        "pv_uid": pv["metadata"]["uid"],
        "pv_spec_sha256": checkpoint_hash(pv["spec"]),
    }


def capacity_database_binding(
    api: Kubernetes,
    binding: dict[str, Any],
    *,
    historical_started_at: datetime | None = None,
    incident_at: datetime | None = None,
) -> dict[str, Any]:
    """Return the stable projection, failing closed on malformed API object shapes."""
    try:
        return _capacity_database_binding(
            api, binding, historical_started_at=historical_started_at, incident_at=incident_at
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise BlockedError("capacity database identity response malformed") from exc


async def verify_capacity_rebind(api: Kubernetes, storage: Any, approval: dict[str, Any]) -> None:
    """Verify current persistent incarnation and parsed full brain config, never write them."""
    rebind = approval["incarnation_rebind"]
    historical = rebind["previous_database"]
    if (
        historical.get("proof_kind") != "reviewed_observed_database_continuity"
        or historical.get("unavailable_owner_mount_spec_acknowledged") is not True
    ):
        raise BlockedError("capacity historical-gap review is missing")
    historical_job = json.loads(rebind["archived_predecessor"]["job_raw"])
    historical_started_at = timestamp(historical_job.get("status", {}).get("startTime"))
    if historical_started_at is None:
        raise BlockedError("capacity historical failed Job start time missing")
    observed = capacity_database_binding(
        api,
        rebind["database"],
        historical_started_at=historical_started_at,
        incident_at=timestamp(rebind["incident"]["node_not_ready_at"]),
    )
    if canonical(observed) != canonical(rebind["database"]):
        raise BlockedError("capacity approved database identity or specification changed")
    brain = await storage.get_brain("default")
    config = getattr(brain, "config", None)
    if (
        getattr(brain, "id", None) != rebind["brain"]["brain_id"]
        or not is_dataclass(config)
        or isinstance(config, type)
        or checkpoint_hash(asdict(config)) != rebind["brain"]["config_sha256"]
    ):
        raise BlockedError("capacity default brain identity or effective configuration changed")


def _verify_reviewed_resume_runtime(
    api: Kubernetes,
    safety: Safety,
    receipt: dict[str, Any],
    approval: dict[str, Any],
    *,
    label: str,
) -> None:
    """Shared source/image/resource checks for reviewed operator intents."""
    from importlib.metadata import version

    from surreal_memory.engine import consolidation_receipts

    runtime = approval["runtime"]
    if (
        ".".join(map(str, sys.version_info[:3])) != runtime["python_version"]
        or version("numpy") != runtime["numpy_version"]
        or version("surreal-memory") != runtime["package_version"]
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runtime["runner_sha256"]
        or hashlib.sha256(Path(consolidation_receipts.__file__).read_bytes()).hexdigest()
        != runtime["receipts_sha256"]
        or safety.policy_sha256 != receipt["resource_policy_sha256"]
        or safety.memory_limit_bytes != approval["memory_limit_bytes"]
        or canonical(safety.baseline) != canonical(receipt["resource_baseline"])
    ):
        raise BlockedError(f"{label} runtime source or resource policy differs")
    identity = api.identity()
    api.no_other_jobs(identity["job_uid"])
    pod = api.request(api.path("pods/" + identity["pod_name"]))
    specs = pod.get("spec", {}).get("containers", [])
    statuses = pod.get("status", {}).get("containerStatuses", [])
    digest = runtime["image_digest"]
    if (
        pod.get("metadata", {}).get("uid") != identity["pod_uid"]
        or len(specs) != 1
        or len(statuses) != 1
        or not specs[0].get("image", "").endswith("@" + digest)
        or not statuses[0].get("imageID", "").endswith(digest)
    ):
        raise BlockedError(f"{label} worker image differs")


async def verify_capacity_resume(
    api: Kubernetes, storage: Any, safety: Safety, receipt: dict[str, Any], cp: dict[str, Any]
) -> None:
    """Recheck deployed source parity and unchanged resource guards before admission/use."""
    if not capacity_resume_matches(receipt, cp):
        raise BlockedError("capacity resume seed is invalid")
    approval = receipt["capacity_resume"]["approval"]
    _verify_reviewed_resume_runtime(api, safety, receipt, approval, label="capacity resume")
    if receipt["capacity_resume"]["version"] == 2:
        previous = json.loads(receipt["capacity_resume"]["previous_receipt_raw"])
        if (
            capacity_predecessor_log(api, previous, approval)
            != receipt["capacity_resume"]["previous_log"]
        ):
            raise BlockedError("capacity archived predecessor log changed")
        await verify_capacity_rebind(api, storage, approval)
    latest, leases = await live(storage)
    if leases or checkpoint_hash(latest) != checkpoint_hash(cp):
        raise BlockedError("capacity resume checkpoint or active lease changed")
    if receipt["capacity_resume"]["version"] == 2:
        # Close resource sampling after the comparatively expensive identity proof reads.
        safety.sample(initial=True)


async def verify_lease_loss_resume(
    api: Kubernetes, storage: Any, safety: Safety, receipt: dict[str, Any], cp: dict[str, Any]
) -> None:
    """Revalidate one immutable lease-loss intent without writing native state."""
    if not lease_loss_resume_matches(receipt, cp):
        raise BlockedError("lease loss resume seed is invalid")
    audit = receipt["lease_loss_resume"]
    approval = audit["approval"]
    if audit["version"] == 2:
        prior_key = "lease_loss_resume_" + approval["prior_admission_sha256"] + ".json"
        if api.state().get("data", {}).get(prior_key) != audit["previous_intent_raw"]:
            raise BlockedError("lease loss resume prior approved intent archive changed")
    _verify_reviewed_resume_runtime(api, safety, receipt, approval, label="lease loss resume")
    previous = json.loads(audit["previous_receipt_raw"])
    if api.previous_log(previous, expected_exit_code=2) != audit["previous_log"]:
        raise BlockedError("lease loss resume predecessor changed")
    brain = await storage.get_brain("default")
    config = getattr(brain, "config", None)
    if (
        brain is None
        or getattr(brain, "id", None) != approval["brain_id"]
        or not is_dataclass(config)
        or isinstance(config, type)
        or checkpoint_hash(asdict(config)) != approval["brain_config_sha256"]
    ):
        raise BlockedError("lease loss resume brain config differs")
    latest, leases = await live(storage)
    if leases or checkpoint_hash(latest) != checkpoint_hash(cp):
        raise BlockedError("lease loss resume checkpoint or active lease changed")
    safety.sample(initial=True)


async def admit_lease_loss_resume(
    api: Kubernetes, storage: Any, safety: Safety, args: argparse.Namespace
) -> int:
    return await _admit_reviewed_resume(api, storage, safety, args, lease_loss=True)


async def admit_capacity_stop(
    api: Kubernetes, storage: Any, safety: Safety, args: argparse.Namespace
) -> int:
    return await _admit_reviewed_resume(api, storage, safety, args, lease_loss=False)


async def admit_supervision_timeout(
    api: Kubernetes, storage: Any, safety: Safety, args: argparse.Namespace
) -> int:
    return await _admit_reviewed_resume(
        api, storage, safety, args, lease_loss=False, supervision_timeout=True
    )


async def _verify_supervision_timeout_resume(
    api: Kubernetes,
    storage: Any,
    safety: Safety,
    receipt: dict[str, Any],
    cp: dict[str, Any],
    *,
    forced: bool = False,
) -> None:
    from surreal_memory.engine.consolidation_receipts import _native_digest

    matches = (
        forced_supervision_timeout_resume_matches if forced else supervision_timeout_resume_matches
    )
    if not matches(receipt, cp):
        raise BlockedError("supervision timeout resume seed is invalid")
    audit = receipt["forced_supervision_timeout_resume" if forced else "supervision_timeout_resume"]
    approval = audit["approval"]
    _verify_reviewed_resume_runtime(
        api, safety, receipt, approval, label="supervision timeout resume"
    )
    await _verify_handoff_brain(storage, approval)
    current, leases = await live(storage)
    if (
        leases != []
        or _native_digest(current) != approval["native_checkpoint_sha256"]
        or checkpoint_hash(current) != approval["checkpoint_sha256"]
    ):
        raise BlockedError("supervision timeout checkpoint or lease changed")
    incident = approval["incident_evidence"]
    if incident["path"] != api.path("pods/" + os.environ["DB_POD"]):
        raise BlockedError("supervision timeout database endpoint changed")
    previous = json.loads(audit["previous_raw"])
    if api.previous_log(previous, expected_exit_code=255 if forced else 2) != audit["previous_log"]:
        raise BlockedError("supervision timeout previous log changed")
    pod = api.request(api.path("pods/" + previous["pod_name"]))
    specs, statuses = (
        pod.get("spec", {}).get("containers", []),
        pod.get("status", {}).get("containerStatuses", []),
    )
    if (
        pod.get("metadata", {}).get("uid") != previous["pod_uid"]
        or len(specs) != 1
        or len(statuses) != 1
        or not specs[0].get("image", "").endswith("@" + approval["previous_image_digest"])
        or not statuses[0].get("imageID", "").endswith(approval["previous_image_digest"])
        or statuses[0].get("restartCount") != 0
    ):
        raise BlockedError("supervision timeout previous runtime changed")
    if forced:
        proof = approval["forced_source"]
        terminated = statuses[0].get("state", {}).get("terminated", {})
        job = api.request(f"/apis/batch/v1/namespaces/{api.namespace}/jobs/{previous['job_name']}")
        status = job.get("status", {})
        failed = [
            c
            for c in status.get("conditions", [])
            if c.get("type") == "Failed" and c.get("status") == "True"
        ]
        owners = [
            o
            for o in pod.get("metadata", {}).get("ownerReferences", [])
            if o.get("kind") == "Job" and o.get("controller") is True
        ]
        if (
            pod.get("status", {}).get("phase") != "Failed"
            or len(owners) != 1
            or owners[0].get("uid") != previous["job_uid"]
            or owners[0].get("name") != previous["job_name"]
            or job.get("metadata", {}).get("name") != previous["job_name"]
            or job.get("metadata", {}).get("uid") != proof["job_uid"]
            or pod.get("metadata", {}).get("uid") != proof["pod_uid"]
            or status.get("active", 0)
            or status.get("succeeded", 0)
            or status.get("failed") != 1
            or len(failed) != 1
            or any(
                c.get("type") == "Complete" and c.get("status") == "True"
                for c in status.get("conditions", [])
            )
            or terminated.get("exitCode") != proof["pod_exit_code"]
            or terminated.get("reason") != proof["pod_reason"]
            or terminated.get("containerID") != proof["pod_container_id"]
            or (
                statuses[0].get("containerID") is not None
                and statuses[0]["containerID"] != proof["pod_container_id"]
            )
            or statuses[0].get("restartCount") != proof["pod_restart_count"]
            or timestamp(terminated.get("finishedAt")) != timestamp(proof["pod_finished_at"])
            or timestamp(failed[0].get("lastTransitionTime")) != timestamp(proof["job_failed_at"])
            or failed[0].get("reason") != proof["job_failed_reason"]
        ):
            raise BlockedError("forced supervision executor termination proof changed")
    db_env = [e.get("value") for e in specs[0].get("env", []) if e.get("name") == "DB_POD"]
    if db_env != [os.environ["DB_POD"]]:
        raise BlockedError("supervision timeout previous database target differs")
    old_runner = (
        (previous.get("supervision_timeout") or {}).get("runner_sha256")
        if previous.get("reason") != "TimeoutError"
        else incident["previous_runner_sha256"]
    )
    if not isinstance(old_runner, str) or re.fullmatch(r"[0-9a-f]{64}", old_runner) is None:
        raise BlockedError("supervision timeout previous runner identity missing")
    volumes = pod.get("spec", {}).get("volumes", [])
    if (
        len(
            [
                v
                for v in volumes
                if v.get("configMap", {}).get("name")
                == "smem-consolidate-hourly-runner-" + old_runner[:8]
            ]
        )
        != 1
    ):
        raise BlockedError("supervision timeout previous runner source differs")
    if audit.get("terminal_predecessor") is not None:
        parent = audit["terminal_predecessor"]
        log = api.previous_log(parent)
        markers = [line[len(MARKER) :] for line in log.splitlines() if line.startswith(MARKER)]
        if len(markers) != 1:
            raise BlockedError("supervision timeout terminal predecessor marker missing")
        try:
            terminal_parent = json.loads(markers[0])
        except ValueError as exc:
            raise BlockedError("supervision timeout terminal predecessor marker malformed") from exc
        if (
            terminal_parent.get("phase") != "terminal"
            or checkpoint_hash(terminal_parent) != parent["receipt_sha256"]
            or any(
                terminal_parent.get(key) != parent.get(key)
                for key in (
                    "job_name",
                    "job_uid",
                    "pod_name",
                    "pod_uid",
                    "status",
                    "cli_rc",
                    "run_id",
                    "checkpoint_sha256",
                    "finished_at",
                )
            )
        ):
            raise BlockedError("supervision timeout terminal predecessor changed")
        return
    state = api.state()
    parent_raw = state.get("data", {}).get(audit["predecessor_archive_key"])
    try:
        parent = json.loads(parent_raw)
    except (TypeError, ValueError) as exc:
        raise BlockedError("supervision timeout predecessor archive missing") from exc
    suffix = audit["predecessor_archive_key"].rsplit("_", 1)[-1][:-5]
    if (
        checkpoint_hash(parent) != suffix
        or parent.get("run_id") != previous["before_run_id"]
        or parent.get("checkpoint_sha256") != previous["before_sha256"]
    ):
        raise BlockedError("supervision timeout predecessor archive changed")


async def verify_supervision_timeout_resume(
    api: Kubernetes, storage: Any, safety: Safety, receipt: dict[str, Any], cp: dict[str, Any]
) -> None:
    await _verify_supervision_timeout_resume(api, storage, safety, receipt, cp)


async def verify_forced_supervision_timeout_resume(
    api: Kubernetes, storage: Any, safety: Safety, receipt: dict[str, Any], cp: dict[str, Any]
) -> None:
    await _verify_supervision_timeout_resume(api, storage, safety, receipt, cp, forced=True)


async def admit_forced_supervision_timeout(
    api: Kubernetes, storage: Any, safety: Safety, args: argparse.Namespace
) -> int:
    return await _admit_reviewed_resume(
        api, storage, safety, args, lease_loss=False, forced_supervision_timeout=True
    )


async def _admit_reviewed_resume(
    api: Kubernetes,
    storage: Any,
    safety: Safety,
    args: argparse.Namespace,
    *,
    lease_loss: bool,
    supervision_timeout: bool = False,
    forced_supervision_timeout: bool = False,
) -> int:
    """Archive one failed receipt and CAS a reviewed resume intent; never launch a CLI."""
    if sum((lease_loss, supervision_timeout, forced_supervision_timeout)) > 1:
        raise BlockedError("resume admission modes conflict")
    is_supervision = supervision_timeout or forced_supervision_timeout
    label = (
        "forced supervision timeout resume"
        if forced_supervision_timeout
        else (
            "supervision timeout resume"
            if supervision_timeout
            else "lease loss resume"
            if lease_loss
            else "capacity resume"
        )
    )
    verifier = (
        verify_forced_supervision_timeout_resume
        if forced_supervision_timeout
        else (
            verify_supervision_timeout_resume
            if supervision_timeout
            else verify_lease_loss_resume
            if lease_loss
            else verify_capacity_resume
        )
    )
    prefix = (
        "forced_supervision_timeout_resume_"
        if forced_supervision_timeout
        else (
            "supervision_timeout_resume_"
            if supervision_timeout
            else "lease_loss_resume_"
            if lease_loss
            else "capacity_resume_"
        )
    )
    cm = api.state()
    raw = cm["data"]["receipt.json"]
    previous = json.loads(raw)
    approval_raw = Path(args.approval_file).read_text(encoding="utf-8")
    if len(approval_raw.encode()) > 128 * 1024:
        raise BlockedError(f"{label} approval is oversized")
    approval = json.loads(approval_raw)
    if forced_supervision_timeout and hashlib.sha256(raw.encode()).hexdigest() != approval.get(
        "previous_receipt_raw_sha256"
    ):
        raise BlockedError("forced supervision source receipt changed")
    if is_supervision and cm.get("metadata", {}).get("resourceVersion") != approval.get(
        "state_resource_version"
    ):
        raise BlockedError("supervision timeout state changed since approval")
    cp, leases = await live(storage)
    if leases or (is_supervision and leases != []):
        raise BlockedError(f"{label} requires zero active leases")
    identity = api.identity()
    api.no_other_jobs(identity["job_uid"])
    log = (
        api.previous_log(previous, expected_exit_code=255 if forced_supervision_timeout else 2)
        if lease_loss or is_supervision
        else capacity_predecessor_log(api, previous, approval)
    )
    observation_raw = cm["data"].get("last_observation.json")
    observation_log = (
        api.previous_log(json.loads(observation_raw), expected_exit_code=2)
        if observation_raw is not None and not lease_loss and not is_supervision
        else None
    )
    safety.sample(initial=True)
    seed = (
        forced_supervision_timeout_resume_seed(
            raw, log, cp, approval=approval, observation_raw=observation_raw
        )
        if forced_supervision_timeout
        else supervision_timeout_resume_seed(
            raw, log, cp, approval=approval, observation_raw=observation_raw
        )
        if supervision_timeout
        else lease_loss_resume_seed(
            raw,
            log,
            cp,
            approval=approval,
            previous_intent_raw=(
                cm["data"].get(
                    "lease_loss_resume_" + str(approval.get("prior_admission_sha256")) + ".json"
                )
                if approval.get("lease_loss_resume_version") == 2
                else None
            ),
        )
        if lease_loss
        else capacity_resume_seed(
            raw,
            log,
            cp,
            approval=approval,
            observation_raw=observation_raw,
            observation_log=observation_log,
        )
    )
    await verifier(api, storage, safety, seed, cp)
    key = prefix + checkpoint_hash(seed) + ".json"
    if key in cm["data"]:
        raise BlockedError(f"{label} archive already exists")
    data = dict(cm["data"], **{key: canonical(seed)})
    if (lease_loss or is_supervision) and observation_raw is not None:
        observation_key = (
            (
                "forced_supervision_timeout_observation_"
                if forced_supervision_timeout
                else "supervision_timeout_observation_"
                if is_supervision
                else "lease_loss_observation_"
            )
            + hashlib.sha256(observation_raw.encode()).hexdigest()
            + ".json"
        )
        if observation_key in data and data[observation_key] != observation_raw:
            raise BlockedError("lease loss observation archive differs")
        data[observation_key] = observation_raw
    data.pop("last_observation.json", None)
    expected_data = dict(data, **{"receipt.json": canonical(seed)})
    if len(canonical(expected_data).encode()) > 900 * 1024:
        raise BlockedError(f"{label} archive exceeds bounded state capacity")
    safety.sample(initial=True)
    await verifier(api, storage, safety, seed, cp)
    try:
        api.cas(dict(cm, data=data), seed)
        readback = api.state()
        safety.sample(initial=True)
        await verifier(api, storage, safety, seed, cp)
        if readback.get("data") != expected_data:
            raise BlockedError(f"{label} readback differs")
    except Exception as exc:
        raise BlockedError(f"{label} outcome uncertain; inspect state without retry") from exc
    print(
        (
            "HOURLY_FORCED_SUPERVISION_TIMEOUT_ADMISSION "
            if forced_supervision_timeout
            else "HOURLY_SUPERVISION_TIMEOUT_ADMISSION "
            if is_supervision
            else "HOURLY_LEASE_LOSS_ADMISSION "
            if lease_loss
            else "HOURLY_CAPACITY_ADMISSION "
        )
        + canonical(
            {
                "checkpoint_sha256": checkpoint_hash(cp),
                "archive_key": key,
                "seed_sha256": checkpoint_hash(seed),
                "historical_cli_rc": previous["cli_rc"],
            }
        ),
        flush=True,
    )
    return 0


async def _verify_handoff_brain(storage: Any, approval: dict[str, Any]) -> None:
    from surreal_memory.engine.consolidation_receipts import _native_digest

    brain = await storage.get_brain("default")
    if (
        brain is None
        or brain.name != "default"
        or brain.id != approval["brain_uuid"]
        or not is_dataclass(brain.config)
        or _native_digest(asdict(brain.config)) != approval["brain_config_sha256"]
    ):
        raise BlockedError("completed semantic handoff brain/configuration changed")


async def verify_completed_semantic_handoff(
    api: Kubernetes, storage: Any, safety: Safety, seed: dict[str, Any], cp: dict[str, Any]
) -> None:
    """Re-read the exact closed-run evidence and current native/resource fences."""
    from surreal_memory.engine.consolidation_receipts import _native_digest

    if not completed_semantic_handoff_matches(seed, cp):
        raise BlockedError("completed semantic handoff checkpoint or approval differs")
    audit = seed["completed_semantic_handoff"]
    approval = audit["approval"]
    _verify_reviewed_resume_runtime(api, safety, seed, approval, label="completed semantic handoff")
    await _verify_handoff_brain(storage, approval)
    current, leases = await live(storage)
    if leases != [] or _native_digest(current) != approval["checkpoint_sha256"]:
        raise BlockedError("completed semantic checkpoint or lease changed")
    source = approval["semantic_source"]
    state = api.request(api.path("configmaps/" + source["state_name"]))
    if (
        state.get("metadata", {}).get("uid") != source["state_uid"]
        or state.get("data", {}).get("receipt.json") != audit["semantic_receipt_raw"]
    ):
        raise BlockedError("semantic completion ConfigMap evidence changed")
    log = api.previous_log(source)
    pod = api.request(api.path("pods/" + source["pod_name"]))
    specs = pod.get("spec", {}).get("containers", [])
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if (
        log != audit["semantic_log"]
        or len(specs) != 1
        or len(statuses) != 1
        or not specs[0].get("image", "").endswith("@" + source["image_digest"])
        or not statuses[0].get("imageID", "").endswith(source["image_digest"])
        or type(statuses[0].get("state", {}).get("terminated", {}).get("exitCode")) is not int
        or statuses[0]["state"]["terminated"]["exitCode"] != 0
        or statuses[0].get("restartCount") != 0
    ):
        raise BlockedError("semantic completion Pod/CLI-zero evidence changed")


async def admit_completed_semantic_handoff(
    api: Kubernetes, storage: Any, safety: Safety, args: argparse.Namespace
) -> int:
    """CAS one audited fresh-ALL intent; preserve history and never launch a CLI."""
    import re

    cm = api.state()
    raw = cm["data"]["receipt.json"]
    approval_raw = Path(args.approval_file).read_text(encoding="utf-8")
    if len(approval_raw.encode()) > 128 * 1024:
        raise BlockedError("completed semantic handoff approval is oversized")
    approval = json.loads(approval_raw)
    if cm.get("metadata", {}).get("resourceVersion") != approval.get("state_resource_version"):
        raise BlockedError("hourly state changed since handoff approval")
    cron = api.request(
        f"/apis/batch/v1/namespaces/{api.namespace}/cronjobs/smem-consolidate-hourly"
    )
    if cron.get("spec", {}).get("suspend") is not True:
        raise BlockedError("completed semantic admission requires hourly still suspended")
    cp, leases = await live(storage)
    if leases != []:
        raise BlockedError("completed semantic handoff requires zero active leases")
    source = approval.get("semantic_source", {})
    for key in ("state_name", "job_name", "pod_name"):
        if (
            not isinstance(source.get(key), str)
            or re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?", source[key]) is None
        ):
            raise BlockedError("semantic source Kubernetes name is invalid")
    state = api.request(api.path("configmaps/" + source["state_name"]))
    semantic_raw = state.get("data", {}).get("receipt.json")
    log = api.previous_log(source)
    observation_raw = cm["data"].get("last_observation.json")
    safety.sample(initial=True)
    seed = completed_semantic_handoff_seed(
        raw,
        cp,
        approval=approval,
        semantic_receipt_raw=semantic_raw,
        semantic_log=log,
        observation_raw=observation_raw,
    )
    await verify_completed_semantic_handoff(api, storage, safety, seed, cp)
    key = "completed_semantic_handoff_" + checkpoint_hash(seed) + ".json"
    if key in cm["data"]:
        raise BlockedError("completed semantic handoff archive already exists")
    data = dict(cm["data"], **{key: canonical(seed)})
    if observation_raw is not None:
        observation_key = (
            "completed_semantic_observation_"
            + hashlib.sha256(observation_raw.encode()).hexdigest()
            + ".json"
        )
        if observation_key in data and data[observation_key] != observation_raw:
            raise BlockedError("completed semantic observation archive differs")
        data[observation_key] = observation_raw
    data.pop("last_observation.json", None)
    expected_data = dict(data, **{"receipt.json": canonical(seed)})
    if len(canonical(expected_data).encode()) > 900 * 1024:
        raise BlockedError("completed semantic handoff archive exceeds bounded state capacity")
    safety.sample(initial=True)
    await verify_completed_semantic_handoff(api, storage, safety, seed, cp)
    try:
        api.cas(dict(cm, data=data), seed)
        readback = api.state()
        safety.sample(initial=True)
        await verify_completed_semantic_handoff(api, storage, safety, seed, cp)
        if readback.get("data") != expected_data:
            raise BlockedError("completed semantic handoff readback differs")
    except Exception as exc:
        raise BlockedError(
            "completed semantic handoff outcome uncertain; inspect state without retry"
        ) from exc
    print(
        "HOURLY_COMPLETED_SEMANTIC_ADMISSION "
        + canonical(
            {
                "archive_key": key,
                "seed_sha256": checkpoint_hash(seed),
                "next_action": "fresh_all",
            }
        ),
        flush=True,
    )
    return 0


def stop(child: subprocess.Popen[Any]) -> bool:
    if child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            return False
    return type(child.returncode) is int


async def verify_failed_runtime_resume(
    api: Kubernetes, storage: Any, safety: Safety, receipt: dict[str, Any], cp: dict[str, Any]
) -> None:
    """Recheck the approved runtime/source against this worker before ownership CAS."""
    from importlib.metadata import version

    import numpy as np

    if not failed_runtime_resume_matches(receipt, cp):
        raise BlockedError("failed runtime resume seed is invalid")
    approval = receipt["failed_runtime_resume"]["approval"]
    runtime = approval["runtime"]
    if (
        ".".join(map(str, sys.version_info[:3])) != runtime["python_version"]
        or version("numpy") != np.__version__
        or np.__version__ != runtime["numpy_version"]
        or version("surreal-memory") != runtime["package_version"]
        or safety.policy_sha256 != receipt["resource_policy_sha256"]
    ):
        raise BlockedError("failed runtime resume installed runtime or policy differs")
    identity = api.identity()
    api.no_other_jobs(identity["job_uid"])
    pod = api.request(api.path("pods/" + identity["pod_name"]))
    specs = pod.get("spec", {}).get("containers", [])
    statuses = pod.get("status", {}).get("containerStatuses", [])
    digest = runtime["image_digest"]
    if (
        pod.get("metadata", {}).get("uid") != identity["pod_uid"]
        or len(specs) != 1
        or len(statuses) != 1
        or not specs[0].get("image", "").endswith("@" + digest)
        or not statuses[0].get("imageID", "").endswith(digest)
    ):
        raise BlockedError("failed runtime resume worker image differs")
    ref = approval["source_state_ref"]
    # The native public loader requires a lease; preclaim proof is one exact indexed read.
    rows = await storage._query(
        "SELECT brain_id, run_id, state_id, revision, source_token, payload "
        "FROM semantic_discovery_state WITH INDEX idx_sds_state_revision "
        "WHERE brain_id = $brain_id AND run_id = $run_id "
        "AND state_id = $state_id AND revision = $revision LIMIT 1",
        brain_id=cp["brain_id"],
        run_id=cp["run_id"],
        state_id=ref["state_id"],
        revision=ref["revision"],
    )
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise BlockedError("failed runtime resume exact source row missing")
    row = rows[0]
    source = row.get("payload")
    expected = {
        "brain_id": cp["brain_id"],
        "run_id": cp["run_id"],
        "state_id": ref["state_id"],
        "revision": ref["revision"],
        "source_token": approval["source_token"],
    }
    if (
        not isinstance(source, dict)
        or any(row.get(key) != value or source.get(key) != value for key, value in expected.items())
        or source.get("state_revision") != ref["revision"]
        or ref.get("source_token") != approval["source_token"]
    ):
        raise BlockedError("failed runtime resume source row or payload identity differs")
    if (
        not isinstance(source, dict)
        or checkpoint_hash(source) != approval["source_state_sha256"]
        or source.get("source_token") != approval["source_token"]
        or (source.get("legacy_checkpoint") or {}).get("similarity_backend")
        != approval["source_backend"]
    ):
        raise BlockedError("failed runtime resume immutable source differs")
    # Native resume validates mutable source via its saved, budgeted validation cursor.


def failed_resume_startup_pending(
    before: dict[str, Any], current: dict[str, Any], active: list[Any]
) -> bool:
    """Recognize only native acquire/claim intervals or a proved same-run continuation."""
    anchor = native_handoff_anchor(before)
    if anchor is None or not current:
        raise BlockedError("failed runtime resume checkpoint proof missing")
    owner = current.get("owner_token")
    lease_owner = active[0].get("owner_token") if len(active) == 1 else None
    owner_valid = isinstance(owner, str) and bool(owner) and owner != before.get("owner_token")
    lease_valid = (
        isinstance(lease_owner, str)
        and bool(lease_owner)
        and lease_owner != before.get("owner_token")
    )
    if checkpoint_hash(current) == checkpoint_hash(before):
        if not active or (len(active) == 1 and lease_valid):
            return True
        raise BlockedError("failed runtime resume startup lease differs")
    # claim_consolidation_progress changes only these two fields before the running save.
    restored = dict(current)
    for key in ("owner_token", "updated_at"):
        if key in before:
            restored[key] = before[key]
        else:
            restored.pop(key, None)
    if (
        owner_valid
        and timestamp(current.get("updated_at")) is not None
        and len(active) == 1
        and lease_owner == owner
        and checkpoint_hash(restored) == checkpoint_hash(before)
    ):
        return True
    claimed = native_handoff_anchor(current)
    if (
        claimed is None
        or any(
            claimed[key] != anchor[key]
            for key in (
                "lineage_root_sha256",
                "run_seq",
                "scope_sha256",
                "run_id",
                "reference_time",
            )
        )
        or claimed["attempt_seq"] <= anchor["attempt_seq"]
        or any(
            claimed["completed_states"].get(name) != digest
            for name, digest in anchor["completed_states"].items()
        )
        or any(
            claimed["counters"].get(name) != anchor["counters"].get(name)
            for name in anchor["completed_states"]
        )
        or current.get("status") not in {"running", "paused", "completed"}
        or checkpoint_errors(current)
        or not owner_valid
        or len(active) > 1
        or (active and lease_owner != owner)
        or (current.get("status") == "running" and not active)
    ):
        raise BlockedError("failed runtime resume native claim differs")
    return False


async def run(api: Kubernetes, storage: Any, safety: Safety) -> int:
    cm = api.state()
    receipt = json.loads(cm["data"]["receipt.json"])
    child = None
    owns = False
    capacity_checkpoint = None
    try:
        if datetime.now(UTC) < datetime.fromisoformat("2026-11-05T13:15:00+00:00"):
            print("HOURLY_DEFERRED not-before", flush=True)
            return 0
        if receipt.get("phase") == "blocked":
            observed = {
                "outcome": "blocked",
                "reason": receipt.get("reason"),
                "observed_at": utcnow(),
                **api.identity(),
            }
            api.cas(cm, receipt, observed)
            print(MARKER + canonical(receipt), flush=True)
            print("HOURLY_OBSERVATION " + canonical(observed), flush=True)
            return 2
        decision_log = None
        prior_observation = cm["data"].get("last_observation.json")
        if prior_observation:
            observed = json.loads(prior_observation)
            log = api.previous_log(observed)
            decision_log = log
            if "HOURLY_OBSERVATION " + canonical(observed) not in log.splitlines():
                raise BlockedError("previous observation logs missing or stale")
        if receipt.get("phase") == "terminal":
            if decision_log is None:
                decision_log = api.previous_log(receipt)
            markers = [line for line in decision_log.splitlines() if line.startswith(MARKER)]
            if markers != [MARKER + canonical(receipt)]:
                raise BlockedError("immediate predecessor decision marker missing or stale")
        cp, leases = await live(storage)
        if leases:
            if receipt.get("phase") == "started":
                print("HOURLY_DEFERRED active-owner", flush=True)
                return 0
            observed = {
                "outcome": "deferred_active_lease",
                "observed_at": utcnow(),
                **api.identity(),
            }
            api.cas(cm, receipt, observed)
            print(MARKER + canonical(receipt), flush=True)
            print("HOURLY_OBSERVATION " + canonical(observed), flush=True)
            return 0
        command = select_command(receipt, cp, decision_log)
        handoff_authorization = (
            receipt if receipt.get("source") == "operator_completed_semantic_handoff" else None
        )
        fresh_all_options = (
            receipt["expected_all_options_fingerprint"]
            if handoff_authorization is not None
            else None
        )
        # Leave 500s below the 3500s Job deadline for setup, health and shutdown.
        command.extend(["--strategy-timeout", "1800", "--total-timeout", "3000"])
        safety.sample(initial=True)
        if handoff_authorization is not None:
            archive = "completed_semantic_handoff_" + checkpoint_hash(receipt) + ".json"
            if cm["data"].get(archive) != canonical(receipt):
                raise BlockedError("completed semantic handoff archive missing or stale")
            await verify_completed_semantic_handoff(api, storage, safety, receipt, cp)
        forced_supervision_authorization = (
            receipt
            if receipt.get("source") == "operator_forced_supervision_timeout_resume"
            else None
        )
        if forced_supervision_authorization is not None:
            archive = "forced_supervision_timeout_resume_" + checkpoint_hash(receipt) + ".json"
            if cm["data"].get(archive) != canonical(receipt):
                raise BlockedError("forced supervision timeout archive missing or stale")
            await verify_forced_supervision_timeout_resume(api, storage, safety, receipt, cp)
        supervision_authorization = (
            receipt if receipt.get("source") == "operator_supervision_timeout_resume" else None
        )
        if supervision_authorization is not None:
            archive = "supervision_timeout_resume_" + checkpoint_hash(receipt) + ".json"
            if cm["data"].get(archive) != canonical(receipt):
                raise BlockedError("supervision timeout admission archive missing or stale")
            await verify_supervision_timeout_resume(api, storage, safety, receipt, cp)
        failed_runtime = receipt.get("source") == "operator_failed_runtime_resume"
        lease_authorization = (
            receipt if receipt.get("source") == "operator_lease_loss_resume" else None
        )
        runtime_resume = (
            failed_runtime
            or lease_authorization is not None
            or supervision_authorization is not None
            or forced_supervision_authorization is not None
        )
        capacity_authorization = (
            receipt if receipt.get("source") == "operator_capacity_resume" else None
        )
        if capacity_authorization is not None:
            archive = "capacity_resume_" + checkpoint_hash(receipt) + ".json"
            if cm["data"].get(archive) != canonical(receipt):
                raise BlockedError("capacity resume archive missing or stale")
            await verify_capacity_resume(api, storage, safety, receipt, cp)
        if failed_runtime:
            archive = "failed_runtime_resume_" + checkpoint_hash(receipt) + ".json"
            if cm["data"].get(archive) != canonical(receipt):
                raise BlockedError("failed runtime resume archive missing or stale")
            await verify_failed_runtime_resume(api, storage, safety, receipt, cp)
        if lease_authorization is not None:
            archive = "lease_loss_resume_" + checkpoint_hash(lease_authorization) + ".json"
            if cm["data"].get(archive) != canonical(lease_authorization):
                raise BlockedError("lease loss resume archive missing or stale")
            await verify_lease_loss_resume(api, storage, safety, lease_authorization, cp)
        if (
            receipt.get("source") == "operator_resource_rebaseline"
            and receipt.get("resource_policy_sha256") != safety.policy_sha256
        ):
            raise BlockedError("approved resource policy changed since operator rebaseline")
        if not receipt.get("resource_baseline") or canonical(
            receipt["resource_baseline"]
        ) != canonical(safety.baseline):
            raise BlockedError("database incarnation or OOM changed since approved predecessor")
        # Close the preflight window before the CAS; the engine still arbitrates its own lease.
        latest, leases = await live(storage)
        if leases:
            if receipt.get("phase") == "started":
                print("HOURLY_DEFERRED active-owner", flush=True)
                return 0
            observed = {
                "outcome": "deferred_active_lease",
                "observed_at": utcnow(),
                **api.identity(),
            }
            api.cas(cm, receipt, observed)
            print(MARKER + canonical(receipt), flush=True)
            print("HOURLY_OBSERVATION " + canonical(observed), flush=True)
            return 0
        if checkpoint_hash(latest) != checkpoint_hash(cp):
            raise BlockedError("checkpoint changed during preflight")
        if (
            runtime_resume
            or capacity_authorization is not None
            or handoff_authorization is not None
            or supervision_authorization is not None
            or forced_supervision_authorization is not None
        ):
            safety.sample(initial=True)
            api.no_other_jobs(api.identity()["job_uid"])
        resume_seed_sha256 = checkpoint_hash(receipt) if failed_runtime else None
        capacity_seed_sha256 = (
            checkpoint_hash(receipt) if capacity_authorization is not None else None
        )
        terminal_parent = None
        parent_fields = (
            "job_name",
            "job_uid",
            "pod_name",
            "pod_uid",
            "status",
            "cli_rc",
            "run_id",
            "checkpoint_sha256",
            "finished_at",
        )
        if (
            receipt.get("phase") == "terminal"
            and all(key in receipt for key in parent_fields)
            and all(
                isinstance(receipt.get(key), str) and receipt[key]
                for key in parent_fields
                if key != "cli_rc"
            )
            and type(receipt.get("cli_rc")) is int
            and timestamp(receipt.get("finished_at")) is not None
        ):
            terminal_parent = {key: receipt[key] for key in parent_fields}
            terminal_parent.update(
                version=1,
                receipt_sha256=checkpoint_hash(receipt),
                before_sha256=checkpoint_hash(cp),
                options_fingerprint=cp["options_fingerprint"],
                reference_time=cp["reference_time"],
                scope_sha256=checkpoint_hash(
                    {
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
                ),
            )
        receipt = {
            **(
                {
                    "forced_supervision_timeout_resume_sha256": checkpoint_hash(
                        forced_supervision_authorization
                    )
                }
                if forced_supervision_authorization is not None
                else {}
            ),
            **({"terminal_predecessor": terminal_parent} if terminal_parent is not None else {}),
            **(
                {"supervision_timeout_resume_sha256": checkpoint_hash(supervision_authorization)}
                if supervision_authorization is not None
                else {}
            ),
            **({"failed_runtime_resume_sha256": resume_seed_sha256} if failed_runtime else {}),
            **(
                {"lease_loss_resume_sha256": checkpoint_hash(lease_authorization)}
                if lease_authorization is not None
                else {}
            ),
            **(
                {"capacity_resume_sha256": capacity_seed_sha256}
                if capacity_authorization is not None
                else {}
            ),
            **(
                {"completed_semantic_handoff_sha256": checkpoint_hash(handoff_authorization)}
                if handoff_authorization is not None
                else {}
            ),
            "phase": "started",
            "started_at": utcnow(),
            "before_sha256": checkpoint_hash(cp),
            "before_run_id": cp["run_id"],
            "before_resources": safety.baseline,
            **api.identity(),
        }
        cm = api.cas(cm, receipt)
        owns = True
        latest, leases = await live(storage)
        if leases or checkpoint_hash(latest) != checkpoint_hash(cp):
            raise BlockedError("checkpoint or lease changed after ownership CAS")
        if (
            runtime_resume
            or capacity_authorization is not None
            or handoff_authorization is not None
            or supervision_authorization is not None
            or forced_supervision_authorization is not None
        ):
            safety.sample(initial=True)
            api.no_other_jobs(api.identity()["job_uid"])
        if capacity_authorization is not None:
            await verify_capacity_resume(api, storage, safety, capacity_authorization, cp)
        if lease_authorization is not None:
            await verify_lease_loss_resume(api, storage, safety, lease_authorization, cp)
        if handoff_authorization is not None:
            await verify_completed_semantic_handoff(api, storage, safety, handoff_authorization, cp)
        if supervision_authorization is not None:
            await verify_supervision_timeout_resume(
                api, storage, safety, supervision_authorization, cp
            )
        if forced_supervision_authorization is not None:
            await verify_forced_supervision_timeout_resume(
                api, storage, safety, forced_supervision_authorization, cp
            )
        awaiting_native_claim = runtime_resume
        native_owner = None
        observed_lease_owner = None
        with tempfile.TemporaryFile(mode="w+t") as output:

            def interrupted(signum: int, frame: Any) -> None:
                raise BlockedError("runner interrupted by signal " + str(signum))

            previous = {
                sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
            }
            try:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    await asyncio.sleep(10)
                    capacity_error = None
                    try:
                        safety.sample()
                    except CapacityError as exc:
                        capacity_error = exc
                    except SupervisionTimeoutError as exc:
                        receipt["supervision_timeout"] = dict(
                            exc.provenance, child_was_running=child.poll() is None
                        )
                        raise
                    current, active = await live(storage)
                    pending = False
                    if runtime_resume:
                        pending = failed_resume_startup_pending(cp, current, active)
                        if active:
                            owner = active[0]["owner_token"]
                            if observed_lease_owner is not None and owner != observed_lease_owner:
                                raise BlockedError("failed runtime resume startup lease changed")
                            observed_lease_owner = owner
                        elif pending and observed_lease_owner is not None:
                            raise BlockedError("failed runtime resume startup lease lost")
                        if (
                            not pending
                            and observed_lease_owner is not None
                            and current.get("owner_token") != observed_lease_owner
                        ):
                            raise BlockedError("failed runtime resume claim owner differs")
                        if pending and not awaiting_native_claim:
                            raise BlockedError("failed runtime resume reverted after native claim")
                        if not pending:
                            if (
                                native_owner is not None
                                and current.get("owner_token") != native_owner
                            ):
                                raise BlockedError("failed runtime resume native owner changed")
                            native_owner = current["owner_token"]
                            awaiting_native_claim = False
                    if (
                        not pending
                        and current
                        and (
                            current.get("last_error")
                            or current.get("status") in {"failed", "lease_lost", "timed_out"}
                        )
                    ):
                        raise BlockedError("live checkpoint reports failure")
                    if (
                        not pending
                        and current
                        and current.get("status") == "running"
                        and (
                            len(active) != 1
                            or active[0].get("owner_token") != current.get("owner_token")
                        )
                    ):
                        raise BlockedError("engine lease absent or lost")
                    if capacity_error is not None:
                        if (
                            not current
                            or current.get("status") not in {"running", "paused", "completed"}
                            or checkpoint_errors(current)
                            or timestamp(current.get("reference_time")) is None
                            or not isinstance(current.get("owner_token"), str)
                            or not current["owner_token"]
                        ):
                            raise BlockedError(
                                "capacity stop checkpoint is unsafe"
                            ) from capacity_error
                        capacity_checkpoint = current
                        receipt["capacity_stop_reason"] = str(capacity_error)
                        if not stop(child):
                            raise BlockedError(
                                "capacity stop required forced termination"
                            ) from capacity_error
                        break
                rc = child.wait()
                receipt["cli_rc"] = rc
            finally:
                if child is not None:
                    graceful = stop(child)
                    if "supervision_timeout" in receipt:
                        receipt["supervision_timeout"].update(
                            stop_graceful=graceful, actual_child_rc=child.returncode
                        )
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                output.seek(0)
                text = output.read()
                if child is not None:
                    if isinstance(child.returncode, int):
                        receipt["cli_rc"] = child.returncode
                    receipt.update(
                        cli_output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        cli_output_tail=text[-4096:],
                    )
                print(text, end="", flush=True)
        try:
            safety.sample()
        except CapacityError:
            if capacity_checkpoint is None:
                raise
        after, leases = await live(storage)
        if leases:
            raise BlockedError("engine lease still active after child exit")
        if runtime_resume:
            if failed_resume_startup_pending(cp, after, leases):
                raise BlockedError("child exited before native claim")
            if (
                observed_lease_owner is not None
                and after.get("owner_token") != observed_lease_owner
            ):
                raise BlockedError("failed runtime resume claim owner differs")
            if native_owner is not None and after.get("owner_token") != native_owner:
                raise BlockedError("failed runtime resume native owner changed")
        if capacity_checkpoint is not None and (
            any(
                after.get(key) != capacity_checkpoint.get(key)
                for key in (
                    "brain_id",
                    "run_id",
                    "owner_token",
                    "options_fingerprint",
                    "reference_time",
                )
            )
            or (
                "--resume" in command
                and any(
                    after.get(key) != cp.get(key)
                    for key in ("brain_id", "run_id", "options_fingerprint", "reference_time")
                )
            )
        ):
            raise BlockedError("capacity stop checkpoint scope changed")
        if handoff_authorization is not None:
            await _verify_handoff_brain(
                storage, handoff_authorization["completed_semantic_handoff"]["approval"]
            )
        terminal(
            after,
            rc,
            cp,
            "--resume" in command,
            text,
            fresh_all_options_fingerprint=fresh_all_options,
        )
        receipt.update(
            phase="terminal",
            status=after["status"],
            cli_rc=rc,
            run_id=after["run_id"],
            checkpoint_sha256=checkpoint_hash(after),
            finished_at=utcnow(),
            resource_baseline=safety.baseline,
            resource_policy_sha256=safety.policy_sha256,
        )
        anchor = native_handoff_anchor(after)
        if anchor is not None:
            receipt["native_handoff_anchor"] = anchor
        proof = confirmed_success(receipt, after)
        if proof is not None:
            receipt["confirmed_success"] = proof
        api.cas(cm, receipt)
        print(MARKER + canonical(receipt), flush=True)
        return 0
    except DeferredError as exc:
        observed = {
            "outcome": "deferred_capacity",
            "reason": str(exc),
            "observed_at": utcnow(),
            **api.identity(),
        }
        api.cas(cm, receipt, observed)
        print(MARKER + canonical(receipt), flush=True)
        print("HOURLY_OBSERVATION " + canonical(observed), flush=True)
        return 0
    except Exception as exc:
        if child is not None:
            stop(child)
        reason = str(exc) if isinstance(exc, BlockedError) else type(exc).__name__
        if (
            capacity_checkpoint is not None
            and reason == "CLI exit disagrees with terminal checkpoint"
            and type(receipt.get("cli_rc")) is int
            and receipt["cli_rc"] == 130
        ):
            receipt["secondary_reason"] = reason
            reason = receipt["capacity_stop_reason"]
        # Never reread and overwrite another pod's CAS winner.
        if owns or receipt.get("phase") not in {"started", "blocked"}:
            receipt.update(
                phase="blocked",
                blocked_at=utcnow(),
                reason=reason,
            )
            try:
                api.cas(cm, receipt)
            except Exception:
                pass
        if receipt.get("secondary_reason"):
            print("HOURLY_SECONDARY_BLOCKED " + receipt["secondary_reason"], flush=True)
        print("HOURLY_BLOCKED " + reason, flush=True)
        return 2


async def main(argv: list[str] | None = None) -> int:
    from surreal_memory.unified_config import get_config, get_shared_storage

    if os.environ.get("SURREAL_MEMORY_STORAGE") != "surrealdb":
        raise BlockedError("persistent backend required")
    get_config()
    get_config(reload=True)
    parser = argparse.ArgumentParser(description=__doc__)
    operator = parser.add_mutually_exclusive_group()
    operator.add_argument("--rebaseline-resources", action="store_true")
    operator.add_argument(
        "--admit-capacity-stop",
        action="store_true",
        help="Archive an approved blocked CLI130 capacity stop and seed one native resume",
    )
    operator.add_argument("--admit-lease-loss-resume", action="store_true")
    operator.add_argument("--admit-completed-semantic-handoff", action="store_true")
    operator.add_argument("--admit-supervision-timeout", action="store_true")
    operator.add_argument("--admit-forced-supervision-timeout", action="store_true")
    parser.add_argument("--approval-file", help="Reviewed exact-state operator admission JSON")
    parser.add_argument("--confirm-run-id")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--previous-receipt-sha256")
    parser.add_argument("--previous-memory-limit-bytes", type=int)
    args = parser.parse_args(argv or [])
    proof = (
        args.confirm_run_id,
        args.checkpoint_sha256,
        args.previous_receipt_sha256,
        args.previous_memory_limit_bytes,
    )
    if args.rebaseline_resources != all(value is not None for value in proof) or (
        not args.rebaseline_resources and any(value is not None for value in proof)
    ):
        parser.error("resource rebaseline requires all explicit predecessor proof arguments")
    if (
        args.admit_capacity_stop
        or args.admit_lease_loss_resume
        or args.admit_completed_semantic_handoff
        or args.admit_supervision_timeout
        or args.admit_forced_supervision_timeout
    ) != (args.approval_file is not None):
        parser.error("operator admission requires its mode and --approval-file")
    api = Kubernetes()
    safety = Safety(api)
    storage: Any
    if (
        args.rebaseline_resources
        or args.admit_capacity_stop
        or args.admit_lease_loss_resume
        or args.admit_completed_semantic_handoff
        or args.admit_supervision_timeout
        or args.admit_forced_supervision_timeout
    ):
        # A normal storage factory initializes schema and can migrate brain data.
        # The operator path must only authenticate/select the existing database.
        from surreal_memory.storage.surrealdb import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("default")
        try:
            await storage._reconnect()
            if args.admit_forced_supervision_timeout:
                return await admit_forced_supervision_timeout(api, storage, safety, args)
            if args.admit_supervision_timeout:
                return await admit_supervision_timeout(api, storage, safety, args)
            if args.admit_completed_semantic_handoff:
                return await admit_completed_semantic_handoff(api, storage, safety, args)
            if args.admit_capacity_stop:
                return await admit_capacity_stop(api, storage, safety, args)
            if args.admit_lease_loss_resume:
                return await admit_lease_loss_resume(api, storage, safety, args)
            return await rebaseline_resources(api, storage, safety, args)
        finally:
            await storage.close()
    storage = await get_shared_storage("default")
    return await run(api, storage, safety)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
