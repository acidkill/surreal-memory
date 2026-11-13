#!/usr/bin/env python3
"""Fail-closed hourly supervisor; never edits consolidation checkpoints or budgets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import signal
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from surreal_memory.engine.consolidation_receipts import (
    MARKER,
    STATE,
    BlockedError,
    canonical,
    checkpoint_hash,
    confirmed_success,
    resource_rebaseline_seed,
    select_command,
    success_timestamp,
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

    def previous_log(self, receipt: dict[str, Any]) -> str:
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
        if status.get("active", 0) or status.get("failed", 0) or status.get("succeeded") != 1:
            raise BlockedError("previous job has not succeeded")
        if pod.get("status", {}).get("phase") != "Succeeded":
            raise BlockedError("previous pod has not succeeded")
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
        pod = self.api.request(self.api.path("pods/" + os.environ["DB_POD"]))
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
    leases = await storage._query(
        "SELECT * FROM consolidation_lease WHERE brain_id = $brain_id AND expires_at > time::now()",
        brain_id="default",
    )
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
        safety.sample(initial=True)
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
        receipt = {
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
                    current, active = await live(storage)
                    if current and (
                        current.get("last_error")
                        or current.get("status") in {"failed", "lease_lost", "timed_out"}
                    ):
                        raise BlockedError("live checkpoint reports failure")
                    if (
                        current
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
                    stop(child)
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
        terminal(after, rc, cp, "--resume" in command, text)
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
            or ("--resume" in command and after.get("reference_time") != cp.get("reference_time"))
        ):
            raise BlockedError("capacity stop checkpoint scope changed")
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
        # Never reread and overwrite another pod's CAS winner.
        if owns or receipt.get("phase") not in {"started", "blocked"}:
            receipt.update(
                phase="blocked",
                blocked_at=utcnow(),
                reason=str(exc) if isinstance(exc, BlockedError) else type(exc).__name__,
            )
            try:
                api.cas(cm, receipt)
            except Exception:
                pass
        print(
            "HOURLY_BLOCKED " + (str(exc) if isinstance(exc, BlockedError) else type(exc).__name__),
            flush=True,
        )
        return 2


async def main(argv: list[str] | None = None) -> int:
    from surreal_memory.unified_config import get_config, get_shared_storage

    if os.environ.get("SURREAL_MEMORY_STORAGE") != "surrealdb":
        raise BlockedError("persistent backend required")
    get_config()
    get_config(reload=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebaseline-resources", action="store_true")
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
    api = Kubernetes()
    safety = Safety(api)
    storage: Any
    if args.rebaseline_resources:
        # A normal storage factory initializes schema and can migrate brain data.
        # The operator path must only authenticate/select the existing database.
        from surreal_memory.storage.surrealdb import SurrealDBStorage

        storage = SurrealDBStorage()
        storage.set_brain("default")
        try:
            await storage._reconnect()
            return await rebaseline_resources(api, storage, safety, args)
        finally:
            await storage.close()
    storage = await get_shared_storage("default")
    return await run(api, storage, safety)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
