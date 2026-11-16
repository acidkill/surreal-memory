"""Offline regressions for reviewed capacity incarnation rebind."""

from __future__ import annotations

import copy
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock
from urllib.error import HTTPError, URLError

import pytest

from surreal_memory.engine import consolidation_receipts as receipts
from tests.unit.test_capacity_stop_admission import rebind_evidence
from tests.unit.test_capacity_stop_admission import (
    synthetic_runtime_versions as synthetic_runtime_versions,
)
from tests.unit.test_cluster_consolidation_hourly import hourly


def http_error(code):
    return HTTPError("https://synthetic.invalid", code, "synthetic", {}, None)


def archived_api(approval, pod_result=None, job_result=None):
    previous = json.loads(rebind_evidence()[0])
    archive = approval["incarnation_rebind"]["archived_predecessor"]
    pod = json.loads(archive["pod_raw"])
    job = json.loads(archive["job_raw"])
    api = hourly.Kubernetes.__new__(hourly.Kubernetes)
    api.namespace = "test"
    api.request = MagicMock(side_effect=[pod_result or http_error(404), job_result or job])
    return api, previous, pod, job


def test_archived_predecessor_only_after_typed_pod_404():
    _, log, _, approval = rebind_evidence()
    api, previous, _, _ = archived_api(approval)
    assert hourly.capacity_predecessor_log(api, previous, approval) == log
    assert api.request.call_count == 2


@pytest.mark.parametrize(
    "error", [http_error(403), http_error(500), TimeoutError(), URLError("offline")]
)
def test_archive_never_masks_unavailable_pod(error):
    _, _, _, approval = rebind_evidence()
    api, previous, _, _ = archived_api(approval, error)
    with pytest.raises((receipts.BlockedError, HTTPError, URLError, TimeoutError)):
        hourly.capacity_predecessor_log(api, previous, approval)
    assert api.request.call_count == 1


@pytest.mark.parametrize(
    "fault", ["job_404", "job_uid", "job_active", "archive_hash", "archive_pod", "archive_log"]
)
def test_archived_predecessor_rejects_untrusted_evidence(fault):
    raw, _, cp, approval = rebind_evidence()
    archive = approval["incarnation_rebind"]["archived_predecessor"]
    api, previous, pod, job = archived_api(approval)
    if fault == "job_404":
        api.request.side_effect = [http_error(404), http_error(404)]
    elif fault == "job_uid":
        job["metadata"]["uid"] = "wrong-job"
    elif fault == "job_active":
        job["status"]["active"] = 1
    elif fault == "archive_hash":
        archive["pod_sha256"] = "0" * 64
    elif fault == "archive_pod":
        pod["metadata"]["uid"] = "wrong-pod"
        archive["pod_raw"] = receipts.canonical(pod)
        archive["pod_sha256"] = hashlib.sha256(archive["pod_raw"].encode()).hexdigest()
    else:
        archive["log"] += "tampered"
        archive["log_sha256"] = hashlib.sha256(archive["log"].encode()).hexdigest()
    with pytest.raises((receipts.BlockedError, HTTPError, URLError, TimeoutError)):
        receipts.capacity_resume_seed(
            raw, hourly.capacity_predecessor_log(api, previous, approval), cp, approval=approval
        )


@pytest.mark.parametrize("fault", ["live_uid", "live_exit"])
def test_live_predecessor_mismatch_does_not_fall_back(fault):
    _, _, _, approval = rebind_evidence()
    api, previous, pod, job = archived_api(approval)
    if fault == "live_uid":
        pod["metadata"]["uid"] = "wrong-pod"
    else:
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 1
    api.request.side_effect = [pod, job]
    with pytest.raises(receipts.BlockedError):
        hourly.capacity_predecessor_log(api, previous, approval)


def database_objects(approval):
    binding = approval["incarnation_rebind"]["database"]
    pod_spec = {
        "nodeName": binding["node_name"],
        "containers": [
            {
                "name": "surrealdb",
                "image": "synthetic/db",
                "resources": {"limits": {"memory": "18Gi"}},
            }
        ],
        "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": binding["pvc_name"]}}],
    }
    rs_spec = {"replicas": 1}
    deployment_spec = {"replicas": 1}
    pvc_spec = {"volumeName": binding["pv_name"]}
    pv_spec = {
        "claimRef": {
            "namespace": binding["namespace"],
            "name": binding["pvc_name"],
            "uid": binding["pvc_uid"],
        }
    }
    pod = {
        "metadata": {
            "name": binding["pod_name"],
            "namespace": binding["namespace"],
            "uid": binding["pod_uid"],
            "ownerReferences": [
                {
                    "kind": "ReplicaSet",
                    "name": binding["replica_set_name"],
                    "uid": binding["replica_set_uid"],
                    "controller": True,
                }
            ],
        },
        "spec": pod_spec,
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "surrealdb",
                    "ready": True,
                    "restartCount": 0,
                    "containerID": "new-container",
                    "imageID": binding["image_id"],
                }
            ],
        },
    }
    rs = {
        "metadata": {
            "name": binding["replica_set_name"],
            "uid": binding["replica_set_uid"],
            "ownerReferences": [
                {
                    "kind": "Deployment",
                    "name": binding["deployment_name"],
                    "uid": binding["deployment_uid"],
                    "controller": True,
                }
            ],
        },
        "spec": rs_spec,
    }
    deployment = {
        "metadata": {"name": binding["deployment_name"], "uid": binding["deployment_uid"]},
        "spec": deployment_spec,
    }
    pvc = {"metadata": {"name": binding["pvc_name"], "uid": binding["pvc_uid"]}, "spec": pvc_spec}
    pv = {"metadata": {"name": binding["pv_name"], "uid": binding["pv_uid"]}, "spec": pv_spec}
    pvc["status"] = pv["status"] = {"phase": "Bound"}
    objects = {"pod": pod, "replica_set": rs, "deployment": deployment, "pvc": pvc, "pv": pv}
    for name in ("pod", "replica_set", "deployment", "pvc"):
        objects[name]["metadata"]["namespace"] = binding["namespace"]
    for name, obj in objects.items():
        obj["metadata"]["creationTimestamp"] = "2025-12-01T00:00:00Z"
        binding[name + "_spec_sha256"] = receipts.checkpoint_hash(obj["spec"])
    pod["metadata"]["creationTimestamp"] = "2026-01-01T01:40:00Z"
    old_pod = copy.deepcopy(pod)
    old_pod["metadata"].update(
        name="historical-db", uid="same-db-uid", creationTimestamp="2026-01-01T00:00:00Z"
    )
    old_pod["status"]["containerStatuses"][0].update(containerID="same-container", restartCount=2)
    old_raw = receipts.canonical(old_pod)
    approval["incarnation_rebind"]["previous_database"] = {
        "pod_raw": old_raw,
        "pod_sha256": hashlib.sha256(old_raw.encode()).hexdigest(),
    }
    return objects


def database_api(objects, binding):
    api = hourly.Kubernetes.__new__(hourly.Kubernetes)
    api.namespace = binding["namespace"]
    paths = {
        api.path("pods/" + binding["pod_name"]): objects["pod"],
        "/apis/apps/v1/namespaces/test/replicasets/" + binding["replica_set_name"]: objects[
            "replica_set"
        ],
        "/apis/apps/v1/namespaces/test/deployments/" + binding["deployment_name"]: objects[
            "deployment"
        ],
        api.path("persistentvolumeclaims/" + binding["pvc_name"]): objects["pvc"],
        "/api/v1/persistentvolumes/" + binding["pv_name"]: objects["pv"],
        "/api/v1/nodes/" + binding["node_name"]: {
            "metadata": {"name": binding["node_name"]},
            "status": {"conditions": [{"type": "Ready", "status": "True"}]},
        },
    }
    api.request = MagicMock(side_effect=lambda path, **kwargs: copy.deepcopy(paths[path]))
    return api


def test_live_database_projection_uses_spec_not_volatile_metadata(monkeypatch):
    _, _, _, approval = rebind_evidence()
    objects = database_objects(approval)
    binding = approval["incarnation_rebind"]["database"]
    monkeypatch.setenv("DB_POD", binding["pod_name"])
    for obj in objects.values():
        obj["metadata"]["resourceVersion"] = "synthetic-changing-version"
    assert hourly.capacity_database_binding(database_api(objects, binding), binding) == binding


@pytest.mark.parametrize(
    "fault", ["owner_uid", "pvc_uid", "pv_claim_uid", "deployment_spec", "image"]
)
def test_live_database_projection_rejects_identity_and_storage_drift(fault, monkeypatch):
    _, _, _, approval = rebind_evidence()
    objects = database_objects(approval)
    binding = approval["incarnation_rebind"]["database"]
    monkeypatch.setenv("DB_POD", binding["pod_name"])
    if fault == "owner_uid":
        objects["pod"]["metadata"]["ownerReferences"][0]["uid"] = "different"
    elif fault == "pvc_uid":
        objects["pvc"]["metadata"]["uid"] = "different"
    elif fault == "pv_claim_uid":
        objects["pv"]["spec"]["claimRef"]["uid"] = "different"
    elif fault == "deployment_spec":
        objects["deployment"]["spec"]["replicas"] = 2
    else:
        objects["pod"]["status"]["containerStatuses"][0]["imageID"] = "different"
    api = database_api(objects, binding)
    try:
        actual = hourly.capacity_database_binding(api, binding)
    except receipts.BlockedError:
        return
    assert actual != binding


def rebind_operator(tmp_path, monkeypatch):
    from dataclasses import asdict
    from io import BytesIO
    from pathlib import Path
    from urllib.parse import parse_qs, urlsplit

    from surreal_memory.core.brain import Brain, BrainConfig
    from tests.unit.test_capacity_stop_admission import operator_fixture

    api, storage, _, args, cm, cp, _ = operator_fixture(tmp_path)
    raw, _, _, approval = rebind_evidence()
    cm["data"]["receipt.json"] = raw
    objects = database_objects(approval)
    binding = approval["incarnation_rebind"]["database"]
    brain = Brain(
        id="synthetic-brain-uuid", name="default", config=BrainConfig(activation_strategy="hybrid")
    )
    approval["incarnation_rebind"]["brain"]["config_sha256"] = receipts.checkpoint_hash(
        asdict(brain.config)
    )
    storage.get_brain = AsyncMock(return_value=brain)
    Path(args.approval_file).write_text(json.dumps(approval))
    api.namespace = "test"
    api.path.side_effect = lambda resource: "/api/v1/namespaces/test/" + resource
    api.previous_log.side_effect = lambda previous, expected_exit_code: (
        hourly.Kubernetes.previous_log(api, previous, expected_exit_code=expected_exit_code)
    )
    api.no_other_jobs.side_effect = lambda uid: hourly.Kubernetes.no_other_jobs(api, uid)
    db_api = database_api(objects, binding)
    operator = {
        "metadata": {"uid": "operator-pod"},
        "spec": {"containers": [{"image": "registry/smem@" + approval["runtime"]["image_digest"]}]},
        "status": {
            "containerStatuses": [
                {"imageID": "registry/smem@" + approval["runtime"]["image_digest"]}
            ]
        },
    }
    archived_job = json.loads(approval["incarnation_rebind"]["archived_predecessor"]["job_raw"])

    def request(path, **kwargs):
        if path == api.path("pods/failed-pod"):
            raise http_error(404)
        if path == "/apis/batch/v1/namespaces/test/jobs/failed-job":
            return copy.deepcopy(archived_job)
        if path == "/apis/batch/v1/namespaces/test/jobs":
            return {"items": [{"metadata": {"uid": "operator-job"}, "status": {"active": 1}}]}
        if path == api.path("pods/operator"):
            return copy.deepcopy(operator)
        return db_api.request(path, **kwargs)

    api.request.side_effect = request
    monkeypatch.setenv("DB_POD", binding["pod_name"])
    monkeypatch.setenv("PROMETHEUS_URL", "https://synthetic.invalid")
    for key in ("current", "working", "available", "oom", "failcnt"):
        monkeypatch.setenv("PROM_QUERY_" + key.upper(), key)

    def metrics(url, timeout):
        query = parse_qs(urlsplit(url).query)["query"][0]
        now = hourly.time.time()
        value = (
            now
            if query.startswith("timestamp(")
            else {
                "current": 1024,
                "working": 512,
                "available": 8 * hourly.GIB,
                "oom": 0,
                "failcnt": 0,
            }[query]
        )
        return BytesIO(
            json.dumps(
                {"status": "success", "data": {"result": [{"value": [now, str(value)]}]}}
            ).encode()
        )

    monkeypatch.setattr(hourly.urllib.request, "urlopen", metrics)
    safety = hourly.Safety.__new__(hourly.Safety)
    safety.api = api
    safety.baseline = None
    safety.memory_limit_bytes = approval["memory_limit_bytes"]
    safety.policy_sha256 = approval["resource_policy_sha256"]
    return api, storage, safety, args, cm, cp, approval, objects


async def test_full_v2_admission_and_single_native_resume(
    tmp_path, monkeypatch, synthetic_runtime_versions
):
    from unittest.mock import patch

    api, storage, safety, args, cm, cp, approval, _ = rebind_operator(tmp_path, monkeypatch)
    before_cp = copy.deepcopy(cp)
    old = cm["data"]["receipt.json"]
    with patch.object(hourly.subprocess, "Popen") as launch:
        assert await hourly.admit_capacity_stop(api, storage, safety, args) == 0
    launch.assert_not_called()
    assert cp == before_cp
    seed = json.loads(cm["data"]["receipt.json"])
    key = "capacity_resume_" + receipts.checkpoint_hash(seed) + ".json"
    assert cm["data"][key] == receipts.canonical(seed)
    assert seed["capacity_resume"]["previous_receipt_raw"] == old
    assert json.loads(old)["cli_rc"] == 130 and seed["cli_rc"] is None
    child = MagicMock(returncode=1)
    child.poll.return_value = child.wait.return_value = 1
    with patch.object(hourly.subprocess, "Popen", return_value=child) as launch:
        assert await hourly.run(api, storage, safety) == 0
    launch.assert_called_once()
    assert launch.call_args.args[0][:5] == ["smem", "consolidate", "--brain", "default", "--resume"]
    assert cm["data"][key] == receipts.canonical(seed)
    assert cp == before_cp


@pytest.mark.parametrize(
    "fault",
    ["brain", "checkpoint", "runtime", "owner", "unready", "restart", "oom", "stale_metrics"],
)
async def test_v2_admission_refuses_drift_before_cas(
    tmp_path, monkeypatch, synthetic_runtime_versions, fault
):
    from dataclasses import replace
    from pathlib import Path
    from unittest.mock import patch

    api, storage, safety, args, cm, cp, approval, objects = rebind_operator(tmp_path, monkeypatch)
    old = copy.deepcopy(cm)
    if fault == "brain":
        storage.get_brain.return_value = replace(storage.get_brain.return_value, id="different")
    elif fault == "checkpoint":
        cp["status"] = "different"
    elif fault == "runtime":
        approval["runtime"]["runner_sha256"] = "0" * 64
        Path(args.approval_file).write_text(json.dumps(approval))
    elif fault == "owner":
        objects["pod"]["metadata"]["ownerReferences"][0]["uid"] = "different"
    elif fault == "unready":
        objects["pod"]["status"]["containerStatuses"][0]["ready"] = False
    elif fault == "restart":
        objects["pod"]["status"]["containerStatuses"][0]["restartCount"] = 1
    elif fault == "oom":
        safety.baseline = (("new-db-uid", '[["surrealdb","new-container",0]]'), 1.0, 0.0)
    else:
        monkeypatch.setattr(
            hourly.urllib.request,
            "urlopen",
            lambda *args, **kwargs: __import__("io").BytesIO(
                b'{"status":"success","data":{"result":[{"value":[0,"0"]}]}}'
            ),
        )
    with patch.object(hourly.subprocess, "Popen") as launch:
        with pytest.raises(receipts.BlockedError):
            await hourly.admit_capacity_stop(api, storage, safety, args)
    launch.assert_not_called()
    api.cas.assert_not_called()
    assert cm == old


async def test_v2_uncertain_cas_is_not_retried(tmp_path, monkeypatch, synthetic_runtime_versions):
    api, storage, safety, args, cm, _, _, _ = rebind_operator(tmp_path, monkeypatch)
    original_cas = api.cas.side_effect

    def uncertain(*args):
        original_cas(*args)
        raise TimeoutError("synthetic ambiguous response")

    api.cas.side_effect = uncertain
    with pytest.raises(receipts.BlockedError, match="uncertain"):
        await hourly.admit_capacity_stop(api, storage, safety, args)
    assert api.cas.call_count == 1
    assert json.loads(cm["data"]["receipt.json"])["cli_rc"] is None


async def test_v2_post_cas_binding_drift_prevents_success(
    tmp_path, monkeypatch, synthetic_runtime_versions
):
    api, storage, safety, args, _, _, _, objects = rebind_operator(tmp_path, monkeypatch)
    original_cas = api.cas.side_effect

    def change_binding(*args):
        result = original_cas(*args)
        objects["pv"]["metadata"]["uid"] = "different"
        return result

    api.cas.side_effect = change_binding
    with pytest.raises(receipts.BlockedError, match="uncertain"):
        await hourly.admit_capacity_stop(api, storage, safety, args)
    assert api.cas.call_count == 1


async def test_v2_prelaunch_drift_after_worker_cas_never_starts_child(
    tmp_path, monkeypatch, synthetic_runtime_versions
):
    from unittest.mock import patch

    api, storage, safety, args, _, _, _, objects = rebind_operator(tmp_path, monkeypatch)
    assert await hourly.admit_capacity_stop(api, storage, safety, args) == 0
    original_cas = api.cas.side_effect

    def drift(*args):
        result = original_cas(*args)
        objects["deployment"]["spec"]["replicas"] = 2
        return result

    api.cas.side_effect = drift
    with patch.object(hourly.subprocess, "Popen") as launch:
        assert await hourly.run(api, storage, safety) == 2
    launch.assert_not_called()


async def test_v2_admission_cannot_be_reused(tmp_path, monkeypatch, synthetic_runtime_versions):
    api, storage, safety, args, cm, _, _, _ = rebind_operator(tmp_path, monkeypatch)
    assert await hourly.admit_capacity_stop(api, storage, safety, args) == 0
    saved = copy.deepcopy(cm)
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_capacity_stop(api, storage, safety, args)
    assert cm == saved


async def test_v2_binding_drift_between_preflight_checks_prevents_cas(
    tmp_path, monkeypatch, synthetic_runtime_versions
):
    api, storage, safety, args, cm, _, approval, objects = rebind_operator(tmp_path, monkeypatch)
    original_request = api.request.side_effect
    original_state = copy.deepcopy(cm)
    reads = 0

    def drift(path, **kwargs):
        nonlocal reads
        if path == api.path("pods/" + approval["incarnation_rebind"]["database"]["pod_name"]):
            reads += 1
            if reads == 4:
                objects["pv"]["metadata"]["uid"] = "different"
        return original_request(path, **kwargs)

    api.request.side_effect = drift
    with pytest.raises(receipts.BlockedError):
        await hourly.admit_capacity_stop(api, storage, safety, args)
    assert reads >= 4
    api.cas.assert_not_called()
    assert cm == original_state
