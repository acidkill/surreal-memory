"""Offline safety checks for the disposable repair benchmark."""

import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_semantic_discovery_prune as bench


def test_benchmark_requires_registered_target(monkeypatch):
    monkeypatch.setattr(bench, "load_target", lambda: None)
    with pytest.raises(RuntimeError, match="registered disposable"):
        bench._benchmark_target()


def test_benchmark_requires_matching_container_pid(tmp_path, monkeypatch):
    descriptor = tmp_path / "target.json"
    descriptor.write_text(json.dumps({"container_id": "synthetic"}))
    monkeypatch.setenv("SMEM_TEST_SURREALDB_TARGET_FILE", str(descriptor))
    monkeypatch.setenv("SURREALDB_CONTAINER_PID", "11")
    monkeypatch.setattr(bench, "load_target", lambda: SimpleNamespace())
    monkeypatch.setattr(bench, "_inspect", lambda _: {"State": {"Pid": 12}})
    with pytest.raises(RuntimeError, match="measurement process"):
        bench._benchmark_target()


def test_repair_scan_plan_does_not_invent_physical_metrics():
    scans = bench._physical_scans({"operator": "IndexScan", "metrics": None})
    assert scans == [
        {"operator": "IndexScan", "context": None, "attributes": None, "metrics": None}
    ]


def test_repair_page_rejects_full_rows_and_overlarge_or_non_id_responses():
    identifier = SimpleNamespace(table_name="fiber")
    page = {"updated_ids": [identifier] * 8, "has_more": True}
    assert bench._repair_page([{"result": [page]}]) == page
    with pytest.raises(AssertionError):
        bench._repair_page([{"result": [{"synapse_ids": ["x"]}, page]}])
    with pytest.raises(AssertionError):
        bench._repair_page([{"result": [{"updated_ids": [identifier] * 9, "has_more": True}]}])
    with pytest.raises(AssertionError):
        bench._repair_page([{"result": [{"updated_ids": [{"id": "fiber:x"}], "has_more": False}]}])


@pytest.mark.parametrize(
    "transport,module_name", [("http", "async_http"), ("websocket", "async_ws")]
)
@pytest.mark.asyncio
async def test_wire_count_records_original_cbor_bytes(monkeypatch, transport, module_name):
    import asyncio

    module = pytest.importorskip(f"surrealdb.connections.{module_name}")
    monkeypatch.setattr(module, "decode", lambda raw: {"decoded": len(raw)})

    async def query(_sql, _params):
        async def receive():
            return module.decode(b"\x01\x02\x03\x04")

        return await asyncio.create_task(receive())

    result, size, observed = await bench._wire_counted_query(query, "synthetic", {})
    assert result == {"decoded": 4}
    assert size == 4
    assert observed == transport


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.asyncio
async def test_wire_count_rejects_zero_or_multiple_decodes(monkeypatch, count):
    module = pytest.importorskip("surrealdb.connections.async_ws")
    monkeypatch.setattr(module, "decode", lambda raw: raw)

    async def query(_sql, _params):
        for _ in range(count):
            module.decode(b"data")
        return {"result": "synthetic"}

    with pytest.raises(RuntimeError, match="one CBOR response"):
        await bench._wire_counted_query(query, "synthetic", {})


@pytest.mark.asyncio
async def test_wire_count_preserves_query_error():
    pytest.importorskip("surrealdb.connections.async_ws")

    async def query(_sql, _params):
        raise ConnectionError("original query failure")

    with pytest.raises(ConnectionError, match="original query failure"):
        await bench._wire_counted_query(query, "synthetic", {})


def test_sampler_process_loss_reports_and_preserves_query_error(monkeypatch, capsys):
    import threading

    failed = threading.Event()
    calls = 0

    def rss(_pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 100
        failed.set()
        raise FileNotFoundError("synthetic process vanished")

    monkeypatch.setattr(bench, "_rss_kib", rss)

    def query():
        with bench._RssSampler(123):
            assert failed.wait(1)
            raise ConnectionError("original query failure")

    with pytest.raises(ConnectionError, match="original query failure"):
        query()
    diagnostic = capsys.readouterr().out
    assert "RSS sampler process unavailable" in diagnostic
    assert "Traceback" not in diagnostic


def test_sampler_process_loss_fails_without_query_error(monkeypatch, capsys):
    import threading

    failed = threading.Event()
    calls = 0

    def rss(_pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 100
        failed.set()
        raise FileNotFoundError("synthetic process vanished")

    monkeypatch.setattr(bench, "_rss_kib", rss)
    with pytest.raises(RuntimeError, match="peak cannot be confirmed"):
        with bench._RssSampler(123):
            assert failed.wait(1)
    assert "RSS sampler process unavailable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mode,oom,marker,expected_rc",
    [
        ("repair-legacy", "true", True, 0),
        ("repair-legacy", "true", False, 1),
        ("repair-legacy", "false", True, 1),
        ("repair-bounded", "true", True, 1),
    ],
)
def test_legacy_oom_acceptance_requires_measurement_marker(
    tmp_path, mode, oom, marker, expected_rc
):
    import subprocess
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    condition = next(
        line.strip()
        for line in workflow.splitlines()
        if "measurement_started" in line and "failed=1" in line
    )
    report = tmp_path / "report.jsonl"
    report.write_text('{"measurement_started": "repair-legacy"}\n' if marker else "")
    result = subprocess.run(  # noqa: S603 - fixed shell and CI condition from this repository
        [
            "/bin/bash",
            "-c",
            "mode=$1; oom=$2; report=$3; diagnostic=0; failed=0\n" + condition + '\nexit "$failed"',
            "synthetic",
            mode,
            oom,
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == expected_rc, result.stderr


def test_scan_evidence_requires_measured_index_output_and_discloses_read_gap():
    evidence = bench._repair_scan_evidence({"operator": "IndexScan", "metrics": {"output_rows": 9}})
    assert evidence["indexed_scan_output_bound"] is True
    assert evidence["scan_output_rows"] == 9
    assert evidence["physical_read_bound"] is None
    assert (
        bench._repair_scan_evidence({"operator": "IndexScan", "metrics": {"output_rows": 10}})[
            "indexed_scan_output_bound"
        ]
        is False
    )
    assert (
        bench._repair_scan_evidence({"operator": "IndexScan", "metrics": None})[
            "indexed_scan_output_bound"
        ]
        is False
    )


def test_pure_helpers_import_without_surrealdb_sdk():
    import subprocess
    import sys
    from pathlib import Path

    code = """
import importlib.abc
import sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'surrealdb' or fullname.startswith('surrealdb.'):
            raise ModuleNotFoundError('SDK intentionally unavailable', name=fullname)
sys.meta_path.insert(0, BlockSDK())
from scripts import benchmark_semantic_discovery_prune as benchmark
assert benchmark._repair_scan_evidence({'operator': 'IndexScan', 'metrics': None})['physical_read_bound'] is None
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test code
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_repair_marker_follows_setup_and_high_water_reset():
    import inspect

    source = inspect.getsource(bench._run_repair)
    assert source.index("_reset_hwm(pid)") < source.index('"measurement_started"')
    assert source.index('"measurement_started"') < source.index("with sampler:")


def test_sampler_preserves_partial_peak_in_json_artifact_on_process_loss(monkeypatch, capsys):
    import threading

    failed = threading.Event()
    calls = 0

    def rss(_pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 321
        failed.set()
        raise FileNotFoundError("synthetic process vanished")

    monkeypatch.setattr(bench, "_rss_kib", rss)

    def query():
        with bench._RssSampler(123):
            assert failed.wait(1)
            raise ConnectionError("original RPC failure")

    with pytest.raises(ConnectionError, match="original RPC failure"):
        query()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[-1]["rss_sampled_peak_kib"] == 321
    assert records[-1]["rss_sample_count"] == 1
    assert records[-1]["rss_sampling_complete"] is False


def test_failed_repair_diagnostic_retains_stage_baseline_and_partial_peak(monkeypatch, capsys):
    def disappeared(_pid):
        raise FileNotFoundError("synthetic dead process")

    monkeypatch.setattr(bench, "_start_ticks", disappeared)
    sampler = SimpleNamespace(samples_kib=[100, 143], sample_error=FileNotFoundError())
    state = {
        "mode": "repair-bounded",
        "stage": "transaction_started",
        "transaction_index": 1,
        "rss_baseline_kib": 100,
        "memory_limit_bytes": 1024**3,
        "lookup_scan_evidence": {"scan_output_rows": 9},
    }
    bench._repair_diagnostic(
        state, 123, "original", sampler=sampler, failure_error_type="ConnectionError"
    )
    event = json.loads(capsys.readouterr().out)
    assert event["stage"] == "transaction_started" and event["transaction_index"] == 1
    assert event["rss_baseline_kib"] == 100
    assert event["rss_sampled_peak_kib"] == 143
    assert event["rss_sampled_peak_delta_kib"] == 43
    assert event["rss_kernel_hwm_kib"] is None
    assert event["rss_current_kib"] is None
    assert event["rss_observation_error"] == "FileNotFoundError"
    assert event["lookup_scan_evidence"] == {"scan_output_rows": 9}


@pytest.mark.parametrize("prior_failure", [0, 1])
def test_diagnostic_failure_preserves_original_acceptance_latch(tmp_path, prior_failure):
    import subprocess
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text()
    condition = next(
        line.strip()
        for line in workflow.splitlines()
        if "measurement_started" in line and "failed=1" in line
    )
    report = tmp_path / "report.jsonl"
    report.write_text("")
    result = subprocess.run(  # noqa: S603 - fixed shell and CI condition from this repository
        [
            "/bin/bash",
            "-c",
            "mode=repair-bounded; oom=true; diagnostic=1; report=$1; failed=$2\n"
            + condition
            + '\nexit "$failed"',
            "synthetic",
            str(report),
            str(prior_failure),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == prior_failure, result.stderr


def test_failed_process_diagnostic_retains_cached_cgroup_peak(monkeypatch, capsys, tmp_path):
    def disappeared(_pid):
        raise FileNotFoundError("synthetic dead process")

    monkeypatch.setattr(bench, "_start_ticks", disappeared)
    (tmp_path / "memory.current").write_text("1024\n")
    (tmp_path / "memory.peak").write_text("1073741824\n")
    (tmp_path / "memory.events").write_text("oom 1\noom_kill 1\n")
    bench._repair_diagnostic({}, 123, "original", cgroup=tmp_path)
    event = json.loads(capsys.readouterr().out)
    assert event["rss_kernel_hwm_kib"] is None
    assert event["cgroup_memory_peak_bytes"] == 1024**3
    assert event["cgroup_memory_events"] == {"oom": 1, "oom_kill": 1}
    assert event["rss_sampled_peak_kib"] is None
    assert event["rss_sampled_peak_delta_kib"] is None
    assert event["rss_sampling_complete"] is False
