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


@pytest.mark.asyncio
async def test_wire_count_records_original_cbor_bytes(monkeypatch):
    from surrealdb.connections import async_http

    monkeypatch.setattr(async_http, "decode", lambda raw: {"decoded": len(raw)})

    async def query(_sql, _params):
        return async_http.decode(b"\x01\x02\x03\x04")

    result, size = await bench._wire_counted_query(query, "synthetic", {})
    assert result == {"decoded": 4}
    assert size == 4


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
