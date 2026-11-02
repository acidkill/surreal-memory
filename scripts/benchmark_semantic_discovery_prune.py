"""Disposable real-SurrealDB memory benchmark for snapshot cleanup."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

from surrealdb import RecordID

from surreal_memory.core.brain import Brain
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

ROWS = 512
PROTECTED_ROWS = 64
PAYLOAD_BYTES = 128 * 1024
READ_PAGE_LIMIT = 64
MAX_READ_PAGES = 4
MAX_DELETES = 8
PHYSICAL_SCAN_OPERATORS = {"TableScan", "RecordIdScan", "IndexScan", "DynamicScan"}


def _rss_kib(pid: int) -> int:
    """Return current resident memory for the isolated SurrealDB process."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    raise RuntimeError(f"VmRSS is unavailable for benchmark process {pid}")


def _hwm_kib(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise RuntimeError(f"VmHWM is unavailable for benchmark process {pid}")


def _start_ticks(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def _reset_hwm(pid: int) -> None:
    import subprocess

    subprocess.run(
        ["/usr/bin/sudo", "-n", "/bin/sh", "-c", f"echo 5 > /proc/{pid}/clear_refs"],
        check=True,
    )


def _physical_scans(plan: Any) -> list[dict[str, Any]]:
    scans: list[dict[str, Any]] = []
    if isinstance(plan, list):
        for child in plan:
            scans.extend(_physical_scans(child))
    elif isinstance(plan, dict):
        operator = plan.get("operator")
        if isinstance(operator, str) and operator in PHYSICAL_SCAN_OPERATORS:
            scans.append(
                {
                    "operator": operator,
                    "context": plan.get("context"),
                    "attributes": plan.get("attributes"),
                    "metrics": plan.get("metrics"),
                }
            )
        for child in plan.values():
            scans.extend(_physical_scans(child))
    return scans


def _assert_physical_read_bound(evidence: dict[str, Any]) -> None:
    scans = evidence["physical_scans"]
    invalid = [
        scan
        for scan in scans
        if not isinstance(scan.get("metrics"), dict)
        or not isinstance(scan["metrics"].get("output_rows"), int)
        or scan["metrics"]["output_rows"] > READ_PAGE_LIMIT
    ]
    if not scans or invalid:
        print(
            json.dumps(
                {
                    "benchmark_failure": "physical scan read bound exceeded or unreported",
                    **evidence,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise AssertionError("physical scan output_rows must be reported and <= 64 per page")


class _RssSampler:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.samples_kib: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.samples_kib.append(_rss_kib(self.pid))
            self._stop.wait(0.01)

    def __enter__(self) -> _RssSampler:
        self.samples_kib.append(_rss_kib(self.pid))
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


async def _run(mode: str, version: str) -> dict[str, Any]:
    url = os.environ["SURREALDB_URL"]
    pid = int(os.environ["SURREALDB_CONTAINER_PID"])
    suffix = uuid.uuid4().hex[:10]
    storage = SurrealDBStorage(
        url=url,
        user=os.getenv("SURREALDB_USER", "root"),
        password=os.getenv("SURREALDB_PASS", "root"),
        namespace=f"sds_bench_{mode}_{suffix}",
        database=f"run_{suffix}",
    )
    await storage.initialize()
    brain = Brain.create(name=f"sds-prune-benchmark-{mode}-{suffix}")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    brain_id = storage._get_brain_id()
    created_at = utcnow() - timedelta(days=30)
    padding = "x" * (PAYLOAD_BYTES - 4)
    rows = [
        {
            "id": RecordID("semantic_discovery_state", f"benchmark_{index:04d}"),
            "state_id": f"benchmark-{mode}-{index:04d}",
            "revision": 1,
            "brain_id": brain_id,
            "run_id": "finished-run" if index < ROWS else "paused-run",
            "owner_token": "synthetic-owner",
            "source_token": "synthetic-source",
            "payload": {"padding": f"{index:04d}" + padding},
            "created_at": created_at,
        }
        for index in range(ROWS + PROTECTED_ROWS)
    ]
    try:
        for start in range(0, len(rows), 32):
            await storage._query(
                "INSERT INTO semantic_discovery_state $rows RETURN NONE",
                rows=rows[start : start + 32],
            )

        measured: dict[str, Any] = {"mode": mode, "db_version": version}
        if mode == "legacy":
            cutoff = utcnow() - timedelta(days=7)
            old_sql = (
                "DELETE (SELECT VALUE id FROM semantic_discovery_state "
                "WITH INDEX idx_sds_created_at WHERE created_at < $cutoff "
                "AND brain_id = $brain_id AND run_id NOT IN $keep LIMIT $limit) "
                "RETURN VALUE $before.id"
            )
            old_plan_sql = (
                "SELECT VALUE id FROM semantic_discovery_state "
                "WITH INDEX idx_sds_created_at WHERE created_at < $cutoff "
                "AND brain_id = $brain_id AND run_id NOT IN $keep LIMIT $limit"
            )
            old_params: dict[str, Any] = {
                "cutoff": cutoff,
                "brain_id": brain_id,
                "keep": ["paused-run"],
                "limit": 500,
            }
            old_plan = await SurrealDBStorage._query_response(
                storage, old_plan_sql + " EXPLAIN FULL", **old_params
            )
            old_plan_text = json.dumps(old_plan, default=str)
            baseline = _rss_kib(pid)
            process_start = _start_ticks(pid)
            _reset_hwm(pid)
            deleted = 0
            batches = 0
            with _RssSampler(pid) as sampler:
                while True:
                    result = await storage._query(old_sql, **old_params)
                    deleted += len(result)
                    batches += 1
                    if len(result) < 500:
                        break
            measured.update(
                reads_unbounded=True,
                deletes=deleted,
                delete_batches=batches,
                plan=old_plan_text,
                rss_baseline_kib=baseline,
                rss_sampled_peak_kib=max(sampler.samples_kib),
                rss_sampled_peak_delta_kib=max(sampler.samples_kib) - baseline,
                rss_kernel_hwm_kib=_hwm_kib(pid),
                rss_sample_interval_ms=10,
                rss_metric="VmHWM reset after setup; /proc VmRSS sampled during cleanup",
            )
            assert deleted == ROWS, f"legacy cleanup deleted {deleted} of {ROWS} eligible rows"
            assert _start_ticks(pid) == process_start, (
                "SurrealDB process changed during measurement"
            )
        else:
            captured: list[tuple[str, dict[str, Any], int]] = []
            original_query = storage._query

            async def capture(sql: str, **params: Any) -> list[dict[str, Any]]:
                result = await original_query(sql, **params)
                if sql.lstrip().upper().startswith(("SELECT ID,", "SELECT CURSOR FROM")):
                    captured.append((sql, params, len(result)))
                return result

            storage._query = capture  # type: ignore[method-assign]
            owner_token = suffix
            assert await storage.acquire_consolidation_lease(
                brain_id, owner_token, lease_seconds=120
            )
            warm_start = len(captured)
            warm_result = await storage.prune_semantic_discovery_state(
                older_than=timedelta(days=7),
                keep_run_ids={"paused-run", "finished-run"},
                owner_token=owner_token,
            )
            warm_scans = captured[warm_start:]
            assert warm_result["deleted"] == 0
            assert warm_result["more"]
            assert warm_scans
            warm_page_scans = [
                item for item in warm_scans if item[0].lstrip().upper().startswith("SELECT ID,")
            ]
            assert all(count <= READ_PAGE_LIMIT for _, _, count in warm_page_scans)
            measured_start = len(captured)
            baseline = _rss_kib(pid)
            process_start = _start_ticks(pid)
            _reset_hwm(pid)
            with _RssSampler(pid) as sampler:
                result = await storage.prune_semantic_discovery_state(
                    older_than=timedelta(days=7),
                    keep_run_ids={"paused-run"},
                    owner_token=owner_token,
                )
            sampled_peak = max(sampler.samples_kib)
            kernel_hwm = _hwm_kib(pid)
            assert _start_ticks(pid) == process_start, (
                "SurrealDB process changed during measurement"
            )
            storage._query = original_query  # type: ignore[method-assign]
            measured_scans = captured[measured_start:]
            page_scans = [
                item for item in measured_scans if item[0].lstrip().upper().startswith("SELECT ID,")
            ]
            cursor_reads = [
                item
                for item in measured_scans
                if item[0].lstrip().upper().startswith("SELECT CURSOR FROM")
            ]
            scan_rows = sum(count for _, _, count in page_scans)
            reads = sum(count for _, _, count in measured_scans)
            assert len(page_scans) <= MAX_READ_PAGES, f"scan exceeded {MAX_READ_PAGES} pages"
            assert len(cursor_reads) <= MAX_READ_PAGES, "cursor lookup query budget exceeded"
            assert all(count <= 1 for _, _, count in cursor_reads), (
                "cursor lookup returned multiple rows"
            )
            assert all(count <= READ_PAGE_LIMIT for _, _, count in page_scans), (
                "scan returned a page larger than its bound"
            )
            assert reads <= (READ_PAGE_LIMIT + 1) * MAX_READ_PAGES, "read budget exceeded"
            assert result["deleted"] <= MAX_DELETES, "delete budget exceeded"
            assert result["deleted"] > 0, "bounded prune did not delete an eligible row"
            assert isinstance(result["more"], bool), "prune result omitted continuation state"

            first_sql, first_params, _ = warm_page_scans[0]
            scan_sql, scan_params, _ = page_scans[0]
            page_evidence: list[dict[str, Any]] = []
            for phase, queries in (("cursor_seed", warm_page_scans), ("measured", page_scans)):
                for sql, params, result_rows in queries:
                    plan = await SurrealDBStorage._query_response(
                        storage, sql + " EXPLAIN FULL", **params
                    )
                    evidence = {
                        "phase": phase,
                        "sql": sql,
                        "result_rows": result_rows,
                        "physical_scans": _physical_scans(plan),
                        "plan": json.dumps(plan, default=str),
                    }
                    _assert_physical_read_bound(evidence)
                    page_evidence.append(evidence)
            continuation = next(item for item in page_evidence if item["phase"] == "measured")
            first_page = next(item for item in page_evidence if item["phase"] == "cursor_seed")
            plan_text = continuation["plan"]
            first_plan_text = first_page["plan"]
            if not any(
                s["operator"] in {"DynamicScan", "RecordIdScan"}
                for s in continuation["physical_scans"]
            ):
                print(
                    json.dumps(
                        {
                            "benchmark_failure": "continuation plan lacks primary record range",
                            "page_scan_evidence": page_evidence,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                raise AssertionError("continuation plan lacks a primary record range scan")
            assert "semantic_discovery_state" in scan_sql
            assert ".." in scan_sql
            measured.update(
                rows_read=reads,
                scan_rows_returned=scan_rows,
                cursor_rows_read=sum(count for _, _, count in cursor_reads),
                read_pages=len(page_scans),
                read_page_max=max((count for _, _, count in page_scans), default=0),
                cursor_seed_rows=sum(count for _, _, count in warm_page_scans),
                first_page_plan=first_plan_text,
                continuation_plan=plan_text,
                page_scan_evidence=page_evidence,
                deletes=result["deleted"],
                more=result["more"],
                max_reads=(READ_PAGE_LIMIT + 1) * MAX_READ_PAGES,
                max_deletes=MAX_DELETES,
                plan=plan_text,
                rss_baseline_kib=baseline,
                rss_sampled_peak_kib=sampled_peak,
                rss_sampled_peak_delta_kib=sampled_peak - baseline,
                rss_kernel_hwm_kib=kernel_hwm,
                rss_sample_interval_ms=10,
                rss_metric="VmHWM reset after setup; /proc VmRSS sampled during prune call",
            )
            await storage.release_consolidation_lease(brain_id, owner_token)

        protected = await storage._query(
            "SELECT count() AS count FROM semantic_discovery_state "
            "WHERE run_id = $run_id GROUP ALL",
            run_id="paused-run",
        )
        measured["protected_rows_remaining"] = int(protected[0]["count"])
        assert measured["protected_rows_remaining"] == PROTECTED_ROWS
        measured.update(
            seeded_rows=len(rows),
            protected_rows_seeded=PROTECTED_ROWS,
            payload_bytes_per_row=PAYLOAD_BYTES,
        )
        return measured
    finally:
        await storage.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("legacy", "bounded"), required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args.mode, args.version)), sort_keys=True))


if __name__ == "__main__":
    main()
