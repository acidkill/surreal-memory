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
from unittest.mock import patch

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.utils.timeutils import utcnow
from tests._disposable_surrealdb import _inspect, load_target

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
        self.sample_error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        try:
            while not self._stop.is_set():
                self.samples_kib.append(_rss_kib(self.pid))
                self._stop.wait(0.01)
        except Exception as exc:
            self.sample_error = exc

    def __enter__(self) -> _RssSampler:
        self.samples_kib.append(_rss_kib(self.pid))
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        if self.sample_error is not None:
            print(
                json.dumps(
                    {
                        "benchmark_failure": "RSS sampler process unavailable",
                        "pid": self.pid,
                        "error_type": type(self.sample_error).__name__,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            if exc_type is None:
                raise RuntimeError(
                    "RSS sampling failed; peak cannot be confirmed"
                ) from self.sample_error


async def _wire_counted_query(
    query_raw: Any, sql: str, params: Any, **kwargs: Any
) -> tuple[Any, int, str]:
    """Count actual CBOR response bytes at the SDK HTTP or WebSocket decoder."""
    from surrealdb.connections import async_http, async_ws

    observed: list[tuple[str, int]] = []
    http_decode, ws_decode = async_http.decode, async_ws.decode

    def capture_http(raw: bytes) -> Any:
        observed.append(("http", len(raw)))
        return http_decode(raw)

    def capture_ws(raw: bytes) -> Any:
        observed.append(("websocket", len(raw)))
        return ws_decode(raw)

    with (
        patch.object(async_http, "decode", side_effect=capture_http),
        patch.object(async_ws, "decode", side_effect=capture_ws),
    ):
        result = await query_raw(sql, params, **kwargs)
    if len(observed) != 1:
        raise RuntimeError("could not prove one CBOR response for transaction")
    transport, size = observed[0]
    return result, size, transport


def _repair_scan_evidence(plan: Any) -> dict[str, Any]:
    scans = _physical_scans(plan)
    outputs = [
        scan.get("metrics", {}).get("output_rows")
        if isinstance(scan.get("metrics"), dict)
        else None
        for scan in scans
    ]
    measured = bool(scans) and all(type(value) is int and value >= 0 for value in outputs)
    count = sum(outputs) if measured else None
    return {
        "physical_scans": scans,
        "scan_output_rows": count,
        "indexed_scan_output_bound": measured
        and count <= 9
        and any(scan["operator"] == "IndexScan" for scan in scans),
        "physical_read_bound": None,
        "physical_read_gap": "EXPLAIN output_rows counts emitted rows, not underlying index entries read",
    }


def _repair_page(statements: list[dict[str, Any]]) -> dict[str, Any]:
    assert all("synapse_ids" not in str(statement.get("result")) for statement in statements)
    pages = [
        row
        for statement in statements
        for row in (statement.get("result") if isinstance(statement.get("result"), list) else [])
        if isinstance(row, dict) and set(row) == {"updated_ids", "has_more"}
    ]
    assert len(pages) == 1
    ids = pages[0]["updated_ids"]
    assert isinstance(ids, list)
    assert len(ids) <= 8
    assert all(
        getattr(item, "table_name", None) == "fiber"
        or (isinstance(item, str) and item.startswith("fiber:"))
        for item in ids
    )
    assert type(pages[0]["has_more"]) is bool
    return pages[0]


async def _run_repair(mode: str, version: str, options: dict[str, str], pid: int) -> dict[str, Any]:
    """Compare full-row legacy repair with the bounded real storage method."""
    from surrealdb import RecordID

    from surreal_memory.storage.surrealdb.prune import _checked_prune_results
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(**options, database=f"repair_{uuid.uuid4().hex[:10]}")
    await storage.initialize()
    brain = Brain.create(name="synthetic-repair-benchmark")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    owner = uuid.uuid4().hex
    try:
        nodes = [Neuron.create(NeuronType.CONCEPT, f"repair-node-{i}") for i in range(2)]
        for node in nodes:
            await storage.add_neuron(node)
        live = Synapse.create(nodes[0].id, nodes[1].id, SynapseType.RELATED_TO)
        await storage.add_synapse(live)
        missing = ["missing-repair-a", "missing-repair-b"]
        unrelated = [f"unrelated-{i:05d}" for i in range(30_000)]
        expected = set(unrelated) | {live.id}
        refs = unrelated + missing + [live.id]
        fiber_count = 24
        for start in range(0, fiber_count, 4):
            rows = [
                {
                    "id": RecordID("fiber", f"repair_{i:04d}"),
                    "brain_id": brain.id,
                    "neuron_ids": [nodes[0].id],
                    "anchor_neuron_id": nodes[0].id,
                    "synapse_ids": refs,
                    "metadata": {"padding": "x" * (128 * 1024)},
                }
                for i in range(start, min(start + 4, fiber_count))
            ]
            await storage._query("INSERT INTO fiber $rows RETURN NONE", rows=rows)
        assert await storage.acquire_consolidation_lease(brain.id, owner, lease_seconds=300)
        if mode == "repair-bounded":
            select_sql = (
                "SELECT id FROM fiber WITH INDEX idx_fiber_synapses "
                "WHERE brain_id = $brain_id AND synapse_ids CONTAINSANY $absent LIMIT $probe_limit"
            )
            plan_params: dict[str, Any] = {
                "brain_id": brain.id,
                "absent": missing,
                "probe_limit": 9,
            }
        else:
            select_sql = (
                "SELECT VALUE id FROM fiber WITH INDEX idx_fiber_synapses "
                "WHERE brain_id = $brain_id AND synapse_ids CONTAINS $sid"
            )
            plan_params = {"brain_id": brain.id, "sid": missing[0]}
        plan = await storage._query_response(select_sql + " EXPLAIN FULL", **plan_params)
        process_start = _start_ticks(pid)
        baseline = _rss_kib(pid)
        _reset_hwm(pid)
        print(
            json.dumps({"measurement_started": mode, "db_version": version}, sort_keys=True),
            flush=True,
        )
        transaction_sizes: list[int] = []
        response_bytes: list[int] = []
        response_transports: list[str] = []
        response_shapes: list[dict[str, Any]] = []
        with _RssSampler(pid) as sampler:
            if mode == "repair-legacy":
                # Reproduce the former unbounded per-edge UPDATE and full-row reply.
                from hashlib import sha256

                statements = [
                    "BEGIN TRANSACTION;",
                    "LET $lease = (UPDATE type::record('consolidation_lease', $lease_id) "
                    "SET sweep_fenced_at = time::now() WHERE brain_id = $brain_id "
                    "AND owner_token = $owner AND expires_at > time::now() RETURN AFTER);",
                    "IF array::len($lease) != 1 { THROW 'prune lease lost'; };",
                ]
                params: dict[str, Any] = {
                    "brain_id": brain.id,
                    "owner": owner,
                    "lease_id": sha256(brain.id.encode()).hexdigest(),
                }
                for i, sid in enumerate(missing + [live.id]):
                    params[f"sid_{i}"] = sid
                    params[f"record_{i}"] = sid
                    statements.append(
                        "UPDATE (SELECT VALUE id FROM fiber WITH INDEX idx_fiber_synapses "  # noqa: S608
                        f"WHERE brain_id = $brain_id AND synapse_ids CONTAINS $sid_{i}) "
                        f"SET synapse_ids = array::difference(synapse_ids, [$sid_{i}]) "
                        f"WHERE brain_id = $brain_id AND synapse_ids CONTAINS $sid_{i} "
                        f"AND array::len((SELECT id FROM type::record('synapse', $record_{i}))) = 0 "
                        "RETURN AFTER;"
                    )
                statements.extend(
                    [
                        "LET $final_lease = (UPDATE type::record('consolidation_lease', $lease_id) "
                        "SET sweep_fenced_at = time::now() WHERE brain_id = $brain_id "
                        "AND owner_token = $owner AND expires_at > time::now() RETURN AFTER);",
                        "IF array::len($final_lease) != 1 { THROW 'prune lease lost'; };",
                        "COMMIT TRANSACTION;",
                    ]
                )
                raw, size, transport = await _wire_counted_query(
                    storage._ensure_conn().query_raw, "\n".join(statements), params
                )
                response_bytes.append(size)
                response_transports.append(transport)
                results = _checked_prune_results(raw)
                rows = [
                    row
                    for statement in results
                    for row in (
                        statement.get("result") if isinstance(statement.get("result"), list) else []
                    )
                    if isinstance(row, dict) and "synapse_ids" in row
                ]
                transaction_sizes.append(len(rows))
                response_shapes.append({"full_fiber_rows": len(rows)})
                updated = len(rows)
            else:
                connection = storage._ensure_conn()
                original_query_raw = connection.query_raw

                async def capture(sql: str, params: Any = None, **kwargs: Any) -> Any:
                    if "-- bounded legacy prune repair" not in sql:
                        return await original_query_raw(sql, params, **kwargs)
                    raw, size, transport = await _wire_counted_query(
                        original_query_raw, sql, params, **kwargs
                    )
                    response_bytes.append(size)
                    response_transports.append(transport)
                    if "-- bounded legacy prune repair" in sql:
                        statements = _checked_prune_results(raw)
                        page = _repair_page(statements)
                        ids = page["updated_ids"]
                        transaction_sizes.append(len(ids))
                        response_shapes.append(
                            {"updated_ids": len(ids), "has_more": page["has_more"]}
                        )
                    return raw

                connection.query_raw = capture
                try:
                    updated = await storage.repair_prune_synapse_refs_if_absent(
                        missing + [live.id], lease_owner_token=owner
                    )
                finally:
                    connection.query_raw = original_query_raw
                assert transaction_sizes == [8, 8, 8]
                assert [page["has_more"] for page in response_shapes] == [True, True, False]
                assert updated == fiber_count
        if mode == "repair-legacy":
            assert updated == fiber_count * len(missing)
        assert _start_ticks(pid) == process_start, "SurrealDB process changed during repair"
        sample_peak = max(sampler.samples_kib)
        kernel_hwm = _hwm_kib(pid)
        rows = await storage._query(
            "SELECT synapse_ids, metadata FROM fiber WHERE brain_id = $brain_id", brain_id=brain.id
        )
        assert len(rows) == fiber_count
        assert all(set(row["synapse_ids"]) == expected for row in rows)
        assert all(row["metadata"]["padding"] == "x" * (128 * 1024) for row in rows)
        assert await storage.get_synapses_by_ids([live.id])
        await storage.release_consolidation_lease(brain.id, owner)
        return {
            "mode": mode,
            "db_version": version,
            "fibers": fiber_count,
            "unrelated_refs_per_fiber": len(unrelated),
            "missing_edges": len(missing),
            "updates": updated,
            "transaction_count": len(transaction_sizes),
            "transaction_mutations": transaction_sizes,
            "transaction_rows_returned": sum(transaction_sizes),
            "response_shapes": response_shapes,
            "response_wire_bytes": response_bytes,
            "response_wire_format": "CBOR message bytes before SDK decode",
            "response_wire_transports": response_transports,
            "lookup_plan": json.dumps(plan, default=str),
            "lookup_plan_scope": "EXPLAIN FULL for initial selector before repair; physical scope not assumed",
            "logical_probe_limit": 9 if mode == "repair-bounded" else None,
            "lookup_scan_evidence": _repair_scan_evidence(plan),
            "rss_baseline_kib": baseline,
            "rss_sampled_peak_kib": sample_peak,
            "rss_sampled_peak_delta_kib": sample_peak - baseline,
            "rss_kernel_hwm_kib": kernel_hwm,
            "rss_sample_interval_ms": 10,
            "rss_metric": "VmHWM reset after setup; /proc VmRSS sampled during repair",
        }
    finally:
        await storage.close()


def _benchmark_target() -> tuple[dict[str, str], int]:
    target = load_target()
    if target is None:
        raise RuntimeError("registered disposable container is required")
    descriptor = json.loads(Path(os.environ["SMEM_TEST_SURREALDB_TARGET_FILE"]).read_text())
    pid = int(os.environ["SURREALDB_CONTAINER_PID"])
    if _inspect(descriptor["container_id"])["State"]["Pid"] != pid:
        raise RuntimeError("measurement process does not match registered container")
    return target.child().storage_kwargs(), pid


async def _run(mode: str, version: str) -> dict[str, Any]:
    options, pid = _benchmark_target()
    if mode.startswith("repair-"):
        return await _run_repair(mode, version, options, pid)
    from surrealdb import RecordID

    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    suffix = uuid.uuid4().hex[:10]
    storage = SurrealDBStorage(**options, database=f"run_{suffix}")
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
    parser.add_argument(
        "--mode", choices=("legacy", "bounded", "repair-legacy", "repair-bounded"), required=True
    )
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    result = asyncio.run(_run(args.mode, args.version))
    print(json.dumps(result, sort_keys=True))
    if (
        args.mode == "repair-bounded"
        and not result["lookup_scan_evidence"]["indexed_scan_output_bound"]
    ):
        raise SystemExit("bounded selector scan output is unreported or exceeds nine rows")


if __name__ == "__main__":
    main()
