"""Disposable real-SurrealDB memory benchmark for snapshot cleanup."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import threading
import uuid
from collections.abc import Callable
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
                        "rss_sampled_peak_kib": max(self.samples_kib, default=None),
                        "rss_sample_count": len(self.samples_kib),
                        "rss_sampling_complete": False,
                    }
                ),
                flush=True,
            )
            if exc_type is None:
                raise RuntimeError(
                    "RSS sampling failed; peak cannot be confirmed"
                ) from self.sample_error


def _cgroup_directory(pid: int) -> Path | None:
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    return None


def _repair_diagnostic(
    state: dict[str, Any],
    pid: int,
    process_start: str,
    *,
    sampler: _RssSampler | None = None,
    cgroup: Path | None = None,
    stage: str | None = None,
    **fields: Any,
) -> None:
    """Flush observed values even when the database vanished; missing peaks stay null."""
    if stage is not None:
        state["stage"] = stage
    state.update(fields)
    memory: dict[str, Any] = {
        "rss_current_kib": None,
        "rss_kernel_hwm_kib": None,
        "rss_observation_error": None,
        "process_identity_match": None,
    }
    try:
        memory["process_identity_match"] = _start_ticks(pid) == process_start
        if not memory["process_identity_match"]:
            raise RuntimeError("measurement process identity changed")
        memory["rss_current_kib"] = _rss_kib(pid)
        memory["rss_kernel_hwm_kib"] = _hwm_kib(pid)
    except (OSError, RuntimeError) as exc:
        memory["rss_observation_error"] = type(exc).__name__
    for filename, key in (
        ("memory.current", "cgroup_memory_current_bytes"),
        ("memory.peak", "cgroup_memory_peak_bytes"),
    ):
        memory[key] = None
        if cgroup is not None:
            try:
                memory[key] = int((cgroup / filename).read_text().strip())
            except (OSError, ValueError):
                pass
    memory["cgroup_memory_events"] = None
    if cgroup is not None:
        try:
            memory["cgroup_memory_events"] = {
                key: int(value)
                for key, value in (
                    line.split() for line in (cgroup / "memory.events").read_text().splitlines()
                )
            }
        except (OSError, ValueError):
            pass
    peak = max(sampler.samples_kib, default=None) if sampler is not None else None
    baseline = state.get("rss_baseline_kib")
    print(
        json.dumps(
            {
                **state,
                **memory,
                "rss_sampled_peak_kib": peak,
                "rss_sampled_peak_delta_kib": peak - baseline
                if peak is not None and baseline is not None
                else None,
                "rss_sample_count": len(sampler.samples_kib) if sampler is not None else 0,
                "rss_sampling_complete": sampler is not None
                and sampler.sample_error is None
                and state.get("measurement_complete") is True,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


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


def _legacy_repair_query(
    brain_id: str, owner: str, synapse_ids: list[str]
) -> tuple[str, dict[str, Any]]:
    """Reproduce the original per-edge repair, including storage ID normalization."""
    from surreal_memory.storage.surrealdb._ids import _to_surreal_id

    statements = [
        "BEGIN TRANSACTION;",
        "LET $lease = (UPDATE type::record('consolidation_lease', $lease_id) "
        "SET sweep_fenced_at = time::now() WHERE brain_id = $brain_id "
        "AND owner_token = $owner AND expires_at > time::now() RETURN AFTER);",
        "IF array::len($lease) != 1 { THROW 'prune lease lost'; };",
    ]
    params: dict[str, Any] = {
        "brain_id": brain_id,
        "owner": owner,
        "lease_id": hashlib.sha256(brain_id.encode()).hexdigest(),
    }
    for i, sid in enumerate(synapse_ids):
        params[f"sid_{i}"] = sid
        params[f"record_{i}"] = _to_surreal_id(sid)
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
    return "\n".join(statements), params


async def _verify_repair_fibers(
    storage: Any,
    fiber_ids: list[Any],
    brain_id: str,
    expected_refs: set[str],
    padding: str,
    mark: Callable[..., None],
) -> None:
    """Verify every synthetic row without materializing all fat fibers together."""
    for index, record in enumerate(fiber_ids):
        mark("verification_fiber_started", verification_index=index)
        rows = await storage._query(
            "SELECT id, synapse_ids, metadata FROM $record WHERE brain_id = $brain_id",
            record=record,
            brain_id=brain_id,
        )
        assert len(rows) == 1
        assert rows[0]["id"] == record
        assert set(rows[0]["synapse_ids"]) == expected_refs
        assert rows[0]["metadata"] == {"padding": padding}
        mark("verification_fiber_completed", verification_index=index)


async def _run_repair(
    mode: str, version: str, options: dict[str, str], pid: int, *, diagnostic_only: bool = False
) -> dict[str, Any]:
    """Compare full-row legacy repair with the bounded real storage method."""
    from surrealdb import RecordID

    from surreal_memory.storage.surrealdb.prune import (
        _PRUNE_REPAIR_SELECTOR,
        _checked_prune_results,
    )
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(**options, database=f"repair_{uuid.uuid4().hex[:10]}")
    process_start = _start_ticks(pid)
    descriptor = json.loads(Path(os.environ["SMEM_TEST_SURREALDB_TARGET_FILE"]).read_text())
    instance = _inspect(descriptor["container_id"])
    assert instance["State"]["Pid"] == pid
    memory_limit = int(instance["HostConfig"]["Memory"])
    assert memory_limit == (2 if diagnostic_only else 1) * 1024**3
    swap_limit = int(instance["HostConfig"]["MemorySwap"])
    cpu_limit = int(instance["HostConfig"]["NanoCpus"])
    assert swap_limit == memory_limit, "benchmark must disable extra swap"
    assert cpu_limit == 2_000_000_000, "benchmark must retain the two-CPU limit"
    cgroup = _cgroup_directory(pid)
    state: dict[str, Any] = {
        "mode": mode,
        "db_version": version,
        "diagnostic_only": diagnostic_only,
        "datastore": "memory",
        "memory_limit_bytes": memory_limit,
        "memory_swap_limit_bytes": swap_limit,
        "cpu_limit_nanos": cpu_limit,
        "cgroup_peak_scope": "entire disposable container since startup; not reset",
        "rss_baseline_kib": None,
        "transaction_index": 0,
        "completed_transaction_mutations": [],
        "completed_response_wire_bytes": [],
    }
    sampler = None

    def mark(stage: str | None, **fields: Any) -> None:
        _repair_diagnostic(
            state, pid, process_start, sampler=sampler, cgroup=cgroup, stage=stage, **fields
        )

    owner = uuid.uuid4().hex
    try:
        mark("storage_initialization")
        await storage.initialize()
        brain = Brain.create(name="synthetic-repair-benchmark")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
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
        fiber_ids = []
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
            fiber_ids.extend(row["id"] for row in rows)
            mark("fixture_seed_started", seed_batch_start=start, seed_batch_rows=len(rows))
            await storage._query("INSERT INTO fiber $rows RETURN NONE", rows=rows)
            mark(
                "fixture_seed_completed",
                seeded_fibers=start + len(rows),
                indexed_memberships=(start + len(rows)) * len(refs),
            )
        assert await storage.acquire_consolidation_lease(brain.id, owner, lease_seconds=300)
        if mode == "repair-bounded":
            select_sql = _PRUNE_REPAIR_SELECTOR
            plan_params: dict[str, Any] = {
                "brain_id": brain.id,
                "probe_sid": missing[0],
            }
        else:
            select_sql = (
                "SELECT VALUE id FROM fiber WITH INDEX idx_fiber_synapses "
                "WHERE brain_id = $brain_id AND synapse_ids CONTAINS $sid"
            )
            plan_params = {"brain_id": brain.id, "sid": missing[0]}
        mark("selector_explain_started", selector_sql=select_sql)
        plan = await storage._query_response(select_sql + " EXPLAIN FULL", **plan_params)
        mark(
            "selector_explain_completed",
            lookup_scan_evidence=_repair_scan_evidence(plan),
            lookup_plan=plan,
        )
        baseline = _rss_kib(pid)
        _reset_hwm(pid)
        mark(
            "measurement_started",
            measurement_started=mode,
            rss_baseline_kib=baseline,
            rss_headroom_to_cap_kib=memory_limit // 1024 - baseline,
        )
        transaction_sizes: list[int] = []
        response_bytes: list[int] = []
        response_transports: list[str] = []
        response_shapes: list[dict[str, Any]] = []
        sampler = _RssSampler(pid)
        with sampler:
            if mode == "repair-legacy":
                sql, params = _legacy_repair_query(brain.id, owner, missing + [live.id])
                mark(
                    "transaction_started",
                    transaction_index=1,
                    transaction_sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
                    inflight_response_wire_bytes=None,
                )
                raw, size, transport = await _wire_counted_query(
                    storage._ensure_conn().query_raw, sql, params
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
                mark(
                    "transaction_completed",
                    completed_transaction_mutations=transaction_sizes,
                    completed_response_wire_bytes=response_bytes,
                    inflight_response_wire_bytes=size,
                )
                updated = len(rows)
            else:
                connection = storage._ensure_conn()
                original_query_raw = connection.query_raw

                async def capture(sql: str, params: Any = None, **kwargs: Any) -> Any:
                    if "-- bounded legacy prune repair" not in sql:
                        return await original_query_raw(sql, params, **kwargs)
                    mark(
                        "transaction_started",
                        transaction_index=len(transaction_sizes) + 1,
                        transaction_sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
                        inflight_response_wire_bytes=None,
                    )
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
                    mark(
                        "transaction_completed",
                        completed_transaction_mutations=transaction_sizes,
                        completed_response_wire_bytes=response_bytes,
                        inflight_response_wire_bytes=size,
                    )
                    return raw

                connection.query_raw = capture
                try:
                    updated = await storage.repair_prune_synapse_refs_if_absent(
                        missing + [live.id], lease_owner_token=owner
                    )
                finally:
                    connection.query_raw = original_query_raw
                assert transaction_sizes == [8, 8, 8, 0, 0]
                assert [page["has_more"] for page in response_shapes] == [
                    True,
                    True,
                    False,
                    False,
                    False,
                ]
                assert updated == fiber_count
        if mode == "repair-legacy":
            assert updated == fiber_count * len(missing)
        assert _start_ticks(pid) == process_start, "SurrealDB process changed during repair"
        sample_peak = max(sampler.samples_kib)
        kernel_hwm = _hwm_kib(pid)
        mark(
            "measurement_completed",
            measurement_complete=True,
            repair_rss_sampled_peak_kib=sample_peak,
            repair_rss_kernel_hwm_kib=kernel_hwm,
        )
        assert len(fiber_ids) == fiber_count
        await _verify_repair_fibers(
            storage, fiber_ids, brain.id, expected, "x" * (128 * 1024), mark
        )
        mark("verification_synapse_started")
        assert await storage.get_synapses_by_ids([live.id])
        mark("verification_completed", verified_fibers=len(fiber_ids))
        mark("lease_release_started")
        await storage.release_consolidation_lease(brain.id, owner)
        mark("lease_release_completed")
        return {
            "mode": mode,
            "db_version": version,
            "diagnostic_only": diagnostic_only,
            "measurement_complete": True,
            "memory_limit_bytes": memory_limit,
            "datastore": "memory",
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
    except BaseException as exc:
        mark(
            None,
            benchmark_failed=True,
            measurement_failed=state.get("measurement_complete") is not True,
            failure_error_type=type(exc).__name__,
        )
        raise
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


async def _run(mode: str, version: str, *, diagnostic_only: bool = False) -> dict[str, Any]:
    options, pid = _benchmark_target()
    if mode.startswith("repair-"):
        return await _run_repair(mode, version, options, pid, diagnostic_only=diagnostic_only)
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
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()

    async def execute() -> dict[str, Any]:
        work = _run(args.mode, args.version, diagnostic_only=args.diagnostic_only)
        return await asyncio.wait_for(work, timeout=120) if args.diagnostic_only else await work

    result = asyncio.run(execute())
    print(json.dumps(result, sort_keys=True))
    if (
        args.mode == "repair-bounded"
        and not result["lookup_scan_evidence"]["indexed_scan_output_bound"]
    ):
        raise SystemExit("bounded selector scan output is unreported or exceeds nine rows")


if __name__ == "__main__":
    main()
