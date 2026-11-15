"""Offline adapter plumbing; actual transaction semantics live in DB integration tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from surreal_memory.engine.consolidation_progress import ConsolidationLeaseLostError
from surreal_memory.engine.prune_policy import PrunePolicy
from surreal_memory.storage.surrealdb.prune import SurrealDBPruneMixin
from tests.unit.test_consolidation_prune_resumable import REFERENCE_TIME, _weak_synapse


class ScriptedPrune(SurrealDBPruneMixin):
    def __init__(self):
        self._change_seq = 0
        self.calls = []
        self.receipts = {}
        self.raw_response = {"result": [{"status": "OK", "result": None}]}
        self.conn = self

    def _get_brain_id(self):
        return "synthetic-brain"

    def _ensure_conn(self):
        return self.conn

    async def _ensure_conn_on_current_loop(self):
        pass

    async def query_raw(self, sql, params):
        self.calls.append(params)
        for item in params["items"]:
            self.receipts.setdefault(item["receipt_id"], item["change"])
        return self.raw_response

    async def _query(self, sql, **params):
        return [
            self.receipts[value]
            for key, value in params.items()
            if key.startswith("receipt_") and value in self.receipts
        ]

    async def get_synapses_by_ids(self, synapse_ids):
        return []


async def call(storage, edges):
    return await storage.prune_synapses_if_unchanged(
        edges,
        pending_ids=[edge.id for edge in edges],
        policy=PrunePolicy(REFERENCE_TIME, 0.05, 7, 10),
        run_id="run",
        operation_id="operation",
        lease_owner_token="synthetic-owner",  # noqa: S106 - synthetic lease identity
    )


async def test_atomic_chunks_and_deterministic_existing_journal_receipts():
    storage = ScriptedPrune()
    edges = [replace(_weak_synapse(), id=f"edge-{index}") for index in range(70)]
    result = await call(storage, edges)
    assert set(result.values()) == {"deleted"}
    assert [len(call["items"]) for call in storage.calls] == [32, 32, 6]
    before = dict(storage.receipts)
    assert await call(storage, edges) == result
    assert storage.receipts == before
    payload = next(iter(before.values()))["payload"]
    assert payload["_prune"]["decision"]["reference_time"] == REFERENCE_TIME
    assert payload["_prune"]["decision"]["stored_weight"] == 0.001
    assert payload["source_id"] == "source-1"


@pytest.mark.parametrize("raw", [{}, {"result": []}, {"error": "not-confirmed"}])
async def test_unconfirmed_raw_response_never_becomes_success(raw):
    storage = ScriptedPrune()
    storage.raw_response = raw
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await call(storage, [_weak_synapse()])
    assert len(storage.calls) == 1


async def test_checks_later_statement_errors_instead_of_only_sdk_first_result():
    storage = ScriptedPrune()
    storage.raw_response["result"].append({"status": "ERR", "result": "prune lease lost"})
    with pytest.raises(ConsolidationLeaseLostError):
        await call(storage, [_weak_synapse()])


async def test_fiber_probe_batch_retains_each_nested_result():
    class ProbeStorage(ScriptedPrune):
        async def _query(self, sql, **params: Any):
            assert "idx_fiber_synapses" in sql and "LIMIT 1" in sql
            return [[{"id": "fiber"}]] if len(params) == 2 else [[], [{"id": "fiber"}], []]

    storage = ProbeStorage()
    assert await storage.get_fiber_synapse_ids_for(["absent-a", "referenced", "absent-b"]) == {
        "referenced"
    }
    assert await storage.get_fiber_synapse_ids_for(["referenced"]) == {"referenced"}
    assert await storage.get_fiber_synapse_ids_for([]) == set()


@pytest.mark.parametrize("envelope", [True, False])
async def test_transaction_accepts_rpc_envelope_and_statement_list(envelope):
    storage = ScriptedPrune()
    if not envelope:
        storage.raw_response = storage.raw_response["result"]
    assert await call(storage, [_weak_synapse()]) == {"edge-1": "deleted"}


async def test_rpc_parse_error_keeps_code_and_message_without_request_parameters():
    from surreal_memory.storage.surrealdb.prune import _checked_prune_results

    with pytest.raises(RuntimeError, match="RPC error -32700: Parse error") as error:
        _checked_prune_results(
            {
                "error": {"code": -32700, "message": "Parse error"},
                "params": {"synthetic_private_value": "do not expose"},
            }
        )
    assert "synthetic_private_value" not in str(error.value)
    assert "do not expose" not in str(error.value)


@pytest.mark.parametrize(
    "raw", [None, [], {"result": [None]}, {"result": [{"status": "UNKNOWN"}]}, {"result": True}]
)
async def test_malformed_transaction_shapes_fail_closed(raw):
    from surreal_memory.storage.surrealdb.prune import _checked_prune_results

    with pytest.raises(RuntimeError, match="unconfirmed"):
        _checked_prune_results(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"error": {"code": "40001", "message": "serialization failure"}},
        {
            "result": [
                {"status": "OK", "result": None},
                {
                    "status": "ERR",
                    "result": "Transaction conflict: Write conflict, retry the transaction",
                },
            ]
        },
    ],
)
async def test_only_confirmed_server_conflicts_have_retryable_type(raw):
    from surreal_memory.storage.errors import PruneTransactionConflictError
    from surreal_memory.storage.surrealdb.prune import _checked_prune_results

    with pytest.raises(PruneTransactionConflictError):
        _checked_prune_results(raw)


@pytest.mark.parametrize("envelope", [True, False])
async def test_originating_transaction_error_is_not_masked_by_rollback_placeholders(envelope):
    from surreal_memory.storage.surrealdb.prune import _checked_prune_results

    rows = [
        {"status": "ERR", "result": "The query was not executed due to a failed transaction"},
        {"status": "ERR", "result": "synthetic originating statement error"},
        {"status": "ERR", "result": "The query was not executed due to a cancelled transaction"},
    ]
    raw = {"result": rows} if envelope else rows
    with pytest.raises(RuntimeError, match="statement 2: synthetic originating statement error"):
        _checked_prune_results(raw)
