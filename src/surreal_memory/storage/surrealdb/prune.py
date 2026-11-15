"""Lease-fenced, replayable pruning; generic deletion remains independent."""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

from surreal_memory.core.synapse import Synapse
from surreal_memory.engine.prune_policy import PrunePolicy
from surreal_memory.storage.errors import PruneTransactionConflictError
from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_public_id, _to_surreal_id

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

PruneOutcome = Literal["deleted", "protected", "changed", "missing"]
_REPAIR_FIBERS_PER_TRANSACTION = 8
_PRUNE_REPAIR_TRANSACTION = """
-- bounded legacy prune repair
BEGIN TRANSACTION;
LET $lease = (UPDATE type::record('consolidation_lease', $lease_id)
    SET sweep_fenced_at = time::now()
    WHERE brain_id = $brain_id AND owner_token = $owner
      AND expires_at > time::now() RETURN AFTER);
IF array::len($lease) != 1 { THROW 'prune lease lost'; };
LET $present = (SELECT id FROM __PRUNE_RECORDS__);
LET $present_ids = array::map($present, |$row| $row.id);
LET $absent = array::map(
    array::filter($candidates, |$candidate| !(type::record('synapse', $candidate.record_id) IN $present_ids)),
    |$candidate| $candidate.public_id
);
LET $matches = (SELECT id, brain_id FROM fiber WITH INDEX idx_fiber_synapses
    WHERE brain_id = $brain_id AND synapse_ids CONTAINSANY $absent
    ORDER BY brain_id LIMIT $probe_limit);
LET $targets = array::map(array::slice($matches, 0, $fiber_limit), |$row| $row.id);
LET $updated = (UPDATE $targets SET synapse_ids = array::difference(synapse_ids, $absent)
    WHERE brain_id = $brain_id AND synapse_ids CONTAINSANY $absent RETURN id);
LET $final_lease = (UPDATE type::record('consolidation_lease', $lease_id)
    SET sweep_fenced_at = time::now()
    WHERE brain_id = $brain_id AND owner_token = $owner
      AND expires_at > time::now() RETURN AFTER);
IF array::len($final_lease) != 1 { THROW 'prune lease lost'; };
SELECT * FROM [{updated_ids: array::map($updated, |$row| $row.id), has_more: array::len($matches) > $fiber_limit}];
COMMIT TRANSACTION;
"""

# Each transaction reads its own coherent protection snapshot. Independently
# overlapping fiber/pin writes may linearize later; same-edge writes conflict.
_PRUNE_TRANSACTION = """
BEGIN TRANSACTION;
LET $lease = (UPDATE type::record('consolidation_lease', $lease_id)
    SET sweep_fenced_at = time::now()
    WHERE brain_id = $brain_id AND owner_token = $owner
      AND expires_at > time::now() RETURN AFTER);
IF array::len($lease) != 1 { THROW 'prune lease lost'; };
FOR $item IN $items {
    LET $prior = (SELECT * FROM type::record('change_log', $item.receipt_id)
        WHERE brain_id = $brain_id AND entity_type = 'synapse' AND operation = 'delete'
          AND payload._prune.run_id = $run_id AND payload._prune.operation_id = $operation_id);
    IF array::len($prior) = 0 {
        LET $rows = (SELECT * FROM type::record('synapse', $item.record_id)
            WHERE brain_id = $brain_id);
        LET $row = $rows[0];
        LET $same = array::len($rows) = 1 AND type::is_object($item.snapshot)
            AND $row.in = type::record('neuron', $item.source_record)
            AND $row.out = type::record('neuron', $item.target_record)
            AND $row.type = $item.snapshot.type AND $row.direction = $item.snapshot.direction
            AND $row.weight = $item.snapshot.weight
            AND $row.created_at = $item.snapshot.created_at
            AND ($row.last_activated ?? NULL) = ($item.snapshot.last_activated ?? NULL)
            AND $row.reinforced_count = $item.snapshot.reinforced_count
            AND $row.metadata = $item.snapshot.metadata;
        LET $refs = IF $same {
            (SELECT id FROM fiber WITH INDEX idx_fiber_synapses
             WHERE brain_id = $brain_id AND synapse_ids CONTAINS $item.public_id LIMIT 1)
        } ELSE { [] };
        LET $source_protection = IF $same {
            (SELECT id FROM fiber WITH INDEX idx_fiber_neurons
             WHERE brain_id = $brain_id AND neuron_ids CONTAINS $item.source_id
             AND (pinned = true OR salience > 0.8) LIMIT 1)
        } ELSE { [] };
        LET $target_protection = IF $same {
            (SELECT id FROM fiber WITH INDEX idx_fiber_neurons
             WHERE brain_id = $brain_id AND neuron_ids CONTAINS $item.target_id
             AND (pinned = true OR salience > 0.8) LIMIT 1)
        } ELSE { [] };
        LET $targets = IF $same AND $row.weight >= 0.02 {
            (SELECT VALUE array::len(array::group(out)) FROM synapse WITH INDEX idx_synapse_in
             WHERE brain_id = $brain_id AND in = $row.in GROUP ALL)
        } ELSE { [2] };
        LET $protected = !$item.eligible OR array::len($refs) > 0
            OR array::len($source_protection) > 0 OR array::len($target_protection) > 0
            OR ($row.weight >= 0.02 AND ($targets[0] ?? 0) <= 1);
        LET $outcome = IF array::len($rows) = 0 { 'missing' }
            ELSE IF !$same { 'changed' }
            ELSE IF $protected { 'protected' } ELSE { 'deleted' };
        IF $outcome = 'deleted' {
            DELETE type::record('synapse', $item.record_id) WHERE brain_id = $brain_id;
            CREATE type::record('change_log', $item.receipt_id) CONTENT $item.change;
        };
    };
};
LET $final_lease = (UPDATE type::record('consolidation_lease', $lease_id)
    SET sweep_fenced_at = time::now()
    WHERE brain_id = $brain_id AND owner_token = $owner
      AND expires_at > time::now() RETURN AFTER);
IF array::len($final_lease) != 1 { THROW 'prune lease lost'; };
COMMIT TRANSACTION;
"""


def _checked_prune_results(raw: Any) -> list[dict[str, Any]]:
    """Check every statement, retaining safe RPC diagnostics without parameters."""
    from surreal_memory.engine.consolidation_progress import ConsolidationLeaseLostError

    if isinstance(raw, dict) and raw.get("error"):
        error = raw["error"]
        if isinstance(error, dict):
            message = str(error.get("message", "no message"))
            if "prune lease lost" in message:
                raise ConsolidationLeaseLostError("prune lease lost")
            if str(error.get("code")) == "40001" or any(
                term in message.lower() for term in ("transaction conflict", "write conflict")
            ):
                raise PruneTransactionConflictError("prune transaction write conflict")
            raise RuntimeError(
                f"prune transaction RPC error {error.get('code', 'unknown')}: "
                f"{str(error.get('message', 'no message'))[:1000]}"
            )
        raise RuntimeError("prune transaction response is unconfirmed (RPC error)")
    results = raw.get("result") if isinstance(raw, dict) else raw
    if (
        not isinstance(results, list)
        or not results
        or any(
            not isinstance(row, dict) or row.get("status") not in {"OK", "ERR"} for row in results
        )
    ):
        raise RuntimeError("prune transaction response is unconfirmed")
    errors = [
        (index, row.get("result")) for index, row in enumerate(results) if row["status"] != "OK"
    ]
    if errors:
        if any("prune lease lost" in str(error) for _, error in errors):
            raise ConsolidationLeaseLostError("prune lease lost")
        if any(
            term in str(error).lower()
            for _, error in errors
            for term in ("transaction conflict", "write conflict")
        ):
            raise PruneTransactionConflictError("prune transaction write conflict")
        rollback_messages = {
            "the query was not executed due to a failed transaction",
            "the query was not executed due to a cancelled transaction",
        }
        index, cause = next(
            (
                (index, error)
                for index, error in errors
                if str(error).strip().lower().rstrip(".") not in rollback_messages
            ),
            errors[0],
        )
        raise RuntimeError(
            f"prune transaction failed at statement {index + 1}: {str(cause)[:1000]}"
        )
    return results


class SurrealDBPruneMixin:
    _change_seq: int

    if TYPE_CHECKING:

        async def _record_changes_bulk(
            self, entity_type: str, operation: str, entities: list[Any]
        ) -> None: ...

    async def get_synapses_by_ids(self, synapse_ids: list[str] | set[str]) -> list[Synapse]:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    async def _ensure_conn_on_current_loop(self) -> None:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_fiber_synapse_ids_for(self, synapse_ids: list[str] | set[str]) -> set[str]:
        """Use scalar array-index probes; never fetch whole fiber payloads."""
        found: set[str] = set()
        ids = list(dict.fromkeys(synapse_ids))
        for start in range(0, len(ids), 32):
            batch = ids[start : start + 32]
            params: dict[str, Any] = {"brain_id": self._get_brain_id()}
            probes = []
            for i, sid in enumerate(batch):
                params[f"sid_{i}"] = sid
                probes.append(
                    "(SELECT VALUE id FROM fiber WITH INDEX idx_fiber_synapses "
                    f"WHERE brain_id = $brain_id AND synapse_ids CONTAINS $sid_{i} LIMIT 1)"
                )
            rows = await self._query("RETURN [" + ", ".join(probes) + "]", **params)
            if len(rows) != len(batch):
                raise RuntimeError("prune fiber lookup returned incomplete results")
            found.update(sid for sid, matches in zip(batch, rows, strict=True) if matches)
        return found

    async def repair_prune_synapse_refs_if_absent(
        self,
        synapse_ids: list[str],
        *,
        lease_owner_token: str,
        budget_check: Callable[[], Awaitable[None]] | None = None,
    ) -> int:
        """Repair at most eight fibers per transaction, once per missing-ID group."""
        if not lease_owner_token:
            raise ValueError("prune repair requires a lease owner")
        unique_ids = list(dict.fromkeys(synapse_ids))
        count = 0
        for start in range(0, len(unique_ids), 32):
            params: dict[str, Any] = {
                "brain_id": self._get_brain_id(),
                "owner": lease_owner_token,
                "lease_id": sha256(self._get_brain_id().encode()).hexdigest(),
                "fiber_limit": _REPAIR_FIBERS_PER_TRANSACTION,
                "probe_limit": _REPAIR_FIBERS_PER_TRANSACTION + 1,
            }
            candidates = []
            targets = []
            for index, sid in enumerate(unique_ids[start : start + 32]):
                params[f"record_{index}"] = _to_surreal_id(sid)
                candidates.append({"public_id": sid, "record_id": _to_surreal_id(sid)})
                targets.append(f"type::record('synapse', $record_{index})")
            params["candidates"] = candidates
            sql = _PRUNE_REPAIR_TRANSACTION.replace(
                "__PRUNE_RECORDS__", "[" + ", ".join(targets) + "]"
            )
            while True:
                if budget_check is not None:
                    await budget_check()
                await self._ensure_conn_on_current_loop()
                results = _checked_prune_results(await self._ensure_conn().query_raw(sql, params))
                pages = [
                    row
                    for statement in results
                    for row in (
                        statement["result"] if isinstance(statement.get("result"), list) else []
                    )
                    if isinstance(row, dict) and set(row) == {"updated_ids", "has_more"}
                ]
                if len(pages) != 1:
                    raise RuntimeError("prune repair page response is unconfirmed")
                page = pages[0]
                updated_ids = page["updated_ids"]
                if (
                    not isinstance(updated_ids, list)
                    or len(updated_ids) > _REPAIR_FIBERS_PER_TRANSACTION
                    or type(page["has_more"]) is not bool
                    or any(not str(rid).startswith("fiber:") for rid in updated_ids)
                ):
                    raise RuntimeError("prune repair page response is invalid")
                public_ids = [_to_public_id(_record_id_part(str(rid))) for rid in updated_ids]
                if len(set(public_ids)) != len(public_ids) or (page["has_more"] and not public_ids):
                    raise RuntimeError("prune repair page made no bounded progress")
                # Fiber updates already have a null journal payload; only IDs are needed.
                await self._record_changes_bulk(
                    "fiber", "update", [SimpleNamespace(id=fid) for fid in public_ids]
                )
                count += len(public_ids)
                if not page["has_more"]:
                    break
        return count

    async def prune_synapses_if_unchanged(
        self,
        expected: list[Synapse],
        *,
        pending_ids: list[str],
        policy: PrunePolicy,
        run_id: str,
        operation_id: str,
        lease_owner_token: str,
    ) -> dict[str, PruneOutcome]:
        """Delete up to 32 edges per atomic decision/journal transaction.

        Stable operation identity lets callers reconcile response loss by replay.
        No mutation is retried here: uncertain execution propagates to the durable
        pending checkpoint. Receipt reads cannot stand in for a confirmed commit.
        """
        if not run_id or not operation_id or not lease_owner_token:
            raise ValueError("prune requires a durable run, operation identity and lease owner")
        from surreal_memory.storage.surrealdb.store import _change_payload
        from surreal_memory.utils.timeutils import utcnow

        brain_id = self._get_brain_id()
        snapshots = {synapse.id: synapse for synapse in expected}
        ids = list(dict.fromkeys(pending_ids))
        if snapshots.keys() - set(ids):
            raise ValueError("prune snapshot contains IDs outside pending operation")
        outcomes: dict[str, PruneOutcome] = {}
        for start in range(0, len(ids), 32):
            batch = ids[start : start + 32]
            items = []
            for sid in batch:
                edge = snapshots.get(sid)
                receipt_id = sha256(
                    f"{brain_id}:{run_id}:{operation_id}:{sid}".encode()
                ).hexdigest()
                self._change_seq += 1
                snapshot = (
                    None
                    if edge is None
                    else {
                        "type": edge.type.value,
                        "direction": edge.direction,
                        "weight": edge.weight,
                        "created_at": edge.created_at,
                        "last_activated": edge.last_activated,
                        "reinforced_count": edge.reinforced_count,
                        "metadata": dict(edge.metadata),
                    }
                )
                decision = (
                    None
                    if edge is None
                    else {
                        "reference_time": policy.reference_time,
                        "stored_weight": edge.weight,
                        "effective_weight": policy.effective_weight(edge),
                        "weight_threshold": policy.weight_threshold,
                        "min_inactive_days": policy.min_inactive_days,
                        "dream_decay_multiplier": policy.dream_decay_multiplier,
                        "created_at": edge.created_at,
                        "last_activated": edge.last_activated,
                        "reinforced_count": edge.reinforced_count,
                        "inferred": bool(edge.metadata.get("_inferred")),
                        "dream": bool(edge.metadata.get("_dream")),
                        "semantic_discovery": bool(edge.metadata.get("_semantic_discovery")),
                    }
                )
                items.append(
                    {
                        "public_id": sid,
                        "record_id": _to_surreal_id(sid),
                        "receipt_id": receipt_id,
                        "source_id": edge.source_id if edge else "",
                        "target_id": edge.target_id if edge else "",
                        "source_record": _to_surreal_id(edge.source_id) if edge else "",
                        "target_record": _to_surreal_id(edge.target_id) if edge else "",
                        "snapshot": snapshot,
                        "decision": decision,
                        "eligible": edge is not None and policy.eligible(edge),
                        "change": {
                            "brain_id": brain_id,
                            "entity_type": "synapse",
                            "entity_id": sid,
                            "operation": "delete",
                            "device_id": "",
                            "payload": {
                                **(_change_payload(edge) or {}),
                                "_prune": {
                                    "run_id": run_id,
                                    "operation_id": operation_id,
                                    "synapse_id": sid,
                                    "outcome": "deleted",
                                    "decision": decision,
                                },
                            },
                            "changed_at": utcnow(),
                            "synced": False,
                            "sequence": self._change_seq,
                        },
                    }
                )
            await self._ensure_conn_on_current_loop()
            # SDK query() checks only its first statement; inspect every result.
            raw = await self._ensure_conn().query_raw(
                _PRUNE_TRANSACTION,
                {
                    "brain_id": brain_id,
                    "lease_id": sha256(brain_id.encode()).hexdigest(),
                    "owner": lease_owner_token,
                    "run_id": run_id,
                    "operation_id": operation_id,
                    "items": items,
                },
            )
            _checked_prune_results(raw)
            params: dict[str, Any] = {
                "brain_id": brain_id,
                "operation_id": operation_id,
                "run_id": run_id,
            }
            records = []
            for index, item in enumerate(items):
                params[f"receipt_{index}"] = item["receipt_id"]
                records.append(f"type::record('change_log', $receipt_{index})")
            receipts = await self._query(
                "SELECT * FROM "
                + ", ".join(records)
                + " WHERE brain_id = $brain_id AND entity_type = 'synapse' AND operation = 'delete'"
                + " AND payload._prune.operation_id = $operation_id AND payload._prune.run_id = $run_id",
                **params,
            )
            deleted_ids: set[str] = set()
            for receipt in receipts:
                proof = (receipt.get("payload") or {}).get("_prune") or {}
                receipt_sid = receipt.get("entity_id")
                if (
                    receipt_sid not in batch
                    or receipt_sid in deleted_ids
                    or proof.get("synapse_id") != receipt_sid
                    or proof.get("outcome") != "deleted"
                ):
                    raise RuntimeError("prune deletion receipt is invalid")
                deleted_ids.add(receipt_sid)
                outcomes[receipt_sid] = "deleted"
            # Only mutations need durable receipts. Non-mutating outcomes can be
            # rechecked after response loss without inventing a deletion history.
            remainder = set(batch) - deleted_ids
            current = (
                {edge.id: edge for edge in await self.get_synapses_by_ids(remainder)}
                if remainder
                else {}
            )
            for sid in remainder:
                if sid not in current:
                    outcomes[sid] = "missing"
                elif current[sid] != snapshots.get(sid):
                    outcomes[sid] = "changed"
                else:
                    outcomes[sid] = "protected"
        return outcomes
