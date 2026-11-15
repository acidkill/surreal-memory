"""Lease-fenced, replayable pruning; generic deletion remains independent."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from surreal_memory.core.synapse import Synapse
from surreal_memory.engine.prune_policy import PrunePolicy
from surreal_memory.storage.surrealdb._ids import _to_surreal_id

PruneOutcome = Literal["deleted", "protected", "changed", "missing"]

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
        LET $same = array::len($rows) = 1 AND $item.snapshot != NULL
            AND $row.in = type::record('neuron', $item.source_record)
            AND $row.out = type::record('neuron', $item.target_record)
            AND $row.type = $item.snapshot.type AND $row.direction = $item.snapshot.direction
            AND $row.weight = $item.snapshot.weight
            AND $row.created_at = $item.snapshot.created_at
            AND ($row.last_activated ?? NULL) = $item.snapshot.last_activated
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
        LET $protected = NOT $item.eligible OR array::len($refs) > 0
            OR array::len($source_protection) > 0 OR array::len($target_protection) > 0
            OR ($row.weight >= 0.02 AND ($targets[0] ?? 0) <= 1);
        LET $outcome = IF array::len($rows) = 0 { 'missing' }
            ELSE IF NOT $same { 'changed' }
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
    ) -> int:
        """Finish old pending cleanup only for edges absent in the repair snapshot."""
        from surreal_memory.engine.consolidation_progress import ConsolidationLeaseLostError
        from surreal_memory.storage.surrealdb.store import _row_to_fiber

        if not lease_owner_token:
            raise ValueError("prune repair requires a lease owner")
        count = 0
        for start in range(0, len(synapse_ids), 32):
            params: dict[str, Any] = {
                "brain_id": self._get_brain_id(),
                "owner": lease_owner_token,
                "lease_id": sha256(self._get_brain_id().encode()).hexdigest(),
            }
            statements = [
                "BEGIN TRANSACTION;",
                "LET $lease = (UPDATE type::record('consolidation_lease', $lease_id) "
                "SET sweep_fenced_at = time::now() WHERE brain_id = $brain_id "
                "AND owner_token = $owner AND expires_at > time::now() RETURN AFTER);",
                "IF array::len($lease) != 1 { THROW 'prune lease lost'; };",
            ]
            for index, sid in enumerate(synapse_ids[start : start + 32]):
                params[f"sid_{index}"] = sid
                params[f"record_{index}"] = _to_surreal_id(sid)
                statements.append(
                    "UPDATE (SELECT VALUE id FROM fiber WITH INDEX idx_fiber_synapses "
                    f"WHERE brain_id = $brain_id AND synapse_ids CONTAINS $sid_{index}) "
                    f"SET synapse_ids = array::difference(synapse_ids, [$sid_{index}]) "
                    "WHERE brain_id = $brain_id "
                    f"AND synapse_ids CONTAINS $sid_{index} "
                    f"AND array::len((SELECT id FROM type::record('synapse', $record_{index}))) = 0 "
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
            await self._ensure_conn_on_current_loop()
            raw = await self._ensure_conn().query_raw("\n".join(statements), params)
            if not isinstance(raw, dict) or raw.get("error") or not raw.get("result"):
                raise RuntimeError("prune repair transaction is unconfirmed")
            errors = [row.get("result") for row in raw["result"] if row.get("status") != "OK"]
            if errors:
                if any("prune lease lost" in str(error) for error in errors):
                    raise ConsolidationLeaseLostError("prune lease lost")
                raise RuntimeError(f"prune repair transaction failed: {errors}")
            updated = [
                _row_to_fiber(row)
                for statement in raw["result"]
                for row in (
                    statement["result"] if isinstance(statement.get("result"), list) else []
                )
                if isinstance(row, dict) and "synapse_ids" in row
            ]
            # Keep the existing legacy cleanup's fiber-update sync journal contract.
            await self._record_changes_bulk("fiber", "update", updated)
            count += len(updated)
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
        from surreal_memory.engine.consolidation_progress import ConsolidationLeaseLostError
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
            if (
                not isinstance(raw, dict)
                or raw.get("error")
                or not isinstance(raw.get("result"), list)
                or not raw["result"]
            ):
                raise RuntimeError("prune transaction response is unconfirmed")
            errors = [row.get("result") for row in raw["result"] if row.get("status") != "OK"]
            if errors:
                if any("prune lease lost" in str(error) for error in errors):
                    raise ConsolidationLeaseLostError("prune lease lost")
                raise RuntimeError(f"prune transaction failed: {errors}")
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
