"""SurrealDB composite storage backend for Neural Memory.

Combines document, graph, and vector search capabilities in a single
SurrealDB instance. Implements the full NeuralStorage interface including
sync engine, change log, Merkle hashes, and typed memories.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from neural_memory.core.brain import Brain, BrainSnapshot
from neural_memory.core.fiber import Fiber
from neural_memory.core.neuron import Neuron, NeuronState, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.surrealdb.schema import ensure_schema
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _to_surreal_id(record_id: str) -> str:
    """Convert a UUID to a valid SurrealDB record name (alphanumeric + _-)."""
    return record_id.replace("-", "_")


def _from_surreal_id(surreal_id: str) -> str:
    """Extract the original ID from a SurrealDB record ID like 'neuron:abc_def'."""
    if ":" in surreal_id:
        surreal_id = surreal_id.rsplit(":", 1)[1]
    return surreal_id.replace("_", "-")


def _parse_datetime(val: Any) -> datetime | None:
    """Parse a SurrealDB datetime value to Python datetime (naive for consistency)."""
    if val is None:
        return None
    if isinstance(val, datetime):
        # Convert to naive datetime for consistency across codebase
        if val.tzinfo is not None:
            return val.replace(tzinfo=None)
        return val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            # Convert to naive datetime
            if parsed.tzinfo is not None:
                return parsed.replace(tzinfo=None)
            return parsed
        except (ValueError, AttributeError):
            return None
    return None


def _ensure_naive(dt: datetime) -> datetime:
    """Convert datetime to naive (no timezone) for comparison."""
    if dt is None:
        return datetime.min
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _row_to_neuron(row: dict[str, Any]) -> Neuron:
    """Convert a SurrealDB neuron record to a Neuron."""
    meta = dict(row.get("metadata") or {})
    rid = row["id"]
    neuron_id = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    # Strip table prefix and convert underscores back to dashes
    if ":" in neuron_id:
        neuron_id = neuron_id.split(":", 1)[1]
    neuron_id = neuron_id.replace("_", "-")
    return Neuron(
        id=neuron_id,
        type=NeuronType(row["type"]),
        content=str(row["content"]),
        metadata=meta,
        content_hash=int(row.get("content_hash", 0)),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        ephemeral=bool(row.get("ephemeral", False)),
    )


def _row_to_neuron_state(row: dict[str, Any]) -> NeuronState:
    """Convert a SurrealDB neuron_state record to NeuronState."""
    nid = row.get("neuron_id", "")
    if hasattr(nid, "record_id"):
        nid = str(nid.record_id)
    else:
        nid = str(nid)
    return NeuronState(
        neuron_id=nid,
        activation_level=float(row.get("activation_level", 0.0)),
        access_frequency=int(row.get("access_frequency", 0)),
        last_activated=_parse_datetime(row.get("last_activated")),
        decay_rate=float(row.get("decay_rate", 0.1)),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        firing_threshold=float(row.get("firing_threshold", 0.3)),
        refractory_until=_parse_datetime(row.get("refractory_until")),
        refractory_period_ms=float(row.get("refractory_period_ms", 500.0)),
        homeostatic_target=float(row.get("homeostatic_target", 0.5)),
    )


def _row_to_synapse(row: dict[str, Any]) -> Synapse:
    """Convert a SurrealDB synapse record to Synapse."""

    rid = row["id"]
    syn_id = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    if ":" in syn_id:
        syn_id = syn_id.split(":", 1)[1]
    syn_id = syn_id.replace("_", "-")
    source_id = str(row.get("source_id", "")).replace("_", "-")
    target_id = str(row.get("target_id", "")).replace("_", "-")
    syn = Synapse(
        id=syn_id,
        type=SynapseType(row["type"]),
        source_id=source_id,
        target_id=target_id,
        weight=float(row.get("weight", 1.0)),
        direction=Direction(str(row.get("direction", "forward"))),
        metadata=dict(row.get("metadata") or {}),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        last_activated=_parse_datetime(row.get("last_activated")),
        reinforced_count=int(row.get("reinforced_count", 0)),
    )
    return syn


def _row_to_fiber(row: dict[str, Any]) -> Fiber:
    """Convert a SurrealDB fiber record to Fiber."""
    rid = row["id"]
    fiber_id = f"{rid.table_name}:{rid.id}" if hasattr(rid, "table_name") else str(rid)
    if ":" in fiber_id:
        fiber_id = fiber_id.split(":", 1)[1]
    return Fiber(
        id=fiber_id,
        neuron_ids=set(row.get("neuron_ids") or []),
        synapse_ids=set(row.get("synapse_ids") or []),
        anchor_neuron_id=str(row.get("anchor_neuron_id", "")),
        pathway=list(row.get("pathway") or []),
        conductivity=float(row.get("conductivity", 1.0)),
        last_conducted=_parse_datetime(row.get("last_conducted")),
        time_start=_parse_datetime(row.get("time_start")),
        time_end=_parse_datetime(row.get("time_end")),
        coherence=float(row.get("coherence", 0.0)),
        salience=float(row.get("salience", 0.0)),
        frequency=int(row.get("frequency", 0)),
        summary=row.get("summary"),
        essence=row.get("essence"),
        auto_tags=set(row.get("auto_tags") or []),
        agent_tags=set(row.get("agent_tags") or []),
        metadata=dict(row.get("metadata") or {}),
        compression_tier=int(row.get("compression_tier", 0)),
        pinned=bool(row.get("pinned", False)),
        created_at=_parse_datetime(row.get("created_at")) or utcnow(),
        last_ghost_shown_at=_parse_datetime(row.get("last_ghost_shown_at")),
    )


class SurrealDBStorage(NeuralStorage):
    """SurrealDB-backed storage for Neural Memory.

    Multi-model: documents (neurons), graphs (synapses via RELATE),
    and vector search (HNSW via embedding_vec) all in one database.

    Usage:
        storage = SurrealDBStorage(url="http://localhost:8001", ...)
        await storage.initialize()
        storage.set_brain("my-brain")
        await storage.add_neuron(neuron)
    """

    def __init__(
        self,
        url: str = "",
        namespace: str = "neural_memory",
        database: str = "default",
        user: str = "root",
        password: str = "root",
        embedding_dim: int = 3072,
    ) -> None:
        self._url = url or os.getenv("SURREALDB_URL", "http://localhost:8001")
        self._namespace = namespace or os.getenv("SURREALDB_NS", "neural_memory")
        self._database = database or os.getenv("SURREALDB_DB", "default")
        self._user = user or os.getenv("SURREALDB_USER", "root")
        self._password = password or os.getenv("SURREALDB_PASS", "root")
        self._embedding_dim = embedding_dim
        self._conn: Any = None
        self._current_brain_id: str | None = None
        self._change_seq: int = 0

    async def initialize(self) -> None:
        """Connect to SurrealDB and apply schema."""
        from surrealdb import AsyncSurreal

        self._conn = AsyncSurreal(self._url)
        await self._conn.signin({"username": self._user, "password": self._password})
        await self._conn.use(self._namespace, self._database)
        await ensure_schema(self._conn)
        logger.info(
            "SurrealDB connected: %s ns=%s db=%s",
            self._url,
            self._namespace,
            self._database,
        )

    async def close(self) -> None:
        """Close SurrealDB connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def brain_id(self) -> str | None:
        return self._current_brain_id

    def set_brain(self, brain_id: str) -> None:
        self._current_brain_id = brain_id

    def _get_brain_id(self) -> str:
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    def _ensure_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("SurrealDB not initialized. Call initialize() first.")
        return self._conn

    def disable_auto_save(self) -> None:
        pass

    def enable_auto_save(self) -> None:
        pass

    async def batch_save(self) -> None:
        pass

    # ================================================================
    # Query helper
    # ================================================================

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a SurrealQL query and return result rows."""
        conn = self._ensure_conn()
        result = await conn.query(sql, params)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0] if isinstance(result[0], list) else result
        return []

    # ================================================================
    # Neuron Operations
    # ================================================================

    async def add_neuron(self, neuron: Neuron) -> str:
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()

        sid = _to_surreal_id(neuron.id)
        meta = dict(neuron.metadata)
        embedding_vec = meta.pop("_embedding", None)

        record_data: dict[str, Any] = {
            "id": sid,
            "brain_id": brain_id,
            "type": neuron.type.value,
            "content": neuron.content,
            "content_hash": neuron.content_hash,
            "metadata": meta,
            "ephemeral": neuron.ephemeral,
            "created_at": neuron.created_at,
            "updated_at": utcnow(),
        }
        if embedding_vec is not None:
            record_data["embedding_vec"] = list(embedding_vec)

        await conn.insert("neuron", record_data)

        # Create initial state
        state_data = {
            "id": f"state_{sid}",
            "neuron_id": neuron.id,
            "brain_id": brain_id,
            "activation_level": 0.0,
            "access_frequency": 0,
            "created_at": neuron.created_at,
        }
        try:
            await conn.insert("neuron_state", state_data)
        except Exception:
            pass

        # Record change
        await self._record_change_internal("neuron", neuron.id, "insert", neuron)

        return neuron.id

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(neuron_id)
        try:
            result = await conn.select(f"neuron:{sid}")
            if result and isinstance(result, list) and len(result) > 0:
                return _row_to_neuron(result[0])
        except Exception:
            pass
        return None

    async def get_neurons_batch(self, neuron_ids: list[str]) -> dict[str, Neuron]:
        if not neuron_ids:
            return {}
        brain_id = self._get_brain_id()
        # Use direct select for each ID (more reliable than IN query with params)
        results: dict[str, Neuron] = {}
        for nid in neuron_ids:
            sid = _to_surreal_id(nid)
            try:
                result = await self._conn.select(f"neuron:{sid}")
                if result and isinstance(result, list) and len(result) > 0:
                    neuron = _row_to_neuron(result[0])
                    # Use the converted ID as key
                    results[nid] = neuron
            except Exception:
                pass
        return results

    async def find_neurons(
        self,
        type: NeuronType | None = None,
        content_contains: str | None = None,
        content_exact: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 100,
        offset: int = 0,
        ephemeral: bool | None = None,
    ) -> list[Neuron]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if type is not None:
            conditions.append("type = $ntype")
            params["ntype"] = type.value

        if content_exact is not None:
            conditions.append("content = $content_exact")
            params["content_exact"] = content_exact

        if content_contains is not None:
            conditions.append("content CONTAINS $content_contains")
            params["content_contains"] = content_contains

        if time_range is not None:
            start, end = time_range
            conditions.append("created_at >= $time_start AND created_at <= $time_end")
            params["time_start"] = start.isoformat()
            params["time_end"] = end.isoformat()

        if ephemeral is not None:
            conditions.append("ephemeral = $ephemeral")
            params["ephemeral"] = ephemeral

        where = " AND ".join(conditions)
        rows = await self._query(
            f"SELECT * FROM neuron WHERE {where} ORDER BY id LIMIT {int(limit)} START {int(offset)}",
            **params,
        )
        return [_row_to_neuron(r) for r in rows]

    async def suggest_neurons(
        self,
        prefix: str,
        type_filter: NeuronType | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id", "content CONTAINS $prefix"]
        params: dict[str, Any] = {"brain_id": brain_id, "prefix": prefix}

        if type_filter is not None:
            conditions.append("type = $ntype")
            params["ntype"] = type_filter.value

        where = " AND ".join(conditions)
        rows = await self._query(
            f"SELECT id, content, type FROM neuron WHERE {where} LIMIT {int(min(limit, 20))}",
            **params,
        )

        # Get activation info for ranking
        results = []
        for r in rows:
            nid = str(r["id"]).split(":")[-1]
            state = await self.get_neuron_state(nid)
            results.append(
                {
                    "neuron_id": nid,
                    "content": r["content"],
                    "type": r["type"],
                    "access_frequency": state.access_frequency if state else 0,
                    "activation_level": state.activation_level if state else 0.0,
                    "score": (state.activation_level if state else 0.0)
                    + 0.1 * (state.access_frequency if state else 0),
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def update_neuron(self, neuron: Neuron) -> None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(neuron.id)
        meta = dict(neuron.metadata)
        embedding_vec = meta.pop("_embedding", None)

        update_data: dict[str, Any] = {
            "type": neuron.type.value,
            "content": neuron.content,
            "content_hash": neuron.content_hash,
            "metadata": meta,
            "ephemeral": neuron.ephemeral,
            "updated_at": utcnow(),
        }
        if embedding_vec is not None:
            update_data["embedding_vec"] = list(embedding_vec)

        await conn.merge(f"neuron:{sid}", update_data)
        await self._record_change_internal("neuron", neuron.id, "update", neuron)

    async def delete_neuron(self, neuron_id: str) -> bool:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(neuron_id)

        # Delete connected synapses first
        await self._query(
            "DELETE synapse WHERE brain_id = $brain_id AND (source_id = $nid OR target_id = $nid)",
            brain_id=brain_id,
            nid=sid,
        )
        # Delete related edges
        await self._query(
            "DELETE connects_to WHERE out = $src OR in = $tgt",
            src=f"neuron:{sid}",
            tgt=f"neuron:{sid}",
        )
        # Delete state
        await self._query(f"DELETE neuron_state:{sid}")

        try:
            await conn.delete(f"neuron:{sid}")
            await self._record_change_internal("neuron", neuron_id, "delete")
            return True
        except Exception:
            return False

    async def has_neuron_by_content_hash(self, content_hash: int) -> bool:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM neuron WHERE brain_id = $brain_id AND content_hash = $ch LIMIT 1",
            brain_id=brain_id,
            ch=content_hash,
        )
        return len(rows) > 0

    # ================================================================
    # Neuron State Operations
    # ================================================================

    async def get_neuron_state(self, neuron_id: str) -> NeuronState | None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(neuron_id)
        try:
            result = await conn.select(f"neuron_state:{sid}")
            if result:
                return _row_to_neuron_state(result[0] if isinstance(result, list) else result)
        except Exception:
            pass
        return None

    async def update_neuron_state(self, state: NeuronState) -> None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(state.neuron_id)

        update_data: dict[str, Any] = {
            "activation_level": state.activation_level,
            "access_frequency": state.access_frequency,
            "decay_rate": state.decay_rate,
            "firing_threshold": state.firing_threshold,
            "refractory_period_ms": state.refractory_period_ms,
            "homeostatic_target": state.homeostatic_target,
        }
        if state.last_activated:
            update_data["last_activated"] = state.last_activated
        if state.refractory_until:
            update_data["refractory_until"] = state.refractory_until

        await conn.merge(f"neuron_state:{sid}", update_data)

    # ================================================================
    # Synapse Operations
    # ================================================================

    async def add_synapse(self, synapse: Synapse) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        sid = _to_surreal_id(synapse.id)
        ss = _to_surreal_id(synapse.source_id)
        st = _to_surreal_id(synapse.target_id)

        record_data: dict[str, Any] = {
            "id": sid,
            "brain_id": brain_id,
            "type": synapse.type.value,
            "source_id": ss,
            "target_id": st,
            "weight": synapse.weight,
            "direction": synapse.direction,
            "metadata": dict(synapse.metadata),
            "created_at": synapse.created_at,
            "reinforced_count": synapse.reinforced_count,
        }
        await conn.insert("synapse", record_data)

        # Create graph edge for traversal
        try:
            await self._query(
                "RELATE neuron:$src -> connects_to -> neuron:$tgt SET brain_id = $brain_id",
                src=ss,
                tgt=st,
                brain_id=brain_id,
            )
        except Exception:
            pass

        await self._record_change_internal("synapse", synapse.id, "insert", synapse)
        return synapse.id

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(synapse_id)
        try:
            result = await conn.select(f"synapse:{sid}")
            if result:
                return _row_to_synapse(result[0] if isinstance(result, list) else result)
        except Exception:
            pass
        return None

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if source_id is not None:
            conditions.append("source_id = $source_id")
            params["source_id"] = source_id
        if target_id is not None:
            conditions.append("target_id = $target_id")
            params["target_id"] = target_id
        if type is not None:
            conditions.append("type = $stype")
            params["stype"] = type.value
        if min_weight is not None:
            conditions.append("weight >= $min_weight")
            params["min_weight"] = min_weight

        where = " AND ".join(conditions)
        rows = await self._query(f"SELECT * FROM synapse WHERE {where}", **params)
        return [_row_to_synapse(r) for r in rows]

    async def update_synapse(self, synapse: Synapse) -> None:
        conn = self._ensure_conn()
        sid = _to_surreal_id(synapse.id)

        update_data: dict[str, Any] = {
            "type": synapse.type.value,
            "weight": synapse.weight,
            "direction": synapse.direction,
            "metadata": dict(synapse.metadata),
            "reinforced_count": synapse.reinforced_count,
        }
        if synapse.last_activated:
            update_data["last_activated"] = synapse.last_activated

        await conn.merge(f"synapse:{sid}", update_data)
        await self._record_change_internal("synapse", synapse.id, "update", synapse)

    async def delete_synapse(self, synapse_id: str) -> bool:
        conn = self._ensure_conn()
        sid = _to_surreal_id(synapse_id)
        try:
            await conn.delete(f"synapse:{sid}")
            await self._record_change_internal("synapse", synapse_id, "delete")
            return True
        except Exception:
            return False

    # ================================================================
    # Graph Traversal
    # ================================================================

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        brain_id = self._get_brain_id()

        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if direction == "out":
            conditions.append("source_id = $nid")
        elif direction == "in":
            conditions.append("target_id = $nid")
        else:
            conditions.append("(source_id = $nid OR target_id = $nid)")
        params["nid"] = _to_surreal_id(neuron_id)

        if synapse_types:
            type_vals = [t.value for t in synapse_types]
            type_list = ", ".join(f"'{t}'" for t in type_vals)
            conditions.append(f"type IN [{type_list}]")

        if min_weight is not None:
            conditions.append("weight >= $min_weight")
            params["min_weight"] = min_weight

        where = " AND ".join(conditions)
        syn_rows = await self._query(f"SELECT * FROM synapse WHERE {where}", **params)

        results: list[tuple[Neuron, Synapse]] = []
        for sr in syn_rows:
            syn = _row_to_synapse(sr)
            # Get the neighbor neuron
            neighbor_id = syn.target_id if syn.source_id == neuron_id else syn.source_id
            neighbor = await self.get_neuron(neighbor_id)
            if neighbor is not None:
                results.append((neighbor, syn))
        return results

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        bidirectional: bool = False,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """BFS path finding between two neurons via synapses."""
        if source_id == target_id:
            src = await self.get_neuron(source_id)
            return (
                [(src, Synapse.create(source_id, target_id, SynapseType.RELATED_TO))]
                if src
                else None
            )

        visited: set[str] = {source_id}
        queue: list[tuple[str, list[tuple[Neuron, Synapse]]]] = [(source_id, [])]

        while queue and len(queue[0][1]) < max_hops:
            current_id, path = queue.pop(0)
            dir_literal: Literal["out", "in", "both"] = "both" if bidirectional else "out"
            neighbors = await self.get_neighbors(current_id, direction=dir_literal)

            for neighbor, synapse in neighbors:
                if neighbor.id == target_id:
                    return path + [(neighbor, synapse)]
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [(neighbor, synapse)]))

        return None

    # ================================================================
    # Fiber Operations
    # ================================================================

    async def add_fiber(self, fiber: Fiber) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        fid = _to_surreal_id(fiber.id)

        record_data: dict[str, Any] = {
            "id": fid,
            "brain_id": brain_id,
            "neuron_ids": list(fiber.neuron_ids),
            "synapse_ids": list(fiber.synapse_ids),
            "anchor_neuron_id": fiber.anchor_neuron_id,
            "pathway": fiber.pathway,
            "conductivity": fiber.conductivity,
            "coherence": fiber.coherence,
            "salience": fiber.salience,
            "frequency": fiber.frequency,
            "summary": fiber.summary,
            "essence": fiber.essence,
            "auto_tags": list(fiber.auto_tags),
            "agent_tags": list(fiber.agent_tags),
            "metadata": dict(fiber.metadata),
            "compression_tier": fiber.compression_tier,
            "pinned": fiber.pinned,
            "created_at": fiber.created_at,
        }
        if fiber.last_conducted:
            record_data["last_conducted"] = fiber.last_conducted
        if fiber.time_start:
            record_data["time_start"] = fiber.time_start
        if fiber.time_end:
            record_data["time_end"] = fiber.time_end
        if fiber.last_ghost_shown_at:
            record_data["last_ghost_shown_at"] = fiber.last_ghost_shown_at

        await conn.insert("fiber", record_data)
        await self._record_change_internal("fiber", fiber.id, "insert")
        return fiber.id

    async def get_fiber(self, fiber_id: str) -> Fiber | None:
        conn = self._ensure_conn()
        fid = _to_surreal_id(fiber_id)
        try:
            result = await conn.select(f"fiber:{fid}")
            if result:
                return _row_to_fiber(result[0] if isinstance(result, list) else result)
        except Exception:
            pass
        return None

    async def find_fibers(
        self,
        contains_neuron: str | None = None,
        time_overlaps: tuple[datetime, datetime] | None = None,
        tags: set[str] | None = None,
        min_salience: float | None = None,
        metadata_key: str | None = None,
        limit: int = 100,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if contains_neuron:
            conditions.append("$contains_neuron IN neuron_ids")
            params["contains_neuron"] = contains_neuron
        if min_salience is not None:
            conditions.append("salience >= $min_salience")
            params["min_salience"] = min_salience

        where = " AND ".join(conditions)
        rows = await self._query(f"SELECT * FROM fiber WHERE {where} LIMIT {int(limit)}", **params)

        fibers = [_row_to_fiber(r) for r in rows]

        # Post-filter for complex conditions
        if tags:
            fibers = [f for f in fibers if tags.issubset(f.tags)]
        if time_overlaps:
            start, end = time_overlaps
            # Normalize to naive UTC for comparison
            if start.tzinfo is not None:
                start = start.replace(tzinfo=None)
            if end.tzinfo is not None:
                end = end.replace(tzinfo=None)
            fibers = [
                f
                for f in fibers
                if f.time_start
                and f.time_end
                and _ensure_naive(f.time_start) <= end
                and _ensure_naive(f.time_end) >= start
            ]
        if metadata_key:
            fibers = [f for f in fibers if metadata_key in f.metadata]

        return fibers[:limit]

    async def update_fiber(self, fiber: Fiber) -> None:
        conn = self._ensure_conn()
        fid = _to_surreal_id(fiber.id)

        update_data: dict[str, Any] = {
            "neuron_ids": list(fiber.neuron_ids),
            "synapse_ids": list(fiber.synapse_ids),
            "anchor_neuron_id": fiber.anchor_neuron_id,
            "pathway": fiber.pathway,
            "conductivity": fiber.conductivity,
            "coherence": fiber.coherence,
            "salience": fiber.salience,
            "frequency": fiber.frequency,
            "summary": fiber.summary,
            "essence": fiber.essence,
            "auto_tags": list(fiber.auto_tags),
            "agent_tags": list(fiber.agent_tags),
            "metadata": dict(fiber.metadata),
            "compression_tier": fiber.compression_tier,
            "pinned": fiber.pinned,
        }
        if fiber.last_conducted:
            update_data["last_conducted"] = fiber.last_conducted
        if fiber.time_start:
            update_data["time_start"] = fiber.time_start
        if fiber.time_end:
            update_data["time_end"] = fiber.time_end
        if fiber.last_ghost_shown_at:
            update_data["last_ghost_shown_at"] = fiber.last_ghost_shown_at

        await conn.merge(f"fiber:{fid}", update_data)
        await self._record_change_internal("fiber", fiber.id, "update")

    async def delete_fiber(self, fiber_id: str) -> bool:
        conn = self._ensure_conn()
        fid = _to_surreal_id(fiber_id)
        try:
            await conn.delete(f"fiber:{fid}")
            await self._record_change_internal("fiber", fiber_id, "delete")
            return True
        except Exception:
            return False

    async def get_fibers(
        self,
        limit: int = 10,
        order_by: Literal["created_at", "salience", "frequency"] = "created_at",
        descending: bool = True,
    ) -> list[Fiber]:
        brain_id = self._get_brain_id()
        order_dir = "DESC" if descending else "ASC"
        rows = await self._query(
            f"SELECT * FROM fiber WHERE brain_id = $brain_id ORDER BY {order_by} {order_dir} LIMIT {int(limit)}",
            brain_id=brain_id,
        )
        return [_row_to_fiber(r) for r in rows]

    # ================================================================
    # Brain Operations
    # ================================================================

    async def save_brain(self, brain: Brain) -> None:
        conn = self._ensure_conn()

        record_data: dict[str, Any] = {
            "id": brain.id,  # Use original ID to avoid underscore conversion
            "name": brain.name,
            "config": dict(brain.metadata),
            "metadata": dict(brain.metadata),
            "created_at": brain.created_at,
            "updated_at": brain.updated_at,
        }
        try:
            await conn.insert("brain", record_data)
        except Exception as e:
            # Try to update existing record
            try:
                await conn.merge(f"brain:{brain.id}", record_data)
            except Exception:
                # Query and update if merge also fails
                rows = await self._query(
                    "SELECT * FROM brain WHERE id = $id", id=f"brain:{brain.id}"
                )
                if rows:
                    for field, value in record_data.items():
                        await self._query(
                            f"UPDATE brain:{brain.id} SET {field} = $value", value=value
                        )

    async def get_brain(self, brain_id: str) -> Brain | None:
        conn = self._ensure_conn()
        try:
            # Query all brains and filter manually (string matching in SurrealDB is problematic)
            rows = await self._query("SELECT * FROM brain")
            if rows and len(rows) > 0:
                target_prefix = f"brain:{brain_id}"
                for r in rows:
                    rid = r["id"]
                    # Compare record ID string
                    rid_str = str(rid) if not hasattr(rid, "id") else f"brain:{rid.id}"
                    if brain_id in rid_str or rid_str.endswith(f":{brain_id}"):
                        bid_str = (
                            str(rid.id).replace("_", "-")
                            if hasattr(rid, "id")
                            else str(rid).split(":")[-1].replace("_", "-")
                        )
                        return Brain(
                            id=bid_str,
                            name=str(r["name"]),
                            metadata=dict(r.get("metadata") or {}),
                            created_at=_parse_datetime(r.get("created_at")) or utcnow(),
                            updated_at=_parse_datetime(r.get("updated_at")) or utcnow(),
                        )
        except Exception:
            pass
        return None

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError(f"Brain {brain_id} not found")

        raw_neurons = await self.find_neurons(limit=100000)
        raw_synapses = await self.get_synapses()
        raw_fibers = await self.get_fibers(limit=100000)

        neurons: list[dict[str, Any]] = [
            {
                "id": n.id,
                "type": n.type.value,
                "content": n.content,
                "metadata": dict(n.metadata),
                "created_at": n.created_at.isoformat(),
            }
            for n in raw_neurons
        ]
        synapses: list[dict[str, Any]] = [
            {
                "id": s.id,
                "source_id": s.source_id,
                "target_id": s.target_id,
                "type": s.type.value,
                "weight": s.weight,
                "direction": s.direction.value,
                "metadata": dict(s.metadata),
            }
            for s in raw_synapses
        ]
        fibers: list[dict[str, Any]] = [
            {
                "id": f.id,
                "neuron_ids": list(f.neuron_ids),
                "synapse_ids": list(f.synapse_ids),
                "anchor_neuron_id": f.anchor_neuron_id,
                "pathway": f.pathway,
                "conductivity": f.conductivity,
                "salience": f.salience,
            }
            for f in raw_fibers
        ]

        return BrainSnapshot(
            brain_id=brain_id,
            brain_name=brain.name,
            exported_at=utcnow(),
            version="0.1.0",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config=dict(brain.metadata),
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        bid = target_brain_id or snapshot.brain_id
        self.set_brain(bid)

        brain = Brain(
            id=bid,
            name=snapshot.brain_name,
            metadata=dict(snapshot.config),
        )
        await self.save_brain(brain)

        for nd in snapshot.neurons:
            try:
                neuron = Neuron(
                    id=str(nd.get("id", "")),
                    type=NeuronType(nd["type"]),
                    content=str(nd["content"]),
                    metadata=dict(nd.get("metadata") or {}),
                    created_at=_parse_datetime(nd.get("created_at")) or utcnow(),
                )
                await self.add_neuron(neuron)
            except Exception:
                pass

        for sd in snapshot.synapses:
            try:
                synapse = Synapse(
                    id=str(sd.get("id", "")),
                    source_id=str(sd.get("source_id", "")),
                    target_id=str(sd.get("target_id", "")),
                    type=SynapseType(sd["type"]),
                    weight=float(sd.get("weight", 1.0)),
                    direction=Direction(str(sd.get("direction", "forward"))),
                    metadata=dict(sd.get("metadata") or {}),
                    created_at=_parse_datetime(sd.get("created_at")) or utcnow(),
                )
                await self.add_synapse(synapse)
            except Exception:
                pass

        for fd in snapshot.fibers:
            try:
                fiber = Fiber(
                    id=str(fd.get("id", "")),
                    neuron_ids=set(fd.get("neuron_ids") or []),
                    synapse_ids=set(fd.get("synapse_ids") or []),
                    anchor_neuron_id=str(fd.get("anchor_neuron_id", "")),
                    pathway=list(fd.get("pathway") or []),
                    conductivity=float(fd.get("conductivity", 1.0)),
                    salience=float(fd.get("salience", 0.0)),
                )
                await self.add_fiber(fiber)
            except Exception:
                pass

        return bid

    # ================================================================
    # Statistics
    # ================================================================

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        neuron_rows = await self._query(
            "SELECT count() AS c FROM neuron WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )
        synapse_rows = await self._query(
            "SELECT count() AS c FROM synapse WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )
        fiber_rows = await self._query(
            "SELECT count() AS c FROM fiber WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )

        def _count(rows: list[Any]) -> int:
            if rows and len(rows) > 0:
                return int(rows[0].get("c", 0))
            return 0

        return {
            "neuron_count": _count(neuron_rows),
            "synapse_count": _count(synapse_rows),
            "fiber_count": _count(fiber_rows),
        }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        stats = await self.get_stats(brain_id)

        # Type breakdown
        type_rows = await self._query(
            "SELECT type, count() AS c FROM neuron WHERE brain_id = $bid GROUP BY type",
            bid=brain_id,
        )
        type_counts = {str(r.get("type", "unknown")): int(r.get("c", 0)) for r in type_rows}

        return {
            **stats,
            "neuron_types": type_counts,
        }

    # ================================================================
    # Clear
    # ================================================================

    async def clear(self, brain_id: str) -> None:
        await self._query("DELETE neuron WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE neuron_state WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE synapse WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE fiber WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE change_log WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE device WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE merkle_hash WHERE brain_id = $bid", bid=brain_id)
        await self._query("DELETE typed_memory WHERE brain_id = $bid", bid=brain_id)

    # ================================================================
    # Vector Search (for cone queries / semantic search)
    # ================================================================

    async def find_neurons_by_embedding(
        self,
        query_embedding: list[float],
        limit: int = 10,
        type_filter: NeuronType | None = None,
    ) -> list[tuple[Neuron, float]]:
        """Find neurons by vector similarity using SurrealDB KNN operator."""
        brain_id = self._get_brain_id()
        conditions = [
            "brain_id = $brain_id",
        ]
        params: dict[str, Any] = {
            "brain_id": brain_id,
            "vec": query_embedding,
        }

        if type_filter is not None:
            conditions.append("type = $ntype")
            params["ntype"] = type_filter.value

        where = " AND ".join(conditions)
        # SurrealDB KNN syntax: WHERE embedding_vec <|k, ef|> $vec
        rows = await self._query(
            f"SELECT *, vector::distance::knn() AS score "
            f"FROM neuron WHERE {where} AND embedding_vec <|{int(limit)},100|> $vec",
            **params,
        )

        results: list[tuple[Neuron, float]] = []
        for r in rows:
            raw_score = r.pop("score", None)
            score = float(raw_score) if raw_score is not None else 0.0
            # SurrealDB returns distance (lower = more similar), convert to similarity
            similarity = 1.0 / (1.0 + score) if score >= 0 else 0.0
            results.append((_row_to_neuron(r), similarity))
        return results

    # ================================================================
    # Change Log (for sync)
    # ================================================================

    async def _record_change_internal(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        entity: Any | None = None,
        device_id: str = "",
    ) -> None:
        """Internal helper to record a change."""
        try:
            brain_id = self._get_brain_id()
        except ValueError:
            return

        payload: dict[str, Any] | None = None
        if entity is not None:
            if isinstance(entity, Neuron):
                meta = dict(entity.metadata)
                meta.pop("_embedding", None)
                payload = {
                    "type": entity.type.value,
                    "content": entity.content,
                    "metadata": meta,
                    "content_hash": entity.content_hash,
                    "ephemeral": entity.ephemeral,
                }
            elif isinstance(entity, Synapse):
                payload = {
                    "type": entity.type.value,
                    "source_id": entity.source_id,
                    "target_id": entity.target_id,
                    "weight": entity.weight,
                    "direction": entity.direction,
                }

        conn = self._ensure_conn()
        self._change_seq += 1
        change_id = f"change_{self._change_seq}_{uuid4().hex[:8]}"
        record = {
            "id": change_id,
            "brain_id": brain_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation,
            "device_id": device_id,
            "payload": payload,
            "changed_at": utcnow(),
            "synced": False,
            "sequence": self._change_seq,
        }
        try:
            await conn.insert("change_log", record)
        except Exception:
            pass

    async def record_change(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        device_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> int:
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        self._change_seq += 1
        change_id = f"change_{self._change_seq}_{uuid4().hex[:8]}"
        record = {
            "id": change_id,
            "brain_id": brain_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "operation": operation,
            "device_id": device_id,
            "payload": payload,
            "changed_at": utcnow(),
            "synced": False,
            "sequence": self._change_seq,
        }
        await conn.insert("change_log", record)
        return self._change_seq

    async def get_changes_since(self, sequence: int = 0, limit: int = 1000) -> list[Any]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM change_log WHERE brain_id = $brain_id AND sequence > $seq "
            "ORDER BY sequence ASC LIMIT $limit",
            brain_id=brain_id,
            seq=sequence,
            limit=limit,
        )
        from neural_memory.sync.protocol import SyncChange

        changes = []
        for r in rows:
            changes.append(
                SyncChange(
                    sequence=int(r.get("sequence", 0)),
                    entity_type=str(r.get("entity_type", "")),
                    entity_id=str(r.get("entity_id", "")),
                    operation=str(r.get("operation", "")),
                    device_id=str(r.get("device_id", "")),
                    changed_at=str(r.get("changed_at", "")),
                    payload=r.get("payload") or {},
                )
            )
        return changes

    async def get_unsynced_changes(self, limit: int = 1000) -> list[Any]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM change_log WHERE brain_id = $brain_id AND synced = false "
            "ORDER BY sequence ASC LIMIT $limit",
            brain_id=brain_id,
            limit=limit,
        )
        from neural_memory.sync.protocol import SyncChange

        return [
            SyncChange(
                sequence=int(r.get("sequence", 0)),
                entity_type=str(r.get("entity_type", "")),
                entity_id=str(r.get("entity_id", "")),
                operation=str(r.get("operation", "")),
                device_id=str(r.get("device_id", "")),
                changed_at=str(r.get("changed_at", "")),
                payload=r.get("payload") or {},
            )
            for r in rows
        ]

    async def mark_synced(self, up_to_sequence: int) -> int:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM change_log WHERE brain_id = $brain_id AND sequence <= $seq AND synced = false",
            brain_id=brain_id,
            seq=up_to_sequence,
        )
        count = 0
        for r in rows:
            cid = str(r.get("id", ""))
            try:
                await self._conn.merge(cid, {"synced": True})
                count += 1
            except Exception:
                pass
        return count

    async def prune_synced_changes(self, older_than_days: int = 30) -> int:
        brain_id = self._get_brain_id()
        await self._query(
            "DELETE FROM change_log WHERE brain_id = $brain_id AND synced = true "
            "AND changed_at < time::ago($days, 'd')",
            brain_id=brain_id,
            days=older_than_days,
        )
        return 0

    async def seed_change_log(self, device_id: str = "") -> dict[str, int]:
        brain_id = self._get_brain_id()

        # Check what's already seeded
        existing = await self._query(
            "SELECT entity_id FROM change_log WHERE brain_id = $brain_id AND operation = 'insert' GROUP BY entity_id",
            brain_id=brain_id,
        )
        existing_ids = {str(r.get("entity_id", "")) for r in existing}

        counts = {"neurons": 0, "synapses": 0, "fibers": 0}

        # Seed neurons
        neurons = await self.find_neurons(limit=100000)
        for n in neurons:
            if n.id not in existing_ids:
                await self.record_change("neuron", n.id, "insert", device_id)
                counts["neurons"] += 1

        # Seed synapses
        synapses = await self.get_synapses()
        for s in synapses:
            if s.id not in existing_ids:
                await self.record_change("synapse", s.id, "insert", device_id)
                counts["synapses"] += 1

        # Seed fibers
        fibers = await self.get_fibers(limit=100000)
        for f in fibers:
            if f.id not in existing_ids:
                await self.record_change("fiber", f.id, "insert", device_id)
                counts["fibers"] += 1

        return counts

    async def get_change_log_stats(self) -> dict[str, Any]:
        brain_id = self._get_brain_id()
        total_rows = await self._query(
            "SELECT count() AS c FROM change_log WHERE brain_id = $bid GROUP ALL",
            bid=brain_id,
        )
        pending_rows = await self._query(
            "SELECT count() AS c FROM change_log WHERE brain_id = $bid AND synced = false GROUP ALL",
            bid=brain_id,
        )
        synced_rows = await self._query(
            "SELECT count() AS c FROM change_log WHERE brain_id = $bid AND synced = true GROUP ALL",
            bid=brain_id,
        )
        last_seq_rows = await self._query(
            "SELECT sequence FROM change_log WHERE brain_id = $bid ORDER BY sequence DESC LIMIT 1",
            bid=brain_id,
        )

        def _cnt(rows: list[Any]) -> int:
            return int(rows[0].get("c", 0)) if rows else 0


        def _max(rows: list) -> int:
            return int(rows[0].get("sequence", 0)) if rows else 0

        return {
            "total": _cnt(total_rows),
            "pending": _cnt(pending_rows),
            "synced": _cnt(synced_rows),
            "last_sequence": _max(last_seq_rows),
        }

    # ================================================================
    # Device Registry
    # ================================================================

    async def register_device(self, device_id: str, device_name: str = "") -> Any:
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        did = _to_surreal_id(device_id)

        from neural_memory.sync.device import DeviceInfo

        record = {
            "id": f"{brain_id}_{did}",
            "device_id": device_id,
            "brain_id": brain_id,
            "device_name": device_name,
            "registered_at": utcnow(),
            "last_sync_sequence": 0,
        }
        try:
            await conn.insert("device", record)
        except Exception:
            await conn.merge(f"device:{brain_id}_{did}", record)

        return DeviceInfo(
            device_id=device_id,
            device_name=device_name,
            registered_at=utcnow(),
        )

    async def get_device(self, device_id: str) -> Any | None:
        brain_id = self._get_brain_id()
        did = _to_surreal_id(device_id)
        try:
            result = await self._conn.select(f"device:{brain_id}_{did}")
            if result:
                r = result[0] if isinstance(result, list) else result
                from neural_memory.sync.device import DeviceInfo

                return DeviceInfo(
                    device_id=str(r.get("device_id", device_id)),
                    device_name=str(r.get("device_name", "")),
                    registered_at=_parse_datetime(r.get("registered_at")) or utcnow(),
                )
        except Exception:
            pass
        return None

    async def list_devices(self) -> list[Any]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM device WHERE brain_id = $brain_id ORDER BY registered_at ASC",
            brain_id=brain_id,
        )
        from neural_memory.sync.device import DeviceInfo

        return [
            DeviceInfo(
                device_id=str(r.get("device_id", "")),
                device_name=str(r.get("device_name", "")),
                registered_at=_parse_datetime(r.get("registered_at")) or utcnow(),
            )
            for r in rows
        ]

    async def update_device_sync(self, device_id: str, last_sync_sequence: int) -> None:
        brain_id = self._get_brain_id()
        did = _to_surreal_id(device_id)
        try:
            await self._conn.merge(
                f"device:{brain_id}_{did}",
                {
                    "last_sync_at": utcnow(),
                    "last_sync_sequence": last_sync_sequence,
                },
            )
        except Exception:
            pass

    async def remove_device(self, device_id: str) -> bool:
        brain_id = self._get_brain_id()
        did = _to_surreal_id(device_id)
        try:
            await self._conn.delete(f"device:{brain_id}_{did}")
            return True
        except Exception:
            return False

    # ================================================================
    # Merkle Hash Operations
    # ================================================================

    async def compute_merkle_root(self, entity_type: str, *, is_pro: bool = False) -> str | None:
        if not is_pro:
            return None

        brain_id = self._get_brain_id()
        rows = await self._query(
            f"SELECT id, updated_at FROM {entity_type} WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )

        # Build 2-level prefix tree
        buckets: dict[str, list[str]] = {}
        for r in rows:
            eid = str(r.get("id", "")).split(":")[-1]
            if len(eid) >= 2:
                prefix = eid[:2]
                buckets.setdefault(prefix, []).append(eid)

        conn = self._ensure_conn()
        # Build lookup for updated_at by entity ID
        updated_lookup = {}
        for row in rows:
            eid_raw = str(row.get("id", "")).split(":")[-1]
            updated_lookup[eid_raw] = str(row.get("updated_at", ""))

        # Compute and store bucket hashes
        for prefix, ids in sorted(buckets.items()):
            ids_sorted = sorted(ids)
            leaf_hashes = []
            for eid in ids_sorted:
                updated = updated_lookup.get(eid, "")
                leaf_hash = sha256(f"{eid}|{updated}".encode()).hexdigest()
                leaf_hashes.append(leaf_hash)

            bucket_hash = sha256("|".join(sorted(leaf_hashes)).encode()).hexdigest()
            merkle_id = f"{brain_id}_{entity_type}_{prefix}"

            try:
                await conn.insert(
                    "merkle_hash",
                    {
                        "id": _to_surreal_id(merkle_id),
                        "brain_id": brain_id,
                        "entity_type": entity_type,
                        "prefix": prefix,
                        "hash": bucket_hash,
                        "computed_at": utcnow(),
                    },
                )
            except Exception:
                await conn.merge(
                    f"merkle_hash:{_to_surreal_id(merkle_id)}",
                    {"hash": bucket_hash, "computed_at": utcnow()},
                )

        # Compute root from bucket hashes
        all_hashes = sorted(
            [sha256("|".join(sorted(buckets[p])).encode()).hexdigest() for p in sorted(buckets)]
        )
        root = sha256("|".join(all_hashes).encode()).hexdigest()
        return root

    async def get_merkle_tree(self, entity_type: str, *, is_pro: bool = False) -> dict[str, str]:
        if not is_pro:
            return {}
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT prefix, hash FROM merkle_hash WHERE brain_id = $brain_id AND entity_type = $et",
            brain_id=brain_id,
            et=entity_type,
        )
        return {str(r["prefix"]): str(r["hash"]) for r in rows}

    async def invalidate_merkle_prefix(
        self, entity_type: str, entity_id: str, *, is_pro: bool = False
    ) -> None:
        if not is_pro:
            return
        # Just recompute on next sync — no-op here is fine for v1

    async def get_merkle_root(self, *, is_pro: bool = False) -> str | None:
        if not is_pro:
            return None
        roots = []
        for et in ("neuron", "synapse", "fiber"):
            r = await self.compute_merkle_root(et, is_pro=is_pro)
            if r:
                roots.append(r)
        if not roots:
            return None
        return sha256("|".join(sorted(roots)).encode()).hexdigest()

    async def get_bucket_entity_ids(
        self, entity_type: str, prefix: str, *, is_pro: bool = False
    ) -> list[str]:
        if not is_pro:
            return []
        brain_id = self._get_brain_id()
        rows = await self._query(
            f"SELECT id FROM {entity_type} WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        return [
            str(r["id"]).split(":")[-1]
            for r in rows
            if str(r["id"]).split(":")[-1].startswith(prefix)
        ]

    # ================================================================
    # Lifecycle Operations
    # ================================================================

    async def update_neuron_lifecycle(self, neuron_id: str, lifecycle_state: str) -> None:
        sid = _to_surreal_id(neuron_id)
        try:
            await self._conn.merge(f"neuron:{sid}", {"lifecycle_state": lifecycle_state})
        except Exception:
            pass

    async def update_neuron_frozen(self, neuron_id: str, frozen: bool) -> None:
        sid = _to_surreal_id(neuron_id)
        try:
            await self._conn.merge(f"neuron:{sid}", {"frozen": frozen})
        except Exception:
            pass

    async def get_lifecycle_distribution(self) -> dict[str, int]:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT lifecycle_state, count() AS c FROM neuron WHERE brain_id = $bid GROUP BY lifecycle_state",
            bid=brain_id,
        )
        return {str(r.get("lifecycle_state", "active")): int(r.get("c", 0)) for r in rows}
