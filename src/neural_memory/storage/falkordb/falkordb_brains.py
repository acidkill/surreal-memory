"""FalkorDB brain operations (multi-graph isolation)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
from datetime import datetime
from typing import Any

from neural_memory.core.brain import Brain, BrainConfig, BrainSnapshot
from neural_memory.storage.falkordb.falkordb_base import FalkorDBBaseMixin
from neural_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


class FalkorDBBrainMixin(FalkorDBBaseMixin):
    """FalkorDB brain operations using multi-graph for isolation.

    Each brain maps to a separate FalkorDB graph (brain_{id}).
    Brain metadata is stored in a dedicated 'brain_registry' graph.
    """

    _REGISTRY_GRAPH = "brain_registry"
    _registry_initialized: bool = False

    async def _ensure_registry(self) -> None:
        """Ensure the brain registry graph exists with indexes (once per session)."""
        if self._registry_initialized:
            return
        db = self._ensure_db()
        if self._REGISTRY_GRAPH not in self._graphs:
            self._graphs[self._REGISTRY_GRAPH] = db.select_graph(self._REGISTRY_GRAPH)
        graph = self._graphs[self._REGISTRY_GRAPH]
        try:
            await graph.query(
                "CREATE INDEX IF NOT EXISTS FOR (b:BrainMeta) ON (b.id)"
            )
            await graph.query(
                "CREATE INDEX IF NOT EXISTS FOR (b:BrainMeta) ON (b.name)"
            )
        except Exception:
            logger.debug("Registry index creation may have failed", exc_info=True)
            return
        self._registry_initialized = True

    async def _registry_query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        """Execute a write query against the brain registry graph."""
        await self._ensure_registry()
        graph = self._graphs[self._REGISTRY_GRAPH]
        result = await graph.query(cypher, params=params or {})
        return result.result_set if result.result_set else []

    async def _registry_query_ro(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        """Execute a read-only query against the brain registry graph."""
        await self._ensure_registry()
        graph = self._graphs[self._REGISTRY_GRAPH]
        try:
            result = await graph.ro_query(cypher, params=params or {})
        except AttributeError:
            result = await graph.query(cypher, params=params or {})
        return result.result_set if result.result_set else []

    async def save_brain(self, brain: Brain) -> None:
        config_json = json.dumps(asdict(brain.config), default=str)
        shared_json = json.dumps(brain.shared_with)
        metadata_json = self._serialize_metadata(brain.metadata)

        await self._registry_query(
            """
            MERGE (b:BrainMeta {id: $id})
            SET b.name = $name,
                b.config = $config,
                b.owner_id = $owner_id,
                b.is_public = $is_public,
                b.shared_with = $shared_with,
                b.neuron_count = $neuron_count,
                b.synapse_count = $synapse_count,
                b.fiber_count = $fiber_count,
                b.metadata = $metadata,
                b.created_at = $created_at,
                b.updated_at = $updated_at
            """,
            {
                "id": brain.id,
                "name": brain.name,
                "config": config_json,
                "owner_id": brain.owner_id,
                "is_public": brain.is_public,
                "shared_with": shared_json,
                "neuron_count": brain.neuron_count,
                "synapse_count": brain.synapse_count,
                "fiber_count": brain.fiber_count,
                "metadata": metadata_json,
                "created_at": self._dt_to_str(brain.created_at),
                "updated_at": self._dt_to_str(utcnow()),
            },
        )

    async def get_brain(self, brain_id: str) -> Brain | None:
        rows = await self._registry_query_ro(
            """
            MATCH (b:BrainMeta {id: $id})
            RETURN b.id, b.name, b.config, b.owner_id, b.is_public,
                   b.shared_with, b.neuron_count, b.synapse_count,
                   b.fiber_count, b.metadata, b.created_at, b.updated_at
            """,
            {"id": brain_id},
        )
        if not rows:
            return None
        return self._row_to_brain(rows[0])

    async def find_brain_by_name(self, name: str) -> Brain | None:
        rows = await self._registry_query_ro(
            """
            MATCH (b:BrainMeta {name: $name})
            RETURN b.id, b.name, b.config, b.owner_id, b.is_public,
                   b.shared_with, b.neuron_count, b.synapse_count,
                   b.fiber_count, b.metadata, b.created_at, b.updated_at
            """,
            {"name": name},
        )
        if not rows:
            return None
        return self._row_to_brain(rows[0])

    async def export_brain(self, brain_id: str) -> BrainSnapshot:
        brain = await self.get_brain(brain_id)
        if brain is None:
            raise ValueError("Brain not found")

        # Export all neurons
        neuron_rows = await self._query_ro(
            """
            MATCH (n:Neuron)
            RETURN n.id, n.type, n.content, n.metadata,
                   n.content_hash, n.created_at,
                   n.activation_level, n.access_frequency,
                   n.last_activated, n.decay_rate,
                   n.firing_threshold, n.refractory_until,
                   n.refractory_period_ms, n.homeostatic_target
            """,
            brain_id=brain_id,
        )
        neurons = []
        for r in neuron_rows:
            neurons.append({
                "id": r[0], "type": r[1], "content": r[2],
                "metadata": r[3], "content_hash": r[4],
                "created_at": r[5],
                "activation_level": r[6], "access_frequency": r[7],
                "last_activated": r[8], "decay_rate": r[9],
                "firing_threshold": r[10], "refractory_until": r[11],
                "refractory_period_ms": r[12], "homeostatic_target": r[13],
            })

        # Export all synapses
        synapse_rows = await self._query_ro(
            """
            MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
            RETURN s.id, a.id, b.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count, s.last_activated,
                   s.created_at
            """,
            brain_id=brain_id,
        )
        synapses = []
        for r in synapse_rows:
            synapses.append({
                "id": r[0], "source_id": r[1], "target_id": r[2],
                "type": r[3], "weight": r[4], "direction": r[5],
                "metadata": r[6], "reinforced_count": r[7],
                "last_activated": r[8], "created_at": r[9],
            })

        # Export all fibers
        fiber_rows = await self._query_ro(
            """
            MATCH (f:Fiber)
            RETURN f.id, f.anchor_neuron_id, f.pathway, f.conductivity,
                   f.last_conducted, f.time_start, f.time_end,
                   f.coherence, f.salience, f.frequency, f.summary,
                   f.auto_tags, f.agent_tags, f.metadata,
                   f.compression_tier, f.created_at,
                   f.neuron_ids, f.synapse_ids
            """,
            brain_id=brain_id,
        )
        fibers = []
        for r in fiber_rows:
            fibers.append({
                "id": r[0], "anchor_neuron_id": r[1], "pathway": r[2],
                "conductivity": r[3], "last_conducted": r[4],
                "time_start": r[5], "time_end": r[6],
                "coherence": r[7], "salience": r[8], "frequency": r[9],
                "summary": r[10], "auto_tags": r[11], "agent_tags": r[12],
                "metadata": r[13], "compression_tier": r[14],
                "created_at": r[15], "neuron_ids": r[16], "synapse_ids": r[17],
            })

        return BrainSnapshot(
            brain_id=brain.id,
            brain_name=brain.name,
            exported_at=utcnow(),
            version="falkordb-1",
            neurons=neurons,
            synapses=synapses,
            fibers=fibers,
            config=asdict(brain.config),
            metadata=brain.metadata,
        )

    async def import_brain(
        self,
        snapshot: BrainSnapshot,
        target_brain_id: str | None = None,
    ) -> str:
        bid = target_brain_id or snapshot.brain_id

        # Clear existing data in target graph
        await self._query(
            "MATCH (n) DETACH DELETE n",
            brain_id=bid,
        )

        # Import neurons
        for n in snapshot.neurons:
            metadata = n.get("metadata", "{}")
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata, default=str)
            await self._query(
                """
                CREATE (n:Neuron {
                    id: $id, type: $type, content: $content,
                    metadata: $metadata, content_hash: $content_hash,
                    created_at: $created_at,
                    activation_level: $activation_level,
                    access_frequency: $access_frequency,
                    last_activated: $last_activated,
                    decay_rate: $decay_rate,
                    firing_threshold: $firing_threshold,
                    refractory_until: $refractory_until,
                    refractory_period_ms: $refractory_period_ms,
                    homeostatic_target: $homeostatic_target
                })
                """,
                {
                    "id": n["id"],
                    "type": n.get("type", "concept"),
                    "content": n.get("content", ""),
                    "metadata": metadata,
                    "content_hash": n.get("content_hash", 0),
                    "created_at": n.get("created_at"),
                    "activation_level": n.get("activation_level", 0.0),
                    "access_frequency": n.get("access_frequency", 0),
                    "last_activated": n.get("last_activated"),
                    "decay_rate": n.get("decay_rate", 0.1),
                    "firing_threshold": n.get("firing_threshold", 0.3),
                    "refractory_until": n.get("refractory_until"),
                    "refractory_period_ms": n.get("refractory_period_ms", 500.0),
                    "homeostatic_target": n.get("homeostatic_target", 0.5),
                },
                brain_id=bid,
            )

        # Import synapses
        for s in snapshot.synapses:
            metadata = s.get("metadata", "{}")
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata, default=str)
            await self._query(
                """
                MATCH (a:Neuron {id: $src}), (b:Neuron {id: $tgt})
                CREATE (a)-[:SYNAPSE {
                    id: $id, type: $type, weight: $weight,
                    direction: $direction, metadata: $metadata,
                    reinforced_count: $reinforced_count,
                    last_activated: $last_activated,
                    created_at: $created_at
                }]->(b)
                """,
                {
                    "id": s["id"],
                    "src": s["source_id"],
                    "tgt": s["target_id"],
                    "type": s.get("type", "related_to"),
                    "weight": s.get("weight", 0.5),
                    "direction": s.get("direction", "uni"),
                    "metadata": metadata,
                    "reinforced_count": s.get("reinforced_count", 0),
                    "last_activated": s.get("last_activated"),
                    "created_at": s.get("created_at"),
                },
                brain_id=bid,
            )

        # Import fibers (simplified - no [:CONTAINS] edges for now)
        for f in snapshot.fibers:
            metadata = f.get("metadata", "{}")
            if isinstance(metadata, dict):
                metadata = json.dumps(metadata, default=str)
            pathway = f.get("pathway", "[]")
            if isinstance(pathway, list):
                pathway = json.dumps(pathway)
            neuron_ids = f.get("neuron_ids", "[]")
            if isinstance(neuron_ids, (set, list)):
                neuron_ids = json.dumps(sorted(neuron_ids))
            synapse_ids = f.get("synapse_ids", "[]")
            if isinstance(synapse_ids, (set, list)):
                synapse_ids = json.dumps(sorted(synapse_ids))

            auto_tags = f.get("auto_tags", "")
            if isinstance(auto_tags, (set, list)):
                auto_tags = ",".join(sorted(auto_tags))
            agent_tags = f.get("agent_tags", "")
            if isinstance(agent_tags, (set, list)):
                agent_tags = ",".join(sorted(agent_tags))

            await self._query(
                """
                CREATE (fb:Fiber {
                    id: $id, anchor_neuron_id: $anchor,
                    pathway: $pathway, conductivity: $conductivity,
                    last_conducted: $last_conducted,
                    time_start: $time_start, time_end: $time_end,
                    coherence: $coherence, salience: $salience,
                    frequency: $frequency, summary: $summary,
                    auto_tags: $auto_tags, agent_tags: $agent_tags,
                    metadata: $metadata, compression_tier: $compression_tier,
                    created_at: $created_at,
                    neuron_ids: $neuron_ids, synapse_ids: $synapse_ids
                })
                """,
                {
                    "id": f["id"],
                    "anchor": f.get("anchor_neuron_id", ""),
                    "pathway": pathway,
                    "conductivity": f.get("conductivity", 1.0),
                    "last_conducted": f.get("last_conducted"),
                    "time_start": f.get("time_start"),
                    "time_end": f.get("time_end"),
                    "coherence": f.get("coherence", 0.0),
                    "salience": f.get("salience", 0.0),
                    "frequency": f.get("frequency", 0),
                    "summary": f.get("summary"),
                    "auto_tags": auto_tags,
                    "agent_tags": agent_tags,
                    "metadata": metadata,
                    "compression_tier": f.get("compression_tier", 0),
                    "created_at": f.get("created_at"),
                    "neuron_ids": neuron_ids,
                    "synapse_ids": synapse_ids,
                },
                brain_id=bid,
            )

        # Ensure indexes on the imported graph
        await self._ensure_indexes(brain_id=bid)

        # Save brain metadata
        config = BrainConfig(**snapshot.config) if snapshot.config else BrainConfig()
        brain = Brain(
            id=bid,
            name=snapshot.brain_name,
            config=config,
            neuron_count=len(snapshot.neurons),
            synapse_count=len(snapshot.synapses),
            fiber_count=len(snapshot.fibers),
            metadata=snapshot.metadata,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        await self.save_brain(brain)

        return bid

    async def get_stats(self, brain_id: str) -> dict[str, int]:
        neuron_rows = await self._query_ro(
            "MATCH (n:Neuron) RETURN count(n)",
            brain_id=brain_id,
        )
        synapse_rows = await self._query_ro(
            "MATCH ()-[s:SYNAPSE]->() RETURN count(s)",
            brain_id=brain_id,
        )
        fiber_rows = await self._query_ro(
            "MATCH (f:Fiber) RETURN count(f)",
            brain_id=brain_id,
        )
        return {
            "neuron_count": neuron_rows[0][0] if neuron_rows else 0,
            "synapse_count": synapse_rows[0][0] if synapse_rows else 0,
            "fiber_count": fiber_rows[0][0] if fiber_rows else 0,
        }

    async def get_enhanced_stats(self, brain_id: str) -> dict[str, Any]:
        basic = await self.get_stats(brain_id)

        # Hot neurons (top 10 by activation)
        hot_rows = await self._query_ro(
            """
            MATCH (n:Neuron)
            WHERE n.activation_level > 0
            RETURN n.id, n.content, n.activation_level, n.access_frequency
            ORDER BY n.activation_level DESC
            LIMIT 10
            """,
            brain_id=brain_id,
        )
        hot_neurons = [
            {
                "id": r[0], "content": r[1],
                "activation_level": r[2], "access_frequency": r[3],
            }
            for r in hot_rows
        ]

        # Neuron type breakdown
        type_rows = await self._query_ro(
            """
            MATCH (n:Neuron)
            RETURN n.type, count(n) AS cnt
            ORDER BY cnt DESC
            """,
            brain_id=brain_id,
        )
        type_breakdown = {r[0]: r[1] for r in type_rows}

        return {
            **basic,
            "hot_neurons": hot_neurons,
            "neuron_type_breakdown": type_breakdown,
            "storage_backend": "falkordb",
        }

    async def clear(self, brain_id: str) -> None:
        """Clear all data for a brain graph."""
        await self._query(
            "MATCH (n) DETACH DELETE n",
            brain_id=brain_id,
        )

    # ========== Row Mapper ==========

    def _row_to_brain(self, row: list[Any]) -> Brain:
        """Convert registry row to Brain dataclass."""
        config_raw = row[2]
        if isinstance(config_raw, str):
            config_dict = json.loads(config_raw) if config_raw else {}
        else:
            config_dict = config_raw or {}

        # Filter valid BrainConfig fields
        valid_fields = {f.name for f in fields(BrainConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

        shared_raw = row[5]
        if isinstance(shared_raw, str):
            shared = json.loads(shared_raw) if shared_raw else []
        else:
            shared = list(shared_raw) if shared_raw else []

        return Brain(
            id=row[0],
            name=row[1],
            config=BrainConfig(**filtered_config) if filtered_config else BrainConfig(),
            owner_id=row[3],
            is_public=bool(row[4]) if row[4] is not None else False,
            shared_with=shared,
            neuron_count=row[6] if row[6] is not None else 0,
            synapse_count=row[7] if row[7] is not None else 0,
            fiber_count=row[8] if row[8] is not None else 0,
            metadata=self._deserialize_metadata(row[9]),
            created_at=self._str_to_dt(row[10]) or datetime.min,
            updated_at=self._str_to_dt(row[11]) or datetime.min,
        )
