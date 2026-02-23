"""FalkorDB graph traversal operations (HOTPATH for spreading activation)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.falkordb.falkordb_base import FalkorDBBaseMixin


class FalkorDBGraphMixin(FalkorDBBaseMixin):
    """FalkorDB graph traversal operations.

    These are the HOTPATH methods called thousands of times during
    spreading activation. OpenCypher MATCH replaces SQL JOINs.
    """

    async def get_neighbors(
        self,
        neuron_id: str,
        direction: Literal["out", "in", "both"] = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
    ) -> list[tuple[Neuron, Synapse]]:
        """Get neighboring neurons connected by synapses.

        This is the critical hotpath for spreading activation.
        FalkorDB's GraphBLAS adjacency lookup is O(k) vs SQL O(n).

        Returns:
            List of (neighbor_neuron, connecting_synapse) tuples.
        """
        results: list[tuple[Neuron, Synapse]] = []

        # Outgoing: (neuron)-[s]->(neighbor)
        if direction in ("out", "both"):
            out_rows = await self._query_neighbors_out(neuron_id, synapse_types, min_weight)
            for row in out_rows:
                neighbor = self._neighbor_row_to_neuron(row, offset=0)
                syn = self._neighbor_row_to_synapse(
                    row, source_id=neuron_id, target_id=neighbor.id, offset=6
                )
                results.append((neighbor, syn))

        # Incoming: (neighbor)-[s]->(neuron)
        if direction in ("in", "both"):
            in_rows = await self._query_neighbors_in(neuron_id, synapse_types, min_weight)
            for row in in_rows:
                neighbor = self._neighbor_row_to_neuron(row, offset=0)
                syn = self._neighbor_row_to_synapse(
                    row, source_id=neighbor.id, target_id=neuron_id, offset=6
                )
                results.append((neighbor, syn))

        # Bidirectional synapses: also check reverse direction edges
        # that have direction='bi' (stored as directed but traversable both ways)
        if direction in ("out", "both"):
            bi_rows = await self._query_neighbors_bi_reverse(neuron_id, synapse_types, min_weight)
            seen_ids = {s.id for _, s in results}
            for row in bi_rows:
                neighbor = self._neighbor_row_to_neuron(row, offset=0)
                syn = self._neighbor_row_to_synapse(
                    row, source_id=neighbor.id, target_id=neuron_id, offset=6
                )
                if syn.id not in seen_ids:
                    results.append((neighbor, syn))
                    seen_ids.add(syn.id)

        return results

    async def _query_neighbors_out(
        self,
        neuron_id: str,
        synapse_types: list[SynapseType] | None,
        min_weight: float | None,
    ) -> list[list[Any]]:
        """Query outgoing neighbors."""
        conditions = ["a.id = $nid"]
        params: dict[str, Any] = {"nid": neuron_id}

        if synapse_types:
            conditions.append("s.type IN $stypes")
            params["stypes"] = [st.value for st in synapse_types]
        if min_weight is not None:
            conditions.append("s.weight >= $min_w")
            params["min_w"] = min_weight

        where = " AND ".join(conditions)
        return await self._query_ro(
            f"""
            MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
            WHERE {where}
            RETURN b.id, b.type, b.content, b.metadata,
                   b.content_hash, b.created_at,
                   s.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count,
                   s.last_activated, s.created_at
            """,
            params,
        )

    async def _query_neighbors_in(
        self,
        neuron_id: str,
        synapse_types: list[SynapseType] | None,
        min_weight: float | None,
    ) -> list[list[Any]]:
        """Query incoming neighbors."""
        conditions = ["b.id = $nid"]
        params: dict[str, Any] = {"nid": neuron_id}

        if synapse_types:
            conditions.append("s.type IN $stypes")
            params["stypes"] = [st.value for st in synapse_types]
        if min_weight is not None:
            conditions.append("s.weight >= $min_w")
            params["min_w"] = min_weight

        where = " AND ".join(conditions)
        return await self._query_ro(
            f"""
            MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
            WHERE {where}
            RETURN a.id, a.type, a.content, a.metadata,
                   a.content_hash, a.created_at,
                   s.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count,
                   s.last_activated, s.created_at
            """,
            params,
        )

    async def _query_neighbors_bi_reverse(
        self,
        neuron_id: str,
        synapse_types: list[SynapseType] | None,
        min_weight: float | None,
    ) -> list[list[Any]]:
        """Query bidirectional synapses where neuron is the target.

        These are edges stored as (other)-[s {direction:'bi'}]->(neuron)
        that should also be traversable as outgoing from neuron's perspective.
        """
        conditions = ["b.id = $nid", "s.direction = $bi_dir"]
        params: dict[str, Any] = {"nid": neuron_id, "bi_dir": Direction.BIDIRECTIONAL.value}

        if synapse_types:
            conditions.append("s.type IN $stypes")
            params["stypes"] = [st.value for st in synapse_types]
        if min_weight is not None:
            conditions.append("s.weight >= $min_w")
            params["min_w"] = min_weight

        where = " AND ".join(conditions)
        return await self._query_ro(
            f"""
            MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
            WHERE {where}
            RETURN a.id, a.type, a.content, a.metadata,
                   a.content_hash, a.created_at,
                   s.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count,
                   s.last_activated, s.created_at
            """,
            params,
        )

    async def get_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> list[tuple[Neuron, Synapse]] | None:
        """Find shortest path between two neurons.

        Uses FalkorDB's native shortestPath() for optimal performance.
        Max hops is capped at 10 to prevent runaway traversals.

        Returns:
            List of (neuron, synapse) pairs along the path, or None.
        """
        max_hops = min(max_hops, 10)

        rows = await self._query_ro(
            f"""
            MATCH (a:Neuron {{id: $src}}), (b:Neuron {{id: $tgt}})
            MATCH path = shortestPath((a)-[:SYNAPSE*1..{max_hops}]-(b))
            RETURN nodes(path) AS path_nodes,
                   relationships(path) AS path_rels
            """,
            {"src": source_id, "tgt": target_id},
        )

        if not rows:
            return None

        path_nodes = rows[0][0]
        path_rels = rows[0][1]

        if not path_nodes or not path_rels:
            return None

        result: list[tuple[Neuron, Synapse]] = []
        for i, rel in enumerate(path_rels):
            # Path nodes: [start, ..., end] with len = len(rels) + 1
            # Each step: node[i+1] is the next neuron, rel[i] is the edge
            if i + 1 < len(path_nodes):
                node = path_nodes[i + 1]
                neuron = self._graph_entity_to_neuron(node)
                synapse = self._graph_entity_to_synapse(rel, path_nodes, i)
                result.append((neuron, synapse))

        return result if result else None

    # ========== Helper converters ==========

    def _neighbor_row_to_neuron(self, row: list[Any], offset: int = 0) -> Neuron:
        """Extract Neuron from a neighbor query result row."""
        return Neuron(
            id=row[offset],
            type=NeuronType(row[offset + 1]),
            content=row[offset + 2],
            metadata=self._deserialize_metadata(row[offset + 3]),
            content_hash=row[offset + 4] if row[offset + 4] is not None else 0,
            created_at=self._str_to_dt(row[offset + 5]) or datetime.min,
        )

    def _neighbor_row_to_synapse(
        self,
        row: list[Any],
        source_id: str,
        target_id: str,
        offset: int = 6,
    ) -> Synapse:
        """Extract Synapse from a neighbor query result row."""
        return Synapse(
            id=row[offset],
            source_id=source_id,
            target_id=target_id,
            type=SynapseType(row[offset + 1]),
            weight=row[offset + 2] if row[offset + 2] is not None else 0.5,
            direction=(Direction(row[offset + 3]) if row[offset + 3] else Direction.UNIDIRECTIONAL),
            metadata=self._deserialize_metadata(row[offset + 4]),
            reinforced_count=row[offset + 5] if row[offset + 5] is not None else 0,
            last_activated=self._str_to_dt(row[offset + 6]),
            created_at=self._str_to_dt(row[offset + 7]) or datetime.min,
        )

    def _graph_entity_to_neuron(self, node: Any) -> Neuron:
        """Convert a FalkorDB graph node entity to Neuron.

        Graph entities from nodes(path) have .properties dict access.
        """
        props = node.properties if hasattr(node, "properties") else node
        if isinstance(props, dict):
            return Neuron(
                id=props.get("id", ""),
                type=NeuronType(props.get("type", "concept")),
                content=props.get("content", ""),
                metadata=self._deserialize_metadata(props.get("metadata")),
                content_hash=props.get("content_hash", 0),
                created_at=self._str_to_dt(props.get("created_at")) or datetime.min,
            )
        # Fallback: treat as list (positional)
        return self._neighbor_row_to_neuron(list(props), offset=0)

    def _graph_entity_to_synapse(self, rel: Any, path_nodes: list[Any], rel_index: int) -> Synapse:
        """Convert a FalkorDB graph relationship entity to Synapse.

        Graph entities from relationships(path) have .properties dict access.
        """
        props = rel.properties if hasattr(rel, "properties") else rel

        # Determine source/target from path node order
        src_node = path_nodes[rel_index]
        tgt_node = path_nodes[rel_index + 1]
        src_props = src_node.properties if hasattr(src_node, "properties") else src_node
        tgt_props = tgt_node.properties if hasattr(tgt_node, "properties") else tgt_node

        src_id = src_props.get("id", "") if isinstance(src_props, dict) else ""
        tgt_id = tgt_props.get("id", "") if isinstance(tgt_props, dict) else ""

        if isinstance(props, dict):
            return Synapse(
                id=props.get("id", ""),
                source_id=src_id,
                target_id=tgt_id,
                type=SynapseType(props.get("type", "related_to")),
                weight=props.get("weight", 0.5),
                direction=(
                    Direction(props["direction"])
                    if props.get("direction")
                    else Direction.UNIDIRECTIONAL
                ),
                metadata=self._deserialize_metadata(props.get("metadata")),
                reinforced_count=props.get("reinforced_count", 0),
                last_activated=self._str_to_dt(props.get("last_activated")),
                created_at=(self._str_to_dt(props.get("created_at")) or datetime.min),
            )
        return Synapse(
            id="",
            source_id=src_id,
            target_id=tgt_id,
            type=SynapseType.RELATED_TO,
            weight=0.5,
        )
