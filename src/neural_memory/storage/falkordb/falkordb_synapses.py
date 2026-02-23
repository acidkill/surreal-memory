"""FalkorDB synapse CRUD operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from neural_memory.core.synapse import Direction, Synapse, SynapseType
from neural_memory.storage.falkordb.falkordb_base import FalkorDBBaseMixin


class FalkorDBSynapseMixin(FalkorDBBaseMixin):
    """FalkorDB implementation of synapse CRUD operations.

    Synapses are stored as directed edges [:SYNAPSE] between :Neuron nodes.
    Bidirectional synapses are stored as a single directed edge with
    direction='bi'; the graph traversal layer handles both directions.
    """

    async def add_synapse(self, synapse: Synapse) -> str:
        # Check for duplicate
        existing = await self._query_ro(
            """
            MATCH (:Neuron)-[s:SYNAPSE {id: $id}]->(:Neuron)
            RETURN s.id
            """,
            {"id": synapse.id},
        )
        if existing:
            raise ValueError(f"Synapse {synapse.id} already exists")

        # Verify both neurons exist
        neuron_check = await self._query_ro(
            """
            MATCH (a:Neuron {id: $src}), (b:Neuron {id: $tgt})
            RETURN a.id, b.id
            """,
            {"src": synapse.source_id, "tgt": synapse.target_id},
        )
        if not neuron_check:
            raise ValueError(
                f"Source neuron {synapse.source_id} or target neuron "
                f"{synapse.target_id} not found"
            )

        await self._query(
            """
            MATCH (a:Neuron {id: $source_id}), (b:Neuron {id: $target_id})
            CREATE (a)-[:SYNAPSE {
                id: $id,
                type: $type,
                weight: $weight,
                direction: $direction,
                metadata: $metadata,
                reinforced_count: $reinforced_count,
                last_activated: $last_activated,
                created_at: $created_at
            }]->(b)
            """,
            {
                "id": synapse.id,
                "source_id": synapse.source_id,
                "target_id": synapse.target_id,
                "type": synapse.type.value,
                "weight": synapse.weight,
                "direction": synapse.direction.value,
                "metadata": self._serialize_metadata(synapse.metadata),
                "reinforced_count": synapse.reinforced_count,
                "last_activated": self._dt_to_str(synapse.last_activated),
                "created_at": self._dt_to_str(synapse.created_at),
            },
        )
        return synapse.id

    async def get_synapse(self, synapse_id: str) -> Synapse | None:
        rows = await self._query_ro(
            """
            MATCH (a:Neuron)-[s:SYNAPSE {id: $id}]->(b:Neuron)
            RETURN s.id, a.id, b.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count, s.last_activated,
                   s.created_at
            """,
            {"id": synapse_id},
        )
        if not rows:
            return None
        return self._row_to_synapse(rows[0])

    async def get_synapses(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        type: SynapseType | None = None,
        min_weight: float | None = None,
    ) -> list[Synapse]:
        conditions: list[str] = []
        params: dict[str, Any] = {}

        if source_id is not None:
            conditions.append("a.id = $source_id")
            params["source_id"] = source_id
        if target_id is not None:
            conditions.append("b.id = $target_id")
            params["target_id"] = target_id
        if type is not None:
            conditions.append("s.type = $syn_type")
            params["syn_type"] = type.value
        if min_weight is not None:
            conditions.append("s.weight >= $min_weight")
            params["min_weight"] = min_weight

        where = " AND ".join(conditions) if conditions else "true"
        rows = await self._query_ro(
            f"""
            MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
            WHERE {where}
            RETURN s.id, a.id, b.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count, s.last_activated,
                   s.created_at
            """,
            params,
        )
        return [self._row_to_synapse(r) for r in rows]

    async def update_synapse(self, synapse: Synapse) -> None:
        rows = await self._query(
            """
            MATCH (a:Neuron)-[s:SYNAPSE {id: $id}]->(b:Neuron)
            SET s.type = $type,
                s.weight = $weight,
                s.direction = $direction,
                s.metadata = $metadata,
                s.reinforced_count = $reinforced_count,
                s.last_activated = $last_activated
            RETURN s.id
            """,
            {
                "id": synapse.id,
                "type": synapse.type.value,
                "weight": synapse.weight,
                "direction": synapse.direction.value,
                "metadata": self._serialize_metadata(synapse.metadata),
                "reinforced_count": synapse.reinforced_count,
                "last_activated": self._dt_to_str(synapse.last_activated),
            },
        )
        if not rows:
            raise ValueError(f"Synapse {synapse.id} not found")

    async def delete_synapse(self, synapse_id: str) -> bool:
        existing = await self._query_ro(
            """
            MATCH (:Neuron)-[s:SYNAPSE {id: $id}]->(:Neuron)
            RETURN s.id
            """,
            {"id": synapse_id},
        )
        if not existing:
            return False
        await self._query(
            """
            MATCH (:Neuron)-[s:SYNAPSE {id: $id}]->(:Neuron)
            DELETE s
            """,
            {"id": synapse_id},
        )
        return True

    async def get_all_synapses(self) -> list[Synapse]:
        rows = await self._query_ro(
            """
            MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
            RETURN s.id, a.id, b.id, s.type, s.weight, s.direction,
                   s.metadata, s.reinforced_count, s.last_activated,
                   s.created_at
            """
        )
        return [self._row_to_synapse(r) for r in rows]

    async def get_synapses_for_neurons(
        self,
        neuron_ids: list[str],
        direction: str = "out",
    ) -> dict[str, list[Synapse]]:
        if not neuron_ids:
            return {}

        if direction == "out":
            rows = await self._query_ro(
                """
                MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
                WHERE a.id IN $ids
                RETURN s.id, a.id, b.id, s.type, s.weight, s.direction,
                       s.metadata, s.reinforced_count, s.last_activated,
                       s.created_at
                """,
                {"ids": neuron_ids},
            )
        else:
            rows = await self._query_ro(
                """
                MATCH (a:Neuron)-[s:SYNAPSE]->(b:Neuron)
                WHERE b.id IN $ids
                RETURN s.id, a.id, b.id, s.type, s.weight, s.direction,
                       s.metadata, s.reinforced_count, s.last_activated,
                       s.created_at
                """,
                {"ids": neuron_ids},
            )

        result: dict[str, list[Synapse]] = {nid: [] for nid in neuron_ids}
        for row in rows:
            syn = self._row_to_synapse(row)
            key = syn.source_id if direction == "out" else syn.target_id
            if key in result:
                result[key].append(syn)
        return result

    # ========== Row Mapper ==========

    def _row_to_synapse(self, row: list[Any]) -> Synapse:
        """Convert FalkorDB result row to Synapse dataclass.

        Expected columns: id, source_id, target_id, type, weight, direction,
                         metadata, reinforced_count, last_activated, created_at
        """
        return Synapse(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            type=SynapseType(row[3]),
            weight=row[4] if row[4] is not None else 0.5,
            direction=Direction(row[5]) if row[5] else Direction.UNIDIRECTIONAL,
            metadata=self._deserialize_metadata(row[6]),
            reinforced_count=row[7] if row[7] is not None else 0,
            last_activated=self._str_to_dt(row[8]),
            created_at=self._str_to_dt(row[9]) or datetime.min,
        )
