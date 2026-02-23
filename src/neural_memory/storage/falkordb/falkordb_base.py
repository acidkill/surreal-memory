"""FalkorDB base connection and query infrastructure."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB, Graph

logger = logging.getLogger(__name__)

# Graph schema version for FalkorDB backend
FALKORDB_SCHEMA_VERSION = 1


class FalkorDBBaseMixin:
    """Base mixin providing FalkorDB connection management and query helpers.

    Concrete storage class must set:
        _db: FalkorDB async client instance
        _current_brain_id: str | None
        _graphs: dict[str, Graph]  (cache of selected graphs)
    """

    _db: FalkorDB | None
    _current_brain_id: str | None
    _graphs: dict[str, Any]

    def _get_brain_id(self) -> str:
        if self._current_brain_id is None:
            raise ValueError("No brain context set. Call set_brain() first.")
        return self._current_brain_id

    def _ensure_db(self) -> FalkorDB:
        if self._db is None:
            raise RuntimeError("FalkorDB not initialized. Call initialize() first.")
        return self._db

    async def _get_graph(self, brain_id: str | None = None) -> Graph:
        """Get or create a FalkorDB graph for the given brain.

        Each brain maps to a separate FalkorDB graph for native isolation.
        """
        bid = brain_id or self._get_brain_id()
        graph_name = f"brain_{bid}"

        if graph_name not in self._graphs:
            db = self._ensure_db()
            self._graphs[graph_name] = db.select_graph(graph_name)

        return self._graphs[graph_name]

    async def _query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        brain_id: str | None = None,
    ) -> list[list[Any]]:
        """Execute a Cypher query and return result_set rows.

        Args:
            cypher: OpenCypher query string with $param placeholders
            params: Query parameters (safe from injection)
            brain_id: Override brain context (default: current brain)

        Returns:
            List of result rows (each row is a list of values)
        """
        graph = await self._get_graph(brain_id)
        result = await graph.query(cypher, params=params or {})
        return result.result_set if result.result_set else []

    async def _query_ro(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        brain_id: str | None = None,
    ) -> list[list[Any]]:
        """Execute a read-only Cypher query.

        Uses ro_query for potential read replica routing.
        """
        graph = await self._get_graph(brain_id)
        try:
            result = await graph.ro_query(cypher, params=params or {})
        except AttributeError:
            result = await graph.query(cypher, params=params or {})
        return result.result_set if result.result_set else []

    async def _ensure_indexes(self, brain_id: str | None = None) -> None:
        """Create indexes if they don't exist for a brain graph."""
        graph = await self._get_graph(brain_id)

        index_queries = [
            # Neuron range indexes
            "CREATE INDEX IF NOT EXISTS FOR (n:Neuron) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Neuron) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Neuron) ON (n.content_hash)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Neuron) ON (n.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Neuron) ON (n.activation_level)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Neuron) ON (n.access_frequency)",
            # Fiber range indexes
            "CREATE INDEX IF NOT EXISTS FOR (f:Fiber) ON (f.id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Fiber) ON (f.anchor_neuron_id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Fiber) ON (f.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Fiber) ON (f.salience)",
        ]

        for q in index_queries:
            try:
                await graph.query(q)
            except Exception:
                logger.warning("Index creation failed: %s", q, exc_info=True)

        # Fulltext index on Neuron.content (RediSearch)
        try:
            await graph.query(
                "CALL db.idx.fulltext.createNodeIndex('Neuron', 'content')"
            )
        except Exception:
            logger.debug("Fulltext index on Neuron.content may already exist", exc_info=True)

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> str:
        """Serialize metadata dict to JSON string for graph storage."""
        return json.dumps(metadata, default=str) if metadata else "{}"

    @staticmethod
    def _deserialize_metadata(raw: str | None) -> dict[str, Any]:
        """Deserialize metadata JSON string from graph storage."""
        if not raw:
            return {}
        try:
            result: dict[str, Any] = json.loads(raw)
            return result
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _dt_to_str(dt: Any | None) -> str | None:
        """Convert datetime to ISO string for storage, or None."""
        if dt is None:
            return None
        if hasattr(dt, "isoformat"):
            return str(dt.isoformat())
        return str(dt)

    @staticmethod
    def _str_to_dt(s: str | None) -> datetime | None:
        """Parse ISO datetime string, or None."""
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None
