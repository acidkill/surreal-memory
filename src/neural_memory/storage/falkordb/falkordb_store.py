"""FalkorDB composite storage backend for Neural Memory.

Composes FalkorDB graph mixins for core neuron/synapse/fiber/brain
operations. Non-graph relational features (typed_memories, reviews,
versions, etc.) fall back to base class NotImplementedError defaults
and can be enabled via SQLite sidecar in a future phase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.storage.base import NeuralStorage
from neural_memory.storage.falkordb.falkordb_brains import FalkorDBBrainMixin
from neural_memory.storage.falkordb.falkordb_fibers import FalkorDBFiberMixin
from neural_memory.storage.falkordb.falkordb_graph import FalkorDBGraphMixin
from neural_memory.storage.falkordb.falkordb_neurons import FalkorDBNeuronMixin
from neural_memory.storage.falkordb.falkordb_synapses import FalkorDBSynapseMixin

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB

logger = logging.getLogger(__name__)


class FalkorDBStorage(
    FalkorDBNeuronMixin,
    FalkorDBSynapseMixin,
    FalkorDBFiberMixin,
    FalkorDBGraphMixin,
    FalkorDBBrainMixin,
    NeuralStorage,
):
    """FalkorDB-backed storage for Neural Memory.

    Each brain maps to an isolated FalkorDB graph (brain_{id}).
    Graph-native traversal powers spreading activation hotpath.

    Usage:
        storage = FalkorDBStorage(host="localhost", port=6379)
        await storage.initialize()
        storage.set_brain("my-brain")
        await storage.add_neuron(neuron)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password

        # Base mixin state (used by FalkorDBBaseMixin)
        self._db: FalkorDB | None = None
        self._current_brain_id: str | None = None
        self._graphs: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize FalkorDB connection."""
        from falkordb.asyncio import FalkorDB as AsyncFalkorDB
        from redis.asyncio import BlockingConnectionPool

        pool = BlockingConnectionPool(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            decode_responses=True,
            max_connections=16,
            socket_timeout=10.0,
            socket_connect_timeout=5.0,
        )
        self._db = AsyncFalkorDB(connection_pool=pool)
        logger.info("FalkorDB connected: %s:%d", self._host, self._port)

    async def close(self) -> None:
        """Close FalkorDB connection pool."""
        if self._db is not None:
            try:
                pool = self._db.connection
                if hasattr(pool, "disconnect"):
                    await pool.disconnect()
                elif hasattr(pool, "aclose"):
                    await pool.aclose()
            except Exception:
                logger.warning("Error closing FalkorDB pool", exc_info=True)
            self._db = None
        self._graphs.clear()

    @property
    def current_brain_id(self) -> str | None:
        """The active brain ID, or None if not set."""
        return self._current_brain_id

    def set_brain(self, brain_id: str) -> None:
        """Set the active brain context.

        This determines which FalkorDB graph is used for queries.
        """
        self._current_brain_id = brain_id

    async def set_brain_with_indexes(self, brain_id: str) -> None:
        """Set brain context and ensure graph indexes exist."""
        self._current_brain_id = brain_id
        await self._ensure_indexes(brain_id)
        await self._ensure_registry()

    # ========== Batch Operations (no-op for FalkorDB) ==========

    def disable_auto_save(self) -> None:
        """No-op for FalkorDB (immediate writes)."""

    def enable_auto_save(self) -> None:
        """No-op for FalkorDB (immediate writes)."""

    async def batch_save(self) -> None:
        """No-op for FalkorDB (immediate writes)."""
