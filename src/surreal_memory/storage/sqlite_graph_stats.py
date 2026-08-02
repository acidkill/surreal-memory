"""SQLite mixin for graph statistics.

This module is what remains of ``sqlite_calibration``. The retrieval-calibration
EMA it used to hold — per-gate accuracy stats (``save_calibration_record``,
``get_recent_calibration``, ``get_gate_ema_stats``) and per-brain RRF retriever
weights (``save_retriever_outcome``, ``get_retriever_weights``) — was removed
along with its call sites. It existed only on ``SQLiteStorage`` and every caller
swallowed the resulting ``AttributeError``, so on SurrealDB, the production
backend since 2.0.0, it never recorded a single sample or influenced a single
retrieval. Rather than build it a second time on a backend that has done without
it since day one, the feature is gone: sufficiency gates use their documented
thresholds and RRF uses ``DEFAULT_RETRIEVER_WEIGHTS``, which is exactly what
every SurrealDB brain was already doing.

``get_graph_density`` stays because it is not a statistic — it selects the
activation strategy for ``activation_strategy="auto"``, and returning nothing
pinned that setting to classic BFS.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteGraphStatsMixin:
    """Mixin providing graph-shape statistics for the current brain."""

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def get_graph_density(self) -> float:
        """Compute average synapses per neuron for the current brain.

        Returns 0.0 if no neurons exist.
        Used by retrieval engine to auto-select activation strategy.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM neurons WHERE brain_id = ?",
            (brain_id,),
        )
        row = await cursor.fetchone()
        neuron_count = row[0] if row else 0
        if neuron_count == 0:
            return 0.0

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM synapses WHERE brain_id = ?",
            (brain_id,),
        )
        row = await cursor.fetchone()
        synapse_count = row[0] if row else 0

        return synapse_count / neuron_count
