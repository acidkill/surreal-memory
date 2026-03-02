"""SQLite mixin for retrieval sufficiency calibration persistence."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)

# Cap per brain to prevent unbounded growth
_MAX_RECORDS_PER_BRAIN = 10_000


class SQLiteCalibrationMixin:
    """Mixin providing CRUD for the retrieval_calibration table."""

    def _ensure_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _ensure_read_conn(self) -> aiosqlite.Connection:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def save_calibration_record(
        self,
        gate: str,
        predicted_sufficient: bool,
        actual_confidence: float,
        actual_fibers: int,
        query_intent: str = "",
        metrics_json: dict[str, Any] | None = None,
    ) -> None:
        """Insert a calibration feedback record."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        await conn.execute(
            """INSERT INTO retrieval_calibration
               (brain_id, gate, predicted_sufficient, actual_confidence,
                actual_fibers, query_intent, metrics_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                brain_id,
                gate,
                1 if predicted_sufficient else 0,
                actual_confidence,
                actual_fibers,
                query_intent,
                json.dumps(metrics_json or {}),
                utcnow().isoformat(),
            ),
        )
        await conn.commit()

    async def get_recent_calibration(
        self,
        gate: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch recent calibration records, optionally filtered by gate."""
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_limit = min(limit, 200)

        if gate:
            cursor = await conn.execute(
                """SELECT * FROM retrieval_calibration
                   WHERE brain_id = ? AND gate = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (brain_id, gate, capped_limit),
            )
        else:
            cursor = await conn.execute(
                """SELECT * FROM retrieval_calibration
                   WHERE brain_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (brain_id, capped_limit),
            )

        rows = await cursor.fetchall()
        col_names = [d[0] for d in (cursor.description or [])]
        return [dict(zip(col_names, r, strict=False)) for r in rows]

    async def prune_old_calibration(self, keep_days: int = 90) -> int:
        """Delete calibration records older than keep_days. Returns count deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        from datetime import timedelta

        cutoff = (utcnow() - timedelta(days=keep_days)).isoformat()

        cursor = await conn.execute(
            "DELETE FROM retrieval_calibration WHERE brain_id = ? AND created_at < ?",
            (brain_id, cutoff),
        )
        await conn.commit()
        return cursor.rowcount

    async def cap_calibration_records(self) -> int:
        """Enforce max record limit per brain. Returns count deleted."""
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()

        cursor = await conn.execute(
            "SELECT COUNT(*) FROM retrieval_calibration WHERE brain_id = ?",
            (brain_id,),
        )
        row = await cursor.fetchone()
        count = row[0] if row else 0

        if count <= _MAX_RECORDS_PER_BRAIN:
            return 0

        excess = count - _MAX_RECORDS_PER_BRAIN
        cursor = await conn.execute(
            """DELETE FROM retrieval_calibration WHERE id IN (
                SELECT id FROM retrieval_calibration
                WHERE brain_id = ?
                ORDER BY created_at ASC LIMIT ?
            )""",
            (brain_id, excess),
        )
        await conn.commit()
        return cursor.rowcount

    async def get_gate_ema_stats(
        self,
        window: int = 50,
    ) -> dict[str, dict[str, float]]:
        """Compute EMA accuracy stats per gate over recent records.

        Returns a dict keyed by gate name, each containing:
        - accuracy: EMA of correct predictions (predicted_sufficient matches
          actual_confidence >= 0.3 as true positive threshold)
        - avg_confidence: EMA of actual_confidence for that gate
        - sample_count: number of records used

        EMA decays older records toward the tail (most recent data weighted
        highest). Alpha = 2 / (window + 1) per standard EMA convention.
        """
        conn = self._ensure_read_conn()
        brain_id = self._get_brain_id()
        capped_window = min(window, 500)

        cursor = await conn.execute(
            """SELECT gate, predicted_sufficient, actual_confidence
               FROM retrieval_calibration
               WHERE brain_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (brain_id, capped_window * 20),  # fetch more to group by gate
        )
        rows = await cursor.fetchall()

        # Group rows by gate (rows are DESC order — most recent first)
        gate_rows: dict[str, list[tuple[int, float]]] = {}
        for gate, predicted, actual_conf in rows:
            if gate not in gate_rows:
                gate_rows[gate] = []
            gate_rows[gate].append((predicted, actual_conf))

        result: dict[str, dict[str, float]] = {}
        alpha = 2.0 / (capped_window + 1)

        for gate, gate_data in gate_rows.items():
            # Take at most capped_window records per gate (most recent first)
            gate_data = gate_data[:capped_window]
            sample_count = len(gate_data)

            if sample_count == 0:
                continue

            # Compute EMA on reversed list (oldest first for forward EMA)
            oldest_to_newest = list(reversed(gate_data))

            def _is_correct(predicted: int, actual_conf: float) -> float:
                """Return 1.0 if prediction matches actual outcome, else 0.0."""
                actual_sufficient = actual_conf >= 0.3
                return float(int(bool(predicted)) == int(actual_sufficient))

            first_predicted, first_conf = oldest_to_newest[0]
            ema_accuracy = _is_correct(first_predicted, first_conf)
            ema_confidence = first_conf

            for predicted, actual_conf in oldest_to_newest[1:]:
                correct = _is_correct(predicted, actual_conf)
                ema_accuracy = alpha * correct + (1.0 - alpha) * ema_accuracy
                ema_confidence = alpha * actual_conf + (1.0 - alpha) * ema_confidence

            result[gate] = {
                "accuracy": round(max(0.0, min(1.0, ema_accuracy)), 4),
                "avg_confidence": round(max(0.0, min(1.0, ema_confidence)), 4),
                "sample_count": float(sample_count),
            }

        return result
