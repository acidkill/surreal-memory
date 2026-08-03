"""Regression test for the dashboard <-> CLI metric divergence (no live DB).

SurrealDB ``get_enhanced_stats`` must return a ``synapse_stats`` block with
per-type counts. Without it, ``DiagnosticsEngine`` computed ``diversity = 0``
and ``recall_confidence = 0`` on the SurrealDB backend (while the SQLite
backend computed real values), so the dashboard and the ``smem`` CLI reported
different grades for the very same brain — the "Grade F vs D" divergence.

It must also return the same five fields ``InMemoryStorage.get_enhanced_stats``
computes (``today_fibers_count``, ``newest_memory``, ``oldest_memory``,
``hot_neurons``, ``neuron_type_breakdown``) under the *same* key names its
callers (``cli/commands/info.py``, ``server/routes/brain.py``) actually read —
on SurrealDB, the only backend that ships, those five were previously always
0/null/empty regardless of the data present.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _make_store_with_scripted_query() -> SurrealDBStorage:
    """A SurrealDBStorage whose ``_query`` is scripted (no real connection)."""
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"

    # Keyed by the underscore form: get_neurons_batch runs each id through
    # _to_surreal_id (dash -> underscore) before it reaches "FROM neuron:{sid}".
    neurons_by_id = {
        "hot_1": {"id": "hot_1", "type": "entity", "content": "most accessed"},
        "hot_2": {"id": "hot_2", "type": "concept", "content": "second most accessed"},
    }

    async def fake_query(sql: str, **_params: Any) -> list[dict[str, Any]]:
        s = sql.lower()
        if "from fiber" in s and "created_at >=" in s and "group all" in s:
            return [{"c": 7}]
        if "from neuron" in s and "count()" in s and "group all" in s:
            return [{"c": 100}]
        if "from synapse" in s and "count()" in s and "group all" in s:
            return [{"c": 300}]
        if "from fiber" in s and "count()" in s and "group all" in s:
            return [{"c": 40}]
        if "from neuron" in s and "group by type" in s:
            return [{"type": "memory", "c": 100}]
        if "from synapse" in s and "group by type" in s:
            return [
                {"type": "related_to", "cnt": 150, "avg_w": 0.5, "total_r": 0},
                {"type": "after", "cnt": 100, "avg_w": 0.3, "total_r": 2},
                {"type": "co_occurs", "cnt": 50, "avg_w": 0.9, "total_r": 1},
            ]
        if "from neuron_state" in s and "order by access_frequency desc" in s:
            return [
                {"neuron_id": "hot-1", "activation_level": 0.9, "access_frequency": 42},
                {"neuron_id": "hot-2", "activation_level": 0.4, "access_frequency": 17},
            ]
        if "from neuron:" in s:
            m = re.search(r"from neuron:(\S+)", s)
            nid = m.group(1) if m else ""
            row = neurons_by_id.get(nid)
            return [row] if row else []
        if "from fiber" in s and "order by created_at asc" in s:
            return [{"created_at": "2026-07-01T00:00:00+00:00"}]
        if "from fiber" in s and "order by created_at desc" in s:
            return [{"created_at": "2026-08-03T09:57:30.954674+00:00"}]
        return []

    store._query = fake_query  # type: ignore[assignment,method-assign]
    return store


def test_enhanced_stats_includes_synapse_stats_by_type() -> None:
    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))

    assert "synapse_stats" in stats, "synapse_stats missing -> diversity would be 0"
    by_type = stats["synapse_stats"]["by_type"]
    assert set(by_type) == {"related_to", "after", "co_occurs"}
    # DiagnosticsEngine._compute_diversity reads entry["count"].
    assert by_type["related_to"]["count"] == 150
    assert by_type["after"]["total_reinforcements"] == 2
    assert stats["synapse_stats"]["avg_weight"] > 0


def test_enhanced_stats_synapse_stats_yields_nonzero_diversity() -> None:
    """The whole point: with synapse_stats present, diversity computes > 0."""
    from surreal_memory.engine.diagnostics import DiagnosticsEngine

    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))
    diversity = DiagnosticsEngine._compute_diversity(stats["synapse_stats"])
    assert diversity > 0.0


def test_enhanced_stats_reports_neuron_type_breakdown_under_the_key_callers_read() -> None:
    """The backend computed this under ``neuron_types``; every caller (CLI info,
    the dashboard's brain route) reads ``neuron_type_breakdown`` — the names
    never matched, so the ~2.6s query it took to build was discarded on arrival.
    """
    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))

    assert "neuron_type_breakdown" in stats
    assert stats["neuron_type_breakdown"] == {"memory": 100}
    assert "neuron_types" not in stats


def test_enhanced_stats_reports_today_fibers_count() -> None:
    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))

    assert stats["today_fibers_count"] == 7


def test_enhanced_stats_reports_hot_neurons_ordered_by_access_frequency() -> None:
    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))

    hot = stats["hot_neurons"]
    assert [h["neuron_id"] for h in hot] == ["hot-1", "hot-2"]
    assert hot[0]["access_frequency"] == 42
    assert hot[0]["content"] == "most accessed"
    assert hot[0]["type"] == "entity"
    assert hot[1]["access_frequency"] == 17


def test_enhanced_stats_reports_oldest_and_newest_memory() -> None:
    """``_parse_datetime`` strips tzinfo ("naive for consistency across the
    codebase" — store.py's own docstring), so the isoformat output carries no
    UTC offset even though the scripted rows do. Matches
    ``InMemoryStorage.get_enhanced_stats``, which formats its own naive
    ``utcnow()``-based timestamps the same way.
    """
    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))

    assert stats["oldest_memory"] == "2026-07-01T00:00:00"
    assert stats["newest_memory"] == "2026-08-03T09:57:30.954674"


def test_enhanced_stats_on_an_empty_brain_reports_none_not_a_crash() -> None:
    """No fibers at all -> oldest/newest must be None, hot_neurons empty — not
    an IndexError from indexing an empty query result.
    """
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "empty-brain"

    async def empty_query(sql: str, **_params: Any) -> list[dict[str, Any]]:
        s = sql.lower()
        if "count()" in s and "group all" in s:
            return [{"c": 0}]
        return []

    store._query = empty_query  # type: ignore[assignment,method-assign]
    stats = asyncio.run(store.get_enhanced_stats("empty-brain"))

    assert stats["oldest_memory"] is None
    assert stats["newest_memory"] is None
    assert stats["hot_neurons"] == []
    assert stats["today_fibers_count"] == 0
