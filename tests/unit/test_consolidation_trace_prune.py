"""U4: scheduled consolidation prunes old retrieval traces (TTL + max cap)."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationReport
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import TraceConfig, UnifiedConfig
from surreal_memory.utils.timeutils import utcnow


def _cfg(retention_days: int = 30) -> UnifiedConfig:
    cfg = UnifiedConfig()
    cfg.trace = TraceConfig(enabled=True, retention_days=retention_days, max_traces=5000)
    return cfg


class TestConsolidationTracePrune:
    async def test_prune_removes_old_traces(self) -> None:
        storage = InMemoryStorage()
        storage.set_brain("b")
        now = utcnow()
        old = RetrievalTrace(brain_id="b", query="old", created_at=now - timedelta(days=40))
        fresh = RetrievalTrace(brain_id="b", query="fresh", created_at=now)
        await storage.add_retrieval_trace(old)
        await storage.add_retrieval_trace(fresh)

        engine = ConsolidationEngine(storage)
        report = ConsolidationReport()
        with patch("surreal_memory.unified_config.get_config", return_value=_cfg(30)):
            await engine._prune(report, now, dry_run=False)

        assert report.retrieval_traces_pruned == 1
        remaining = {t.id for t in await storage.find_retrieval_traces(limit=20)}
        assert fresh.id in remaining
        assert old.id not in remaining

    async def test_dry_run_does_not_prune(self) -> None:
        storage = InMemoryStorage()
        storage.set_brain("b")
        now = utcnow()
        old = RetrievalTrace(brain_id="b", query="old", created_at=now - timedelta(days=40))
        await storage.add_retrieval_trace(old)

        engine = ConsolidationEngine(storage)
        report = ConsolidationReport()
        with patch("surreal_memory.unified_config.get_config", return_value=_cfg(30)):
            await engine._prune(report, now, dry_run=True)

        assert report.retrieval_traces_pruned == 0
        assert len(await storage.find_retrieval_traces(limit=20)) == 1
