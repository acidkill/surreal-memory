"""U5: build_uncertainty_block aggregates cheap recall-uncertainty signals."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from surreal_memory.engine.uncertainty_report import build_uncertainty_block


class _Config:
    sufficiency_threshold = 0.7


def _result(
    fibers: list[str], *, confidence: float = 0.9, disputed: list[str] | None = None
) -> Any:
    return SimpleNamespace(
        fibers_matched=fibers,
        confidence=confidence,
        metadata={"disputed_ids": disputed or []},
    )


def _storage(
    *,
    tms: dict[str, Any] | None = None,
    expiring: list[Any] | None = None,
    neurons: dict[str, Any] | None = None,
    drift: list[Any] | None = None,
) -> Any:
    ns = SimpleNamespace()
    ns.get_typed_memories_batch = AsyncMock(return_value=tms or {})
    ns.get_expiring_memories_for_fibers = AsyncMock(return_value=expiring or [])
    ns.get_neurons_batch = AsyncMock(return_value=neurons or {})
    if drift is not None:
        ns.get_drift_clusters = AsyncMock(return_value=drift)
    return ns


def _tm(fiber_id: str, *, superseded_by: str | None = None, tags: list[str] | None = None) -> Any:
    return SimpleNamespace(
        fiber_id=fiber_id, superseded_by=superseded_by, tags=tags or [], expires_at=None
    )


class TestBuildUncertaintyBlock:
    async def test_no_signal_returns_none(self) -> None:
        storage = _storage(tms={"f1": _tm("f1")})
        block = await build_uncertainty_block(storage, _result(["f1"]), _Config())
        assert block is None

    async def test_contradictions_are_high(self) -> None:
        storage = _storage(neurons={"n1": SimpleNamespace(content="disputed content")})
        block = await build_uncertainty_block(storage, _result(["f1"], disputed=["n1"]), _Config())
        assert block is not None
        assert block["level"] == "high"
        assert block["counts"]["contradictions"] == 1
        assert block["contradictions"][0]["neuron_id"] == "n1"
        assert block["contradictions"][0]["content"] == "disputed content"

    async def test_low_confidence_is_high(self) -> None:
        storage = _storage(tms={"f1": _tm("f1")})
        block = await build_uncertainty_block(storage, _result(["f1"], confidence=0.5), _Config())
        assert block is not None
        assert block["level"] == "high"
        assert block["low_confidence"] == {"confidence": 0.5, "threshold": 0.7}

    async def test_superseded_is_medium(self) -> None:
        storage = _storage(tms={"f1": _tm("f1", superseded_by="f2", tags=["geo"])})
        block = await build_uncertainty_block(storage, _result(["f1"]), _Config())
        assert block is not None
        assert block["level"] == "medium"
        assert block["superseded"][0] == {"fiber_id": "f1", "superseded_by": "f2"}

    async def test_expiring_is_medium(self) -> None:
        storage = _storage(expiring=[_tm("f1")])
        block = await build_uncertainty_block(storage, _result(["f1"]), _Config())
        assert block is not None
        assert block["level"] == "medium"
        assert block["counts"]["expiring"] == 1

    async def test_drift_absent_backend_degrades_gracefully(self) -> None:
        # No get_drift_clusters on this storage (e.g. SurrealDB) → empty, no crash.
        storage = _storage(tms={"f1": _tm("f1", superseded_by="f2")})
        block = await build_uncertainty_block(storage, _result(["f1"]), _Config())
        assert block is not None
        assert block["drift_clusters"] == []

    async def test_drift_scoped_to_returned_tags(self) -> None:
        storage = _storage(
            tms={"f1": _tm("f1", tags=["geo"])},
            drift=[
                {"id": "d1", "canonical": "geo", "members": [], "confidence": 0.8},
                {"id": "d2", "canonical": "unrelated", "members": [], "confidence": 0.9},
            ],
        )
        block = await build_uncertainty_block(storage, _result(["f1"]), _Config())
        assert block is not None
        ids = [d["id"] for d in block["drift_clusters"]]
        assert ids == ["d1"]  # only the tag-matching cluster

    async def test_never_raises_on_storage_errors(self) -> None:
        storage = SimpleNamespace(
            get_typed_memories_batch=AsyncMock(side_effect=RuntimeError("boom")),
            get_expiring_memories_for_fibers=AsyncMock(side_effect=RuntimeError("boom")),
            get_neurons_batch=AsyncMock(side_effect=RuntimeError("boom")),
        )
        # Every storage source fails, but disputed_ids come from result.metadata (not a
        # storage call), so that signal survives with empty content — and nothing raises.
        block = await build_uncertainty_block(storage, _result(["f1"], disputed=["n1"]), _Config())
        assert block is not None
        assert block["level"] == "high"
        assert block["contradictions"] == [{"neuron_id": "n1", "content": ""}]
        assert block["superseded"] == []
        assert block["expiring"] == []
        assert block["drift_clusters"] == []

    async def test_storage_errors_with_no_metadata_signal_returns_none(self) -> None:
        storage = SimpleNamespace(
            get_typed_memories_batch=AsyncMock(side_effect=RuntimeError("boom")),
            get_expiring_memories_for_fibers=AsyncMock(side_effect=RuntimeError("boom")),
            get_neurons_batch=AsyncMock(side_effect=RuntimeError("boom")),
        )
        # No disputed ids and confidence above threshold → truly no signal → None.
        block = await build_uncertainty_block(storage, _result(["f1"], confidence=0.95), _Config())
        assert block is None


def _brain_storage(
    *,
    synapses: list[Any] | None = None,
    expiring_count: int = 0,
    typed: list[Any] | None = None,
    total: int = 0,
    drift: list[Any] | None = None,
) -> Any:
    ns = SimpleNamespace()
    ns.get_synapses = AsyncMock(return_value=synapses or [])
    ns.get_expiring_memory_count = AsyncMock(return_value=expiring_count)
    ns.find_typed_memories = AsyncMock(return_value=typed or [])
    ns.count_typed_memories = AsyncMock(return_value=total)
    if drift is not None:
        ns.get_drift_clusters = AsyncMock(return_value=drift)
    return ns


class TestBuildBrainUncertainty:
    async def test_aggregates_all_signals(self) -> None:
        from surreal_memory.core.synapse import SynapseType
        from surreal_memory.engine.uncertainty_report import build_brain_uncertainty

        syn = SimpleNamespace(type=SynapseType.CONTRADICTS, metadata={})
        low = SimpleNamespace(fiber_id="f1", trust_score=0.2, superseded_by=None)
        sup = SimpleNamespace(fiber_id="f2", trust_score=0.9, superseded_by="f3")
        storage = _brain_storage(synapses=[syn], expiring_count=2, typed=[low, sup], total=10)

        block = await build_brain_uncertainty(storage)
        assert block["level"] == "high"
        assert block["counts"] == {
            "contradictions": 1,
            "low_evidence": 1,
            "superseded": 1,
            "expiring": 2,
            "drift_clusters": 0,
        }
        assert block["contradiction_rate"] == 0.1
        assert block["total_memories"] == 10
        assert block["scan"]["typed_scan_truncated"] is False
        assert block["scan"]["contradictions_capped"] is False

    async def test_drift_absent_backend_is_zero(self) -> None:
        from surreal_memory.engine.uncertainty_report import build_brain_uncertainty

        # No get_drift_clusters (SurrealDB) → drift_clusters 0, no raise.
        storage = _brain_storage(total=0)
        block = await build_brain_uncertainty(storage)
        assert block["counts"]["drift_clusters"] == 0
        assert block["contradiction_rate"] == 0.0  # total 0 → no divide-by-zero
