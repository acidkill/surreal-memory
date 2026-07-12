"""U3 auto-hook: conflict resolution → per-fact supersession lineage.

Covers the two halves of the automatic path:
* ``ConflictDetectionStep`` records superseded old anchors on the pipeline context.
* ``RememberHandler._apply_supersessions`` turns those into real A-side validity +
  SUPERSEDES synapse + old-anchor metadata after the new fiber is saved.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import SynapseType
from surreal_memory.engine.encoder import EncodingResult
from surreal_memory.engine.pipeline import PipelineContext
from surreal_memory.engine.pipeline_steps import ConflictDetectionStep
from surreal_memory.mcp.remember_handler import _apply_supersessions
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="supersession-auto")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


async def _make_fact(store: InMemoryStorage, hint: str) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"fact-{hint}")
    await store.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=f"fact {hint}",
    )
    await store.add_fiber(fiber)
    await store.add_typed_memory(TypedMemory.create(fiber_id=fiber.id, memory_type=MemoryType.FACT))
    return fiber


class TestApplySupersessions:
    """RememberHandler._apply_supersessions on real InMemoryStorage."""

    async def test_applies_lineage_for_pending_old_anchor(self, storage: InMemoryStorage) -> None:
        old = await _make_fact(storage, "oslo")
        new = await _make_fact(storage, "bergen")
        result = EncodingResult(
            fiber=new,
            neurons_created=[],
            neurons_linked=[],
            synapses_created=[],
            pending_supersessions=[old.anchor_neuron_id],
        )

        applied = await _apply_supersessions(storage, result)

        assert applied == 1
        tm = await storage.get_typed_memory(old.id)
        assert tm is not None
        assert tm.superseded_by == new.id
        assert tm.valid_until is not None
        synapses = await storage.get_synapses(
            source_id=new.anchor_neuron_id, target_id=old.anchor_neuron_id
        )
        assert any(s.type == SynapseType.SUPERSEDES for s in synapses)

    async def test_empty_pending_is_noop(self, storage: InMemoryStorage) -> None:
        new = await _make_fact(storage, "solo")
        result = EncodingResult(
            fiber=new,
            neurons_created=[],
            neurons_linked=[],
            synapses_created=[],
            pending_supersessions=[],
        )
        assert await _apply_supersessions(storage, result) == 0

    async def test_ambiguous_old_anchor_skipped(self, storage: InMemoryStorage) -> None:
        # An anchor that maps to two fibers is ambiguous → not superseded.
        shared = Neuron.create(type=NeuronType.CONCEPT, content="shared")
        await storage.add_neuron(shared)
        for i in range(2):
            fib = Fiber.create(
                neuron_ids={shared.id},
                synapse_ids=set(),
                anchor_neuron_id=shared.id,
                summary=f"dup {i}",
            )
            await storage.add_fiber(fib)
            await storage.add_typed_memory(
                TypedMemory.create(fiber_id=fib.id, memory_type=MemoryType.FACT)
            )
        new = await _make_fact(storage, "winner")
        result = EncodingResult(
            fiber=new,
            neurons_created=[],
            neurons_linked=[],
            synapses_created=[],
            pending_supersessions=[shared.id],
        )
        assert await _apply_supersessions(storage, result) == 0


class TestConflictStepCollectsSupersessions:
    """ConflictDetectionStep records superseded old anchors on the context."""

    async def test_superseded_standard_resolution_recorded(
        self, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conflict = MagicMock(existing_neuron_id="old-anchor")
        resolution = MagicMock(superseded=True, contradicts_synapse=MagicMock(), conflict=conflict)

        async def fake_detect(**_: object) -> list[object]:
            return [conflict]

        async def fake_auto(*_: object, **__: object) -> MagicMock:
            return MagicMock(auto_resolved=False, resolution="")

        async def fake_resolve(**_: object) -> list[object]:
            return [resolution]

        monkeypatch.setattr(
            "surreal_memory.engine.conflict_detection.detect_conflicts", fake_detect
        )
        monkeypatch.setattr(
            "surreal_memory.engine.conflict_detection.resolve_conflicts", fake_resolve
        )
        monkeypatch.setattr(
            "surreal_memory.engine.conflict_auto_resolve.try_auto_resolve", fake_auto
        )

        ctx = PipelineContext(
            content="Emma lives in Bergen",
            timestamp=utcnow(),
            metadata={"type": "fact"},
            tags=set(),
            language="en",
        )
        ctx.anchor_neuron = MagicMock(id="new-anchor")
        ctx.effective_metadata = {"type": "fact"}
        ctx.merged_tags = set()

        out = await ConflictDetectionStep().execute(ctx, storage, BrainConfig())

        assert out.pending_supersessions == ["old-anchor"]

    async def test_keep_new_auto_resolution_recorded(
        self, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conflict = MagicMock(existing_neuron_id="old-anchor-2")

        async def fake_detect(**_: object) -> list[object]:
            return [conflict]

        async def fake_auto(*_: object, **__: object) -> MagicMock:
            return MagicMock(auto_resolved=True, resolution="keep_new")

        monkeypatch.setattr(
            "surreal_memory.engine.conflict_detection.detect_conflicts", fake_detect
        )
        monkeypatch.setattr(
            "surreal_memory.engine.conflict_auto_resolve.try_auto_resolve", fake_auto
        )

        ctx = PipelineContext(
            content="Emma lives in Bergen",
            timestamp=utcnow(),
            metadata={"type": "fact"},
            tags=set(),
            language="en",
        )
        ctx.anchor_neuron = MagicMock(id="new-anchor-2")
        ctx.effective_metadata = {"type": "fact"}
        ctx.merged_tags = set()

        out = await ConflictDetectionStep().execute(ctx, storage, BrainConfig())

        assert out.pending_supersessions == ["old-anchor-2"]
