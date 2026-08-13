"""K4 (run 013) — organic connectivity in maintenance pulse + stats hints.

Proves the two dup call-sites (`maintenance_handler._health_pulse`,
`stats_handler._generate_stats_hints`) and the `_evaluate_thresholds` static
helper now score connectivity on the ORGANIC subgraph, and that the 1.5 / 2.0
thresholds (left UNCHANGED) still fire correctly:

- a genuinely thin ORGANIC brain fires both the enrich hint and the auto-dream
  trigger (the rescue path is intact);
- a code-heavy brain whose organic graph is HEALTHY fires NEITHER — this used
  to be a false positive (code-index noise deflated the ratio) and going silent
  is the FIX, not a regression.

See DECISIONS_K4.md for why the thresholds were NOT recalibrated.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.mcp.maintenance_handler import (
    HealthPulse,
    MaintenanceHandler,
    _evaluate_thresholds,
)
from surreal_memory.mcp.stats_handler import StatsHandler
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import MaintenanceConfig, UnifiedConfig


async def _seed(
    *, indexed: int, organic: int, edges_per_organic: int
) -> tuple[InMemoryStorage, str]:
    store = InMemoryStorage()
    brain = Brain.create(name="k4", config=BrainConfig(), owner_id="test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    for i in range(indexed):
        await store.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT,
                content=f"idx-{i}",
                neuron_id=f"idx-{i}",
                metadata={"indexed": True},
            )
        )
    for i in range(organic):
        await store.add_neuron(
            Neuron.create(type=NeuronType.ENTITY, content=f"org-{i}", neuron_id=f"org-{i}")
        )
    sids: set[str] = set()
    if organic > 0:
        for i in range(edges_per_organic * organic):
            sid = f"e-{i}"
            await store.add_synapse(
                Synapse.create(
                    source_id=f"org-{i % organic}",
                    target_id=f"org-{(i + 1) % organic}",
                    type=SynapseType.RELATED_TO,
                    weight=0.6,
                    synapse_id=sid,
                )
            )
            sids.add(sid)
    await store.add_fiber(
        Fiber.create(
            neuron_ids={f"idx-{i}" for i in range(indexed)} | {f"org-{i}" for i in range(organic)},
            synapse_ids=sids,
            anchor_neuron_id="org-0" if organic else "idx-0",
            fiber_id="f-0",
        )
    )
    return store, brain.id


class _FakeServer(MaintenanceHandler):
    def __init__(self, storage: InMemoryStorage) -> None:
        self._storage = storage
        self.config = UnifiedConfig(maintenance=MaintenanceConfig())

    async def get_storage(self) -> InMemoryStorage:
        return self._storage

    async def _maybe_run_expiry_cleanup(self) -> int:
        return 0


async def _pulse(store: InMemoryStorage) -> HealthPulse:
    p = await _FakeServer(store)._health_pulse()
    assert p is not None
    return p


async def _stats_hint(store: InMemoryStorage, brain_id: str) -> str | None:
    handler = StatsHandler.__new__(StatsHandler)
    stats = await store.get_enhanced_stats(brain_id)
    hints = await handler._generate_stats_hints(store, brain_id, stats)
    return next((h for h in hints if "connectivity" in h.lower()), None)


class TestThinOrganicBrainFiresRescue:
    @pytest.mark.asyncio
    async def test_thin_organic_brain_fires_stats_hint_and_dream(self) -> None:
        # 60 organic neurons, 0.5 edges/neuron → organic connectivity 0.5 < 2.0.
        store, brain_id = await _seed(indexed=0, organic=60, edges_per_organic=0)
        # add 30 organic edges manually → 0.5/neuron
        for i in range(30):
            await store.add_synapse(
                Synapse.create(
                    source_id=f"org-{i % 60}",
                    target_id=f"org-{(i + 1) % 60}",
                    type=SynapseType.RELATED_TO,
                    weight=0.6,
                    synapse_id=f"s-{i}",
                )
            )

        hint = await _stats_hint(store, brain_id)
        assert hint is not None
        assert "Low connectivity" in hint

        pulse = await _pulse(store)
        assert pulse.connectivity < 1.5
        assert any("connectivity" in h.message.lower() for h in pulse.hints)


class TestCodeHeavyHealthyOrganicDoesNotFire:
    @pytest.mark.asyncio
    async def test_code_heavy_healthy_organic_brain_silent(self) -> None:
        # 200 indexed neurons (code-index) + 60 organic at 5/neuron (healthy).
        # Organic connectivity 5.0 > 2.0 → no hint, no dream. Pre-fix the indexed
        # neurons deflated the ratio and fired a false alarm.
        store, brain_id = await _seed(indexed=200, organic=60, edges_per_organic=5)

        hint = await _stats_hint(store, brain_id)
        assert hint is None, f"unexpected low-connectivity hint: {hint}"

        pulse = await _pulse(store)
        # organic connectivity ~5.0 → above the 1.5 dream threshold
        assert pulse.connectivity >= 1.5
        assert not any("connectivity" in h.message.lower() for h in pulse.hints)


class TestThresholdsUnchangedSemantics:
    def test_evaluate_thresholds_uses_organic_connectivity(self) -> None:
        # connectivity passed in is already the organic ratio (as the pulse now
        # provides). 1.0 organic → low-connectivity hint fires.
        hints = _evaluate_thresholds(
            fiber_count=100,
            neuron_count=60,
            synapse_count=60,
            connectivity=1.0,
            orphan_ratio=0.0,
            cfg=MaintenanceConfig(),
        )
        assert any("connectivity" in h.message.lower() for h in hints)
