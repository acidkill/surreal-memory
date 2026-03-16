"""Tests for CrossMemoryLinkStep — anchor-to-anchor linking via shared entities."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import Synapse, SynapseType
from neural_memory.engine.pipeline_steps import CrossMemoryLinkStep
from neural_memory.utils.timeutils import utcnow


def _make_neuron(neuron_id: str, content: str, ntype: NeuronType = NeuronType.ENTITY) -> Neuron:
    return Neuron(
        id=neuron_id,
        type=ntype,
        content=content,
        created_at=utcnow(),
    )


def _make_ctx(
    anchor_id: str = "anchor-new",
    entity_neurons: list[Neuron] | None = None,
) -> SimpleNamespace:
    entities = entity_neurons or []
    anchor = _make_neuron(anchor_id, "new memory content")
    return SimpleNamespace(
        anchor_neuron=anchor,
        entity_neurons=entities,
        neurons_created=[anchor] + entities,
        neurons_linked=[],
        synapses_created=[],
    )


def _involves_synapse(source_id: str, target_id: str) -> Synapse:
    """Create an INVOLVES synapse (anchor → entity)."""
    return Synapse.create(
        source_id=source_id,
        target_id=target_id,
        type=SynapseType.INVOLVES,
        weight=0.7,
    )


class TestCrossMemoryLinkStep:
    @pytest.mark.asyncio
    async def test_creates_link_via_shared_entity(self) -> None:
        """Two memories sharing an entity should get linked."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        entity = _make_neuron("entity-python", "Python")
        ctx = _make_ctx(entity_neurons=[entity])

        # Old anchor "anchor-old" also has INVOLVES → entity-python
        storage.get_synapses.return_value = [
            _involves_synapse("anchor-old", "entity-python"),
        ]

        result = await step.execute(ctx, storage, config)

        storage.add_synapse.assert_called_once()
        synapse = storage.add_synapse.call_args[0][0]
        assert synapse.source_id == "anchor-new"
        assert synapse.target_id == "anchor-old"
        assert synapse.type == SynapseType.RELATED_TO
        assert synapse.weight == 0.3  # base weight, 1 shared entity
        assert synapse.metadata["_cross_memory"] is True
        assert synapse.metadata["_shared_entity_count"] == 1
        assert "anchor-old" in result.neurons_linked

    @pytest.mark.asyncio
    async def test_weight_scales_with_shared_entities(self) -> None:
        """More shared entities → stronger link weight."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        entity_a = _make_neuron("entity-python", "Python")
        entity_b = _make_neuron("entity-fastapi", "FastAPI")
        ctx = _make_ctx(entity_neurons=[entity_a, entity_b])

        # Same old anchor shares both entities
        storage.get_synapses.side_effect = [
            [_involves_synapse("anchor-old", "entity-python")],
            [_involves_synapse("anchor-old", "entity-fastapi")],
        ]

        await step.execute(ctx, storage, config)

        storage.add_synapse.assert_called_once()
        synapse = storage.add_synapse.call_args[0][0]
        # 2 shared entities: 0.3 + 0.1 * (2-1) = 0.4
        assert synapse.weight == pytest.approx(0.4)
        assert synapse.metadata["_shared_entity_count"] == 2

    @pytest.mark.asyncio
    async def test_skips_self_links(self) -> None:
        """Should not create link from anchor to itself."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        entity = _make_neuron("entity-python", "Python")
        ctx = _make_ctx(entity_neurons=[entity])

        # Only synapse points back to the new anchor itself
        storage.get_synapses.return_value = [
            _involves_synapse("anchor-new", "entity-python"),
        ]

        await step.execute(ctx, storage, config)
        storage.add_synapse.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_common_entities(self) -> None:
        """Entities appearing in >50 anchors should be skipped."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        entity = _make_neuron("entity-code", "code")
        ctx = _make_ctx(entity_neurons=[entity])

        # 51 synapses → exceeds frequency cap
        storage.get_synapses.return_value = [
            _involves_synapse(f"anchor-{i}", "entity-code") for i in range(51)
        ]

        await step.execute(ctx, storage, config)
        storage.add_synapse.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_max_links_per_encode(self) -> None:
        """Should cap total cross-memory links per encode."""
        step = CrossMemoryLinkStep()
        step.MAX_LINKS_PER_ENCODE = 3
        storage = AsyncMock()
        config = AsyncMock()

        entity = _make_neuron("entity-python", "Python")
        ctx = _make_ctx(entity_neurons=[entity])

        # 10 old anchors share this entity
        storage.get_synapses.return_value = [
            _involves_synapse(f"anchor-old-{i}", "entity-python") for i in range(10)
        ]

        await step.execute(ctx, storage, config)
        assert storage.add_synapse.call_count == 3

    @pytest.mark.asyncio
    async def test_no_entities_no_links(self) -> None:
        """No entity neurons → no links created."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx(entity_neurons=[])
        await step.execute(ctx, storage, config)
        storage.get_synapses.assert_not_called()
        storage.add_synapse.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_anchor_no_links(self) -> None:
        """No anchor neuron → no links created."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = SimpleNamespace(
            anchor_neuron=None,
            entity_neurons=[_make_neuron("e1", "Python")],
        )
        await step.execute(ctx, storage, config)
        storage.get_synapses.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_synapse_handled(self) -> None:
        """Should gracefully handle duplicate synapse errors."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        entity = _make_neuron("entity-python", "Python")
        ctx = _make_ctx(entity_neurons=[entity])

        storage.get_synapses.return_value = [
            _involves_synapse("anchor-old", "entity-python"),
        ]
        storage.add_synapse.side_effect = ValueError("Duplicate synapse")

        # Should not raise
        result = await step.execute(ctx, storage, config)
        assert "anchor-old" not in result.neurons_linked

    @pytest.mark.asyncio
    async def test_weight_capped_at_max(self) -> None:
        """Weight should never exceed WEIGHT_CAP."""
        step = CrossMemoryLinkStep()
        storage = AsyncMock()
        config = AsyncMock()

        # 10 entities all shared with same anchor
        entities = [_make_neuron(f"entity-{i}", f"Entity{i}") for i in range(10)]
        ctx = _make_ctx(entity_neurons=entities)

        storage.get_synapses.side_effect = [
            [_involves_synapse("anchor-old", f"entity-{i}")] for i in range(10)
        ]

        await step.execute(ctx, storage, config)

        synapse = storage.add_synapse.call_args[0][0]
        assert synapse.weight <= step.WEIGHT_CAP
