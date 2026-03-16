"""Tests for IDF-weighted keyword synapse creation (B4)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.core.synapse import SynapseType
from neural_memory.engine.pipeline_steps import CreateSynapsesStep
from neural_memory.utils.timeutils import utcnow


def _make_neuron(neuron_id: str, content: str, ntype: NeuronType = NeuronType.ENTITY) -> Neuron:
    return Neuron(
        id=neuron_id,
        type=ntype,
        content=content,
        created_at=utcnow(),
    )


def _make_ctx(
    content: str = "PostgreSQL database performance optimization",
    concept_neurons: list[Neuron] | None = None,
) -> SimpleNamespace:
    anchor = _make_neuron("anchor-1", content, NeuronType.ENTITY)
    concepts = concept_neurons or []
    return SimpleNamespace(
        content=content,
        language="en",
        anchor_neuron=anchor,
        time_neurons=[],
        entity_neurons=[],
        concept_neurons=concepts,
        action_neurons=[],
        intent_neurons=[],
        neurons_created=[anchor] + concepts,
        neurons_linked=[],
        synapses_created=[],
        effective_metadata={},
    )


class TestIDFWeighting:
    """IDF adjusts keyword synapse weights based on corpus frequency."""

    @pytest.mark.asyncio
    async def test_cold_start_uses_position_weights(self) -> None:
        """With < 5 fibers, IDF is skipped (position-only weights)."""
        step = CreateSynapsesStep()
        storage = AsyncMock()
        config = AsyncMock()

        concept = _make_neuron("c1", "postgresql", NeuronType.CONCEPT)
        ctx = _make_ctx(
            content="PostgreSQL database optimization",
            concept_neurons=[concept],
        )

        # Cold start: only 3 fibers
        storage.get_total_fiber_count.return_value = 3
        storage.get_keyword_df_batch.return_value = {}

        await step.execute(ctx, storage, config)

        # Should NOT query DF (cold start skip)
        storage.get_keyword_df_batch.assert_not_called()
        # Should still create synapse with position-based weight
        assert storage.add_synapse.call_count >= 1

    @pytest.mark.asyncio
    async def test_rare_keyword_gets_high_weight(self) -> None:
        """Rare keywords (low DF) get IDF boost → higher synapse weight."""
        step = CreateSynapsesStep()
        storage = AsyncMock()
        config = AsyncMock()

        concept = _make_neuron("c1", "postgresql", NeuronType.CONCEPT)
        ctx = _make_ctx(
            content="PostgreSQL database optimization",
            concept_neurons=[concept],
        )

        # 100 fibers, "postgresql" appears in only 2
        storage.get_total_fiber_count.return_value = 100
        storage.get_keyword_df_batch.return_value = {"postgresql": 2}

        await step.execute(ctx, storage, config)

        # Find the concept synapse
        concept_synapses = [
            call[0][0]
            for call in storage.add_synapse.call_args_list
            if call[0][0].type == SynapseType.RELATED_TO and call[0][0].target_id == "c1"
        ]
        assert len(concept_synapses) == 1
        # Rare keyword → IDF close to 1.0 → high final weight
        assert concept_synapses[0].weight > 0.5

    @pytest.mark.asyncio
    async def test_common_keyword_gets_low_weight(self) -> None:
        """Common keywords (high DF) get IDF penalty → lower synapse weight."""
        step = CreateSynapsesStep()
        storage = AsyncMock()
        config = AsyncMock()

        concept = _make_neuron("c1", "code", NeuronType.CONCEPT)
        ctx = _make_ctx(
            content="code review process",
            concept_neurons=[concept],
        )

        # 100 fibers, "code" appears in 90 of them
        storage.get_total_fiber_count.return_value = 100
        storage.get_keyword_df_batch.return_value = {"code": 90}

        await step.execute(ctx, storage, config)

        concept_synapses = [
            call[0][0]
            for call in storage.add_synapse.call_args_list
            if call[0][0].type == SynapseType.RELATED_TO and call[0][0].target_id == "c1"
        ]
        assert len(concept_synapses) == 1
        # Common keyword → low IDF → lower weight
        # IDF floor is 0.2, so weight should be lower
        assert concept_synapses[0].weight < 0.7

    @pytest.mark.asyncio
    async def test_idf_floor_prevents_zero_weight(self) -> None:
        """IDF factor never goes below 0.2 (floor)."""
        step = CreateSynapsesStep()
        storage = AsyncMock()
        config = AsyncMock()

        concept = _make_neuron("c1", "code", NeuronType.CONCEPT)
        ctx = _make_ctx(
            content="code review process",
            concept_neurons=[concept],
        )

        # "code" in ALL 100 fibers
        storage.get_total_fiber_count.return_value = 100
        storage.get_keyword_df_batch.return_value = {"code": 100}

        await step.execute(ctx, storage, config)

        concept_synapses = [
            call[0][0]
            for call in storage.add_synapse.call_args_list
            if call[0][0].type == SynapseType.RELATED_TO and call[0][0].target_id == "c1"
        ]
        assert len(concept_synapses) == 1
        # Even with max DF, weight should be > 0 (floor=0.2)
        assert concept_synapses[0].weight > 0.4  # min(0.8, 0.4 + 0.3 * 0.2 * position_weight)

    @pytest.mark.asyncio
    async def test_df_updated_after_encode(self) -> None:
        """DF table is updated with keywords from the encoded memory."""
        step = CreateSynapsesStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx(
            content="PostgreSQL database optimization",
            concept_neurons=[],
        )

        storage.get_total_fiber_count.return_value = 10
        storage.get_keyword_df_batch.return_value = {}

        await step.execute(ctx, storage, config)

        # increment_keyword_df should be called with extracted keywords
        storage.increment_keyword_df.assert_called_once()
        keywords = storage.increment_keyword_df.call_args[0][0]
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # All keywords should be lowercased
        assert all(kw == kw.lower() for kw in keywords)

    @pytest.mark.asyncio
    async def test_no_keywords_no_df_update(self) -> None:
        """Empty content → no keywords → no DF update."""
        step = CreateSynapsesStep()
        storage = AsyncMock()
        config = AsyncMock()

        ctx = _make_ctx(content="a", concept_neurons=[])

        await step.execute(ctx, storage, config)

        # With single-char content, no keywords extracted → no IDF queries
        storage.get_total_fiber_count.assert_not_called()
        storage.increment_keyword_df.assert_not_called()

    @pytest.mark.asyncio
    async def test_idf_math_correctness(self) -> None:
        """Verify IDF calculation matches expected formula."""
        # IDF = log((N+1) / (1+df)) / log(N+1)
        total = 100
        df = 5
        idf_raw = math.log((total + 1) / (1 + df))
        idf_max = math.log(total + 1)
        idf_normalized = idf_raw / idf_max
        # Should be in (0, 1)
        assert 0.0 < idf_normalized < 1.0
        # For df=5 out of 100, should be relatively high
        assert idf_normalized > 0.5

        # Edge: df = total (appears in every fiber)
        df_all = 100
        idf_raw_all = math.log((total + 1) / (1 + df_all))
        idf_all = idf_raw_all / idf_max
        # log((101)/(101)) = log(1) = 0 → IDF is 0, floor kicks in at 0.2
        assert idf_all == 0.0
        # In CreateSynapsesStep, max(0.2, idf_all) → 0.2 (floor)
