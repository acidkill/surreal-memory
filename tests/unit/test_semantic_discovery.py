"""Tests for semantic synapse discovery."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
from surreal_memory.engine.edge_identity import deterministic_edge_id
from surreal_memory.engine.semantic_discovery import (
    SemanticDiscoveryResult,
    _cosine_similarity,
    _provider_cache,
    discover_semantic_synapses,
)
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest.fixture
def brain_config() -> BrainConfig:
    return BrainConfig(
        embedding_enabled=True,
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=100,
    )


@pytest.fixture
def brain_config_disabled() -> BrainConfig:
    return BrainConfig(embedding_enabled=False)


@pytest.fixture
def brain(brain_config: BrainConfig) -> Brain:
    return Brain.create(name="test", config=brain_config)


@pytest.fixture
async def storage(brain: Brain) -> InMemoryStorage:
    store = InMemoryStorage()
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


@pytest.fixture(autouse=True)
def _effective_from_brain_config() -> Any:
    """Make _effective_embedding derive from the passed brain config.

    "Effective config wins" resolves embedding settings from the unified config
    (the real ~/.surrealmemory/config.toml), which would make these unit tests
    depend on the developer's machine. Pin it to the brain config each test
    constructs so the tests stay deterministic and assert their intended logic.
    Individual tests may re-patch _effective_embedding to test precedence.
    """

    def _derive(config: BrainConfig) -> tuple[bool, str, str]:
        return (
            config.embedding_enabled,
            config.embedding_provider,
            config.embedding_model,
        )

    with patch(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        side_effect=_derive,
    ):
        yield


class TestCosimeSimilarity:
    """Tests for cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_similar_vectors(self) -> None:
        sim = _cosine_similarity([1.0, 0.5], [1.0, 0.6])
        assert sim > 0.9


class TestSemanticDiscoveryResult:
    """Tests for result dataclass."""

    def test_defaults(self) -> None:
        r = SemanticDiscoveryResult()
        assert r.neurons_embedded == 0
        assert r.pairs_evaluated == 0
        assert r.synapses_created == 0
        assert r.skipped_existing == 0
        assert r.synapses == []

    def test_frozen(self) -> None:
        r = SemanticDiscoveryResult(neurons_embedded=5)
        with pytest.raises(AttributeError):
            r.neurons_embedded = 10  # type: ignore[misc]


class TestDiscoverSemanticSynapses:
    """Tests for the main discovery function."""

    async def test_skips_when_embedding_disabled(
        self, storage: InMemoryStorage, brain_config_disabled: BrainConfig
    ) -> None:
        result = await discover_semantic_synapses(storage, brain_config_disabled)
        assert result.neurons_embedded == 0
        assert result.synapses_created == 0

    async def test_skips_when_provider_unavailable(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider",
            side_effect=ImportError("no provider"),
        ):
            result = await discover_semantic_synapses(storage, brain_config)
            assert result.neurons_embedded == 0

    async def test_skips_fewer_than_two_neurons(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="only one", neuron_id="n1")
        await storage.add_neuron(n1)

        mock_provider = AsyncMock()
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)
            assert result.neurons_embedded == 0

    async def test_discovers_similar_neurons(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Two similar neurons should produce a SIMILAR_TO synapse."""
        # Discovery reads the embedding STORED on each neuron
        # (metadata["_embedding"]) and never re-embeds: n1 and n2 are very
        # similar, n3 is different.
        n1 = Neuron.create(
            type=NeuronType.CONCEPT,
            content="machine learning",
            neuron_id="n1",
            metadata={"_embedding": [0.9, 0.1, 0.0]},
        )
        n2 = Neuron.create(
            type=NeuronType.CONCEPT,
            content="deep learning",
            neuron_id="n2",
            metadata={"_embedding": [0.85, 0.15, 0.0]},  # similar to n1
        )
        n3 = Neuron.create(
            type=NeuronType.ENTITY,
            content="pizza recipe",
            neuron_id="n3",
            metadata={"_embedding": [0.0, 0.1, 0.9]},  # different
        )
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        result = await discover_semantic_synapses(storage, brain_config)

        assert result.neurons_embedded == 3
        assert result.synapses_created >= 1

        # Check the created synapse
        found_similar = False
        for syn in result.synapses:
            if syn.type == SynapseType.SIMILAR_TO:
                found_similar = True
                assert syn.metadata.get("_semantic_discovery") is True
                assert "cosine_similarity" in syn.metadata
                # Weight should be similarity * 0.6
                assert syn.weight > 0.0
                assert syn.weight <= 0.6
        assert found_similar

    async def test_skips_existing_synapse_pairs(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Should not create duplicate synapses."""
        n1 = Neuron.create(
            type=NeuronType.CONCEPT,
            content="alpha",
            neuron_id="n1",
            metadata={"_embedding": [1.0, 0.0]},
        )
        n2 = Neuron.create(
            type=NeuronType.CONCEPT,
            content="beta",
            neuron_id="n2",
            metadata={"_embedding": [0.99, 0.01]},  # very similar to n1
        )
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Pre-existing synapse
        existing = Synapse.create(
            source_id="n1", target_id="n2", type=SynapseType.RELATED_TO, weight=0.5
        )
        await storage.add_synapse(existing)

        result = await discover_semantic_synapses(storage, brain_config)

        assert result.skipped_existing >= 1
        assert result.synapses_created == 0

    async def test_respects_max_pairs(self, storage: InMemoryStorage) -> None:
        """Should cap results at max_pairs."""
        config = BrainConfig(
            embedding_enabled=True,
            semantic_discovery_similarity_threshold=0.0,  # Accept all
            semantic_discovery_max_pairs=2,
        )

        for i in range(5):
            n = Neuron.create(type=NeuronType.CONCEPT, content=f"concept {i}", neuron_id=f"n{i}")
            await storage.add_neuron(n)

        mock_provider = AsyncMock()
        # All similar to each other
        mock_provider.embed_batch.return_value = [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
        ]

        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, config)

        # Should be capped at 2
        assert result.synapses_created <= 2

    async def test_ignores_non_concept_entity_neurons(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Only CONCEPT and ENTITY neurons should be considered."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="valid concept", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.TIME, content="2026-01-01", neuron_id="n2")
        n3 = Neuron.create(type=NeuronType.STATE, content="happy", neuron_id="n3")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)
        await storage.add_neuron(n3)

        mock_provider = AsyncMock()
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)

        # Only 1 eligible neuron (n1), so < 2, should skip
        assert result.neurons_embedded == 0

    async def test_handles_embed_batch_failure(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Should gracefully handle embedding failures."""
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="alpha", neuron_id="n1")
        n2 = Neuron.create(type=NeuronType.CONCEPT, content="beta", neuron_id="n2")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        mock_provider = AsyncMock()
        mock_provider.embed_batch.side_effect = RuntimeError("embedding failed")

        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider",
            return_value=mock_provider,
        ):
            result = await discover_semantic_synapses(storage, brain_config)

        assert result.neurons_embedded == 0


class TestConsolidationIntegration:
    """Tests for semantic_link integration in ConsolidationEngine."""

    async def test_semantic_link_strategy_exists(self) -> None:
        """SEMANTIC_LINK should be a valid strategy."""
        assert ConsolidationStrategy.SEMANTIC_LINK == "semantic_link"

    async def test_semantic_link_in_tier(self) -> None:
        """SEMANTIC_LINK should be in a tier."""
        found = False
        for tier in ConsolidationEngine.STRATEGY_TIERS:
            if ConsolidationStrategy.SEMANTIC_LINK in tier:
                found = True
        assert found

    async def test_semantic_link_runs_in_consolidation(self, storage: InMemoryStorage) -> None:
        """Running SEMANTIC_LINK strategy should call discover_semantic_synapses."""
        brain_config = BrainConfig(embedding_enabled=False)
        brain = Brain.create(name="test", config=brain_config)
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        engine = ConsolidationEngine(storage)
        report = await engine.run(strategies=[ConsolidationStrategy.SEMANTIC_LINK])
        # Should complete without error even if embedding is disabled
        assert report.semantic_synapses_created == 0

    async def test_report_has_semantic_synapses_field(self, storage: InMemoryStorage) -> None:
        """ConsolidationReport should track semantic_synapses_created."""
        brain = Brain.create(name="test", config=BrainConfig())
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        engine = ConsolidationEngine(storage)
        report = await engine.run(strategies=[ConsolidationStrategy.SEMANTIC_LINK])
        assert hasattr(report, "semantic_synapses_created")
        assert report.semantic_synapses_created == 0

    async def test_semantic_discovery_metadata_flag(self) -> None:
        """Semantic synapses should have _semantic_discovery metadata."""
        synapse = Synapse.create(
            source_id="a",
            target_id="b",
            type=SynapseType.SIMILAR_TO,
            weight=0.4,
            metadata={"_semantic_discovery": True},
        )
        assert synapse.metadata.get("_semantic_discovery") is True


class TestProviderCache:
    """Tests for embedding provider singleton cache (#100)."""

    def test_cache_returns_same_instance(self) -> None:
        """_create_provider should return cached instance on second call."""
        from surreal_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(
            embedding_provider="sentence_transformer",
            embedding_model="test-model",
        )
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "sentence_transformer", "test-model"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="",
            ),
            patch(
                "surreal_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding",
            ) as mock_st,
        ):
            p1 = _create_provider(config)
            p2 = _create_provider(config)
            assert p1 is p2
            # Constructor called only once
            mock_st.assert_called_once_with(model_name="test-model")

        # Cleanup cache to avoid polluting other tests
        _provider_cache.clear()

    def test_cache_keys_by_provider_and_model(self) -> None:
        """Different model names should get separate cache entries."""
        from unittest.mock import MagicMock

        from surreal_memory.engine.semantic_discovery import _create_provider

        config_a = BrainConfig(
            embedding_provider="sentence_transformer",
            embedding_model="model-a",
        )
        config_b = BrainConfig(
            embedding_provider="sentence_transformer",
            embedding_model="model-b",
        )

        def _effective(cfg: BrainConfig) -> tuple[bool, str, str]:
            return (True, cfg.embedding_provider, cfg.embedding_model)

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                side_effect=_effective,
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="",
            ),
            patch(
                "surreal_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding",
                side_effect=lambda **kw: MagicMock(name=f"ST({kw})"),
            ) as mock_st,
        ):
            p1 = _create_provider(config_a)
            p2 = _create_provider(config_b)
            assert p1 is not p2
            assert mock_st.call_count == 2

        _provider_cache.clear()

    def test_effective_config_wins_over_stale_brain_config(self) -> None:
        """When the stored brain config is stale, _create_provider must use the
        effective (unified) provider/model, not the brain config's values.
        """
        from surreal_memory.engine.semantic_discovery import _create_provider

        # Stale brain config: disabled + sentence_transformer default.
        stale = BrainConfig(
            embedding_enabled=False,
            embedding_provider="sentence_transformer",
            embedding_model="all-MiniLM-L6-v2",
        )

        # Effective (unified) config says gemini/enabled — overrides the autouse
        # fixture for this test to assert the precedence directly.
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "gemini", "gemini-embedding-001"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="",
            ),
            patch(
                "surreal_memory.engine.embedding.gemini_embedding.GeminiEmbedding",
            ) as mock_gemini,
            patch(
                "surreal_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding",
            ) as mock_st,
        ):
            _create_provider(stale)
            mock_gemini.assert_called_once()
            mock_st.assert_not_called()

        _provider_cache.clear()

    def test_openai_provider_receives_configured_endpoint(self) -> None:
        """The previously-dead [embedding] endpoint config field must now
        reach OpenAIEmbedding as base_url (fix for #6 in the index-perf plan:
        _create_provider never called resolved_endpoint())."""
        from surreal_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(embedding_provider="openai", embedding_model="text-embed")
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "openai", "text-embed"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="http://127.0.0.1:11435/v1",
            ),
            patch(
                "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding",
            ) as mock_openai,
        ):
            _create_provider(config)
            mock_openai.assert_called_once_with(
                model="text-embed", base_url="http://127.0.0.1:11435/v1"
            )

        _provider_cache.clear()

    def test_openai_provider_no_endpoint_configured(self) -> None:
        """No config/env endpoint set → base_url=None (provider falls back to
        its own env check), NOT an empty string."""
        from surreal_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(embedding_provider="openai", embedding_model="text-embed")
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "openai", "text-embed"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="",
            ),
            patch(
                "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding",
            ) as mock_openai,
        ):
            _create_provider(config)
            mock_openai.assert_called_once_with(model="text-embed", base_url=None)

        _provider_cache.clear()

    def test_ollama_provider_receives_configured_endpoint(self) -> None:
        from surreal_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(embedding_provider="ollama", embedding_model="nomic-embed-text")
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "ollama", "nomic-embed-text"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="http://127.0.0.1:11434",
            ),
            patch(
                "surreal_memory.engine.embedding.ollama_embedding.OllamaEmbedding",
            ) as mock_ollama,
        ):
            _create_provider(config)
            mock_ollama.assert_called_once_with(
                model="nomic-embed-text", base_url="http://127.0.0.1:11434"
            )

        _provider_cache.clear()

    def test_ollama_provider_no_endpoint_uses_default(self) -> None:
        """No configured endpoint → base_url kwarg is omitted entirely, so
        OllamaEmbedding's own module-level default (OLLAMA_BASE_URL) applies —
        its constructor requires a str, never None."""
        from surreal_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(embedding_provider="ollama", embedding_model="nomic-embed-text")
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "ollama", "nomic-embed-text"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                return_value="",
            ),
            patch(
                "surreal_memory.engine.embedding.ollama_embedding.OllamaEmbedding",
            ) as mock_ollama,
        ):
            _create_provider(config)
            mock_ollama.assert_called_once_with(model="nomic-embed-text")

        _provider_cache.clear()

    def test_cache_keys_by_endpoint_too(self) -> None:
        """Same provider/model, different endpoint → different cache entry,
        so editing [embedding] endpoint doesn't return a stale provider."""
        from unittest.mock import MagicMock

        from surreal_memory.engine.semantic_discovery import _create_provider

        config = BrainConfig(embedding_provider="openai", embedding_model="text-embed")
        endpoints = iter(["http://host-a/v1", "http://host-b/v1"])
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding",
                return_value=(True, "openai", "text-embed"),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._effective_embedding_endpoint",
                side_effect=lambda: next(endpoints),
            ),
            patch(
                "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding",
                side_effect=lambda **kw: MagicMock(name=f"openai({kw})"),
            ) as mock_openai,
        ):
            p1 = _create_provider(config)
            p2 = _create_provider(config)
            assert p1 is not p2
            assert mock_openai.call_count == 2

        _provider_cache.clear()


def _embedded(neuron_id: str, vector: list[float]) -> Neuron:
    return Neuron.create(
        type=NeuronType.CONCEPT,
        content=f"content-{neuron_id}",
        neuron_id=neuron_id,
        metadata={"_embedding": vector},
    )


class TestStructuralIdempotency:
    """The pair, not the row, is the identity of a SIMILAR_TO edge.

    Measured on the live brain before this change: the snapshot guard did hold
    (two consecutive runs added 2000 then 1077 rows and never re-minted an edge
    over an existing pair), but idempotency rested entirely on reading the whole
    synapse table into memory first. These tests move that guarantee into the id.
    """

    async def test_created_synapses_carry_the_deterministic_id(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        await storage.add_neuron(_embedded("n1", [0.9, 0.1, 0.0]))
        await storage.add_neuron(_embedded("n2", [0.85, 0.15, 0.0]))

        result = await discover_semantic_synapses(storage, brain_config)

        assert result.synapses
        for syn in result.synapses:
            assert syn.id == deterministic_edge_id(
                SynapseType.SIMILAR_TO, syn.source_id, syn.target_id
            )

    async def test_both_orderings_of_a_pair_share_one_id(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """Whichever endpoint discovery happens to visit first, it is one edge."""
        await storage.add_neuron(_embedded("n1", [0.9, 0.1, 0.0]))
        await storage.add_neuron(_embedded("n2", [0.85, 0.15, 0.0]))

        result = await discover_semantic_synapses(storage, brain_config)

        ids = {s.id for s in result.synapses}
        assert len(ids) == 1, "one pair must not be expressed as two edges"

    async def test_second_run_creates_nothing_over_the_same_pairs(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """The acceptance criterion, stated correctly.

        "Second run == 0 created" only holds once the neighbourhood is
        saturated -- while a backlog remains a second run legitimately creates
        new pairs. Here the brain is tiny, so run 1 saturates it and run 2 must
        create nothing and count the pairs as skipped.
        """
        await storage.add_neuron(_embedded("n1", [0.9, 0.1, 0.0]))
        await storage.add_neuron(_embedded("n2", [0.85, 0.15, 0.0]))

        first = await discover_semantic_synapses(storage, brain_config)
        for syn in first.synapses:
            await storage.add_synapse(syn)

        second = await discover_semantic_synapses(storage, brain_config)

        assert first.synapses_created >= 1
        assert second.synapses_created == 0
        assert second.skipped_existing >= 1

    async def test_a_pair_joined_by_any_type_is_left_alone(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """The snapshot's job: never lay a second edge over a connected pair."""
        await storage.add_neuron(_embedded("n1", [0.9, 0.1, 0.0]))
        await storage.add_neuron(_embedded("n2", [0.85, 0.15, 0.0]))
        await storage.add_synapse(
            Synapse.create(
                source_id="n1",
                target_id="n2",
                type=SynapseType.RELATED_TO,
                weight=0.5,
            )
        )

        result = await discover_semantic_synapses(storage, brain_config)

        assert result.synapses_created == 0
        assert result.skipped_existing >= 1


class TestTruncationIsReported:
    async def test_truncated_is_false_when_under_the_cap(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        await storage.add_neuron(_embedded("n1", [0.9, 0.1, 0.0]))
        await storage.add_neuron(_embedded("n2", [0.85, 0.15, 0.0]))

        result = await discover_semantic_synapses(storage, brain_config)

        assert result.truncated is False

    async def test_truncated_is_true_when_the_cap_bites(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        """A capped run must say so, or it reads as a stuck system."""
        capped = BrainConfig(
            embedding_enabled=True,
            semantic_discovery_similarity_threshold=0.5,
            semantic_discovery_max_pairs=1,
        )
        for i in range(4):
            await storage.add_neuron(_embedded(f"n{i}", [0.9 - i * 0.01, 0.1, 0.0]))

        result = await discover_semantic_synapses(storage, capped)

        assert result.synapses_created == 1
        assert result.truncated is True


class TestSnapshotIsPaged:
    async def test_existing_pairs_are_read_page_by_page(
        self, storage: InMemoryStorage, brain_config: BrainConfig
    ) -> None:
        """An unbounded read of this table is the '[Errno 104]' failure mode."""
        await storage.add_neuron(_embedded("n1", [0.9, 0.1, 0.0]))
        await storage.add_neuron(_embedded("n2", [0.85, 0.15, 0.0]))

        calls: list[dict[str, Any]] = []
        real = storage.get_synapses

        async def _spy(**kwargs: Any) -> list[Synapse]:
            calls.append(kwargs)
            return await real(**kwargs)

        with patch.object(storage, "get_synapses", side_effect=_spy):
            await discover_semantic_synapses(storage, brain_config)

        assert calls, "the snapshot must still be taken"
        assert all(c.get("limit") for c in calls), "every snapshot read must be bounded"
        assert calls[0].get("offset") == 0
