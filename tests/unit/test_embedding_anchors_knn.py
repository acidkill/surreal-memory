"""``ReflexPipeline._find_embedding_anchors_outcome`` — the KNN anchor path.

Semantic anchors used to come from one path only: probe ``find_neurons(limit=20)``,
then score a capped ``find_neurons(limit=1000)`` page in Python. That page is
ordered by id, so on a brain larger than the cap the semantic retriever only ever
saw its oldest slice — newer memories were structurally invisible to it, no matter
how well they matched.

``embedding_anchor_mode`` adds a second path that asks the storage backend's
vector index directly (``find_neurons_by_embedding``) for the real nearest
neighbours, with the old behaviour kept byte-for-byte as ``"scan"`` and as the
fallback of ``"auto"``. The mocks in this file stand in for that backend, so
every branch of the outcome — success, threshold, tombstone displacement, the
one-shot wider retry, an unsupported/broken backend, and a failed embed — can be
proven without a live SurrealDB. ``tests/integration/test_surrealdb_knn_anchors.py``
is the live counterpart that proves the real index actually reaches memories a
scan cannot see.

These tests exercise ``_find_embedding_anchors_outcome`` directly rather than the
thin ``_find_embedding_anchors`` wrapper, because the wrapper throws away exactly
the provenance (``source``, ``knn_rows``, ``tombstones``, ``above_threshold``)
these tests are here to pin. ``TestQueryMetadataCarriesEmbeddingAnchorProvenance``
is the exception: it drives the real ``query()`` end to end (over a real
``InMemoryStorage``, not a mock) because the thing it proves — that the outcome
reaches ``RetrievalResult.metadata`` on both the early "insufficient signal"
return and the normal return — is a property of ``query()``'s wiring, not of the
anchor lookup itself.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.constants import GRAPH_ONLY_PLACEHOLDER
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage

_LOGGER_NAME = "surreal_memory.engine.retrieval"


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_fibers_batch = AsyncMock(return_value=[])
    storage.get_neurons_batch = AsyncMock(return_value={})
    storage.get_fibers = AsyncMock(return_value=[])
    storage.get_synapses_for_neurons = AsyncMock(return_value={})
    return storage


@pytest.fixture
def mock_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider.similarity = AsyncMock(return_value=0.9)
    return provider


def _make_config(mode: object = "auto", threshold: float = 0.7) -> MagicMock:
    config = MagicMock()
    config.max_context_tokens = 500
    config.max_spread_hops = 3
    config.activation_threshold = 0.1
    config.lateral_inhibition_k = 10
    config.lateral_inhibition_factor = 0.3
    config.reinforcement_delta = 0.05
    config.hebbian_threshold = 0.5
    config.hebbian_delta = 0.1
    config.hebbian_initial_weight = 0.3
    config.embedding_enabled = False
    config.embedding_similarity_threshold = threshold
    config.embedding_anchor_mode = mode
    return config


def _make_pipeline(
    storage: AsyncMock, provider: AsyncMock, mode: object = "auto", threshold: float = 0.7
) -> ReflexPipeline:
    return ReflexPipeline(
        storage=storage,
        config=_make_config(mode=mode, threshold=threshold),
        embedding_provider=provider,
    )


def _neuron(neuron_id: str, *, content: str | None = None) -> Neuron:
    return Neuron.create(type=NeuronType.CONCEPT, content=content or neuron_id, neuron_id=neuron_id)


class TestKnnAnchorLookup:
    """``mode="knn"`` asks the vector index and never touches the scan path."""

    async def test_anchors_come_back_sorted_by_similarity_not_backend_order(
        self, mock_storage: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        low, mid, high = _neuron("low"), _neuron("mid"), _neuron("high")
        # Deliberately out of similarity order — the pipeline, not the mock
        # backend, must be what sorts these.
        mock_storage.find_neurons_by_embedding = AsyncMock(
            return_value=[(low, 0.75), (high, 0.95), (mid, 0.85)]
        )
        pipeline = _make_pipeline(mock_storage, mock_provider, mode="knn")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.anchor_ids == ["high", "mid", "low"]
        assert outcome.source == "knn"
        mock_storage.find_neurons.assert_not_called()

    @pytest.mark.parametrize("top_k, expected_limit", [(10, 30), (20, 60)])
    async def test_backend_is_asked_for_three_times_top_k_neighbours(
        self, mock_storage: AsyncMock, mock_provider: AsyncMock, top_k: int, expected_limit: int
    ) -> None:
        mock_storage.find_neurons_by_embedding = AsyncMock(return_value=[])
        pipeline = _make_pipeline(mock_storage, mock_provider, mode="knn")

        await pipeline._find_embedding_anchors_outcome("query", top_k=top_k)

        mock_storage.find_neurons_by_embedding.assert_awaited_once()
        _, kwargs = mock_storage.find_neurons_by_embedding.await_args
        assert kwargs["limit"] == expected_limit


class TestSimilarityThreshold:
    """The threshold is applied to the cosine the backend itself returns."""

    async def test_cosine_at_0_72_passes_a_0_7_threshold_and_0_68_does_not(
        self, mock_storage: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        passes, fails = _neuron("passes"), _neuron("fails")
        mock_storage.find_neurons_by_embedding = AsyncMock(
            return_value=[(passes, 0.72), (fails, 0.68)]
        )
        pipeline = _make_pipeline(mock_storage, mock_provider, mode="knn", threshold=0.7)

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.anchor_ids == ["passes"]
        assert outcome.above_threshold == 1


class TestTombstoneFiltering:
    """``GRAPH_ONLY_PLACEHOLDER`` rows must never occupy an anchor slot."""

    async def test_tombstones_never_occupy_an_anchor_slot(
        self, mock_storage: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        tombstones = [
            Neuron.create(
                type=NeuronType.CONCEPT, content=GRAPH_ONLY_PLACEHOLDER, neuron_id=f"tomb-{i}"
            )
            for i in range(12)
        ]
        genuine = _neuron("genuine")
        # Tombstones score higher than the genuine memory — if they were not
        # excluded first, ordering alone would let them win every slot.
        rows = [(t, 0.99) for t in tombstones] + [(genuine, 0.9)]
        mock_storage.find_neurons_by_embedding = AsyncMock(return_value=rows)
        pipeline = _make_pipeline(mock_storage, mock_provider, mode="knn")

        # top_k=1 keeps this isolated from the retry-on-shortfall behaviour
        # covered by TestTombstoneRetry below: one surviving anchor already
        # satisfies top_k=1, so no wider re-query is triggered here.
        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=1)

        assert outcome.anchor_ids == ["genuine"]
        assert outcome.tombstones == 12


class TestTombstoneRetry:
    """One bounded wider re-query when tombstones ate the anchor slots."""

    async def test_retries_once_with_a_wider_limit_when_tombstones_leave_too_few_anchors(
        self, mock_storage: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        tombstones = [
            Neuron.create(
                type=NeuronType.CONCEPT, content=GRAPH_ONLY_PLACEHOLDER, neuron_id=f"tomb-{i}"
            )
            for i in range(5)
        ]
        thin_round = [(t, 0.9) for t in tombstones] + [(_neuron("a"), 0.85)]
        wide_genuine = [_neuron(f"g{i}") for i in range(10)]
        wide_round = [(t, 0.9) for t in tombstones] + [(n, 0.85) for n in wide_genuine]
        mock_storage.find_neurons_by_embedding = AsyncMock(side_effect=[thin_round, wide_round])
        pipeline = _make_pipeline(mock_storage, mock_provider, mode="knn")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert mock_storage.find_neurons_by_embedding.await_count == 2
        _, second_kwargs = mock_storage.find_neurons_by_embedding.await_args_list[1]
        assert second_kwargs["limit"] == 120
        assert set(outcome.anchor_ids) == {f"g{i}" for i in range(10)}

    async def test_no_retry_when_the_shortfall_has_no_tombstones_to_blame(
        self, mock_storage: AsyncMock, mock_provider: AsyncMock
    ) -> None:
        few_genuine = [(_neuron(f"g{i}"), 0.85) for i in range(2)]
        mock_storage.find_neurons_by_embedding = AsyncMock(return_value=few_genuine)
        pipeline = _make_pipeline(mock_storage, mock_provider, mode="knn")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert mock_storage.find_neurons_by_embedding.await_count == 1
        assert len(outcome.anchor_ids) == 2


class TestAutoModeFallsBackToScan:
    """``mode="auto"`` prefers the index but degrades to the scan, loudly."""

    async def test_a_result_of_the_wrong_shape_falls_back_to_scan_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bare ``AsyncMock`` answers every call, but not with ``(Neuron,
        float)`` pairs — the shape check, not ``hasattr``, must catch this."""
        storage = AsyncMock()
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2])
        pipeline = _make_pipeline(storage, provider, mode="auto")

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.source == "scan-fallback:knn-invalid-result:AsyncMock"
        assert outcome.anchor_ids == []
        assert any(r.levelname == "WARNING" for r in caplog.records)

    async def test_an_unsupported_backend_falls_back_to_a_nonempty_scan_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(side_effect=NotImplementedError)
        scanned = Neuron.create(
            type=NeuronType.CONCEPT,
            content="scanned memory",
            neuron_id="scanned",
            metadata={"_embedding": [0.1, 0.2, 0.3]},
        )
        storage.find_neurons = AsyncMock(return_value=[scanned])
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        provider.similarity = AsyncMock(return_value=0.9)
        pipeline = _make_pipeline(storage, provider, mode="auto")

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.source.startswith("scan-fallback:knn-unsupported:")
        assert outcome.anchor_ids == ["scanned"]
        assert any(r.levelname == "WARNING" for r in caplog.records)


class TestStrictKnnModeNeverScans:
    """``mode="knn"`` is a hard gate: a broken index yields no anchors, not a scan."""

    async def test_unsupported_backend_yields_no_anchors_and_never_touches_scan(
        self, mock_provider: AsyncMock
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(side_effect=NotImplementedError)
        storage.find_neurons = AsyncMock(return_value=[])
        pipeline = _make_pipeline(storage, mock_provider, mode="knn")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.anchor_ids == []
        assert outcome.source.startswith("none:knn-unsupported:")
        storage.find_neurons.assert_not_called()


class TestNullDistanceIsReportedNotHiddenAsAPerfectMatch:
    """A backend that can't report a distance must fail loudly (``ValueError``),
    never silently read as distance 0 — a perfect match for every row."""

    async def test_null_distance_falls_back_to_scan_in_auto_mode(
        self, mock_provider: AsyncMock
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(
            side_effect=ValueError("null distance from index")
        )
        storage.find_neurons = AsyncMock(return_value=[])
        pipeline = _make_pipeline(storage, mock_provider, mode="auto")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.source == "scan-fallback:knn-bad-distance"

    async def test_null_distance_yields_no_anchors_in_strict_knn_mode(
        self, mock_provider: AsyncMock
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(
            side_effect=ValueError("null distance from index")
        )
        pipeline = _make_pipeline(storage, mock_provider, mode="knn")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.anchor_ids == []
        assert outcome.source == "none:knn-bad-distance"


class TestEmptyKnnResultIsNotAFallback:
    """Zero neighbours found is a legitimate KNN answer, not a degradation."""

    async def test_empty_vector_index_result_stays_source_knn_without_a_warning(
        self, mock_provider: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(return_value=[])
        pipeline = _make_pipeline(storage, mock_provider, mode="knn")

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.source == "knn"
        assert outcome.knn_rows == 0
        assert outcome.anchor_ids == []
        assert not any(r.levelname == "WARNING" for r in caplog.records)


class TestEmbedFailure:
    """Losing the query embedding must be visible, not a quiet debug line."""

    async def test_embed_failure_is_a_warning_and_yields_no_anchors(
        self, mock_storage: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = AsyncMock()
        provider.embed = AsyncMock(side_effect=RuntimeError("embedding service down"))
        pipeline = _make_pipeline(mock_storage, provider, mode="auto")

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.anchor_ids == []
        assert outcome.source == "none:embed-failed:RuntimeError"
        assert any(r.levelname == "WARNING" for r in caplog.records)


class TestModeNormalization:
    """``embedding_anchor_mode`` is strip+lower normalized in ``__init__``."""

    async def test_mode_with_padding_and_mixed_case_behaves_as_strict_knn(
        self, mock_provider: AsyncMock
    ) -> None:
        """ "KNN " must normalize to "knn", not "scan" or "auto" — proven by
        checking it never falls back to the scan when the index is unusable."""
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(side_effect=NotImplementedError)
        storage.find_neurons = AsyncMock(return_value=[])
        pipeline = _make_pipeline(storage, mock_provider, mode="KNN ")

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        assert outcome.source.startswith("none:knn-unsupported:")
        storage.find_neurons.assert_not_called()

    async def test_non_string_mode_warns_at_pipeline_construction(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _make_pipeline(AsyncMock(), AsyncMock(), mode=MagicMock())

        assert any(
            r.levelname == "WARNING" and "embedding_anchor_mode" in r.message
            for r in caplog.records
        )

    async def test_non_string_mode_behaves_as_auto_not_strict_knn(
        self,
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(side_effect=NotImplementedError)
        scanned = Neuron.create(
            type=NeuronType.CONCEPT,
            content="scanned",
            neuron_id="scanned",
            metadata={"_embedding": [0.1, 0.2, 0.3]},
        )
        storage.find_neurons = AsyncMock(return_value=[scanned])
        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        provider.similarity = AsyncMock(return_value=0.9)
        pipeline = _make_pipeline(storage, provider, mode=MagicMock())

        outcome = await pipeline._find_embedding_anchors_outcome("query", top_k=10)

        # Normalized to "auto", not strict "knn": an unsupported backend still
        # falls back to a non-empty scan instead of returning nothing.
        assert outcome.anchor_ids == ["scanned"]
        assert outcome.source.startswith("scan-fallback:knn-unsupported:")

    async def test_unknown_string_mode_warns_at_pipeline_construction(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            _make_pipeline(AsyncMock(), AsyncMock(), mode="bogus-mode")

        assert any(
            r.levelname == "WARNING" and "embedding_anchor_mode" in r.message
            for r in caplog.records
        )


class TestEmbeddingAnchorModeConfigSurface:
    """``embedding_anchor_mode`` must reach a stored brain and survive persistence.

    It is deliberately NOT in ``BrainSettings._EXPLICIT_KEYS`` (see
    ``unified_config.py``), so it has to migrate onto an already-stored brain
    through ``extras`` instead, and round-trip through the SurrealDB
    ``config`` column like any other ``BrainConfig`` field.
    """

    def test_default_is_auto(self) -> None:
        assert BrainConfig().embedding_anchor_mode == "auto"

    def test_reaches_a_stored_brain_through_config_toml_extras(self) -> None:
        """``BrainSettings`` carries a handful of *explicit* keys (``freshness_weight``
        among them) that ``runtime_overrides()`` deliberately does NOT migrate, so a
        stored brain keeps whatever it was created with (issue #168). Any new knob has
        to travel through ``extras`` instead — this test pins that it does, because a
        mode that cannot reach the production brain is a mode that does nothing.
        """
        from surreal_memory.unified_config import BrainSettings

        settings = BrainSettings.from_dict({"embedding_anchor_mode": "knn"})
        assert settings.extras["embedding_anchor_mode"] == "knn"
        assert settings.runtime_overrides()["embedding_anchor_mode"] == "knn"

    def test_deserializing_a_stored_config_without_the_key_defaults_to_auto(self) -> None:
        """A row written before this field existed has no ``embedding_anchor_mode``
        key at all — it must not crash and must not silently pick a non-default mode."""
        from surreal_memory.storage.surrealdb.store import _deserialize_brain_config

        stored = {"reranker_enabled": True}  # a pre-KNN row: the key is simply absent
        restored = _deserialize_brain_config(stored)
        assert restored.embedding_anchor_mode == "auto"

    def test_deserializing_an_unknown_extra_key_does_not_raise(self) -> None:
        """A future field removed again after this one must not take the whole
        stored config down with it — unknown keys are dropped, not fatal."""
        from surreal_memory.storage.surrealdb.store import (
            _deserialize_brain_config,
            _serialize_brain_config,
        )

        stored = _serialize_brain_config(BrainConfig(embedding_anchor_mode="scan"))
        stored["a_field_removed_in_a_future_version"] = 123
        restored = _deserialize_brain_config(stored)
        assert restored.embedding_anchor_mode == "scan"


class TestUnsupportedWarningIsRateLimitedPerPipeline:
    """The unsupported-backend warning fires once per instance, then drops to debug
    — one misconfigured backend must not flood the log on every query."""

    async def test_two_queries_on_the_same_pipeline_log_the_warning_only_once(
        self, mock_provider: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        storage = AsyncMock()
        storage.find_neurons_by_embedding = AsyncMock(side_effect=NotImplementedError)
        storage.find_neurons = AsyncMock(return_value=[])
        pipeline = _make_pipeline(storage, mock_provider, mode="auto")

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            await pipeline._find_embedding_anchors_outcome("first query", top_k=10)
            await pipeline._find_embedding_anchors_outcome("second query", top_k=10)

        vector_index_warnings = [
            r for r in caplog.records if r.levelname == "WARNING" and "vector index" in r.message
        ]
        assert len(vector_index_warnings) == 1


class TestQueryMetadataCarriesEmbeddingAnchorProvenance:
    """``query()`` must surface the embedding-anchor outcome, on every return path.

    A degraded semantic retriever (a fallback, an unsupported backend, a failed
    embed) must be visible in the answer's own metadata, not only in the log —
    otherwise a caller reading ``RetrievalResult`` alone cannot tell a query
    answered from the real vector index apart from one answered from a scan, or
    from no semantic signal at all.
    """

    async def test_early_insufficient_signal_return_still_carries_the_provenance_keys(
        self,
    ) -> None:
        storage = InMemoryStorage()
        brain = Brain.create(name="embedding-anchor-meta-empty-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        # A completely empty brain guarantees zero anchors and zero
        # activations — the sufficiency gate cannot call this anything but
        # insufficient, which is exactly the early-return branch this test
        # needs to exercise.

        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        config = BrainConfig(embedding_anchor_mode="knn", embedding_similarity_threshold=0.7)
        pipeline = ReflexPipeline(storage=storage, config=config, embedding_provider=provider)

        result = await pipeline.query("a query that matches nothing at all")

        assert result.synthesis_method == "insufficient_signal"
        assert result.metadata["embedding_anchor_source"] == "knn"
        assert result.metadata["embedding_anchor_count"] == 0
        assert result.metadata["embedding_anchor_knn_rows"] == 0
        assert isinstance(result.metadata["embedding_anchor_ms"], float)

    async def test_normal_return_also_carries_the_provenance_keys(self) -> None:
        storage = InMemoryStorage()
        brain = Brain.create(name="embedding-anchor-meta-normal-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        neuron = Neuron.create(type=NeuronType.CONCEPT, content="widget alpha protocol handshake")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="widget alpha protocol handshake",
        )
        await storage.add_fiber(fiber)

        provider = AsyncMock()
        provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        config = BrainConfig(embedding_anchor_mode="knn", embedding_similarity_threshold=0.7)
        pipeline = ReflexPipeline(storage=storage, config=config, embedding_provider=provider)

        result = await pipeline.query("widget alpha protocol handshake")

        assert result.synthesis_method != "insufficient_signal"
        assert fiber.id in result.fibers_matched
        # The neuron above was never given a stored embedding, so the vector
        # index legitimately finds nothing — the point here is that the keys
        # exist on the SUCCESSFUL path too, not what their values are.
        assert result.metadata["embedding_anchor_source"] == "knn"
        assert result.metadata["embedding_anchor_count"] == 0
        assert result.metadata["embedding_anchor_knn_rows"] == 0
        assert isinstance(result.metadata["embedding_anchor_ms"], float)
