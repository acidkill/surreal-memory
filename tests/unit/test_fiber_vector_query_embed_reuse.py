"""The fiber-vector retriever must reuse the query vector, not embed the query twice.

``_find_anchors_ranked`` runs two retrievers that need the embedded query: step 4
(``_find_embedding_anchors_outcome``) and step 6 (FIBER VECTOR ANCHORS).
Both sit behind the same guard — ``self._embedding_provider is not None`` — so whenever step 6
runs, step 4 has already embedded this exact query a few lines above. Embedding is a network
round-trip to the embedder and is the dominant cost of step 6; doing it a second time doubles
that cost for a vector already in hand, and it is invisible in the results, which is why it
survived the first review of the branch.

``EmbeddingAnchorOutcome.query_vec`` is how the vector travels between them. These tests pin
both halves of the contract: exactly one embed per query, AND step 6 still actually running on
that shared vector — an assertion on the call count alone would pass just as happily if the
whole step had been deleted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.extraction.parser import Perspective, QueryIntent, Stimulus

_QUERY_VECTOR = [0.11, 0.22, 0.33]


def _make_config() -> MagicMock:
    """Everything off except the fiber-vector step — the retriever under test in isolation."""
    config = MagicMock()
    config.max_context_tokens = 500
    config.max_spread_hops = 3
    config.activation_threshold = 0.1
    config.embedding_enabled = False
    config.embedding_similarity_threshold = 0.7
    config.embedding_anchor_mode = "scan"
    config.idf_anchor_enabled = False
    config.fuzzy_search_enabled = False
    config.graph_expansion_enabled = False
    config.query_expansion_synonyms = {}
    config.query_expansion_abbreviations = {}
    config.query_expansion_max_per_term = 0
    config.fiber_vector_enabled = True
    config.fiber_vector_top_n = 10
    return config


def _make_stimulus(query: str = "what did we decide about the vector index") -> Stimulus:
    return Stimulus(
        time_hints=[],
        keywords=["vector", "index"],
        entities=[],
        intent=QueryIntent.RECALL,
        perspective=Perspective.RECALL,
        raw_query=query,
    )


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_fibers_batch = AsyncMock(return_value=[])
    storage.get_neurons_batch = AsyncMock(return_value={})
    storage.get_fibers = AsyncMock(return_value=[])
    storage.get_synapses_for_neurons = AsyncMock(return_value={})
    fiber = Fiber.create(neuron_ids={"neuron-x"}, synapse_ids=set(), anchor_neuron_id="neuron-x")
    storage.find_fibers_by_embedding = AsyncMock(return_value=[(fiber, 0.93)])
    return storage


@pytest.fixture
def counting_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=list(_QUERY_VECTOR))
    provider.similarity = AsyncMock(return_value=0.9)
    return provider


def _pipeline(storage: AsyncMock, provider: AsyncMock) -> ReflexPipeline:
    return ReflexPipeline(
        storage=storage, config=_make_config(), use_reflex=True, embedding_provider=provider
    )


class TestQueryIsEmbeddedOncePerQuery:
    async def test_fiber_vector_step_reuses_the_vector_instead_of_embedding_again(
        self, mock_storage: AsyncMock, counting_provider: AsyncMock
    ) -> None:
        pipeline = _pipeline(mock_storage, counting_provider)

        _sets, ranked_lists, outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert counting_provider.embed.await_count == 1, (
            "the query must be embedded once per query, not once per retriever that needs it"
        )
        # ...and the step it feeds really ran, on that same vector. Without this half, deleting
        # the fiber-vector step entirely would also satisfy the count above.
        mock_storage.find_fibers_by_embedding.assert_awaited_once()
        used_vector = mock_storage.find_fibers_by_embedding.await_args.args[0]
        assert used_vector == _QUERY_VECTOR
        assert outcome.query_vec == _QUERY_VECTOR
        retrievers = {a.retriever for lst in ranked_lists for a in lst}
        assert "fiber_vector" in retrievers

    async def test_no_fiber_vector_lookup_when_the_query_could_not_be_embedded(
        self, mock_storage: AsyncMock, counting_provider: AsyncMock
    ) -> None:
        """A failed embed already loses the semantic anchors and says so in ``source``. Step 6
        has no vector to search with either, so it must be skipped — not retried into the same
        failure, which is what the second embed call used to do."""
        counting_provider.embed = AsyncMock(side_effect=RuntimeError("embedder unreachable"))
        pipeline = _pipeline(mock_storage, counting_provider)

        _sets, ranked_lists, outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert outcome.query_vec is None
        assert outcome.source.startswith("none:embed-failed")
        mock_storage.find_fibers_by_embedding.assert_not_awaited()
        assert not any(a.retriever == "fiber_vector" for lst in ranked_lists for a in lst)

    async def test_the_vector_never_reaches_the_result_metadata(
        self, mock_storage: AsyncMock, counting_provider: AsyncMock
    ) -> None:
        """``EmbeddingAnchorOutcome`` is the object whose provenance fields are copied into
        ``RetrievalResult.metadata``. The vector rides along for reuse only — a thousand floats
        in every recall's metadata would be a real cost, so pin that it stays out of ``repr``
        (the usual accidental route into a log line)."""
        pipeline = _pipeline(mock_storage, counting_provider)

        _sets, _ranked, outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert outcome.query_vec is not None
        assert "query_vec" not in repr(outcome)
        # Assert on the vector as a WHOLE, never on one of its floats: the repr also
        # carries elapsed_ms, and a timing of 0.11xx ms made "0.11" match inside that
        # field (1 of 3 full -n auto runs red, 0 of 30 in isolation — V-GATE round 4, I1).
        assert str(_QUERY_VECTOR) not in repr(outcome)
        assert "0.11, 0.22" not in repr(outcome)
