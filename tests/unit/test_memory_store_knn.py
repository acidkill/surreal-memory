"""``InMemoryStorage.find_neurons_by_embedding`` — brute-force cosine parity.

SurrealDB answers semantic-anchor lookups through its HNSW index;
``InMemoryStorage`` has to honour the same *contract* — best-first cosine
similarity, bounded by ``limit``, filterable by ``type_filter``, scoped to the
current brain, tolerant of rows with no vector or a degenerate one — with a
plain Python scan, since it is the fixture every unit test in this suite runs
retrieval logic against (see the module docstring of ``storage/memory_store.py``).
A silent divergence here would mean the in-memory suite validates a different
ranking than the one SurrealDB ever produces in production.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage


def _neuron(
    neuron_id: str,
    embedding: list[float] | None,
    *,
    type_: NeuronType = NeuronType.CONCEPT,
    content: str | None = None,
) -> Neuron:
    metadata = {} if embedding is None else {"_embedding": embedding}
    return Neuron.create(
        type=type_,
        content=content or neuron_id,
        metadata=metadata,
        neuron_id=neuron_id,
    )


class TestFindNeuronsByEmbeddingOrdering:
    """Best-first cosine similarity over the current brain's neurons."""

    async def test_results_are_sorted_by_cosine_similarity_descending(
        self, storage: InMemoryStorage
    ) -> None:
        # Inserted out of similarity order on purpose — the method, not
        # insertion order, must be what decides the ranking.
        await storage.add_neuron(_neuron("low", [0.0, 1.0]))
        await storage.add_neuron(_neuron("high", [1.0, 0.0]))
        await storage.add_neuron(_neuron("mid", [1.0, 1.0]))

        results = await storage.find_neurons_by_embedding([1.0, 0.0], limit=10)

        assert [neuron.id for neuron, _ in results] == ["high", "mid", "low"]
        scores = {neuron.id: score for neuron, score in results}
        assert scores["high"] == pytest.approx(1.0)
        assert scores["mid"] == pytest.approx(0.7071, abs=1e-3)
        assert scores["low"] == pytest.approx(0.0)

    async def test_limit_caps_the_number_of_results_returned(
        self, storage: InMemoryStorage
    ) -> None:
        await storage.add_neuron(_neuron("high", [1.0, 0.0]))
        await storage.add_neuron(_neuron("mid", [1.0, 1.0]))
        await storage.add_neuron(_neuron("low", [0.0, 1.0]))

        results = await storage.find_neurons_by_embedding([1.0, 0.0], limit=2)

        assert [neuron.id for neuron, _ in results] == ["high", "mid"]

    async def test_type_filter_excludes_a_better_match_of_the_wrong_type(
        self, storage: InMemoryStorage
    ) -> None:
        # The best cosine match is an ENTITY neuron; only the weaker CONCEPT
        # match may come back once the caller asks for CONCEPT only.
        await storage.add_neuron(_neuron("entity-best", [1.0, 0.0], type_=NeuronType.ENTITY))
        await storage.add_neuron(_neuron("concept-ok", [1.0, 0.5], type_=NeuronType.CONCEPT))

        results = await storage.find_neurons_by_embedding(
            [1.0, 0.0], limit=10, type_filter=NeuronType.CONCEPT
        )

        assert [neuron.id for neuron, _ in results] == ["concept-ok"]

    async def test_neuron_with_no_stored_embedding_is_skipped_not_crashed(
        self, storage: InMemoryStorage
    ) -> None:
        await storage.add_neuron(_neuron("no-vector", None))
        await storage.add_neuron(_neuron("has-vector", [1.0, 0.0]))

        results = await storage.find_neurons_by_embedding([1.0, 0.0], limit=10)

        assert [neuron.id for neuron, _ in results] == ["has-vector"]

    async def test_neuron_with_a_zero_norm_embedding_is_skipped(
        self, storage: InMemoryStorage
    ) -> None:
        # A zero vector has an undefined cosine (division by zero) — it must
        # be excluded outright, not silently scored as a 0.0 similarity match.
        await storage.add_neuron(_neuron("zero-vector", [0.0, 0.0]))
        await storage.add_neuron(_neuron("has-vector", [1.0, 0.0]))

        results = await storage.find_neurons_by_embedding([1.0, 0.0], limit=10)

        assert [neuron.id for neuron, _ in results] == ["has-vector"]

    async def test_neurons_from_another_brain_never_leak_into_the_result(
        self, storage: InMemoryStorage, brain: Brain
    ) -> None:
        # brain2's neuron is a PERFECT match for the query and would rank
        # first if brain scoping leaked — proving isolation, not just that
        # the filter happens to agree with the ranking.
        await storage.add_neuron(_neuron("brain1-neuron", [1.0, 1.0]))

        other_brain = Brain.create(name="other_brain_knn_test")
        await storage.save_brain(other_brain)
        storage.set_brain(other_brain.id)
        await storage.add_neuron(_neuron("brain2-neuron", [1.0, 0.0]))

        storage.set_brain(brain.id)
        results = await storage.find_neurons_by_embedding([1.0, 0.0], limit=10)

        ids = [neuron.id for neuron, _ in results]
        assert ids == ["brain1-neuron"]
