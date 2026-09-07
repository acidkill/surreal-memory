"""reindex --stale detects vectors that no longer describe their text (#199).

``--missing-only`` is blind to a stored vector describing content the neuron
no longer contains; ``--all`` rewrites everything. The stale mode re-embeds
what it inspects, compares cosine(stored, fresh) against the threshold, and
rewrites only what diverges. These tests pin the selection inverse, the
cosine helper, and an end-to-end run against the in-memory backend with a
canned provider.
"""

from __future__ import annotations

import math

import pytest

from surreal_memory.cli.commands import reindex as reindex_mod
from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class TestHelpers:
    def test_stale_selection_is_the_inverse(self) -> None:
        with_vec = Neuron.create(
            type=NeuronType.CONCEPT, content="x", neuron_id="a"
        )
        with_vec.metadata["_embedding"] = [0.1, 0.2]
        without_vec = Neuron.create(
            type=NeuronType.CONCEPT, content="x", neuron_id="b"
        )
        assert reindex_mod._needs_embedding(with_vec, all_neurons=False, stale=True)
        assert not reindex_mod._needs_embedding(
            without_vec, all_neurons=False, stale=True
        )

    def test_cosine_identical_vectors_is_one(self) -> None:
        v = _unit([1.0, 2.0, 3.0])
        assert reindex_mod._cosine_similarity(v, v[:]) == pytest.approx(1.0)

    def test_cosine_orthogonal_is_zero(self) -> None:
        assert reindex_mod._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


class _CannedProvider:
    """Returns a fixed vector per text: 'stale' text maps elsewhere."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            _unit([1.0, 0.0]) if "stale" in t else _unit([0.0, 1.0]) for t in texts
        ]


@pytest.mark.asyncio
async def test_stale_run_rewrites_only_diverged(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = InMemoryStorage()
    brain = Brain.create(name="stale-test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    fresh = Neuron.create(type=NeuronType.CONCEPT, content="fresh content", neuron_id="n1")
    fresh.metadata["_embedding"] = _unit([0.0, 1.0])  # matches canned output
    stale = Neuron.create(type=NeuronType.CONCEPT, content="stale content", neuron_id="n2")
    stale.metadata["_embedding"] = _unit([0.0, 1.0])  # describes the OLD text;
    # the canned provider embeds "stale content" to [1, 0] — orthogonal → diverged
    missing = Neuron.create(type=NeuronType.CONCEPT, content="no vector", neuron_id="n3")
    for n in (fresh, stale, missing):
        await storage.add_neuron(n)

    provider = _CannedProvider()
    import surreal_memory.engine.semantic_discovery as sd

    # reindex imports these inside the function body — patch the SOURCE module.
    monkeypatch.setattr(sd, "_effective_embedding", lambda cfg: (True, "openai", "bge-m3"))
    monkeypatch.setattr(sd, "_create_provider", lambda cfg, task_type=None: provider)
    monkeypatch.setattr(reindex_mod, "get_config", lambda: type("C", (), {"data_dir": None})())
    monkeypatch.setattr(reindex_mod, "get_storage", lambda *a, **k: _ret(storage))

    async def _ret(s: InMemoryStorage) -> InMemoryStorage:
        return s
    written: list[tuple[str, list[float]]] = []

    async def _capture(pairs: list[tuple[str, list[float]]]) -> None:
        written.extend(pairs)

    monkeypatch.setattr(storage, "update_neuron_embeddings", _capture)

    await reindex_mod._reindex_async(
        brain="", dry_run=False, all_neurons=False, stale=True,
        threshold=0.98, batch_size=10, json_output=True,
    )

    assert [nid for nid, _ in written] == ["n2"], "only the diverged vector is rewritten"
    assert provider.calls >= 1
