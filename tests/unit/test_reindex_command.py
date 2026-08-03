"""Tests for the ``smem reindex`` CLI command.

These exercise the async core (`_reindex_async`) with a mocked storage and a
mocked embedding provider — they never hit a real database or embedding API.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.cli.commands.reindex import _needs_embedding, _reindex_async
from surreal_memory.core.neuron import Neuron, NeuronType


def _neuron(content: str, *, embedding: list[float] | None = None) -> Neuron:
    meta: dict[str, Any] = {}
    if embedding is not None:
        meta["_embedding"] = embedding
    return Neuron.create(type=NeuronType.CONCEPT, content=content, metadata=meta)


def _make_storage(neurons: list[Neuron]) -> MagicMock:
    storage = MagicMock()
    storage.brain_id = "default"
    brain = MagicMock()
    brain.config = MagicMock(
        embedding_enabled=False,
        embedding_provider="sentence_transformer",
        embedding_model="all-MiniLM-L6-v2",
    )
    storage.get_brain = AsyncMock(return_value=brain)

    async def _find(limit: int = 100, offset: int = 0, **_: Any) -> list[Neuron]:
        return neurons[offset : offset + limit]

    storage.find_neurons = AsyncMock(side_effect=_find)
    storage.update_neuron_embeddings = AsyncMock()
    return storage


def _patches(storage: MagicMock, provider: Any, *, enabled: bool = True):
    return (
        patch("surreal_memory.cli.commands.reindex.get_config", return_value=MagicMock()),
        patch(
            "surreal_memory.cli.commands.reindex.get_storage",
            new=AsyncMock(return_value=storage),
        ),
        patch(
            "surreal_memory.engine.semantic_discovery._effective_embedding",
            return_value=(enabled, "gemini", "gemini-embedding-001"),
        ),
        patch(
            "surreal_memory.engine.semantic_discovery._create_provider",
            return_value=provider,
        ),
    )


class TestNeedsEmbedding:
    def test_skips_blank_content(self) -> None:
        assert _needs_embedding(_neuron("   "), all_neurons=False) is False

    def test_missing_only_skips_existing_vector(self) -> None:
        assert _needs_embedding(_neuron("x", embedding=[0.1]), all_neurons=False) is False

    def test_missing_only_includes_unembedded(self) -> None:
        assert _needs_embedding(_neuron("x"), all_neurons=False) is True

    def test_all_includes_embedded(self) -> None:
        assert _needs_embedding(_neuron("x", embedding=[0.1]), all_neurons=True) is True


class TestReindexAsync:
    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self) -> None:
        storage = _make_storage([_neuron("a"), _neuron("b", embedding=[0.1])])
        provider = MagicMock()
        p1, p2, p3, p4 = _patches(storage, provider)
        with p1, p2, p3, p4:
            await _reindex_async(
                brain="", dry_run=True, all_neurons=False, batch_size=64, json_output=False
            )
        storage.update_neuron_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_embeds_only_missing(self) -> None:
        storage = _make_storage([_neuron("a"), _neuron("b", embedding=[0.1, 0.2]), _neuron("c")])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[[1.0], [2.0]])
        p1, p2, p3, p4 = _patches(storage, provider)
        with p1, p2, p3, p4:
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=64, json_output=False
            )
        # Both unembedded neurons land in ONE update_neuron_embeddings call —
        # not two update_neuron round-trips.
        storage.update_neuron_embeddings.assert_awaited_once()
        pairs = storage.update_neuron_embeddings.await_args.args[0]
        assert len(pairs) == 2
        assert {p[1][0] for p in pairs} == {1.0, 2.0}
        provider.embed_batch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_all_reembeds_everything(self) -> None:
        storage = _make_storage([_neuron("a", embedding=[0.1]), _neuron("b")])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[[1.0], [2.0]])
        p1, p2, p3, p4 = _patches(storage, provider)
        with p1, p2, p3, p4:
            await _reindex_async(
                brain="", dry_run=False, all_neurons=True, batch_size=64, json_output=False
            )
        storage.update_neuron_embeddings.assert_awaited_once()
        assert len(storage.update_neuron_embeddings.await_args.args[0]) == 2

    @pytest.mark.asyncio
    async def test_disabled_effective_config_exits(self) -> None:
        import typer

        storage = _make_storage([_neuron("a")])
        provider = MagicMock()
        p1, p2, p3, p4 = _patches(storage, provider, enabled=False)
        with p1, p2, p3, p4, pytest.raises(typer.Exit):
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=64, json_output=False
            )
        storage.update_neuron_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_fail_soft_per_batch(self) -> None:
        """A write failure for one embedding batch must not abort the run —
        with batched writes the fail-soft unit is a batch, not a neuron."""
        storage = _make_storage([_neuron("a"), _neuron("b")])
        storage.update_neuron_embeddings = AsyncMock(side_effect=[None, RuntimeError("db error")])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[[1.0]])
        p1, p2, p3, p4 = _patches(storage, provider)
        # batch_size=1 forces two separate write batches for the two neurons.
        with p1, p2, p3, p4:
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=1, json_output=False
            )
        assert storage.update_neuron_embeddings.await_count == 2


class TestFailureReporting:
    """A run that embeds nothing must say WHY, stop early, and exit non-zero.

    The pre-fix behaviour printed one identical low-information line per batch
    ("batch N-M failed (skipped)"), ground through every remaining neuron
    repeating the same error, and exited 0 — so the real cause (an HTTP 400
    naming the wrong host) was invisible and any caller recorded a clean run.
    """

    @pytest.mark.asyncio
    async def test_real_error_is_reported_once(self, capsys: pytest.CaptureFixture[str]) -> None:
        storage = _make_storage([_neuron(f"n{i}") for i in range(8)])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(side_effect=RuntimeError("Unknown Model 'bge-m3'"))
        p1, p2, p3, p4 = _patches(storage, provider)
        import typer

        with p1, p2, p3, p4, pytest.raises(typer.Exit):
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=1, json_output=False
            )

        err = capsys.readouterr().err
        assert "Unknown Model 'bge-m3'" in err, "the real exception was swallowed"
        assert err.count("Unknown Model") == 1, "the error was repeated per batch"

    @pytest.mark.asyncio
    async def test_aborts_early_when_nothing_succeeds(self) -> None:
        """Stop after a few consecutive failures instead of retrying thousands."""
        storage = _make_storage([_neuron(f"n{i}") for i in range(50)])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(side_effect=RuntimeError("endpoint refused"))
        p1, p2, p3, p4 = _patches(storage, provider)
        import typer

        with p1, p2, p3, p4, pytest.raises(typer.Exit):
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=1, json_output=False
            )

        assert provider.embed_batch.await_count <= 5, (
            f"kept going for {provider.embed_batch.await_count} batches with nothing embedded"
        )

    @pytest.mark.asyncio
    async def test_total_failure_exits_non_zero(self) -> None:
        storage = _make_storage([_neuron("a")])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(side_effect=RuntimeError("boom"))
        p1, p2, p3, p4 = _patches(storage, provider)
        import typer

        with p1, p2, p3, p4, pytest.raises(typer.Exit) as exc:
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=64, json_output=False
            )
        assert exc.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_partial_success_still_exits_zero(self) -> None:
        """Some progress is not a failed run — only a total wash-out exits 1."""
        storage = _make_storage([_neuron("a"), _neuron("b")])
        provider = MagicMock()
        provider.embed_batch = AsyncMock(side_effect=[[[1.0]], RuntimeError("later boom")])
        p1, p2, p3, p4 = _patches(storage, provider)

        with p1, p2, p3, p4:
            await _reindex_async(
                brain="", dry_run=False, all_neurons=False, batch_size=1, json_output=False
            )
