"""Regression: an unavailable embed provider must log at the same level as a
timeout, with the same actionable hint.

Both failure paths in `_embed_created_neurons` produce the same observable
outcome — the neuron is saved keyword-only, no vector is written, and
`smem reindex` is the fix — but the `TimeoutError` branch logs at WARNING
with a hint, while the `Exception` branch logs at DEBUG with no hint at
all. Since debug is invisible under any default logging config, a
provider-down failure was silent by default while the (rarer) timeout
case was loud.

#99 added the timeout branch and framed it as producing "the same outcome"
as the existing debug branch, so this divergence in visibility reads as an
oversight rather than a deliberate design.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from surreal_memory.engine.encoder import MemoryEncoder


class _Neuron:
    """Minimal stand-in for a created neuron (mirrors test_inline_embed_timeout.py)."""

    def __init__(self, nid: str) -> None:
        from surreal_memory.core.neuron import NeuronType

        self.id = nid
        self.content = "alpha content"
        self.metadata: dict[str, Any] = {}
        self.ephemeral = False
        self.type = NeuronType.CONCEPT


class _Ctx:
    def __init__(self, neurons: list[Any]) -> None:
        self.neurons_created = neurons
        self.anchor_neuron = None


class _FailingProvider:
    """Stands in for a provider whose endpoint is down / returns an error."""

    async def embed_batch(self, _texts: list[str]) -> list[list[float]]:
        raise RuntimeError("connection refused: bge-m3 endpoint unreachable")


@pytest.mark.asyncio
async def test_provider_unavailable_logs_warning_with_reindex_hint(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The regression: WARNING with the `smem reindex` hint, not DEBUG-silent."""
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _cfg: (True, None, None),
        raising=True,
    )
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._create_provider",
        lambda _cfg, task_type=None: _FailingProvider(),
        raising=True,
    )

    class _Storage:
        async def update_neuron(self, neuron: Any) -> None:
            raise AssertionError("update_neuron must not be called when embed failed")

    encoder = MemoryEncoder.__new__(MemoryEncoder)
    encoder._storage = _Storage()  # type: ignore[attr-defined]
    encoder._config = object()  # type: ignore[attr-defined]

    ctx = _Ctx([_Neuron("n1"), _Neuron("n2")])

    with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.encoder"):
        await encoder._embed_created_neurons(ctx)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # Exactly one warning about the skipped embed; no DEBUG-silent burial.
    assert len(warnings) == 1, (
        f"expected exactly one WARNING about the skipped embed; got: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
    )
    msg = warnings[0].getMessage()
    assert "smem reindex" in msg, f"warning must include the `smem reindex` hint; got: {msg!r}"
    assert "provider unavailable" in msg, f"warning must name the failure category; got: {msg!r}"
    # exc_info=True — so the operator can see WHY the provider was down, not
    # just that it was. Matches the sibling TimeoutError branch's contract.
    assert warnings[0].exc_info is not None, (
        "warning must carry the exception chain via exc_info=True"
    )


class TestProviderUnavailableThrottling:
    """The write path calls the encoder in a loop (train, train-db,
    remember_batch), so an unthrottled WARNING+exc_info per neuron turns a
    down provider into a ~1 KiB-per-record log flood. Policy: first
    occurrence and every 100th warn in full; the rest log at DEBUG with the
    running count."""

    @staticmethod
    def _make_encoder(monkeypatch: pytest.MonkeyPatch) -> MemoryEncoder:
        import surreal_memory.engine.encoder as encoder_mod

        monkeypatch.setattr(encoder_mod, "_EMBED_UNAVAILABLE_COUNT", 0)
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery._effective_embedding",
            lambda _cfg: (True, None, None),
            raising=True,
        )
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery._create_provider",
            lambda _cfg, task_type=None: _FailingProvider(),
            raising=True,
        )

        class _Storage:
            async def update_neuron(self, neuron: Any) -> None:
                raise AssertionError("update_neuron must not be called when embed failed")

        encoder = MemoryEncoder.__new__(MemoryEncoder)
        encoder._storage = _Storage()  # type: ignore[attr-defined]
        encoder._config = object()  # type: ignore[attr-defined]
        return encoder

    @pytest.mark.asyncio
    async def test_three_failures_produce_one_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        encoder = self._make_encoder(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger="surreal_memory.engine.encoder"):
            for _ in range(3):
                await encoder._embed_created_neurons(_Ctx([_Neuron("n1"), _Neuron("n2")]))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            f"expected exactly 1 WARNING for 3 failures; got "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        assert "smem reindex" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debugs) == 2, "the other two failures must remain visible at DEBUG"

    @pytest.mark.asyncio
    async def test_warning_recurs_every_hundredth_occurrence(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        encoder = self._make_encoder(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.encoder"):
            for _ in range(101):
                await encoder._embed_created_neurons(_Ctx([_Neuron("n1")]))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2, f"expected warnings at occurrence 1 and 100; got {len(warnings)}"
        assert "occurrence 1" in warnings[0].getMessage()
        assert "occurrence 100" in warnings[1].getMessage()
