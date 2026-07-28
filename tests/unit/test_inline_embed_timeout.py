"""Regression test for #79: a slow embedding provider must not fail the write.

``_embed_created_neurons`` runs inside the write path so a fresh memory is
semantically recallable immediately. It was already fail-soft about a *missing*
provider ("keyword-only memory, no error"), but not about a *slow* one: a remote
or rate-limited endpoint blocked until its own timeout, pushing ``smem_remember``
past the 30s tool-call cap MCP hosts impose. The write had already succeeded, so
the caller saw a timeout and could not tell whether the memory landed — the exact
loss this tool exists to prevent.

The bound turns "slow provider" into "vector arrives later" rather than
"caller cannot tell what happened".
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from surreal_memory.engine.encoder import MemoryEncoder, _inline_embed_timeout


class TestInlineEmbedTimeoutConfig:
    def test_default_sits_under_the_mcp_tool_cap(self, monkeypatch: Any) -> None:
        """Must be well below the 30s cap, or the bound does not solve #79."""
        monkeypatch.delenv("SURREAL_MEMORY_INLINE_EMBED_TIMEOUT", raising=False)
        assert 0 < _inline_embed_timeout() < 30.0

    def test_env_override_is_honoured(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_INLINE_EMBED_TIMEOUT", "2.5")
        assert _inline_embed_timeout() == 2.5

    def test_non_positive_disables_the_bound(self, monkeypatch: Any) -> None:
        """Escape hatch for anyone who prefers the old blocking behaviour."""
        monkeypatch.setenv("SURREAL_MEMORY_INLINE_EMBED_TIMEOUT", "0")
        assert _inline_embed_timeout() == 0.0

    def test_garbage_falls_back_instead_of_raising(self, monkeypatch: Any) -> None:
        """A typo in the env must not take down every write."""
        monkeypatch.setenv("SURREAL_MEMORY_INLINE_EMBED_TIMEOUT", "soon")
        assert _inline_embed_timeout() == 10.0


class _Neuron:
    """Minimal stand-in for a created neuron."""

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


class _HangingProvider:
    """Never answers — stands in for a rate-limited / cold-start remote endpoint."""

    async def embed_batch(self, _texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_slow_provider_does_not_block_the_write(monkeypatch: Any) -> None:
    """The write path returns promptly; the memory is simply left keyword-only."""
    monkeypatch.setenv("SURREAL_MEMORY_INLINE_EMBED_TIMEOUT", "0.2")
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _cfg: (True, None, None),
        raising=True,
    )
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._create_provider",
        lambda _cfg, task_type=None: _HangingProvider(),
        raising=True,
    )

    updates: list[Any] = []

    class _Storage:
        async def update_neuron(self, neuron: Any) -> None:
            updates.append(neuron)

    encoder = MemoryEncoder.__new__(MemoryEncoder)
    encoder._storage = _Storage()  # type: ignore[attr-defined]
    encoder._config = object()  # type: ignore[attr-defined]

    ctx = _Ctx([_Neuron("n1")])

    # Generous ceiling: the point is that it returns on the 0.2s budget, not on
    # the provider's own (here: never).
    await asyncio.wait_for(encoder._embed_created_neurons(ctx), timeout=10)

    # Fail-soft: no vector was written, and crucially no exception propagated —
    # the neuron itself was already persisted upstream.
    assert updates == []
