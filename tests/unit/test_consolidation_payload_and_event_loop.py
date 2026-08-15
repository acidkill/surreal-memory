"""Consolidation must not drag vectors it never reads, nor hold the event loop.

Three defects with one shared symptom — a `smem consolidate` that stalls and then
floods the terminal with `[Errno 104] Connection reset by peer`:

1. Passes that only look at ids, tags or the synapse graph still asked storage
   for every neuron's 1024-float embedding, because `find_neurons` defaults
   `include_embedding=True`. `dream` pulled 10 000 of them in one response.
2. `semantic_discovery` scanned the WHOLE neuron table to keep the CONCEPT/ENTITY
   minority, and then spent minutes in a pure-CPU similarity loop without ever
   awaiting — long enough to miss the WebSocket keepalive, after which the peer
   drops the connection and every later query fails.
3. A stage that raised anything other than `TimeoutError` killed the pass, and a
   report of zeros is indistinguishable from "there was nothing to do".
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType


class _RecordingStorage:
    """Records every ``find_neurons`` call and serves a fixed neuron pool."""

    def __init__(self, neurons: list[Neuron] | None = None) -> None:
        self.current_brain_id = "default"
        self.brain_id = "default"
        self.calls: list[dict[str, Any]] = []
        self._neurons = neurons or []

    async def find_neurons(self, **kwargs: Any) -> list[Neuron]:
        self.calls.append(kwargs)
        wanted = kwargs.get("type")
        pool = [n for n in self._neurons if wanted is None or n.type is wanted]
        offset = int(kwargs.get("offset", 0))
        limit = int(kwargs.get("limit", 100))
        return pool[offset : offset + limit]

    async def get_synapses(self, **_kwargs: Any) -> list[Any]:
        return []


def _neuron(nid: str, ntype: NeuronType, vector: list[float] | None = None) -> Neuron:
    neuron = Neuron.create(content=f"content-{nid}", type=ntype)
    object.__setattr__(neuron, "id", nid)
    if vector is not None:
        neuron.metadata["_embedding"] = vector
    return neuron


class TestPassesDoNotFetchEmbeddings:
    """`include_embedding=True` is the default, so every reader must opt out."""

    @pytest.mark.asyncio
    async def test_dream_never_fetches_embeddings(self) -> None:
        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine.dream import dream

        storage = _RecordingStorage([_neuron(f"n{i}", NeuronType.CONCEPT) for i in range(3)])

        try:
            await dream(storage, BrainConfig(), seed=1)  # type: ignore[arg-type]
        except Exception:
            # Later steps need more of the engine than this stub provides; the
            # fetch is what this test pins down.
            pass

        assert storage.calls, "dream did not fetch neurons at all"
        for call in storage.calls:
            assert call.get("include_embedding") is False, (
                f"dream fetched embeddings ({call}) — it samples ids and spreads activation "
                "over the synapse graph; it never reads a vector"
            )

    @pytest.mark.asyncio
    async def test_interference_scans_never_fetch_embeddings(self) -> None:
        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine.interference import batch_interference_scan, detect_interference

        pool = [_neuron(f"n{i}", NeuronType.CONCEPT) for i in range(3)]
        for n in pool:
            n.metadata["tags"] = ["shared"]

        storage = _RecordingStorage(pool)
        probe = _neuron("probe", NeuronType.CONCEPT)
        probe.metadata["tags"] = ["shared"]

        config = BrainConfig()
        # Both scans return early unless interference detection is on.
        object.__setattr__(config, "interference_detection_enabled", True)

        try:
            await detect_interference(probe, storage, config)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            await batch_interference_scan(storage, config)  # type: ignore[arg-type]
        except Exception:
            pass

        assert storage.calls, "interference did not fetch neurons at all"
        for call in storage.calls:
            assert call.get("include_embedding") is False, (
                f"interference fetched embeddings ({call}) — it reads tags and simhashes only"
            )


class TestSemanticDiscoveryFetch:
    """The type filter belongs in the DB index, not in a client-side loop."""

    @pytest.mark.asyncio
    async def test_pages_per_type_with_a_small_page(self, monkeypatch: Any) -> None:
        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine import semantic_discovery as sd

        monkeypatch.setattr(sd, "_effective_embedding", lambda _c: (True, "stub", "stub"))
        pool = [
            _neuron("c1", NeuronType.CONCEPT, [1.0, 0.0]),
            _neuron("e1", NeuronType.ENTITY, [0.0, 1.0]),
            _neuron("a1", NeuronType.ACTION, [1.0, 1.0]),
        ]
        storage = _RecordingStorage(pool)

        await sd.discover_semantic_synapses(storage, BrainConfig())  # type: ignore[arg-type]

        requested = [c.get("type") for c in storage.calls]
        assert NeuronType.CONCEPT in requested and NeuronType.ENTITY in requested, (
            f"semantic discovery did not ask the DB for the two eligible types: {requested}"
        )
        assert all(t is not None for t in requested), (
            "a type-agnostic page is still there — that scans the whole neuron table "
            "(vectors included) to keep the CONCEPT/ENTITY minority"
        )
        assert all(int(c.get("limit", 0)) <= sd._EMBEDDING_PAGE_SIZE for c in storage.calls), (
            "these rows carry 1024-float vectors; the page must stay small"
        )

    @pytest.mark.asyncio
    async def test_similarity_pass_yields_to_the_event_loop(self, monkeypatch: Any) -> None:
        """A CPU pass that never awaits starves the connection's keepalive."""
        from surreal_memory.core.brain import BrainConfig
        from surreal_memory.engine import semantic_discovery as sd

        monkeypatch.setattr(sd, "_effective_embedding", lambda _c: (True, "stub", "stub"))
        monkeypatch.setattr(sd, "_YIELD_EVERY_ROWS", 2)
        pool = [_neuron(f"c{i}", NeuronType.CONCEPT, [1.0, float(i) / 100.0]) for i in range(12)]
        storage = _RecordingStorage(pool)

        ticks = 0

        async def _heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0)
                ticks += 1

        beat = asyncio.create_task(_heartbeat())
        try:
            await sd.discover_semantic_synapses(storage, BrainConfig())  # type: ignore[arg-type]
        finally:
            beat.cancel()

        assert ticks > 1, (
            "the similarity loop ran to completion without handing the event loop back — "
            "that is what lets the WebSocket keepalive lapse mid-pass"
        )


class TestLeastConnectedRotationSurvives:
    """Run 012's rotation must keep working now that the fetch is per-type.

    Selecting the least-connected neurons is what makes repeated passes attack a
    different part of the graph instead of resaturating the same slice, so it is
    load-bearing behaviour, not an optimisation.
    """

    @pytest.mark.asyncio
    async def test_selection_prefers_the_least_connected(self, monkeypatch: Any) -> None:
        from surreal_memory.engine import semantic_discovery as sd

        monkeypatch.setattr(sd, "MAX_NEURONS_TO_LINK", 2)

        eligible = [
            _neuron("busy", NeuronType.CONCEPT),
            _neuron("quiet", NeuronType.CONCEPT),
            _neuron("lonely", NeuronType.CONCEPT),
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        class _DegreeStorage(_RecordingStorage):
            async def get_synapse_degrees(self) -> dict[str, int]:
                return {"busy": 40, "quiet": 5, "lonely": 0}

        kept, kept_vectors = await sd._select_candidates(
            _DegreeStorage(),  # type: ignore[arg-type]
            eligible,
            vectors,
        )

        assert [n.id for n in kept] == ["lonely", "quiet"]
        assert kept_vectors == [[1.0, 1.0], [0.0, 1.0]], (
            "vectors must stay aligned with the neurons they belong to"
        )


class TestReportNamesWhatFailed:
    """Zeros from a broken pass must not read like zeros from an idle one."""

    @pytest.mark.asyncio
    async def test_a_failing_stage_does_not_stop_the_rest_and_is_reported(
        self, monkeypatch: Any
    ) -> None:
        from surreal_memory.engine.consolidation import (
            ConsolidationConfig,
            ConsolidationEngine,
            ConsolidationStrategy,
        )

        engine = ConsolidationEngine.__new__(ConsolidationEngine)
        engine._storage = _RecordingStorage()  # type: ignore[attr-defined]
        engine._config = ConsolidationConfig()  # type: ignore[attr-defined]
        engine._tier_config = None  # type: ignore[attr-defined]

        ran: list[str] = []

        async def _fake_run_strategy(
            strategy: ConsolidationStrategy, *_args: Any, **_kwargs: Any
        ) -> None:
            ran.append(strategy.value)
            if strategy is ConsolidationStrategy.PRUNE:
                raise ConnectionResetError(104, "Connection reset by peer")

        monkeypatch.setattr(engine, "_run_strategy", _fake_run_strategy, raising=False)

        report = await engine.run(
            strategies=[ConsolidationStrategy.PRUNE, ConsolidationStrategy.DREAM],
            dry_run=True,
        )

        assert "prune" in ran and "dream" in ran, (
            f"a raising stage aborted the pass; only {ran} ran"
        )
        failed = report.extra.get("failed_strategies")
        assert failed and any(f.startswith("prune") for f in failed), (
            f"the failure vanished from the report: {report.extra}"
        )
        assert "ConnectionResetError" in " ".join(failed)

        summary = report.summary()
        assert "Stages failed:" in summary, (
            "the summary printed only zeros — indistinguishable from an idle run"
        )
        assert "prune" in summary.split("Stages failed:")[1].splitlines()[0]

    def test_a_clean_report_stays_quiet(self) -> None:
        from surreal_memory.engine.consolidation import ConsolidationReport

        summary = ConsolidationReport().summary()

        assert "Stages failed:" not in summary
        assert "Stages timed out:" not in summary

    def test_timed_out_stages_are_surfaced_too(self) -> None:
        """`timed_out_strategies` was recorded but never printed."""
        from surreal_memory.engine.consolidation import ConsolidationReport

        report = ConsolidationReport()
        report.extra["timed_out_strategies"] = ["compress"]

        assert "Stages timed out: compress" in report.summary()
