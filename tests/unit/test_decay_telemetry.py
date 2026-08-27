"""Per-pass decay telemetry (#183).

Split out of the change-log work: collapsing superseded pending updates is
lossless for replication but discards the weight trajectory, which had no other
home because the only place recording it was the replication queue.

The shape here is deliberate. ONE aggregate row per pass, never a row per edge:
the decay pass touches every synapse, so per-edge rows would reintroduce exactly
the unbounded growth that made change_log a problem in the first place.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace

import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.lifecycle import DecayManager, DecayReport, _weight_bucket
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import DecayTelemetryConfig

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "surreal_memory"


@pytest_asyncio.fixture
async def storage() -> InMemoryStorage:
    store = InMemoryStorage()
    brain = Brain.create(name="decay-telemetry", config=BrainConfig())
    await store.save_brain(brain)
    store.set_brain(brain.id)
    yield store
    await store.close()


class TestOptIn:
    def test_disabled_by_default(self) -> None:
        """A row per pass is cheap; the default still has to be off.

        Telemetry nobody asked for is how a table starts growing unnoticed.
        """
        assert DecayTelemetryConfig().enabled is False

    def test_config_roundtrips(self) -> None:
        cfg = DecayTelemetryConfig(enabled=True, retention_days=7, max_records=5)
        assert DecayTelemetryConfig.from_dict(cfg.to_dict()) == cfg


class TestPerPassNotPerEdge:
    """The whole point of the aggregate shape."""

    def test_no_per_edge_write_in_the_decay_loop(self) -> None:
        import ast

        source = (SRC_ROOT / "engine" / "lifecycle.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        offenders = []
        for node in ast.walk(tree):
            # only the per-entity loops, not the whole module
            if not (isinstance(node, ast.For) and isinstance(node.target, ast.Name)):
                continue
            if node.target.id not in {"synapse", "state"}:
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "add_decay_pass"
                ):
                    offenders.append(inner.lineno)

        assert not offenders, (
            "telemetry must be written once per pass, not inside a per-entity "
            f"loop — the pass touches every synapse (line {offenders})"
        )

    async def test_one_row_per_pass(self, storage: InMemoryStorage, monkeypatch) -> None:
        _enable(monkeypatch)
        await _seed(storage)

        await DecayManager().apply_decay(storage)
        await DecayManager().apply_decay(storage)

        assert len(await storage.find_decay_passes()) == 2


class TestGatesAreExplained:
    """`processed` minus `decayed` must not be an unexplained number."""

    def test_report_separates_the_three_gates(self) -> None:
        report = DecayReport(
            synapses_processed=10,
            synapses_decayed=4,
            synapses_skipped_pinned=1,
            synapses_skipped_idle_gate=2,
            synapses_skipped_bookmark=3,
        )
        text = report.summary()
        assert "1 pinned" in text
        assert "2 not idle long enough" in text
        assert "3 already charged" in text

    def test_a_clean_pass_stays_quiet(self) -> None:
        report = DecayReport(synapses_processed=4, synapses_decayed=4)
        assert "skipped" not in report.summary()


class TestWeightDistribution:
    def test_buckets_split_near_zero(self) -> None:
        """Density near zero is where the prune threshold lives."""
        assert _weight_bucket(0.005) == "0-0.01"
        assert _weight_bucket(0.9) == "0.75-1"

    def test_records_both_sides_of_the_change(self) -> None:
        report = DecayReport()
        report.record_weights(0.9, 0.4)
        assert report.weight_before == {"0.75-1": 1}
        assert report.weight_after == {"0.25-0.5": 1}


class TestRetention:
    async def test_prune_respects_the_record_cap(self, storage: InMemoryStorage) -> None:
        for _ in range(5):
            await storage.add_decay_pass({"counters": {}})

        removed = await storage.prune_decay_passes(retention_days=365, max_records=2)

        assert removed == 3
        assert len(await storage.find_decay_passes()) == 2

    def test_consolidation_prunes_even_when_disabled(self) -> None:
        """Turning telemetry off must still clean up what it left behind."""
        source = (SRC_ROOT / "engine" / "consolidation.py").read_text(encoding="utf-8")
        assert "prune_decay_passes" in source
        block = source.split("prune_decay_passes", 1)[0].rsplit("if not dry_run", 1)[1]
        assert "enabled" not in block, (
            "the prune must not be gated on decay_telemetry.enabled — otherwise "
            "disabling telemetry strands its rows forever"
        )


class TestFailSoft:
    async def test_a_broken_telemetry_write_does_not_break_decay(
        self, storage: InMemoryStorage, monkeypatch
    ) -> None:
        """Telemetry that can break the pass it observes is worse than none."""
        _enable(monkeypatch)
        await _seed(storage)

        async def _boom(record):
            raise RuntimeError("storage is down")

        monkeypatch.setattr(storage, "add_decay_pass", _boom)

        report = await DecayManager().apply_decay(storage)

        assert report.synapses_processed >= 1, "decay itself must still have run"


def _enable(monkeypatch) -> None:
    from surreal_memory import unified_config

    cfg = unified_config.get_config()
    monkeypatch.setattr(
        unified_config,
        "get_config",
        lambda: replace(cfg, decay_telemetry=DecayTelemetryConfig(enabled=True)),
    )


async def _seed(storage: InMemoryStorage) -> None:
    a = Neuron.create(type=NeuronType.CONCEPT, content="a")
    b = Neuron.create(type=NeuronType.CONCEPT, content="b")
    await storage.add_neuron(a)
    await storage.add_neuron(b)
    await storage.add_synapse(
        Synapse.create(source_id=a.id, target_id=b.id, type=SynapseType.SIMILAR_TO, weight=0.9)
    )
