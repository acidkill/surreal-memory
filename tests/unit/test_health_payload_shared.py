"""The two health surfaces are serialized once, and the CLI renders what it gets.

``TestHealthPayloadExposesMaturationFields`` (test_maturation_rehearsal.py) pins
the MCP side of the fix. These pin the half that had no coverage at all:

* both surfaces go through one serializer, so a field added to the report cannot
  reach `smem_health` and miss `smem health --json` again — which is exactly how
  `stage_distribution`/`semantic_gate_blockers` (missing from both) and
  `top_penalties` (missing from the CLI only) diverged in the first place;
* `smem health` actually prints the maturation breakdown and the penalty
  remedies, rather than merely carrying them in a dict nobody displays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from surreal_memory.engine.diagnostics import (
    BrainHealthReport,
    _rank_penalty_factors,
    build_health_payload,
)

if TYPE_CHECKING:
    import pytest

BRAIN_NAME = "default"

_STAGE_DISTRIBUTION = {"stm": 40, "working": 12, "episodic": 30, "semantic": 8}
_GATE_BLOCKERS = {"time_gate": 18, "spacing_gate": 9, "ready": 3}

_COMPONENT_SCORES = {
    "connectivity": 0.4,
    "diversity": 0.7,
    "freshness": 0.9,
    "consolidation_ratio": 0.09,
    "orphan_rate": 0.2,
    "activation_efficiency": 0.5,
    "recall_confidence": 0.6,
}


def _make_report(
    *,
    stage_distribution: dict[str, int] | None = None,
    semantic_gate_blockers: dict[str, int] | None = None,
) -> BrainHealthReport:
    return BrainHealthReport(
        purity_score=61.2,
        grade="C",
        connectivity=0.4,
        diversity=0.7,
        freshness=0.9,
        consolidation_ratio=0.09,
        orphan_rate=0.2,
        activation_efficiency=0.5,
        recall_confidence=0.6,
        neuron_count=250,
        synapse_count=900,
        fiber_count=90,
        warnings=(),
        recommendations=("Recall memories by topic.",),
        top_penalties=_rank_penalty_factors(
            _COMPONENT_SCORES,
            metrics={"fiber_count": 90, "neuron_count": 250, "consolidation_ratio": 0.09},
        ),
        stage_distribution=stage_distribution,
        semantic_gate_blockers=semantic_gate_blockers,
    )


class _FakeBrain:
    id = "00313cb4-61ca-4e69-9784-e51431e99ad7"
    name = BRAIN_NAME


class _FakeStorage:
    def __init__(self) -> None:
        self.brain_id = BRAIN_NAME

    async def get_brain(self, _brain_id: str) -> _FakeBrain:
        return _FakeBrain()


def _patch_cli_health(monkeypatch: pytest.MonkeyPatch, report: BrainHealthReport) -> None:
    from surreal_memory.cli.commands import info
    from surreal_memory.engine import diagnostics as diagnostics_mod

    storage = _FakeStorage()

    async def fake_get_storage(_config: Any) -> _FakeStorage:
        return storage

    class _FakeEngine:
        def __init__(self, _storage: Any) -> None: ...

        async def analyze(self, _brain_id: str) -> BrainHealthReport:
            return report

    monkeypatch.setattr(info, "get_config", lambda: object())
    monkeypatch.setattr(info, "get_storage", fake_get_storage)
    monkeypatch.setattr(diagnostics_mod, "DiagnosticsEngine", _FakeEngine)


# ── The shared serializer ────────────────────────────────────────


class TestBuildHealthPayload:
    def test_carries_the_maturation_fields_and_the_penalty_remedies(self) -> None:
        payload = build_health_payload(
            _make_report(
                stage_distribution=_STAGE_DISTRIBUTION,
                semantic_gate_blockers=_GATE_BLOCKERS,
            ),
            brain=BRAIN_NAME,
        )

        assert payload["stage_distribution"] == _STAGE_DISTRIBUTION
        assert payload["semantic_gate_blockers"] == _GATE_BLOCKERS
        assert payload["brain"] == BRAIN_NAME
        assert all(p["action"] for p in payload["top_penalties"])

    def test_omits_rather_than_nulls_what_the_backend_cannot_report(self) -> None:
        payload = build_health_payload(_make_report(), brain=BRAIN_NAME)

        assert "stage_distribution" not in payload
        assert "semantic_gate_blockers" not in payload

    def test_copies_the_dicts(self) -> None:
        """Mutating the payload must not reach back into the report."""
        report = _make_report(
            stage_distribution=_STAGE_DISTRIBUTION,
            semantic_gate_blockers=_GATE_BLOCKERS,
        )
        payload = build_health_payload(report, brain=BRAIN_NAME)
        payload["stage_distribution"]["semantic"] = 999

        assert report.stage_distribution == _STAGE_DISTRIBUTION


class TestBothSurfacesShareOneShape:
    def test_cli_json_matches_the_mcp_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CLI's JSON is the MCP payload plus that tool's own extras.

        Pinning the relationship, not two hand-copied key lists: a field added
        to one and forgotten on the other is what this whole fix is about.
        """
        import asyncio

        from surreal_memory.cli.commands import info
        from surreal_memory.engine import diagnostics as diagnostics_mod
        from surreal_memory.mcp.stats_handler import StatsHandler

        report = _make_report(
            stage_distribution=_STAGE_DISTRIBUTION,
            semantic_gate_blockers=_GATE_BLOCKERS,
        )
        _patch_cli_health(monkeypatch, report)

        cli_result: dict[str, Any] = {}
        monkeypatch.setattr(info, "output_result", lambda result, _json: cli_result.update(result))
        info.health(json_output=True)

        class _FakeEngine:
            def __init__(self, _storage: Any) -> None: ...

            async def analyze(self, _brain_id: str) -> BrainHealthReport:
                return report

        monkeypatch.setattr(diagnostics_mod, "DiagnosticsEngine", _FakeEngine)
        handler = StatsHandler.__new__(StatsHandler)

        async def fake_get_storage() -> _FakeStorage:
            return _FakeStorage()

        handler.get_storage = fake_get_storage  # type: ignore[method-assign]
        mcp_result = asyncio.run(handler._health({}))

        # Everything the CLI reports, the MCP tool reports identically.
        assert cli_result == {k: v for k, v in mcp_result.items() if k in cli_result}
        # The MCP tool adds its own; the CLI adds none of its own.
        assert set(mcp_result) - set(cli_result) <= {"roadmap", "embedding", "gromov", "pro_hints"}
        assert cli_result["stage_distribution"] == _STAGE_DISTRIBUTION
        assert cli_result["semantic_gate_blockers"] == _GATE_BLOCKERS
        assert cli_result["top_penalties"]


# ── What `smem health` actually prints ───────────────────────────


class TestCliHealthRendered:
    def test_renders_the_maturation_section(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from surreal_memory.cli.commands import info

        _patch_cli_health(
            monkeypatch,
            _make_report(
                stage_distribution=_STAGE_DISTRIBUTION,
                semantic_gate_blockers=_GATE_BLOCKERS,
            ),
        )

        info.health(json_output=False)
        out = capsys.readouterr().out

        assert "Maturation:" in out
        assert "short-term: 40" in out
        assert "semantic: 8" in out
        # Spelled out, not "time: 18  spacing: 9  ready: 3".
        assert "18 waiting on dwell time" in out
        assert "9 waiting on recall spacing" in out
        assert "3 ready" in out
        # `ready` is the one bucket a command moves — and the only place the
        # rendered view is allowed to recommend consolidate for the gate.
        assert "smem consolidate` pass advances them" in out

    def test_renders_the_penalty_remedies(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from surreal_memory.cli.commands import info

        report = _make_report(
            stage_distribution=_STAGE_DISTRIBUTION,
            semantic_gate_blockers=_GATE_BLOCKERS,
        )
        _patch_cli_health(monkeypatch, report)

        info.health(json_output=False)
        out = capsys.readouterr().out

        assert "Biggest penalties:" in out
        for penalty in report.top_penalties:
            assert penalty.action in out

    def test_maturation_section_absent_when_the_backend_cannot_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from surreal_memory.cli.commands import info

        _patch_cli_health(monkeypatch, _make_report())

        info.health(json_output=False)
        out = capsys.readouterr().out

        assert "Maturation:" not in out
        # The rest of the view is unaffected.
        assert "Grade: C" in out
        assert "Biggest penalties:" in out

    def test_an_unknown_stage_is_not_dropped(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from surreal_memory.cli.commands import info

        _patch_cli_health(
            monkeypatch,
            _make_report(stage_distribution={"stm": 2, "hibernating": 7}),
        )

        info.health(json_output=False)
        out = capsys.readouterr().out

        assert "hibernating: 7" in out
