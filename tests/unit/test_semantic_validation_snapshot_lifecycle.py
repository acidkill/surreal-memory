"""Validation proof loading accepts the actual pending-apply envelope size."""

from __future__ import annotations

import json

import pytest

from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.semantic_validation import SemanticSourceValidation
from tests.unit.test_consolidation_mid_strategies_resume import _engine
from tests.unit.test_semantic_discovery_cap_continuation import _semantic_link_fixture
from tests.unit.test_semantic_validation_integration import _FenceSession, _install_fence


@pytest.mark.asyncio
@pytest.mark.parametrize("context,count", [("x" * 100, 4000), ("ą" * 100, 2500)])
async def test_restore_finds_published_root_in_oversized_apply_manifest(
    monkeypatch: pytest.MonkeyPatch, context: str, count: int
) -> None:
    storage, progress, reference_time = _semantic_link_fixture(monkeypatch)
    _install_fence(storage, progress, _FenceSession("pending"))
    with pytest.raises(ConsolidationPausedError, match="source validation"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    checkpoint = progress.strategy_state("semantic_link")["pending"][0]
    reference = json.loads(checkpoint)["source_validation_ref"]
    root_key = (reference["state_id"], reference["revision"])
    root = storage.semantic_discovery_states[root_key]
    assert storage.snapshot_rows[root_key]["payload"] == root
    assert len(json.dumps(root).encode("utf-8")) < 1_000_000

    # The pending-apply envelope can exceed 1 MB while its proof root is small.
    synapse = {
        "id": "synapse:fixture",
        "source_id": "neuron:source",
        "target_id": "neuron:target",
        "type": "semantic",
        "weight": 0.5,
        "direction": "forward",
        "metadata": {"context": context},
        "reinforced_count": 0,
        "last_activated": None,
        "created_at": reference_time.isoformat(),
    }
    apply_manifest = json.dumps(
        {
            "kind": "semantic_link_synapses",
            "version": 4,
            "source_validation_ref": reference,
            "synapses": [dict(synapse, id=f"synapse:fixture-{index}") for index in range(count)],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert len(apply_manifest.encode("utf-8")) > 1_000_000
    if not context.isascii():
        assert len(apply_manifest) < 1_000_000
    progress.strategy_state("semantic_link")["pending"] = [apply_manifest]
    _install_fence(storage, progress, _FenceSession("verified"))

    validation = SemanticSourceValidation(
        storage,
        "0",
        reference_time=reference_time,
        run_id=progress.state["run_id"],
        owner_token=progress.owner_token,
        reference=reference,
    )
    await validation.restore()
    assert validation.session.watermark == root["watermark"]

    manifest = json.loads(apply_manifest)
    for field, invalid in (
        ("state_id", "other-state"),
        ("state_revision", 2),
        ("source_token", "other-source"),
    ):
        manifest["source_validation_ref"] = {**reference, field: invalid}
        progress.strategy_state("semantic_link")["pending"] = [
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        ]
        unauthorized = SemanticSourceValidation(
            storage,
            "0",
            reference_time=reference_time,
            run_id=progress.state["run_id"],
            owner_token=progress.owner_token,
            reference=reference,
        )
        with pytest.raises(RuntimeError, match="semantic source validation snapshot unavailable"):
            await unauthorized.restore()
