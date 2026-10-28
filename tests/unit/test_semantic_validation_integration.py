"""Offline engine regressions for durable semantic source validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from surreal_memory.cli.commands import tools
from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationLeaseLostError,
    ConsolidationPausedError,
)
from surreal_memory.storage.surrealdb.semantic_discovery_state import (
    SurrealDBSemanticDiscoveryStateMixin,
)
from tests.unit.test_consolidation_mid_strategies_resume import _engine
from tests.unit.test_semantic_discovery_cap_continuation import _semantic_link_fixture


class _FenceSession:
    def __init__(self, *statuses: str, on_pending: Callable[[], None] | None = None) -> None:
        self.statuses = list(statuses)
        self.on_pending = on_pending
        self.budget_checks = 0
        self.watermark = 100
        self.validation_token = "validation-token"  # noqa: S105 - synthetic token
        self.target_token: str | None = None
        self.scanned_database_rows = 0
        self.scanned_source_rows = {"neuron": 0, "synapse": 0}
        self.post_reference_ids = {
            "neuron": set[str](),
            "synapse": set[str](),
        }
        self.restored: list[tuple[int, set[str]]] = []

    async def verify(self, *, budget_check: Callable[[], Any] | None = None) -> str:
        self.restored.append((self.watermark, set(self.post_reference_ids["synapse"])))
        if budget_check is not None:
            self.budget_checks += 1
            if not await budget_check():
                return "pending"
        status = self.statuses.pop(0) if self.statuses else "verified"
        if status == "pending":
            self.watermark += 1
            self.scanned_database_rows += 1
            self.scanned_source_rows["synapse"] += 1
            self.target_token = "barrier-token"  # noqa: S105 - synthetic token
            self.post_reference_ids["neuron"].add("neuron:post-reference")
            self.post_reference_ids["synapse"].add("synapse:post-reference")
            if self.on_pending is not None:
                self.on_pending()
        return status


def _install_fence(
    storage: Any, progress: Any, session: _FenceSession
) -> tuple[list[str], list[str]]:
    reads: list[str] = []
    writes: list[str] = []
    storage.session = session
    storage._get_brain_id = lambda: storage.current_brain_id  # type: ignore[attr-defined]
    storage._manifest_references = SurrealDBSemanticDiscoveryStateMixin._manifest_references  # type: ignore[attr-defined]
    storage._contains_vector_field = SurrealDBSemanticDiscoveryStateMixin._contains_vector_field  # type: ignore[attr-defined]
    progress.state["owner_token"] = progress.owner_token
    storage.progress = progress.state
    storage.lease = {"owner_token": progress.owner_token}
    if not hasattr(storage, "snapshot_rows"):
        storage.snapshot_rows = {}

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        if sql.startswith("SELECT * FROM semantic_discovery_state WITH INDEX"):
            row = storage.snapshot_rows.get((params["state_id"], params["revision"]))
            return [row] if row is not None else []
        if sql.startswith("SELECT * FROM consolidation_progress"):
            return [storage.progress]
        if sql.startswith("SELECT * FROM consolidation_lease"):
            return [storage.lease]
        return []

    storage._query = query  # type: ignore[attr-defined]
    storage.load_semantic_discovery_state = (  # type: ignore[attr-defined]
        lambda state_id, revision: (
            SurrealDBSemanticDiscoveryStateMixin.load_semantic_discovery_state(
                storage, state_id, revision
            )
        )
    )
    storage._load_semantic_discovery_state_page = (  # type: ignore[attr-defined]
        lambda state_id, revision, **kwargs: (
            SurrealDBSemanticDiscoveryStateMixin._load_semantic_discovery_state_page(
                storage, state_id, revision, **kwargs
            )
        )
    )
    storage._create_semantic_source_fence_session = (  # type: ignore[attr-defined]
        lambda token, **_kwargs: (
            session
            if token == "0"  # noqa: S105 - synthetic token
            else pytest.fail("wrong source token")
        )
    )

    original_save = storage.save_semantic_discovery_state

    async def save_state(state_id: str, revision: int, payload: dict[str, Any]) -> None:
        await original_save(state_id, revision, payload)
        stored = storage.semantic_discovery_states[(state_id, revision)]
        storage.snapshot_rows[(state_id, revision)] = {
            "state_id": state_id,
            "revision": revision,
            "brain_id": stored["brain_id"],
            "run_id": stored["run_id"],
            "owner_token": stored["owner_token"],
            "source_token": stored["source_token"],
            "payload": stored,
        }

    storage.save_semantic_discovery_state = save_state
    original_checkpoint = progress.checkpoint

    async def sync_checkpoint(*args: Any, **kwargs: Any) -> None:
        await original_checkpoint(*args, **kwargs)
        storage.progress = progress.state

    progress.checkpoint = sync_checkpoint
    for method_name, event in (
        ("find_neurons_after_id", "neuron-page"),
        ("get_synapses_after_id", "synapse-page"),
        ("find_existing_synapse_pairs", "pair-lookup"),
        ("get_synapse", "get-synapse"),
    ):
        original = getattr(storage, method_name)

        async def tracked(
            *args: Any, _original: Any = original, _event: str = event, **kwargs: Any
        ) -> Any:
            reads.append(_event)
            return await _original(*args, **kwargs)

        setattr(storage, method_name, tracked)
    original_add = storage.add_synapse

    async def tracked_add(synapse: Any) -> Any:
        writes.append(str(synapse.id))
        return await original_add(synapse)

    storage.add_synapse = tracked_add
    storage.get_synapse_degrees = lambda **_kwargs: {}  # type: ignore[attr-defined]
    return reads, writes


def _prior_progress(progress: Any) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    completed = [
        strategy.value
        for strategy in ConsolidationStrategy
        if strategy not in {ConsolidationStrategy.ALL, ConsolidationStrategy.SEMANTIC_LINK}
    ]
    counters = {name: {"synthetic_count": index + 1} for index, name in enumerate(completed)}
    states = {
        name: {
            "phase": f"{name}_complete",
            "cursor": f"cursor-{name}",
            "counters": dict(counters[name]),
        }
        for name in completed
    }
    progress.state["completed_strategies"] = list(completed)
    progress.state["counters"] = json.loads(json.dumps(counters))
    progress.state["strategy_states"].update(json.loads(json.dumps(states)))
    progress.strategy_state("semantic_link")["counters"] = {
        "semantic_synapses_created": 7,
        "semantic_synapses_skipped": 3,
        "semantic_link_failures": 0,
    }
    return completed, counters, states


@pytest.mark.asyncio
async def test_discovery_validation_pauses_before_source_work_and_restores_after_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _ = _semantic_link_fixture(monkeypatch)
    session = _FenceSession("pending")
    reads, writes = _install_fence(storage, progress, session)
    completed, prior_counters, prior_states = _prior_progress(progress)
    old_link_counters = dict(progress.strategy_state("semantic_link")["counters"])

    with pytest.raises(ConsolidationPausedError, match="source validation"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )

    assert reads == []
    assert writes == []
    state = progress.strategy_state("semantic_link")
    checkpoint = json.loads(state["pending"][0])
    assert checkpoint["kind"] == "semantic_link_discovery"
    assert checkpoint["version"] == 5
    assert checkpoint["stage"] == "validation"
    proof_ref = checkpoint["source_validation_ref"]
    assert storage.semantic_discovery_states[(proof_ref["state_id"], proof_ref["revision"])]
    assert progress.state["completed_strategies"] == completed
    assert progress.state["counters"] == prior_counters
    assert {name: progress.strategy_state(name) for name in completed} == prior_states
    assert {key: state["counters"][key] for key in old_link_counters} == old_link_counters

    # A fresh process-local fence session must restore its cursor and ID registry.
    resumed_session = _FenceSession("verified")
    _install_fence(storage, progress, resumed_session)
    try:
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    except ConsolidationPausedError:
        pass  # A bounded discovery batch is a normal resumable outcome.
    assert resumed_session.restored[0] == (101, {"synapse:post-reference"})
    assert resumed_session.budget_checks == len(resumed_session.restored)
    assert "neuron-page" in reads
    continuation = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    assert continuation["version"] == 5
    assert continuation["source_validation_ref"]
    assert progress.state["completed_strategies"] == completed
    assert progress.state["counters"] == prior_counters
    assert {name: progress.strategy_state(name) for name in completed} == prior_states


@pytest.mark.asyncio
async def test_pending_apply_uses_same_validation_gate_without_rewriting_committed_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _ = _semantic_link_fixture(monkeypatch, pair_cap=20)
    session = _FenceSession()
    reads, writes = _install_fence(storage, progress, session)
    completed, prior_counters, prior_states = _prior_progress(progress)
    progress.pause_phase = "semantic_link_apply"

    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    already_written = list(writes)
    assert already_written
    manifest = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    assert manifest["kind"] == "semantic_link_synapses"
    assert manifest["version"] == 4
    assert manifest["source_validation_ref"]
    apply_cursor = progress.strategy_state("semantic_link")["cursor"]
    assert progress.strategy_state("semantic_link")["counters"][
        "semantic_synapses_created"
    ] == 7 + len(already_written)

    resumed_session = _FenceSession("pending")
    _install_fence(storage, progress, resumed_session)
    reads.clear()
    with pytest.raises(ConsolidationPausedError, match="source validation"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert reads == []
    assert writes == already_written
    assert progress.state["completed_strategies"] == completed
    assert progress.state["counters"] == prior_counters
    assert {name: progress.strategy_state(name) for name in completed} == prior_states
    assert progress.strategy_state("semantic_link")["cursor"] == apply_cursor
    assert progress.strategy_state("semantic_link")["counters"][
        "semantic_synapses_created"
    ] == 7 + len(already_written)

    _install_fence(storage, progress, _FenceSession("verified"))
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )
    assert writes == already_written
    assert len(storage.synapses) == len(set(writes))
    assert progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete"
    assert {name: progress.strategy_state(name) for name in completed} == prior_states


@pytest.mark.asyncio
async def test_restart_rejects_tampered_rooted_classifier_page_before_restoring_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _ = _semantic_link_fixture(monkeypatch)
    reads, writes = _install_fence(storage, progress, _FenceSession("pending"))
    with pytest.raises(ConsolidationPausedError, match="source validation"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    checkpoint = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    proof_ref = checkpoint["source_validation_ref"]
    root = storage.semantic_discovery_states[(proof_ref["state_id"], proof_ref["revision"])]
    page_ref = root["classifier_pages"][0]
    page_row = storage.snapshot_rows[(page_ref["state_id"], page_ref["revision"])]
    page_row["payload"]["records"].append(["synapse", "synapse:tampered"])

    session = _FenceSession("verified")
    _install_fence(storage, progress, session)
    with pytest.raises(RuntimeError, match="page hash mismatch"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert session.watermark == 100
    assert session.post_reference_ids == {"neuron": set(), "synapse": set()}
    assert reads == []
    assert writes == []


@pytest.mark.asyncio
async def test_lease_loss_and_unconfirmed_validation_page_never_publish_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, progress, _ = _semantic_link_fixture(monkeypatch)
    reads, writes = _install_fence(storage, progress, _FenceSession("pending"))
    original_save = storage.save_semantic_discovery_state

    async def fail_validation_page(state_id: str, revision: int, payload: dict[str, Any]) -> None:
        if payload.get("kind") == "semantic_source_validation_ids":
            raise RuntimeError("validation snapshot write not confirmed")
        await original_save(state_id, revision, payload)

    storage.save_semantic_discovery_state = fail_validation_page
    with pytest.raises(RuntimeError, match="validation snapshot write not confirmed"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert reads == []
    assert writes == []
    assert not any(
        payload.get("kind") == "semantic_source_validation"
        for payload in storage.semantic_discovery_states.values()
    )
    storage.save_semantic_discovery_state = original_save

    storage, progress, _ = _semantic_link_fixture(monkeypatch)
    progress.brain_id = "brain-1"
    progress.lease_lost = False
    session = _FenceSession("pending", on_pending=lambda: setattr(progress, "lease_lost", True))
    _install_fence(storage, progress, session)
    original_checkpoint = progress.checkpoint

    async def lease_fenced_checkpoint(*args: Any, **kwargs: Any) -> None:
        if progress.lease_lost:
            raise ConsolidationLeaseLostError("synthetic lease loss before proof publication")
        await original_checkpoint(*args, **kwargs)

    progress.checkpoint = lease_fenced_checkpoint
    with pytest.raises(ConsolidationLeaseLostError, match="before proof publication"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert any(
        payload.get("kind") == "semantic_source_validation"
        for payload in storage.semantic_discovery_states.values()
    )  # immutable proof bytes may remain orphaned when the lease fence rejects publication
    assert not any(
        json.loads(item).get("source_validation_ref")
        for item in progress.strategy_state("semantic_link").get("pending", [])
    )


def test_paused_validation_report_stays_human_readable_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    delta = MagicMock()
    delta.summary.return_value = (
        "Consolidation paused: semantic_link source validation cursor=101. "
        "The next smem consolidate will resume this checkpoint."
    )
    delta.report.extra = {
        "consolidation_status": "paused",
        "last_checkpoint": "semantic_link source validation cursor=101",
    }

    with (
        patch.object(tools, "get_config", return_value=MagicMock()),
        patch.object(tools, "resolve_brain", return_value="test-brain"),
        patch.object(tools, "get_storage", new_callable=AsyncMock, return_value=MagicMock()),
        patch.object(tools, "run_async", side_effect=lambda awaitable: asyncio.run(awaitable)),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ),
    ):
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate(strategy="semantic_link")

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert exc_info.value.exit_code == 1
    assert "last checkpoint" in output.lower() or "cursor=101" in output
    assert "resume this checkpoint" in output
    assert "Traceback" not in output
