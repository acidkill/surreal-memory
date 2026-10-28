"""Durable source-validation progress is separate from consolidation progress."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any

import pytest

from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.semantic_validation import SemanticSourceValidation

_ORIGINAL = "original"
_BARRIER = "barrier"
_OWNER = "owner-a"


class Session:
    def __init__(self) -> None:
        self.token = _ORIGINAL
        self.validation_token = _ORIGINAL
        self.target_token: str | None = None
        self.watermark = 100
        self.scanned_database_rows = 0
        self.scanned_source_rows = {"neuron": 0, "synapse": 0}
        self.post_reference_ids: dict[str, set[str]] = {"neuron": set(), "synapse": set()}
        self.status = "pending"

    async def verify(self) -> str:
        self.watermark += 1
        self.scanned_database_rows += 1
        self.scanned_source_rows["synapse"] += 1
        self.target_token = _BARRIER
        self.post_reference_ids["synapse"].add("synapse:after-reference")
        return self.status


class Storage:
    current_brain_id = "brain-a"

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        self.session = Session()
        self.published: dict[str, Any] | None = None
        self.fail_save = False

    def _create_semantic_source_fence_session(self, token: str, **kwargs: Any) -> Session:
        assert token == _ORIGINAL
        return self.session

    async def save_semantic_discovery_state(
        self, state_id: str, revision: int, payload: dict[str, Any]
    ) -> None:
        if self.fail_save:
            raise RuntimeError("write not confirmed")
        self.rows[(state_id, revision)] = copy.deepcopy(payload)

    async def load_semantic_discovery_state(self, state_id: str, revision: int) -> Any:
        assert self.published is not None
        assert self.published["state_id"] == state_id
        return copy.deepcopy(self.rows.get((state_id, revision)))

    async def _load_semantic_discovery_state_page(
        self, state_id: str, revision: int, **kwargs: Any
    ) -> Any:
        assert kwargs["parent_ref"] == self.published
        return copy.deepcopy(self.rows.get((state_id, revision)))


def validation(storage: Storage, **kwargs: Any) -> SemanticSourceValidation:
    return SemanticSourceValidation(
        storage,
        "original",
        reference_time=datetime(2026, 9, 24, 12),
        run_id="run-a",
        owner_token=_OWNER,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pending_publishes_complete_proof_before_normal_pause_and_restores() -> None:
    storage = Storage()
    proof = validation(storage)

    async def publish(ref: dict[str, Any], counters: dict[str, int]) -> None:
        storage.published = ref
        assert counters["validation_database_events"] == 1

    with pytest.raises(ConsolidationPausedError, match="source validation"):
        await proof.assert_unchanged(publish)
    assert storage.published is not None
    storage.session = Session()
    restored = validation(storage, reference=storage.published)
    await restored.restore()
    assert storage.session.watermark == 101
    assert storage.session.target_token == _BARRIER
    assert storage.session.post_reference_ids["synapse"] == {"synapse:after-reference"}


@pytest.mark.asyncio
async def test_unconfirmed_page_does_not_publish_or_allow_work() -> None:
    storage = Storage()
    storage.fail_save = True
    proof = validation(storage)

    async def publish(ref: dict[str, Any], counters: dict[str, int]) -> None:
        pytest.fail("unconfirmed snapshots cannot be published")

    with pytest.raises(RuntimeError, match="not confirmed"):
        await proof.assert_unchanged(publish)
    assert storage.published is None


@pytest.mark.asyncio
async def test_registry_pages_are_bounded_and_checked_after_restart(monkeypatch: Any) -> None:
    import surreal_memory.engine.semantic_validation as module

    monkeypatch.setattr(module, "_IDS_PER_PAGE", 2)
    storage = Storage()
    storage.session.post_reference_ids["neuron"] = {f"neuron:{i}" for i in range(7)}
    proof = validation(storage)
    ref = await proof.snapshot()
    assert ref is not None
    storage.published = ref
    root = storage.rows[(ref["state_id"], ref["revision"])]
    assert len(root["classifier_pages"]) == 4
    page_ref = root["classifier_pages"][0]
    storage.rows[(page_ref["state_id"], page_ref["revision"])]["records"].append(
        ["neuron", "neuron:tampered"]
    )
    storage.session = Session()
    with pytest.raises(RuntimeError, match="hash"):
        await validation(storage, reference=ref).restore()
    assert storage.session.watermark == 100


@pytest.mark.asyncio
async def test_scope_change_rejects_proof_before_restoring_cursor() -> None:
    storage = Storage()
    proof = validation(storage)
    ref = await proof.snapshot()
    assert ref is not None
    storage.published = ref
    root = storage.rows[(ref["state_id"], ref["revision"])]
    root["reference_time"] = "2026-09-25T12:00:00.000000"
    ref["sha256"] = hashlib.sha256(
        json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="scope"):
        await validation(storage, reference=ref).restore()
