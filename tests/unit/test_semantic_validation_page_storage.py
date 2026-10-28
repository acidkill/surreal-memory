from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.semantic_discovery_state import (
    SurrealDBSemanticDiscoveryStateMixin,
)


class _PageStorage(SurrealDBSemanticDiscoveryStateMixin):
    def __init__(self) -> None:
        self.current_brain_id = "brain-a"
        self.rows: dict[str, dict[str, Any]] = {}
        self.progress: dict[str, Any] = {}
        self.lease: dict[str, Any] = {}
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if sql.startswith("CREATE type::record('semantic_discovery_state'"):
            record_id = str(params["record_id"])
            if record_id in self.rows:
                raise RuntimeError("record already exists")
            row = dict(params["record"])
            row["id"] = f"semantic_discovery_state:{record_id}"
            self.rows[record_id] = row
            return [row]
        if sql.startswith("SELECT * FROM semantic_discovery_state WITH INDEX"):
            return [
                row
                for row in self.rows.values()
                if row.get("brain_id") == params["brain_id"]
                and row.get("state_id") == params["state_id"]
                and row.get("revision") == params["revision"]
            ][:1]
        if sql.startswith("SELECT * FROM consolidation_progress"):
            return [self.progress] if self.progress else []
        if sql.startswith("SELECT * FROM consolidation_lease"):
            return [self.lease] if self.lease else []
        return []


def _canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _published_storage() -> tuple[_PageStorage, dict[str, Any]]:
    storage = _PageStorage()
    child_payload = {
        "brain_id": "brain-a",
        "run_id": "run-1",
        "owner_token": "owner-1",
        "source_token": "source-1",
        "kind": "semantic_source_validation_ids",
        "version": 1,
        "records": [["memory", "record-1"]],
    }
    child_ref = {
        "state_id": "ids-page-1",
        "revision": 1,
        "state_revision": 1,
        "source_token": "source-1",
        "sha256": _canonical_sha(child_payload),
    }
    root_payload = {
        "brain_id": "brain-a",
        "run_id": "run-1",
        "owner_token": "owner-1",
        "source_token": "source-1",
        "kind": "semantic_source_validation",
        "version": 1,
        "original_source_token": "source-1",
        "validation_token": "validation-1",
        "target_token": "target-1",
        "watermark": "watermark-1",
        "cutoff": "2026-10-28T00:00:00Z",
        "counters": {},
        "classifier_pages": [child_ref],
    }
    await storage.save_semantic_discovery_state("validation-root", 1, root_payload)
    await storage.save_semantic_discovery_state("ids-page-1", 1, child_payload)
    storage.lease = {"owner_token": "owner-1"}
    storage.progress = {
        "brain_id": "brain-a",
        "run_id": "run-1",
        "owner_token": "owner-1",
        "strategy_states": {
            "semantic_source_validation": {
                "state_id": "validation-root",
                "state_revision": 1,
                "source_token": "source-1",
            }
        },
    }
    parent_ref = {
        "state_id": "validation-root",
        "revision": 1,
        "state_revision": 1,
        "source_token": "source-1",
    }
    return storage, parent_ref


@pytest.mark.asyncio
async def test_load_trusted_child_page_requires_published_root_and_exact_hash() -> None:
    storage, parent_ref = await _published_storage()

    loaded = await storage._load_semantic_discovery_state_page(
        "ids-page-1", 1, parent_ref=parent_ref
    )

    assert loaded is not None
    assert loaded["records"] == [["memory", "record-1"]]
    page_queries = [
        (sql, params)
        for sql, params in storage.queries
        if sql.startswith("SELECT * FROM semantic_discovery_state WITH INDEX")
    ]
    assert page_queries[-1][1] == {
        "brain_id": "brain-a",
        "state_id": "ids-page-1",
        "revision": 1,
    }
    assert "idx_sds_state_revision" in page_queries[-1][0]


@pytest.mark.asyncio
async def test_load_child_page_rejects_unpublished_root_and_stale_lease() -> None:
    storage, parent_ref = await _published_storage()
    storage.progress["strategy_states"] = {}
    assert (
        await storage._load_semantic_discovery_state_page("ids-page-1", 1, parent_ref=parent_ref)
        is None
    )

    storage, parent_ref = await _published_storage()
    storage.lease["owner_token"] = "owner-rotated"  # noqa: S105 - synthetic lease owner
    assert (
        await storage._load_semantic_discovery_state_page("ids-page-1", 1, parent_ref=parent_ref)
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        ("source", None),
        ("run", None),
        ("hash", None),
        ("corrupt", None),
        ("oversized", None),
    ],
)
async def test_load_child_page_rejects_untrusted_page_content(tamper: str, expected: None) -> None:
    storage, parent_ref = await _published_storage()
    record_id = storage._record_id("ids-page-1", 1)
    row = storage.rows[record_id]
    if tamper == "source":
        row["source_token"] = "other-source"  # noqa: S105 - synthetic token
        row["payload"]["source_token"] = "other-source"  # noqa: S105 - synthetic token
    elif tamper == "run":
        row["run_id"] = "old-run"
        row["payload"]["run_id"] = "old-run"
    elif tamper == "hash":
        row["payload"]["records"] = [["memory", "changed"]]
    elif tamper == "corrupt":
        row["payload"] = "not-a-mapping"
    else:
        row["payload"] = {"x": "z" * 1_000_001}

    assert (
        await storage._load_semantic_discovery_state_page("ids-page-1", 1, parent_ref=parent_ref)
        is expected
    )
