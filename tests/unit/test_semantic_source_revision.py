from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceChangedError,
    SemanticSourceFenceExpiredError,
    SemanticSourceFenceUnavailableError,
    SurrealDBSemanticSourceRevisionMixin,
)


class _RevisionStorage(SurrealDBSemanticSourceRevisionMixin):
    def __init__(self) -> None:
        self.current_brain_id = "brain-a"
        self.now = datetime(2026, 9, 24, 12, 0)
        self.change_rows: dict[str, list[dict[str, Any]]] = {
            "neuron": [],
            "synapse": [],
        }
        self.fail_table: str | None = None
        self.fail_message = "test transport failure"
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.marker_id: str | None = None
        self.marker_versionstamp = (100 << 16) + 9
        self.created_at_readonly = True
        self.date_cursor_empty = False

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def _query_response(self, sql: str, **params: Any) -> Any:
        if sql.startswith("INFO FOR TABLE "):
            readonly = " READONLY" if self.created_at_readonly else ""
            return {"fields": {"created_at": f"DEFINE FIELD created_at{readonly}"}}
        assert sql == "RETURN time::now()"
        return self.now

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if sql.startswith("DELETE semantic_source_barrier WHERE"):
            return []
        if sql.startswith("CREATE type::record('semantic_source_barrier'"):
            self.marker_id = str(params["token_id"])
            return []
        if sql.startswith("SHOW CHANGES FOR TABLE semantic_source_barrier"):
            assert self.marker_id is not None
            if self.date_cursor_empty and ' SINCE d"' in sql:
                return []
            return [
                {
                    "versionstamp": self.marker_versionstamp,
                    "changes": [{"update": {"id": f"semantic_source_barrier:{self.marker_id}"}}],
                }
            ]
        table = "neuron" if "neuron" in sql else "synapse"
        if self.fail_table == table:
            raise RuntimeError(self.fail_message)
        return self.change_rows[table]


def _token(
    *,
    brain_id: str = "brain-a",
    captured_at: datetime,
    versionstamp: int = 100,
    created_at_readonly: bool = False,
) -> str:
    data: dict[str, Any] = {
        "version": 1,
        "brain_id": brain_id,
        "versionstamp": versionstamp,
        "captured_at": captured_at.isoformat(timespec="microseconds") + "Z",
    }
    if created_at_readonly:
        data["created_at_readonly"] = True
    return json.dumps(data)


async def test_captures_server_time_and_queries_both_source_feeds() -> None:
    storage = _RevisionStorage()

    token = await storage.capture_semantic_source_token()
    payload = json.loads(token)
    assert payload["brain_id"] == "brain-a"
    assert payload["versionstamp"] == storage.marker_versionstamp
    assert payload["created_at_readonly"] is True
    assert datetime.fromisoformat(payload["captured_at"].replace("Z", "")) == storage.now
    assert any("CREATE type::record('semantic_source_barrier'" in sql for sql, _ in storage.queries)
    assert any("DELETE type::record('semantic_source_barrier'" in sql for sql, _ in storage.queries)
    barrier_queries = [
        sql
        for sql, _ in storage.queries
        if sql.startswith("SHOW CHANGES FOR TABLE semantic_source_barrier")
    ]
    assert barrier_queries[0].startswith(
        'SHOW CHANGES FOR TABLE semantic_source_barrier SINCE d"2026-09-24T11:59:59'
    )

    await storage.assert_semantic_source_unchanged(token)
    source_queries = [
        (sql, params)
        for sql, params in storage.queries
        if sql.startswith(("SHOW CHANGES FOR TABLE neuron", "SHOW CHANGES FOR TABLE synapse"))
    ]
    assert len(source_queries) == 2
    assert all(f"SINCE {storage.marker_versionstamp} LIMIT 10" in sql for sql, _ in source_queries)
    assert {sql.split("TABLE ", 1)[1].split()[0] for sql, _ in source_queries} == {
        "neuron",
        "synapse",
    }


@pytest.mark.asyncio
async def test_capture_does_not_claim_readonly_capability_without_both_schema_fields() -> None:
    storage = _RevisionStorage()
    storage.created_at_readonly = False

    token = await storage.capture_semantic_source_token()

    assert json.loads(token)["created_at_readonly"] is False


@pytest.mark.asyncio
async def test_capture_retries_raw_cursor_when_date_cursor_has_no_events() -> None:
    storage = _RevisionStorage()
    storage.date_cursor_empty = True

    token = await storage.capture_semantic_source_token()

    assert json.loads(token)["versionstamp"] == storage.marker_versionstamp
    barrier_queries = [
        sql
        for sql, _ in storage.queries
        if sql.startswith("SHOW CHANGES FOR TABLE semantic_source_barrier")
    ]
    assert ' SINCE d"' in barrier_queries[0]
    assert " SINCE 0 " in barrier_queries[1]


@pytest.mark.asyncio
async def test_any_source_table_mutation_invalidates_token() -> None:
    storage = _RevisionStorage()
    token = _token(captured_at=storage.now - timedelta(seconds=1))
    storage.change_rows["synapse"] = [{"versionstamp": 42, "changes": [{"delete": {}}]}]

    with pytest.raises(SemanticSourceChangedError, match="synapse changed"):
        await storage.assert_semantic_source_unchanged(token)


@pytest.mark.asyncio
async def test_expired_and_cross_brain_tokens_fail_closed() -> None:
    storage = _RevisionStorage()

    with pytest.raises(SemanticSourceFenceExpiredError, match="safe changefeed window"):
        await storage.assert_semantic_source_unchanged(
            _token(captured_at=storage.now - timedelta(days=6, hours=16))
        )
    storage.current_brain_id = "brain-b"
    with pytest.raises(SemanticSourceFenceExpiredError, match="malformed"):
        await storage.assert_semantic_source_unchanged(
            _token(captured_at=storage.now - timedelta(seconds=1))
        )


@pytest.mark.asyncio
async def test_changefeed_query_error_fails_closed() -> None:
    storage = _RevisionStorage()
    storage.fail_table = "neuron"
    token = _token(captured_at=storage.now - timedelta(seconds=1))

    with pytest.raises(SemanticSourceFenceUnavailableError, match="could not inspect neuron"):
        await storage.assert_semantic_source_unchanged(token)


@pytest.mark.asyncio
async def test_frozen_source_fence_ignores_full_rows_created_after_reference_time() -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    storage.change_rows["neuron"] = [
        {
            "versionstamp": 101,
            "changes": [
                {
                    "update": {
                        "id": "neuron:late-node",
                        "brain_id": "brain-a",
                        "created_at": cutoff + timedelta(seconds=1),
                        "updated_at": cutoff + timedelta(seconds=1),
                    }
                }
            ],
        }
    ]

    await storage.assert_semantic_source_unchanged(
        _token(
            captured_at=cutoff - timedelta(seconds=1),
            created_at_readonly=True,
        ),
        created_before=cutoff,
    )


def _table_reassertion(table: str) -> dict[str, Any]:
    return {
        "id": 25,
        "name": table,
        "changefeed": {"expiry": "1w", "original": False},
        "drop": False,
        "kind": (
            {"kind": "ANY"}
            if table == "neuron"
            else {"kind": "RELATION", "in": ["neuron"], "out": ["neuron"], "enforced": False}
        ),
        "permissions": {"create": False, "delete": False, "select": False, "update": False},
        "schemafull": table == "synapse",
    }


@pytest.mark.asyncio
async def test_frozen_fence_ignores_only_unchanged_source_table_reassertions() -> None:
    storage = _RevisionStorage()
    token = _token(captured_at=storage.now - timedelta(seconds=1), created_at_readonly=True)
    for table in ("neuron", "synapse"):
        storage.change_rows[table] = [
            {"versionstamp": 101, "changes": [{"define_table": _table_reassertion(table)}]}
        ]

    await storage.assert_semantic_source_unchanged(token, created_before=storage.now)


@pytest.mark.asyncio
async def test_frozen_fence_accepts_python_sdk_table_and_duration_values() -> None:
    try:
        from surrealdb.data.types.duration import Duration
        from surrealdb.data.types.table import Table
    except ImportError:
        pytest.skip("requires the optional SurrealDB Python SDK")

    storage = _RevisionStorage()
    definition = _table_reassertion("synapse")
    definition["changefeed"]["expiry"] = Duration.parse("7d")
    definition["kind"]["in"] = [Table("neuron")]
    definition["kind"]["out"] = [Table("neuron")]
    storage.change_rows["synapse"] = [
        {"versionstamp": 101, "changes": [{"define_table": definition}]}
    ]

    await storage.assert_semantic_source_unchanged(
        _token(captured_at=storage.now - timedelta(seconds=1), created_at_readonly=True),
        created_before=storage.now,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_field", ["schemafull", "permissions", "kind", "changefeed"])
async def test_frozen_fence_rejects_modified_source_table_definition(changed_field: str) -> None:
    storage = _RevisionStorage()
    definition = _table_reassertion("neuron")
    definition[changed_field] = {"unexpected": True}
    storage.change_rows["neuron"] = [
        {"versionstamp": 101, "changes": [{"define_table": definition}]}
    ]

    with pytest.raises(SemanticSourceFenceUnavailableError, match="unexpected table definition"):
        await storage.assert_semantic_source_unchanged(
            _token(captured_at=storage.now - timedelta(seconds=1), created_at_readonly=True),
            created_before=storage.now,
        )


@pytest.mark.asyncio
async def test_frozen_fence_checks_record_mutations_after_table_reassertion() -> None:
    storage = _RevisionStorage()
    storage.change_rows["neuron"] = [
        {
            "versionstamp": 101,
            "changes": [
                {"define_table": _table_reassertion("neuron")},
                {"update": {"id": "neuron:old", "brain_id": "brain-a", "created_at": storage.now}},
            ],
        }
    ]
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await storage.assert_semantic_source_unchanged(
            _token(captured_at=storage.now - timedelta(seconds=1), created_at_readonly=True),
            created_before=storage.now,
        )


@pytest.mark.asyncio
async def test_frozen_source_fence_rejects_pre_reference_update_and_unknown_delete() -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    token = _token(captured_at=cutoff - timedelta(seconds=1))
    storage.change_rows["neuron"] = [
        {
            "versionstamp": 101,
            "changes": [
                {
                    "update": {
                        "id": "neuron:old-node",
                        "brain_id": "brain-a",
                        # This is the exact unsafe shape: an old row was edited
                        # and its mutable created_at was rewritten past cutoff.
                        "created_at": cutoff + timedelta(seconds=1),
                        "updated_at": cutoff + timedelta(seconds=1),
                    }
                }
            ],
        }
    ]
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await storage.assert_semantic_source_unchanged(token, created_before=cutoff)

    storage.change_rows["neuron"] = [
        {"versionstamp": 102, "changes": [{"delete": {"id": "neuron:unknown"}}]}
    ]
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await storage.assert_semantic_source_unchanged(token, created_before=cutoff)


@pytest.mark.asyncio
async def test_frozen_source_fence_allows_delete_only_after_proving_post_reference_create() -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    token = _token(captured_at=cutoff - timedelta(seconds=1))
    storage.change_rows["synapse"] = [
        {
            "versionstamp": 101,
            "changes": [
                {
                    "create": {
                        "id": "synapse:late-edge",
                        "brain_id": "brain-a",
                        "created_at": cutoff + timedelta(seconds=1),
                    }
                }
            ],
        },
        {
            "versionstamp": 102,
            "changes": [
                {
                    "update": {
                        "id": "synapse:late-edge",
                        "brain_id": "brain-a",
                        # Once its post-reference create is proven, later
                        # edits to that out-of-scope row remain irrelevant.
                        "created_at": cutoff + timedelta(seconds=2),
                    }
                }
            ],
        },
        {"versionstamp": 103, "changes": [{"delete": {"id": "synapse:late-edge"}}]},
    ]

    await storage.assert_semantic_source_unchanged(token, created_before=cutoff)
