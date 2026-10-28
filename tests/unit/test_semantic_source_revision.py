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
        self.database_events: list[dict[str, Any]] = []

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def _query_response(self, sql: str, **params: Any) -> Any:
        if sql.startswith("INFO FOR TABLE "):
            readonly = " READONLY" if self.created_at_readonly else ""
            return {"fields": {"created_at": f"DEFINE FIELD created_at{readonly}"}}
        assert sql == "RETURN time::now()"
        return self.now

    async def _show_table_changes(
        self, table: str, since: int | str, limit: int
    ) -> list[dict[str, Any]]:
        # This fake models each table's feed directly; the database-feed paging
        # itself is covered against a live SurrealDB in the integration tests.
        return await self._query(f"SHOW CHANGES FOR TABLE {table} SINCE {since} LIMIT {limit}")

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if sql.startswith("DELETE semantic_source_barrier WHERE"):
            return []
        if sql.startswith("CREATE type::record('semantic_source_barrier'"):
            self.marker_id = str(params["token_id"])
            source_events = [event for rows in self.change_rows.values() for event in rows]
            grouped: dict[int, list[Any]] = {}
            for event in source_events:
                grouped.setdefault(event["versionstamp"], []).extend(event.get("changes", []))
            self.database_events = [
                _feed_event(stamp, {"create": {"id": f"test:noise-{stamp}"}})
                for stamp in sorted(grouped)
            ]
            for target, source in zip(self.database_events, sorted(grouped), strict=True):
                target["changes"] = grouped[source]
            self.database_events.append(
                _feed_event(
                    self.marker_versionstamp,
                    {"create": {"id": f"semantic_source_barrier:{self.marker_id}"}},
                )
            )
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
        if sql.startswith("SHOW CHANGES FOR DATABASE"):
            cursor_part, raw_limit = sql.rsplit(" LIMIT ", 1)
            cursor = int(cursor_part.rsplit(" ", 1)[1])
            return [event for event in self.database_events if event["versionstamp"] >= cursor][
                : int(raw_limit)
            ]
        table = "neuron" if "neuron" in sql else "synapse"
        if self.fail_table == table:
            raise RuntimeError(self.fail_message)
        return self.change_rows[table]


class _DatabaseFeedSessionStorage(SurrealDBSemanticSourceRevisionMixin):
    """Synthetic inclusive database-feed source for fence-session tests."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.current_brain_id = "brain-a"
        self.now = datetime(2026, 9, 24, 12, 0)
        self.events = list(events)
        self.queries: list[tuple[int, int]] = []

    def _get_brain_id(self) -> str:
        return self.current_brain_id

    async def _server_now(self) -> datetime:
        return self.now

    async def capture_semantic_source_token(self) -> str:
        versionstamp = max((event["versionstamp"] for event in self.events), default=100) + 1
        self.events.append(
            _feed_event(versionstamp, {"create": {"id": f"semantic_source_barrier:{versionstamp}"}})
        )
        return _token(captured_at=self.now, versionstamp=versionstamp)

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        assert sql.startswith("SHOW CHANGES FOR DATABASE SINCE ")
        cursor, raw_limit = sql.rsplit(" LIMIT ", 1)
        since = int(cursor.rsplit(" ", 1)[1])
        limit = int(raw_limit)
        self.queries.append((since, limit))
        return [event for event in self.events if event["versionstamp"] >= since][:limit]


def _feed_event(versionstamp: int, change: dict[str, Any]) -> dict[str, Any]:
    return {"versionstamp": versionstamp, "changes": [change]}


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "current", "reverse_patch"),
    [
        (
            "neuron",
            {
                "id": "neuron:recall",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "updated_at": datetime(2026, 9, 24, 12, 0),
                "access_frequency": 4,
                "type": "concept",
                "content": "stable content",
                "embedding_vec": [0.1, 0.2],
                "metadata": {"_embedding": [0.1, 0.2], "topic": "stable"},
            },
            [
                {"op": "replace", "path": "/updated_at", "value": "2026-09-24T11:59:00Z"},
                {"op": "replace", "path": "/access_frequency", "value": 3},
                {"op": "replace", "path": "/metadata/last_recalled", "value": None},
            ],
        ),
        (
            "neuron",
            {
                "id": "neuron:embedding",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "updated_at": datetime(2026, 9, 24, 12, 0),
                "metadata": {"_embedding": [0.1, 0.2], "recall_count": 4},
            },
            [
                {
                    "op": "replace",
                    "path": "/metadata",
                    "value": {"_embedding": [0.1, 0.2], "recall_count": 3},
                }
            ],
        ),
        (
            "neuron",
            {
                "id": "neuron:metadata-without-embedding",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "updated_at": datetime(2026, 9, 24, 12, 0),
                "metadata": {"recall_count": 4},
            },
            [{"op": "remove", "path": "/metadata"}],
        ),
        (
            "synapse",
            {
                "id": "synapse:recall-edge",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "in": "neuron:recall",
                "out": "neuron:target",
                "type": "related_to",
                "weight": 0.75,
                "last_activated": datetime(2026, 9, 24, 12, 0),
                "reinforced_count": 4,
                "metadata": {"seen": True},
            },
            [
                {"op": "replace", "path": "/weight", "value": 0.5},
                {"op": "replace", "path": "/last_activated", "value": None},
                {"op": "replace", "path": "/reinforced_count", "value": 3},
                {"op": "replace", "path": "/metadata/seen", "value": False},
            ],
        ),
    ],
)
async def test_frozen_fence_accepts_actual_reverse_diff_for_projection_neutral_updates(
    table: str, current: dict[str, Any], reverse_patch: list[dict[str, Any]]
) -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    storage.change_rows[table] = [
        {
            "versionstamp": 101,
            "changes": [{"current": current, "update": reverse_patch}],
        }
    ]

    await storage.assert_semantic_source_unchanged(
        _token(captured_at=cutoff - timedelta(seconds=1), created_at_readonly=True),
        created_before=cutoff,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table", "current", "reverse_patch"),
    [
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "content": "after",
            },
            [{"op": "replace", "path": "/content", "value": "before"}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "type": "fact",
            },
            [{"op": "replace", "path": "/type", "value": "concept"}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "embedding_vec": [0.3],
            },
            [{"op": "replace", "path": "/embedding_vec", "value": [0.1]}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "metadata": {"_embedding": [0.3]},
            },
            [{"op": "replace", "path": "/metadata/_embedding", "value": [0.1]}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "metadata": {"_embedding": [0.3], "recall_count": 4},
            },
            [
                {
                    "op": "replace",
                    "path": "/metadata",
                    "value": {"_embedding": [0.1], "recall_count": 3},
                }
            ],
        ),
        (
            "neuron",
            {
                "id": "neuron:embedding-remove",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "metadata": {"_embedding": [0.3]},
            },
            [{"op": "remove", "path": "/metadata"}],
        ),
        (
            "neuron",
            {
                "id": "neuron:new-id",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
            },
            [{"op": "replace", "path": "/id", "value": "neuron:old-id"}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "updated_at": datetime(2026, 9, 24),
            },
            [{"op": "replace", "path": "", "value": {"id": "neuron:n"}}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "updated_at": datetime(2026, 9, 24),
            },
            [{"op": "move", "path": "/updated_at", "from": "/access_frequency"}],
        ),
        (
            "neuron",
            {
                "id": "neuron:n",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "updated_at": datetime(2026, 9, 24),
            },
            [{"op": "replace", "path": "/mystery", "value": 1}],
        ),
        (
            "synapse",
            {
                "id": "synapse:e",
                "brain_id": "brain-a",
                "created_at": datetime(2026, 9, 20),
                "in": "neuron:a",
                "out": "neuron:b",
                "type": "related_to",
            },
            [{"op": "replace", "path": "/in", "value": "neuron:other"}],
        ),
    ],
)
async def test_frozen_fence_rejects_semantic_identity_and_malformed_reverse_diffs(
    table: str, current: dict[str, Any], reverse_patch: list[dict[str, Any]]
) -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    storage.change_rows[table] = [
        {
            "versionstamp": 101,
            "changes": [{"current": current, "update": reverse_patch}],
        }
    ]

    with pytest.raises((SemanticSourceChangedError, SemanticSourceFenceUnavailableError)):
        await storage.assert_semantic_source_unchanged(
            _token(captured_at=cutoff - timedelta(seconds=1), created_at_readonly=True),
            created_before=cutoff,
        )


def test_surreal_text_change_reverse_diffs_follow_the_projection() -> None:
    classify = SurrealDBSemanticSourceRevisionMixin._projection_neutral_update
    diff = "@@ -1,5 +1,6 @@\n tobi\n-e\n+as\n"
    assert classify("neuron", {}, [{"op": "change", "path": "/content", "value": diff}]) is False
    assert (
        classify("neuron", {}, [{"op": "change", "path": "/metadata/context", "value": diff}])
        is True
    )
    assert classify("neuron", {}, [{"op": "change", "path": "/content", "value": 1}]) is None


@pytest.mark.asyncio
async def test_frozen_fence_rejects_invalid_unknown_missing_and_cross_brain_reverse_diffs() -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    base = {
        "id": "neuron:n",
        "brain_id": "brain-a",
        "created_at": datetime(2026, 9, 20),
        "updated_at": cutoff,
    }
    malformed = [
        {"current": {**base, "id": "invalid-id"}, "update": []},
        {
            "current": {**base, "brain_id": "brain-b"},
            "update": [
                {"op": "replace", "path": "/updated_at", "value": cutoff - timedelta(seconds=1)}
            ],
        },
        {"current": base, "update": [{"op": "replace", "path": "/updated_at"}]},
        {"current": base},
    ]
    token = _token(captured_at=cutoff - timedelta(seconds=1), created_at_readonly=True)
    for change in malformed:
        storage.change_rows["neuron"] = [{"versionstamp": 101, "changes": [change]}]
        with pytest.raises((SemanticSourceChangedError, SemanticSourceFenceUnavailableError)):
            await storage.assert_semantic_source_unchanged(token, created_before=cutoff)


@pytest.mark.asyncio
async def test_empty_reverse_diff_proves_no_change_only_with_frozen_cutoff() -> None:
    storage = _RevisionStorage()
    token = _token(captured_at=storage.now - timedelta(seconds=1), created_at_readonly=True)
    storage.change_rows["neuron"] = [
        {
            "versionstamp": 101,
            "changes": [
                {
                    "current": {
                        "id": "neuron:n",
                        "brain_id": "brain-a",
                        "created_at": storage.now - timedelta(days=1),
                    },
                    "update": [],
                }
            ],
        }
    ]
    await storage.assert_semantic_source_unchanged(token, created_before=storage.now)
    with pytest.raises(SemanticSourceChangedError):
        await storage.assert_semantic_source_unchanged(token)


@pytest.mark.asyncio
async def test_reverse_diff_is_strict_without_frozen_cutoff_and_plain_updates_stay_strict() -> None:
    storage = _RevisionStorage()
    cutoff = storage.now
    token = _token(captured_at=cutoff - timedelta(seconds=1), created_at_readonly=True)
    storage.change_rows["neuron"] = [
        {
            "versionstamp": 101,
            "changes": [
                {
                    "current": {
                        "id": "neuron:n",
                        "brain_id": "brain-a",
                        "created_at": cutoff - timedelta(days=1),
                        "updated_at": cutoff,
                        "content": "same",
                    },
                    "update": [
                        {
                            "op": "replace",
                            "path": "/updated_at",
                            "value": cutoff - timedelta(seconds=1),
                        }
                    ],
                }
            ],
        }
    ]
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await storage.assert_semantic_source_unchanged(token)

    storage.change_rows["neuron"] = [
        {
            "versionstamp": 102,
            "changes": [
                {
                    "update": {
                        "id": "neuron:n",
                        "brain_id": "brain-a",
                        "created_at": cutoff + timedelta(seconds=1),
                        "updated_at": cutoff + timedelta(seconds=1),
                    }
                }
            ],
        }
    ]
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await storage.assert_semantic_source_unchanged(
            _token(captured_at=cutoff - timedelta(seconds=1), created_at_readonly=False),
            created_before=cutoff,
        )


@pytest.mark.asyncio
async def test_change_table_attributes_actual_reverse_diff_under_database_feed_noise() -> None:
    storage = _RevisionStorage()
    event = {
        "versionstamp": 17,
        "changes": [
            {"current": {"id": "neuron:changed"}, "update": []},
            {"create": {"id": "fence_noise:1"}},
        ],
    }

    assert SurrealDBSemanticSourceRevisionMixin._change_table(event["changes"][0]) == "neuron"

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        return [event]

    storage._query = query  # type: ignore[method-assign]
    assert await SurrealDBSemanticSourceRevisionMixin._show_table_changes(
        storage, "neuron", 0, 1
    ) == [{**event, "changes": [event["changes"][0]]}]


def _table_reassertion(table: str, *, original: bool = False) -> dict[str, Any]:
    return {
        "id": 25,
        "name": table,
        "changefeed": {"expiry": "1w", "original": original},
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
@pytest.mark.parametrize("original", [False, True])
async def test_frozen_fence_ignores_only_unchanged_source_table_reassertions(
    original: bool,
) -> None:
    storage = _RevisionStorage()
    token = _token(captured_at=storage.now - timedelta(seconds=1), created_at_readonly=True)
    for table in ("neuron", "synapse"):
        storage.change_rows[table] = [
            {
                "versionstamp": 101,
                "changes": [{"define_table": _table_reassertion(table, original=original)}],
            }
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


@pytest.mark.asyncio
async def test_database_feed_pages_past_other_tables_and_skips_inclusive_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 2)
    storage = _RevisionStorage()
    pages = [
        [
            {"versionstamp": 1, "changes": [{"create": {"id": "noise:1"}}]},
            {"versionstamp": 2, "changes": [{"create": {"id": "noise:2"}}]},
        ],
        [
            {"versionstamp": 2, "changes": [{"create": {"id": "noise:2"}}]},
            {"versionstamp": 3, "changes": [{"create": {"id": "neuron:1"}}]},
        ],
        [
            {"versionstamp": 3, "changes": [{"create": {"id": "neuron:1"}}]},
            {"versionstamp": 4, "changes": [{"create": {"id": "neuron:2"}}]},
        ],
    ]

    async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
        storage.queries.append((sql, params))
        return pages.pop(0)

    monkeypatch.setattr(storage, "_query", query)

    events = await SurrealDBSemanticSourceRevisionMixin._show_table_changes(storage, "neuron", 0, 2)

    assert [event["versionstamp"] for event in events] == [3, 4]
    assert [sql for sql, _ in storage.queries] == [
        "SHOW CHANGES FOR DATABASE SINCE 0 LIMIT 2",
        "SHOW CHANGES FOR DATABASE SINCE 2 LIMIT 2",
        "SHOW CHANGES FOR DATABASE SINCE 3 LIMIT 2",
    ]


@pytest.mark.asyncio
async def test_fence_session_advances_one_global_watermark_instead_of_replaying_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 4)
    cutoff = datetime(2026, 9, 24, 12, 0)
    events = [
        _feed_event(100, {"create": {"id": "semantic_source_barrier:token"}}),
        *[
            _feed_event(stamp, {"create": {"id": f"semantic_source_barrier:noise-{stamp}"}})
            for stamp in range(101, 107)
        ],
    ]
    storage = _DatabaseFeedSessionStorage(events)
    token = _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100)
    session = storage._create_semantic_source_fence_session(token, created_before=cutoff)

    await session.assert_unchanged()
    calls_after_full_validation = len(storage.queries)
    assert calls_after_full_validation > 1

    await session.assert_unchanged()

    assert len(storage.queries) - calls_after_full_validation == 1
    assert storage.queries[-1] == (107, 4)

    # A new process/session must revalidate from its own token; no cursor is
    # persisted or borrowed from the previous invocation.
    fresh_session = storage._create_semantic_source_fence_session(token, created_before=cutoff)
    first_fresh_query = len(storage.queries)
    await fresh_session.assert_unchanged()
    assert storage.queries[first_fresh_query][0] == 100


@pytest.mark.asyncio
async def test_fence_session_reduces_repeated_database_feed_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 4)
    cutoff = datetime(2026, 9, 24, 12, 0)
    events = [
        _feed_event(100, {"create": {"id": "semantic_source_barrier:token"}}),
        *[
            _feed_event(stamp, {"create": {"id": f"semantic_source_barrier:noise-{stamp}"}})
            for stamp in range(101, 107)
        ],
    ]
    token = _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100)

    old_storage = _DatabaseFeedSessionStorage(events)
    await old_storage.assert_semantic_source_unchanged(token, created_before=cutoff)
    await old_storage.assert_semantic_source_unchanged(token, created_before=cutoff)
    old_reads = len(old_storage.queries)

    session_storage = _DatabaseFeedSessionStorage(events)
    session = session_storage._create_semantic_source_fence_session(token, created_before=cutoff)
    await session.assert_unchanged()
    await session.assert_unchanged()

    assert old_reads == 6
    assert len(session_storage.queries) == 4


@pytest.mark.asyncio
async def test_fence_session_applies_all_changes_in_one_versionstamp() -> None:
    cutoff = datetime(2026, 9, 24, 12, 0)
    created = {
        "id": "synapse:one-stamp-edge",
        "brain_id": "brain-a",
        "created_at": cutoff + timedelta(seconds=1),
    }
    neuron = {
        "id": "neuron:one-stamp-node",
        "brain_id": "brain-a",
        "created_at": cutoff + timedelta(seconds=1),
    }
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:token"}}),
            {
                "versionstamp": 101,
                "changes": [
                    {"create": neuron},
                    {"create": created},
                    {
                        "current": created,
                        "update": [{"op": "replace", "path": "/weight", "value": 0.5}],
                    },
                    {"delete": {"id": "synapse:one-stamp-edge"}},
                ],
            },
        ]
    )
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )

    await session.assert_unchanged()

    assert session.watermark == 102
    assert session.post_reference_ids["neuron"] == {"neuron:one-stamp-node"}
    assert session.post_reference_ids["synapse"] == set()
    assert session.scanned_source_rows["neuron"] == 1
    assert session.scanned_source_rows["synapse"] == 1


@pytest.mark.asyncio
async def test_fence_session_fails_closed_on_unknown_or_unattributable_changes() -> None:
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [_feed_event(100, {"create": {"id": "semantic_source_barrier:token"}})]
    )
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )
    await session.assert_unchanged()

    storage.events.append(
        _feed_event(
            102,
            {
                "surprise": {
                    "id": "neuron:unknown-action",
                    "brain_id": "brain-a",
                    "created_at": cutoff + timedelta(seconds=1),
                }
            },
        )
    )
    with pytest.raises(SemanticSourceFenceUnavailableError, match="unknown change type"):
        await session.assert_unchanged()
    assert session.watermark == 101

    storage.events.pop()
    storage.events.pop()
    storage.events.append(_feed_event(103, {"create": {"name": "unattributable"}}))
    with pytest.raises(SemanticSourceFenceUnavailableError, match="unclassifiable change"):
        await session.assert_unchanged()
    assert session.watermark == 101


@pytest.mark.asyncio
async def test_fence_session_cancellation_does_not_publish_partial_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 2)
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:token"}}),
            _feed_event(
                101,
                {
                    "create": {
                        "id": "neuron:cancelled-page-node",
                        "brain_id": "brain-a",
                        "created_at": cutoff + timedelta(seconds=1),
                    }
                },
            ),
            _feed_event(102, {"create": {"id": "semantic_source_barrier:noise-2"}}),
        ]
    )
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )
    waiting = asyncio.Event()
    release = asyncio.Event()
    original_query = storage._query

    async def blocked_second_query(sql: str, **params: Any) -> list[dict[str, Any]]:
        if len(storage.queries) == 1:
            waiting.set()
            await release.wait()
        return await original_query(sql, **params)

    storage._query = blocked_second_query  # type: ignore[method-assign]
    task = asyncio.create_task(session.assert_unchanged())
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session.watermark == 100
    assert session.scanned_database_rows == 0
    assert session.post_reference_ids == {"neuron": set(), "synapse": set()}


@pytest.mark.asyncio
async def test_fence_session_keeps_post_reference_ids_across_checks_and_failures() -> None:
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [_feed_event(100, {"create": {"id": "semantic_source_barrier:token"}})]
    )
    token = _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100)
    session = storage._create_semantic_source_fence_session(token, created_before=cutoff)
    created = {
        "id": "synapse:run-edge",
        "brain_id": "brain-a",
        "created_at": cutoff + timedelta(seconds=1),
    }

    storage.events.append(_feed_event(102, {"create": created}))
    await session.assert_unchanged()
    storage.events.append(
        _feed_event(
            103,
            {
                "current": created,
                "update": [{"op": "replace", "path": "/weight", "value": 0.25}],
            },
        )
    )
    await session.assert_unchanged()
    storage.events.append(_feed_event(104, {"delete": {"id": "synapse:run-edge"}}))
    await session.assert_unchanged()

    watermark_before_failure = 105
    storage.events.append(
        _feed_event(
            106,
            {
                "create": {
                    "id": "neuron:foreign",
                    "brain_id": "brain-b",
                    "created_at": cutoff + timedelta(seconds=2),
                }
            },
        )
    )
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await session.assert_unchanged()

    # Failed validation must not publish the new watermark. Removing the bad
    # synthetic event and retrying still starts from the last complete scan.
    storage.events.pop()
    storage.events.pop()
    storage.events.append(
        _feed_event(
            106,
            {
                "create": {
                    "id": "synapse:good-after-retry",
                    "brain_id": "brain-a",
                    "created_at": cutoff + timedelta(seconds=3),
                }
            },
        )
    )
    retry_query_index = len(storage.queries)
    await session.assert_unchanged()
    assert storage.queries[retry_query_index][0] == watermark_before_failure


@pytest.mark.asyncio
async def test_fence_session_preserves_legacy_readonly_and_feed_error_fences() -> None:
    cutoff = datetime(2026, 9, 24, 12, 0)
    legacy_token = _token(
        captured_at=cutoff - timedelta(seconds=1),
        versionstamp=100,
        created_at_readonly=False,
    )
    legacy_storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:token"}}),
            _feed_event(
                101,
                {
                    "update": {
                        "id": "neuron:old-row",
                        "brain_id": "brain-a",
                        "created_at": cutoff + timedelta(seconds=1),
                    }
                },
            ),
        ]
    )
    legacy_session = legacy_storage._create_semantic_source_fence_session(
        legacy_token, created_before=cutoff
    )
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await legacy_session.assert_unchanged()

    failing_storage = _DatabaseFeedSessionStorage(
        [_feed_event(100, {"create": {"id": "semantic_source_barrier:token"}})]
    )
    session = failing_storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )
    await session.assert_unchanged()
    prior_watermark = session.watermark

    async def failed_query(sql: str, **params: Any) -> list[dict[str, Any]]:
        cursor, raw_limit = sql.rsplit(" LIMIT ", 1)
        failing_storage.queries.append((int(cursor.rsplit(" ", 1)[1]), int(raw_limit)))
        raise RuntimeError("synthetic transport failure")

    failing_storage._query = failed_query  # type: ignore[method-assign]
    with pytest.raises(SemanticSourceFenceUnavailableError):
        await session.assert_unchanged()
    assert failing_storage.queries[-1][0] == prior_watermark

    failing_storage._query = _DatabaseFeedSessionStorage._query.__get__(
        failing_storage, _DatabaseFeedSessionStorage
    )
    failing_storage.events.append(
        _feed_event(
            103,
            {
                "create": {
                    "id": "synapse:retry-edge",
                    "brain_id": "brain-a",
                    "created_at": cutoff + timedelta(seconds=1),
                }
            },
        )
    )
    retry_query_index = len(failing_storage.queries)
    await session.assert_unchanged()
    assert failing_storage.queries[retry_query_index][0] == prior_watermark

    expired_storage = _DatabaseFeedSessionStorage(
        [_feed_event(100, {"create": {"id": "semantic_source_barrier:token"}})]
    )
    expired_storage.now = cutoff + timedelta(days=7)
    expired_session = expired_storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )
    with pytest.raises(SemanticSourceFenceExpiredError):
        await expired_session.assert_unchanged()
    assert expired_storage.queries == []


@pytest.mark.asyncio
async def test_fence_session_applies_source_row_budget_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 4)
    monkeypatch.setattr(revision_module, "_SOURCE_CHANGEFEED_MAX_ROWS", 3)
    cutoff = datetime(2026, 9, 24, 12, 0)
    first = {
        "id": "neuron:late-1",
        "brain_id": "brain-a",
        "created_at": cutoff + timedelta(seconds=1),
    }
    second = {
        "id": "neuron:late-2",
        "brain_id": "brain-a",
        "created_at": cutoff + timedelta(seconds=2),
    }
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:token"}}),
            _feed_event(101, {"create": first}),
            _feed_event(102, {"create": second}),
        ]
    )
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )
    await session.assert_unchanged()
    storage.events.extend(
        _feed_event(
            104 + offset,
            {
                "create": {
                    "id": f"neuron:late-{offset + 3}",
                    "brain_id": "brain-a",
                    "created_at": cutoff + timedelta(seconds=offset + 3),
                }
            },
        )
        for offset in range(4)
    )

    assert await session.verify() == "pending"
    assert session.scanned_source_rows["neuron"] == 5
    assert session.target_token is not None
    assert await session.verify() == "verified"
    assert session.scanned_source_rows["neuron"] == 6


@pytest.mark.asyncio
async def test_fence_session_returns_pending_when_invocation_budget_ends_before_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 4)
    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
            _feed_event(101, {"create": {"id": "noise:one"}}),
            _feed_event(102, {"create": {"id": "noise:two"}}),
            _feed_event(103, {"create": {"id": "semantic_source_barrier:target"}}),
        ]
    )
    target = _token(captured_at=cutoff, versionstamp=103)

    async def capture_target() -> str:
        return target

    storage.capture_semantic_source_token = capture_target  # type: ignore[method-assign]
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )

    assert await session.verify() == "pending"
    assert session.watermark == 102
    assert session.target_token == target
    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 0)
    with pytest.raises(revision_module.SemanticSourceValidationPending):
        await session.assert_unchanged()
    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    assert await session.verify() == "verified"


@pytest.mark.asyncio
async def test_fence_session_verifies_when_barrier_is_exactly_at_database_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 4)
    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
            _feed_event(101, {"create": {"id": "noise:one"}}),
            _feed_event(102, {"create": {"id": "semantic_source_barrier:target"}}),
        ]
    )

    async def capture_target() -> str:
        return _token(captured_at=cutoff, versionstamp=102)

    storage.capture_semantic_source_token = capture_target  # type: ignore[method-assign]
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )

    assert await session.verify() == "verified"
    assert session.watermark == 102
    assert session.target_token is None


@pytest.mark.asyncio
async def test_fence_session_budget_callback_commits_only_complete_page_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_PAGE_SIZE", 2)
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
            _feed_event(101, {"create": {"id": "noise:one"}}),
            _feed_event(102, {"create": {"id": "noise:two"}}),
            _feed_event(103, {"create": {"id": "semantic_source_barrier:target"}}),
        ]
    )
    target = _token(captured_at=cutoff, versionstamp=103)

    async def capture_target() -> str:
        return target

    storage.capture_semantic_source_token = capture_target  # type: ignore[method-assign]
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )
    checks = 0

    async def stop_after_one_page() -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    assert await session.verify(budget_check=stop_after_one_page) == "pending"
    assert session.watermark == 101
    assert session.target_token == target
    assert await session.verify() == "verified"
    assert session.watermark == 103


@pytest.mark.asyncio
async def test_fence_session_restores_pending_checkpoint_and_reaches_saved_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
            _feed_event(101, {"create": {"id": "noise:one"}}),
            _feed_event(102, {"create": {"id": "noise:two"}}),
            _feed_event(103, {"create": {"id": "semantic_source_barrier:target"}}),
        ]
    )
    original = _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100)
    target = _token(captured_at=cutoff, versionstamp=103)

    async def capture_target() -> str:
        return target

    storage.capture_semantic_source_token = capture_target  # type: ignore[method-assign]
    session = storage._create_semantic_source_fence_session(original, created_before=cutoff)

    assert await session.verify() == "pending"
    saved = {
        "token": session.token,
        "brain_id": session.brain_id,
        "token_versionstamp": session.token_versionstamp,
        "created_before": session.created_before,
        "validation_token": session.validation_token,
        "target_token": session.target_token,
        "watermark": session.watermark,
        "scanned_database_rows": session.scanned_database_rows,
        "scanned_source_rows": dict(session.scanned_source_rows),
        "post_reference_ids": {
            key: set(value) for key, value in session.post_reference_ids.items()
        },
    }
    assert saved["token"] == original
    assert saved["target_token"] == target

    restored = storage._create_semantic_source_fence_session(original, created_before=cutoff)
    for name, value in saved.items():
        setattr(restored, name, value)
    assert await restored.verify() == "verified"
    assert restored.validation_token == target
    assert restored.target_token is None
    assert restored.watermark == 103


@pytest.mark.asyncio
async def test_fence_session_rejects_mutation_after_pending_prefix_without_publishing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 2)
    cutoff = datetime(2026, 9, 24, 12, 0)
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
            _feed_event(101, {"create": {"id": "noise:one"}}),
            _feed_event(
                102,
                {
                    "create": {
                        "id": "neuron:late-prefix",
                        "brain_id": "brain-a",
                        "created_at": cutoff + timedelta(seconds=1),
                    }
                },
            ),
            _feed_event(
                103,
                {
                    "update": {
                        "id": "neuron:old-mutated",
                        "brain_id": "brain-a",
                        "created_at": cutoff - timedelta(seconds=1),
                    }
                },
            ),
        ]
    )
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )

    assert await session.verify() == "pending"
    assert session.post_reference_ids["neuron"] == {"neuron:late-prefix"}
    prior_state = (
        session.watermark,
        dict(session.scanned_source_rows),
        {table: set(ids) for table, ids in session.post_reference_ids.items()},
        session.target_token,
    )
    with pytest.raises(SemanticSourceChangedError, match="neuron changed"):
        await session.verify()
    assert (
        session.watermark,
        session.scanned_source_rows,
        session.post_reference_ids,
        session.target_token,
    ) == prior_state


@pytest.mark.asyncio
async def test_fence_session_resumes_proven_create_then_delete_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 1)
    cutoff = datetime(2026, 9, 24, 12, 0)
    created = {
        "id": "neuron:late-safe-delete",
        "brain_id": "brain-a",
        "created_at": cutoff + timedelta(seconds=1),
    }
    storage = _DatabaseFeedSessionStorage(
        [
            _feed_event(100, {"create": {"id": "semantic_source_barrier:origin"}}),
            _feed_event(101, {"create": created}),
            _feed_event(102, {"create": {"id": "noise:two"}}),
        ]
    )
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=100),
        created_before=cutoff,
    )

    assert await session.verify() == "pending"
    assert session.post_reference_ids["neuron"] == {"neuron:late-safe-delete"}
    monkeypatch.setattr(revision_module, "_DATABASE_FEED_MAX_ROWS", 4)
    assert await session.verify() == "verified"
    storage.events.append(_feed_event(104, {"delete": {"id": "neuron:late-safe-delete"}}))
    assert await session.verify() == "verified"
    assert session.post_reference_ids["neuron"] == set()


@pytest.mark.asyncio
async def test_fence_session_streams_beyond_the_real_two_million_row_invocation_budget() -> None:
    import surreal_memory.storage.surrealdb.semantic_source_revision as revision_module

    budget = revision_module._DATABASE_FEED_MAX_ROWS
    target_stamp = budget + 1
    cutoff = datetime(2026, 9, 24, 12, 0)

    class StreamingStorage(SurrealDBSemanticSourceRevisionMixin):
        def _get_brain_id(self) -> str:
            return "brain-a"

        async def _server_now(self) -> datetime:
            return cutoff

        async def capture_semantic_source_token(self) -> str:
            return _token(captured_at=cutoff, versionstamp=target_stamp)

        async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
            cursor_part, raw_limit = sql.rsplit(" LIMIT ", 1)
            cursor = int(cursor_part.rsplit(" ", 1)[1])
            limit = int(raw_limit)
            stamps = range(cursor, min(cursor + limit, target_stamp + 1))
            return [
                _feed_event(
                    stamp,
                    {
                        "create": {
                            "id": (
                                "semantic_source_barrier:target"
                                if stamp == target_stamp
                                else f"feed_noise:{stamp}"
                            )
                        }
                    },
                )
                for stamp in stamps
            ]

    storage = StreamingStorage()
    session = storage._create_semantic_source_fence_session(
        _token(captured_at=cutoff - timedelta(seconds=1), versionstamp=0),
        created_before=cutoff,
    )

    assert await session.verify() == "pending"
    assert session.watermark == budget
    assert session.scanned_database_rows == budget
    assert session.target_token is not None
    assert await session.verify() == "verified"
    assert session.watermark == target_stamp
