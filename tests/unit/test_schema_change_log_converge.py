"""ensure_schema converges a legacy SCHEMALESS change_log table (issue #230).

Databases that predate the change_log table declaration (or were carried
through export/import) hold it as SCHEMALESS; the `DEFINE FIELD … FLEXIBLE`
for `payload` is only valid on a SCHEMAFULL table, so every startup warned and
the field kept its old shape. The fix probes `INFO FOR DB` and runs
`ALTER TABLE change_log SCHEMAFULL` before the statement loop — fail-soft, so
a failed probe changes nothing relative to the previous behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb.schema import ensure_schema


class _FakeConn:
    """Records queries; answers INFO FOR DB from a canned table map."""

    def __init__(
        self,
        tables: dict[str, str] | None = None,
        fail_alter: bool = False,
        fail_probe: bool = False,
    ) -> None:
        self._tables = tables if tables is not None else {}
        self._fail_alter = fail_alter
        self._fail_probe = fail_probe
        self.queries: list[str] = []

    async def query(self, q: str) -> Any:
        self.queries.append(q)
        if q.startswith("INFO FOR DB"):
            if self._fail_probe:
                raise RuntimeError("probe unavailable")
            return [{"tables": dict(self._tables)}]
        if q.startswith("ALTER TABLE change_log") and self._fail_alter:
            raise RuntimeError("alter rejected")
        return []


def _alter_queries(conn: _FakeConn) -> list[str]:
    return [q for q in conn.queries if q.startswith("ALTER TABLE change_log")]


def _change_log_field_defines(conn: _FakeConn) -> list[str]:
    return [q for q in conn.queries if "DEFINE FIELD" in q and "ON change_log" in q]


@pytest.mark.asyncio
async def test_schemaless_table_is_converted_before_field_defines() -> None:
    conn = _FakeConn(tables={"change_log": "DEFINE TABLE change_log TYPE ANY SCHEMALESS"})
    await ensure_schema(conn, embedding_dim=1024)
    assert _alter_queries(conn) == ["ALTER TABLE change_log SCHEMAFULL;"]
    fields = _change_log_field_defines(conn)
    assert fields, "the change_log field defines must still run"
    alter_at = conn.queries.index("ALTER TABLE change_log SCHEMAFULL;")
    assert alter_at < conn.queries.index(fields[0]), (
        "the ALTER must precede the field defines it unblocks"
    )


@pytest.mark.asyncio
async def test_schemafull_table_is_left_alone() -> None:
    conn = _FakeConn(tables={"change_log": "DEFINE TABLE change_log SCHEMAFULL"})
    await ensure_schema(conn, embedding_dim=1024)
    assert _alter_queries(conn) == []


@pytest.mark.asyncio
async def test_fresh_database_without_change_log_skips_alter() -> None:
    conn = _FakeConn(tables={})
    await ensure_schema(conn, embedding_dim=1024)
    assert _alter_queries(conn) == []
    assert any(q.startswith("DEFINE TABLE change_log SCHEMAFULL") for q in conn.queries), (
        "fresh databases still get the SCHEMAFULL declaration from SCHEMA_SQL"
    )


@pytest.mark.asyncio
async def test_failed_alter_is_fail_soft() -> None:
    conn = _FakeConn(
        tables={"change_log": "DEFINE TABLE change_log TYPE ANY SCHEMALESS"},
        fail_alter=True,
    )
    await ensure_schema(conn, embedding_dim=1024)  # must not raise
    assert _change_log_field_defines(conn), "defines still attempted after a failed ALTER"


@pytest.mark.asyncio
async def test_failed_probe_is_fail_soft() -> None:
    conn = _FakeConn(fail_probe=True)
    await ensure_schema(conn, embedding_dim=1024)  # must not raise
    assert _alter_queries(conn) == []
    assert _change_log_field_defines(conn), "defines still attempted after a failed probe"
