"""Regression tests for the schema statement parser (dropped-DDL bug).

The old parser split SCHEMA_SQL on ";" and discarded any chunk whose stripped
text started with "--". A statement sitting directly under an explanatory
comment block therefore vanished before it ever reached the connection: all 26
DEFINE TABLE statements and — critically since 2.7.4 — the smem_content
FULLTEXT analyzer, without which idx_neuron_content_fts fails to build and the
@@ operator behind find_neurons matches nothing.
"""

from __future__ import annotations

import re

from surreal_memory.storage.surrealdb.schema import (
    SCHEMA_SQL,
    _parse_schema_statements,
    ensure_schema,
)


class TestParseSchemaStatements:
    def test_statement_under_a_comment_block_survives(self) -> None:
        sql = (
            "-- ====================\n"
            "-- Explanatory header\n"
            "-- ====================\n"
            "\n"
            "-- This comment explains the analyzer in some detail,\n"
            "-- spanning several lines.\n"
            "DEFINE ANALYZER a TOKENIZERS blank;\n"
            "DEFINE TABLE t SCHEMAFULL;\n"
        )
        statements = _parse_schema_statements(sql)
        assert "DEFINE ANALYZER a TOKENIZERS blank" in statements
        assert "DEFINE TABLE t SCHEMAFULL" in statements

    def test_pure_comment_chunks_are_dropped(self) -> None:
        sql = "-- only a comment\n\n-- another one;\nDEFINE TABLE t SCHEMALESS;"
        statements = _parse_schema_statements(sql)
        assert statements == ["DEFINE TABLE t SCHEMALESS"]

    def test_no_comment_line_leaks_into_any_statement(self) -> None:
        for stmt in _parse_schema_statements(SCHEMA_SQL):
            for line in stmt.splitlines():
                assert not line.strip().startswith("--"), stmt

    def test_every_define_in_the_real_schema_survives_parsing(self) -> None:
        # Every DEFINE line present in the raw script must reach the executable
        # statement list — this is exactly the invariant the old parser broke.
        expected = re.findall(r"^DEFINE [A-Z]+ [^\n;]+", SCHEMA_SQL, flags=re.M)
        assert expected, "sanity: the schema script contains DEFINE statements"
        joined = "\n".join(_parse_schema_statements(SCHEMA_SQL))
        missing = [head for head in expected if head not in joined]
        assert not missing, f"DEFINE statements dropped by the parser: {missing}"

    def test_fulltext_analyzer_index_and_tables_present(self) -> None:
        statements = _parse_schema_statements(SCHEMA_SQL)
        assert any(
            s.startswith("DEFINE ANALYZER IF NOT EXISTS smem_content") for s in statements
        ), "the smem_content analyzer must survive parsing (2.7.4 FTS regression)"
        assert any("idx_neuron_content_fts" in s for s in statements)
        table_defines = [s for s in statements if s.startswith("DEFINE TABLE ")]
        assert len(table_defines) >= 26, table_defines


class _RecordingConn:
    """Minimal storage-connection stub recording every executed statement."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    async def query(self, sql: str) -> None:
        self.executed.append(sql)


class TestEnsureSchemaExecution:
    async def test_analyzer_fts_index_and_tables_reach_the_connection(self) -> None:
        conn = _RecordingConn()
        await ensure_schema(conn, embedding_dim=1024)
        joined = "\n".join(conn.executed)
        assert "DEFINE ANALYZER IF NOT EXISTS smem_content" in joined
        assert "idx_neuron_content_fts" in joined
        # neuron (and the other arbitrary-``metadata`` tables) are SCHEMALESS so
        # their ``TYPE object`` metadata accepts nested keys without FLEXIBLE, which
        # SurrealDB rejects on a SCHEMALESS table — see SCHEMA_SQL rationale comment.
        assert "DEFINE TABLE neuron SCHEMALESS" in joined
        assert "HNSW DIMENSION 1024" in joined

    async def test_a_failing_statement_does_not_abort_the_rest(self) -> None:
        class _FlakyConn(_RecordingConn):
            async def query(self, sql: str) -> None:
                await super().query(sql)
                if "DEFINE TABLE neuron " in sql:
                    raise RuntimeError("The table 'neuron' already exists")

        conn = _FlakyConn()
        await ensure_schema(conn, embedding_dim=1024)
        # Statements after the failing one still execute — including the
        # analyzer-dependent FULLTEXT index.
        assert any("idx_neuron_content_fts" in s for s in conn.executed)
