"""Regression tests for the synapse edge model (schema v8).

As of v8 the ``synapse`` table is a native SurrealDB ``RELATION``: endpoints live
in the built-in ``in`` / ``out`` RecordID fields instead of ``source_id`` /
``target_id`` string columns, so GQL ``MATCH`` and graph-traversal pushdown work
over real edges. The canonical DDL is ``SYNAPSE_V8_DDL`` (single source of truth,
applied by both ``ensure_schema`` and the synapse->RELATE migration). These tests
pin that shape; the store-side query rewrite is covered by the store tests.
"""

from __future__ import annotations

import re

from surreal_memory.storage.surrealdb.schema import (
    SCHEMA_SQL,
    SCHEMA_VERSION,
    SYNAPSE_V8_DDL,
)

# Collapse runs of horizontal whitespace so assertions do not depend on the
# column alignment used in the schema source.
_DDL = re.sub(r"[ \t]+", " ", " ; ".join(SYNAPSE_V8_DDL))
_SCHEMA = re.sub(r"[ \t]+", " ", SCHEMA_SQL)


class TestSynapseRelationModel:
    def test_schema_version_is_current(self) -> None:
        # Renamed: the old name said 8 while asserting 9, so it stopped
        # describing what it checked the first time the schema moved.
        assert SCHEMA_VERSION == 10

    def test_synapse_is_native_relation(self) -> None:
        assert "DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron SCHEMAFULL" in _DDL

    def test_edge_indexes_cover_in_and_out(self) -> None:
        assert "DEFINE INDEX idx_synapse_in ON synapse FIELDS brain_id, in" in _DDL
        assert "DEFINE INDEX idx_synapse_out ON synapse FIELDS brain_id, out" in _DDL
        assert "DEFINE INDEX idx_synapse_type ON synapse FIELDS brain_id, type" in _DDL

    def test_source_target_fields_removed(self) -> None:
        # The flat document columns are gone: endpoints are the RELATION in/out.
        assert "DEFINE FIELD source_id ON synapse" not in _DDL
        assert "DEFINE FIELD target_id ON synapse" not in _DDL
        assert "DEFINE FIELD source_id ON synapse" not in _SCHEMA
        assert "DEFINE FIELD target_id ON synapse" not in _SCHEMA

    def test_old_source_target_indexes_removed(self) -> None:
        assert "idx_synapse_source" not in _DDL and "idx_synapse_source" not in _SCHEMA
        assert "idx_synapse_target" not in _DDL and "idx_synapse_target" not in _SCHEMA

    def test_schema_meta_table_defined(self) -> None:
        # The migration bookkeeping table (version + lock + state) is part of the schema.
        assert "DEFINE TABLE schema_meta SCHEMALESS" in _SCHEMA

    def test_connects_to_table_removed(self) -> None:
        # connects_to was a write-only RELATE that never parsed; nothing read it.
        assert "connects_to" not in SCHEMA_SQL

    def test_synapse_not_redefined_in_schema_sql(self) -> None:
        # synapse DDL lives only in SYNAPSE_V8_DDL now, not inline in SCHEMA_SQL,
        # so there is exactly one source of truth for the edge table definition.
        assert "DEFINE TABLE synapse" not in _SCHEMA
