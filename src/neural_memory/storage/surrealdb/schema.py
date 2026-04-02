"""SurrealDB schema initialization for Neural Memory."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- ============================================================
-- Neural Memory SurrealDB Schema
-- ============================================================

-- Neurons (primary memory units)
DEFINE TABLE neuron SCHEMAFULL;
DEFINE FIELD id              ON neuron TYPE string;
DEFINE FIELD brain_id        ON neuron TYPE string;
DEFINE FIELD type            ON neuron TYPE string;
DEFINE FIELD content         ON neuron TYPE string;
DEFINE FIELD content_hash    ON neuron TYPE int DEFAULT 0;
DEFINE FIELD metadata        ON neuron TYPE object DEFAULT {{}};
DEFINE FIELD embedding_vec   ON neuron TYPE option<array<float>>;
DEFINE FIELD ephemeral       ON neuron TYPE bool DEFAULT false;
DEFINE FIELD created_at      ON neuron TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at      ON neuron TYPE datetime DEFAULT time::now();
DEFINE FIELD access_frequency ON neuron TYPE int DEFAULT 0;
DEFINE FIELD last_activated  ON neuron TYPE option<datetime>;
DEFINE FIELD compression_tier ON neuron TYPE int DEFAULT 0;
DEFINE FIELD lifecycle_state ON neuron TYPE string DEFAULT 'active';
DEFINE FIELD frozen          ON neuron TYPE bool DEFAULT false;
DEFINE FIELD last_accessed_at ON neuron TYPE option<datetime>;
DEFINE INDEX idx_neuron_brain    ON neuron FIELDS brain_id;
DEFINE INDEX idx_neuron_type     ON neuron FIELDS brain_id, type;
DEFINE INDEX idx_neuron_hash     ON neuron FIELDS brain_id, content_hash;
DEFINE INDEX idx_neuron_content  ON neuron FIELDS brain_id, content;
DEFINE INDEX idx_neuron_embedding ON neuron FIELDS embedding_vec HNSW DIMENSION 3072 DIST COSINE;

-- Neuron activation states
DEFINE TABLE neuron_state SCHEMAFULL;
DEFINE FIELD neuron_id         ON neuron_state TYPE string;
DEFINE FIELD brain_id          ON neuron_state TYPE string;
DEFINE FIELD activation_level  ON neuron_state TYPE float DEFAULT 0.0;
DEFINE FIELD access_frequency  ON neuron_state TYPE int DEFAULT 0;
DEFINE FIELD last_activated    ON neuron_state TYPE option<datetime>;
DEFINE FIELD decay_rate        ON neuron_state TYPE float DEFAULT 0.1;
DEFINE FIELD firing_threshold  ON neuron_state TYPE float DEFAULT 0.3;
DEFINE FIELD refractory_until  ON neuron_state TYPE option<datetime>;
DEFINE FIELD refractory_period_ms ON neuron_state TYPE float DEFAULT 500.0;
DEFINE FIELD homeostatic_target   ON neuron_state TYPE float DEFAULT 0.5;
DEFINE FIELD created_at        ON neuron_state TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_state_neuron  ON neuron_state FIELDS brain_id, neuron_id UNIQUE;

-- Synapses (graph edges between neurons)
DEFINE TABLE synapse SCHEMAFULL;
DEFINE FIELD id           ON synapse TYPE string;
DEFINE FIELD brain_id     ON synapse TYPE string;
DEFINE FIELD type         ON synapse TYPE string;
DEFINE FIELD weight       ON synapse TYPE float DEFAULT 1.0;
DEFINE FIELD direction    ON synapse TYPE string DEFAULT 'forward';
DEFINE FIELD metadata     ON synapse TYPE object DEFAULT {{}};
DEFINE FIELD created_at   ON synapse TYPE datetime DEFAULT time::now();
DEFINE FIELD last_activated    ON synapse TYPE option<datetime>;
DEFINE FIELD reinforced_count  ON synapse TYPE int DEFAULT 0;
DEFINE INDEX idx_synapse_brain ON synapse FIELDS brain_id;
DEFINE INDEX idx_synapse_source ON synapse FIELDS brain_id, out;
DEFINE INDEX idx_synapse_target ON synapse FIELDS brain_id, in;

-- Synapse edge connections (for graph traversal)
DEFINE TABLE connects_to SCHEMAFULL;
DEFINE FIELD brain_id ON connects_to TYPE string;

-- Fibers (memory clusters / signal pathways)
DEFINE TABLE fiber SCHEMAFULL;
DEFINE FIELD id              ON fiber TYPE string;
DEFINE FIELD brain_id        ON fiber TYPE string;
DEFINE FIELD neuron_ids      ON fiber TYPE array<string>;
DEFINE FIELD synapse_ids     ON fiber TYPE array<string>;
DEFINE FIELD anchor_neuron_id ON fiber TYPE string;
DEFINE FIELD pathway         ON fiber TYPE array<string> DEFAULT [];
DEFINE FIELD conductivity    ON fiber TYPE float DEFAULT 1.0;
DEFINE FIELD last_conducted  ON fiber TYPE option<datetime>;
DEFINE FIELD time_start      ON fiber TYPE option<datetime>;
DEFINE FIELD time_end        ON fiber TYPE option<datetime>;
DEFINE FIELD coherence       ON fiber TYPE float DEFAULT 0.0;
DEFINE FIELD salience        ON fiber TYPE float DEFAULT 0.0;
DEFINE FIELD frequency       ON fiber TYPE int DEFAULT 0;
DEFINE FIELD summary         ON fiber TYPE option<string>;
DEFINE FIELD essence         ON fiber TYPE option<string>;
DEFINE FIELD auto_tags       ON fiber TYPE array<string> DEFAULT [];
DEFINE FIELD agent_tags      ON fiber TYPE array<string> DEFAULT [];
DEFINE FIELD metadata        ON fiber TYPE object DEFAULT {{}};
DEFINE FIELD compression_tier ON fiber TYPE int DEFAULT 0;
DEFINE FIELD pinned           ON fiber TYPE bool DEFAULT false;
DEFINE FIELD created_at       ON fiber TYPE datetime DEFAULT time::now();
DEFINE FIELD last_ghost_shown_at ON fiber TYPE option<datetime>;
DEFINE INDEX idx_fiber_brain  ON fiber FIELDS brain_id;
DEFINE INDEX idx_fiber_anchor ON fiber FIELDS brain_id, anchor_neuron_id;

-- Brains (top-level containers)
DEFINE TABLE brain SCHEMAFULL;
DEFINE FIELD id          ON brain TYPE string;
DEFINE FIELD name        ON brain TYPE string;
DEFINE FIELD config      ON brain TYPE object DEFAULT {{}};
DEFINE FIELD metadata    ON brain TYPE object DEFAULT {{}};
DEFINE FIELD created_at  ON brain TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at  ON brain TYPE datetime DEFAULT time::now();
-- Removed unique index on name - multiple brains can have same name

-- Change log (for multi-device sync)
DEFINE TABLE change_log SCHEMAFULL;
DEFINE FIELD brain_id      ON change_log TYPE string;
DEFINE FIELD entity_type   ON change_log TYPE string;
DEFINE FIELD entity_id     ON change_log TYPE string;
DEFINE FIELD operation     ON change_log TYPE string;
DEFINE FIELD device_id     ON change_log TYPE string DEFAULT '';
DEFINE FIELD payload        ON change_log TYPE option<object>;
DEFINE FIELD changed_at     ON change_log TYPE datetime DEFAULT time::now();
DEFINE FIELD synced         ON change_log TYPE bool DEFAULT false;
DEFINE FIELD sequence       ON change_log TYPE int;
DEFINE INDEX idx_changelog_brain  ON change_log FIELDS brain_id;
DEFINE INDEX idx_changelog_seq    ON change_log FIELDS brain_id, sequence;

-- Devices (sync device registry)
DEFINE TABLE device SCHEMAFULL;
DEFINE FIELD device_id          ON device TYPE string;
DEFINE FIELD brain_id           ON device TYPE string;
DEFINE FIELD device_name        ON device TYPE string DEFAULT '';
DEFINE FIELD registered_at      ON device TYPE datetime DEFAULT time::now();
DEFINE FIELD last_sync_at       ON device TYPE option<datetime>;
DEFINE FIELD last_sync_sequence ON device TYPE int DEFAULT 0;
DEFINE INDEX idx_device_brain   ON device FIELDS brain_id, device_id UNIQUE;

-- Merkle hashes (for efficient delta sync)
DEFINE TABLE merkle_hash SCHEMAFULL;
DEFINE FIELD brain_id      ON merkle_hash TYPE string;
DEFINE FIELD entity_type   ON merkle_hash TYPE string;
DEFINE FIELD prefix        ON merkle_hash TYPE string;
DEFINE FIELD hash          ON merkle_hash TYPE string;
DEFINE FIELD computed_at   ON merkle_hash TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_merkle    ON merkle_hash FIELDS brain_id, entity_type, prefix UNIQUE;

-- Typed memories
DEFINE TABLE typed_memory SCHEMAFULL;
DEFINE FIELD fiber_id      ON typed_memory TYPE string;
DEFINE FIELD brain_id      ON typed_memory TYPE string;
DEFINE FIELD memory_type   ON typed_memory TYPE string;
DEFINE FIELD priority      ON typed_memory TYPE string DEFAULT 'medium';
DEFINE FIELD content       ON typed_memory TYPE string;
DEFINE FIELD tags          ON typed_memory TYPE array<string> DEFAULT [];
DEFINE FIELD trust_score   ON typed_memory TYPE float DEFAULT 0.5;
DEFINE FIELD source        ON typed_memory TYPE option<string>;
DEFINE FIELD project_id    ON typed_memory TYPE option<string>;
DEFINE FIELD expires_at    ON typed_memory TYPE option<datetime>;
DEFINE FIELD tier          ON typed_memory TYPE string DEFAULT 'warm';
DEFINE FIELD metadata      ON typed_memory TYPE object DEFAULT {{}};
DEFINE FIELD created_at    ON typed_memory TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at    ON typed_memory TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_typed_brain ON typed_memory FIELDS brain_id;
DEFINE INDEX idx_typed_type  ON typed_memory FIELDS brain_id, memory_type;
"""


async def ensure_schema(conn: Any) -> None:
    """Apply schema to SurrealDB. Safe to call multiple times."""
    logger.info("Applying SurrealDB schema (v%d)...", SCHEMA_VERSION)
    statements = [
        s.strip() for s in SCHEMA_SQL.split(";") if s.strip() and not s.strip().startswith("--")
    ]
    for stmt in statements:
        try:
            await conn.query(stmt + ";")
        except Exception:
            # Index/table may already exist — that's fine
            pass
    logger.info("SurrealDB schema ready (v%d)", SCHEMA_VERSION)
