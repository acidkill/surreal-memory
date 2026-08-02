"""SurrealDB schema initialization for Surreal-Memory."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 9

SCHEMA_SQL = """
-- ============================================================
-- Surreal-Memory SurrealDB Schema
-- ============================================================

-- Full-text analyzer for neuron.content search (the @@ operator). Defined
-- before the index that uses it. Tokenizes on whitespace/class and lowercases,
-- so keyword/entity lookups become case-insensitive and index-backed instead of
-- full-scanning with CONTAINS.
DEFINE ANALYZER IF NOT EXISTS smem_content TOKENIZERS blank, class FILTERS lowercase, ascii;

-- Tables that carry an arbitrary-key ``metadata``/``config`` object are declared
-- SCHEMALESS (not SCHEMAFULL). Those nested keys require FLEXIBLE on a SCHEMAFULL
-- table, but SurrealDB rejects ``DEFINE FIELD ... FLEXIBLE`` on any table that is
-- already SCHEMALESS ("FLEXIBLE can only be used in SCHEMAFULL tables"). Because a
-- bare ``DEFINE TABLE X SCHEMAFULL`` is swallowed as "already exists" on an upgraded
-- DB, a legacy table first created SCHEMALESS never converges to SCHEMAFULL, so
-- every FLEXIBLE field-def then errors on each ensure_schema()/consolidate. And
-- converting such a table to SCHEMAFULL would BRICK updates on any row still holding
-- a legacy field absent from the schema. A SCHEMALESS table already accepts
-- arbitrary nested object keys WITHOUT FLEXIBLE, so these tables are SCHEMALESS +
-- plain ``TYPE object`` — correct and identical on fresh and upgraded databases.
-- (synapse + retrieval_trace stay SCHEMAFULL: a migration drops & recreates them,
-- so their FLEXIBLE fields are always valid.)

-- Neurons (primary memory units)
DEFINE TABLE neuron SCHEMALESS;
DEFINE FIELD id              ON neuron TYPE string;
DEFINE FIELD brain_id        ON neuron TYPE string;
DEFINE FIELD type            ON neuron TYPE string;
DEFINE FIELD content         ON neuron TYPE string;
DEFINE FIELD content_hash    ON neuron TYPE int DEFAULT 0;
DEFINE FIELD metadata        ON neuron TYPE object DEFAULT {};
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
DEFINE INDEX IF NOT EXISTS idx_neuron_content_fts ON neuron FIELDS content FULLTEXT ANALYZER smem_content BM25;
-- idx_neuron_embedding (HNSW) is defined dynamically in ensure_schema() with the
-- configured embedding dimension, so the vector index always matches the model.

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

-- Synapses are native RELATION edges as of schema v8. Their DDL lives in the
-- SYNAPSE_V8_DDL list below (single source of truth) and is applied both by
-- ensure_schema() and by the synapse->RELATE migration (migrations.py). Edge
-- endpoints live in the built-in `in`/`out` fields, not source_id/target_id.

-- Schema migration metadata: version stamp + migration lock + migration state.
DEFINE TABLE schema_meta SCHEMALESS;

-- Fibers (memory clusters / signal pathways)
DEFINE TABLE fiber SCHEMALESS;
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
DEFINE FIELD metadata        ON fiber TYPE object DEFAULT {};
DEFINE FIELD compression_tier ON fiber TYPE int DEFAULT 0;
DEFINE FIELD pinned           ON fiber TYPE bool DEFAULT false;
DEFINE FIELD created_at       ON fiber TYPE datetime DEFAULT time::now();
DEFINE FIELD last_ghost_shown_at ON fiber TYPE option<datetime>;
DEFINE INDEX idx_fiber_brain  ON fiber FIELDS brain_id;
DEFINE INDEX idx_fiber_anchor ON fiber FIELDS brain_id, anchor_neuron_id;

-- Brains (top-level containers)
DEFINE TABLE brain SCHEMALESS;
DEFINE FIELD id          ON brain TYPE string;
DEFINE FIELD name        ON brain TYPE string;
DEFINE FIELD config      ON brain TYPE object DEFAULT {};
DEFINE FIELD metadata    ON brain TYPE object DEFAULT {};
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
DEFINE TABLE typed_memory SCHEMALESS;
DEFINE FIELD fiber_id      ON typed_memory TYPE string;
DEFINE FIELD brain_id      ON typed_memory TYPE string;
DEFINE FIELD memory_type   ON typed_memory TYPE string;
DEFINE FIELD priority      ON typed_memory TYPE string DEFAULT '5';
DEFINE FIELD content       ON typed_memory TYPE option<string>;
DEFINE FIELD tags          ON typed_memory TYPE array<string> DEFAULT [];
DEFINE FIELD trust_score   ON typed_memory TYPE option<float>;
DEFINE FIELD source        ON typed_memory TYPE option<string>;
DEFINE FIELD project_id    ON typed_memory TYPE option<string>;
DEFINE FIELD expires_at    ON typed_memory TYPE option<datetime>;
DEFINE FIELD tier          ON typed_memory TYPE string DEFAULT 'warm';
DEFINE FIELD metadata      ON typed_memory TYPE object DEFAULT {};
DEFINE FIELD created_at    ON typed_memory TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at    ON typed_memory TYPE datetime DEFAULT time::now();
DEFINE FIELD valid_from    ON typed_memory TYPE option<datetime>;
DEFINE FIELD valid_until   ON typed_memory TYPE option<datetime>;
DEFINE FIELD superseded_by ON typed_memory TYPE option<string>;
DEFINE INDEX idx_typed_brain   ON typed_memory FIELDS brain_id;
DEFINE INDEX idx_typed_type    ON typed_memory FIELDS brain_id, memory_type;
DEFINE INDEX idx_typed_fiber   ON typed_memory FIELDS brain_id, fiber_id UNIQUE;
DEFINE INDEX idx_typed_valid   ON typed_memory FIELDS brain_id, valid_until;
DEFINE INDEX idx_typed_expires ON typed_memory FIELDS brain_id, expires_at;

-- Projects (named scopes for grouping memories)
DEFINE TABLE project SCHEMALESS;
DEFINE FIELD brain_id    ON project TYPE string;
DEFINE FIELD uid         ON project TYPE string;
DEFINE FIELD name        ON project TYPE string;
DEFINE INDEX idx_project_brain ON project FIELDS brain_id;
DEFINE INDEX idx_project_uid   ON project FIELDS brain_id, uid UNIQUE;

-- Sources (memory origin registry)
DEFINE TABLE source SCHEMALESS;
DEFINE FIELD id             ON source TYPE string;
DEFINE FIELD brain_id       ON source TYPE string;
DEFINE FIELD name           ON source TYPE string;
DEFINE FIELD source_type    ON source TYPE string DEFAULT 'document';
DEFINE FIELD version        ON source TYPE string DEFAULT '';
DEFINE FIELD effective_date ON source TYPE option<datetime>;
DEFINE FIELD expires_at     ON source TYPE option<datetime>;
DEFINE FIELD status         ON source TYPE string DEFAULT 'active';
DEFINE FIELD file_hash      ON source TYPE string DEFAULT '';
DEFINE FIELD metadata       ON source TYPE object DEFAULT {};
DEFINE FIELD created_at     ON source TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at     ON source TYPE datetime DEFAULT time::now();
DEFINE FIELD trust          ON source TYPE option<float>;
DEFINE INDEX idx_source_brain ON source FIELDS brain_id;
DEFINE INDEX idx_source_name  ON source FIELDS brain_id, name;

-- Retrieval traces (queryable recall provenance / telemetry)
DEFINE TABLE retrieval_trace SCHEMAFULL;
DEFINE FIELD id           ON retrieval_trace TYPE string;
DEFINE FIELD brain_id     ON retrieval_trace TYPE string;
DEFINE FIELD session_id   ON retrieval_trace TYPE option<string>;
DEFINE FIELD query        ON retrieval_trace TYPE string DEFAULT '';
DEFINE FIELD depth_used   ON retrieval_trace TYPE int DEFAULT 0;
DEFINE FIELD mode         ON retrieval_trace TYPE string DEFAULT '';
DEFINE FIELD confidence   ON retrieval_trace TYPE float DEFAULT 0.0;
DEFINE FIELD latency_ms   ON retrieval_trace TYPE float DEFAULT 0.0;
DEFINE FIELD fiber_ids    ON retrieval_trace TYPE array<string> DEFAULT [];
DEFINE FIELD payload      ON retrieval_trace TYPE object FLEXIBLE DEFAULT {};
DEFINE FIELD created_at   ON retrieval_trace TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_trace_brain   ON retrieval_trace FIELDS brain_id;
DEFINE INDEX idx_trace_created ON retrieval_trace FIELDS brain_id, created_at;

-- Alerts (system-generated warnings and recommendations)
DEFINE TABLE alerts SCHEMALESS;
DEFINE FIELD id                 ON alerts TYPE string;
DEFINE FIELD brain_id           ON alerts TYPE string;
DEFINE FIELD alert_type         ON alerts TYPE string;
DEFINE FIELD severity           ON alerts TYPE string DEFAULT 'low';
DEFINE FIELD message            ON alerts TYPE string DEFAULT '';
DEFINE FIELD recommended_action ON alerts TYPE string DEFAULT '';
DEFINE FIELD status             ON alerts TYPE string DEFAULT 'active';
DEFINE FIELD created_at         ON alerts TYPE datetime DEFAULT time::now();
DEFINE FIELD seen_at            ON alerts TYPE option<datetime>;
DEFINE FIELD acknowledged_at    ON alerts TYPE option<datetime>;
DEFINE FIELD resolved_at        ON alerts TYPE option<datetime>;
DEFINE FIELD metadata           ON alerts TYPE object DEFAULT {};
DEFINE INDEX idx_alerts_brain   ON alerts FIELDS brain_id;
DEFINE INDEX idx_alerts_status  ON alerts FIELDS brain_id, status;

-- Cognitive state (per-neuron belief tracking + predictions when predicted_at is set)
DEFINE TABLE cognitive_state SCHEMAFULL;
DEFINE FIELD brain_id               ON cognitive_state TYPE string;
DEFINE FIELD neuron_id              ON cognitive_state TYPE string;
DEFINE FIELD confidence             ON cognitive_state TYPE float DEFAULT 0.5;
DEFINE FIELD evidence_for_count     ON cognitive_state TYPE int DEFAULT 0;
DEFINE FIELD evidence_against_count ON cognitive_state TYPE int DEFAULT 0;
DEFINE FIELD status                 ON cognitive_state TYPE string DEFAULT 'active';
DEFINE FIELD predicted_at           ON cognitive_state TYPE option<datetime>;
DEFINE FIELD resolved_at            ON cognitive_state TYPE option<datetime>;
DEFINE FIELD schema_version         ON cognitive_state TYPE int DEFAULT 1;
DEFINE FIELD parent_schema_id       ON cognitive_state TYPE option<string>;
DEFINE FIELD last_evidence_at       ON cognitive_state TYPE option<datetime>;
DEFINE FIELD created_at             ON cognitive_state TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_cog_brain_neuron   ON cognitive_state FIELDS brain_id, neuron_id UNIQUE;
DEFINE INDEX idx_cog_predicted      ON cognitive_state FIELDS brain_id, predicted_at;

-- Hot index (top-N high-priority neurons surfaced in context)
DEFINE TABLE hot_index SCHEMAFULL;
DEFINE FIELD brain_id           ON hot_index TYPE string;
DEFINE FIELD slot               ON hot_index TYPE int;
DEFINE FIELD category           ON hot_index TYPE string DEFAULT '';
DEFINE FIELD neuron_id          ON hot_index TYPE string;
DEFINE FIELD summary            ON hot_index TYPE string DEFAULT '';
DEFINE FIELD confidence         ON hot_index TYPE option<float>;
DEFINE FIELD score              ON hot_index TYPE float DEFAULT 0.0;
DEFINE FIELD updated_at         ON hot_index TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_hot_brain      ON hot_index FIELDS brain_id;
DEFINE INDEX idx_hot_score      ON hot_index FIELDS brain_id, score;

-- Knowledge gaps (identified missing knowledge areas)
DEFINE TABLE knowledge_gaps SCHEMAFULL;
DEFINE FIELD id                     ON knowledge_gaps TYPE string;
DEFINE FIELD brain_id               ON knowledge_gaps TYPE string;
DEFINE FIELD topic                  ON knowledge_gaps TYPE string DEFAULT '';
DEFINE FIELD detected_at            ON knowledge_gaps TYPE datetime DEFAULT time::now();
DEFINE FIELD detection_source       ON knowledge_gaps TYPE string DEFAULT '';
DEFINE FIELD related_neuron_ids     ON knowledge_gaps TYPE array<string> DEFAULT [];
DEFINE FIELD priority               ON knowledge_gaps TYPE float DEFAULT 0.5;
DEFINE FIELD resolved_at            ON knowledge_gaps TYPE option<datetime>;
DEFINE FIELD resolved_by_neuron_id  ON knowledge_gaps TYPE option<string>;
DEFINE INDEX idx_gaps_brain         ON knowledge_gaps FIELDS brain_id;
DEFINE INDEX idx_gaps_priority      ON knowledge_gaps FIELDS brain_id, priority;

-- Review schedules (Leitner-box spaced repetition by fiber)
DEFINE TABLE review_schedules SCHEMAFULL;
DEFINE FIELD fiber_id           ON review_schedules TYPE string;
DEFINE FIELD brain_id           ON review_schedules TYPE string;
DEFINE FIELD box                ON review_schedules TYPE int DEFAULT 1;
DEFINE FIELD next_review        ON review_schedules TYPE option<datetime>;
DEFINE FIELD last_reviewed      ON review_schedules TYPE option<datetime>;
DEFINE FIELD review_count       ON review_schedules TYPE int DEFAULT 0;
DEFINE FIELD streak             ON review_schedules TYPE int DEFAULT 0;
DEFINE FIELD created_at         ON review_schedules TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_review_brain   ON review_schedules FIELDS brain_id;
DEFINE INDEX idx_review_fiber   ON review_schedules FIELDS brain_id, fiber_id UNIQUE;
DEFINE INDEX idx_review_due     ON review_schedules FIELDS brain_id, next_review;

-- Memory maturation (STM -> Working -> Episodic -> Semantic, one record per fiber)
DEFINE TABLE maturation SCHEMAFULL;
DEFINE FIELD fiber_id                 ON maturation TYPE string;
DEFINE FIELD brain_id                 ON maturation TYPE string;
DEFINE FIELD stage                    ON maturation TYPE string DEFAULT 'stm';
DEFINE FIELD stage_entered_at         ON maturation TYPE datetime DEFAULT time::now();
DEFINE FIELD rehearsal_count          ON maturation TYPE int DEFAULT 0;
DEFINE FIELD reinforcement_timestamps ON maturation TYPE array<string> DEFAULT [];
DEFINE INDEX idx_maturation_brain     ON maturation FIELDS brain_id;
DEFINE INDEX idx_maturation_fiber     ON maturation FIELDS brain_id, fiber_id UNIQUE;
DEFINE INDEX idx_maturation_stage     ON maturation FIELDS brain_id, stage;

-- Brain versions (snapshot/checkpoint history with compressed snapshots)
DEFINE TABLE brain_versions SCHEMALESS;
DEFINE FIELD id              ON brain_versions TYPE string;
DEFINE FIELD brain_id        ON brain_versions TYPE string;
DEFINE FIELD version_name    ON brain_versions TYPE string DEFAULT '';
DEFINE FIELD version_number  ON brain_versions TYPE int DEFAULT 1;
DEFINE FIELD description     ON brain_versions TYPE string DEFAULT '';
DEFINE FIELD neuron_count    ON brain_versions TYPE int DEFAULT 0;
DEFINE FIELD synapse_count   ON brain_versions TYPE int DEFAULT 0;
DEFINE FIELD fiber_count     ON brain_versions TYPE int DEFAULT 0;
DEFINE FIELD snapshot_hash   ON brain_versions TYPE string DEFAULT '';
DEFINE FIELD snapshot_data   ON brain_versions TYPE string DEFAULT '';
DEFINE FIELD created_at      ON brain_versions TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata        ON brain_versions TYPE object DEFAULT {};
DEFINE INDEX idx_versions_brain  ON brain_versions FIELDS brain_id;
DEFINE INDEX idx_versions_number ON brain_versions FIELDS brain_id, version_number;

-- Keyword document-frequency (BM25 / TF-IDF ranking — fiber_count is # of fibers per keyword)
DEFINE TABLE keyword_document_frequency SCHEMAFULL;
DEFINE FIELD brain_id           ON keyword_document_frequency TYPE string;
DEFINE FIELD keyword            ON keyword_document_frequency TYPE string;
DEFINE FIELD fiber_count        ON keyword_document_frequency TYPE int DEFAULT 0;
DEFINE FIELD last_updated       ON keyword_document_frequency TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_kdf_brain      ON keyword_document_frequency FIELDS brain_id;
DEFINE INDEX idx_kdf_keyword    ON keyword_document_frequency FIELDS brain_id, keyword UNIQUE;

-- Entity references (per-fiber entity mentions, INSERT OR IGNORE on the triple)
DEFINE TABLE entity_refs SCHEMAFULL;
DEFINE FIELD brain_id       ON entity_refs TYPE string;
DEFINE FIELD entity_text    ON entity_refs TYPE string;
DEFINE FIELD fiber_id       ON entity_refs TYPE string;
DEFINE FIELD created_at     ON entity_refs TYPE datetime DEFAULT time::now();
DEFINE FIELD promoted       ON entity_refs TYPE bool DEFAULT false;
DEFINE INDEX idx_eref_brain   ON entity_refs FIELDS brain_id;
DEFINE INDEX idx_eref_entity  ON entity_refs FIELDS brain_id, entity_text;
DEFINE INDEX idx_eref_unique  ON entity_refs FIELDS brain_id, entity_text, fiber_id UNIQUE;

-- Compression backups (pre-compression originals — fiber level)
DEFINE TABLE compression_backups SCHEMAFULL;
DEFINE FIELD id                     ON compression_backups TYPE string;
DEFINE FIELD brain_id               ON compression_backups TYPE string;
DEFINE FIELD fiber_id               ON compression_backups TYPE string;
DEFINE FIELD original_content       ON compression_backups TYPE string DEFAULT '';
DEFINE FIELD compression_tier       ON compression_backups TYPE int DEFAULT 0;
DEFINE FIELD original_token_count   ON compression_backups TYPE int DEFAULT 0;
DEFINE FIELD compressed_token_count ON compression_backups TYPE int DEFAULT 0;
DEFINE FIELD compressed_at          ON compression_backups TYPE datetime DEFAULT time::now();
DEFINE FIELD created_at             ON compression_backups TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_cbk_unique         ON compression_backups FIELDS brain_id, fiber_id UNIQUE;
DEFINE INDEX idx_cbk_brain          ON compression_backups FIELDS brain_id;

-- Neuron snapshots (pre-compression originals — neuron level, tier 3-4)
DEFINE TABLE neuron_snapshots SCHEMAFULL;
DEFINE FIELD id               ON neuron_snapshots TYPE string;
DEFINE FIELD brain_id         ON neuron_snapshots TYPE string;
DEFINE FIELD neuron_id        ON neuron_snapshots TYPE string;
DEFINE FIELD original_content ON neuron_snapshots TYPE string DEFAULT '';
DEFINE FIELD compressed_at    ON neuron_snapshots TYPE string DEFAULT '';
DEFINE FIELD tier             ON neuron_snapshots TYPE int DEFAULT 3;
DEFINE FIELD created_at       ON neuron_snapshots TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_nsnap_unique ON neuron_snapshots FIELDS brain_id, neuron_id UNIQUE;
DEFINE INDEX idx_nsnap_brain  ON neuron_snapshots FIELDS brain_id;

-- Co-activations (Hebbian learning events)
DEFINE TABLE co_activations SCHEMAFULL;
DEFINE FIELD id               ON co_activations TYPE string;
DEFINE FIELD brain_id         ON co_activations TYPE string;
DEFINE FIELD neuron_a         ON co_activations TYPE string;
DEFINE FIELD neuron_b         ON co_activations TYPE string;
DEFINE FIELD binding_strength ON co_activations TYPE float DEFAULT 0.0;
DEFINE FIELD source_anchor    ON co_activations TYPE option<string>;
DEFINE FIELD created_at       ON co_activations TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_coact_brain  ON co_activations FIELDS brain_id;
DEFINE INDEX idx_coact_pair   ON co_activations FIELDS brain_id, neuron_a, neuron_b;
DEFINE INDEX idx_coact_time   ON co_activations FIELDS brain_id, created_at;

-- Action log (hippocampal buffer — habit learning)
DEFINE TABLE action_log SCHEMAFULL;
DEFINE FIELD id             ON action_log TYPE string;
DEFINE FIELD brain_id       ON action_log TYPE string;
DEFINE FIELD action_type    ON action_log TYPE string DEFAULT '';
DEFINE FIELD action_context ON action_log TYPE string DEFAULT '';
DEFINE FIELD tags           ON action_log TYPE array<string> DEFAULT [];
DEFINE FIELD session_id     ON action_log TYPE option<string>;
DEFINE FIELD fiber_id       ON action_log TYPE option<string>;
DEFINE FIELD created_at     ON action_log TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_alog_brain   ON action_log FIELDS brain_id;
DEFINE INDEX idx_alog_session ON action_log FIELDS brain_id, session_id;
DEFINE INDEX idx_alog_time    ON action_log FIELDS brain_id, created_at;

-- Depth priors (Bayesian adaptive retrieval depth selection)
DEFINE TABLE depth_priors SCHEMAFULL;
DEFINE FIELD id            ON depth_priors TYPE string;
DEFINE FIELD brain_id      ON depth_priors TYPE string;
DEFINE FIELD entity_text   ON depth_priors TYPE string;
DEFINE FIELD depth_level   ON depth_priors TYPE int;
DEFINE FIELD alpha         ON depth_priors TYPE float DEFAULT 1.0;
DEFINE FIELD beta          ON depth_priors TYPE float DEFAULT 1.0;
DEFINE FIELD total_queries ON depth_priors TYPE int DEFAULT 0;
DEFINE FIELD last_updated  ON depth_priors TYPE datetime DEFAULT time::now();
DEFINE FIELD created_at    ON depth_priors TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_dprior_unique ON depth_priors FIELDS brain_id, entity_text, depth_level UNIQUE;
DEFINE INDEX idx_dprior_brain  ON depth_priors FIELDS brain_id;
DEFINE INDEX idx_dprior_stale  ON depth_priors FIELDS brain_id, last_updated;

-- Tool events (staging buffer for tool-usage pattern mining / consolidation)
DEFINE TABLE tool_events SCHEMAFULL;
DEFINE FIELD id           ON tool_events TYPE string;
DEFINE FIELD event_id     ON tool_events TYPE string;
DEFINE FIELD brain_id     ON tool_events TYPE string;
DEFINE FIELD tool_name    ON tool_events TYPE string DEFAULT '';
DEFINE FIELD server_name  ON tool_events TYPE string DEFAULT '';
DEFINE FIELD args_summary ON tool_events TYPE string DEFAULT '';
DEFINE FIELD success      ON tool_events TYPE bool DEFAULT true;
DEFINE FIELD duration_ms  ON tool_events TYPE int DEFAULT 0;
DEFINE FIELD session_id   ON tool_events TYPE string DEFAULT '';
DEFINE FIELD task_context ON tool_events TYPE string DEFAULT '';
DEFINE FIELD processed    ON tool_events TYPE bool DEFAULT false;
DEFINE FIELD created_at   ON tool_events TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_tevt_brain   ON tool_events FIELDS brain_id;
DEFINE INDEX idx_tevt_unproc  ON tool_events FIELDS brain_id, processed;
DEFINE INDEX idx_tevt_eventid ON tool_events FIELDS brain_id, event_id;
DEFINE INDEX idx_tevt_time    ON tool_events FIELDS brain_id, created_at;

-- Reasoning traces (staging buffer for reasoning-trace mining / distillation)
DEFINE TABLE reasoning_traces SCHEMAFULL;
DEFINE FIELD id            ON reasoning_traces TYPE string;
DEFINE FIELD trace_hash    ON reasoning_traces TYPE string;
DEFINE FIELD brain_id      ON reasoning_traces TYPE string;
DEFINE FIELD model         ON reasoning_traces TYPE string DEFAULT '';
DEFINE FIELD session_id    ON reasoning_traces TYPE string DEFAULT '';
DEFINE FIELD project       ON reasoning_traces TYPE string DEFAULT '';
DEFINE FIELD task_context  ON reasoning_traces TYPE string DEFAULT '';
DEFINE FIELD content       ON reasoning_traces TYPE string DEFAULT '';
DEFINE FIELD content_chars ON reasoning_traces TYPE int DEFAULT 0;
DEFINE FIELD category      ON reasoning_traces TYPE string DEFAULT '';
DEFINE FIELD processed     ON reasoning_traces TYPE bool DEFAULT false;
DEFINE FIELD created_at    ON reasoning_traces TYPE datetime DEFAULT time::now();
DEFINE FIELD ingested_at   ON reasoning_traces TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_rtr_brain  ON reasoning_traces FIELDS brain_id;
DEFINE INDEX idx_rtr_hash   ON reasoning_traces FIELDS brain_id, trace_hash;
DEFINE INDEX idx_rtr_model  ON reasoning_traces FIELDS brain_id, model;
DEFINE INDEX idx_rtr_unproc ON reasoning_traces FIELDS brain_id, processed;
DEFINE INDEX idx_rtr_time   ON reasoning_traces FIELDS brain_id, created_at;

-- Document-training files (content-hash dedup + resume for `smem train`).
-- Purely additive, so it ships in SCHEMA_SQL rather than behind a SCHEMA_VERSION
-- bump: ensure_schema is idempotent and runs before apply_migrations on every
-- initialize(), so existing databases pick this up on their next start. Bumping
-- the version would break them — MIGRATIONS keys off TARGET_VERSION, so raising
-- it to 10 rewrites the (8, 9) entry to (8, 10) and strands every v8 database.
DEFINE TABLE training_files SCHEMAFULL;
DEFINE FIELD id               ON training_files TYPE string;
DEFINE FIELD brain_id         ON training_files TYPE string;
DEFINE FIELD file_hash        ON training_files TYPE string;
DEFINE FIELD file_path        ON training_files TYPE string DEFAULT '';
DEFINE FIELD file_size        ON training_files TYPE int DEFAULT 0;
DEFINE FIELD chunks_total     ON training_files TYPE int DEFAULT 0;
DEFINE FIELD chunks_completed ON training_files TYPE int DEFAULT 0;
DEFINE FIELD status           ON training_files TYPE string DEFAULT 'pending';
DEFINE FIELD domain_tag       ON training_files TYPE string DEFAULT '';
DEFINE FIELD trained_at       ON training_files TYPE option<datetime>;
DEFINE FIELD created_at       ON training_files TYPE datetime DEFAULT time::now();
DEFINE INDEX idx_tfile_brain  ON training_files FIELDS brain_id;
DEFINE INDEX idx_tfile_hash   ON training_files FIELDS brain_id, file_hash UNIQUE;
DEFINE INDEX idx_tfile_status ON training_files FIELDS brain_id, status;
"""


# Native RELATION edge DDL for the synapse table (schema v8). Single source of
# truth shared by ensure_schema() (fresh DBs) and the synapse->RELATE migration
# in migrations.py (which REMOVEs the old flat table then re-applies these).
# Endpoints are the built-in `in`/`out` edge fields; there is intentionally no
# `id`/`source_id`/`target_id` FIELD (RELATION supplies id; endpoints are in/out).
# NOT ENFORCED: orphan edges (endpoint neuron missing) are tolerated, matching
# the pre-migration behaviour.
SYNAPSE_V8_DDL: list[str] = [
    "DEFINE TABLE synapse TYPE RELATION IN neuron OUT neuron SCHEMAFULL",
    "DEFINE FIELD brain_id ON synapse TYPE string DEFAULT 'default'",
    "DEFINE FIELD type ON synapse TYPE string",
    "DEFINE FIELD weight ON synapse TYPE float DEFAULT 1.0",
    "DEFINE FIELD direction ON synapse TYPE string DEFAULT 'forward'",
    # FLEXIBLE is REQUIRED: synapse metadata carries arbitrary nested keys
    # (e.g. {"_dedup": true}). On a SCHEMAFULL table a plain `TYPE object` field
    # rejects any undefined nested key ("Found field 'metadata._dedup', but no
    # such field exists"), which silently skipped every synapse with non-empty
    # metadata during the v7->v8 migration. FLEXIBLE (after TYPE) allows nested keys.
    "DEFINE FIELD metadata ON synapse TYPE object FLEXIBLE DEFAULT {}",
    "DEFINE FIELD created_at ON synapse TYPE datetime DEFAULT time::now()",
    "DEFINE FIELD last_activated ON synapse TYPE option<datetime>",
    "DEFINE FIELD reinforced_count ON synapse TYPE int DEFAULT 0",
    "DEFINE INDEX idx_synapse_brain ON synapse FIELDS brain_id",
    "DEFINE INDEX idx_synapse_in ON synapse FIELDS brain_id, in",
    "DEFINE INDEX idx_synapse_out ON synapse FIELDS brain_id, out",
    "DEFINE INDEX idx_synapse_type ON synapse FIELDS brain_id, type",
]


def _parse_schema_statements(sql: str) -> list[str]:
    """Split a SurrealQL schema script into executable statements.

    Comment LINES are stripped inside each ``;``-separated chunk. The previous
    approach — dropping any whole chunk whose stripped text *started* with
    ``--`` — silently discarded every statement that sits directly under an
    explanatory comment block: all 26 ``DEFINE TABLE`` statements and, since
    2.7.4, the ``smem_content`` FULLTEXT analyzer. Without the analyzer the
    ``idx_neuron_content_fts`` DEFINE fails (and was swallowed below), so the
    ``@@`` operator behind ``find_neurons`` matched nothing — keyword recall
    was silently dead on any database relying on ``ensure_schema``.
    """
    statements: list[str] = []
    for chunk in sql.split(";"):
        stmt = "\n".join(
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        ).strip()
        if stmt:
            statements.append(stmt)
    return statements


async def ensure_schema(conn: Any, embedding_dim: int = 3072) -> None:
    """Apply schema to SurrealDB. Safe to call multiple times.

    The neuron embedding HNSW index dimension is parameterized by ``embedding_dim``
    so the vector index always matches the configured embedding model (e.g. 1024
    for bge-m3 via a local OpenAI-compatible server, 3072 for Gemini). SurrealDB
    rejects vectors whose length differs from the index dimension, so this MUST
    equal the provider's output dimension.

    Applies the monolithic SCHEMA_SQL first, then the parameterized neuron HNSW
    index, then the native-RELATION synapse DDL (SYNAPSE_V8_DDL). On an existing
    v7 database the flat ``synapse`` table still exists, so ``DEFINE TABLE synapse
    TYPE RELATION`` raises ``AlreadyExistsError`` here and is swallowed — the old
    table survives until ``apply_migrations`` (migrations.py) converts it. On a
    fresh database the RELATION table is created directly.

    Note: a plain ``DEFINE INDEX`` errors when the index already exists (swallowed
    below), so this does NOT change the dimension of an EXISTING index — it only
    sets it on first creation. Changing a populated index requires an explicit
    ``REMOVE INDEX`` + re-``DEFINE`` migration.
    """
    dim = int(embedding_dim) if embedding_dim and int(embedding_dim) > 0 else 3072
    logger.info("Applying SurrealDB schema (v%d, embedding_dim=%d)...", SCHEMA_VERSION, dim)
    statements = _parse_schema_statements(SCHEMA_SQL)
    statements.append(
        "DEFINE INDEX idx_neuron_embedding ON neuron "
        f"FIELDS embedding_vec HNSW DIMENSION {dim} DIST COSINE"
    )
    statements.extend(SYNAPSE_V8_DDL)
    for stmt in statements:
        try:
            await conn.query(stmt + ";")
        except Exception as exc:
            # A bare DEFINE on an existing index/table raises AlreadyExists on
            # every start (including the flat synapse table blocking the
            # RELATION re-definition on v7 — the migration handles conversion);
            # that stays at debug. Anything else used to vanish here — the
            # dropped-analyzer bug hid behind this very handler — so real
            # failures are now logged without breaking startup.
            head = stmt.splitlines()[0][:88]
            if "already exists" in str(exc).lower():
                logger.debug("Schema statement skipped (already exists): %s", head)
            else:
                logger.warning("Schema statement failed: %s (%s)", head, exc)
    logger.info("SurrealDB schema ready (v%d)", SCHEMA_VERSION)
