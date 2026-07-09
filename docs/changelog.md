# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Source of truth:** [`CHANGELOG.md`](https://github.com/acidkill/surreal-memory/blob/main/CHANGELOG.md)
> in the repository root. This page covers the **surreal-memory** (post-fork) release line. The full
> history — including the pre-fork upstream [neural-memory](https://github.com/nhadaututtheky/neural-memory)
> v4.x → v0.x — lives in that root file.

## [Unreleased]

## [2.7.3] — 2026-07-09

### Fixed

- **Dashboard endpoints are now fast on the large station brain** (follow-up to
  2.7.2, measured on 64k neurons / 266k synapses):
  - Parameterized `WHERE brain_id = $bid` defeated the `brain_id` index in
    SurrealDB 3.2.0 (only an inline literal uses it), so `count() … GROUP ALL`
    full-scanned the vector-laden neuron table (~2.5 s → 0.01 s). `get_stats`
    now inlines the validated brain_id and runs the counts concurrently.
  - Diagnostics skips the unused neuron type-breakdown (~2.6 s), runs its
    independent reads concurrently (`asyncio.gather`), and only fully analyzes
    the active brain in `/api/dashboard/stats`.
  - Net: `/api/dashboard/stats` ~27 s -> ~2.8 s, `/api/graph` 40 s+ -> ~4 s,
    `/api/dashboard/timeline` ~3 s -> ~0.06 s.
- **The web-UI container binds loopback only again**: `Dockerfile.surrealdb`
  hardcoded `uvicorn --host 0.0.0.0`, overriding the compose
  `SURREAL_MEMORY_HOST=127.0.0.1`. The CMD now honours `SURREAL_MEMORY_HOST` /
  `SURREAL_MEMORY_PORT`.

## [2.7.2] — 2026-07-08

### Fixed

- **Dashboard is fast again on large, fully-embedded brains.** After a full
  re-embed every neuron row carries a 1024-float `embedding_vec`, and the
  dashboard's `SELECT *` scans dragged those vectors (plus two unbounded
  full-table scans) into Python — Overview took ~27 s and the Graph view 40 s+
  on a 64k-neuron / 185k-synapse brain. Now:
  - `find_neurons(include_embedding=False)` projects `SELECT * OMIT embedding_vec`;
    the timeline and daily-stats endpoints use it (and push the time window into
    the query).
  - The graph view no longer loads every neuron and synapse: node degree is
    aggregated in the DB (`GROUP BY in/out` on the RELATE edge), the selected
    core's edges come from the indexed `->synapse` graph traversal, and only the
    rendered nodes are fetched (`find_neurons_by_ids`, no vectors).
  - Diagnostics (health grade/purity) replaces its unbounded synapse and
    neuron_state scans with DB aggregates (`get_connected_neuron_ids`,
    `count_activated_neuron_states`) and no longer duplicates the count queries.

### Changed

- **The web-UI container runs on host networking** (`docker-compose.surrealdb.yml`).
  The app shares the host loopback, so it reaches host-local services —
  llamastash embeddings/reranker on `127.0.0.1:11435` and SurrealDB via its
  published `127.0.0.1:8001` — without `host.docker.internal` and without
  binding llamastash to a docker bridge (smaller exposure). The dashboard now
  binds loopback-only by default (`SURREAL_MEMORY_HOST=127.0.0.1`); set it to
  `0.0.0.0` in `.env` to expose it on the LAN.

## [2.7.1] — 2026-07-08

### Fixed

- **Reranker no longer flip-flops on a shared brain.** Reranking is
  deployment/runtime config, but 2.7.0 persisted it onto the per-brain
  `BrainConfig` and re-applied it on every connect. With multiple clients on one
  brain (e.g. the CLI/MCP with reranking on and the web-UI container with it off),
  whichever connected last flipped the flag for everyone. Reranking is now read
  from the effective **app config** at recall time (`ReflexPipeline`), and
  `_migrate_brain_runtime_config` no longer touches the brain's reranker fields —
  each client uses its own endpoint independently.

### Changed

- **Docker Compose defaults to local BGE embeddings.** `docker-compose.surrealdb.yml`
  no longer hard-codes Gemini; it inherits the embedding provider/model/dimension
  from `.env` (defaulting to the OpenAI-compatible `bge-m3` on llamastash, 1024-dim)
  and reaches the host's llamastash via `host.docker.internal`, so the web UI and
  the CLI/MCP share one local embedding backend.

## [2.7.0] — 2026-07-07

### Added

- **Cross-encoder reranking is now a fully wired, config-driven recall stage.**
  Spreading activation over-fetches candidates, which are then reranked by a
  cross-encoder scoring `(query, memory)` relevance; the final ordering blends the
  reranker score with the activation level (`blend_weight`, default `0.7`). Enable
  it in `config.toml`:

  ```toml
  [reranker]
  enabled = true
  endpoint = "http://127.0.0.1:11435/v1"   # OpenAI-compatible /rerank (e.g. llamastash)
  model_name = "BAAI/bge-reranker-v2-m3"
  blend_weight = 0.7
  ```

- **HTTP reranking over an OpenAI-compatible `/rerank` server** (`HttpReranker`).
  The new `[reranker].endpoint` runs the cross-encoder on a shared inference server
  (e.g. llamastash / llama.cpp on GPU) instead of loading an in-process
  sentence-transformers model — reranking then needs no `torch` dependency. When
  the endpoint is unset it falls back to the `SURREAL_MEMORY_RERANKER_ENDPOINT`
  env var, then to a local `CrossEncoder`. Raw llama.cpp relevance logits are
  min-max normalised within the candidate set before blending. Reranking never
  breaks recall: any error falls back to the spreading-activation ordering.

### Fixed

- **Per-brain `BrainConfig` was never persisted.** `save_brain` stored a copy of the
  brain *metadata* in the `config` column, and `get_brain`/`find_brain_by_name` never
  loaded it, so every brain came back with a **default** config. This made the
  reranker — and any non-default per-brain retrieval knob — dead code. The full
  `BrainConfig` is now serialised on save and restored on load (unknown keys are
  dropped for forward/backward compatibility; legacy pre-2.7.0 rows that stored
  metadata in the `config` column fall back to defaults). `config.toml [reranker]` is
  also layered onto already-stored brains on connect, so enabling reranking takes
  effect without recreating a brain.

## [2.6.1] — 2026-07-07

### Fixed

- **CRITICAL: the v7→v8 synapse migration silently dropped every synapse with
  non-empty `metadata`.** The v8 `synapse` RELATION table is `SCHEMAFULL`, but its
  `metadata` field was defined as a plain `TYPE object`, which rejects arbitrary
  nested keys (e.g. `{"_dedup": true}`). On a real database the migration therefore
  skipped the majority of edges as "data loss" and `store.initialize()` aborted at
  the verification step. The field is now `TYPE object FLEXIBLE`, so nested metadata
  is preserved and the migration completes losslessly. Originals are always kept in
  `synapse_migration_backup` and the pre-migration data was never modified, so **no
  data was lost** — but a 2.6.0 upgrade could not complete. **Upgrade to 2.6.1
  before migrating an existing database.**
- The migration's `converting` phase now always rebuilds the RELATION table from the
  complete backup on entry (including on resume), so a migration interrupted by the
  above bug recovers every row instead of resuming past the ones it skipped.
- **The same `SCHEMAFULL` + plain-`TYPE object` gap affected every table with a
  `metadata`/`config` object field on a *fresh* database** — `neuron`, `fiber`,
  `brain` (config + metadata), `typed_memory`, `source`, `alerts` and
  `brain_versions`. Any write carrying a nested key (e.g. a structured `context`)
  would have been rejected. All object fields are now `TYPE object FLEXIBLE`.

## [2.6.0] — 2026-07-07

### BREAKING

- **Requires SurrealDB ≥ 3.2.0.** `store.initialize()` now hard-fails with a clear upgrade
  hint (`StorageVersionError`) when it detects an older server. **Back up the
  `surrealdb_data` volume before upgrading.**
- **The `synapse` graph auto-migrates to native RELATE edges on first connect** after the
  upgrade: the flat `source_id`/`target_id` columns become the built-in `in`/`out` edge
  endpoints. Existing synapse ids, `fiber.synapse_ids`, `change_log` entries and the Merkle
  root are preserved. The pre-migration rows are kept in a `synapse_migration_backup` table
  for rollback (clean up later with `smem doctor --synapse-migration purge-backup`).

### Added

- GQL-accelerated `get_path` shortest-path with an automatic BFS fallback — uses SurrealDB
  3.2's internal ISO GQL when the server exposes it (optional capability flags), and falls
  back to BFS otherwise.
- `smem doctor` **SurrealDB version check** (TIER_CORE; FAILs when the server is < 3.2.0).
- `smem doctor --synapse-migration {status|retry|purge-backup}` to inspect, resume, or
  clean up the synapse→RELATE migration.
- **Parametric embedding dimension** — the HNSW vector index (`idx_neuron_embedding`) is now built to
  match the embedding provider's output dimension. New `SURREAL_MEMORY_EMBEDDING_DIMENSION` env /
  `[embedding].dimension` config (`0` = auto-derive, the default). Fixes silently-broken semantic search
  when the index dimension disagreed with the model.
- **SurrealDB maturation storage** — maturation stages now persist on the SurrealDB backend (previously a
  base no-op), so long-lived memories report their real semantic maturity instead of 0%.

### Changed

- `docker-compose.surrealdb.yml` now runs `surrealdb/surrealdb:v3.2.0` with
  `--allow-experimental gql --allow-eval-query`. The datastore path must be given **before**
  the capability flags (the multi-valued `--allow-eval-query` would otherwise consume it).

### Performance

- **Semantic discovery** now ranks candidate pairs over each neuron's **stored** embedding instead of
  re-embedding on every run (vectorised top-K with a pure-python fallback) and raises the candidate caps —
  much cheaper and surfaces far more cross-domain links.
- **Edge-first graph selection** on the dashboard graph endpoint picks the most-connected nodes and keeps
  an edge only when both endpoints survive (`edge_cap=4000`), fixing the near-empty graph that
  node-capping by id produced.

### Fixed

- **Soft `forget` is excluded from recall immediately.** A soft-forgotten memory (expired `typed_memory`)
  no longer resurfaces in recall until the next consolidation — recall post-filters expired fibers and
  rebuilds the answer context from the survivors.
- **Config-cache refresh** — the REST process picks up new sync/embedding config after `set_config(...)`
  without a restart.
- **Rename-safe persistence** — cognitive/compression/review-schedule upserts and `consolidate` now target
  the current brain id after a rename instead of silently no-op'ing against a stale id.

### Known behaviour

- During the *converting* phase of the migration on a very large brain, synapse reads return
  empty until conversion completes (rows are paged in batches of 500). This window is brief
  for typical brains and the migration is crash-resumable.
- After the upgrade, external writers that insert **flat** rows (`source_id`/`target_id`)
  directly into the `synapse` table will fail — `synapse` is now a native RELATION and
  requires `in`/`out` edge endpoints (such writes were already violating the schema).

## [2.5.0] — 2026-06-23

### Added

- **`chat-heavy` config preset** for conversational agents (fast decay, recency-biased, compact). (#31)
- **`smem_offload` / `smem_inflate` / `smem_situation` MCP tools** — ephemeral tool-output offload +
  one-shot session snapshot (agent ergonomics). (#31)
- **`prefer_recent` recall flag** and **`verbose_extraction` remember flag**. (#31)
- **`[brain]` config extras pass-through** so new `BrainConfig` knobs are config-controllable. (#31, upstream #168)
- **Case-insensitive tag matching** at all write/read boundaries. (#33)
- **Dashboard Storage tab rebuilt for SurrealDB** — live backend status (URL, namespace, database,
  health, counts) via new `GET /api/dashboard/storage/status`. (#34)

### Changed

- **Lighter PostToolUse hook** (stdlib-only, noise filter, lock-safe append, Codex session id). (#32)
- **Plugin hook de-duplication** — plugin installs no longer double-register hooks. (#31, upstream #169)
- **MCP tool count is now 56** (was 53).

### Removed

- **Dead dashboard Storage migration UI** (`MigrationCard` / `MigrationProgress`) — SurrealDB-only,
  the migration flow was removed. (#34)

### Maintenance

- **Repository file-permission normalization** (`a4f27d0`, 2026-05-05): 963 tracked files had
  their mode bits drift from `100644` (regular) to `100755` (executable), likely due to a
  `core.fileMode` mismatch on the host filesystem. Zero content changes — this commit restores
  the intended permission state so future diffs reflect only real code changes.

## [2.4.0] — 2026-06-22

All changes in this release were contributed by [@RobertSigmundsson](https://github.com/RobertSigmundsson), who adopted surreal-memory as the production memory engine of the Uruboros multi-agent swarm. Huge thanks.

### Added

- `get_synapses(..., limit=None)` — optional cap on returned synapses, mirroring `find_neurons`. Bounds memory/latency on dense graphs (consolidation, replay). (#26)
- `GeminiEmbedding` honours `GOOGLE_GEMINI_BASE_URL` / `GOOGLE_GEMINI_API_VERSION` for gateway/proxy routing. (#27)

### Fixed

- **Activation persistence restored:** `neuron_state` records are addressed as `neuron_state:state_<sid>` on read/update/delete, matching the writer in `add_neuron`. The missing `state_` prefix made every state read miss and every update a silent no-op, leaving the activation→decay→tiering→consolidation loop dormant. (#29, re-scoped from #16)
- Never auto-prune pinned isolated (orphan) neurons; the pinned guard now covers both the orphan and dead-neuron prune paths. (#17)
- Pin `surrealdb` SDK to `>=2.0.0,<3.0.0`; the 2.x API is required and the old `>=0.4.0` floor allowed incompatible installs with opaque `AttributeError`s. (#18)
- `GeminiEmbedding.embed_batch` wraps each text in its own content, fixing N-texts→1-embedding under `google-genai >= 2.0` (which broke `reindex`). (#19)
- Tolerant neuron-type parsing; an unknown stored `type` falls back to `concept` with a warning instead of breaking recall for the whole brain. (#20)
- Remove leftover literal `{{}}` in nine `SCHEMA_SQL` DEFAULT clauses (invalid SurrealQL, so the DEFAULTs silently never applied). (#21)
- Default `synapse.brain_id` to `'default'` (was undeclared → NONE-coercion when omitted). (#22)
- `_to_surreal_id` strips an existing table prefix to prevent `neuron:neuron:…` id doubling (all three copies). (#23)
- Add `FORWARD`/`BACKWARD` to the `Direction` enum so the `'forward'` default in `_row_to_synapse` is valid (was a latent `ValueError`). (#24)
- Drop the write-only `connects_to` edge table; declare `source_id`/`target_id` on the synapse table and repoint the source/target indexes at those populated columns (Discussion #15, option A). (#25)

## [2.3.2] — 2026-06-01

### Fixed
- **SurrealDB auth fail-fast** — `SurrealDBStorage.initialize()` and `_reconnect()` now
  raise `StorageAuthError` (actionable) instead of propagating the raw SDK
  `NotAllowedError`. The MCP server surfaces this as JSON-RPC code `-32001` with a
  hint pointing to `SURREALDB_PASS` and `smem doctor --fix`, replacing the opaque
  `-32000 "failed unexpectedly"` that made bad-credential failures invisible.
- **Default password unified** — the silent default `SURREALDB_PASS=root` (which never
  matched the Docker default `surrealmemory`) is replaced by a single source of truth
  in `storage/surrealdb/connection.py`. Both `store.py` and `unified_config.py` now
  derive the default from this module, eliminating the drift that caused clean-install
  auth failures.

### Added
- **`storage/surrealdb/connection.py`** (new module) — `SurrealSettings.from_env()`,
  `StorageAuthError`, `is_credential_error()`, `build_mcp_env()`; single source of
  truth for all SurrealDB connection defaults.
- **Claude Desktop MCP support** — `smem init` and `smem setup mcp` now write the
  `surreal-memory` entry (including the full `env` block with `SURREALDB_PASS`) to
  `claude_desktop_config.json` on Linux, macOS, and Windows. Existing entries without
  `env` are backfilled automatically.
- **`env` block in all MCP configs** — `find_smem_command()` always returns an `env`
  dict so newly written Claude Code and Cursor configs include SurrealDB connection
  variables, preventing the "empty env" bug on clean installs.
- **`smem doctor` SurrealDB checks** — two new diagnostic checks:
  - `SurrealDB connection` (TIER_CORE): live auth test with 5-second timeout; FAIL
    with actionable fix on `StorageAuthError`.
  - `MCP env completeness` (TIER_RECOMMENDED): verifies `SURREALDB_PASS` is present
    in the `env` block of each MCP client config.
  - `smem doctor --fix` backfills missing env in all detected client configs.
- **`_warn_missing_surreal_pass()`** — one-time warning when `storage=surrealdb` is
  active but `SURREALDB_PASS` is unset.

### Changed
- `_check_brain` and `_check_schema_version` in `smem doctor` now return `SKIP`
  (not `FAIL`) when the SurrealDB backend is active — those checks are SQLite-only.
- `setup_mcp_claude()` uses JSON write path exclusively (the `claude mcp add` CLI
  does not support the `env` block). Behaviour from the user perspective is identical.
- `SURREALDB_PASS` default (`surrealmemory`) documented in installation and
  contributing guides.

## [2.3.1] — 2026-05-31

### Fixed
- **Dashboard ⇄ CLI metric parity** — `SurrealDBStorage.get_enhanced_stats` now
  returns a `synapse_stats` block (per-type counts), so `DiagnosticsEngine`
  computes `diversity` and `recall_confidence` on the SurrealDB backend exactly
  as it does on SQLite. Previously both were `0` on SurrealDB, so the dashboard
  and the `smem` CLI reported different health grades (e.g. F vs D) for the same
  brain.
- **Consistent brain grade across endpoints** — `/api/dashboard/brains` now runs
  diagnostics like `/api/dashboard/stats`, so the Brains table and the stats
  cards report the same grade. Per-brain analysis runs sequentially to avoid
  racing the shared SurrealDB storage singleton.
- **Resilient SurrealDB connection** — `SurrealDBStorage._query` re-authenticates
  and retries once on an expired/closed connection (HTTP 401), so long-lived MCP
  and CLI processes survive SurrealDB restarts and root-token expiry instead of
  failing every subsequent call.
- **Accurate orphan rate** — `DiagnosticsEngine.analyze` pins the storage brain
  context before its reads, preventing a false high orphan rate when multiple
  brains are analyzed concurrently.

### Added
- **SQLite misconfiguration guard** — emit a loud, one-time warning when the
  active storage backend resolves to SQLite, with a targeted message when
  SurrealDB connection vars are set. surreal-memory targets SurrealDB; this
  surfaces the "memories silently written to a local SQLite brain that diverges
  from the SurrealDB the dashboard reads" footgun instead of failing silently.

### Changed
- Pin the `surrealdb` Docker image to `v3.1.1` in `docker-compose.surrealdb.yml`.

## [2.3.0] — 2026-05-29

### Added
- **SurrealDB tool-event storage** — new `tool_events` table (schema v6) brings the
  SurrealDB backend to parity with SQLite. Powers the dashboard **Tool Stats** page
  and consolidation's tool-usage pattern mining on the SurrealDB backend (previously
  raised `AttributeError`).

### Fixed
- **Dashboard is fully free** — removed leftover Pro-tier gating that survived the
  SurrealDB-only switch. **Evolution** and **Visualize** no longer show a "PRO FEATURE"
  overlay, the **Embedding Provider** settings are editable (no 403), and **Settings →
  General** reports a `FULL` license with no upgrade prompt.
- **Storage page** — rebuilt for the SurrealDB-only model. It now shows the active
  SurrealDB backend, neuron/synapse/fiber counts, and tier distribution from the live
  `/stats` + `/tier-stats` endpoints, instead of calling the removed `/storage/status`
  endpoint that left the page blank.
- **Brain lookup by name** — `SurrealDBStorage.get_brain` now matches the `name` field,
  fixing an orphan-row leak where the bootstrap re-created a fresh brain on every start
  (active brain reported 0 neurons even when the store held data).
- **Dashboard brain enumeration** — `/api/dashboard/stats` and `/api/dashboard/brains`
  now list brains from the active SurrealDB store (`list_available_brains`) instead of
  only local SQLite fixture files, so the dashboard no longer shows zero brains.

### Changed
- `UnifiedConfig.is_pro()` always returns `True` and `/api/dashboard/license` reports the
  `full` tier — Surreal-Memory is fully free; every feature is unlocked for everyone.
- A fresh process now honors `SURREAL_MEMORY_STORAGE` before a `config.toml` exists, so it
  no longer caches a SQLite singleton while the environment asks for SurrealDB.

## [2.2.0] — 2026-05-28

### Added
- **Embedding env overrides** — the unified config now honors
  `SURREAL_MEMORY_EMBEDDING_ENABLED` / `_PROVIDER` / `_MODEL` /
  `_SIMILARITY_THRESHOLD` (precedence: env > `config.toml` > default), so the
  MCP server and CLI follow the embedding provider set in their environment.
- **`smem reindex`** — (re)embed a brain's neurons with the effective provider.
  Flags: `--dry-run`, `--missing-only` (default) / `--all`, `--batch-size`;
  idempotent and fail-soft per neuron.

### Changed
- **Effective config wins** — embedding `enabled`/`provider`/`model` now resolve
  from the effective config (`config.toml` + env) instead of the stale stored
  `brain.config`. Fixes embeddings silently staying disabled after a user edits
  their config/env. `smem_health` now reports the effective embedding state.

### Performance
- The Stop hook no longer loads a local `sentence-transformers` model on every
  session end (it was the dominant session-save latency). Semantic dedup uses a
  local Ollama server when one is running, otherwise it is skipped.

## [2.1.0] — 2026-05-28

### Added
- **Project-aware memory hooks** — SessionStart, PreCompact, and Stop hooks scope
  captured memories to the current project (git repo basename as `project_id`);
  SessionStart injects only the current project's memories.
- **Task-context hook** — new `smem-hook-task-context` entry point persists a rich,
  structured per-task note as one project-scoped `context` memory.
- **SurrealDB Project entity** — `add_project` / `get_project` / `get_project_by_name`
  / `list_projects` / `update_project` / `delete_project` restored on the SurrealDB
  backend (parity broken by the v2.0.0 SurrealDB-only refactor); new `project` table,
  schema version 5.
- `get_project_memories` declared on the `NeuralStorage` base interface.

### Fixed
- **Connection close** — `SurrealDBStorage.close()` tolerates transports that don't
  implement `close()` (the HTTP connection raises `NotImplementedError`), fixing a
  long-running MCP server degrading to "No brain configured".
- **CLI regression** — restored `surreal_memory/utils/sandbox.py`; `smem` no longer
  fails with `ModuleNotFoundError` (regression from the v2.0.0 refactor).
- **Embedding pipeline hardening** — retry/backoff, embedding-capability probe, and
  removal of a decommissioned default model.
- **Latent recall bug** — save hooks persist the verbatim text as the fiber summary,
  so SessionStart actually injects context (previously `fiber.summary`/`essence` were
  always `None`).
- Cross-backend parity test fixture (`connect()` → `initialize()`, and skip when the
  optional `surrealdb` package is absent).

### Documentation
- README: new **Embeddings** section (Gemini `gemini-embedding-001` recommended; local
  `sentence-transformers` `all-MiniLM-L6-v2` / `paraphrase-multilingual-MiniLM-L12-v2`
  as the no-API-key fallback; Ollama / OpenAI / OpenRouter; `auto` detection).
- Fixed rename-rot ("What's Different From NeuralMemory?", `~/.neuralmemory` migration)
  and corrected counts: 15 memory types, 41 synapse types, 5500+ tests.
- INSTALL_PROMPT: fixed stale repository URLs; Gemini recommended (not required) with a
  documented local no-key path.
- `.env.example`, `AGENTS.md`, `CONTRIBUTING.md` corrections.

## [2.0.0] — 2026-05-27

### Removed — InfinityDB Pro chain + SQLite/InMemory demoted to test fixtures (BREAKING)

Surreal-Memory is now **SurrealDB-only** on the public surface. The
InfinityDB Pro plugin chain is gone; SQLite and InMemory remain in the
tree but only as internal test infrastructure — they are no longer
documented, no longer offered through the CLI, and no longer reachable
through any public configuration path.

Deletions:
- `src/surreal_memory/cli/commands/migrate.py` (no alternative backends
  to migrate to).
- `tests/unit/test_infinitydb_integration.py`.
- `tests/unit/test_storage_migration_api.py`.
- ~400 lines of Storage Management code in
  `src/surreal_memory/server/routes/dashboard_api.py`:
  `MigrationJobStatus`, `StorageStatusResponse`,
  `StartMigrationRequest`, `SetBackendRequest`, the
  `GET /storage/status`, `POST /storage/migrate`,
  `GET /storage/migrate/{job_id}`, `POST /storage/backend` endpoints,
  `_run_migration_task`, `_open_sqlite_storage`,
  `_open_infinitydb_storage`.

Code surgery:
- `src/surreal_memory/cli/main.py`: stopped importing/registering
  `migrate`.
- `src/surreal_memory/cli/commands/storage.py`: rewritten. Only
  `smem storage status` remains; it probes the SurrealDB connection
  instead of describing SQLite/InfinityDB files. `storage switch` is
  gone — nothing to switch between.
- `src/surreal_memory/cli/commands/shared.py`: removed the
  "Pro activated -> upgrade to InfinityDB" hint block.
- `src/surreal_memory/engine/consolidation.py`: removed
  `ConsolidationStrategy.SMART_MERGE` and `_smart_merge_pro`.
- `src/surreal_memory/engine/retrieval.py`: `"cone"` strategy now logs
  a debug message and falls back to classic activation.
- `src/surreal_memory/mcp/stats_handler.py`,
  `src/surreal_memory/mcp/sync_handler.py`,
  `src/surreal_memory/server/app.py`: removed every "Pro tip:
  InfinityDB ..." upsell.
- `src/surreal_memory/unified_config.py`: removed
  `_get_infinitydb_storage`, the `infinitydb` dispatch branch, and the
  InfinityDB-directory fall-through in `list_brains()`.
- `src/surreal_memory/storage/factory.py`: dropped `_try_pro_storage`.
- `src/surreal_memory/plugins/__init__.py`,
  `src/surreal_memory/plugins/base.py`,
  `src/surreal_memory/plugins/community.py`: dropped
  `get_storage_class()`.

Test-only surface markings:
- `src/surreal_memory/storage/sqlite_store.py` and
  `src/surreal_memory/storage/memory_store.py` now carry an explicit
  TEST FIXTURE ONLY header so contributors don't mistake them for
  production paths.

Config and docs:
- `.env.example`: `SURREAL_MEMORY_STORAGE=surrealdb` is uncommented and
  the comment explains that `sqlite` is not a production option.
- `docs/landing/pro.md`: dropped the InfinityDB row from the backend
  table.
- `ROADMAP.md`: current-state line updated.
- `dashboard/src/i18n/en.json` + `vi.json`: dropped
  `storage.infinitydb*` / `enableInfinitydb` / migration UI strings.
- `docs/getting-started/cli-reference.md`, `docs/api/mcp-tools.md`:
  regenerated.

Verification:
- `ruff check src/ tests/` clean.
- `mypy src/ --ignore-missing-imports` clean (334 files).
- `pytest --co tests/unit`: 5515 tests collected, zero import errors.
- `pytest test_unified_config + test_dx_wizard + test_brain_isolation +
  test_health_fixes`: 93/93 passed locally.

**BREAKING CHANGE:** anyone running the Pro InfinityDB chain on v1.x
must export their brain to JSON and re-import on SurrealDB before
upgrading. `smem migrate` and `smem storage switch` are gone — point
users at `docker-compose.surrealdb.yml` instead.

### Removed — FalkorDB and PostgreSQL backends (BREAKING)

Surreal-Memory is **SurrealDB-only** from v2.0.0 onwards. The opt-in
FalkorDB and PostgreSQL backends added in upstream v4.7 are gone:

- Deleted `src/surreal_memory/storage/falkordb/` (8 mixin files + store).
- Deleted `src/surreal_memory/storage/postgres/` (11 mixin files + store).
- Deleted `docker-compose.falkordb.yml` and `docker-compose.postgres.yml`.
- Deleted `scripts/postgres-init.sh`.
- Deleted FalkorDB integration test `tests/integration/test_falkordb_spreading.py`.
- Deleted FalkorDB storage tests in `tests/storage/test_falkordb_*.py` (5 files)
  and the entire `tests/storage/postgres/` suite (5 files + conftest).
- Deleted `tests/unit/test_postgres_migration.py`.

Code paths trimmed:

- `pyproject.toml`: dropped `[project.optional-dependencies] falkordb` and
  `postgres` extras (and the matching ruff per-file-ignores rule).
- `src/surreal_memory/unified_config.py`: removed `FalkorDBConfig`,
  `PostgresConfig`, `_get_falkordb_storage`, `_get_postgres_storage`, the
  cached module globals, the TOML serializers, and the dispatch branches.
  `_VALID_STORAGE_BACKENDS` is now `{"sqlite", "surrealdb"}` (InfinityDB
  remains available via the Pro plugin).
- `src/surreal_memory/utils/config.py`: dropped `falkordb_*` fields.
- `src/surreal_memory/storage/__init__.py`: removed lazy `__getattr__`
  branches for `FalkorDBStorage` / `PostgreSQLStorage`.
- `src/surreal_memory/cli/commands/migrate.py`: rewritten — only
  `infinitydb` and `sqlite` (no-op) targets remain; FalkorDB/Postgres
  targets emit a deprecation hint pointing to docker-compose.surrealdb.yml.
- `src/surreal_memory/cli/commands/storage.py`: help text reads
  "SQLite, SurrealDB, InfinityDB" only.
- `docker-compose.yml`: removed the `falkordb` optional service; readers
  are routed to `docker-compose.surrealdb.yml`.
- `.env.example`: removed `FALKORDB_*` block; storage options list shows
  `sqlite, surrealdb`.

Docs synced:

- `ROADMAP.md`: replaced "PostgreSQL Backend Parity" milestone with
  "SurrealDB Backend Parity"; updated current-state line and the C1
  tiered storage section.
- `docs/contributing.md`, `docs/FAQ.md`, `docs/landing/pro.md`,
  `docs/promo/reddit-localllama.md`: backend table and prose updated to
  reflect the new surface.
- `docs/getting-started/cli-reference.md`: regenerated from the new
  `smem migrate` signature.

**Migration:** if you were running on PostgreSQL or FalkorDB on v1.x,
export your brain to JSON before upgrading and re-import on SurrealDB
(or stay on v1.x — that line is still supported for one minor release).

### Fixed — Concept Neuron Noise Filtering (#156)

Short and casual text no longer creates low-signal concept neurons that pollute
recall context. `ExtractConceptNeuronsStep` now:

- Raises min keyword length from 3 to 4 chars (filters `AI`, `OS`, `It`)
- Scales concept floor from 5 to 3 for content under 100 chars
- Skips keywords already captured as entity neurons (avoids duplicates)
- Filters known noise words (`use`, `run`, `new`, `got`, etc.)

Aligns with the F2 Fiber Precision & Density roadmap item.

### Fixed — Advisory hints stripped from machine output (#155)

CLI update notices are now skipped for machine-oriented commands
(`context`, `recall`, `stats`, `status`) and any invocation with `--json`.
MCP `strip_hints` now strips advisory fields even in non-compact mode.
Adds an Agent Memory Governance guide.

### Added — Contributor dev diagnostics (#154)

`smem doctor --dev` now reports source checkout detection, editable install
status, dev dependencies, and checkout/package version parity for
contributors working from a source checkout.

### Fixed — Coroutine warning on sandbox fail-fast (#153)

CLI commands that fail fast in restricted sandboxes no longer emit
`RuntimeWarning: coroutine was never awaited`. The unawaited command
coroutine is now explicitly closed before re-raising the sandbox exit.

## [1.0.0] — 2026-05-04

First stable release of **surreal-memory** as an independent PyPI package. This version forks from
surreal-memory 4.24.0, replaces SQLite with SurrealDB, and unlocks all Pro-tier features for free
via the bundled community plugin.

### Added

- **SurrealDB storage backend — fully implemented** (163/163 methods): Ten mixin classes covering
  typed memory, sources, alerts, cognitive state, review schedules, versioning, keyword/entity
  extraction, compression, activity tracking, and depth priors.
- **`get_project_memories` parity**: `SurrealDBTypedMemoryMixin` implements `get_project_memories`
  matching the `SQLiteTypedMemoryMixin` signature.
- **Memory type classifier expansion**: `suggest_memory_type()` extended from 9 to 12 covered
  types. New branches: `BOUNDARY`, `TOOL`, `CONTEXT`.
- **`INSTALL_PROMPT.md`**: 9-step Claude Code installation prompt covering prerequisites, Docker
  setup, pipx install, env config, MCP registration, doctor verification, and CLAUDE.md injection.
- **Community plugin** (`src/surreal_memory/plugins/community.py`): Bypasses Pro feature gates.
  Provides cone queries (HNSW vector search), smart merge, and directional compression.

### Fixed

- **`ensure_schema` never applied on connect** (F821 runtime bug): Schema now correctly
  initialises on every SurrealDB connection.
- **Mypy / ruff clean build**: Removed unused locals, typed `_max()`, added `# noqa` for
  intentionally naive datetime sentinels.
- **SQLite FK constraint in tests**: Test suite now creates `projects` rows before seeding
  `typed_memory` rows.
- **Taskmaster project-locality**: CWD-walking resolver — each project uses its own
  `.taskmaster/tasks.json`.

### Improved

- **`docs/getting-started/installation.md`** rewritten with accurate surreal-memory instructions.
- **`pyproject.toml` project URLs** corrected to `acidkill/surreal-memory-surrealdb-version`.
- **`README.md`** Quick Start: automated setup via Claude Code listed first; badge URLs corrected.

### Tests

- **150+ new parametrised tests** across four new test files covering `get_project_memories`
  parity, `suggest_memory_type` coverage (128 tests), remember-handler all-types (19 tests), and
  SurrealDB typed-memory integration (31 tests, skipped without `SURREALDB_URL`).

---

> **Earlier history.** Releases before the fork (upstream neural-memory **v4.53.4 → v0.x**) are not
> repeated here. See the full [`CHANGELOG.md`](https://github.com/acidkill/surreal-memory/blob/main/CHANGELOG.md)
> in the repository root.
