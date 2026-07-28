# Architecture Overview

Surreal-Memory's layered architecture for memory management, backed by SurrealDB's
multi-model (document + graph + vector) engine as the production storage backend.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI / MCP Server / REST API               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Encoder    │  │  Retrieval   │  │   Lifecycle  │       │
│  │              │  │   Pipeline   │  │   Manager    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  DocTrainer  │  │  DBTrainer   │  │Consolidation │       │
│  │              │  │              │  │   Engine     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Context     │  │  Query       │  │  Alert       │       │
│  │  Optimizer   │  │  Patterns    │  │  Handler     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    Extraction Layer                          │
│  ┌───────────┐  ┌────────────┐  ┌──────────────────┐        │
│  │QueryParser│  │QueryRouter │  │TemporalExtractor │        │
│  └───────────┘  └────────────┘  └──────────────────┘        │
├─────────────────────────────────────────────────────────────┤
│              Storage Interface (NeuralStorage ABC)            │
│  ┌─────────────────┐  ┌───────────────────┐  ┌─────────────┐│
│  │    SurrealDB     │  │  SQLite/InMemory  │  │SharedStorage││
│  │ (production —   │  │  (test fixtures    │  │  (HTTP,     ││
│  │  doc+graph+vec) │  │   only, see below) │  │  BrainMode  ││
│  │                 │  │                    │  │ .SHARED)    ││
│  └─────────────────┘  └───────────────────┘  └─────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Core Layer                                │
│  ┌──────┐  ┌───────┐  ┌─────┐  ┌─────┐  ┌───────────┐      │
│  │Neuron│  │Synapse│  │Fiber│  │Brain│  │TypedMemory│      │
│  └──────┘  └───────┘  └─────┘  └─────┘  └───────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Layers

### Interface Layer

Entry points for users and applications:

- **CLI** - Command-line interface (`smem` commands)
- **MCP Server** - Model Context Protocol for Claude integration (58 tools)
- **REST API** - FastAPI-based HTTP server, backs the web dashboard at `/ui`

### Engine Layer

`src/surreal_memory/engine/` has grown to ~90 modules. Grouped by responsibility:

- **Encoding** - `encoder.py` (MemoryEncoder), `pipeline.py`/`pipeline_steps.py` (composable
  async steps), `chunking.py`, `doc_chunker.py`/`doc_extractor.py`, `codebase_encoder.py`,
  `idf_anchor.py` (IDF-weighted keyword synapses)
- **Retrieval & Activation** - `retrieval.py` (ReflexPipeline), `activation.py` (classic
  spreading activation), `reflex_activation.py` (trail-based `ReflexActivation` +
  `CoActivation`), `ppr_activation.py` (Personalized PageRank, opt-in), `score_fusion.py`
  (RRF multi-retriever fusion), `reranker.py` (optional cross-encoder reranking),
  `sufficiency.py` (algorithmic sufficiency gate), `causal_traversal.py`
- **Lifecycle & Neuroscience** - `lifecycle.py` (decay/reinforcement), `consolidation.py`/
  `consolidation_delta.py` (sleep strategies), `dream.py`, `reflection.py`,
  `hippocampal_replay.py`, `reconsolidation.py`, `interference.py`, `prediction_error.py`,
  `temporal_binding.py`, `schema_assimilation.py`, `memory_stages.py`, `tier_engine.py`
  (HOT/WARM/COLD), `spaced_repetition.py`
- **Training** - `doc_trainer.py`, `db_trainer.py`, `db_introspector.py`, `db_knowledge.py`,
  `file_watcher.py`/`watch_state.py`
- **Sync & Multi-Brain** - `merge.py`, `cross_brain.py`, `brain_transplant.py`,
  `brain_versioning.py`, `conflict_detection.py`/`conflict_auto_resolve.py`
- **Cognitive Reasoning** - `cognitive.py`, `associative_inference.py`,
  `drift_detection.py`, `semantic_discovery.py`, `decision_intel.py`
- **Embeddings** - `embedding/` subpackage (Gemini, OpenAI, OpenRouter, local
  sentence-transformers, Ollama providers)
- **Context & Observability** - `context_optimizer.py`, `context_retrieval.py`,
  `context_merger.py`, `token_budget.py`, `diagnostics.py`, `brain_evolution.py`,
  `query_pattern_mining.py`, `chart_generator.py`, `narrative.py`

### Extraction Layer

NLP and parsing utilities:

- **QueryParser** (`parser.py`) - Decomposes queries into signals
- **QueryRouter** (`router.py`) - Determines query intent and depth
- **TemporalExtractor** (`temporal.py`) - Extracts time references
- **EntityExtractor** (`entities.py`), **keywords.py** - Entity/keyword extraction
- **structure_detector.py**, **relations.py**, **sentiment.py**, **codebase.py**,
  **llm_provider.py** - Structured data detection, relation extraction, sentiment,
  codebase indexing, optional LLM-assisted extraction

### Storage Layer

Pluggable storage backends implementing `NeuralStorage`. **SurrealDB is the production
backend**; SQLite and InMemory are retained solely as test fixtures (see `ROADMAP.md` and
the [Storage Selection](#storage-selection) note below).

- **SurrealDBStorage** (`storage/surrealdb/`) - Multi-model production backend: neurons as
  documents, synapses as native SurrealDB `RELATE` graph edges, HNSW vector search over
  embeddings — all in one SurrealDB instance
- **SQLiteStorage** (`storage/sqlite_store.py` + mixins) - Persistent, single-process; test
  fixture only. A parity pass (ROADMAP "A1: SurrealDB Backend Parity") tracks remaining gaps
  between the SQLite and SurrealDB feature surfaces
- **InMemoryStorage** (`storage/memory_store.py`) - NetworkX-based; test fixture only
- **SharedStorage** (`storage/shared_store.py`) - HTTP client for a remote Surreal-Memory
  server (`BrainMode.SHARED`)
- **HybridStorage** (`storage/factory.py`) - Offline-first: writes to local SQLite, syncs to
  a remote server (`BrainMode.HYBRID`)

#### Storage Selection

There are two independent storage-selection mechanisms in the codebase — worth calling out
explicitly since they don't compose:

1. **`unified_config.get_shared_storage()`** (`src/surreal_memory/unified_config.py`) - the
   real entry point used by the CLI, MCP server, and REST server. It branches on
   `config.storage_backend` (`"surrealdb"` or `"sqlite"`, set via `SURREAL_MEMORY_STORAGE`
   or `config.toml`) and returns either a cached `SurrealDBStorage` or `SQLiteStorage`.
   `storage_backend` defaults to `"sqlite"` for backward compatibility, but production setups
   must set it to `"surrealdb"` — the CLI (`smem doctor`) and dashboard warn loudly if
   SurrealDB connection env vars are present while `storage_backend` is still `"sqlite"`,
   since that silently splits memories across two stores.
2. **`storage.factory.create_storage(BrainModeConfig, ...)`** - a lower-level Python API
   keyed off `BrainMode` (`LOCAL` → SQLite/InMemory, `SHARED` → `SharedStorage` HTTP client,
   `HYBRID` → `HybridStorage`). This path has no `SURREALDB` mode and never returns a
   `SurrealDBStorage` — it predates the SurrealDB migration and is used directly only by the
   lower-level Python API (see README "Python API" section), not by the CLI/MCP/server.

`SurrealDBStorage` is intentionally not re-exported from `storage/__init__.py` — import it
directly from `surreal_memory.storage.surrealdb`.

### Core Layer

Fundamental data structures (`src/surreal_memory/core/`):

- **Neuron** - Atomic information unit
- **Synapse** - Typed connection between neurons (41 types)
- **Fiber** - Signal pathway with conductivity
- **Brain** / **BrainConfig** - Container with configuration
- **TypedMemory** (`memory_types.py`) - Metadata layer for memories (type, priority, tier)
- **BrainMode** / **SharedConfig** / **HybridConfig** (`brain_mode.py`) - `create_storage()`
  mode toggle (see [Storage Selection](#storage-selection))
- **Alert**, **Project**, **Source**, **ReviewSchedule** - Supporting entities for proactive
  alerts, project scoping, source-aware memory, and spaced-repetition review

## Data Flow

### Encoding Flow

```
Input Text
    │
    ▼
┌─────────────────┐
│  QueryParser    │  Extract entities, time, concepts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MemoryEncoder  │  Create neurons and synapses
│                 │  · CreateSynapsesStep: IDF-weighted keywords (B4)
│                 │  · CrossMemoryLinkStep: RELATED_TO via shared entities (B3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Storage        │  Persist via NeuralStorage — SurrealDB in production
│                 │  (SQLite schema v38 / SurrealDB schema v8 as test fixtures)
└─────────────────┘
```

### Retrieval Flow

```
Query
    │
    ▼
┌─────────────────┐
│  QueryParser    │  Decompose query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QueryRouter    │  Determine depth, intent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Find Anchors   │  Time-first anchor selection
│  (Time-First)   │  Time(1.0) → Entity(0.8) → Action(0.6)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Find Fibers    │  Get fiber pathways for anchors
└────────┬────────┘
         │
         ├─── reflex mode ──┐
         │                   ▼
         │          ┌─────────────────┐
         │          │  Trail          │  Activate along fiber
         │          │  Activation     │  pathways with decay
         │          └────────┬────────┘
         │                   │
         │                   ▼
         │          ┌─────────────────┐
         │          │  Co-Activation  │  Hebbian binding
         │          └────────┬────────┘
         │                   │
         ├─── classic mode ──┤
         │                   │
         │          ┌─────────────────┐
         │          │  Spreading      │  BFS with decay
         │          │  Activation     │
         │          └────────┬────────┘
         │                   │
         ├───────────────────┘
         │
         ▼
┌─────────────────┐
│  Score Fibers   │  Fiber-level recall scoring (B5)
│                 │  base_quality × activation_signal × stage_multiplier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract        │  Build response context
│  Subgraph       │  · Contextual compression by age (B6)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reinforce      │  Update fiber conductivity
│  Fibers         │  · Adaptive synapse decay (B8)
└─────────────────┘
```

## Storage Interface

All storage backends implement `NeuralStorage` (`src/surreal_memory/storage/base.py`). The
interface has grown well beyond the original CRUD surface — it now also declares (mostly
non-abstract, default-`NotImplementedError` or no-op) hooks for typed memories, cognitive
reasoning (hypotheses/evidence/predictions/knowledge gaps), proactive alerts, brain
versioning, compression backups, Merkle-hash sync, entity-ref lazy promotion, co-activation
tracking, and review schedules. The core abstract methods remain:

```python
class NeuralStorage(ABC):
    # Neuron operations
    async def add_neuron(self, neuron: Neuron) -> str
    async def get_neuron(self, neuron_id: str) -> Neuron | None
    async def find_neurons(self, **filters) -> list[Neuron]

    # Synapse operations
    async def add_synapse(self, synapse: Synapse) -> str
    async def get_synapses(self, **filters) -> list[Synapse]

    # Graph traversal
    async def get_neighbors(self, neuron_id: str, ...) -> list[tuple]

    # Fiber operations
    async def add_fiber(self, fiber: Fiber) -> str
    async def get_fiber(self, fiber_id: str) -> Fiber | None

    # Brain operations
    async def export_brain(self, brain_id: str) -> BrainSnapshot
    async def import_brain(self, snapshot: BrainSnapshot, brain_id: str)
```

Benefits:

- Swap backends without changing application code
- Production runs on SurrealDB (one multi-model instance: document + graph + vector);
  SQLite and InMemory are test fixtures only, not deployment targets
- `SharedStorage`/`HybridStorage` give a remote-server and offline-first path for
  multi-device sync, independent of which local backend is active

## Configuration

### Brain Configuration

```python
@dataclass
class BrainConfig:
    decay_rate: float = 0.1
    reinforcement_delta: float = 0.05
    activation_threshold: float = 0.2
    max_spread_hops: int = 4
    max_context_tokens: int = 1500
    default_synapse_weight: float = 0.5
    # ...abbreviated — 50+ additional tuning fields for Hebbian learning, lateral
    # inhibition, novelty boost, co-activation, embeddings, and reranking now live
    # here too. See src/surreal_memory/core/brain.py for the full dataclass.
```

### CLI Configuration

Stored in `~/.surrealmemory/config.toml` (the pre-rename `~/.surreal-memory/` path is
recognized only as a legacy migration source):

```toml
version = "1.0"
current_brain = "default"
storage_backend = "surrealdb"   # "surrealdb" (production) or "sqlite" (test fixture)

[brain]
decay_rate = 0.1
max_context_tokens = 1500

[embedding]
enabled = true
provider = "gemini"

[auto]
min_confidence = 0.7
capture_decisions = true
capture_errors = true
```

SurrealDB connection settings are **not** stored in `config.toml` — they come from
environment variables (`SURREALDB_URL`, `SURREALDB_NS`, `SURREALDB_DB`, `SURREALDB_USER`,
`SURREALDB_PASS`), read by `storage/surrealdb/connection.py::SurrealSettings.from_env()`.
This lets the same brain config be shared across machines while each machine points at its
own SurrealDB instance (or Docker Compose service).

## File Structure

```
~/.surrealmemory/
├── config.toml           # User configuration (storage_backend, brain, embedding, auto...)
├── brains/
│   ├── default.db        # SQLite brain database (test-fixture backend only)
│   ├── work.db
│   └── personal.db
└── cache/
    └── ...
```

This local layout applies only when `storage_backend = "sqlite"` (or InMemory, which
persists nothing). When `storage_backend = "surrealdb"` (the production default), brain
data lives inside the SurrealDB instance itself — e.g. the `surrealdb_data` Docker volume
used by `docker-compose.surrealdb.yml` — not under `~/.surrealmemory/brains/`.

## Module Organization

```
src/surreal_memory/
├── __init__.py                # Public API exports
├── py.typed                   # PEP 561 marker
├── core/
│   ├── brain.py                # Brain, BrainConfig
│   ├── brain_mode.py            # BrainMode, SharedConfig, HybridConfig
│   ├── neuron.py                # Neuron, NeuronType, NeuronState
│   ├── synapse.py               # Synapse, SynapseType, Direction
│   ├── fiber.py                 # Fiber (with pathway, conductivity)
│   ├── memory_types.py          # TypedMemory, MemoryType, Priority
│   ├── alert.py                 # Alert, AlertType, AlertStatus
│   ├── project.py               # Project
│   ├── source.py                # Source (source-aware memory registry)
│   ├── review_schedule.py       # ReviewSchedule (Leitner-box spaced repetition)
│   ├── eternal_context.py       # Eternal context snapshotting
│   ├── action_event.py          # ActionEvent (habit-learning action log)
│   └── trigger_engine.py        # Rule-based trigger evaluation
├── engine/                     # ~90 modules — see "Engine Layer" above for groupings
│   ├── encoder.py                # MemoryEncoder
│   ├── retrieval.py              # ReflexPipeline
│   ├── retrieval_types.py        # DepthLevel, Subgraph, RetrievalResult
│   ├── retrieval_context.py      # reconstitute_answer, format_context
│   ├── activation.py             # SpreadingActivation (classic)
│   ├── reflex_activation.py      # ReflexActivation, CoActivation
│   ├── reranker.py                # Optional cross-encoder reranking
│   ├── lifecycle.py              # DecayManager, ReinforcementManager
│   ├── db_trainer.py / db_introspector.py / db_knowledge.py   # DB-to-Brain training
│   ├── doc_trainer.py / doc_chunker.py                        # Doc-to-Brain training
│   ├── context_optimizer.py      # Smart context scoring + dedup + budgeting
│   ├── embedding/                # Gemini / OpenAI / OpenRouter / local providers
│   └── ...                       # consolidation, dream, cognitive reasoning, sync/merge,
│                                  # tiering, diagnostics — see engine/ for the full list
├── extraction/
│   ├── parser.py               # QueryParser, Stimulus
│   ├── router.py               # QueryRouter
│   ├── entities.py             # EntityExtractor
│   ├── keywords.py             # extract_keywords, STOP_WORDS
│   ├── temporal.py             # TemporalExtractor
│   ├── structure_detector.py   # Table/CSV/JSON structure detection
│   ├── relations.py            # Relation extraction
│   ├── sentiment.py            # Sentiment/emotional valence signals
│   ├── codebase.py             # Codebase indexing extraction
│   └── llm_provider.py         # Optional LLM-assisted extraction
├── storage/
│   ├── base.py                     # NeuralStorage ABC (see "Storage Interface" above)
│   ├── factory.py                  # create_storage() (BrainMode-based) + HybridStorage
│   │
│   ├── surrealdb/                   # PRODUCTION backend
│   │   ├── store.py                  # SurrealDBStorage — composes the mixins below
│   │   ├── schema.py                 # DEFINE TABLE/FIELD/INDEX DDL, ensure_schema()
│   │   ├── migrations.py             # Schema migrations (e.g. synapse → RELATE edges)
│   │   ├── connection.py             # SurrealSettings (env-driven connection config)
│   │   ├── typed_memory.py           # TypedMemory CRUD mixin
│   │   ├── projects.py               # Project CRUD mixin
│   │   ├── sources.py                # Source registry mixin
│   │   ├── alerts.py                 # Proactive alerts mixin
│   │   ├── cognitive.py              # Cognitive state / predictions / knowledge gaps
│   │   ├── review_schedules.py       # Leitner-box review schedule mixin
│   │   ├── maturation.py             # Memory maturation stage mixin
│   │   ├── versions.py               # Brain version snapshot mixin
│   │   ├── keyword_entity.py         # Keyword DF + lazy entity-ref promotion mixin
│   │   ├── compression.py            # Compression backup / neuron snapshot mixin
│   │   ├── activity.py               # Change log, device registry, Merkle hash mixin
│   │   ├── depth_priors.py           # Bayesian depth prior mixin
│   │   └── tool_events.py            # Tool-call event logging mixin
│   │
│   ├── sqlite_store.py              # SQLiteStorage (core) — TEST FIXTURE
│   ├── sqlite_schema.py             # Schema DDL and migrations (schema v38)
│   ├── sqlite_row_mappers.py        # Row-to-object converters
│   ├── sqlite_neurons.py / sqlite_synapses.py / sqlite_fibers.py   # Core CRUD mixins
│   ├── sqlite_typed.py / sqlite_projects.py / sqlite_alerts.py     # Feature CRUD mixins
│   ├── sqlite_brain_ops.py / sqlite_versioning.py / sqlite_reviews.py
│   ├── sqlite_maturation.py / sqlite_cognitive.py / sqlite_compression.py
│   ├── sqlite_depth_priors.py / sqlite_coactivation.py / sqlite_drift.py
│   ├── sqlite_change_log.py / sqlite_sync_state.py / sqlite_devices.py / sqlite_merkle.py
│   ├── sqlite_action_log.py / sqlite_calibration.py / sqlite_entity_refs.py
│   ├── sqlite_sources.py / sqlite_sessions.py / sqlite_training_files.py
│   ├── sqlite_tool_events.py
│   ├── read_pool.py                 # Read-only connection pool for parallel WAL reads
│   ├── neuron_cache.py               # TTL cache for repeated exact-match neuron lookups
│   │
│   ├── memory_store.py              # InMemoryStorage (core, NetworkX-based) — TEST FIXTURE
│   ├── memory_brain_ops.py          # InMemory brain operations mixin
│   ├── memory_collections.py        # InMemory neuron/synapse/fiber ops mixin
│   ├── memory_reviews.py            # InMemory review schedule mixin
│   │
│   ├── shared_store.py              # SharedStorage HTTP client (core)
│   ├── shared_store_mappers.py      # dict_to_* converters
│   └── shared_store_collections.py  # Fiber/brain HTTP mixin + SharedStorageError
├── server/
│   ├── app.py                  # FastAPI application
│   ├── dependencies.py         # Shared FastAPI dependencies (storage, config)
│   ├── models.py               # Pydantic models
│   ├── routes/                 # API route handlers (brain, memory, sync, hub, oauth...)
│   └── static/                 # Built dashboard assets served at /ui
├── mcp/                        # ~45 modules — server + one handler file per tool group
│   ├── server.py / __main__.py / http_transport.py   # Server entrypoint + transport
│   ├── tool_schemas.py / tool_handlers.py            # MCP tool registration
│   ├── remember_handler.py / recall_handler.py / stats_handler.py   # Core 3-tool handlers
│   ├── auto_capture.py                               # Pattern-based memory detection
│   ├── db_train_handler.py / train_handler.py        # Training/ingestion tools
│   ├── alert_handler.py / maintenance_handler.py      # Proactive alerts + maintenance
│   ├── cognitive_handler.py / drift_handler.py        # Cognitive reasoning tools
│   ├── sync_handler.py / mem0_sync_handler.py         # Multi-device + import sync
│   └── prompt.py                                      # System prompts
├── cli/
│   ├── main.py                # Entry point, app registration
│   ├── doctor.py               # `smem doctor` — setup diagnostics (incl. SurrealDB probe)
│   └── commands/               # One file per command group (storage.py manages SurrealDB)
├── plugins/                   # CommunityPlugin — cone queries, smart merge, compression
├── safety/                    # Encryption, sensitive-content detection, input firewall
├── sync/                      # Multi-device sync engine, Merkle delta, device registry
├── integration/ + integrations/  # ChromaDB/Mem0/Cognee/Graphiti/LlamaIndex import adapters,
│                                 # Telegram, OpenClaw, nanobot integrations
├── hooks/                     # Claude Code hooks (session_start, pre_compact, stop, ...)
├── surface/                   # Knowledge Surface (.nm) generation/parsing
├── skills/                    # Bundled Claude skills (memory-audit, memory-evolution, ...)
└── utils/                     # Shared utilities (SimHash, timeutils, tag_normalizer, ...)

dashboard/                    # React web dashboard source (built into server/static/)
├── src/
│   ├── api/                    # REST client
│   ├── components/ features/   # Pages: overview, health, graph, timeline, storage, ...
│   ├── stores/                 # State management
│   └── i18n/

vscode-extension/
├── src/
│   ├── extension.ts           # Entry point
│   ├── commands/               # Command handlers
│   ├── editors/                # CodeLens, decorations
│   ├── server/                 # HTTP client, WebSocket, lifecycle
│   ├── utils/
│   └── views/
│       ├── MemoryTreeProvider.ts  # Activity-bar memory tree
│       ├── StatusBarManager.ts
│       └── graph/
│           ├── GraphPanel.ts      # Panel controller
│           └── graphTemplate.ts   # Cytoscape.js HTML template
└── test/                      # Unit, integration, perf tests
```
