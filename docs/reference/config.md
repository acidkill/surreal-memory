# Configuration Reference

Every setting in `~/.surrealmemory/config.toml`, generated from the dataclasses
in `unified_config.py`. Unknown keys are ignored on load, so a file written by
an older version keeps working.

Run `smem init` to write a file with the current defaults.

## Top level

Keys that sit outside any `[section]`.

| Setting | Type | Default | Description |
|---|---|---|---|
| `data_dir` | `Path` | `~/.surrealmemory` |  |
| `current_brain` | `str` | `default` |  |
| `device_id` | `str` | `""` |  |
| `storage_backend` | `str` | `surrealdb` |  |
| `json_output` | `bool` | `false` |  |
| `default_depth` | `int \| None` | `None` |  |
| `default_max_tokens` | `int` | `500` |  |
| `version` | `str` | `1.0` |  |

## `[brain]`

Settings for brain behavior.

| Setting | Type | Default | Description |
|---|---|---|---|
| `decay_rate` | `float` | `0.1` |  |
| `reinforcement_delta` | `float` | `0.05` |  |
| `reinforcement_neuron_limit` | `int` | `15` |  |
| `activation_threshold` | `float` | `0.2` |  |
| `max_spread_hops` | `int` | `4` |  |
| `max_context_tokens` | `int` | `1500` |  |
| `freshness_weight` | `float` | `0.0` |  |
| `extras` | `dict[str, Any]` | `{}` |  |

## `[embedding]`

Settings for embedding-based semantic recall.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `provider` | `str` | `sentence_transformer` |  |
| `model` | `str` | `all-MiniLM-L6-v2` |  |
| `similarity_threshold` | `float` | `0.7` |  |
| `dimension` | `int` | `0` |  |
| `endpoint` | `str` | `""` |  |

## `[auto]`

Auto-capture configuration for MCP server.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `capture_decisions` | `bool` | `true` |  |
| `capture_errors` | `bool` | `true` |  |
| `capture_todos` | `bool` | `true` |  |
| `capture_facts` | `bool` | `true` |  |
| `capture_insights` | `bool` | `true` |  |
| `capture_preferences` | `bool` | `true` |  |
| `min_confidence` | `float` | `0.7` |  |

## `[eternal]`

Eternal context auto-save configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `notifications` | `bool` | `true` |  |
| `auto_save_interval` | `int` | `15` |  |
| `context_warning_threshold` | `float` | `0.8` |  |
| `max_context_tokens` | `int` | `128000` |  |

## `[maintenance]`

Proactive brain maintenance configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `check_interval` | `int` | `25` |  |
| `fiber_warn_threshold` | `int` | `500` |  |
| `neuron_warn_threshold` | `int` | `2000` |  |
| `synapse_warn_threshold` | `int` | `5000` |  |
| `orphan_ratio_threshold` | `float` | `0.25` |  |
| `expired_memory_warn_threshold` | `int` | `10` |  |
| `stale_fiber_ratio_threshold` | `float` | `0.3` |  |
| `stale_fiber_days` | `int` | `90` |  |
| `consolidation_ratio_threshold` | `float` | `0.1` |  |
| `auto_consolidate` | `bool` | `true` |  |
| `auto_consolidate_strategies` | `tuple[str, ...]` | `['prune', 'merge', 'mature', 'infer']` |  |
| `consolidate_cooldown_minutes` | `int` | `30` |  |
| `dream_cooldown_hours` | `int` | `24` |  |
| `expiry_cleanup_enabled` | `bool` | `true` |  |
| `expiry_cleanup_interval_hours` | `int` | `12` |  |
| `expiry_cleanup_max_per_run` | `int` | `100` |  |
| `scheduled_consolidation_enabled` | `bool` | `true` |  |
| `scheduled_consolidation_interval_hours` | `int` | `24` |  |
| `scheduled_consolidation_strategies` | `tuple[str, ...]` | `['prune', 'merge', 'enrich']` |  |
| `version_check_enabled` | `bool` | `true` |  |
| `version_check_interval_hours` | `int` | `24` |  |
| `decay_enabled` | `bool` | `true` |  |
| `decay_interval_hours` | `int` | `12` |  |
| `reindex_enabled` | `bool` | `false` |  |
| `reindex_paths` | `tuple[str, ...]` | `[]` |  |
| `reindex_interval_hours` | `int` | `168` | weekly |
| `reindex_extensions` | `tuple[str, ...]` | `['.md', '.txt', '.py', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.rst', '.html', '.css']` |  |
| `notifications_enabled` | `bool` | `false` |  |
| `notifications_webhook_url` | `str` | `""` |  |
| `notifications_health_threshold` | `str` | `D` | alert at D or F |
| `notifications_daily_summary` | `bool` | `false` |  |
| `notifications_zero_activity_alert` | `bool` | `true` |  |

## `[safety]`

Safety and auto-redaction configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `auto_redact_min_severity` | `int` | `3` | Auto-redact severity 3+ by default |

## `[encryption]`

Encryption configuration for sensitive memory content.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `auto_encrypt_sensitive` | `bool` | `true` |  |
| `keys_dir` | `str` | `""` | empty = use {data_dir}/keys/ |

## `[write_gate]`

Write-gate configuration for memory quality enforcement.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | opt-in, backward compat (True == enforce) |
| `mode` | `str` | `off` | off | shadow | enforce. Overrides `enabled` when not "off". |
| `auto_capture_mode` | `str` | `""` | "" (inherit) | off | shadow | enforce |
| `min_length` | `int` | `30` | reject content shorter than this |
| `min_quality_score` | `int` | `3` | reject score below this (0-10 scale) |
| `auto_capture_min_score` | `int` | `5` | stricter threshold for passive captures |
| `max_content_length` | `int` | `2000` | reject wall-of-text above this |
| `reject_generic_filler` | `bool` | `true` | reject "done", "ok", "completed" etc. |

## `[dedup]`

LLM-powered deduplication settings.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `simhash_threshold` | `int` | `7` | tighter: ~89% similarity (was 10 / ~85%) |
| `embedding_threshold` | `float` | `0.85` |  |
| `embedding_ambiguous_low` | `float` | `0.75` |  |
| `llm_enabled` | `bool` | `false` |  |
| `llm_provider` | `str` | `none` |  |
| `llm_model` | `str` | `""` |  |
| `llm_max_pairs_per_encode` | `int` | `3` |  |
| `merge_strategy` | `str` | `keep_newer` |  |
| `max_candidates` | `int` | `30` | wider search (was 10) |
| `consolidation_max_anchors` | `int` | `2000` |  |

## `[tool_tier]`

MCP tool tier configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `tier` | `str` | `full` |  |

## `[mem0_sync]`

Auto-sync configuration for Mem0 integration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `self_hosted` | `bool` | `false` |  |
| `user_id` | `str` | `""` |  |
| `agent_id` | `str` | `""` |  |
| `cooldown_minutes` | `int` | `60` |  |
| `sync_on_startup` | `bool` | `true` |  |
| `limit` | `int \| None` | `None` |  |

## `[sync]`

Multi-device sync configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `hub_url` | `str` | `""` |  |
| `api_key` | `str` | `""` |  |
| `auto_sync` | `bool` | `false` |  |
| `sync_interval_seconds` | `int` | `300` |  |
| `conflict_strategy` | `str` | `prefer_recent` |  |

## `[tool_memory]`

Tool memory auto-capture configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `blacklist` | `tuple[str, ...]` | `[]` | Tool name prefixes to skip |
| `cooccurrence_window_s` | `int` | `60` | Seconds for USED_WITH detection |
| `min_frequency` | `int` | `3` | Min calls before creating a tool neuron |
| `max_buffer_lines` | `int` | `10000` | Truncate JSONL buffer beyond this |
| `process_batch_size` | `int` | `200` | Max events per processing cycle |

## `[telegram]`

Telegram backup integration configuration.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `chat_ids` | `tuple[str, ...]` | `[]` |  |
| `max_file_size_mb` | `int` | `50` |  |
| `backup_on_consolidation` | `bool` | `false` |  |

## `[license]`

License tier information — set via smem_sync_config(action='activate').

| Setting | Type | Default | Description |
|---|---|---|---|
| `tier` | `str` | `free` | "free" | "pro" | "team" |
| `activated_at` | `str` | `""` |  |
| `expires_at` | `str` | `""` |  |

## `[reranker]`

Settings for optional cross-encoder reranking after spreading activation.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `model_name` | `str` | `BAAI/bge-reranker-v2-m3` |  |
| `blend_weight` | `float` | `0.7` | Reranker weight (SA gets 1 - this) |
| `min_score` | `float` | `0.15` |  |
| `max_candidates` | `int` | `30` | Safety cap on overfetch |
| `endpoint` | `str` | `""` |  |

## `[tiers]`

Auto-tier promotion/demotion configuration (Pro feature).

| Setting | Type | Default | Description |
|---|---|---|---|
| `auto_enabled` | `bool` | `false` | Pro only — free users keep manual tiers |
| `promote_threshold` | `int` | `5` | access_frequency >= N → WARM→HOT |
| `demote_inactive_days` | `int` | `30` | no access in N days → HOT→WARM |
| `cold_archive_days` | `int` | `90` | no access in N days → WARM→COLD |
| `max_hot_memories` | `int` | `100` | cap HOT tier size |

## `[watcher]`

Settings for file watcher auto-ingestion.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `paths` | `tuple[str, ...]` | `[]` |  |
| `extensions` | `tuple[str, ...]` | `['.md', '.txt', '.pdf', '.docx', '.pptx', '.html', '.json', '.csv', '.xlsx', '.py', '.ts', '.js']` |  |
| `ignore_patterns` | `tuple[str, ...]` | `['__pycache__', '.git', 'node_modules', '.venv', '.env']` |  |
| `debounce_seconds` | `float` | `2.0` |  |
| `max_file_size_mb` | `int` | `10` |  |
| `max_watched_dirs` | `int` | `10` |  |
| `memory_type` | `str` | `fact` |  |
| `domain_tag` | `str` | `""` |  |

## `[response]`

MCP response compaction settings.

| Setting | Type | Default | Description |
|---|---|---|---|
| `compact_mode` | `bool` | `false` |  |
| `max_list_items` | `int` | `10` |  |
| `strip_hints` | `bool` | `true` |  |
| `content_preview_length` | `int` | `120` |  |
| `auto_compact_threshold` | `int` | `20` |  |

## `[budget]`

Token budget configuration for retrieval context allocation.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` |  |
| `default_tokens` | `int` | `4000` |  |
| `system_overhead` | `int` | `50` |  |
| `per_fiber_overhead` | `int` | `15` |  |

## `[trace]`

Retrieval-trace telemetry configuration (schema v9, opt-in).

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | opt-in, no traces persisted by default |
| `sample_rate` | `float` | `1.0` | fraction of recalls to trace when enabled |
| `retention_days` | `int` | `30` | prune traces older than this |
| `max_traces` | `int` | `5000` | cap total stored traces (delete-oldest) |

## `[decay_telemetry]`

Per-pass decay telemetry.

| Setting | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` |  |
| `retention_days` | `int` | `90` |  |
| `max_records` | `int` | `2000` |  |

## `[reasoning_training]`

Reasoning-training configuration (mining reasoning traces + injection).

| Setting | Type | Default | Description |
|---|---|---|---|
| `mining_enabled` | `bool` | `false` | opt-in (privacy): reads no transcripts until True |
| `injection_enabled` | `bool` | `false` | opt-in: inject learned strategies into sessions |
| `mining_models` | `tuple[str, ...]` | `[]` |  |
| `extra_transcript_dirs` | `tuple[str, ...]` | `[]` |  |
| `injection_map` | `tuple[tuple[str, str], ...]` | `[]` |  |
| `categories` | `tuple[str, ...]` | `['debugging', 'planning', 'implementation', 'refactoring', 'research', 'verification', 'architecture', 'data-analysis']` |  |
| `min_trace_chars` | `int` | `200` |  |
| `max_trace_chars` | `int` | `100000` |  |
| `scan_lookback_days` | `int` | `30` | 0 = full backfill |
| `retention_days` | `int` | `90` |  |
| `max_traces_total` | `int` | `20000` |  |
| `min_cluster_support` | `int` | `3` |  |
| `cluster_cosine` | `float` | `0.75` |  |
| `min_confidence` | `float` | `0.2` |  |
| `min_patterns_per_category` | `int` | `3` |  |
| `injection_max_patterns` | `int` | `5` |  |
| `injection_max_chars` | `int` | `4000` |  |
| `distill_use_llm` | `bool` | `false` |  |
| `distill_llm_model` | `str` | `""` |  |
| `distill_llm_endpoint` | `str` | `""` |  |
| `allow_remote_endpoints` | `bool` | `false` |  |
| `distill_llm_unload_cmd` | `tuple[str, ...]` | `[]` |  |
| `distill_llm_load_cmd` | `tuple[str, ...]` | `[]` |  |
| `redact_secrets` | `bool` | `true` |  |
| `pattern_targets` | `dict[str, int]` | `{}` |  |

## Environment variables

Read straight from the environment. Where the same setting exists in both
places, the environment wins.

| Variable | Read in |
|---|---|
| `BGE_M3_API_KEY` | `src/surreal_memory/engine/reranker.py` |
| `CLAUDE_SESSION_ID` | `src/surreal_memory/hooks/capture_state.py`, `src/surreal_memory/hooks/post_tool_use.py` |
| `CLIPROXY_URL` | `src/surreal_memory/server/routes/oauth.py` |
| `CODEX_SESSION_ID` | `src/surreal_memory/hooks/post_tool_use.py` |
| `COGNEE_API_KEY` | `src/surreal_memory/integration/adapters/cognee_adapter.py`, `src/surreal_memory/mcp/index_handler.py` |
| `GEMINI_API_KEY` | `src/surreal_memory/engine/embedding/gemini_embedding.py`, `src/surreal_memory/engine/semantic_discovery.py` |
| `GOOGLE_API_KEY` | `src/surreal_memory/engine/embedding/gemini_embedding.py`, `src/surreal_memory/engine/semantic_discovery.py` |
| `GOOGLE_GEMINI_API_VERSION` | `src/surreal_memory/engine/embedding/gemini_embedding.py` |
| `GOOGLE_GEMINI_BASE_URL` | `src/surreal_memory/engine/embedding/gemini_embedding.py` |
| `GRAPHITI_GROUP_ID` | `src/surreal_memory/integration/adapters/graphiti_adapter.py` |
| `GRAPHITI_URI` | `src/surreal_memory/integration/adapters/graphiti_adapter.py` |
| `MEM0_API_KEY` | `src/surreal_memory/integration/adapters/mem0_adapter.py`, `src/surreal_memory/mcp/index_handler.py`, `src/surreal_memory/mcp/mem0_sync_handler.py` |
| `OLLAMA_BASE_URL` | `src/surreal_memory/engine/embedding/ollama_embedding.py`, `src/surreal_memory/engine/reasoning_distiller.py` |
| `OPENAI_API_KEY` | `src/surreal_memory/engine/semantic_discovery.py` |
| `OPENAI_BASE_URL` | `src/surreal_memory/engine/embedding/openai_embedding.py` |
| `OPENROUTER_API_KEY` | `src/surreal_memory/engine/semantic_discovery.py` |
| `SMEM_PROJECT` | `src/surreal_memory/hooks/project_context.py` |
| `SURREALDB_AUTH_LEVEL` | `src/surreal_memory/storage/surrealdb/connection.py` |
| `SURREALDB_DB` | `src/surreal_memory/cli/commands/storage.py`, `src/surreal_memory/storage/surrealdb/connection.py` |
| `SURREALDB_NS` | `src/surreal_memory/cli/commands/storage.py`, `src/surreal_memory/storage/surrealdb/connection.py` |
| `SURREALDB_PASS` | `src/surreal_memory/storage/surrealdb/connection.py`, `src/surreal_memory/unified_config.py` |
| `SURREALDB_URL` | `src/surreal_memory/cli/commands/storage.py`, `src/surreal_memory/storage/surrealdb/connection.py` |
| `SURREALDB_USER` | `src/surreal_memory/storage/surrealdb/connection.py` |
| `SURREAL_MEMORY_API_KEY` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_BRAIN` | `src/surreal_memory/cli/_helpers.py`, `src/surreal_memory/cli/commands/brain.py`, `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_DASHBOARD_CACHE_TTL` | `src/surreal_memory/server/dashboard_cache.py` |
| `SURREAL_MEMORY_DIR` | `src/surreal_memory/cli/config.py`, `src/surreal_memory/cli/update_check.py`, `src/surreal_memory/engine/reasoning_injection.py`, +3 more |
| `SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER` | `src/surreal_memory/mcp/recall_handler.py` |
| `SURREAL_MEMORY_EMBEDDING_API_KEY` | `src/surreal_memory/engine/embedding/bge_m3_embedding.py` |
| `SURREAL_MEMORY_EMBEDDING_DIMENSION` | `src/surreal_memory/engine/embedding/bge_m3_embedding.py`, `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_EMBEDDING_ENABLED` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_EMBEDDING_ENDPOINT` | `src/surreal_memory/engine/embedding/bge_m3_embedding.py`, `src/surreal_memory/engine/embedding/openai_embedding.py`, `src/surreal_memory/engine/reasoning_distiller.py`, +2 more |
| `SURREAL_MEMORY_EMBEDDING_MODEL` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_EMBEDDING_PROVIDER` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_HOST` | `src/surreal_memory/utils/config.py` |
| `SURREAL_MEMORY_HUB_URL` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_INLINE_EMBED_TIMEOUT` | `src/surreal_memory/engine/encoder.py` |
| `SURREAL_MEMORY_REASONING_ALLOW_REMOTE` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_REASONING_EXTRA_DIRS` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_REASONING_INJECTION` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_REASONING_INJECTION_MAP` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_REASONING_MINING` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_REASONING_MODELS` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_RERANKER_API_KEY` | `src/surreal_memory/engine/reranker.py` |
| `SURREAL_MEMORY_RERANKER_ENDPOINT` | `src/surreal_memory/engine/reranker.py` |
| `SURREAL_MEMORY_SOURCE` | `src/surreal_memory/mcp/evolution_handler.py`, `src/surreal_memory/mcp/remember_handler.py` |
| `SURREAL_MEMORY_SQLITE_PATH` | `src/surreal_memory/utils/config.py` |
| `SURREAL_MEMORY_STORAGE` | `src/surreal_memory/cli/doctor.py`, `src/surreal_memory/unified_config.py`, `src/surreal_memory/utils/config.py` |
| `SURREAL_MEMORY_SYNC_AUTO` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_SYNC_ENABLED` | `src/surreal_memory/unified_config.py` |
| `SURREAL_MEMORY_TELEGRAM_BOT_TOKEN` | `src/surreal_memory/integration/telegram.py` |

---

*Auto-generated by `scripts/gen_config_docs.py` from `unified_config.py` — 23 sections, 52 environment variables.*
