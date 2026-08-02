# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.2] — 2026-08-02 — `smem brain` and the dashboard see brains that live only in SurrealDB

### Fixed

Two separate, same-named `list_brains()` methods existed with no relation to
each other: `UnifiedConfig.list_brains()` and `CLIConfig.list_brains()` both
only glob local `*.db`/`*.json` fixture files — they never query SurrealDB,
the only production backend since v2.0.0. Five `smem brain` subcommands
(`list`, `use`, `create`, `import`, `delete`) and two dashboard endpoints
(`POST /brains/switch`, `GET /brain-files`) called one of these instead of
the correct `unified_config.list_available_brains()`, which queries the
active backend directly. Net effect on a SurrealDB-backed install: `smem
brain list` and the dashboard's brain-files panel always reported zero
brains, and switching to a brain via the dashboard 404'd even though the
same dashboard's `/brains` endpoint listed it correctly.

All five CLI call sites and both dashboard endpoints now route through
`list_available_brains()`. `smem brain delete` also gained a guard: a brain
that exists only in SurrealDB has no local file to delete, so attempting to
delete one now reports a clear "not supported yet" message instead of
crashing with an unhandled `FileNotFoundError` — deleting SurrealDB-backed
brains programmatically is not yet implemented.

## [3.0.1] — 2026-08-02 — `smem doctor` no longer carries dead SQLite-era checks

### Fixed

Two `smem doctor` checks became permanently inert when the SQLite backend was
removed in 3.0.0: `Schema version` could only ever return `SKIP` for either
remaining storage backend (SurrealDB applies its schema idempotently with no
stored version marker to compare against), and `Brain database` fell through
to a dead code path that checked for a local `.db` file that no longer exists
for any valid backend. Both permanently capped the doctor summary below 100%
even on a fully healthy install.

`Schema version` is removed. `Brain database` now does a real check for the
`surrealdb` backend: it looks up the configured brain in SurrealDB and reports
whether it exists, instead of unconditionally skipping.

## [3.0.0] — 2026-08-02 — The SQLite storage backend is removed

### Breaking: `storage_backend = "sqlite"` is now a hard error

SurrealDB has been the production backend since 2.0.0; SQLite was kept only as
an internal test fixture, deprecated since 2.21.0. All 31 `storage/sqlite_*.py`
modules, `storage/factory.py` (`HybridStorage`, `create_storage`),
`storage/read_pool.py` and `storage/neuron_cache.py` — roughly 10,200 lines —
are deleted. `InMemoryStorage` (opt in with `SURREAL_MEMORY_STORAGE=memory`)
is now the only non-SurrealDB backend, and now implements the full
`NeuralStorage` interface (previously 64 of 172 methods were inherited
`NotImplementedError` stubs, tolerable only because SQLite covered the gap in
tests).

Setting `storage_backend = "sqlite"` (via `config.toml` or
`SURREAL_MEMORY_STORAGE`) now raises immediately with the two supported
alternatives and a link to the migration guide, instead of silently falling
back to something else — a silent fallback here would look exactly like data
loss. **Existing SQLite brains at `~/.surrealmemory/brains/*.db` are never
read, written or deleted by 3.0.0** — installing a 2.x release restores full
access to them at any time. See `docs/guides/migrating-to-3.0.md` to move a
brain to SurrealDB.

### Pinning, training dedup and graph density only ever worked on SQLite

Six storage methods lived on `SQLiteStorage` and were never declared on the
`NeuralStorage` interface. Every caller reached for them through `hasattr` or a
swallowed exception, so on SurrealDB — the production backend since 2.0.0 — they
did nothing at all, and nothing reported it.

The consequences were not cosmetic:

- **Pinned memories were decayed and pruned.** `Fiber.pinned` round-tripped
  fine, and compression, the tier engine and the typed-memory TTL sweep all
  honoured it, but decay and prune resolve pinned neurons through
  `get_pinned_neuron_ids()`. Without it they treated pinned fibers like any
  other, deleting exactly the content `smem_train` documents as a permanent
  knowledge base.
- **`smem_pin` refused every action** — pin, unpin *and* list — answering
  "Storage does not support pinning".
- **`smem train` was not idempotent.** With no training-file tracking, each run
  re-encoded the whole corpus and duplicated it.
- **`activation_strategy="auto"` never left classic BFS**, because selecting PPR
  or hybrid needs `get_graph_density()`.

All of these are now declared on `NeuralStorage` and implemented on SurrealDB
and the in-memory backend. Migration is automatic and additive: the `pinned`
field was already in the SurrealDB schema, and the new `training_files` table is
created by the idempotent `ensure_schema()` on next start, with no schema-version
bump.

### `smem_pin(action="list")` was broken on every backend

The SQLite query selected `type` and `priority` from the `fibers` table, which
has neither column, so the call raised `no such column: type` — on the one
backend that implemented it at all. Both fields live on `typed_memories` and are
now read from there. No test covered this path; the pin tests now run against
every backend, as do the document-training tests.

### Removed: retrieval calibration EMA

The per-gate accuracy EMA and per-brain RRF retriever weights
(`save_calibration_record`, `get_gate_ema_stats`, `get_retriever_weights` and
their companions) are gone, along with `GateCalibration` and the two SQLite
tables behind them. They were SQLite-only too, so no SurrealDB brain ever
recorded a sample or had a gate decision or fusion weight adjusted by one.
Sufficiency gates use their documented thresholds and RRF uses
`DEFAULT_RETRIEVER_WEIGHTS` — precisely what every SurrealDB brain was already
doing. `get_graph_density()` was kept and implemented, because it changes
behaviour rather than reporting on it.

## [2.20.1] — A resolved model alias no longer points at a prior generation

### The reasoning-injection alias table pointed at a superseded model

`resolve_active_model`'s last-resort fallback maps a short alias (from `~/.claude/settings.json`'s
`model` field) to a canonical model id when no other signal is available. `"opus"` and `"opusplan"`
resolved to a prior model generation's id, so on that fallback path, pattern injection looked up a
source model that no longer matches the current lineup. Both aliases now resolve to the current
generation.

### Release drafts now publish themselves

Preparing a draft GitHub Release does not push a real tag — the tag only becomes real once the
draft is published, and that is what actually triggers the publish pipeline (`on: push: tags: v*`).
Nothing published a release's draft automatically once its own PR merged, so a draft could sit
unpublished after merge until someone did it by hand.

A new workflow watches for a merged PR that itself bumped `pyproject.toml`'s version and publishes
the matching draft release, if one is still pending. It only acts on that specific signal, so an
unrelated PR merging while an earlier release's draft awaits review cannot trigger a publish it has
nothing to do with.

## [2.20.0] — The embedding path stops being outperformed by its own fallback

### Community — Spanish README (#127)

Thanks to [WebBrain](https://github.com/webbrain-one) for this release's first outside
contribution: a full Spanish translation, `README.es-ES.md`, alongside the original.

### Pattern clustering used a threshold that was never calibrated

Distillation groups traces into patterns by cosine similarity when an embedder is available, and
falls back to move-set Jaccard when one is not. The cosine threshold was a module constant — and
because `_get_embedder()` read the environment instead of the configured embedder, it could return
nothing on a correctly configured machine, so the embedding branch never ran there and the constant
was never exercised against a real embedding model.

Embedding models do not share a similarity scale. Measured against a local bge-m3 corpus, the
shipped constant sat **above the 99th percentile** of pairwise trace similarity in every category:
the largest categories produced no clusters at all, and the embedding path yielded roughly a fifth
of what the move-set fallback it is meant to supersede produced over the same traces. Mining
reported patterns learned while whole categories stayed empty, which reads as "not enough material"
rather than as a threshold defect.

The threshold is now `reasoning_training.cluster_cosine` (default `0.75`, clamped to `0.05`–`1.0`),
so it travels with the configured embedder instead of being frozen in the module. The floor exists
because single-linkage clustering collapses into a single component as the threshold approaches the
corpus baseline — a very low value yields fewer patterns, not more.

If a brain's category coverage looks stuck, re-run distillation after checking this value; a raised
`pattern_targets` cannot help when nothing clusters in the first place.

```toml
[reasoning_training]
cluster_cosine = 0.75
```

A non-finite value in that field is rejected rather than clamped: `NaN` propagates through the
bounds check into the FLOOR, which would merge every trace into a single cluster instead of falling
back to a usable threshold.

### The embedder ignored its own configuration

`_get_embedder()` probed the environment rather than reading `[embedding]`, so a configured local
provider was skipped whenever an unrelated cloud API key happened to be present, and classification
silently degraded to keyword matching. It now reads the configuration first and delegates to the
canonical provider factory, falling back to environment probing only when embedding is disabled.

`[embedding] endpoint` joins the reranker's equivalent knob, so a local OpenAI-compatible embedding
server can be pinned in `config.toml` instead of only through an environment variable.

### A validated endpoint was not the endpoint that got connected to

The loopback gate is meant to guarantee that reasoning traces — private user data — never leave the
machine. For two providers the value it checked had no causal connection to the URL the HTTP client
actually opened, so the gate passed while requests went to a remote host:

- **openrouter**: the endpoint was loopback-checked, but the provider was then built through the
  factory, which supplies no `base_url` and falls back to a hardcoded remote default. No attacker
  required — just the documented configuration surface.
- **openai**: `[embedding] endpoint` is read from `config.toml` first, but the client resolved its
  own base URL from the environment variable alone. A loopback endpoint set only in the config file
  — exactly what that field is for — passed the check and then fell back to the hosted default.
- **ollama**: short-circuit evaluation meant the check was never called for this provider at all,
  and its base URL comes from `OLLAMA_BASE_URL`, which nothing validated.

The endpoint that clears the gate is now passed explicitly as the client's `base_url`, in both the
configured and the auto-detect paths, and the ollama gate tests the URL that provider really opens.
The factory delegation that re-resolved the endpoint independently is gone.

### Loopback validation accepted hostnames that only looked local

The check guarding local-only endpoints tested for a `127.` prefix, which accepts
`127.0.0.1.attacker.example` — a remote host. It now parses the host and asks the address itself
whether it is a loopback address.

## [2.19.0] — Learned reasoning patterns can be rebuilt, and pattern reads follow the request's brain

### Losing pattern fibers was permanent

Distillation marks every trace it consumes as processed — including the ones it discards
(off-category, or in a cluster too small to support a pattern) — and nothing could ever clear
that flag. So a brain whose pattern fibers were lost had no path back. Re-running the miner
looked like it should fix it and could not: the transcript re-scan finds the same traces,
ingest deduplicates them on `trace_hash`, and distillation only ever reads *unprocessed*
rows. The run reported traces processed and zero patterns learned, and the dashboard's
category coverage stayed frozen at whatever the surviving patterns happened to add up to,
run after run.

This was reachable through no fault of the operator: patterns mined before 2.17.0 were
written under a brain scope no read path consults (fixed there), and once those rows were
removed as orphans, the traces that produced them were still marked processed.

`reprocess` re-opens that backlog. It is available on `POST /api/dashboard/reasoning/mine`,
as a checkbox beside *Backfill* in the dashboard's mining card, on `smem_reasoning`
(`action: "mine"`), and as `smem reasoning mine --reprocess`. Repeating it is safe: pattern
signatures make a second pass a no-op rather than a duplicate-maker. It respects
`mining_models` — which are globs, so they are resolved against the models actually present
before anything is re-opened, and a filter matching nothing re-opens nothing rather than
widening into a blanket reset. `dry_run` still wins: re-opening the backlog is a write.

One limit is inherent: a distill run prunes processed traces past `retention_days`, so
`reprocess` can only rebuild from what is still staged. Anything older needs `backfill` to
re-ingest it from the transcripts, and rotated transcripts are gone for good.

### Consolidation merged learned patterns away as fast as mining produced them

`_merge` builds one fiber from a group of overlapping ones, and the merged fiber
REPLACES its members' metadata wholesale before the members are deleted. Learned
reasoning patterns are the most exposed fibers in the graph to that: every
pattern in a category is attached to that category's concept neuron, and shared
neurons are exactly what the overlap check keys on. A mining run's whole output
therefore collapsed into a single metadata-less "Merged from N fibers" row, its
`_source_model` / `_reasoning_category` / `_reasoning_confidence` gone, and
category coverage fell straight back to zero — with the source traces already
marked processed, which before the reprocess path above meant it stayed there.

This is why coverage could look frozen while mining reported patterns learned
run after run: they were being produced and consumed in the same cycle.

The same defect had already been found and fixed for `_habit_pattern`, whose
guard carries a comment describing this exact failure. It simply never covered
`_reasoning_pattern`. Both markers now share one guard.

### A per-model pattern target above 100 was silently reduced

`pattern_targets` was clamped to 100 by the config loader and rejected above 100
by the config endpoint, with nothing announcing either: a higher target written
into `config.toml` came back as 100 on the next read, and distillation stopped
there while thousands of staged traces still had patterns to give. The ceiling
guards against a runaway configuration, and a model's own backlog bounds it long
before 100 does. Both sites now share `MAX_PATTERN_TARGET` (1000), so the
endpoint can no longer accept a value the loader would quietly reduce.

### Status and pattern endpoints could mix two brains in one response

Reasoning traces are read with an explicit `brain_id` taken from the request's brain, but the
fiber API has no such parameter — `find_fibers`, `get_fiber` and `delete_fiber` filter on
whatever brain the storage instance is bound to, and the server hands out the process-wide
instance bound at startup without rebinding it per request. A request carrying an
`X-Brain-ID` naming any other brain therefore reported one brain's traces beside another
brain's patterns, with coverage computed from the wrong half and no indication anything was
amiss. `DELETE /patterns?model=` was worse: it listed victims from one scope and issued the
deletes against another, reporting a count for rows it had not removed.

Pattern reads and deletes now resolve under the same scope the traces use. When the request's
brain already matches the bound one — the common case — nothing changes and no extra
connection is opened.

## [2.18.2] — Docs catch up with the maturation model, and the attribution rule stops contradicting itself

No behavior change. Documentation and contributor-process only.

### The README described consolidation as something a command does

`Memory consolidation — episodic memories mature into semantic knowledge` was the entire
description, which reads as though `smem consolidate` performs the maturation. It does not,
and 2.18.0 removed that claim from the code's own remedy text without updating the README
that still implied it. The entry now states the actual mechanism: maturation happens through
spaced recall (7 days dwell plus reinforcement across 3+ distinct days, or 15+ rehearsals
across 5+ windows), names the two `smem health` fields that show where each memory sits and
what it is waiting on, and points at `brain.reinforcement_neuron_limit` for widening how many
of a recall's memories get rehearsed.

`distill_llm_load_cmd` (2.18.0) was missing from the reasoning-training example entirely, so
the documented config could only unload a model it never controlled the loading of.

### AGENTS.md required the attribution that CLAUDE.md forbids

`AGENTS.md` Hard Rule #2 was "Tool Transparency": it instructed contributors to add a
`Built with: …` line and to *keep* `Co-Authored-By: Claude <noreply@anthropic.com>` trailers.
`CONTRIBUTING.md`'s quality gate listed the same trailer as a requirement. Both are now the
opposite rule, because the repo's release process forbids AI attribution and the two
documents were pulling contributors in opposite directions — with the versioned, visible one
winning by default for anyone who could not see the ignored one.

Because a documented rule had already failed to prevent this twice, it is now also enforced
mechanically: a `commit-msg` pre-commit hook rejects any commit message carrying an agent
attribution trailer.

## [2.18.1] — The maturation view actually reaches `smem_health`

2.18.0 added `stage_distribution` and `semantic_gate_blockers` to the health report and
computed both on every call. Neither reached anyone. The `smem_health` MCP handler and the
`smem health` CLI command each build an explicit response dict, and both simply omitted the
new keys — the fields existed on the report object, were populated correctly, and were
dropped at the serialization boundary. Verifying through `DiagnosticsEngine.analyze()`
passes on the object that boundary then discards, which is exactly why this survived the
original release.

Both surfaces now carry them, omitted rather than nulled when a backend cannot answer,
matching how `smem_evolution` already treats the same two fields.

`smem health` also never showed `top_penalties` at all, so 2.18.0's corrected remedy text —
the one that names spaced recall and says `smem consolidate` will not raise the ratio on its
own — was reachable only over MCP and the dashboard. A CLI user saw a low consolidation bar
and no explanation for it. The CLI now prints the biggest penalties with their remedy, plus
the stage distribution and the semantic-gate breakdown. The semantic-gate line says what its
numbers mean (`18 waiting on dwell time, 9 waiting on recall spacing, 3 ready`), and calls
out the `ready` bucket as the only one a `smem consolidate` run moves.

Both payloads are now produced by one serializer rather than two hand-maintained dicts.
That duplication is the actual defect here: two lists of keys, written out by hand, drift
apart the moment the report grows a field — which is how these two ended up missing from
both surfaces while `top_penalties` was missing from only one. A field added to the report
and to the serializer is now on both surfaces or on neither. `smem health --json`
consequently also gained `contradiction_count`, `conflict_rate`, and the `weight` /
`penalty_points` the MCP payload had always carried on each penalty.

`docs/guides/brain-health.md` documents the maturation fields, and its example
`top_penalties[].action` no longer quotes the pre-2.18.0 remedy that told the reader to run
`smem consolidate`.

## [2.18.0] — Maturation reachability

2.16.0 made the consolidation report honest about what it counts. It did not make
`consolidation_ratio` reachable. This release closes the gap between the metric and the
mechanism that is supposed to move it: five separate defects that each, independently,
kept fibers from ever reaching SEMANTIC or kept the report from saying why.

### The remedy stopped recommending the wrong command

`smem_health`'s low-`consolidation_ratio` penalty, `smem_stats`' low-consolidation hint, and
the `NO_CONSOLIDATION` warning all told the operator to run `smem consolidate` to raise the
ratio. Nothing about `consolidate` moves a fiber toward SEMANTIC — only spaced recall does
(reinforcement spread across 3+ distinct days, or 15+ rehearsals across 5+ time windows).
All three surfaces now name the actual mechanism and say explicitly that `consolidate` will
not move it on its own.

### Recall's rehearsal step was capped at a hardcoded 10, twice over

Every recall reinforces the fibers connected to its top-activated neurons — the mechanism
spaced recall depends on. Two independent, redundant caps limited this to the first 10
neurons regardless of how many a recall actually activated: one in the retrieval pipeline's
neuron ranking, a second, wholly redundant one inside the reinforcement step itself on the
fibers that made it through the first cap. A recall that activated 50 neurons still only
rehearsed fibers connected to 10 of them, silently discarding the rest of that recall's
contribution to the spacing requirement.

Both caps are fixed: the redundant fiber-side cap is removed, and the neuron-side cap is now
a configurable `brain.reinforcement_neuron_limit` (default 15, up from the old
unconfigurable 10) in `config.toml`. The default was set from a live measurement against the
production storage backend rather than assumed — raising this value is not free on every
backend, and an installation whose storage can absorb more latency can raise it.

### Merging fibers used to erase their progress toward SEMANTIC

Consolidation's `merge` strategy deletes the source fibers and writes one new fiber in their
place. It never carried the sources' maturation records forward, so a fiber one reinforcement
away from SEMANTIC that got merged came back at the beginning: no maturation row at all,
stage STM. The merged fiber now inherits the higher of the sources' stages, the union of
their reinforcement timestamps (not a plain concatenation — a fiber reinforced on the same
calendar day by both sources counts as one distinct day, not two), and the oldest of their
stage-entry timestamps, so a source that had already dwelt in a stage far longer than its
merge partner does not have that dwell time reset to "now."

### The consolidation report and health diagnostics explain themselves now

- The report's flat `Stages advanced: N` is now followed by a breakdown of exactly which
  hop advanced (`stm→working`, `working→episodic`, `episodic→semantic`); the three hop
  counts sum to the same total, so nothing about the existing field changes for anything
  already parsing it.
- The health report now includes a `stage_distribution` (how many fibers sit at each
  maturation stage) and a `semantic_gate_blockers` breakdown of every EPISODIC fiber by what
  is actually blocking it from SEMANTIC: waiting on dwell time, waiting on reinforcement
  spacing, or already eligible and waiting for the next consolidation pass. (Correction: this
  entry originally said "`smem_health` now includes". It did not — both fields were dropped
  at that tool's serialization boundary and reached a caller only through `smem_evolution`
  and the dashboard. Fixed in 2.18.1.) Found and fixed along the
  way: `smem evolution`'s own "closest to semantic" classifier duplicated this threshold
  logic locally and checked only the distinct-days path, so a fiber that qualified through
  the rehearsal-count-and-windows alternative was misreported as still blocked. Both
  diagnostics now share one classifier.
- A consolidation run's own backfill of maturation rows for fibers that predate the
  maturation subsystem is now a visible line in the report summary, not a number that only
  existed in a field nothing printed.

### Distillation can now explicitly load its model before the first request

`reasoning_training.distill_llm_unload_cmd` (2.17.0) explicitly releases a distillation run's
chat model when the run ends. Loading was still implicit — whatever happens on that
endpoint's first request — so a run had no way to control how the model got loaded: with
what context size, how many GPU layers, or with a projector it did not need. A new,
symmetric `distill_llm_load_cmd` runs once, before the first naming request, with the same
argv-only execution (no shell, `{model}` substitution, empty or failing command degrades
silently to the old implicit behavior). The unload command still fires even if a run that
explicitly loaded ends up naming nothing, closing the leak that would otherwise leave the
model resident.

### Fixed

- A configured `distill_llm_model` is sanitized against a looser character set than the
  `distill_llm_load_cmd`/`distill_llm_unload_cmd` argv template it gets substituted into; the
  substituted result is now re-validated against the stricter argv allowlist before
  execution, closing a defense-in-depth gap an independent security review found.
- A `distill_llm_load_cmd`/`distill_llm_unload_cmd` longer than the configured part limit is
  now voided outright, consistent with every other invalid-command case, rather than
  silently truncated to a shorter, different command.
- `SurrealDBStorage.clear()` never deleted maturation rows, leaving them orphaned after a
  brain's fibers and neurons were otherwise fully cleared.

## [2.17.0] — Patterns that read like advice

`reasoning_training.distill_use_llm` has existed as a configuration field since reasoning
training shipped. Nothing read it. Setting it changed the config file and nothing else —
searching for the name found the dataclass attribute, `to_dict`, `from_dict` and the TOML
writer, and no consumer anywhere. This release gives it an implementation.

### What the flag now does

A distilled pattern used to be named after its own mechanics. The title was the cluster's
three most frequent reasoning moves (`debugging: restate-goal, gather-evidence, verify`)
and the description was the first 200 characters of the medoid trace — a raw thinking
fragment, usually cut mid-sentence. Serviceable as an identifier, useless as an
explanation, and it is what `smem reasoning` displays and what injection ships to other
models.

With the flag on, a local model rewrites the prose:

```
title       Diagnose, Investigate, and Confirm Fix
description Debug by restating the objective, gathering diagnostic evidence, then
            confirming the resolution.
strategy    1. Restate the problem to be solved.
            2. Gather evidence — error messages, tracebacks, surrounding code.
            3. Verify the problem no longer occurs.
```

It rewrites **only** prose. `model`, `category`, `confidence`, `frequency` and `signature`
are untouched, and `signature` is derived from the cluster's trace hashes — so toggling
the flag cannot fork a known pattern into a duplicate. Existing patterns keep their names;
naming applies to patterns distilled from then on.

### Traces do not leave the machine

Trace content is raw model reasoning, so the endpoint is **loopback-only**. A remote
address yields no namer and therefore no request at all — an invariant, not a default, and
no setting relaxes it. Ingest-time redaction is upstream and can be switched off, so the
transport guarantee has to stand on its own.

Resolution order is `SURREAL_MEMORY_LLM_ENDPOINT`, then the new `distill_llm_endpoint`
config key, then `SURREAL_MEMORY_EMBEDDING_ENDPOINT` — one local OpenAI-compatible server
commonly serves both roles.

### The model is borrowed, not kept

Local model servers load a chat model on its first request and typically keep it resident
with no idle timeout, so naming a handful of patterns would leave several gigabytes parked
in VRAM indefinitely. Loading stays implicit — the first request pulls the model in — and
`distill_llm_unload_cmd` releases it when the run ends, including when the run fails.

The command is an **argv list executed without a shell**, so a model name can never become
shell syntax; `{model}` is substituted, a part that is not plain command syntax voids the
whole command, and it is read from the config file only, never from an API request. It
fires only if the run actually issued a request, so a model something else loaded is left
alone.

### Failure is invisible

Missing endpoint, missing `httpx`, refused connection, timeout, HTTP error, or a model that
answers in prose instead of JSON — every one falls back to the mechanical naming.
Distillation never fails because naming failed. Three consecutive failures trip a circuit
breaker, so a dead endpoint costs three attempts rather than one timeout per cluster.

Three failure modes that only a live model exposes are handled explicitly, because against
a stubbed transport all of them look like success:

- **A reasoning model spends its budget thinking first.** Too small a token allowance
  returns an empty `content` with `finish_reason: "length"` rather than a short answer. The
  budget now accommodates thinking, truncation is reported as its own diagnosis instead of
  a generic parse failure, and servers that support it are asked to skip the thinking phase
  (a server that rejects that request has it dropped after one attempt).
- **A model asked for "numbered steps" answers with a JSON array**, not a string. Refusing
  a good answer over its container type was wrong; a list of scalars is now joined into
  lines.
- **Markdown fences and surrounding chatter** are located rather than assumed away.

### Added

- `reasoning_training.distill_llm_model` — the chat model to name with.
- `reasoning_training.distill_llm_endpoint` — loopback OpenAI-compatible base URL, for
  deployments that cannot easily add environment variables.
- `reasoning_training.distill_llm_unload_cmd` — argv run once after distillation.

### An impossible embedding configuration is now reported

Every embedding provider guesses a dimension for a model it does not recognise — Gemini
assumes 3072, the OpenAI-compatible providers 1536, Ollama 1024. Each guess is reasonable
alone and dangerous in combination: aim a provider at another provider's model name and it
confidently produces vectors of the wrong width, which the HNSW index then rejects on every
write. `smem doctor` reported that configuration as **ok** as long as the provider's package
was importable, and `smem_health` reported it as available.

Two rules now run in `smem doctor`, in `smem_health` and at MCP startup:

- **A hosted provider's catalogue is closed.** A model outside it is not "unlisted", it is a
  request the API will refuse — reported together with the models that provider does serve.
- **A catalogued model's dimension must match `embedding.dimension`**, since that is what the
  vector index is built from.

Deliberately silent where it cannot know: a local OpenAI-compatible server serves whatever
files it was pointed at, so an unrecognised model name there is ordinary rather than
suspect. Only an exact catalogue hit counts as knowing a dimension — a provider's fallback
is a guess, and calling a configuration broken on the strength of a guess would flag every
local model name and train everyone to ignore the check.

### Fixed

- `probe_embedding_capability` promised in its docstring never to raise, and raised.
  `importlib.util.find_spec("google.genai")` imports the parent package first, so on a
  machine without `google` installed the probe died with `ModuleNotFoundError` instead of
  answering "not installed" — in the one function whose job is to report that calmly, and
  which `smem_health` and MCP startup both call.
- The npm and VS Code package lockfiles had drifted several releases behind their
  `package.json` files; every package is back in version parity.

## [2.16.0] — Consolidation honesty

Every counter in the consolidation report now measures what its name says, and three
subsystems that quietly did nothing have been repaired. Diagnosed by measuring the live
brain rather than reading the code, which is why one widely-believed cause turned out to
be wrong.

### The report stops lying

- **"Duplicates found: N"** was a *census* of anchors that look alike, printed as if it
  were work performed. It is now `Duplicate anchors (census): N (new alias links: M)`,
  where M is the number of ALIAS edges actually created this run. On a steady-state brain
  M goes to zero while N stays high — that is correct, and it is no longer alarming.
- **"Semantic synapses: 2000"** was the `semantic_discovery_max_pairs` cap printed as a
  flat number, so a truncated run and a stuck system looked identical. It is now
  `N created, K skipped (existing) [truncated at cap]`.
- **"Memories promoted"** counted only `context → fact` type promotions. It is relabelled
  `Memories promoted (type)`, and the real stage counter — `Stages advanced` — is finally
  printed. It existed; it was simply never shown.
- The `dedup` strategy's docstring, CLI help and docs all claimed it *merges* memories and
  *redirects fibers*. It does neither: it records ALIAS edges. Fixed in all four places.
  The community `smart_merge` docstring claimed it keeps the more-accessed neuron; it has
  always kept the longer content.

### Correcting a claim from 2.15.0

2.15.0 said memories could finally mature. **Promotable was not the same as promoted.**
`get_promotion_candidates` looked its fiber up with `id = $sid` bound to a `"fiber:<raw>"`
*string*, which can never equal a record id — the same fiber yields `count 0` that way and
`count 1` via `type::record` with a raw id. **No memory could be promoted at all.** That is
fixed, and the query returns candidates again.

Expect promotion to still take time: the gate requires 7 days in stage plus 3 distinct
recall days (or 15 rehearsals across five 2-hour windows), and only `recall`/`review`
rehearse — never `consolidate`. **The first organic promotions land after ≥3 days of
actual recalling.** The consolidation ratio will jump once on upgrade as the backfill runs,
then climb slowly.

### Write-time dedup was running on one of ~18 write paths

`MemoryEncoder` was given a dedup pipeline in exactly one place — the MCP `remember`
handler. Every other entry point built it without one, and `DedupCheckStep` returns
immediately when the pipeline is `None`, so the CLI, all three auto-capture hooks (the
highest-volume writers), the HTTP API, the cognitive and session handlers and the
integrations silently skipped dedup. Four CLI writes of byte-identical content produced
four separate anchors. Construction now lives in one factory that every write path uses;
the bulk trainers and the batch mapper opt out explicitly, at the call site, with reasons.

The alias neuron minted on a dedup hit was also flagged `is_anchor: true`. Both the
consolidation census and the write-time candidate pool select on that flag, so every alias
re-entered as a fresh duplicate on the next pass. It is now `false`.

### Health metrics

`get_fiber_stage_counts` grouped by `compression_tier`, which is `NULL` for every row — the
whole table collapsed into one bucket, so the caller's `.get("semantic")` was always 0 and
`LOW_CONSOLIDATION` was permanently on. It now groups by maturation stage, matching the
storage contract, the other two backends and `DiagnosticsEngine`, so the ratio is computed
from a real stage distribution instead of a single null bucket.

Fibers created before the maturation subsystem existed have no maturation row at all and so can
never advance a stage. The maturation
phase now backfills them (guarded on the counts, so a healthy brain pays two cheap counts),
and reinforcement creates a row on miss instead of silently skipping.

### What we expected to find and did not

The reported "Semantic synapses: 2000 every run" was widely assumed to be lost idempotency
— edges re-minted with fresh UUIDs on every pass. **It was not.** Two consecutive runs against a
real deployment created only genuinely new pairs and re-minted **zero** edges over an existing
pair, leaving the duplicate-row count unchanged. The
existing-pairs guard works, and those 198 rows are historical residue. What was actually
wrong was the *reporting*. Two real defects behind it were fixed anyway: that guard read
the entire synapse table in one response (the `[Errno 104] Connection reset by peer`
failure mode) and is now paged, and `SIMILAR_TO` edges now carry an id derived from the
sorted endpoint pair so idempotency is structural rather than dependent on a full-table
read.

### `smem surface`

`smem doctor` has always prescribed `smem surface generate`. **That command did not exist.**
It does now, with `generate` and `show`, plus `--global-path` for one predictable location —
surface paths were resolved from the process CWD, so `generate` run from home and `doctor`
run from a repo disagreed about which file they meant. Brain names are validated before
becoming filenames: passing a config object where a name was expected could write a surface whose
brain key was an entire `CLIConfig(...)` repr.

### Fixed

- `get_synapses` accepts `offset` on every backend, with a stable order, so large slices can
  be paged instead of loaded whole.
- `find_maturations` is paged rather than capped at 5000 rows.
- A dedup failure can no longer fail a write that would otherwise have succeeded.
- `make verify` passes on a clean checkout again: two stress tests still mocked
  `write_gate.enabled` after the gate moved to `effective_mode`, and `make install-dev`
  omitted the `surrealdb` extra the integration suite needs.
- `scripts/check_distribution.py` no longer reports a permanent false mismatch against a
  test fixture's snapshot-schema version, and the `.claude-plugin` manifests — stale at
  2.0.0 — are back in sync.

## [2.15.0] — Memories can finally mature, and the health metrics stop flattering the graph

Correctness release, and the most consequential one so far for anyone with a brain more
than a few weeks old. Two of these defects meant the system reported health it did not
have while quietly failing to do its central job: consolidating memories.

### ⚠️ Your purity score will drop on upgrade. That is the fix working.

Connectivity previously counted `alias` edges — dedup pointers with weight 0.0 that
carry no meaning. On a real brain they were **89.4% of all synapses** (137,871 of
154,252), so the metric measured plumbing, not knowledge. Connectivity now counts
semantic edges only.

Measured on an 11.4k-neuron brain:

| metric | before | after this release | after optional alias cleanup |
|---|---|---|---|
| connectivity | 1.0 (saturated) | **0.085** | 0.085 |
| diversity | 0.256 | 0.256 | **0.948** |
| purity | — | **−23 points** | **+14 points** |

Nothing degraded. The old number was a saturated sigmoid reading a graph that was 89%
bookkeeping. **Stored purity/grade values from before this release are not comparable
with values after it** — treat the upgrade as a new baseline.

### Fixed

- **Memories never matured (EPISODIC → SEMANTIC promotion was dead).** `Fiber.create`
  mints a dash-form uuid4, but the storage layer folds ids to underscore form. The
  `maturation` table stored whichever form the caller happened to pass, so maturation
  rows and fiber rows never joined. Measured on a live brain: of 1,920 maturation rows,
  1,277 carried a dash-form id and **not one of them had ever been rehearsed or
  promoted** — every rehearsed row (77) and every semantic row (9) was underscore-form.
  `get_maturation` returned `None`, `rehearse()` never ran, and the spacing gate could
  never be met. Pattern extraction was blind to the same 1,277 fibers. A blanket
  `except Exception` had been swallowing the resulting write collisions as "Skipping
  maturation save" for months.
  Lookups now resolve through the record id — the one identifier that was canonical in
  both eras — so legacy rows heal on their next write with no migration required.
  Expect the first consolidation run after upgrading to do noticeably more work: those
  1,277 fibers become promotable again.
- **Decay recomputed from creation on every run.** Nothing marked a synapse as already
  decayed, so each run re-measured from `created_at` and multiplied an already-decayed
  weight again — quadratic in run count (ten daily runs applied ~55 days of decay, not
  10). Decay now bills only the interval since the last decay, via the `_last_decayed`
  bookmark. The exponents telescope, so total decay per day of wall-clock time matches
  the documented Ebbinghaus curve. **Weights now fade slower than in previous builds** —
  that is the bug being removed, not a behaviour change.
  Rows with nothing to bill are skipped entirely, which removes the ~57k pointless
  rewrites per run.
- **Dedup re-inserted the same alias edge on every consolidation.** 2,375 distinct pairs
  had produced 144,565 rows, growing ~40k/day. Alias edges now use a deterministic id
  per pair, so re-running consolidation is idempotent.
- **Knowledge surface reported "not generated" depending on your shell's directory.**
  `$HOME` was being classified as a project root because it contains the global
  `.surrealmemory/` config dir, so `smem surface generate` and `smem doctor` resolved
  different paths. Project-level surfaces still work; the home directory no longer
  masquerades as a project.
- **Connectivity contradicted itself across the product.** `smem health`,
  `smem_stats`, the maintenance handler and topology analysis each computed it
  independently; after excluding structural edges in one place the others would have
  printed 13.5 where health printed 1.4. All now share one definition.
- **Warning texts collided units.** "connectivity 1.0 (target 3-8)" pairs a 0-1
  normalised score with a raw synapses/neuron target, reading as "you are far below
  target" when the raw ratio was 13.5 — far above. Texts now name their unit and state
  where the score sits on its own scale.

### Changed

- `LOW_DIVERSITY` no longer fires on brains that use many synapse types. The live brain
  uses **17** distinct types; the warning claiming "only 1 of 8 used" was reading an
  empty scope, not the real graph.

## [2.14.0] — Brain scoping, honest degradation, and consolidation that finishes

Correctness release. The theme is **silent failure**: three of these bugs made the
system look like it was working while it quietly was not — an empty-looking brain
that held 10k neurons, recall that skipped reranking without saying so, and writes
that landed but reported a timeout.

### ⚠️ Upgrade note

**If a deployment accumulated rows under a UUID brain scope, they need remapping.**
Rows are scoped by brain *name* (`brain_id = "default"`), but several call sites
passed the brain record's UUID primary key. Reasoning mining was one such path, so
its output landed in a scope recall never reads. Before this release a fetch by
record id still returned those rows; #63 now brain-scopes every by-id fetch, so
they become unreadable by *any* path until remapped:

```sql
UPDATE neuron  SET brain_id = "default" WHERE brain_id = "<uuid>";
UPDATE fiber   SET brain_id = "default" WHERE brain_id = "<uuid>";
UPDATE synapse SET brain_id = "default" WHERE brain_id = "<uuid>";
```

Check first with `SELECT brain_id, count() FROM neuron GROUP BY brain_id` — a
healthy install shows one scope per brain name and no UUIDs.

### Security

- **Cross-brain read via record id (#63).** Every fetch-by-record-id did a bare
  `select` with no `WHERE brain_id`, so a caller in brain A could read a record
  owned by brain B just by knowing its id. All six getters (`get_neuron`,
  `get_neuron_state`, `get_synapse`, `get_fiber`, `get_neurons_batch`,
  `find_neurons_by_ids`) now scope to the brain, fail-closed. Fetches stay pinned
  by id, so this is not a table scan — the dashboard-perf property is preserved.
- **Id sanitisers consolidated (#63).** `migrations._sanitize_id` and an inline
  fold in `tool_events` bypassed the single `_to_surreal_id` choke point.

### Fixed

- **Brain-owned rows scoped by UUID instead of name (#97, #98).** Ten call sites
  passed `brain.id` where rows are keyed by brain name. Reads reported an empty
  brain (grade F, `EMPTY_BRAIN`) on a brain holding 10k+ neurons; writes were
  worse — reasoning mining bound its whole job to a UUID scope, so mined neurons
  and fibers landed where recall never looks. `POST /brain` now also pins
  `brain_id` to the name, closing the source rather than the symptoms: previously
  any brain created through the dashboard or API was born with an id that did not
  match its own row scope.
- **Rerank degradation was silent (#97).** With the reranker enabled but
  unreachable, recall fell back to raw spreading-activation order and only logged
  a warning — results indistinguishable from reranked ones. Recall now retries
  once and reports the degradation (`rerank_degraded` in the MCP response,
  `rerank_degraded_warning` in the CLI). The reason string is sanitised, so a
  remote endpoint cannot leak a token through it.
- **Consolidation aborted on large brains (#97).** Per-strategy timeout had
  regressed to 120s, cutting the heavy passes mid-run so consolidation never
  converged; restored to 600s with the total budget raised to 3600s. `_lifecycle`
  fetched 10k neurons *with* embeddings in one response, which SurrealDB dropped
  mid-transfer (`[Errno 104] Connection reset by peer`); it reads only metadata,
  so it now pages with `include_embedding=False`. `_query` retried its reconnect
  once, in the same instant, hitting the same reset — now backs off (0s/1s/3s) and
  still fails fast on auth errors.
- **Writes reported a timeout after succeeding (#79, #99).** Inline embedding ran
  unbounded inside the write path, so a slow or rate-limited provider pushed
  `smem_remember` past the 30s tool-call cap MCP hosts impose — the memory landed
  but the caller could not tell. Now bounded at 10s (tune with
  `SURREAL_MEMORY_INLINE_EMBED_TIMEOUT`, `<= 0` restores the old behaviour); on
  timeout the memory stays keyword-only and `smem reindex` back-fills.
- **Dangling synapses survived neuron deletion (#89).** The cascade filtered on
  `brain_id`, but some write paths create synapses with a NULL `brain_id`, so
  those outlived the neuron they pointed at. A synapse whose endpoint is gone is
  dangling regardless of which brain it claims.
- **`SURREAL_MEMORY_BRAIN` ignored when `--brain` omitted (#90, #98).** Extracted
  into a single `resolve_brain()` helper; the duplicated
  `os.environ.get(X) or os.environ.get(X)` lookup is gone from both remaining
  sites.
- **En-dash was not a clause boundary (#65).** Editors and operating systems emit
  U+2013 as readily as U+2014, so the same cross-clause junk bigram slipped
  through. The ASCII hyphen stays a non-boundary, so `write-gate` still yields
  the bigram `write gate`.

### Added

- **Write-gate shadow/enforce modes (#93).** `mode: off | shadow | enforce`
  replaces the all-or-nothing boolean, with a per-intent `auto_capture_mode` so
  the gate can be enforced on passive auto-captures while interactive writes stay
  in shadow. `shadow` records decisions without rejecting anything, which is what
  makes the thresholds tunable against a real corpus. Backward compatible:
  defaults to `off` and falls back to the legacy `enabled` bool, so behaviour is
  unchanged until opted into. Adds `gate_decision` telemetry.
- **Machine-noise denylist (#94).** Empirically derived from real false-accepts
  (180× `Session activity:`, 30× `<summary>`, …) — session notifications and shell
  output are never knowledge.
- **BGE-M3 embedding provider (#91).** HTTP provider for a self-hosted BGE-M3
  service (1024D, L2-normalised), with a zero-vector guard: on failure it raises
  rather than returning a fabricated `[0.0] * dim`, which would poison top-k KNN
  silently. Reads `SURREAL_MEMORY_EMBEDDING_ENDPOINT` (with
  `SURREAL_MEMORY_EMBEDDING_BASE_URL` kept as a fallback — #98).
- **Reranker Bearer auth (#92).** `HttpReranker` sends `Authorization` when a key
  is configured and accepts both `relevance_score` and `score` response fields.
  An empty key sends no header, so llamastash and llama.cpp are unaffected.

### Changed

- **Lint is pinned (#96).** `ruff` was installed unpinned in CI, making Lint a
  moving target: 0.16.0 started reporting `RUF100`/`RUF036` on untouched files and
  turned `main` and every open PR red on the same day. Now `ruff==0.16.0` in both
  jobs.
- **mypy: ignore missing `sentence_transformers` stubs (#95).**

## [2.13.0] — Full-corpus reasoning mining: deep discovery, per-model targets, live progress

### Added

- **Full-corpus reasoning mining.** The miner now discovers transcripts
  recursively — nested session directories and Task-tool subagent transcripts
  (`projects/<project>/<session>/subagents/agent-*.jsonl`), not just the top
  `projects/<project>/*.jsonl` layer — so the ~1000 previously-invisible files
  are mined. Subagent traces are attributed to their real project, not a
  literal "subagents" pseudo-project.
- **Per-model pattern targets.** Each mineable model has a distillation target
  (0–100), set via a slider on the dashboard Reasoning tab (or
  `[reasoning_training.pattern_targets]` in config / `PUT /config`). A
  preliminary Mine with no targets set only DETECTS models; raising a target
  then distills that model's traces up to the target. Replaces the old global
  `max_patterns_per_run`.
- **Live mining progress.** `GET /api/dashboard/reasoning/status` and the
  dashboard show the phase (scanning → ingesting → distilling → done), files
  scanned/total, traces found/ingested, and per-model distillation progress
  while a run is active. The `smem reasoning mine` CLI prints the same, and
  `docker logs` shows INFO milestones at start / after ingest / at end.
- **Backfill is a true full re-scan.** `--backfill` (CLI / MCP / dashboard) now
  re-reads every transcript from the top, bypassing the incremental
  size+mtime/line skip (trace-hash dedup keeps it idempotent) instead of only
  widening the lookback window. It never deletes scan state, so a later normal
  scan stays cheap.

### Changed

- **opus-4-8 is now mined.** It was wrongly denylisted as "signature-only
  thinking"; the real corpus has ~933 opus-4-8 traces averaging ~1166 chars of
  genuine reasoning. The prefix-denylist mechanism is kept (currently empty) for
  a future thinking-less model.
- **No per-scan trace cap.** `max_traces_per_scan` is removed — mining ingests
  one transcript at a time (bounded memory) and sees ALL traces. The
  `max_trace_chars` content-safety limit is raised 20k → 100k.

### Fixed

- **Docker: reasoning mining found zero traces.** `docker-compose.surrealdb.yml`
  now mounts the host's `~/.claude/projects` read-only into the dashboard
  container at `/home/appuser/.claude/projects`. Without it the in-container miner
  scanned an empty `~/.claude` and every dashboard "Mine" finished instantly with
  0 traces (no error) — looking like nothing happened. Projects-only, read-only
  mount (not the whole `~/.claude`); override via `HOST_CLAUDE_PROJECTS` in `.env`.

## [2.12.1] — Release pipeline: npm packages actually ship again

Patch release. No runtime changes — it exists to fix the release pipeline and
re-align every published package on one version. npm packages had been silently
stuck at 2.10.5: versions 2.11.0 and 2.12.0 were never published to npm (the
registry rejected the re-upload of 2.10.5 with E403 and the workflow masked the
failure as a warning). Those npm versions are intentionally skipped — npm jumps
2.10.5 → 2.12.1.

### Fixed

- **Release workflow syncs npm package versions to the tag** — `publish-npm`
  (`surrealmemory`), `publish-sdk` (`@acidkill/surreal-memory-client`) and
  `publish-vscode` now run `npm version <tag> --no-git-tag-version` before
  building, so a release can never re-publish a stale `package.json` version
  again (the root cause: the pipeline bumped `pyproject.toml`/`__init__.py`
  but never the npm manifests).
- **Publish failures are no longer masked** — the `|| echo "::warning::…"`
  fallbacks are gone; with a token set, a failed `npm publish`/`vsce publish`
  now fails the job loudly. A graceful skip remains only for a missing token
  and for an idempotent re-run (this exact version already live).

### Changed

- `integrations/surrealmemory`, `integrations/surreal-memory-client` and
  `vscode-extension` manifests bumped to 2.12.1 so the repo matches the
  published artifacts.

## [2.12.0] — Reasoning training: learn how models think

An opt-in pipeline that mines a model's `thinking` blocks from `~/.claude`
transcripts, distills them into reusable reasoning-pattern fibers, and injects
the learned strategies into other models' sessions — surfaced across the
dashboard, CLI, and MCP. Off by default: nothing is mined or injected until you
enable it, and traces are redacted before they are ever stored.

### Added

- **`reasoning_traces` staging store** across all four backends (SQLite,
  SurrealDB, in-memory, base) with a schema migration (v39 → v40) and a
  `delete_reasoning_traces_by_model` wipe.
- **`reasoning_miner`** — scans `~/.claude` transcripts for model `thinking`,
  redacts secrets via `safety/sensitive` **before** insert, and stages the traces
  (`PROCESS_REASONING_TRACES` consolidation step).
- **`reasoning_distiller`** — segments reasoning moves → classifies → clusters
  them into ReasoningBank pattern fibers (idempotent by signature), restricted to
  the configured `mining_models` (`LEARN_REASONING` consolidation step).
- **`reasoning_injection`** — `SessionStart` and `UserPromptSubmit` hooks inject
  the learned patterns for the active model.
- **Dashboard reasoning page** at `/ui` — mining config, injection mapping, and a
  patterns table.
- **`smem reasoning` CLI** (`status` / `mine` / `patterns` / `clear`) and the
  **`smem_reasoning` MCP tool** (58 tools total).
- **`[reasoning_training]` config** — `mining_enabled` / `injection_enabled`
  (both default off), `mining_models`, plus backfill and privacy controls.

### Fixed

- **Hook output contract** — hooks now emit
  `{"hookSpecificOutput": {"hookEventName", "additionalContext"}}`; the previous
  `{"type": "context"}` shape was silently ignored by Claude Code, so
  `SessionStart` memory/reasoning injection never actually reached the model.
- **Shared-storage brain race** — long-running mining runs on an isolated,
  non-cached storage instance (`create_isolated_storage`) so a concurrent
  request's `set_brain` cannot redirect its graph writes into the wrong brain.

### Notes

- Fully opt-in and privacy-preserving: thinking traces are redacted before
  storage, no secrets reach evidence, and both mining and injection default off.

## [2.11.0] — Fast bulk doc-training

`smem train` on large documents is dramatically faster on big brains, and the
encoder now batches its writes instead of issuing one round-trip per neuron /
synapse. Three layers of work.

### Added

- `SurrealDBStorage.add_synapses_batch` (multi-statement `INSERT RELATION`)
  plus a base `add_synapses_batch` default with a per-synapse fallback, so
  non-SurrealDB backends stay correct.
- `SurrealDBStorage.find_neurons_exact_batch` override — one `content IN $contents`
  round-trip for N exact-content lookups (the base default was an N+1 sequential
  loop). `brain_id` is inlined as a literal here too.
- `tqdm` progress bar in `smem train` (optional import — tqdm is a transitive
  dependency, not declared — with a `logger.info` fallback every 50 chunks so a
  clean install without tqdm still reports progress).

### Changed

- **`increment_keyword_df`**: per-keyword SELECT-then-merge/insert N+1 → a single
  multi-statement UPSERT (`fiber_count = (fiber_count ?? 0) + 1`). This was the
  #1 per-chunk op count during doc-training (~93 keyword-DF SELECTs/chunk on a
  6651-chunk doc); it is now 1 query/chunk.
- **`CreateSynapsesStep` + `CoOccurrenceStep`** persist their synapses through a
  new `_persist_synapses` helper that uses `add_synapses_batch` (an `asyncio.gather`
  of N `add_synapse` collapses into one round-trip).
- **`find_neurons`**: `brain_id` is now an inline, charset-validated literal
  instead of a parameterized `$brain_id`. SurrealDB 3.2.0's planner only uses the
  `brain_id` index for an inline literal; a parameterized value full-scans the
  neuron table — the root cause of per-chunk doc-train cost scaling with brain
  size. EXPLAIN confirms the plan changes from `TableScan` to
  `IndexScan [idx_neuron_brain]`.

### Performance

- Isolated in-memory SDB 3.2.0, N=100 chunks: batch writes took per-chunk encode
  from **1.446 s → 0.847 s (-41%, under the 1 s target)**.
- DB-op count per chunk is ~10× lower (**581 → 58**) on a full 6651-chunk run.
- `find_neurons` is now index-driven on large brains — the dominant lever on
  disk-backed brains (a manual `smem train` on a 68k-neuron default brain
  previously ran at **7–15 s/chunk** for this reason; the index vs full-scan gap
  is the single biggest per-chunk cost on disk).

### Notes

- The doc-trainer now batches create/insert round-trips and uses the brain_id
  index, but per-chunk reads and Python-side extraction remain the next
  bottleneck on very large (>50k-neuron) brains — material for a follow-up.

## [2.10.5] — Consolidation scales, prune stops eating memories

Full `smem consolidate` now finishes without per-strategy timeouts on large
brains, session-less tool events finally build the USED_WITH tool graph, and
a data-integrity bug in dead-neuron pruning is fixed — see the first entry
below before relying on `prune`/`consolidate` on an existing brain.

### Fixed

- **`prune` no longer deletes live memory content.** Dead-neuron pruning
  (`access_frequency == 0` + old enough) never checked fiber membership,
  unlike orphan detection right above it in the same loop, which does.
  `reinforce()` only bumps `access_frequency` for the top-10
  highest-activation neurons per recall, so most neurons that are genuinely
  part of an actively-recalled fiber read `access_frequency == 0` forever —
  once prune's delete phase could actually complete (see the next entry),
  this deleted real memory content instead of dead junk. Measured live:
  57150 of 63380 neuron_states were fiber members reading
  `access_frequency == 0` (nearly the whole brain was wrongly eligible).
  Dead-neuron pruning now skips any neuron that belongs to a fiber, mirroring
  the orphan-detection guard. **If you've run `consolidate` (prune or `all`)
  non-dry-run on 2.10.0–2.10.4, check your brain's neuron/synapse counts
  against a recent backup** — this bug could only fire once prune's delete
  phase ran to completion, which the timeout below usually prevented, but a
  smaller brain (or one that later grew past the point where prune was
  timing out) could have been silently affected.
- **`prune`'s delete phase no longer costs ~1.2s per neuron.**
  `delete_neuron`'s synapse-cleanup query — `... WHERE brain_id = $b AND
  (in = X OR out = X)` — doesn't hit either `idx_synapse_in`/`idx_synapse_out`
  on SurrealDB 3.2.0: its planner falls back to a full scan across an OR of
  two different fields regardless of whether brain_id/the record id are
  inlined or param-bound. Splitting into two single-field DELETEs (each hits
  its own index) measured ~5ms total instead of ~1.2s — the dominant cost
  behind prune's 120s timeout once the read-side N+1 was fixed. Added
  `delete_neurons_batch`/`delete_synapses_batch` to the SurrealDB backend
  (previously SQLite-only); both are sequential, not concurrent — concurrent
  deletes raised live `Transaction conflict` errors under SurrealDB's
  transaction isolation, unlike concurrent reads, which are safe.
- **`compress` no longer times out on large brains.** On a ~67k-neuron brain
  the strategy hit the 120s per-strategy budget: `compress_fiber` made one
  `get_synapse` round trip per synapse id per fiber (~66k across 4.2k eligible
  fibers) and fetched synapses even for tiers that never use them. Synapses
  are now fetched in one bounded-concurrent batch and only for the
  ENTITY_ONLY/TEMPLATE tiers that actually render relations; neuron batch
  fetches are bounded-concurrent too. The compression run also takes a time
  budget (80% of the strategy timeout) and reports any remainder as
  `fibers_deferred` instead of being cancelled mid-run — deferred fibers stay
  eligible, so repeated runs drain the backlog in O(batch) work per run.
  Measured live: compress 120s-timeout → 72s clean; full `smem consolidate`
  zero timeouts. Note: on SurrealDB 3.2.0 a single `id IN [...]` query over
  RecordIDs measured ~8x *slower* than per-id direct selects (IN-membership
  skips the primary index), hence concurrency-shaped batching rather than an
  IN query.
- **Session-less tool events now form USED_WITH/EFFECTIVE_FOR synapses.**
  `process_events` dropped every tool event with an empty `session_id` from
  co-occurrence grouping, so on brains where tool events carry no session
  (all of them on a hook-fed brain) the tool graph never formed — 0 USED_WITH
  synapses. Session-less events now fold into one shared time-ordered stream
  (the same treatment tool-habit mining already uses), still bounded by the
  co-occurrence window. Verified live: 16 USED_WITH synapses formed from the
  pending backlog.
- **Test suite: deflaked the aiosqlite leak-guard pair under `pytest-xdist`**
  (`KeyError: 'thread'` when its two cooperating tests landed on different
  workers) by pinning them to one worker via `xdist_group` +
  `--dist loadgroup`, and isolated two env-sensitive tests from ambient
  `SURREAL_MEMORY_EMBEDDING_ENDPOINT`/`SURREAL_MEMORY_RERANKER_ENDPOINT`
  developer environments.
- **Test suite: the e2e API tests can no longer write into a configured live
  SurrealDB.** The `client` fixture redirected `SURREAL_MEMORY_DIR` to a temp
  dir, but a fresh config inherits `storage_backend` from the
  `SURREAL_MEMORY_STORAGE` env var — on a dev shell exporting
  `surrealdb` + `SURREALDB_URL`/`SURREALDB_PASS`, the suite created dozens of
  test brains in the production DB and xdist workers aborted each other with
  `Transaction write conflict` setup errors. The fixture now forces the
  sqlite fixture backend, strips the live-DB env vars, and resets the cached
  `_surrealdb_storage` singleton (which bypasses `_storage_cache`).

### Added

- `get_synapses_batch` on all storage backends (batched on SQLite/SurrealDB,
  sequential fallback on the base class), mirroring `get_neurons_batch`.

## [2.10.4] — Tool-usage habits

Tool calls now feed habit learning: repeated tool workflows (Read → Edit →
Bash, …) become listable habits, closing the gap where a months-old brain with
thousands of buffered tool events had almost nothing to show in
`smem habits list`.

### Added

- **Tool-usage habit mining.** The `learn_habits` consolidation strategy now
  also mines the `tool_events` buffer. Tool events carry no session id, so the
  history is treated as one time-ordered stream segmented by the sequential
  window; same-tool self-pairs (Bash → Bash) are dropped, and at most 25 new
  tool habits materialize per run (repeated runs drain the backlog). Learned
  habits are regular `_habit_pattern` WORKFLOW fibers — they show up in
  `smem habits list` alongside action habits, tagged
  `_habit_source: tool_events`. Backends without a tool-event buffer are a
  graceful no-op (`get_tool_events_for_mining` defaults to empty).

### Fixed

- **Habit learning is now idempotent.** Every consolidation run re-created the
  same action habit (e.g. a duplicate `recall-remember` fiber per run) because
  nothing consumed the mined events. Both action and tool habit mining now skip
  step-sequences that already exist as `_habit_pattern` fibers.
- **Habit confidence is clamped to [0, 1]** for tool habits — without session
  data the confidence formula degenerated to the raw frequency and
  `smem habits list` printed values like `Confidence: 93.00`.

## [2.10.3] — Prune completes, habits stick

Two user-reported bugs on large, months-old brains: `consolidate`'s prune step
timed out, and `smem habits list` was always empty.

### Fixed

- **Prune no longer times out on large brains.** The `prune` strategy fetched
  every neuron's activation state one row at a time (a hidden N+1 behind the
  `get_neuron_states_batch` name) and re-ran it for each 5000-neuron page. On a
  ~67k-neuron / ~420k-synapse brain that single step cost ~140s and blew the
  120s per-strategy budget. States are now read in one scan and looked up
  in-memory, and the SurrealDB backend gained a real batched
  `get_neuron_states_batch`. End-to-end prune drops from ~188s to ~39s on that
  brain. (Builds on 2.10.1's N+1 synapse fix and the embedding-vector OMIT.)
- **`smem habits list` now finds learned habits on large brains.** `find_fibers`
  filtered the `_habit_pattern` marker in Python *after* applying the row limit,
  so on a brain with more fibers than the fetch window any habit beyond the first
  1000 rows was invisible. The marker filter is now pushed into the SurrealDB
  query (`WHERE metadata.<key> != NONE`), and the `habits` CLI commands query the
  marker directly instead of `get_fibers(1000)` + a Python filter.
- **`consolidate` reports query patterns separately from habits.** "Habits
  learned: N" previously also counted query-pattern strengthenings — which are
  CONCEPT neurons/synapses, not listable `_habit_pattern` fibers — so the number
  never matched `smem habits list`. Query patterns now have their own
  "Query patterns learned" line.
- **Learned habits survive consolidation.** The `merge` strategy could fold a
  `_habit_pattern` fiber into a summary fiber and drop the marker, silently
  deleting the habit. Habit fibers are now excluded from merge clusters so
  habits accumulate over time.

## [2.10.2] — Faster inline embedding

A follow-up performance patch to [2.10.1]'s inline embedding.

### Fixed

- **Faster memory saves on large brains.** [2.10.1] embeds a memory's neurons as
  part of the save, but wrote each vector with its own database round-trip. Those
  writes are now batched into a single query, so a permanent `remember` on a
  66k-neuron SurrealDB brain drops from ~6s to ~3s. (SurrealDB backend; other
  backends keep the per-neuron path.)

## [2.10.1] — Local-first performance

A performance and correctness patch for the SurrealDB backend, focused on large
brains and local (self-hosted) embedding. No schema migration; recall ranking is
unchanged if you don't change your embedding config.

### What's now possible (and wasn't)

- **Point the embedder at a local, OpenAI-compatible server.** Reranking could
  already run against a local endpoint (`SURREAL_MEMORY_RERANKER_ENDPOINT`); the
  embedder now has the same knob: `SURREAL_MEMORY_EMBEDDING_ENDPOINT`. Set
  `provider = "openai"`, `model = "bge-m3"` (or your model), and
  `SURREAL_MEMORY_EMBEDDING_ENDPOINT = http://127.0.0.1:11435/v1` to embed and
  rerank entirely on your own GPU (e.g. bge-m3 + bge-reranker via llama.cpp /
  llamastash) with no cloud calls. A placeholder API key is supplied automatically
  for keyless local servers.

- **Fresh memories are semantically searchable immediately.** Previously a
  just-saved memory only got its embedding vector on the next batch
  `smem reindex`, so semantic recall missed it until then; it was keyword-only in
  the meantime. `remember`/encode now embed the memory inline as part of the save.
  Fully fail-soft: with embeddings disabled or no provider reachable, saves behave
  exactly as before (keyword-only, no error, no slowdown).

### Fixed

- **`smem consolidate` no longer times out on large brains.** The `prune` strategy
  issued one query *per candidate source neuron* (tens of thousands on a 66k-neuron
  brain) for bridge detection, blowing the 120s per-strategy budget. It now groups
  the already-loaded synapses in memory — zero extra queries — dropping prune's read
  path from ~110s to a few seconds.

- **No more `FLEXIBLE can only be used in SCHEMAFULL tables` warnings on upgraded
  databases.** Tables that carry an arbitrary-key `metadata`/`config` object
  (neuron, fiber, brain, typed_memory, source, alerts, brain_versions) are now
  declared `SCHEMALESS`, which accepts nested keys without `FLEXIBLE`. On a database
  whose tables were first created `SCHEMALESS`, the old `SCHEMAFULL` + `FLEXIBLE`
  definitions failed on every `ensure_schema()`/`consolidate`; converting such
  tables to `SCHEMAFULL` was unsafe (it breaks updates of any row holding a legacy
  field). `synapse` and `retrieval_trace` stay `SCHEMAFULL`.

- **Session-end dedup recognizes a local OpenAI-compatible embedder.** The stop
  hook treated only `ollama` as a cheap local embedder and fell back to simhash for
  everything else; a loopback `SURREAL_MEMORY_EMBEDDING_ENDPOINT` (llamastash bge-m3)
  is now recognized as local-and-cheap, so end-of-session semantic dedup uses it.

## [2.10.0] — Ecosystem

The ecosystem release: recall gains a geographic dimension, and surreal-memory plugs
directly into the LangChain RAG stack. **No schema change and no migration** — both
features are purely additive, and the default recall ranking stays byte-identical if
you don't use them.

### Added

- **Geospatial recall** — recall can now be scoped to "near a place", not just "similar
  in meaning". Attach a location when you store a memory, then filter recall to a
  radius around a point:

  ```python
  # Store a location alongside a memory
  smem_remember(
      content="Met the landlord to sign the lease",
      location={"lat": 59.9139, "lon": 10.7522, "label": "Oslo"},
  )

  # Only recall memories within 50 km of a point
  smem_recall(
      query="lease signing",
      near={"lat": 59.9139, "lon": 10.7522, "radius_m": 50_000},
  )
  ```

  This is useful for agents that operate across multiple sites, travel logs, or any
  memory that's meaningfully tied to a place — "what did we discuss at the Oslo
  office" now works as an actual filter, not just a keyword match. `near` behaves
  like the existing `valid_at` filter: it's exact (real-world great-circle distance,
  not a bounding box), memories without a location are excluded when `near` is set,
  and omitting `near` entirely leaves recall completely unchanged. No schema
  migration is needed — the location lives in the memory's existing metadata field.
  Under the hood, the same distance filter works identically whether your brain runs
  on the in-memory backend, SQLite, or SurrealDB.

- **LangChain adapter** (optional — nothing changes unless you install it):

  ```bash
  pip install surreal-memory[langchain]
  ```

  ```python
  from surreal_memory.adapters.langchain import (
      SurrealMemoryRetriever,
      SurrealMemoryChatMessageHistory,
  )

  # Drop surreal-memory into any LangChain RAG chain as the retriever
  retriever = SurrealMemoryRetriever(brain_name="my_brain", k=5)
  docs = await retriever.ainvoke("what did we decide about the backend?")

  # Or use it as a chat history — every turn is stored as a real, queryable memory
  history = SurrealMemoryChatMessageHistory("session-42", brain_name="my_brain")
  ```

  This means you can build a LangChain agent or RAG pipeline that uses surreal-memory
  as its long-term memory with a couple of lines, instead of writing custom glue code
  against the MCP tools. Runs in-process (no server required), works with both async
  and sync LangChain code. See `examples/langchain_rag.py` for a full working example
  (a LangChain chain that retrieves context and remembers the conversation).

### Fixed

- `find_fibers(tags=...)` now pushes the tag predicate into the database query itself,
  so results are filtered *before* the result-count limit is applied instead of after.
  In practice: on a large brain, a tagged subset (e.g. one chat session's messages)
  could previously be cut off by the limit before the tag filter even ran, silently
  losing part of that session's history. That's fixed for all three backends.

## [2.9.0] — Memory you can trust

The trust release: memories now carry validity over time, recall stops surfacing facts
that have been replaced by newer ones, and both the MCP tools and the dashboard can
tell you *how much to trust an answer* — not just what the answer is.

**In practice**, this is the difference between:

```
smem_remember "Alex lives in Oslo"
smem_remember "Alex moved to Bergen"
smem_recall "where does Alex live?"
```

Before 2.9.0, both facts could surface side by side, leaving it to the reader to guess
which one is current. Now, recall automatically resolves this: the old fact is marked
superseded and stays out of the default answer ("Bergen"), while remaining fully
recoverable — either for its own sake (`include_superseded: true` returns both, with
the timeline) or for a specific point in time (`valid_at: "2026-01-01"` recalls what was
true back then, i.e. "Oslo"). Nothing needs to be deleted or manually corrected — the
old fact's history is preserved, just no longer presented as current by default.

### Changed

- **Superseded facts are hard-filtered from recall by default** (the one intended
  default-behaviour change in this release). When a newer fact replaces an older one,
  the old fact's validity window is closed (`valid_until` set) and it no longer
  surfaces in `smem_recall` by default. Three escape hatches if you need the full
  history: `include_superseded: true` per call, the environment variable
  `SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER` to disable the filter process-wide
  (superseded facts still surface, just demoted in ranking), or `valid_at: "<ISO
  timestamp>"` to recall the state of the world as of a specific moment.
- `smem_stats` now reports `conflict_rate` (active conflicts ÷ neurons); `smem_uncertainty`
  and the dashboard report `contradiction_rate` (active conflicts ÷ typed memories) — two
  distinct, consistently-named metrics instead of one overloaded name.

### Added

- **Schema v9** — the storage schema that makes all of the above possible.
  Migration is automatic, additive, and idempotent (SurrealDB 8→9, SQLite 38→39): it
  only adds new fields (typed-memory `valid_from` / `valid_until` / `superseded_by`, a
  `retrieval_trace` table, `source.trust`) and never touches or removes existing data.
  Upgrading is a normal `pip install --upgrade`; no manual migration step is required.
- **Trust & recency calibration** — you can now tell surreal-memory how much to trust
  different sources, and how quickly relevance should decay with age, instead of
  treating every memory as equally reliable and equally fresh forever. Configurable
  `trust_weight` / `recency_weight` / `trust_default` on the brain, and a `trust` field
  on `smem_source`; recall factors in per-source, per-source-type, and per-label trust
  when ranking results. The defaults are neutral, so existing brains rank results
  exactly as before until you opt in by setting a trust score somewhere.
- **Per-fact supersession** (the mechanism behind the trust behaviour above): a hard
  recall filter plus `valid_at` point-in-time recall plus the escape hatch described
  above. Supersession happens automatically when you remember something that
  contradicts an existing fact, or manually via `smem_conflicts` (`keep_new`).
  `smem_provenance` can trace the "replaced by / replaces" lineage in both directions,
  and `smem_lifecycle action=backfill_supersession` retroactively stamps supersession
  onto conflicts that were resolved before this feature existed.
- **Queryable retrieval traces** — an opt-in, off-by-default record of *why* a recall
  returned what it did. Pass `trace: true` to `smem_recall` to get back a `trace_id`,
  then inspect it later with `smem_provenance action=traces` / `action=trace_get`. Useful
  for debugging "why didn't it remember X" or auditing what informed a given answer;
  traces are pruned automatically during consolidation so they don't accumulate forever.
- **Uncertainty surfacing** — a direct answer to "how much should I trust this?".
  `smem_uncertainty` gives you an overview (or drills into contradictions, drift,
  soon-expiring memories, or low-evidence facts specifically), and `smem_recall
  include_uncertainty: true` attaches a summary of the same signals to a single recall.
- **Dashboard**: a new **Uncertainty** page visualizing all of the above at a glance
  (via a new `/api/dashboard/uncertainty` endpoint), and the existing Health page/report
  now surface `conflict_rate` and `contradiction_count` alongside the brain's grade.

### Fixed

- SurrealDB record-id comparison: `id = $bare` never matched a `RecordID` — sources
  (`get` / `update` / `delete_source`) now match via `type::record`.
- SurrealDB fiber-id normalization: `get_typed_memory` and batch lookups now resolve both the
  original (dash) and loaded (underscore) id forms, restoring recall's sources / trust map.

## [2.8.0] — 2026-07-11

### Removed

- **Vietnamese language support and the cross-language "translation" layer.** The
  long-unmaintained bilingual surface has been dropped — extraction (keywords,
  entities, sentiment, temporal, relations, arousal, prediction-error reversal),
  query expansion, and auto-capture are now **English-only**:
  - Removed the Vietnamese lexicons/patterns and helpers (`_is_vietnamese`,
    `_strip_diacritics`, `normalize_vietnamese_compound`, `_tokenize_vietnamese`,
    `_extract_vietnamese_names`, `_resolve_vi_hour`, `detect_language`,
    `_get_stop_words(language)`), the EN↔VI `CROSS_LANG_MAP` query-expansion pairs,
    and the recall handler's `_check_cross_language_hint`.
  - Dropped the optional `[nlp-vi]` extra and its `underthesea` / `pyvi`
    dependencies (plus the matching mypy overrides and the `pyvi` warning filter).
  - Deleted the Vietnamese and cross-language test suites and pruned Vietnamese
    cases from the shared unit / integration / e2e tests and benchmark scripts.
  - Embedding-level multilingual recall is unaffected — it is a property of the
    embedding model (e.g. Gemini, `paraphrase-multilingual-*`), not of the removed
    extraction layer. The extraction `language` parameter is retained for
    backward-compatible call sites but is ignored (English-only). See
    [`docs/architecture/vietnamese-removal.md`](https://github.com/acidkill/surreal-memory/blob/main/docs/architecture/vietnamese-removal.md).

### Fixed

- **Auto-mode keyword/concept extraction no longer forms cross-clause bigrams or
  drops real words.** Bigrams now require both words in the same clause (split on
  `.,;:!?\n\r` and the em-dash) with a tightened position gap (`<= 2`). A Polish
  stop-word set (`STOP_WORDS_PL`, diacritic + ASCII) is added so bare Polish
  function words stop surviving as keywords; ASCII short-forms that collide with
  domain acronyms (e.g. `ci` → "CI", continuous integration) are excluded from the
  Polish set.

- **`SurrealDBStorage` now reconnects after a dropped transport, not only on a
  401.** A DB container restart (backup, upgrade, reboot) severs the WebSocket and
  surfaces as a connection error rather than an auth error, so the previous
  retry-on-401-only path left the cached dead connection in place — every
  subsequent query failed and long-lived clients returned `-32000` until the
  process was restarted. `_query` now also reconnects on connection/transport
  errors (`_is_connection_error`), using the same single-retry path; genuine query
  errors are unaffected. Connection detection excludes `TimeoutError`, so a
  legitimate slow-query timeout under the HTTP transport is not misread as a drop.

### Security

- **Hardened `_to_surreal_id` against record-id / SurQL injection.** Ids are
  inlined into record literals and raw SurQL across the storage layer; the previous
  sanitiser only replaced `-` with `_`, leaving quotes, braces, semicolons and
  comment markers intact — a statement-literal breakout reachable via the REST
  `get_path` route (and `eval::gql` when enabled). Every character outside
  `[A-Za-z0-9_]` is now folded to `_`, a store-layer `_safe_brain_id` fail-closed
  guard covers the dot/hyphen-bearing brain-id path, and the 13 duplicated sanitiser
  copies are consolidated into one shared implementation so the guarantee holds
  storage-wide. Behaviour-preserving for legitimate ids (UUID4 / content-hash).

## [2.7.4] — 2026-07-11

### Performance

- **`recall` is now index-backed end-to-end (≈25s → ≈8s on a 65k-neuron /
  295k-synapse brain).** Two hot-path scans dominated a deep recall, pushing it
  against the MCP client's 30s tool timeout (which is why the CLI, with no
  timeout, still returned while the MCP tool "didn't respond"):
  - **Content lookup full-scanned the neuron table.** `find_neurons` matched
    `content` with `CONTAINS`, a case-sensitive substring scan (~2.9s/call, ~10
    calls per recall). A BM25 `FULLTEXT` index (`idx_neuron_content_fts`, with a
    lowercase analyzer) plus the `@@` operator brings this to ~0.9ms. Matching is
    now case-insensitive — a net improvement for keyword/entity recall.
    `suggest_neurons` keeps `CONTAINS` (prefix autocomplete needs substring).
  - **Neighbor traversal full-scanned the synapse table.**
    `get_neighbors(direction='both')` matched `(in = .. OR out = ..)`, and an
    `OR` across two indexed columns disables `idx_synapse_in`/`idx_synapse_out`
    (~950ms/call, ~26 per recall). Each direction is now queried with its own
    indexed equality and merged (dedup by id), preserving the inline `in.*/out.*`
    neighbour fetch: ~38ms.

### Changed

- **Dashboard overview and graph are cached per brain for a short window**, so
  repeat loads are instant. The grade/purity (a full `DiagnosticsEngine.analyze`)
  and the graph payload each aggregate over the whole synapse graph — a few
  seconds on a large brain that no index removes — but both are slow-moving, so
  recomputing them on every request was waste. The overview's neuron/synapse/fiber
  counts stay live (they're cheap, indexed `count()`); only the multi-second
  grade/purity and the graph structure are cached. TTL is configurable via
  `SURREAL_MEMORY_DASHBOARD_CACHE_TTL` (seconds, default 60); set it to `0` to
  disable caching entirely.

### Fixed

- **Broken SurrealDB test fixture** — the all-types-roundtrip fixture called
  `SurrealDBStorage.connect()`, which does not exist (the method is
  `initialize()`); it errored whenever a live SurrealDB was configured.

## [2.7.3] — 2026-07-09

### Fixed

- **Dashboard endpoints are now fast on the large station brain.** Follow-up to
  2.7.2 after measuring the real bottlenecks on 64k neurons / 266k synapses:
  - **Parameterized `brain_id` defeated the index.** SurrealDB 3.2.0's planner
    only uses the `brain_id` index for an *inline literal* — `WHERE brain_id = $bid`
    fell back to a full table scan, turning `count() … GROUP ALL` on the neuron
    table (each row carrying a 1024-float vector) from 0.01 s into ~2.5 s.
    `get_stats` now inlines the (strictly validated) brain_id and runs the three
    counts concurrently. This was the single biggest dashboard cost.
  - **Diagnostics skips the neuron type-breakdown** (`get_enhanced_stats(include_neuron_types=False)`)
    — a ~2.6 s `GROUP BY type` scan the health metrics never read.
  - **Independent reads run concurrently** (`asyncio.gather`): the diagnostics
    fibers/connected/activation reads, the connected-neuron `in`/`out` scans, and
    the graph's degree `in`/`out` scans (measured ~1.6x on the shared connection).
  - **Only the active brain gets full diagnostics** in `/api/dashboard/stats`;
    other brains report counts only, so a station carrying integration-test
    residue brains no longer pays a full diagnostic pass for each.

  Net on the station: `/api/dashboard/stats` ~27 s -> ~2.8 s, `/api/graph` 40 s+ ->
  ~4 s, `/api/dashboard/timeline` ~3 s -> ~0.06 s.

- **The web-UI container binds loopback only again.** `Dockerfile.surrealdb`
  hardcoded `uvicorn --host 0.0.0.0`, which silently overrode the compose stack's
  `SURREAL_MEMORY_HOST=127.0.0.1` (added in 2.7.2 for host networking) and exposed
  the dashboard on every interface. The CMD now honours `SURREAL_MEMORY_HOST` /
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

Storage-agnostic improvements ported from upstream
[nhadaututtheky/neural-memory](https://github.com/nhadaututtheky/neural-memory) and adapted
to the SurrealDB-only backend, plus two follow-up fixes.

### Added

- **`chat-heavy` config preset** — a conversational-agent profile (Telegram/Discord/Slack):
  faster decay, recency-biased recall, compact context. Apply with `smem config preset chat-heavy`. (#31)
- **`smem_offload` / `smem_inflate` MCP tools** — store a large tool result as an ephemeral
  neuron (24h TTL) and get back a compact `ref_id` + summary, then drill into the full content
  on demand. Keeps bulky tool output out of the agent's context. (#31)
- **`smem_situation` MCP tool** — one-shot snapshot of the working situation (active session
  task, top recent decisions, open blockers), so an agent can resume context without chaining
  `smem_recap` + multiple `smem_recall` calls. (#31)
- **`prefer_recent` recall flag** — re-ranks matched fibers newest-first (by `time_end`, falling
  back to `created_at`) for "current state" queries; off by default. (#31)
- **`verbose_extraction` flag on `smem_remember`** — surfaces concept-extraction observability
  counters (`dropped_short` / `dropped_noise` / `dropped_duplicate_entity`) so you can see why
  concept neurons were filtered; off by default. (#31)
- **`[brain]` config extras pass-through** — `BrainSettings` now forwards any `[brain]` key in
  `config.toml` that maps to a real `BrainConfig` field, so new tuning knobs become
  config-controllable without a parallel setting each, and already-stored brains pick them up
  on load. (#31, upstream #168)
- **Case-insensitive tag matching** — tags are normalized (lowercased) at every write and read
  boundary, so `KB`, `kb`, and `Kb` all match. (#33)
- **Dashboard Storage tab rebuilt for SurrealDB** — the Storage tab now shows live SurrealDB
  backend status (URL, namespace, database, connection health, and neuron/fiber/synapse counts)
  via the new `GET /api/dashboard/storage/status` endpoint. (#34)

### Changed

- **Lighter PostToolUse hook** — rewritten to be stdlib-only (no heavy imports on the hot path),
  with a noise-tool fast-path filter, lock-safe JSONL append (POSIX `flock` / Windows fallback),
  and `CODEX_SESSION_ID` support alongside `CLAUDE_SESSION_ID`. (#32)
- **Plugin hook de-duplication** — when running as the Claude Code plugin, first-time init skips
  `setup_hooks_claude()` so hooks are not registered twice (the plugin's own `hooks.json` owns
  registration). (#31, upstream #169)
- **MCP tool count is now 56** (was 53) with the three new agent-ergonomics tools above. (#31)

### Removed

- **Dead multi-backend dashboard Storage UI** — the legacy SQLite↔InfinityDB migration components
  (`MigrationCard` / `MigrationProgress`) were removed; surreal-memory is SurrealDB-only, so the
  migration flow no longer applies. (#34)

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

## [4.53.4] — 2026-04-30

### Fixed — Sandbox Hang at Every Storage Entry Point (#151)

Issue #151 reported that `smem` CLI commands hang silently inside the Codex
workspace sandbox. Root-causing surfaced an environmental issue, not a Neural
Memory bug: `aiosqlite` relies on `loop.call_soon_threadsafe()` to wake the
main event loop from its worker thread, but the sandbox blocks that wakeup.
The result is a never-resolving future on `aiosqlite.connect()` — every
storage-backed command (`smem context`, `status`, `stats`, `list`, `today`,
`health`, `doctor`) freezes with no output.

PR #152 (from the issue author) patched only `smem doctor`; this release
expands the fix to all four storage entry points.

#### Shared sandbox probe

- `src/surreal_memory/utils/sandbox.py` (new) — runs a 2-second
  `aiosqlite.connect(":memory:")` probe, caches the verdict process-wide,
  exposes `ensure_aiosqlite_or_exit_cli()` (CLI) and
  `ensure_aiosqlite_or_raise()` (server). Honours
  `NMEM_SKIP_AIOSQLITE_PROBE=1` for explicit opt-out.

#### Wired into every entry point that touches storage

- `cli/_helpers.py:run_async` — covers all storage-backed CLI commands.
  Hangs become a clear `typer.Exit(2)` with hint pointing at issue #151.
- `mcp/server.py` — both stdio (`run_mcp_server`) and HTTP (`main`)
  transports probe before any handler runs. Failure writes a structured
  error to stderr (so MCP clients log the real reason) and `sys.exit(2)`
  instead of leaving clients connected to a hung server.
- `server/app.py:lifespan` — FastAPI startup raises
  `SandboxIncompatibleError` instead of blocking the lifespan event.
- `cli/doctor.py:_check_dependencies` — reuses the shared probe so
  `smem doctor` surfaces the runtime failure beside the importability
  check.

#### Doctor stays responsive in broken sandboxes

- `cli/doctor.py:_check_schema_version` — switched from `aiosqlite` to
  sync `sqlite3`. Doctor now completes (and reports the runtime probe
  failure) even when `aiosqlite` itself is hanging.

### Tests

- `tests/unit/test_sandbox_probe.py` — 23 cases covering probe basics
  (success, caching, force re-run, reset), bypass env var parsing,
  failure paths (mocked timeout + real stuck connect), CLI exit
  contract, server raise contract, doctor integration, and CLI
  `run_async` integration.

## [4.53.3] — 2026-04-28

### Fixed — Lifecycle Integrity & CLI Parity (#148)

Issue #148 reported two confusing failures when soft-deleting a TODO fiber via MCP `smem_forget`: (1) the CLI had no `forget` command, forcing all delete flows through MCP, and (2) `smem recall` continued surfacing fiber IDs that `smem_show`/`smem_forget` already considered "Memory not found". Root-causing surfaced a single class of bug — recall, show, and forget each used a different definition of "fiber exists" — plus a parity gap between the MCP and CLI surfaces.

#### Recall no longer leaks soft-deleted or untyped fibers

- `storage/sqlite_fibers.py`, `storage/sql/mixins/fibers.py`, `storage/postgres/postgres_fibers.py` — `find_fibers_batch` now LEFT JOINs `typed_memories` and filters `expires_at <= now()`. Fibers soft-deleted via `smem_forget` (which only sets `expires_at`) are dropped from recall immediately instead of waiting for the next consolidation pass. Untyped fibers (no `typed_memory` row) still surface — the LEFT JOIN treats them as never-expiring.

#### Show / forget now handle untyped fibers

- `mcp/provenance_handler.py:_show` — used to short-circuit on `typed_mem is None` and return "Memory not found" even when the fiber row existed. Now fetches `typed_memory` and `fiber` independently; returns fiber data with `memory_type=null` plus a warning when no typed_memory exists, so users can `smem forget --hard` the orphan.
- `mcp/lifecycle_handler.py:_forget` — same independent lookup. Hard delete on an untyped fiber now removes the fiber row (and skips the missing typed_memory). Soft delete on an untyped fiber returns a clear error pointing to `--hard`, since there is no `expires_at` to set.

#### CLI parity for the lifecycle CRUD subset

- `smem forget <id> [--hard] [--reason "…"]` — soft- or hard-delete via CLI without an MCP server. Useful for cron cleanup, scripted teardowns, and debugging when MCP is unavailable.
- `smem show <id> [--json]` — inspect a memory by ID. Works for typed fibers, untyped fibers, and bare neurons.
- `smem edit <id> [--type T] [--content X] [--priority N] [--tier T]` — type/content/priority/tier edits.
- `smem pin <id>` / `smem pin list` / `smem unpin <id>` — pin lifecycle parity.

CLI commands reuse the existing MCP handler bodies via a small `_CliMcpFacade(LifecycleHandler, ProvenanceHandler, TrainHandler)` shim, so behavior stays identical across surfaces.

#### Diagnostics

- `smem doctor` now surfaces an "Orphan fibers" check that counts fibers with no `typed_memory` row. Suggests `smem show <id>` + `smem forget --hard <id>` for cleanup.

### Tests

- `tests/unit/test_issue_148_lifecycle.py` — 6 regression cases: soft-deleted fibers excluded from recall, untyped fibers still surface in recall, future-dated `expires_at` still surfaces, `_show` returns untyped fibers with a warning, `_forget --hard` removes an untyped fiber, `_forget` (soft) on an untyped fiber rejects with `hard=true` hint.
- 108/108 lifecycle/forget/edit/tag-filter/pinned/related/performance tests green; full mypy clean.

### Strategic note

Audited all 57 MCP tools against the CLI surface. Decided against full 1:1 parity — most agent-only tools (cognitive, narrative, suggest, auto, batch, recap, …) would just bloat the CLI without serving humans. Only the CRUD lifecycle subset (forget/show/edit/pin) was added; ops tools like `conflicts`, `review`, `budget`, `tier` remain MCP-only until a concrete user need lands.

## [4.53.2] — 2026-04-26

### Fixed — Dashboard Migration Endpoint (#147 follow-up)

A post-ship audit of v4.53.1 caught two more occurrences of the same class of bug the CLI fix had just shipped: the dashboard's `POST /api/storage/migrate` endpoint had the broken `from surreal_memory.storage.sqlite import SQLiteStorage` (module is `sqlite_store`) and a missing `SQLiteStorage.list_brains()` call. CLI users were already safe — dashboard users still hit `ImportError` / `AttributeError` until this patch.

- `server/routes/dashboard_api.py:2158` import switched to `sqlite_store` (same drift as `cli/commands/migrate.py:152` had).
- `SQLiteStorage` now exposes `list_brains()` returning `[{"id": ..., "name": ...}]`, mirroring `InfinityDBStorage` / `SQLStorage` so backend-agnostic callers (`_run_migration_task`) work uniformly across all 3 backends. Convention is still single-brain-per-file but we read the `brains` table rather than assume.

### Improved — Fiber Round-Trip Fidelity

- `pro/infinitydb/migrator.py` fiber loop now also preserves `summary` under `metadata["summary"]`. It was already surfaced as `name` / `description` post-migration, but downstream readers using the `summary` key wouldn't find it. Additive fix — no breaking change.

### Tests

- `TestSQLiteListBrains` (3 cases: single brain, empty DB, multiple brains in created_at order).
- `test_fiber_summary_preserved_in_metadata` proves the `summary` field round-trips through InfinityDB metadata.
- 506/506 Pro + storage + migration suite green.

### Side benefit — Hidden mypy bug surfaced

With the `sqlite_store` import path now correct, mypy could finally resolve `SQLiteStorage` and caught a pre-existing `Optional[str]` narrowing issue at `dashboard_api.py:2184`. Fixed under the same patch — not a runtime bug, but mypy was effectively blind to this file before.

## [4.53.1] — 2026-04-26

### Fixed — InfinityDB Migration End-to-End (#147)

Issue #147 reported `smem migrate infinitydb` failing on a fresh v4.53.0 install. Five interlocking bugs caused the command to crash, and even when worked around manually, the migrated InfinityDB read as empty under `smem health`. All five fixed in a single patch:

- **Wrong import** — `cli/commands/migrate.py:152` imported `surreal_memory.storage.sqlite` (module renamed `sqlite_store` long ago). Crashed on first run with `ModuleNotFoundError`. Fix: drop the SQLiteStorage adapter detour entirely; drive `SQLiteToInfinityMigrator` directly the way the user proved works.
- **Missing API** — `migrate.py:158` called `await source.list_brains()` which does not exist on SQLiteStorage. The migrate command no longer needs it after the rewrite.
- **Path mismatch (the silent killer)** — `migrate.py:197` opened `InfinityDBStorage(str(brain_dir))`, putting writes at `brains/<name>/default/brain.inf` while runtime reads from `brains/<name>/brain.inf`. Migration now opens `InfinityDB(brains_dir, brain_id=name)` — a literal mirror of `unified_config._build_storage`. Plus a re-verify pass that reopens at the runtime path before declaring success.
- **Fibers silently skipped** — `pro/infinitydb/migrator.py:403` skipped any fiber missing both `id` AND `name`, but the production fibers schema (`sqlite_schema.py:832`) has NO `name` column — `summary` is the canonical label. **100% of modern fibers were dropped** (e.g. 30/30 in the issue). Now skips only on missing `id`, falls back to `summary` then `id` for the InfinityDB label, records a diagnostic when skipping.
- **Missing dep** — `pyproject.toml` `pro` extra missing `sortedcontainers` even though `pro/infinitydb/metadata_store.py` imports `SortedList`. Install succeeded, runtime ImportError on first InfinityDB open. Added `sortedcontainers>=2.0` to the Pro extra.

### Tests

- 2 new regression tests: `test_fibers_modern_schema_no_name_column` proves fibers with `summary`-only schema migrate correctly, `test_runtime_path_round_trip` proves data written via the CLI path is readable through the exact `(base_dir, brain_id)` pair the runtime uses.
- 34/34 migration tests pass, 454/454 Pro + migration suite green.

## [4.53.0] — 2026-04-24

### Performance — Defer Post-Recall Side-Effects (~15-20% recall latency)

Profile of `ReflexPipeline.query()` on a real brain (3,488 neurons) showed ~19ms of the critical path spent on four blocking DB-write blocks that don't contribute to the returned result: reinforcement + `batch_update_last_accessed`, calibration/retriever-outcome/depth-prior writes, deferred write queue flush, reconsolidation loop, and session summary persist.

- **`ReflexPipeline._spawn_background()` + `_background_tasks` set.** Side-effectful writes are scheduled as `asyncio.Task`s; exceptions logged at debug, never raised. The returned `RetrievalResult` reflects everything user-visible (metadata, confidence, context) but no longer waits for the subsequent DB writes.
- **Storage-side task registry (`storage._pipeline_bg_tasks`).** Each spawned task registers on the storage object so backend `close()` can drain pending writes before releasing the file handle. Critical on Windows where `tempfile.TemporaryDirectory` cleanup races aiosqlite connection teardown.
- **`SQLiteStorage.close()` + `SQLStorage.close()` drain pending pipeline tasks** via `asyncio.gather(..., return_exceptions=True)` before running their existing connection-teardown logic.
- **`ReflexPipeline.flush_background_tasks()`** for tests or explicit shutdown paths that need write determinism.
- **Result (30-query real-brain profile):** TOTAL mean 118.8ms (Apr 12 baseline) → 64-80ms (post-fix). Four deferred blocks now contribute ~0ms to the hot path.

### Tests

- `tests/unit/test_pipeline_background_tasks.py` — 5 cases: tasks are spawned on query, `flush_background_tasks` awaits all, flush on empty pipeline is no-op, `storage.close()` drains tasks, failing coroutines don't bubble.

### Tooling

- `scripts/profile_recall.py` retained (unchanged). Re-run after any retrieval change to confirm post-recall side-effects stay off the hot path.

### Added — `smem_causal` temporal query actions

Wires the previously-dead `query_temporal_range()` + `query_temporal_neighborhood()` engine functions in `engine/causal_traversal.py` through to the MCP surface. Two new actions on `smem_causal`:

- **`temporal_range`** — list fibers whose time window intersects `[start, end]` (ISO-8601), sorted chronologically. Required params: `start`, `end`. Optional: `limit` (default 50, max 200).
- **`temporal_neighborhood`** — list fibers temporally adjacent to an anchor fiber, excluding the anchor. Required param: `fiber_id`. Optional: `window_hours` (default 24, max 8760), `limit` (default 10, max 200).

The `_causal` handler is now a pure dispatcher; per-action validation moved into dedicated methods (`_causal_trace`, `_causal_sequence`, `_causal_temporal_range`, `_causal_temporal_neighborhood`). `neuron_id` is no longer in the JSON Schema `required` list — it is conditionally required only for `trace`/`sequence` and checked by the handler.

### Tests

- `tests/unit/test_depth_gap_fixes.py::TestCausalMCPTool` — 7 new cases covering schema shape, missing/invalid ISO bounds, reversed `start`/`end`, happy-path chronology for `temporal_range`, anchor-excluded neighborhood, and window-based filtering. Existing schema assertion updated to reflect conditional `neuron_id` requirement.

### Performance — Related Information Section Compression (-48% Recall Tokens)

Baseline measurement on a real brain (my-brain.v2, 20 queries) found **85.9% of recall context tokens** were emitted by the `## Related Information` section — individual neurons that bypassed the context compiler entirely. The cross-fiber SimHash layer was already at 0% redundancy, so the real bottleneck was this uncompressed per-neuron loop.

- **`_compress_related_neurons()` helper in `engine/retrieval_context.py`.** Applies the same recipe the compiler already uses on fibers: age-tier `compress_for_recall()`, hard per-neuron content cap, SimHash + Hamming distance dedup (intra-section + cross-section against already-emitted fiber text). TIME neurons are still filtered. Disabled path preserves legacy output byte-for-byte for rollback.
- **Two new `BudgetConfig` fields (engine/token_budget.py).** `enable_related_compression: bool = True` (instant kill-switch) and `related_neuron_max_tokens: int = 150` (per-neuron content cap).
- **Wired through both entry points.** `format_context_budgeted()` threads the config down via a new internal `_budget_config` param; direct `format_context()` callers in `ReflexPipeline` (encode + familiarity paths) pick up the default config so they benefit without a signature change.
- **Real-brain result (same 20 queries):** mean total 699 → 362 tokens (-48.2%), Related section 601 → 273 (-54.6%), Related P95 1296 → 598 (-53.9%).

### Tests

- `tests/unit/test_retrieval_context_related_compression.py` — 11 tests covering age compression, content cap, TIME filter, cross-section + intra-section SimHash dedup, `max_neurons` cap, disabled-path legacy preservation, `clean_for_prompt=True` compression (MCP default), and direct `format_context()` call with default config.

### Tooling

- `scripts/measure_token_breakdown.py` — baseline tool that splits recall context into fibers / related / other sections and reports mean/P50/P95 tokens per section. Rerun after any retrieval change to catch recall-token regressions.

## [4.52.2] — 2026-04-20

### Improved — DREAM Hubs Now Consumed by Retrieval

Closes the Section 9 integration gap **DREAM hubs + graph density scaling** — hub synapses were being *written* by consolidation but never *read* back by retrieval.

- **Graph density excludes DREAM hubs by default for strategy selection.** `get_graph_density()` grows an `exclude_hubs: bool = False` parameter on both the SQL mixin (`storage/sql/mixins/calibration.py`) and the legacy SQLite backend (`storage/sqlite_calibration.py`). When True, it filters out synapses whose metadata contains `_hub=True` via `json_extract(metadata, '$._hub') IS NULL` / `metadata->>'_hub' IS NULL`. `retrieval._auto_select_strategy()` now calls with `exclude_hubs=True` so DREAM's synthesized hub links don't inflate density and trick the engine into picking PPR on graphs that are organically sparse.
- **PPR dampens hub edges during push.** `PPRActivation` multiplies the effective weight of `_hub=True` synapses by `BrainConfig.hub_edge_dampening` (default `0.5`) when building the neighbor cache. Hub edges still carry activation — they just can't hijack random walks at the expense of genuine edges. Setting the config to `1.0` disables the dampening for users who want the pre-v4.52.2 behavior.
- **New config field.** `BrainConfig.hub_edge_dampening: float = 0.5`. Documented inline.

### Docs

- `.rune/FEATURE_REGISTRY.md` Section 2c (DREAM hub extraction) and Section 9 (DREAM hubs + graph density scaling) updated to reflect the fix — the gap moves from OPEN → FIXED v4.52.2.

### Tests

- `tests/unit/test_v4_52_2_hub_aware_retrieval.py` — 8 tests: density computation with/without hub exclusion on a real SQLite brain, `_auto_select_strategy` passes `exclude_hubs=True`, PPR hub dampening (hub target gets less activation than plain target at equal base weight), dampening disabled when factor=1.0, default config value.

## [4.52.1] — 2026-04-20

### Improved — Activation Decay Integrated into Consolidation

- **`DECAY` is now a first-class consolidation strategy (Tier 0)**. Previously the Ebbinghaus decay pass only ran on the scheduled 12h cycle, so every consolidation between those cycles worked off stale activation + synapse weights. Old memories kept their full activation and crowded fresh ones out of recall. `ConsolidationEngine` now runs `DecayManager.apply_decay()` as a dedicated first tier before `PRUNE` — so PRUNE sees the actually-decayed activation and can drop items below its threshold on the same run.
- **Safe by construction.** DECAY sits in its own frozenset tier (before the PRUNE/LEARN_HABITS/DEDUP tier) so execution order is explicit, not frozenset-hash-dependent. `min_age_days` in `DecayManager` still guards against double-decaying recently-touched memories. `dry_run=True` propagates correctly — the report records stats without persisting changes. Failures from the decay pass are logged and swallowed (`logger.warning`) so a storage backend issue cannot take consolidation down with it.
- **Report surface.** `ConsolidationReport.extra["decay"]` carries `{neurons_processed, neurons_decayed, synapses_processed, synapses_decayed, duration_ms}` so callers (dashboard, MCP clients, pre-ship checks) can inspect what the decay pass did without a second round-trip.

### Docs

- `.rune/FEATURE_REGISTRY.md` Section 10 updates: freshness weight tweaks marked DONE (already at 15% / 0.15 default on `BrainConfig`) — the "stale" audit from the v4.52.0 review found these shipped separately. Decay ↔ consolidation gap moves from OPEN → FIXED. Context compiler keyword boost marked DONE (already case-normalized).

### Tests

- `tests/unit/test_v4_52_1_decay_in_consolidation.py` — 7 tests covering the DECAY enum, tier ordering (DECAY < PRUNE), dispatch wiring, report surface (DECAY alone + ALL), dry_run propagation, and non-fatal failure handling.

## [4.52.0] — 2026-04-20

### Improved — 3 Cross-Feature Wirings

Closes three integration gaps flagged in Section 9 of `.rune/FEATURE_REGISTRY.md`. Each is small in LOC but meaningful: features that already existed now actually talk to each other.

- **Dynamic abstraction → stratum MMR** (Section 2c/Section 2 integration). `_apply_mmr_diversity` now tracks `abstraction_counts` alongside `schema_counts`. Fibers anchored on a CONCEPT neuron with `_abstraction_induced=True`, or carrying `_abstract_neuron_id` from MERGE consolidation, are capped per-cluster using the same `max_per_stratum` budget. Prevents a single "super-abstract" from dominating top-K results.
- **Vietnamese keyword extraction → query_expander** (Section 2b/Section 2 integration). `expand_terms()` now accepts a `language` parameter. When set to `"vi"` (or auto-detected via Vietnamese diacritics), multi-word phrases (3+ tokens) are run through pyvi's `ViTokenizer` to extract compound tokens — so "học sinh giỏi nhất" now surfaces "học_sinh" as an expansion candidate. Graceful no-op when pyvi is not installed. Wired through `stimulus.language` in `RecallPipeline`.
- **Abstraction → priming** (Section 2e integration). `prime_from_topics` and `prime_from_habits` now apply a +25% boost (`ABSTRACTION_BOOST_MULT = 1.25`) to neurons whose metadata carries `_abstraction_induced=True`. Concept-level summaries surface before raw episodes in primed recall rounds.

### Tests

- `tests/unit/test_v4_52_wirings.py` — 9 tests pinning each wiring independently so future edits don't silently regress the integrations.

### Registry

- `.rune/FEATURE_REGISTRY.md` Section 9 updates: the three gaps above move from OPEN → FIXED. Three other gaps remain (DREAM hubs + graph density, auto-capture + FastAPI, agent ID + provenance) — queued for future versions.

## [4.51.4] — 2026-04-19

### Fixed

- **`ensure_schema` never applied on connect** (F821 runtime bug): `ensure_schema` was called in
  `SurrealDBStorage.connect()` but never imported. Schema now correctly initializes on every
  connection.
- **Mypy / ruff clean build**: Removed unused locals (`brain_id` at line 340, `conn` at 903,
  `target_prefix` at 908); typed `_max(rows: list[Any])`; added `# noqa: DTZ901` for
  intentionally naive `datetime.min` sentinel; fixed `get_schema_history` parent_raw type widening.
- **SQLite FK constraint in tests**: `typed_memory.project_id → projects.id` FK requires calling
  `add_project()` before `add_typed_memory()` when `project_id` is set. Test suite updated to
  create project rows before seeding typed memory rows.
- **Taskmaster project-locality**: Global `taskmaster` was resolving its tasks file via
  `realpath(__file__)`, hardcoding it to the L260639 project. Fixed with CWD-walking resolver —
  each project now uses its own `.taskmaster/tasks.json`.

### Improved

- **`docs/getting-started/installation.md`** rewritten: Replaces stale upstream `surreal-memory`
  content with accurate surreal-memory instructions — pipx from GitHub, Docker setup, extras
  table, dev install steps, env variable reference, and cross-link to `INSTALL_PROMPT.md`.
- **`pyproject.toml` project URLs** corrected to `acidkill/surreal-memory-surrealdb-version`.
- **`README.md`** Quick Start section: Automated Setup via Claude Code listed first; badge URLs
  corrected to fork repo.

### Tests

- **150+ new parametrized tests** across four new test files:
  - `test_get_project_memories.py`: Cross-backend parity (SQLite + InMemory; SurrealDB skipped
    without `SURREALDB_URL`). Covers project isolation, expiry filtering, empty result.
  - `test_suggest_memory_type.py`: 128 tests — corpus coverage (12 types × 10+ sentences),
    precedence collision suite (BOUNDARY > INSTRUCTION, BOUNDARY > TODO, TOOL after WORKFLOW,
    CONTEXT before FACT).
  - `test_remember_handler_all_types.py`: 19 lightweight tests — constructor round-trip for all
    15 types, classifier never emits cognitive-only types, all 12 non-cognitive types reachable.
  - `test_surrealdb_typed_memory_all_types.py`: 31 parametrized integration tests (skipped
    without `SURREALDB_URL`).

### Implications of 1.0.0

- **Stable public API**: The `NeuralStorage` ABC interface and all 163 SurrealDB method
  implementations are considered stable. Breaking changes will require a 2.0 bump.
- **SurrealDB is the default recommended backend**: SQLite remains supported for local-only use
  but lacks vector search, graph traversal, and real-time sync.
- **All features are free**: The community plugin ships with every install. No license key,
  no Pro gate, no paywalled tools.
- **`surreal-memory` is an independent package**: Upstream `surreal-memory` versioning (4.x)
  is frozen as the fork baseline. This project follows its own semantic versioning from 1.0.0.

## [4.24.0] — 2026-03-31

### Added

- **Auto-tier engine** (B5 Phase 1, Pro): Automatic WARM→HOT promotion, HOT→WARM demotion, WARM→COLD archival based on access patterns. Protection for BOUNDARY types and pinned fibers. Oscillation prevention. `smem_tier` MCP tool with status/evaluate/apply/history/config actions.
- **Decision intelligence** (B5 Phase 2): Extract structured decision components (chosen, alternatives, reasoning, confidence) from DECISION-type memories. Detect overlapping prior decisions, classify relationships (confirms/contradicts/evolves), create EVOLVES_FROM synapses, boost recall scores for domain-relevant decisions.
- **Dashboard Phosphor icons**: Migrated all 19 component files from Lucide to Phosphor Icons (`@phosphor-icons/react`). Added Playwright E2E smoke tests (8 tests).

### Fixed

- **CLI ignores `SURREAL_MEMORY_BRAIN` env var** (#123): CLI `get_storage()` now respects `SURREAL_MEMORY_BRAIN`/`SURREAL_MEMORY_BRAIN` env vars. Priority: explicit arg > env var > config file. `brain list` shows effective brain from env var.
- **Handler monolith split**: Split `tool_handlers.py` (2030 LOC) into 7 domain-specific handler modules. Fixed circular imports, removed duplicate utility functions.
- **Input firewall hardening**: Added bounds validation, type checks, and range clamping across handler modules (lifecycle, provenance, evolution, stats).

### Improved

- Auto-tier config: `cold_archive_days` invariant (must be ≥ `demote_inactive_days`), Pro gate in consolidation engine
- MemoryTier constants used consistently (no string literals in handlers)
- MCP tool count: 52 → 53

### Tests

- 50+ new tests across tier engine, decision intelligence, brain isolation, E2E smoke tests
- Fixed stale `ReflexPipeline` patch targets and MagicMock config attrs in test fixtures

## [4.23.4] — 2026-03-30

### Fixed

- **macOS SSL cert failures** (#120): Added `ssl_helper.py` with certifi-based SSL context, patched all 11 aiohttp session locations
- **`smem init --full` hang** (#121): Added `--skip-embeddings` flag and non-interactive terminal guard to prevent hang in pipes/CI
- **`find_spec` crash** (#122): Handle `ImportError` from namespace packages (e.g. `google-cloud-storage`) in `_is_module_available`

### Tests

- 11 new tests: `_is_module_available` edge cases (6), SSL helper (4), skip-embeddings (1)

## [4.23.3] — 2026-03-30

### Improved

- **Landing page**: Added "Install Free" CTA, quickstart guide for new users, post-purchase activation steps
- **ClawHub**: Fixed display name from ".Claude Plugin" to "Surreal-Memory", published v4.23.3
- **CLI docs**: Regenerated CLI reference for storage commands

## [4.23.2] — 2026-03-30

### Added

- **Pro upgrade URL**: Free users see `upgrade_url` in MCP stats, CLI status, and dashboard license API — agents and UI can guide users to purchase page
- **CLI license info**: `smem shared status` now shows license tier and upgrade link for free users

## [4.23.1] — 2026-03-30

### Fixed

- **Dashboard live reload**: License, storage status, and backend switch endpoints now reload config from disk — CLI changes (Pro activation, backend switch) reflected without server restart
- **Dashboard activation**: `/license/activate` now uses pay-hub directly (no sync config required), matches MCP + CLI behavior
- **Dashboard activation**: Adds `next_step` InfinityDB guidance hint when activating on SQLite

## [4.23.0] — 2026-03-30

### Added

- **Storage visibility**: `smem_stats` now shows `storage_backend`, `pro_installed`, `is_pro` fields
- **Storage CLI**: `smem storage status` — shows backend, Pro status, data file existence + sizes
- **Storage CLI**: `smem storage switch <sqlite|infinitydb>` — switch with Pro/data guards
- **Migration**: `smem migrate infinitydb` — SQLite → InfinityDB via export/import (Pro required)
- **Activation guidance**: Pro activation (MCP + CLI) now shows next_step hint to InfinityDB
- **Stats hint**: When Pro active but on SQLite, suggests InfinityDB upgrade path

## [4.22.2] — 2026-03-30

### Fixed

- **Pro activation**: Decoupled license activation from sync config — no longer requires hub_url + api_key
- **Pro activation**: Both MCP tool and CLI now call pay-hub directly with just the license key
- **Config**: ISO datetime sanitizer now accepts space-separated timestamps (pay-hub format)
- **Pro activation**: `activated_at` now populated with actual activation timestamp

## [4.22.1] — 2026-03-30

### Fixed

- **L4**: `with_priority()` now preserves `trust_score` and `source` fields (pre-existing bug)
- **L2**: HOT tier injection catches all exceptions, not just TypeError/AttributeError
- **M1**: Boundary auto-promote in `_edit` moved before tier assignment — eliminates dead code path
- **M2**: Tier distribution counts use `count_typed_memories()` SQL COUNT — no 1000-row display cap
- **L1**: Schema v38 migration promotes pre-v37 BOUNDARY memories from default "warm" to "hot"
- **L3**: Tier param normalized to lowercase in `smem_remember`, `smem_recall`, and `smem_edit`

## [4.22.0] — 2026-03-29

### Added

- **A6 Tiered Memory Loading** — HOT/WARM/COLD tier system for context priority and decay behavior
  - **HOT**: Always injected into context, 0.5× decay rate, activation floor at 0.5 — memories never fade below half strength
  - **WARM**: Default tier, standard semantic-match retrieval, normal decay
  - **COLD**: Excluded from auto-context, 2× decay rate — archive-grade memories accessible only via explicit recall
  - **BOUNDARY safety invariant**: `MemoryType.BOUNDARY` memories always auto-promote to HOT tier (enforced in create, edit, pin, and decay)
- **Tier parameter** on `smem_remember`, `smem_edit`, `smem_pin`, and `smem_train` tools
- **Tier filter** on `smem_recall` — filter recall results by specific tier
- **Schema v37** — `tier` column on `typed_memories` table with index
- **Dashboard**: Storage page with TierDistribution card (progress bars: red HOT, amber WARM, blue COLD)
- **`MAX_HOT_CONTEXT_MEMORIES`** constant (50) caps auto-injected HOT memories per recall

### Improved

- **Context optimizer** — HOT tier gets +0.3 score boost, COLD excluded by default (`exclude_cold=True`)
- **Lifecycle decay** — tier-aware decay with per-tier multipliers and floors, batched fiber lookups
- **Recall handler** — combined trust + tier filtering into single loop, HOT memories always injected regardless of `fresh_only`

### Tests

- 42 new unit tests across 4 phase files (`test_tiered_memory_phase1-4.py`)
- Covers: schema migration, tier constants, decay math, context optimizer, recall filter, lifecycle integration, dashboard API, tier stats

## [4.21.1] — 2026-03-28

### Fixed

- **Multilingual neuro engine** — arousal detection + prediction error reversal now support Vietnamese via pattern registries, with language-agnostic fallback for all other languages (closes #116, #119)
- **Auto-ingest noise stripping** — input firewall strips NM context headers, neuron-type bullets, and metadata wrappers before re-encoding, preventing self-referential memory pollution (closes #118)
- **OpenClaw hook migration** — migrated all legacy hooks to current API (`before_prompt_build`, `before_compaction`, `before_reset`, `gateway_start`)
- **Gemini SDK import** — updated `google.generativeai` → `google.genai` + default model to `gemini-2.0-flash` (#117)

### Added

- **`clean_for_prompt` recall mode** — new parameter on `smem_recall` strips section headers and type tags from output, reducing noise when injecting context into prompts
- **Shared `detect_language()`** — deduplicated language detection from arousal + prediction_error into `extraction/parser.py`

### Improved

- **OpenClaw plugin v1.16.0** — auto-context recall uses `clean_for_prompt`, `sanitizeAutoCapture()` strips NM noise + short acknowledgements before re-ingest

## [4.21.0] — 2026-03-26

### Added

- **Neuroscience Engine** — 10 brain-inspired improvements across 4 phases:
  - **Phase 1**: Temporal binding (TEMPORAL synapses between nearby memories) + arousal detection (emotional valence scoring via sentiment/punctuation/caps)
  - **Phase 2**: Prediction error encoding (novelty-based priority boost via SimHash) + retrieval reconsolidation (context drift detection, context anchors for shifted memories)
  - **Phase 3**: Context-dependent retrieval (encoding fingerprint stored per fiber, Jaccard similarity scoring at recall) + hippocampal replay (LTP/LTD synapse strengthening during consolidation) + cognitive chunking (greedy clustering of retrieval results by activation + synapse connectivity)
  - **Phase 4**: Schema assimilation (auto-creates SCHEMA neurons when tag clusters exceed threshold, Piaget assimilate/accommodate) + interference forgetting (SimHash-based retroactive/proactive/fan-effect detection, CONTRADICTS synapses)
- **Post-encode hooks** — schema assimilation + interference detection auto-run after every `encode()` when enabled (non-critical, swallowed on error)
- **Real activation scores** — chunking now uses per-neuron activation levels from retrieval instead of dummy values
- **Paginated tag fetch** — `_find_neurons_by_tags()` helper pages through large brains (1000/page) instead of fixed limit
- 10 new `BrainConfig` fields (all default OFF except `context_retrieval_enabled` and `chunking_enabled`)
- 107 new unit tests across all neuro engine modules

### Fixed

- **`list(int)` bug** in recall_handler chunking — `result.neurons_activated` is int, not iterable
- **`replay_enabled` gate** — `hippocampal_replay()` now checks flag directly (was only checked at dispatcher level)
- **Small brain skip** — post-encode schema hook checks `get_stats()` neuron count before querying

## [4.20.4] — 2026-03-25

### Fixed

- **`_essence_backfill` pagination bug** — used broken cursor-based pagination with `offset=` param that `get_fibers()` doesn't support. Replaced with single-batch fetch (limit=1000) + safety cap of 2000 fibers
- **`_summarize` O(N²) pair explosion** — no cap on candidate pairs or fiber count. Added: cap fibers at 1000 (highest-salience), skip tags shared by >100 fibers, cap pairs at 50K, yield every 1000 pairs
- **Unbounded `get_synapses()` in dream engine** — filtered by `RELATED_TO` type (the only type dream creates), reducing memory footprint
- **`_prune` event loop blocking** — added `asyncio.sleep(0)` yield every 500 synapses in prune loop
- **Dormant neuron selection bias** — `_dream_cycle` always picked first 20 dormant neurons instead of randomizing

### Improved

- **Yield frequency** — cross-cluster enrichment 50→20 iterations, SimHash dedup 100→50 iterations for more responsive timeout cancellation
- **Encryptor cache TTL** — `_get_encryptor()` in retrieval engine now has 5-minute TTL instead of caching forever (picks up config changes mid-session)

## [4.20.3] — 2026-03-25

### Fixed

- **Consolidation CPU hang** — consolidation could run 1+ hours at 100% CPU on large brains. CPU-bound O(N²) loops in `_dedup`, `_merge`, and cross-cluster enrichment blocked the event loop, preventing `asyncio.wait_for` timeout from ever firing
  - Added `asyncio.sleep(0)` yields in all O(N²) loops so timeouts actually work
  - Capped dedup anchors at 2000, merge candidate pairs at 50K
  - Skip overly-shared neurons (>100 fibers) in merge candidate generation
  - Cross-cluster Jaccard loop now yields every 50 iterations
- **Dashboard search overlay** — command palette (Ctrl+K) had z-index stacking issue causing partial dark overlay. Fixed by rendering via `createPortal` to `document.body` with `z-[100]`

## [4.20.2] — 2026-03-25

### Fixed

- **Consolidation timeout** — full consolidation could run for hours on brains with 5K+ neurons/20K+ synapses. Root causes: dream engine O(N²) pair generation from unbounded activated neurons, enrichment O(N²) Jaccard fiber comparison on up to 10K fibers, and no timeout on any strategy
  - Added per-strategy timeout (120s) and total timeout (600s) via `asyncio.wait_for()`
  - Capped dream activated neurons at 500, reduced max pairs from 50K to 5K, max new dream synapses capped at 200
  - Capped enrichment fiber clustering at 1000 highest-salience fibers (was unbounded up to 10K)

### Improved

- **Consolidation progress logging** — each strategy now logs start/finish with duration, making it easy to identify bottlenecks
- **Timeout reporting** — timed-out strategies are listed in the consolidation report (`report.extra["timed_out_strategies"]`)

## [4.20.1] — 2026-03-25

### Fixed

- **Consolidate prune crash** (#113) — `consolidate --strategy prune` crashed with `TypeError: can't subtract offset-naive and offset-aware datetimes`. Added `ensure_naive_utc()` helper to normalize timezone-aware reference times in `synapse.time_decay()`, `consolidation.run()`, and `compression.run()`
- **CLI packaging regression** (#114, #115) — v4.20.0 wheel published to PyPI was missing `cli/commands/` directory, breaking all `smem` CLI commands. Rebuilt wheel includes all command modules. Added import smoke tests to prevent regression

### Tests

- 7 new tests: timezone-aware decay (2), `ensure_naive_utc` helper (3), package integrity smoke (2)

## [4.20.0] — 2026-03-23

### Added

- **Ctrl+K Command Palette** — dashboard-wide search: navigate pages, search fibers by summary, search neurons by content (debounced). Pro upsell hints for semantic search and cross-brain search
- **Mindmap fiber names** — fiber list and root node now show human-readable summaries instead of UUIDs

### Fixed

- **activate.ts security** — license key moved from query param to POST body, upstream error messages no longer forwarded, strict regex validation (`nm_pro_*`/`nm_team_*`), tier whitelist validation before D1 write, removed dead code
- **Stale `type: ignore`** in `file_watcher.py` — removed unused mypy suppression

## [4.19.0] — 2026-03-22

### Added

- **Fidelity layers** — memories decay through 4 levels (FULL → SUMMARY → ESSENCE → GHOST) based on activation, importance, and time. Budget pressure shifts thresholds upward, automatically compressing aged memories to save tokens
- **Extractive essence engine** — sentence-level scoring using entity density and position bias. No LLM required, generates single-sentence distillations (max 150 chars)
- **LLM essence generator** — optional abstractive essence via configured provider with cost guard (skips LLM for priority < 3). Factory pattern with `extractive` (default) and `llm` strategies
- **Ghost recall** — faded memories render as `[~] tags | age | links | recall:fiber:{id}`. Users can restore full content via the recall key
- **Ghost visibility boost** — fibers shown as ghosts within 24h get +0.1 fidelity score, preventing repeated ghost cycling
- **Budget-aware context assembly** — `optimize_context()` now scores each fiber's fidelity, renders at appropriate level, and tracks fidelity stats (full/summary/essence/ghost counts)
- **`include_ghosts` parameter** on `smem_context` — controls ghost section visibility in context output
- **Schema v33→35** — `essence` column on fibers (v34), `last_ghost_shown_at` column (v35)
- **BrainConfig fidelity fields** — `fidelity_enabled`, `fidelity_full_threshold`, `fidelity_summary_threshold`, `fidelity_essence_threshold`, `decay_floor`, `essence_generator`
- **Consolidation essence backfill** — cursor-based pagination for existing fibers without essence

### Fixed

- **13 pre-existing mypy errors** — Anthropic SDK union type narrowing in `llm_judge.py`, tags `set`/`list` type mismatch in recall handler
- **Doctor check count** — updated assertions for `_check_config_freshness` addition (11→12 checks)

### Tests

- 73 new fidelity tests across 4 phases (essence extraction, fidelity scoring, ghost rendering, generators)

## [4.18.1] — 2026-03-21

### Added

- **`smem lifecycle` CLI** — manage memory lifecycle states from CLI: `status` (distribution), `freeze` (prevent compression), `thaw` (resume lifecycle), `recover` (rehydrate compressed memory). Mirrors MCP `smem_lifecycle` tool (#97)
- **Config freshness check** — `smem doctor` now detects missing config sections from newer versions. `smem doctor --fix` auto-adds them with defaults (#97)

### Fixed

- **Write gate scope clarification** — changelog now documents that write gate applies to MCP pipeline only; CLI `smem remember` bypasses it (explicit user intent). Disabled by default (#97)

## [4.18.0] — 2026-03-21

### Added

- **Write gate** — hard quality filter before storage with configurable thresholds (`min_length`, `min_quality_score`, `reject_generic_filler`, `max_content_length`). Applies to MCP pipeline only (auto-capture + `smem_remember` tool). CLI `smem remember` bypasses write gate (explicit user intent). Disabled by default (`enabled = false`) — opt-in via `config.toml` `[write_gate]` section
- **Agent identity capture** — MCP `clientInfo.name` auto-injected as `agent:` tag on every memory, enabling per-agent filtering in multi-agent setups
- **Consolidation lock** — atomic file-based lock (`O_CREAT|O_EXCL`) with per-brain isolation and cross-platform PID check (Windows + Unix), prevents concurrent consolidation corruption
- **Sync dedup** — content hash check on neuron import (skip duplicates), fiber anchor match with tag union merge on sync
- **Dead neuron pruning** — auto-prune neurons with `access_frequency=0` older than configurable `prune_dead_neuron_days` (default 14) during consolidation

### Improved

- **Dedup tuning** — simhash threshold 10→7 and max_candidates 10→30 for tighter duplicate detection
- **Recall quality** — configurable recency sigmoid halflife (`recency_halflife_hours`, default 168h), tag-aware scoring with additive boost for matching tags
- **BrainConfig** — 3 new fields: `recency_halflife_hours`, `tag_match_boost`, `prune_dead_neuron_days`
- **Session-end consolidation** — now includes DEDUP strategy alongside MATURE/INFER/ENRICH

### Fixed

- **CRITICAL: TOCTOU race** in consolidation lock — replaced check-then-write with atomic file creation
- **HIGH: Windows PID check** — `os.kill(pid, 0)` doesn't work on Windows; now uses `kernel32.OpenProcess`
- **HIGH: `_auto_capture` bypass** — parameter now popped from args so users cannot override auto-capture quality threshold
- **HIGH: Sync dedup abstraction** — replaced raw `_read_pool` SQL with `has_neuron_by_content_hash()` storage method

### Tests

- 95 new tests across 6 files (test_write_gate, test_dedup_improvements, test_recall_quality, test_multi_agent, test_sync_safety, test_dedup_config updates)
- Total: 4480 passed

## [4.17.0] — 2026-03-21

### Fixed

- **Per-project Knowledge Surface** — `save_surface_text()` always wrote to global `~/.surrealmemory/surfaces/` because `get_surface_path()` only returned project path when the file already existed (chicken-and-egg bug). Now uses `for_write=True` to prefer project-level `<root>/.surrealmemory/surface.nm` when a project root is detected, regardless of whether the file exists yet
- **Stale global surface warning** — logs `INFO` when both project and global surface files coexist, alerting users to the stale global copy

### Tests

- 2 new resolver tests: write-mode project path creation, read-mode global fallback

## [4.16.0] — 2026-03-21

### Improved

- **Agent instruction prompts** — audited and optimized all 10 instruction surfaces
  - Deduplicated SYSTEM_PROMPT cognitive sections (merged 2 → 1, ~50 lines saved)
  - Strengthened OpenClaw `buildToolInstructions()` from 5-line stub to full RECALL/SAVE/EPHEMERAL/COMPACT guide
  - Removed marketing copy from SKILL.md — agents see usage instructions, not feature lists
  - Fixed stale `fresh_only=true` param in `.cursorrules` and `CLAUDE.md` template
  - Added `ephemeral=true` docs to all surfaces (MCP_INSTRUCTIONS, SKILL.md, .cursorrules, CLAUDE.md, plugin.json)
  - Added `compact=true` + `token_budget` mention to all surfaces
  - Added `tags=[...]` to all SYSTEM_PROMPT examples for consistency
  - Removed hardcoded tool count from SKILL.md

## [4.15.0] — 2026-03-21

### Added

- **Ephemeral memories** (#91) — session-scoped scratch notes that auto-expire
  - `smem_remember(content="temp", ephemeral=true)` — stores with 24h TTL, excluded from consolidation and cloud sync
  - `smem_recall(query="temp", permanent_only=true)` — filter out ephemeral memories from results
  - `smem_remember_batch` supports `ephemeral` per item
  - Auto-cleanup of expired ephemeral neurons at session end (`smem_auto(action="process")`)
  - Schema migration v32→v33: `ephemeral` column + index on neurons table

### Tests

- 14 new tests for ephemeral memories (`test_ephemeral.py`)

## [4.14.0] — 2026-03-21

### Fixed

- **Vietnamese auto-capture quality** (#94) — dramatically reduce low-quality Vietnamese memories
  - Quality gate: reject captures where >60% of words are Vietnamese stop words
  - TODO patterns: require compound forms (`cần phải`, `nhớ là`) — bare `cần`/`phải`/`nên` no longer match
  - Preference patterns: require explicit subject (`tôi`/`mình`/`em`/`anh`) + minimum content length
  - Correction patterns: require minimum 10-char capture content
  - Confidence penalty increased (0.7 → 0.55) for all Vietnamese regex captures
  - Minimum capture length raised (15 → 25 chars) for Vietnamese patterns
  - One-time pyvi missing warning when Vietnamese text detected in auto-capture

### Tests

- 22 new tests for Vietnamese capture quality (`test_vietnamese_capture.py`)
- Updated existing Vietnamese preference test for tighter pattern

## [4.13.0] — 2026-03-20

### Added

- **Memory Lifecycle Engine** — Heat-based compression resistance inspired by TEMM1E
  - Heat score: weighted combination of access recency, frequency, priority (exponential decay)
  - Lifecycle states: ACTIVE → WARM → COOL → COMPRESSED → ARCHIVED
  - Hot memories resist compression by 1 tier; frozen memories never compress
  - Neuron snapshots: recoverable content even after destructive Tier 3-4 compression
  - Access tracking: batch update `last_accessed_at` on every recall
  - `smem_lifecycle` tool: status/recover/freeze/thaw actions
  - Schema v32: `lifecycle_state`, `frozen`, `last_accessed_at` columns + `neuron_snapshots` table
- **Adaptive Instructions** — Self-improving procedural memory
  - Auto-populate instruction metadata: version, execution_count, success_rate, trigger_patterns
  - `smem_refine` tool: version instructions with refinement history, add failure modes/triggers
  - `smem_report_outcome` tool: track execution success/failure, recompute success_rate
  - Recall boost: proven instructions (high success_rate) rank higher via activation bonus
  - Trigger pattern matching: instruction keywords boost relevance when query overlaps
- **Budget-Aware Retrieval** — Token cost management for context-efficient recall
  - Token cost estimator: estimate fiber tokens from content length
  - Greedy value-per-token allocation within context budget
  - `smem_budget` tool: estimate/analyze/optimize token usage
  - `recall_token_budget` param on `smem_recall` for opt-in budget-aware formatting
- 4 new MCP tools (47→50): `smem_lifecycle`, `smem_refine`, `smem_report_outcome`, `smem_budget`
- 133 new tests across 3 test files

## [Unreleased]

### Added

- **Input firewall (Gate 1)** — Security gate blocking garbage/adversarial content from auto-capture pipeline
  - Blocks: oversized content (>10KB), control sequences (`<ctrl*>`, fake role tags), JSON metadata injection, base64/binary blocks, repetitive content, low-entropy data
  - `FirewallResult` dataclass with `blocked`, `reason`, `sanitized` fields
  - Integrated into all 3 auto-capture entry points: stop hook, precompact hook, post-tool passive capture
  - 30 new tests (`test_input_firewall.py`)
- **Stop hook role filtering** — JSONL transcript entries classified by role; tool results skipped, assistant messages filtered by memory markers
- **Embedding semantic dedup** — Removes near-duplicate auto-captures using local embedding cosine similarity (sentence_transformer/ollama only)
- **Compact response mode** — Reduce MCP tool response tokens by 60-80%
  - `compact=true` param on all 46 MCP tools to strip metadata hints and truncate lists
  - `token_budget=N` param for progressive response size enforcement
  - Auto-compact: responses with >20 list items are compacted automatically
  - Content preview: list items show truncated content with `_content_truncated` flag
  - Count-replace: `fibers_matched`, `conflicts`, `expiry_warnings` → count only
  - Long string truncation: `markdown` field capped at 500 chars
  - `ResponseConfig` in config.toml: `compact_mode`, `max_list_items`, `strip_hints`, `content_preview_length`, `auto_compact_threshold`
  - 47 new tests (`test_response_compactor.py`)

### Fixed

- **Memory poisoning prevention** — Garbage content (chat control sequences, fake role injection, 270KB payloads) no longer enters brain through hooks (#94)
- **PreCompact emergency threshold** — Raised from 0.5 to 0.65 to reduce false positive captures
- **fiber.metadata type sync** — `smem_edit` now syncs type changes into `fiber.metadata` (cherry-picked from PR #85)
- **Compression size guard** — Skip compression when summary is not smaller than original (#92)

## [4.11.0] - 2026-03-17

### Added

- **Diminishing returns gate (v4.0 Phase 5)** — Stop spreading activation early when new hops add insufficient signal
  - `ActivationTrace` dataclass: per-hop tracking of new neurons and activation gain
  - `should_stop_spreading()`: absolute (< min neurons) + relative (gain ratio < threshold) criteria
  - Wired into all 3 activation engines: BFS, PPR, Reflex
  - 4 new `BrainConfig` fields: `diminishing_returns_enabled/threshold/min_neurons/grace_hops`
  - 25 new tests (`test_diminishing_returns.py`)

### Improved

- **Roadmap cleanup** — Removed 45 completed/obsolete plan files, consolidated remaining plans
  - File watcher plan added (3 phases, Issue #66)
  - Brain Quality Track C1+C2 merged
  - v4.0 master plan: all 5 phases complete

### Tests

- 4140 passed, 92 skipped, 1 xfailed

## [4.10.0] - 2026-03-16

### Added

- **Onboarding overhaul (Issue #82)** — Reduce 26 manual setup steps to 1 command
  - `smem init --full`: auto-detect embeddings, enable dedup, generate maintenance script, print guide URL
  - `smem doctor` enhanced: 11 checks (was 8), `--fix` flag for auto-remediation (hooks, dedup, embedding)
  - Interactive quickstart guide page (MkDocs + animated terminal demos, scroll reveals, feature cards)
  - Dashboard `GuideCard` for new users (<50 neurons) — dismissible, persisted via localStorage
  - Help button (?) in dashboard TopBar linking to quickstart guide
  - CLI banners link to guide URL after init and doctor
  - 35 new tests (test_full_setup + test_doctor_enhanced)

### Fixed

- **Windows npm install**: OpenClaw plugin postinstall uses cross-platform Node.js instead of Unix shell syntax

## [4.9.0] - 2026-03-16

### Added

- **Knowledge Surface (.nm format)** — Two-tier memory architecture: Tier 1 = `.nm` flat file (~1000 tokens, loaded every session), Tier 2 = `brain.db` SQLite graph (queried on-demand)
  - `.nm` format with 5 sections: GRAPH (causal edges), CLUSTERS (topic groups), SIGNALS (urgent/watching/uncertain), DEPTH MAP (self-routing hints), META (brain stats)
  - `SurfaceGenerator` — algorithmic extraction from brain.db using composite scoring (activation + recency + connections + priority)
  - Depth-aware recall routing: SUFFICIENT entities answered from surface (0 latency), NEEDS_DEEP triggers depth=2 recall
  - Auto-injected into MCP `instructions` on session init for immediate agent context
  - `smem_surface` MCP tool — generate (rebuild from brain.db) and show (inspect current surface)
  - Auto-regeneration on `smem_auto(action="process")` session-end
  - Atomic file writes (tmp + rename), project-level and global surface resolution
  - Surface reload on brain switch, cached by brain name
  - 73 new tests across 4 test files

### Fixed

- **CI fixes**: doc_trainer mock using real `BrainConfig` instead of `MagicMock` (lazy entity promotion attrs), auto_tags tests accept bigrams from keyword extractor
- **Docs freshness**: regenerated CLI reference (new PostgreSQL migrate options)

## [4.8.0] - 2026-03-16

### Added

- **B7: Lazy Entity Promotion** — Entities need 2+ mentions before becoming neurons; `entity_refs` table (schema v29), retroactive synapses on promotion, high-confidence/user-tagged exceptions
- **A4: Auto-Importance Scoring** — Heuristic priority when user doesn't set explicit priority; type bonus, causal/comparative language signals, entity richness
- **A4: Reflection Engine** — Accumulates importance from saved memories, detects patterns (recurring entities, temporal sequences, contradictions) at threshold
- **PostgreSQL Migration** — `smem migrate postgres` CLI command with full connection params (#80)
- **B1-B6, B8: Brain Quality Track B** — Auto-consolidation, Hebbian retrieval, cross-memory linking, IDF keywords, fiber scoring, contextual compression, adaptive decay
- **A1: Smart Instructions** — Decision framework injected into MCP `instructions` to guide proactive memory saving
- **Schema v29** — `entity_refs` table for lazy entity promotion + `keyword_document_frequency` for IDF scoring
- **73 new tests**: lazy entity (11), importance (16), reflection (12), compression (12), adaptive decay (11), postgres migration (5), cross-memory link (9), IDF (7), fiber scoring (8)

### Improved

- All quality improvements are purely algorithmic — zero LLM calls added
- Pipeline steps use `getattr` for backward compat with SimpleNamespace contexts
- Entity ref operations gracefully degrade when table doesn't exist

## [4.7.0] - 2026-03-16

### Added

- **PostgreSQL + pgvector backend** — Full async storage backend via `asyncpg` with vector similarity search. Supports neurons, synapses, fibers, brains, typed queries. Docker Compose included. Contributed by @zsecducna (#56)
- **Surreal-Memory vs Mem0 benchmark** — Head-to-head comparison: 121x faster writes, equal accuracy, 0 API calls vs 70. Script at `scripts/benchmark_mem0_vs_nm.py`
- **Chatbot v2** — Upgraded HF Spaces chatbot with conversation memory, cognitive reasoning for low-confidence answers, source citations, and retrieval stats panel

### Fixed

- `ReinforcementManager.reinforce()` test — updated assertion to match batch API (`update_neuron_states_batch`)
- `check_distribution.py` — Fixed ClawHub JSON parser, Windows shell compat, independent version channels

## [4.6.0] - 2026-03-14

### Added

- **`smem setup rules`** — IDE rules file generator for multi-agent adoption. Generates `.cursorrules`, `.windsurfrules`, `.clinerules`, `GEMINI.md`, and `AGENTS.md` with NM usage instructions. Supports `--all`, `--ide <name>`, `--force`, and interactive selection
- **17 new tests** for IDE rules generator

## [4.5.0] - 2026-03-14

### Added

- **Context merger (Phase A)** — `smem_remember` accepts optional `context` dict (e.g. `{reason, alternatives, cause, fix, steps}`) that gets merged into content server-side using type-specific templates. Works with any agent — no need to craft perfect prose
- **Quality scorer (Phase B)** — Every `smem_remember` response now includes `quality` ("low"/"medium"/"high"), `score` (0-10), and `hints` (actionable improvement suggestions). Soft gate: always stores, never rejects
- **36 new tests** for quality scorer (20) and context merger (16)

### Fixed

- **Tool memory config default** — test assertion updated to match `enabled=True` default

## [4.4.1] - 2026-03-14

### Improved

- **Embedding config-status 3-state detection** — Quick Actions card now distinguishes "configured", "installed but disabled", and "not installed" for embedding provider, with actionable enable/disable commands

## [4.4.0] - 2026-03-14

### Added

- **Dashboard Quick Actions card** — Overview page now shows configuration status for 6 features (tool memory, cloud sync, embedding, consolidation, review queue, orphan rate) with actionable shortcut commands and copy buttons
- **`/api/dashboard/config-status` endpoint** — returns per-feature config status with status badges and commands
- **Source-Aware Brain plan** — 4-phase architecture plan for smart index with exact citations from source documents

### Fixed

- **Plugin skills path (#71)** — `skills` field in `plugin.json` changed from `"./SKILL.md"` (file) to `"./skills"` (directory) to match Claude Code's expected format. Fixes 2 load errors on plugin install
- **Tool stats empty** — `tool_memory.enabled` defaulted to `false`, causing dashboard Tool Stats page to show no data. Now defaults to `true` — tool usage tracking works out of the box
- **E2E health test** — fixed assertion mismatch (`"healthy"` vs `"ok"`)

### Added

- **Source-Aware Brain plan** — 4-phase architecture plan for smart index with exact citations from source documents (source locators, `smem_cite` tool, source refresh, cloud resolvers)

## [4.3.1] - 2026-03-14

### Fixed

- **Plugin manifest validation (#70)** — removed invalid `features`, `instructions`, `agents` keys from `plugin.json` that broke Claude Code plugin install
- **Doc trainer orphan neurons** — heading-less chunks now get synthetic heading from filename; added per-file tags for cross-cluster ENRICH linking; increased heading dedup limit 20→100 for common headings like "Overview"
- **Chatbot brain loading** — use `find_brain_by_name("surrealmemory-docs")` instead of non-existent `list_brains()` method
- **HF deploy script username** — fixed `nhadaututtheky` typo (double t)

### Added

- **`/health` + `/ready` endpoints** — `smem serve` now exposes health check (brain name, uptime, schema version) and readiness probe (503 when uninitialized) for production monitoring
- **Cloud sync privacy docs** — privacy model table, encryption details, CF free tier limits in `docs/guides/cloud-sync.md`

### Improved

- **Self-hosted cloud sync** — switched default from shared hub to self-hosted model. Users deploy their own CF Worker + D1 database. Data stays on user's own Cloudflare account
- **Sync setup instructions** — updated README, FAQ, dashboard SyncPage, and MCP setup flow to guide self-hosted deployment first

### Tests

- 14 new health endpoint tests
- Total: 3748 passing

## [4.3.0] - 2026-03-13

### Added

- **`smem_tool_stats` MCP tool** — exposes tool usage analytics (summary + daily breakdown) via MCP (#63)
- **`/api/dashboard/tool-stats` REST endpoint** — tool usage analytics for dashboard integration
- **Dashboard: Tool Stats page** — top tools bar chart, usage-over-time line chart, detailed table with success rates and durations (#63)
- **Background consolidation daemon** — `smem serve` now runs periodic consolidation using existing `maintenance.scheduled_consolidation_*` config (#65)
- **HuggingFace Spaces deployment** — chatbot ready for HF Spaces with proper metadata, async Gradio handlers, deploy script, and docs guide (#60)
- **Cascading retrieval with fiber summary tier** — FTS5 search on fiber summaries as step 2.8 before neuron pipeline, sufficiency gate for early termination, schema v27 (#61, #62)

### Improved

- **Docs messaging** — restructured README and mcp-server.md with "3 tools you need, 41 the agent handles" hierarchy (#59)

### Fixed

- **`smem doctor` schema version check** — was using `PRAGMA user_version` (always 0) instead of `schema_version` table; now correctly reports v26
- **`smem brain health` crash in shared mode** — hardcoded `limit=10000` exceeded server max (1000), causing 422 errors (#67)
- **`smem info` crash in shared mode** — same limit issue for typed memories query
- **`smem consolidate` FK crash** — summarize strategy referenced anchor neurons pruned by earlier tier; now validates neuron existence before creating summary fibers (#68)

## [4.1.1] - 2026-03-12

### Fixed

- **`smem doctor` crash** — fixed `No module named 'surreal_memory.storage.sqlite'` caused by stale import after storage restructuring (now imports from `sqlite_schema`)
- **`smem_pin action=list`** — new `list` action to query pinned fibers (#57)

### Improved

- **Stale references audit** — updated tool counts (39→44), schema version (v22→v26), test counts across README, ROADMAP, plugin.json, mcp-server.md
- **FAQ** — added "Why is my consolidation 0%?" entry
- **Regenerated docs** — MCP tools + CLI reference refreshed for v4.1.x

## [4.1.0] - 2026-03-12

### Added

- **Auto-generated MCP Tool Reference** — `scripts/gen_mcp_docs.py` introspects all 44 MCP tool schemas and generates `docs/api/mcp-tools.md` with parameter tables, categories, and tier badges
- **Auto-generated CLI Reference** — `scripts/gen_cli_docs.py` introspects all 66 CLI commands (Typer/Click) and generates `docs/getting-started/cli-reference.md`
- **Documentation Chatbot** — Gradio UI (`chatbot/app.py`) powered by Surreal-Memory's ReflexPipeline, answers docs questions without an LLM using spreading activation retrieval
- **Docs Brain Trainer** — `chatbot/train_docs_brain.py` trains a brain from project docs (40 files → 1045 chunks → 9175 neurons)
- **CI Docs Freshness Check** — new `docs` job in GitHub Actions runs `--check` mode on both generators, fails CI when auto-generated docs are stale

### Fixed

- **Brain lookup fallback** — `get_brain(name)` now falls back to `find_brain_by_name()` when id-based lookup fails, preventing duplicate "brain.v2" creation for users upgrading from older versions with UUID-based brain ids

### Improved

- **Docs navigation** — added orphan pages (Companion Setup, Lessons Learned) to mkdocs.yml nav
- **Cross-links** — CLI Guide, CLI Reference, and MCP Tools Reference now link to each other via admonition boxes
- **CLI Guide renamed** — title changed from "CLI Reference" to "CLI Guide" to avoid confusion with auto-generated reference

## [4.0.1] - 2026-03-12

### Security

- **Fix path traversal** in `index_handler.py` — adapter connection paths now validated with `is_relative_to()` against allowed directories (cwd, home, temp)
- **Fix path traversal** in `pre_compact.py` hook — stdin transcript path now validated against `~/.claude` directory
- **Update `cryptography>=46.0.5`** — fix CVE-2026-26007
- **Add `python-multipart>=0.0.22`** floor constraint — fix CVE-2026-24486
- **Remove internal info from error messages** — 9 locations no longer leak memory IDs, hypothesis IDs, or filesystem paths to clients
- **CORS hardening** — replace `localhost:*` wildcard with explicit port list (3000, 3001, 5173, 5174, 8000, 8080, 8888)

### Fixed

- Fix 8 silent `except Exception: pass` blocks — all now log at DEBUG level with `exc_info=True`
- Fix 14 redundant exception tuples (`except (AttributeError, Exception)` → `except Exception`)
- Remove unused `python-dateutil` from core dependencies

## [4.0.0] - 2026-03-12

### Added

- **Semantic Drift Detection** — Find tag synonyms/aliases via Jaccard similarity on co-occurrence data
- **Tag Co-Occurrence Matrix** — Automatically recorded on every memory encode, tracks which tags appear together
- **Union-Find Clustering** — Groups related tags with confidence thresholds: merge (>0.7), alias (>0.4), review (>0.3)
- **Temporal Drift Detection** — Compares early vs recent session topics to detect terminology shifts
- **`smem_drift` MCP Tool** — detect/list/merge/alias/dismiss actions for managing drift clusters
- **`detect_drift` Consolidation Strategy** — Runs drift analysis during periodic consolidation
- **Schema v26** — New `tag_cooccurrence` and `drift_clusters` tables

### Improved

- **Brain Intelligence Complete** — v4.0 milestone: session intelligence, adaptive depth, predictive priming, and semantic drift detection work together as feedback loops
- Consolidation engine now includes drift detection in the final tier alongside semantic_link

### Tests

- 51 new drift detection tests (Jaccard, clustering, storage, MCP handler, Union-Find)
- Total: 3810 passing

## [3.5.0] - 2026-03-12

### Added

- **Predictive Priming** — Brain anticipates next query from session context with 4-source priming engine
- **Activation Cache** — Recent query results carry forward as soft activation with exponential decay (`0.7^n` per query)
- **Topic Pre-Warming** — Session topics with EMA > 0.5 pre-warm related neurons before query parsing (truly predictive)
- **Habit-Based Priming** — Query pattern co-occurrence (CONCEPT neurons + BEFORE synapses) predicts next topic, max 3 predicted topics
- **Co-Activation Priming** — Hebbian binding data (strength >= 0.5, count >= 3) boosts associated neurons
- **Priming Metrics** — Hit rate tracking with auto-adjusted aggressiveness (0.5x-1.5x) based on priming effectiveness
- **Session priming fields** — `priming_hit_rate`, `priming_total` exposed in session summaries and result metadata

### Tests

- 57 new tests covering all priming sources, metrics, orchestration, merging, backward compat
- Total: 3759 passing

## [3.4.0] - 2026-03-12

### Added

- **Session-aware depth selection** — Primed topics go shallower (already in context), new topics go deeper (need exploration). Uses session EMA topic weights
- **Calibration-driven gate tuning** — High-accuracy gates get confidence boost (+10%), low-accuracy gates get dampened (-30%), very low avg_confidence triggers downgrade to insufficient
- **Agent feedback signal** — `agent_used_result` parameter: remember-after-recall = strong positive, unused recall = raised bar for success
- **Dynamic RRF weights** — Per-brain retriever weights evolve from outcome history via `retriever_calibration` table and EMA
- **Auto activation strategy** — `activation_strategy="auto"` selects classic/PPR/hybrid based on graph density (synapses/neuron ratio)
- **Schema v25** — `retriever_calibration` table + `graph_density` column on brains

### Tests

- 30 new tests covering all 5 features + backward compatibility
- Total: 3702 passing

## [3.3.0] - 2026-03-12

### Added

- **Cloud Sync Hub** — Cloudflare Workers + D1 sync hub with API key auth, brain ownership, device management. Live at `surreal-memory-sync-hub.vietnam11399.workers.dev`
- **API key auth** — `nmk_` prefixed keys, SHA-256 hashed storage, Bearer token transport, key masking in all outputs
- **`smem_sync_config(action='setup')`** — Guided onboarding flow for cloud sync setup
- **URL versioning** — Cloud hub uses `/v1/` prefix, localhost preserves backward-compatible paths
- **HTTP error mapping** — User-friendly messages for 401/403/413/429 status codes
- **Cloud profile in `smem_sync_status`** — Shows tier, email, usage when connected to cloud hub
- **HTTPS enforcement** — Refuses non-HTTPS for cloud hub URLs (localhost exempt)

### Tests

- 22 new tests: SyncConfig api_key, key masking, URL versioning, HTTP error handling
- Sync hub: 10 Vitest tests (health, auth, validation, type shapes)
- Total: 3672 passing

## [3.2.0] - 2026-03-11

### Added

- **Session Intelligence (v4.0 Phase 1)** — In-memory session state tracking across MCP calls with topic EMA scoring, LRU eviction (max 10 sessions), 2h auto-expiry, and SQLite persistence via `session_summaries` table (schema v24)
- **Dashboard assets in wheel** — Bundled `server/static/dist/` via hatch artifacts config, fixing blank dashboard on pip install (#54)

### Fixed

- **Config singleton mutation** — `wizard.py` and `embedding_setup.py` now use immutable `replace()` pattern instead of mutating the cached config singleton (H1/H2)
- **Structure detector false positives** — Added 4096-char size guard and CSV all-text column rejection heuristic (H4/H5)
- **Source registry validation** — `_row_to_source()` handles invalid SourceType/SourceStatus gracefully, `update_source()` validates before SQL write (H2/H3)
- **Source handler error handling** — `_require_brain_id()` and `Source.create()` wrapped in try/except ValueError (H1/M1)

### Tests

- 40 new tests for session intelligence (QueryRecord, SessionState EMA, SessionManager LRU, SQLite persistence)
- Total: 3650 passing

## [3.1.0] - 2026-03-11

### Added

- **Source-Aware Memory (v3.0 Pillar 4)** — Brain that knows its sources. 6-phase plan fully shipped.
- **`smem_show` tool** — Retrieve exact verbatim content of a memory by fiber ID
- **Exact recall mode** — `mode="exact"` in `smem_recall` returns verbatim content without summarization
- **Source Registry** — Schema v23 with `sources` table, `SOURCE_OF` synapse type, `smem_source` tool for registering and querying memory provenance
- **Structured encoding** — Schema-aware encoder detects tabular data (CSV, markdown tables, JSON arrays) and preserves structure through the pipeline
- **Citation engine** — `citation.py` generates citation metadata with audit synapses linking memories to their sources
- **`smem init --wizard`** — Interactive first-run wizard: brain name → embedding provider → MCP config → test memory
- **`smem doctor`** — System health diagnostics with 8 checks (Python, config, brain, deps, embeddings, schema, MCP, CLI tools)
- **`smem setup embeddings`** — Interactive embedding provider setup with installation status and API key detection
- **Change log tracking** — `sqlite_change_log.py` records schema and data mutations for audit trail

### Fixed

- **SharedStorage brain_id parity** — Abstract `brain_id` property on base class, all backends implement consistently (#53)
- **Hub auto-creates brain** — First sync or device registration no longer fails on missing brain
- **Error message leaks** — Batch remember no longer exposes `str(e)` exception details to clients

### Improved

- **DX Sprint** — Actionable error messages across CLI and MCP, embedding setup guides new users through provider selection
- **VS Code extension v0.5.0** — 6 lifecycle and config bug fixes

### Tests

- 200+ new tests across all v3.0 phases (show handler, source registry, structured encoding, citation, audit synapses, DX wizard/doctor/embedding)
- Total: 3515 passing

## [2.29.0] - 2026-03-10

### Added

- **Reciprocal Rank Fusion (RRF)** — Multi-retriever score blending for anchor ranking. Combines BM25/FTS5, embedding similarity, and graph expansion ranks into unified scores using the RRF formula (`score = Σ weight_i / (k + rank_i)`). Anchors now start with differentiated activation levels instead of uniform 1.0. Config: `rrf_k` (default 60).
- **Graph-based query expansion** — 1-hop neighbor traversal from entity/concept anchors adds soft expansion anchors. Exploits knowledge graph structure for associative priming (e.g., "auth" → OAuth2 → JWT, session). Config: `graph_expansion_enabled`, `graph_expansion_max`, `graph_expansion_min_weight`.
- **Personalized PageRank (PPR) activation** — Optional replacement for classic BFS spreading activation. Distributes activation proportional to edge weights / out-degree with damping (teleport back to seed set), naturally handling hub dampening. Opt-in via `activation_strategy = "ppr"` or `"hybrid"` (PPR + reflex). Config: `ppr_damping`, `ppr_iterations`, `ppr_epsilon`.
- **Tag filtering in Query API and MCP** — `POST /query` accepts `tags: list[str]` (AND filter, max 20). `smem_recall` accepts `tags: list[str]` to scope results to specific tag sets. Filters across `tags`, `auto_tags`, and `agent_tags` columns. Backward compatible — `tags=None` returns all results as before.

### Fixed

- **Marketplace plugin install** — Removed unrecognized `features` key from `marketplace.json` that caused Claude Code `/plugin marketplace add` to fail with schema validation error (#49).

## [2.28.0] - 2026-03-08

### Added

- **`smem_remember_batch`** — Bulk remember up to 20 memories in a single call. Partial success supported (individual failures don't block others). Added to `standard` tool tier.
- **Trust score** — First-class `trust_score` (0.0–1.0) and `source` fields on TypedMemory. Source-specific ceiling caps: `user_input=0.9`, `ai_inference=0.7`, `auto_capture=0.5`, `verified=1.0`. Schema v22 migration adds columns + index.
- **`min_trust` filter** — `smem_recall` accepts optional `min_trust` parameter to filter out low-confidence memories.
- **Auto-promote context→fact** — Frequently-recalled context memories (frequency ≥ 5) are automatically promoted to `fact` during consolidation. Audit trail in metadata (`auto_promoted`, `promoted_from`, `promoted_at`).
- **SEMANTIC alternative path** — Memories can reach SEMANTIC stage via intensive reinforcement (`rehearsal_count ≥ 15` + `5 distinct 2h-windows`) as alternative to the 3-distinct-days spacing requirement. Enables agents with burst usage patterns.

### Fixed

- **FK constraint race condition** — `update_fiber()` no longer raises ValueError when a fiber is deleted between deferred-write enqueue and flush. Gracefully skips with debug log.

### Changed

- **MCP startup 3x faster** — Lazy-import `cli.setup` (defer until first-time init actually needed) and `sync.client`/`sync.sync_engine` (defer aiohttp until first sync call). Cold start: 611ms → 197ms.

## [2.27.3] - 2026-03-08

### Fixed

- **OpenAI-compatible client HTTP 400** — Tool schemas now include `parameters` alias alongside `inputSchema`, fixing "schema must be type object, got type None" errors when MCP tools are forwarded through OpenAI-compatible bridges (Cursor, LiteLLM, etc.)

### Added

- **Cognitive Reasoning Guide** — Full workflow documentation: hypothesize, evidence, predict, verify loop with Bayesian confidence formula, end-to-end examples (`docs/guides/cognitive-reasoning.md`)
- **Schema v21 Migration Guide** — New tables, auto-migration behavior, rollback instructions (`docs/guides/schema-v21-migration.md`)
- **Learning Habits Guide** — 3-stage pipeline, thresholds, confidence calculation, suggestion engine (`docs/guides/learning-habits.md`)
- **Pre-ship smoke tests** — Auto-type classifier (13 cases) and cognitive engine integration test in `scripts/pre_ship.py`

## [2.27.2] - 2026-03-07

### Fixed

- **OpenClaw plugin: lazy auto-connect** — Fixed tools returning "Surreal-Memory service not running" when OpenClaw calls `register()` multiple times across subsystems (gateway, agent worker, CLI). Agent worker instance now lazily connects on first tool call via `ensureConnected()` with connection mutex to prevent race conditions (#38)

## [2.27.1] - 2026-03-06

### Added

- **`smem_edit`** — Edit memory type, content, or priority by fiber ID. Preserves all neural connections. Supports typed_memory path (type/priority) and anchor neuron path (content update)
- **`smem_forget`** — Soft delete (sets expires_at for natural decay) or hard delete (permanent removal with cascade to fiber + typed_memory). Also handles orphan neuron deletion
- **Enhanced MCP instructions** — Richer behavioral directives: brain growth tips, rich language patterns (causal/temporal/relational/decisional/comparative), memory correction guidance, all 38 tools listed
- **Enhanced plugin instructions** — Comprehensive agent guidance in `.claude-plugin/plugin.json` for proactive memory usage

### Fixed

- **FK constraint errors** — `INSERT OR REPLACE INTO neuron_states` and `save_maturation` now catch `sqlite3.IntegrityError` when neuron was deleted by consolidation prune (previously crashed with FOREIGN KEY constraint failed)
- **Auto-type classifier bias** — Reordered `suggest_memory_type()`: DECISION now checked before INSIGHT to prevent "because" from hijacking decisions. Removed overly broad "because"/"pattern" from INSIGHT keywords. Added "rejected"/"went with" to DECISION, "prefers"/"preferred" to PREFERENCE. Tightened TODO keywords and added guard against descriptive "should"
- **DECISION_PATTERNS greediness** — Removed overly broad patterns (`"we're going to"`, `"let's use"`, `"going to"`) from `auto_capture.py` that caused false decision captures
- **Synapse FK error message** — Distinguished FOREIGN KEY violations from UNIQUE violations in `add_synapse()` for clearer error messages

- **Cognitive Reasoning Layer** — 8 new MCP tools for hypothesis-driven reasoning (38 tools total)
  - `smem_hypothesize` — Create and manage hypotheses with Bayesian confidence tracking and auto-resolution
  - `smem_evidence` — Submit evidence for/against hypotheses, auto-updates confidence via sigmoid-dampened shift
  - `smem_predict` — Make falsifiable predictions with deadlines, linked to hypotheses via PREDICTED synapse
  - `smem_verify` — Verify predictions as correct/wrong, propagates result to linked hypothesis
  - `smem_cognitive` — Hot index: ranked summary of active hypotheses + pending predictions with calibration score
  - `smem_gaps` — Knowledge gap metacognition: detect, track, prioritize, and resolve what the brain doesn't know
  - `smem_schema` — Schema evolution: evolve hypotheses into new versions via SUPERSEDES synapse chain
  - `smem_explain` — (moved to cognitive) Trace shortest path between concepts with evidence
- **Schema v21** — Three new tables: `cognitive_state` (hypothesis/prediction tracking), `hot_index` (ranked cognitive summary), `knowledge_gaps` (metacognition)
- **Pure cognitive engine** (`engine/cognitive.py`) — Stateless functions: `update_confidence`, `detect_auto_resolution`, `compute_calibration`, `score_hypothesis`, `score_prediction`, `gap_priority`
- **Bayesian confidence model** — Sigmoid-dampened shift with surprise factor and diminishing returns from total evidence
- **Auto-resolution** — Hypotheses with confidence ≥0.9 + 3 supporting evidence auto-confirm; ≤0.1 + 3 against auto-refute
- **Prediction calibration** — Tracks correct/wrong ratio across all resolved predictions
- **Schema version chain** — `parent_schema_id` column + `get_schema_history()` walks the SUPERSEDES chain with cycle guard
- **Knowledge gap detection sources** — `contradiction`, `low_confidence_hypothesis`, `user_flagged`, `recall_miss`, `stale_schema`

## [2.26.1] - 2026-03-05

### Added

- **Dashboard: actionable health penalties** — Top penalties section shows ranked cards with score bar, penalty points lost, estimated gain if fixed, and exact action to improve each component
- **API: `top_penalties` field** in `/api/dashboard/health` response — exposes diagnostics engine penalty analysis to frontend
- **i18n: penalty translations** — English and Vietnamese keys for top penalties section

## [2.26.0] - 2026-03-05

### Added

- **Brain Health Guide** (`docs/guides/brain-health.md`) — comprehensive guide explaining all 7 health metrics, thresholds, improvement roadmap (F through A), common issues, maintenance schedule
- **Connection Tracing docs** (`smem_explain`) — added to README, MCP prompt, brain health guide. Previously undocumented feature that traces shortest path between concepts
- **Embedding auto-detection** (`provider = "auto"`) — automatically detects best available embedding provider: Ollama → sentence-transformers → Gemini → OpenAI. Lowers barrier for cross-language recall
- **Consolidation post-run hints** — warns about orphan neurons (>20%) and missing consolidation after running `smem consolidate`
- **Pre-ship verification script** (`scripts/pre_ship.py`) — automated quality gate: version consistency, ruff, mypy, import smoke test, fast tests, plugin checks
- **MCP instructions update** — health interpretation, priority scale, tagging strategy, maintenance schedule added to system prompt

### Changed

- README: added smem_explain to tools table, brain health section, connection tracing section, embedding auto-detect
- OpenClaw npm package renamed to `surrealmemory` (published on npm)

## [2.25.1] - 2026-03-05

### Fixed

- **`smem flush` stdin blocking** — Process hangs forever when spawned as subprocess without piped input; `sys.stdin.read()` blocks because no EOF is sent. Added 5s timeout via `ThreadPoolExecutor` (fixes #27)
- **Consolidation prune** — Protects fiber members from orphan pruning + invariant tests
- **Orphan rate** — Counts fiber membership correctly, isolated E2E tests from production DB
- **Dashboard dist** — Bundled for `pip install` compatibility

### Changed

- Published v2.25.0 release (was stuck in draft)

## [OpenClaw Plugin 1.5.0] - 2026-03-05

### Fixed

- **Plugin ID mismatch warning** — Renamed package from `@surrealmemory/openclaw-plugin` to `surrealmemory` to match manifest `id`. OpenClaw's `deriveIdHint()` extracts the unscoped package name as `idHint`, which previously produced `openclaw-plugin` ≠ `surrealmemory`
- **Tool schema provider compatibility** — Replaced `integer` with `number` (Gemini rejects `integer`), added `additionalProperties: false` (OpenAI strict mode), removed constraint keywords (`maxLength`, `maxItems`, `minimum`, `maximum`) that some providers reject. MCP server validates these server-side
- **Pre-existing test bugs** — Config test missing `initTimeout` in expected defaults; execute tests passing args as `id` parameter

## [2.25.0] - 2026-03-04

### Added

- **Proactive Memory Auto-Save** — 4-layer system ensures agents use Surreal-Memory without explicit instructions
  - **MCP `instructions`** — Behavioral directives in InitializeResult, auto-injected into agent context
  - **Post-tool passive capture** — Server-side auto-analysis of recall/context/recap/explain results with rate limiting (3/min)
  - **Plugin `instructions` field** — Short nudge for all plugin users
  - **Enhanced stop hook** — Transcript capture 80→150 lines, session summary extraction, always saves at least one context memory
- **Ollama embedding provider** — Local zero-cost inference via Ollama API (contributed by @xthanhn91)

### Fixed

- **Scale performance bottlenecks** — Consolidation prune, neuron dedup, cache improvements (PR #23)
- **OpenClaw plugin `execute()` signature** — Missing `id` parameter broke all agent tool calls (issue #19)
- **Auto-consolidation crash** — `ValueError: 'none' is not a valid ConsolidationStrategy` (issue #20)
- **`smem remember --stdin`** — CLI now supports piped input for safe shell usage (issue #21)
- **CI test compatibility** — `test_remember_sensitive_content` mock fix for Python 3.11

## [2.24.2] - 2026-03-03

### Added

- **Dashboard Phase 2** — Complete visual dashboard overhaul
  - **Sigma.js graph visualization** — WebGL-rendered neural graph with ForceAtlas2 layout, node limit selector (100-1000), click-to-inspect detail panel, color-coded by neuron type
  - **ReactFlow mindmap** — Interactive fiber mindmap with dagre left-to-right tree layout, custom nodes (root/group/leaf), MiniMap, zoom/pan, click-to-select neuron details
  - **Theme toggle** — Light / Dark / System cycle button in TopBar, warm cream light mode (`#faf8f3`), class-based TailwindCSS 4 dark mode via `@custom-variant`
  - **Delete brain** — Trash icon on inactive brains in Overview table with confirmation dialog
  - **Click-to-switch brain** — Click inactive brain row to switch active brain
- **CLI update check fix** — Editable/dev installs no longer show misleading "Update available" prompts

### Removed

- **Legacy dashboard UI** — Removed `dashboard.html`, `index.html`, legacy JS/CSS/locales (4,451 LOC), `/static` mount from FastAPI

### Dependencies

- Added `@xyflow/react`, `@dagrejs/dagre` (ReactFlow mindmap)
- Added `graphology-layout-forceatlas2` (Sigma.js graph layout)

## [2.24.1] - 2026-03-03

### Fixed

- **IntegrityError in consolidation** — `save_maturation` FK constraint failed when orphaned maturation records referenced deleted fibers
  - Added `cleanup_orphaned_maturations()` to purge stale records before stage advancement
  - Defensive try/except for any remaining FK errors during `_mature()`

### Tests

- 2 new tests for orphaned maturation handling
- Total: 3145 passing

## [2.24.0] - 2026-03-03

### Fixed

- **[CRITICAL] SQL Injection Prevention** — `get_synapses_for_neurons` direction param validated against whitelist instead of raw f-string
- **[HIGH] BFS max_hops off-by-one** — Nodes at depth=max_hops no longer uselessly enqueued then discarded
- **[HIGH] Bidirectional path search** — `memory_store.get_path()` now respects `bidirectional=True` via `to_undirected()`
- **[HIGH] JSON-RPC parse errors** — Returns proper `{"code": -32700}` error instead of silently dropping malformed messages
- **[HIGH] Encryption failure policy** — Returns error instead of silently storing plaintext when encryption fails
- **[HIGH] `disable_auto_save` placement** — Moved inside `try` block in tool_handlers and conflict_handler so `finally` always re-enables
- **[HIGH] Cross-brain depth validation** — Added int coercion + 0-3 clamping for depth parameter
- **[HIGH] Factory sync exception handling** — Narrowed bare `except Exception` to specific exception types
- **[HIGH] SSN pattern false positives** — Excluded invalid prefixes (000, 666, 900-999); raised base64/hex minimums to 64 chars
- **[MEDIUM] MCP notification handling** — Unknown notifications return None instead of error responses
- **[MEDIUM] Brain ID error propagation** — New `_get_brain_or_error()` helper prevents uncaught ValueError in 6 handlers
- **[MEDIUM] Connection handler I/O** — Removed unused brain fetch in `_explain`
- **[MEDIUM] Evidence fetch optimization** — Removed wasted source neuron from evidence query
- **[MEDIUM] Narrative date validation** — Added `end_date < start_date` guard
- **[MEDIUM] CORS port handling** — Enumerate common dev ports instead of invalid `:*` wildcard
- **[MEDIUM] Embedding config** — Graceful fallback instead of crash on invalid provider
- **[LOW] Type coercion** — max_hops/max_fibers/max_depth safely coerced to int
- **[LOW] Immutability** — Dict mutations replaced with spread patterns in review_handler and encoder
- **[LOW] Schema cleanup** — Removed empty `"required": []` from smem_suggest

### Tests

- Fixed and added 5 tests (max_hops_capped, avg_weight, default_hops, tier assertions, embedding fallback)
- Total: 3143 passing

## [2.23.0] - 2026-03-03

### Added

- **smem_explain — Connection Explainer** — New MCP tool to explain how two entities are related
  - Finds shortest path through synapse graph via bidirectional BFS
  - Hydrates path with fiber evidence (memory summaries)
  - Returns structured steps + human-readable markdown explanation
  - New engine module: `connection_explainer.py` with `ConnectionStep` and `ConnectionExplanation` dataclasses
  - New handler mixin: `ConnectionHandler` following established mixin pattern
  - Args: `from_entity`, `to_entity` (required), `max_hops` (optional, 1-10, default 6)

### Fixed

- **OpenClaw Compatibility** — Handle JSON string arguments in MCP `tools/call` handler
  - OpenClaw sends `arguments` as JSON string instead of dict — now auto-parsed
  - Prevents crash when receiving `"arguments": "{\"content\": \"...\"}"` format

### Improved

- **Bidirectional BFS** — `get_path()` in SQLite storage now supports `bidirectional=True`
  - Uses `UNION ALL` to traverse both outgoing and incoming synapse edges
  - Updated abstract base + all 5 storage implementations

### Tests

- 11 new tests for connection explainer (engine + MCP handler + integration)
- Total: 3140+ passing

## [2.22.0] - 2026-03-03

### Fixed

- **#12 Version Mismatch** — Detect editable installs in update hint, show version in `smem_stats`
- **#14 Dedup on Remember** — Enable SimHash dedup (Tier 1) by default, surface `dedup_hint` in remember response, skip content < 20 chars
- **#11 SEMANTIC Stage Blocked** — Rehearse maturation records on retrieval so memories can reach SEMANTIC stage (requires 3+ distinct reinforcement days)
- **#15 Low Activation Efficiency** — Fix Hebbian learning None activation floor (0.1 instead of None → delta > 0), add dormant neuron reactivation during consolidation

### Added

- **#10 Semantic Linking** — `SemanticLinkingStep` cross-links entity/concept neurons to existing similar neurons (reduces orphan rate)
- **#13 Neuron Diversity** — `ExtractActionNeuronsStep` + `ExtractIntentNeuronsStep` extract ACTION/INTENT neurons from verb/goal phrases (improves type diversity from 4-5 to 6-7 of 8 types)
- **Dormant Reactivation** — Consolidation ENRICH tier bumps up to 20 dormant neurons (access_frequency=0) with +0.05 activation

### Tests

- 55 new tests across 6 test files: version check (12), dedup default (9), maturation rehearsal (5), semantic linking (6), action/intent extraction (15), activation efficiency (8)
- Total: 3127 passing

## [2.21.0] - 2026-03-03

### Added

- **Cross-Language Recall Hint** — Smart detection when recall misses due to language mismatch
  - Detects query language vs brain majority language (Vietnamese ↔ English)
  - Shows actionable `cross_language_hint` in recall response when embedding is not enabled
  - Suggests `pip install` if sentence-transformers not installed, config-only if already installed
  - `detect_language()` extracted as reusable module-level function with Vietnamese-unique char detection

- **Embedding Setup Guide** — Comprehensive docs for all embedding providers
  - New `docs/guides/embedding-setup.md` with provider comparison, config examples, troubleshooting
  - Free multilingual model recommendations: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 384D, ~440MB)
  - Provider comparison table: sentence_transformer (free/local) vs Gemini vs OpenAI

- **Embedding Documentation & Onboarding**
  - README: updated "None — pure algorithmic" → "Optional", added embedding quick-start section
  - `.env.example`: added `GEMINI_API_KEY`, `OPENAI_API_KEY` vars
  - Onboarding step 6: suggests cross-language recall setup for new users

### Improved

- **Vietnamese Language Detection** — More accurate short-text detection
  - Added `_VI_UNIQUE_CHARS` set (chars exclusive to Vietnamese, not shared with French/Spanish)
  - Short text like "lỗi xác thực" now correctly detected as Vietnamese

### Tests

- 18 new tests in `test_cross_language_hint.py` (8 detect_language + 10 hint logic)
- All 3090+ tests pass

## [2.20.0] - 2026-03-03

### Added

- **Gemini Embedding Provider** — Cross-language recall via Google Gemini embeddings (PR #9 by @xthanhn91)
  - `GeminiEmbedding` provider: `gemini-embedding-001` (3072D), `text-embedding-004` (768D)
  - Parallel anchor sources: embedding + FTS5 run concurrently (not fallback-only)
  - Config pipeline: `config.toml[embedding]` → `EmbeddingSettings` → `BrainConfig` → SQLite
  - Doc training embeds anchor neurons for cross-language retrieval
  - E2E validated: 100/100 Vietnamese queries on English KB (avg confidence 0.98)
  - Optional dependency: `pip install 'surreal-memory[embeddings-gemini]'`

- **Sufficiency Enhancements** — Smarter retrieval gating
  - EMA calibration: per-gate accuracy tracking, auto-downgrade unreliable gates
  - Per-query-type thresholds: strict (factual), lenient (exploratory), default profiles
  - Diminishing returns gate: early-exit when multi-pass retrieval plateaus

### Fixed

- **Comprehensive Audit** — 7 CRITICAL, 17 HIGH, 18 MEDIUM fixes
  - Security: auth guard on consolidation routes, CORS wildcard removal, path traversal fix
  - Performance: `@lru_cache` regex, cached QueryRouter/MemoryEncryptor, `asyncio.gather` embeddings
  - Infrastructure: `.dockerignore`, `.env.example`, bounded exports, async cursor managers
- **PR #9 Review Fixes** — 3 HIGH, 6 MEDIUM, 3 LOW
  - Bare except → specific exceptions in doc_trainer
  - `EmbeddingSettings` frozen + validated (rejects invalid providers)
  - Probe-first early exit in embedding anchor scan (performance)
  - Correct task_type for semantic discovery consolidation
  - Hardcoded paths → env vars in E2E scripts

### Tests

- 33 new sufficiency tests (EMA calibration, query profiles, diminishing returns)
- 6 new EmbeddingSettings validation tests
- 13 new Gemini embedding provider tests
- Full suite: 3054 passed, 0 failed

## [2.19.0] - 2026-03-02

### Added

- **React Dashboard** — Modern dashboard replacing legacy Alpine.js/vis.js
  - Vite 7 + React 19 + TypeScript + TailwindCSS 4 + shadcn/ui
  - Warm cream light theme (`#faf8f3`) with dark mode support
  - 7 pages: Overview, Health (Recharts radar), Graph, Timeline, Evolution, Diagrams, Settings
  - TanStack Query 5 for data fetching, Zustand 5 for state
  - Lazy-loaded routes with skeleton loaders
  - `/ui` and `/dashboard` serve React SPA, legacy at `/ui-legacy` and `/dashboard-legacy`
  - Brain file info: paths, sizes, disk usage in Settings page

- **Telegram Backup Integration** — Send brain `.db` files to Telegram
  - `TelegramClient` (aiohttp): `send_message` (auto-split >4096 chars), `send_document`, `backup_brain`
  - `TelegramConfig` frozen dataclass in `unified_config.py` (`[telegram]` TOML section)
  - CLI: `smem telegram status`, `smem telegram test`, `smem telegram backup [--brain NAME]`
  - MCP tool: `smem_telegram_backup` (28 total tools)
  - Dashboard API: `GET /api/dashboard/telegram/status`, `POST .../test`, `POST .../backup`
  - Dashboard Settings page: status indicator, test button, backup button
  - Bot token via `SURREAL_MEMORY_TELEGRAM_BOT_TOKEN` env var only (never in config file)
  - Chat IDs in `config.toml` under `[telegram]` section

- **Brain Files API** — `GET /api/dashboard/brain-files`
  - Returns brains directory path, per-brain file path + size, total disk usage

### Tests

- 15 new Telegram tests: config, token, client, status, MCP handler
- MCP tool count updated (27→28)

## [2.18.0] - 2026-03-02

### Added

- **Export Markdown** — `smem brain export --format markdown -o brain.md`
  - Human-readable brain export grouped by memory type (facts, decisions, insights, etc.)
  - Tag index with occurrence counts
  - Statistics table with neuron/synapse/fiber breakdowns
  - Pinned memory indicators and sensitive content exclusion support
  - New module: `cli/markdown_export.py` (~180 LOC)

- **Original Timestamp** — `event_at` parameter on `smem_remember`
  - MCP: `smem_remember(content="Meeting at 8am", event_at="2026-03-02T08:00:00")`
  - CLI: `smem remember "Meeting" --timestamp "2026-03-02T08:00:00"`
  - Time neurons and fiber `time_start/time_end` use the original event time
  - Supports ISO format with optional timezone (auto-stripped for UTC storage)

### Changed

- **Health Roadmap Enhancement** — Concrete metrics in improvement actions
  - Actions now include specific numbers: "Store memories to build ~250 more connections (current: 0.5 synapses/neuron, target: 3.0+)"
  - Added `timeframe` field to roadmap: "~2 weeks with regular use"
  - Dynamic action strings computed from actual brain metrics (neuron counts, orphan rate, etc.)
  - Grade transition messages include estimated timeframe

### Tests

- 31 new tests: `test_markdown_export.py` (11), `test_health_roadmap.py` (13), `test_event_timestamp.py` (7)

## [2.17.0] - 2026-03-02

### Added

- **Knowledge Base Training** — Multi-format document extraction with pinned memories
  - 12 supported formats: .md, .mdx, .txt, .rst (passthrough), .pdf, .docx, .pptx, .html/.htm (rich docs), .json, .xlsx, .csv (structured data)
  - `doc_extractor.py` — Format-specific extractors with 50MB file size limit
  - Optional dependencies via `surreal-memory[extract]` for non-text formats (pymupdf4llm, python-docx, python-pptx, beautifulsoup4, markdownify, openpyxl)
- **Pinned Memories** — Permanent knowledge that bypasses decay, pruning, and compression
  - `Fiber.pinned: bool` field — pinned fibers skip all lifecycle operations
  - 4 lifecycle bypass points: decay, pruning, compression, maturation
  - `smem_pin` MCP tool for manual pin/unpin
- **Training File Dedup** — SHA-256 hash tracking prevents re-ingesting same documents
  - `training_files` table with hash, status, progress tracking
  - Resume support for interrupted training sessions
- **Tool Memory System** — Tracks MCP tool usage patterns and effectiveness
  - `MemoryType.TOOL` — New memory type (90-day expiry, 0.06 decay rate)
  - `SynapseType.EFFECTIVE_FOR` + `USED_WITH` — Tool effectiveness and co-occurrence synapses
  - PostToolUse hook — Fast JSONL buffer capture (<50ms, no SQLite on hot path)
  - `engine/tool_memory.py` — Batch processing during consolidation
  - `PROCESS_TOOL_EVENTS` consolidation strategy

### Fixed (Comprehensive Audit — 4 CRITICAL, 8 HIGH, 12 MEDIUM)

- **CRITICAL**: Auth guard on consolidation routes, CORS wildcard removal, path traversal fix, coverage threshold enforcement
- **HIGH**: Reject null client IP, sanitize error messages, Windows ACL key protection, FalkorDB password warning
- **Performance**: Module-level regex compilation with `@lru_cache`, cached QueryRouter + MemoryEncryptor (lazy singleton), `asyncio.gather` for parallel embeddings, batch neuron delete (chunked 500), SQL FILTER clause combining queries
- **Infrastructure**: `.dockerignore`, `.env.example`, bounded export (LIMIT 50000), `asyncio.Lock` for storage cache, cursor context managers

### Changed

- Schema version 18 → 20 (tool_events table, pinned column on fibers, training_files table)
- SynapseType enum: 22 → 24 types (EFFECTIVE_FOR, USED_WITH)
- MemoryType enum: 10 → 11 types (TOOL)
- MCP tools: 26 → 27 (added smem_pin)
- ROADMAP.md — Complete rewrite as forward-looking 5-phase vision
- Agent instructions — 7 new sections covering all 28 MCP tools
- MCP prompt — Added KB training, pin, health, review, import instructions

---

## [2.16.0] - 2026-02-28

### Added

- **Algorithmic Sufficiency Check** — Post-stabilization gate that early-exits when activation signal is too weak
  - 8-gate evaluation (priority-ordered, first match wins): no_anchors, empty_landscape, unstable_noise, ambiguous_spread, intersection_convergence, high_coverage_strong_hit, focused_result, default_pass
  - Unified confidence formula from 7 weighted inputs (activation, focus_ratio, coverage, intersection_ratio, proximity, stability, path_diversity)
  - Conservative bias — false-INSUFFICIENT penalized 10× more than false-SUFFICIENT
  - `engine/sufficiency.py` (~302 LOC), `storage/sqlite_calibration.py` (~133 LOC)
  - Schema migration v17 → v18 (`retrieval_calibration` table)

---

## [2.15.1] - 2026-02-28

### Fixed

- **SharedStorage CRUD Endpoint Mismatch** — Client called endpoints that didn't exist on server
  - Added 14 CRUD endpoints to `server/routes/memory.py` (neurons + synapses full lifecycle, state, neighbors, path)
  - 6 new Pydantic models in `server/models.py`
- **Brain Import Deduplication** — Changed `INSERT` → `INSERT OR REPLACE` in `sqlite_brain_ops.py` for idempotent imports

---

## [2.15.0] - 2026-02-28

### Added

- **Trusted Networks for Docker/Container Deployments** — Configurable non-localhost access via `SURREAL_MEMORY_TRUSTED_NETWORKS` env var (CIDR notation)
  - `is_trusted_host()` function with safe `ipaddress` module validation
  - Default remains localhost-only (secure by default)

### Fixed

- **OpenClaw Plugin Zod Peer Dependency** — Pinned `zod` to `^3.0.0`

---

## [2.14.0] - 2026-02-27

### Added

- **MCP Tool Tiers** — 3-tier system (minimal/standard/full) for controlling exposed tools
  - `ToolTierConfig` frozen dataclass with case-insensitive tier parsing
  - `get_tool_schemas_for_tier()` filters tools by tier level
  - Minimal: 4 core tools, Standard: 8 tools, Full: all 27 tools
  - Hidden tools still callable via dispatch (tier controls visibility, not access)
- **Consolidation Eligibility Hints** — `_eligibility_hints()` explains why 0 changes happened
- **Habits Status** — Progress bars for emerging patterns
- **Diagnostics Improvements** — Actionable recommendations with specific numbers
- **Graph SVG Export** — Pure Python SVG export with dark theme, zero external deps

---

## [2.13.0] - 2026-02-27

### Added

- **Error Resolution Learning** — When a new FACT/INSIGHT contradicts an existing ERROR memory, the system creates a `RESOLVED_BY` synapse linking fix → error instead of just flagging a conflict
  - `RESOLVED_BY` synapse type added to `SynapseType` enum (22 types total)
  - Resolved errors get ≥50% activation demotion (2x stronger than normal conflicts)
  - Error neurons marked with `_conflict_resolved` and `_resolved_by` metadata
  - Auto-detection via neuron metadata `{"type": "error"}` — no caller changes needed
  - Zero-cost: pure graph manipulation, no LLM calls
  - 7 new tests in `test_error_resolution.py`

### Changed

- `resolve_conflicts()` accepts optional `existing_memory_type` parameter
- `conflict_detection.py` now imports `logging` module for RESOLVED_BY synapse debug logging

---

## [2.8.1] - 2026-02-23

### Added

- **FalkorDB Graph Storage Backend** — Optional graph-native storage replacing SQLite for high-performance traversal
  - `FalkorDBStorage` composite class implementing full `NeuralStorage` ABC via 5 specialized mixins
  - `FalkorDBBaseMixin` — connection pooling, query helpers (`_query`, `_query_ro`), index management
  - `FalkorDBNeuronMixin` — neuron CRUD with graph node operations
  - `FalkorDBSynapseMixin` — synapse CRUD with typed graph edges
  - `FalkorDBFiberMixin` — fiber CRUD with `CONTAINS` relationships, batch operations
  - `FalkorDBGraphMixin` — native Cypher spreading activation (1-4 hop BFS via variable-length paths)
  - `FalkorDBBrainMixin` — brain registry graph, import/export, graph-level clear
  - Brain-per-graph isolation (`brain_{id}`) for native multi-tenancy
  - Read-only query routing via `ro_query` for registry reads and fiber lookups
  - Per-neuron limit enforcement in `find_fibers_batch` via UNWIND+collect/slice Cypher pattern
  - Connection health verification via Redis PING with automatic reconnect
  - `docker-compose.falkordb.yml` — standalone FalkorDB service configuration
  - Migration CLI: `smem migrate falkordb` to move SQLite brain data to FalkorDB
  - 69 tests across 6 test files (auto-skip when FalkorDB unavailable)
  - SQLite remains default — FalkorDB is opt-in via `[storage]` TOML config

### Fixed

- **mypy: `set_brain` missing from ABC** — Added `set_brain(brain_id)` to `NeuralStorage` base class, resolving 2 mypy errors in `unified_config.py`
- **Registry reads used write queries** — Added `_registry_query_ro()` for read-only brain registry operations (`get_brain`, `find_brain_by_name`)
- **`find_fibers_batch` ignored `limit_per_neuron`** — Rewrote with UNWIND+collect/slice Cypher for proper per-neuron limiting
- **FalkorDB health check was superficial** — `_get_falkordb_storage()` now performs actual Redis PING instead of just `_db is not None` check
- **`export_brain` leaked `brain_id` in error** — Sanitized to generic "Brain not found" message
- **Import sorting (I001)** — Fixed `falkordb.asyncio` before `redis.asyncio` in `falkordb_store.py`
- **Unused import (F401)** — Removed stale `SQLiteStorage` import from `unified_config.py`
- **Quoted annotation (UP037)** — Unquoted `_storage_cache` and `_falkordb_storage` type annotations
- **Silent error logging** — Upgraded index creation and connection close errors from debug to warning level

## [2.8.0] - 2026-02-22

### Added

- **Adaptive Recall (Bayesian Depth Prior)** — System learns optimal retrieval depth per entity pattern
  - Beta distribution priors per (entity, depth) pair — picks depth with highest E[Beta(a,b)]
  - 5% epsilon exploration to discover better depths for known entities
  - Fallback to rule-based detection when < 5 queries or no priors exist
  - Outcome recording: updates alpha (success) or beta (failure) based on confidence + fibers_matched
  - 30-day decay (a *= 0.9, b *= 0.9) to forget stale patterns
  - `DepthPrior`, `DepthDecision` frozen dataclasses + `AdaptiveDepthSelector` engine
  - `SQLiteDepthPriorMixin` with batch fetch, upsert, stale decay, delete operations
  - Configurable: `adaptive_depth_enabled` (default True), `adaptive_depth_epsilon` (default 0.05)
- **Tiered Memory Compression** — Age-based compression preserving entity graph structure (zero-LLM)
  - 5 tiers: Full (< 7d), Extractive (7-30d), Entity-only (30-90d), Template (90-180d), Graph-only (180d+)
  - Entity density scoring: `count(neurons_referenced) / word_count` per sentence
  - Reversible for tiers 1-2 (backup stored), irreversible for tiers 3-4
  - Integrated as `COMPRESS` strategy in `ConsolidationEngine` (Tier 2)
  - `CompressionTier` IntEnum, `CompressionConfig`, `CompressionResult` frozen dataclasses
  - `SQLiteCompressionMixin` for backup storage with stats
  - Configurable: `compression_enabled` (default True), `compression_tier_thresholds` (7, 30, 90, 180 days)
- **Multi-Device Sync** — Hub-and-spoke incremental sync via change log + sequence numbers
  - **Device Identity**: UUID-based device_id generation, persisted in config, `DeviceInfo` frozen dataclass
  - **Change Tracking**: Append-only `change_log` table recording all neuron/synapse/fiber mutations
    - `ChangeEntry` frozen dataclass, `SQLiteChangeLogMixin` with 6 CRUD methods
    - `record_change()`, `get_changes_since(sequence)`, `mark_synced()`, `prune_synced_changes()`
  - **Incremental Sync Protocol**: Delta-based merge using neural-aware conflict resolution
    - `SyncRequest`, `SyncResponse`, `SyncChange`, `SyncConflict` frozen dataclasses
    - `ConflictStrategy` enum: prefer_recent, prefer_local, prefer_remote, prefer_stronger
    - Neural merge rules: weight=max, access_frequency=sum, tags=union, conductivity=max, delete wins
  - **Sync Engine**: `SyncEngine` orchestrator with `prepare_sync_request()`, `process_sync_response()`, `handle_hub_sync()`
  - **Hub Server Endpoints** (localhost-only by default):
    - `POST /hub/register` — register device for brain
    - `POST /hub/sync` — push/pull incremental changes
    - `GET /hub/status/{brain_id}` — sync status + device count
    - `GET /hub/devices/{brain_id}` — list registered devices
  - **3 new MCP tools** (full tier only):
    - `smem_sync` — trigger manual sync (push/pull/full)
    - `smem_sync_status` — show pending changes, devices, last sync
    - `smem_sync_config` — configure hub URL, auto-sync, conflict strategy
  - `SyncConfig` frozen dataclass: enabled (default False), hub_url, auto_sync, sync_interval_seconds, conflict_strategy
  - Device tracking columns on neurons/synapses/fibers: `device_id`, `device_origin`, `updated_at`
  - Schema migrations v15 → v16 (depth_priors, compression_backups, fiber compression_tier) → v17 (change_log, devices, device columns)

### Changed

- **SQLite schema** — Version 15 → 17 (two migrations)
- **MCP tools** — Expanded from 23 to 26 (`smem_sync`, `smem_sync_status`, `smem_sync_config`)
- **MCPServer mixin chain** — Added `SyncToolHandler` mixin
- **`Fiber` model** — Added `compression_tier: int = 0` field
- **`BrainConfig`** — Added 4 new fields: `adaptive_depth_enabled`, `adaptive_depth_epsilon`, `compression_enabled`, `compression_tier_thresholds`
- **`UnifiedConfig`** — Added `device_id` field and `SyncConfig` dataclass
- **`ConsolidationEngine`** — Added `COMPRESS` strategy enum + Tier 2 registration + `fibers_compressed`/`tokens_saved` report fields
- **Hub endpoints** — Pydantic request validation with regex-based brain_id/device_id format checks
- Tests: 2687 passed (up from 2527), +160 new tests across 8 test files

## [2.7.1] - 2026-02-21

### Added

- **MCP Tool Tiers** — Config-based filtering to reduce token overhead per API turn
  - 3 tiers: `minimal` (4 tools, ~84% savings), `standard` (8 tools, ~69% savings), `full` (all 23, default)
  - `ToolTierConfig` frozen dataclass in `unified_config.py` with `from_dict()`/`to_dict()`
  - `get_tool_schemas_for_tier(tier)` in `tool_schemas.py` — filters schemas by tier
  - `[tool_tier]` TOML section in `config.toml` for persistent configuration
  - Hidden tools remain callable via dispatch — only schema exposure changes
  - CLI command: `smem config tier [--show | minimal | standard | full]`
- **Description Compression** — All 23 tool descriptions compressed (~22% token reduction at full tier)

### Changed

- `MCPServer.get_tools()` now respects `config.tool_tier.tier` setting
- `tool_schemas.py` refactored: `_ALL_TOOL_SCHEMAS` module-level list + `TOOL_TIERS` dict
- Tests: added 28 new tests in `test_tool_tiers.py`

## [2.7.0] - 2026-02-18

### Added

- **Spaced Repetition Engine** — Leitner box system (5 boxes: 1d, 3d, 7d, 14d, 30d) for memory reinforcement
  - `ReviewSchedule` frozen dataclass: fiber_id, brain_id, box (1–5), next_review, streak, review_count
  - `SpacedRepetitionEngine`: `get_review_queue()`, `process_review()` (calls `ReinforcementManager`), `auto_schedule_fiber()`
  - `advance(success)` returns new schedule instance — box increments on success (max 5), resets to 1 on failure
  - Auto-scheduling: fibers with `priority >= 7` are automatically scheduled in `_remember`
  - `SQLiteReviewsMixin`: upsert, get_due, get_stats with `min(limit, 100)` cap
  - `InMemoryReviewsMixin` for testing
  - `ReviewHandler` MCP mixin: `smem_review` tool (queue/mark/schedule/stats actions)
  - Schema migration v14 → v15 (`review_schedules` table + 2 indexes)
- **Memory Narratives** — Template-based markdown narrative generation (no LLM)
  - 3 modes: `timeline` (date range), `topic` (spreading activation via `ReflexPipeline`), `causal` (CAUSED_BY chain traversal)
  - `NarrativeItem` + `Narrative` frozen dataclasses with `to_markdown()` rendering
  - Timeline mode: queries fibers by date range, sorts chronologically, groups by date headers
  - Topic mode: runs SA query, fetches matched fibers, sorts by relevance
  - Causal mode: uses `trace_causal_chain()` to follow CAUSED_BY synapses, builds cause→effect narrative
  - `NarrativeHandler` MCP mixin: `smem_narrative` tool (timeline/topic/causal actions)
  - Configurable `max_fibers` with server-side cap of 50
- **Semantic Synapse Discovery** — Offline consolidation using embeddings to find latent connections
  - Batch embeds CONCEPT + ENTITY neurons, evaluates cosine similarity pairs above threshold
  - Creates SIMILAR_TO synapses with `weight = similarity * 0.6` and `{"_semantic_discovery": True}` metadata
  - Configurable: `semantic_discovery_similarity_threshold` (default 0.7), `semantic_discovery_max_pairs` (default 100)
  - Integrated as Tier 5 (`SEMANTIC_LINK`) in `ConsolidationEngine` strategy dispatch
  - 2× faster decay for unreinforced semantic synapses in `_prune` (reinforced_count < 2 → decay factor 0.5)
  - Optional — gracefully skipped if `sentence-transformers` not installed
  - `SemanticDiscoveryResult` dataclass: neurons_embedded, pairs_evaluated, synapses_created, skipped_existing
- **Cross-Brain Recall** — Parallel spreading activation across multiple brains
  - Extends `smem_recall` with optional `brains` array parameter (max 5 brains)
  - Resolves brain names → DB paths via `UnifiedConfig`, opens temporary `SQLiteStorage` per brain
  - Parallel query via `asyncio.gather`, each brain runs independent `ReflexPipeline`
  - SimHash-based deduplication across brain results (keeps higher confidence on collision)
  - Confidence-sorted merge with `[brain_name]` prefixed context sections
  - `CrossBrainFiber` + `CrossBrainResult` frozen dataclasses
  - Temporary storage instances closed in `finally` blocks

### Changed

- **MCPServer mixin chain** — Added `ReviewHandler` + `NarrativeHandler` mixins (16 → 18 handler mixins)
- **MCP tools** — Expanded from 21 to 23 (`smem_review`, `smem_narrative`)
- **SQLite schema** — Version 14 → 15 (`review_schedules` table)
- **`smem_recall` schema** — Added `brains` array property for cross-brain queries
- **`BrainConfig`** — Added `semantic_discovery_similarity_threshold` and `semantic_discovery_max_pairs` fields
- **`ConsolidationEngine`** — Added `SEMANTIC_LINK` strategy enum + Tier 5 + `semantic_synapses_created` report field
- **Consolidation prune** — Unreinforced semantic synapses (`_semantic_discovery` metadata) decay at 2× rate
- Tests: 2399 passed (up from 2314), +85 new tests across 4 features

## [2.6.0] - 2026-02-18

### Added

- **Smart Context Optimizer** — Composite scoring replaces naive loop in `smem_context`
  - 5-factor weighted score: activation (0.30) + priority (0.25) + frequency (0.20) + conductivity (0.15) + freshness (0.10)
  - SimHash-based deduplication removes near-duplicate content before token budgeting
  - Proportional token budget allocation: items get budget proportional to their composite score
  - Items below minimum budget (20 tokens) are dropped; oversized items are truncated
  - `optimization_stats` field in response shows `items_dropped` and `top_score`
- **Proactive Alerts Queue** — Persistent brain health alerts with full lifecycle management
  - `Alert` frozen dataclass with `AlertStatus` (active → seen → acknowledged → resolved) and 7 `AlertType` enum values
  - `SQLiteAlertsMixin` with CRUD operations: `record_alert` (6h dedup cooldown), `get_active_alerts`, `mark_alerts_seen`, `mark_alert_acknowledged`, `resolve_alerts_by_type`
  - `AlertHandler` MCP mixin: `smem_alerts` tool (list/acknowledge actions)
  - Auto-creation from health pulse hints; auto-resolution when conditions clear
  - Pending alert count surfaced in `smem_remember`, `smem_recall`, `smem_context` responses
  - Schema migration v13 → v14 (alerts table + indexes)
- **Recall Pattern Learning** — Discover and materialize query topic co-occurrence patterns
  - `extract_topics()` — keyword-based topic extraction from recall queries (min_length=3, cap 10)
  - `mine_query_topic_pairs()` — session-grouped, time-windowed (600s default) pair mining
  - `extract_pattern_candidates()` — frequency filtering + confidence scoring
  - `learn_query_patterns()` — materializes patterns as CONCEPT neurons + BEFORE synapses with `{"_query_pattern": True}` metadata
  - `suggest_follow_up_queries()` — follows BEFORE synapses for related topic suggestions
  - Integrated into LEARN_HABITS consolidation strategy
  - `related_queries` field added to `smem_recall` response

### Changed

- **MCPServer mixin chain** — Added `AlertHandler` mixin (15 → 16 handler mixins)
- **MCP tools** — Expanded from 20 to 21 (`smem_alerts`)
- **SQLite schema** — Version 13 → 14 (alerts table)
- **`smem_context` response** — Now includes `optimization_stats` when items are dropped
- **`smem_recall` response** — Now includes `related_queries` from learned patterns
- Tests: 2314 passed (up from 2291)

## [2.5.0] - 2026-02-18

### Added

- **Onboarding flow** — Detects fresh brain (0 neurons + 0 fibers) and surfaces a 4-step getting-started guide on the first tool call (`_remember`, `_recall`, `_context`, `_stats`). Shows once per server instance.
- **Background expiry cleanup** — Fire-and-forget task auto-deletes expired `TypedMemory` + underlying fibers on a configurable interval (default 12h, max 100/run). Fires `MEMORY_EXPIRED` hooks. Piggybacks on `_check_maintenance()`.
- **Scheduled consolidation** — Background `asyncio` loop runs consolidation every 24h (configurable strategies: prune, merge, enrich). Shares `_last_consolidation_at` with `MaintenanceHandler` to prevent overlap. Initial delay of one full interval avoids triggering on restart.
- **Version check handler** — Background task checks PyPI every 24h for newer versions of `surreal-memory`. Caches result and surfaces `update_hint` in `_remember`, `_recall`, `_stats` responses when an update is available. Uses `urllib` (no extra deps), validates HTTPS scheme.
- **Expiry alerts** — `warn_expiry_days` parameter on `smem_recall`; expiring-soon count in health pulse thresholds
- **Evolution dashboard** — `/api/evolution` REST endpoint + dashboard UI tab for brain maturation metrics (stage distribution, plasticity, proficiency)

### Changed

- **MaintenanceConfig** — Added 8 new config fields: `expiry_cleanup_enabled`, `expiry_cleanup_interval_hours`, `expiry_cleanup_max_per_run`, `scheduled_consolidation_enabled`, `scheduled_consolidation_interval_hours`, `scheduled_consolidation_strategies`, `version_check_enabled`, `version_check_interval_hours`
- **MCPServer mixin chain** — Added `OnboardingHandler`, `ExpiryCleanupHandler`, `ScheduledConsolidationHandler`, `VersionCheckHandler` mixins
- **Server lifecycle** — `run_mcp_server()` now starts scheduled consolidation + version check at startup, cancels all background tasks on shutdown

## [2.4.0] - 2026-02-17

### Security

- **6-phase security audit** — Comprehensive audit across 142K LOC / 190 files covering engine, storage, server, config, MCP/CLI, core, safety, utils, sync, integration, and extraction modules
- **Path traversal fixes** — 3 CRITICAL path injection vulnerabilities in CLI commands (tools, brain import, shortcuts) patched with `resolve()` + `is_relative_to()`
- **CORS hardening** — Replaced wildcard patterns with explicit localhost origins in FastAPI server
- **TOML injection prevention** — Added `_sanitize_toml_str()` for user-provided dedup config fields
- **API key masking** — `BrainModeConfig.to_dict()` now serializes api_key as `"***"` instead of plaintext
- **Info leak prevention** — Removed internal IDs, adapter names, and filesystem paths from 5 error messages across MCP, integration, and sync modules
- **WebSocket validation** — Brain ID format + length validation on subscribe action
- **Path normalization** — `SQLiteStorage` and `SURREAL_MEMORY_DIR` env var paths now resolved with `Path.resolve()`

### Fixed

- **Frozen core models** — `Synapse`, `Fiber`, `NeuronState`, `BrainSnapshot`, `FreshnessResult`, `MemoryFreshnessReport`, `Entity`, `WeightedKeyword`, `TimeHint` dataclasses are now `frozen=True` per immutability contract
- **merge_brain() atomicity** — Restore from backup on import failure instead of leaving empty brain
- **import_brain() orphan** — Brain record INSERT moved inside transaction to prevent orphan on failure
- **Division-by-zero guards** — `_predicates_conflict()` and homeostatic normalization protected against empty inputs
- **Datetime hardening** — 4 `datetime.fromisoformat()` call sites wrapped with try/except + naive UTC enforcement
- **Lateral inhibition** — Ceiling division for fair slot allocation across clusters
- **suggest_memory_type** — Word boundary matching prevents false positives (e.g. "add" no longer matches "address")
- **Git update command** — Detects current branch instead of hardcoded 'main'
- **Dead code removal** — Removed unused `updated_at` field, duplicate index, stale imports

### Performance

- **N+1 query elimination** — `consolidation._prune()` pre-fetches neighbor synapses in batch (was 500+ serial queries); `activation.activate()` caches neighbors + batch state pre-fetch (was ~1000 queries); `conflict_detection` uses `asyncio.gather()` for parallel searches
- **Export safety caps** — `export_brain()` limited to 50K neurons, 100K synapses, 50K fibers
- **Bounds enforcement** — 15+ storage methods capped with `min(limit, MAX)`, schema tool limits enforced
- **Regex pre-compilation** — `sensitive.py` and `trigger_engine.py` patterns compiled at module level with cache
- **Enrichment optimization** — Early exit on empty tags + zero intersection in O(n^2) Jaccard loop
- **ReDoS prevention** — Content length cap (100K chars) before regex matching in sensitive content detection

### Changed

- **BrainConfig.with_updates()** — Replaced 80-line manual field copy with `dataclasses.replace()`
- **DriftReport.variants** — Changed from mutable `list` to `tuple` on frozen dataclass
- **Mutable constants** — `VI_PERSON_PREFIXES` and `LOCATION_INDICATORS` converted to `frozenset`
- **Error handling** — 8 bare `except Exception` blocks narrowed to specific exception types with logging

## [2.2.0] - 2026-02-13

### Added

- **Config presets** — Three built-in profiles: `safe-cost` (token-efficient), `balanced` (defaults), `max-recall` (maximum retention). CLI: `smem config preset <name> [--list] [--dry-run]`
- **Consolidation delta report** — `run_with_delta()` wrapper computes before/after health snapshots around consolidation, showing purity, connectivity, and orphan rate changes. CLI consolidate now shows health delta.

### Fixed

- **CI lint parity** — CI now passes: fixed 14 lint errors in test files (unused imports, sorting, Yoda conditions)
- **Release workflow idempotency** — `gh release create` no longer fails when release already exists; uploads assets to existing release instead
- **CI test timeouts** — Added `pytest-timeout` (60s default) and `timeout-minutes: 15` to prevent stuck CI jobs

### Changed

- **Makefile** — Added `verify` target matching CI exactly (lint + format-check + typecheck + test-cov + security)
- **Auto-consolidation observability** — Background auto-consolidation now logs purity delta for monitoring

## [2.1.0] - 2026-02-13

### Fixed

- **Brain reset on config migration** — When upgrading to unified config (config.toml), `current_brain` is now migrated from legacy config.json so users don't lose their active brain selection
- **EternalHandler stale brain cache** — Eternal context now detects brain switches and re-creates the context instead of caching the initial brain ID indefinitely
- **Ruff lint errors** — Fixed 7 pre-existing lint violations (unused imports, naming convention, import ordering)
- **Mypy type errors** — Fixed 2 pre-existing type errors (`Any` import, `set()` arg-type)

### Added

- **CLI `--version` flag** — `smem --version` / `smem -V` now prints version and exits (standard CLI convention)
- **Actionable health scoring** — `smem_health` now returns `top_penalties`: top 3 ranked penalty factors with estimated gain and suggested action
- **Semantic stage progress** — `smem_evolution` now returns `stage_distribution` (fiber counts per maturation stage) and `closest_to_semantic` (top 3 EPISODIC fibers with progress % and next step)
- **Composable encoding pipeline** — Refactored monolithic `encode()` into 14 composable async pipeline steps (`PipelineContext` / `PipelineStep` / `Pipeline`)

### Changed

- **Dependency warning suppression** — pyvi/NumPy DeprecationWarnings are now suppressed at import time with targeted `filterwarnings`

## [2.3.1] - 2026-02-17

### Refactored

- **Engine cleanup** — Removed 176 lines of dead code across 6 engine modules
  - Deduplicated stop-word sets into shared `_STOP_WORDS` frozenset in `conflict_detection.py`
  - Replaced manual `Fiber()` constructor with `dc_replace()` in `consolidation.py`
  - Removed unused `reconstitute_answer()` from `retrieval_context.py`
  - Hoisted expansion suffix/prefix constants to module level in `retrieval.py`
  - Used `heapq.nlargest` instead of sorted+slice in retrieval reinforcement
  - Typed consolidation dispatch dict with `Callable[[], Awaitable[None]]` instead of `Any`

### Fixed

- **Unreachable break in dream** — Outer loop guard added to prevent quadratic blowup when activated neuron list is large (max 50K pairs)
- **JSON snapshot validation** — `brain_versioning.py` now validates parsed JSON is a dict before field access

## [2.3.0] - 2026-02-16

### Added

- **PreCompact + Stop auto-flush hooks** — Pre-compaction hook fires before context compression, parallel CI tests support
- **Emergency flush** (`smem_auto action="flush"`) — Pre-compaction emergency capture that skips dedup, lowers confidence threshold to 0.5, enables all memory types regardless of config, and boosts priority +2. Tag `emergency_flush` applied to all captured memories. Inspired by OpenClaw Memory's Layer 3 (`memoryFlush`)
- **Session gap detection** — `smem_session(action="get")` now returns `gap_detected: true` when content may have been lost between sessions (e.g. user ran `/new` without saving). Uses MD5 fingerprint stored on `session_set`/`session_end` to detect gaps from older code paths missing fingerprints
- **Auto-capture preference patterns** — Detects explicit preferences ("I prefer...", "always use..."), corrections ("that's wrong...", "actually, it should be..."), and Vietnamese equivalents. New memory type `preference` with 0.85 confidence
- **Windows surrogate crash fix** — MCP server now strips lone surrogate characters (U+D800-U+DFFF) from tool arguments before processing, preventing `UnicodeEncodeError` on Windows stdio pipes

### Fixed

- **CI lint failure** — Fixed ruff RUF002 (ambiguous EN DASH `–` in docstring) in `mcp/server.py`
- **CI stress test timeouts** — Skipped stress tests on GitHub runners to prevent CI timeout failures

### Changed

- **Release workflow hardened** — `release.yml` now validates tag version matches `pyproject.toml` + `__init__.py` before publishing, and runs full CI (lint + typecheck + test) as a gate before PyPI upload

## [Unreleased]

### Fixed

- **Agent forgets tools after `/new`** — `before_agent_start` hook now always injects `systemPrompt` with tool instructions, ensuring the agent knows about Surreal-Memory tools even after session reset. Previously only `prependContext` (data) was injected, leaving the agent unaware of available tools
- **Agent confuses CLI vs MCP tool calls** — `systemPrompt` injection explicitly states "call as tool, NOT CLI command", preventing agents from running `smem remember` in terminal instead of calling the `smem_remember` tool
- **`openclaw plugins list` not recognizing plugin on Windows** — Changed `main` and `openclaw.extensions` from TypeScript source (`src/index.ts`) to compiled output (`dist/index.js`). Added `prepublishOnly` and `postinstall` build scripts. Fixed `tsconfig.json` module resolution from `bundler` to `Node16` for broader compatibility
- **OpenClaw plugin ID mismatch** — Added explicit `"id": "surrealmemory"` to `openclaw` section in `package.json`, fixing the `plugin id mismatch (manifest uses "surrealmemory", entry hints "openclaw-plugin")` warning
- **Content-Length framing bug** — Switched from string-based buffer to raw `Buffer` for byte-accurate MCP message parsing. Fixes silent data corruption with non-ASCII content (Vietnamese, emoji, CJK)
- **Null dereference after close()** — `writeMessage()` and `notify()` now guard against null process reference
- **Unhandled tool call errors** — `callTool()` exceptions in tools.ts now caught and returned as structured error responses instead of crashing OpenClaw

### Added

- **Configurable MCP timeout** — New `timeout` plugin config option (default: 30s, max: 120s) for users on slow machines or first-time init
- **Actionable MCP error messages** — Initialize failures now include Python stderr output and specific hints:
  - `ENOENT` → tells user to check `pythonPath` in plugin config
  - Exit code 1 → suggests `pip install surreal-memory`
  - Timeout → prints captured stderr + verify command (`python -m surreal_memory.mcp`)

### Security

- **Least-privilege child env** — MCP subprocess now receives only whitelisted env vars (`PATH`, `HOME`, `PYTHONPATH`, `SURREAL_MEMORY_*`) instead of full `process.env`. Prevents leaking API keys and secrets to child process
- **Config validation** — `resolveConfig()` now validates types, ranges, and brain name pattern (`^[a-zA-Z0-9_\-.]{1,64}$`). Invalid values fall back to defaults instead of passing through
- **Input bounds on all tools** — Zod schemas now enforce max lengths: content (100K chars), query (10K), tags (50 items × 100 chars), expires_days (1–3650), context limit (1–200)
- **Buffer overflow protection** — 10 MB cap on stdio buffer; process killed if exceeded
- **Stderr cap** — Max 50 lines collected during init to prevent unbounded memory growth
- **Auto-capture truncation** — Agent messages truncated to 50K chars before sending to MCP
- **Graceful shutdown** — `close()` now removes listeners, waits up to 3s for exit, then escalates to SIGKILL
- **Config schema hardened** — Added `additionalProperties: false` and brain name `pattern` constraint

## [1.7.4] - 2026-02-11

### Fixed

- **Full mypy compliance**: Resolved all 341 mypy errors across 79 files (0 errors in 170 source files)
  - Added `TYPE_CHECKING` protocol stubs to all mixin classes (storage, MCP handlers)
  - Added generic type parameters to all bare `dict`/`list` annotations
  - Narrowed `str | None` → `str` before passing to typed parameters
  - Removed 14 stale `# type: ignore` comments
  - Added proper type annotations to `HybridStorage` factory delegate methods
  - Fixed variable name reuse across different types in same scope
  - Fixed missing `await` on coroutine calls in CLI commands

### Added

- **CLAUDE.md — Type Safety Rules**: New section documenting mixin protocol stubs, generic type params, Optional narrowing, and `# type: ignore` discipline to prevent future mypy regressions

## [1.7.3] - 2026-02-11

### Added

- **Bundled skills** — 3 Claude Code agent skills (memory-intake, memory-audit, memory-evolution) now ship inside the pip package under `src/surreal_memory/skills/`
- **`smem install-skills`** — new CLI command to install skills to `~/.claude/skills/`
  - `--list` shows available skills with descriptions
  - `--force` overwrites existing with latest version
  - Detects unchanged files (skip), changed files (report "update available"), missing `~/.claude/` (graceful error)
- **`smem init --skip-skills`** — skills are now installed as part of `smem init`; use `--skip-skills` to opt out
- Tests: 25 new unit tests for `setup_skills`, `_discover_bundled_skills`, `_classify_status`, `_extract_skill_description`

### Changed

- `_classify_status()` now recognizes "installed" and "updated" as success states
- `skills/README.md` updated: manual copy instructions replaced with `smem install-skills`

## [1.7.2] - 2026-02-11

### Security

- **CORS hardening**: Default CORS origins changed from `["*"]` to `["http://localhost:*", "http://127.0.0.1:*"]` (C2)
- **Bind address**: Default server bind changed from `0.0.0.0` to `127.0.0.1` (C4)
- **Migration safety**: Non-benign migration errors now halt and raise instead of silently advancing schema version (C8)
- **Info leakage**: Removed available brain names from 404 error responses (H21)
- **URI validation**: Graphiti adapter validates `bolt://`/`bolt+s://` URI scheme before connecting (H23)
- **Error masking**: Exception type names no longer leaked in MCP training error responses (H27)
- **Import screening**: `RecordMapper.map_record()` now runs `check_sensitive_content()` before importing external records (H33)

### Fixed

- Fix `RuntimeError: Event loop is closed` from aiosqlite worker thread on CLI exit (Python 3.12+)
  - **Root cause**: 4 CLI commands (`decay`, `consolidate`, `export`, `import`) called `get_shared_storage()` directly, bypassing `_active_storages` tracking — aiosqlite connections were never closed before event loop teardown
  - Route all CLI storage creation through `get_storage()` in `_helpers.py` so connections are properly tracked and cleaned up
  - Add `await asyncio.sleep(0)` after storage cleanup to drain pending aiosqlite worker thread callbacks before `asyncio.run()` tears down the loop
- **Bounds hardening**: MCP `_habits` fiber fetch reduced 10K→1K; `_context` limit capped at 200; REST `list_neurons` capped at 1000; `EncodeRequest.content` max 100K chars (H11-H13, H32)
- **Data integrity**: `import_brain` wrapped in `BEGIN IMMEDIATE` with rollback on failure (H14)
- **Code quality**: AWF adapter gets ImportError guard; redundant `enable_auto_save()` removed from train handler (C7, H26)
- **Public API**: Added `current_brain_id` property to `NeuralStorage`, `SQLiteStorage`, `InMemoryStorage` — replaces private `_current_brain_id` access (H25)

### Added

- **CLAUDE.md**: Project-level AI coding standards (architecture, immutability, datetime, security, bounds, testing, error handling, naming conventions)
- **Quality gates**: Automated enforcement via ruff, mypy, pytest, and CI
  - 8 new ruff rule sets: S (bandit), A (builtins), DTZ (datetimez), T20 (print), PT (pytest), PERF (perflint), PIE, ERA (eradicate)
  - Per-file-ignores for intentional patterns (CLI print, simhash MD5, SQL column names, etc.)
  - Coverage threshold: 67% enforced in CI and Makefile
  - CI: typecheck job now fails build (removed `continue-on-error` and `|| true`); build requires `[lint, typecheck, test]`; added security scan job
  - Pre-commit: updated hooks (ruff v0.9.6, mypy v1.15.0); added `no-commit-to-branch` and `bandit`
  - Makefile: added `security`, `audit` targets; `check` now includes `security`

### Changed

- Tests: 1759 passed (up from 1696)

## [1.7.1] - 2026-02-11

### Fixed

- Fix `__version__` reporting "1.6.1" instead of "1.7.0" in PyPI package (runtime version mismatch)

## [1.7.0] - 2026-02-11

### Added

- **Proactive Brain Intelligence** — 3 features that make the brain self-aware during normal usage
  - **Related Memories on Write** — `smem_remember` now discovers and returns up to 3 related existing memories via 2-hop SpreadingActivation from the new anchor neuron. Always-on (~5-10ms overhead), non-intrusive. Response includes `related_memories` list with `fiber_id`, `preview`, and `similarity` score.
  - **Expired Memory Hint** — Health pulse detects expired memories via cheap COUNT query on `typed_memories` table. Surfaces hint when count exceeds threshold (default: 10): `"N expired memories found. Consider cleanup via smem list --expired."`
  - **Stale Fiber Detection** — Health pulse detects fibers with decayed conductivity (last conducted >90 days ago or never). Surfaces hint when stale ratio exceeds threshold (default: 30%): `"N% of fibers are stale. Consider running smem_health for review."`
- **MaintenanceConfig extensions** — 3 new configuration fields:
  - `expired_memory_warn_threshold` (default: 10)
  - `stale_fiber_ratio_threshold` (default: 0.3)
  - `stale_fiber_days` (default: 90)
- **Storage layer** — 2 new optional methods on `NeuralStorage`:
  - `get_expired_memory_count()` — COUNT of expired typed memories (SQLite + InMemory)
  - `get_stale_fiber_count(brain_id, stale_days)` — COUNT of stale fibers (SQLite + InMemory)
- **HealthPulse extensions** — `expired_memory_count` and `stale_fiber_ratio` fields
- **HEALTH_DEGRADATION trigger** — `TriggerType.HEALTH_DEGRADATION` for maintenance events

### Changed

- Tests: 1696 passed (up from 1695)

## [1.6.1] - 2026-02-10

### Fixed

- CLI brain commands (`export`, `import`, `create`, `delete`, `health`, `transplant`) now work correctly in SQLite mode
- `brain export` no longer produces empty files when brain was created with `brain create`
- `brain delete` correctly removes `.db` files in unified config mode
- `brain health` uses storage-agnostic `find_neurons()` instead of JSON-internal `_neurons` dict
- All `version` subcommands (`create`, `list`, `rollback`, `diff`) now find brains in SQLite mode
- `shared sync` uses correct storage backend

## [1.6.0] - 2026-02-10

### Added

- **DB-to-Brain Schema Training (`smem_train_db`)** — Teach brains to understand database structure
  - 3-layer pipeline: `SchemaIntrospector` → `KnowledgeExtractor` → `DBTrainer`
  - Extracts **schema knowledge** (table structures, relationships, patterns) — NOT raw data rows
  - SQLite dialect (v1) via `aiosqlite` read-only connections
  - Schema fingerprint (SHA256) for re-training detection
- **Schema Introspection** — `engine/db_introspector.py`
  - `SchemaDialect` protocol with `SQLiteDialect` implementation
  - Frozen dataclasses: `ColumnInfo`, `ForeignKeyInfo`, `IndexInfo`, `TableInfo`, `SchemaSnapshot`
  - PRAGMA-based metadata extraction (table_info, foreign_key_list, index_list)
- **Knowledge Extraction** — `engine/db_knowledge.py`
  - FK-to-SynapseType mapping with confidence scoring (IS_A, INVOLVES, AT_LOCATION, RELATED_TO)
  - Structure-based join table detection (2+ FKs, ≤1 business column → CO_OCCURS synapse)
  - 5 schema pattern detectors: audit_trail, soft_delete, tree_hierarchy, polymorphic, enum_table
- **Training Orchestrator** — `engine/db_trainer.py`
  - Mirrors DocTrainer architecture: batch save, per-table error isolation, shared domain neuron
  - Configurable: `max_tables` (1-500), `salience_ceiling`, `consolidate`, `domain_tag`
- **MCP Tool: `smem_train_db`** — `train` and `status` actions

### Fixed

- Security: read-only SQLite connections, absolute path rejection, SQL identifier sanitization, info leakage prevention

### Changed

- MCP tools expanded from 17 to 18
- Tests: 1648 passed (up from 1596)

### Skills

- **3 composable AI agent skills** — ship-faster SKILL.md pattern, installable to `~/.claude/skills/`
  - `memory-intake` — structured memory creation from messy notes, 1-question-at-a-time clarification, batch store with preview
  - `memory-audit` — 6-dimension quality review (purity, freshness, coverage, clarity, relevance, structure), A-F grading
  - `memory-evolution` — evidence-based optimization from usage patterns, consolidation, enrichment, pruning, checkpoint Q&A

## [1.5.0] - 2026-02-10

### Added

- **Conflict Management MCP Tool (`smem_conflicts`)** — List, resolve, and pre-check memory conflicts
  - `list`, `resolve` (keep_existing/keep_new/keep_both), `check` actions
  - `ConflictHandler` mixin with full input validation
- **Recall Conflict Surfacing** — `has_conflicts` flag and `conflict_count` in default recall response
- **Provenance Source Enrichment** — `SURREAL_MEMORY_SOURCE` env var → `mcp:{source}` provenance
- **Purity Score Conflict Penalty** — Unresolved CONTRADICTS reduce health score (max -10 points)

### Fixed

- 20+ performance bottlenecks — storage index optimization, encoder batch operations
- 25+ bugs across engine/storage/MCP — deep audit fixes including deprecated `datetime.utcnow()` replacement

### Changed

- MCP tools expanded from 16 to 17
- Tests: 1372 passed (up from 1352)

## [1.4.0] - 2026-02-09

### Added

- **OpenClaw Memory Plugin** — `@surrealmemory/openclaw-plugin` npm package
  - MCP stdio client: JSON-RPC 2.0 with Content-Length framing
  - 6 core tools, 2 hooks (before_agent_start, agent_end), 1 service
  - Plugin manifest with `configSchema` + `uiHints`

### Changed

- Dashboard Integrations tab simplified to status-only with deep links (Option B)

## [1.3.0] - 2026-02-09

### Added

- **Deep Integration Status** — Enhanced status cards, activity log, setup wizards, import sources
- **Source Attribution** — `SURREAL_MEMORY_SOURCE` env var for integration tracking
- 25 new i18n keys in EN + VI (87 total)

### Changed

- Tests: 1352 passed (up from 1340)

## [1.2.0] - 2026-02-09

### Added

- **Dashboard** — Full-featured SPA at `/dashboard` (Alpine.js + Tailwind CDN, zero-build)
  - 5 tabs: Overview, Neural Graph (Cytoscape.js), Integrations, Health (radar chart), Settings
  - Graph toolbar, toast notifications, skeleton loading, brain management, EN/VI i18n
  - ARIA accessibility, 44px mobile touch targets, design system

### Fixed

- `ModuleNotFoundError: typing_extensions` on fresh Python 3.12 — added dependency

### Changed

- Tests: 1340 passed (up from 1264)

## [1.1.0] - 2026-02-09

### Added

- **ClawHub SKILL.md** — Published `surreal-memory@1.0.0` to ClawHub
- **Nanobot Integration** — 4 tools adapted for Nanobot's action interface
- **Architecture Doc** — `docs/ARCHITECTURE_V1_EXTENDED.md`

### Changed

- OpenClaw PR [#12596](https://github.com/openclaw/openclaw/pull/12596) submitted

## [1.0.2] - 2026-02-09

### Fixed

- Empty recall for broad queries — `format_context()` truncates long fiber content to fit token budget
- Diversity metric normalization — Shannon entropy normalized against 8 expected synapse types
- Temporal synapse diversity — `_link_temporal_neighbors()` creates BEFORE/AFTER instead of always RELATED_TO
- Consolidation prune crash — Fixed `Fiber(tags=...)` TypeError, uses `dataclasses.replace()`

## [1.0.0] - 2026-02-09

### Added

- **Brain Versioning** — Snapshot, rollback, diff (schema v11, `brain_versions` table)
- **Partial Brain Transplant** — Topic-filtered merge between brains with conflict resolution
- **Brain Quality Badge** — Grade A-F from BrainHealthReport, marketplace eligibility
- **Optional Embedding Layer** — SentenceTransformer + OpenAI providers (OFF by default)
- **Optional LLM Extraction** — Enhanced relation extraction beyond regex (OFF by default)

### Changed

- Version 1.0.0 — Production/Stable, schema v10 → v11
- MCP tools expanded from 14 to 16 (smem_version, smem_transplant)

## [0.20.0] - 2026-02-09

### Added

- **Habitual Recall** — ENRICH, DREAM, LEARN_HABITS consolidation strategies
  - Action event log (hippocampal buffer), sequence mining, workflow suggestions
  - `smem_habits` MCP tool, `smem habits` CLI, `smem update` CLI
  - Prune enhancements: dream synapse 10x decay, high-salience resistance
- Schema v10: `action_events` table
- 6 new BrainConfig fields for habit/dream configuration

### Changed

- `ConsolidationStrategy` extended with ENRICH, DREAM, LEARN_HABITS
- Schema version 9 → 10

## [0.19.0] - 2026-02-08

### Added

- **Temporal Reasoning** — Causal chain traversal, temporal range queries, event sequence tracing
  - `trace_causal_chain()`, `query_temporal_range()`, `trace_event_sequence()`
  - `CAUSAL_CHAIN` and `TEMPORAL_SEQUENCE` synthesis methods
  - Pipeline integration: "Why?" → causal, "When?" → temporal, "What happened after?" → event sequence
  - Router enhancement with traversal metadata in `RouteDecision`

### Changed

- Tests: 1019 passed (up from 987)

## [0.17.0] - 2026-02-08

### Added

- **Brain Diagnostics** — `BrainHealthReport` with 7 component scores and composite purity (0-100)
  - Grade A/B/C/D/F, 7 warning codes, automatic recommendations
  - Tag drift detection via `TagNormalizer.detect_drift()`
- **MCP tool: `smem_health`** — Brain health diagnostics
- **CLI command: `smem health`** — Terminal health report with ASCII progress bars

## [0.16.0] - 2026-02-08

### Added

- **Emotional Valence** — Lexicon-based sentiment extraction (EN + VI, zero LLM)
  - `SentimentExtractor`, `Valence` enum, 7 emotion tag categories
  - Negation handling, intensifier detection
  - `FELT` synapses from anchor → emotion STATE neurons
- **Emotional Resonance Scoring** — Up to +0.1 retrieval boost for matching-valence memories
- **Emotional Decay Modulation** — High-intensity emotions decay slower (trauma persistence)

### Changed

- Tests: 950 passed (up from 908)

## [0.15.0] - 2026-02-08

### Added

- **Associative Inference Engine** — Co-activation patterns → persistent CO_OCCURS synapses
  - `compute_inferred_weight()`, `identify_candidates()`, `create_inferred_synapse()`
  - `generate_associative_tags()` from BFS clustering
- **Co-Activation Persistence** — `co_activation_events` table (schema v8 → v9)
  - `record_co_activation()`, `get_co_activation_counts()`, `prune_co_activations()`
- **INFER Consolidation Strategy** — Create synapses from co-activation patterns
- **Tag Normalizer** — ~25 synonym groups + SimHash fuzzy matching + drift detection
- 6 new BrainConfig fields for co-activation configuration

### Changed

- Schema version 8 → 9
- Tests: 908 passed (up from 838)

## [0.14.0] - 2026-02-08

### Added

- **Relation extraction engine**: Regex-based causal, comparative, and sequential pattern detection from content — auto-creates CAUSED_BY, LEADS_TO, BEFORE, SIMILAR_TO, CONTRADICTS synapses during encoding
- **Tag origin tracking**: Separate `auto_tags` (content-derived) from `agent_tags` (user-provided) with backward-compatible `fiber.tags` union property
- **Auto memory type inference**: `suggest_memory_type()` fallback when no explicit type provided at encode time
- **Confirmatory weight boost**: Hebbian +0.1 boost on anchor synapses when agent tags confirm auto tags; RELATED_TO synapses (weight 0.3) for divergent agent tags
- **Bilingual pattern support**: English + Vietnamese regex patterns for causal ("because"/"vì"), comparative ("similar to"), and sequential ("then"/"sau khi") relations
- `RelationType`, `RelationCandidate`, `RelationExtractor` in new `extraction/relations.py`
- `Fiber.auto_tags`, `Fiber.agent_tags` fields with `Fiber.add_auto_tags()` method
- SQLite schema migration v7→v8 with backward-compatible column additions and backfill
- 62 new tests: relation extraction (25), tag origin (10), confirmatory boost (5), relation encoding (7), auto-tags update (15)
- `ROADMAP.md` with versioned plan from v0.14.0 → v1.0.0

### Fixed

- **"Event loop is closed" noise on CLI exit**: aiosqlite connections now properly closed before event loop teardown via centralized `run_async()` helper
- MCP server shutdown now closes storage connection in `finally` block

### Changed

- All 32 CLI `asyncio.run()` calls replaced with `run_async()` for proper cleanup
- Encoder pipeline extended with relation extraction (step 6b) and confirmatory boost (step 6c)
- `Fiber.create(tags=...)` preserved for backward compat — maps to `agent_tags`
- 838 tests passing

## [0.13.0] - 2026-02-07

### Added

- **Ground truth evaluation dataset**: 30 curated memories across 5 sessions (Day 1→Day 30) covering project setup, development, integration, sprint review, and production launch
- **Standard IR metrics**: Precision@K, Recall@K, MRR (Mean Reciprocal Rank), NDCG@K with per-query and per-category aggregation
- **25 evaluation queries**: 8 factual, 6 temporal, 4 causal, 4 pattern, 3 multi-session coherence queries with expected relevant results
- **Naive keyword-overlap baseline**: Tokenize-and-rank strawman that Surreal-Memory's activation-based recall must beat
- **Long-horizon coherence test framework**: 5-session simulation across 30 days with recall tracking per session (target: >= 60% at day 30)
- `benchmarks/ground_truth.py` — ground truth memories, queries, session schedule
- `benchmarks/metrics.py` — IR metrics: `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `evaluate_query`, `BenchmarkReport`
- `benchmarks/naive_baseline.py` — keyword overlap ranking and baseline evaluation
- `benchmarks/coherence_test.py` — multi-session coherence test with `CoherenceReport`
- Ground-truth evaluation section in `run_benchmarks.py` comparing Surreal-Memory vs baseline
- 27 new unit tests: precision (6), recall (4), MRR (5), NDCG (4), query evaluation (1), report aggregation (2), baseline (5)

### Changed

- `run_benchmarks.py` now includes ground-truth evaluation with Surreal-Memory vs naive baseline comparison in generated markdown output

## [0.12.0] - 2026-02-07

### Added

- **Real-time conflict detection**: Detects factual contradictions and decision reversals at encode time using predicate extraction — no LLM required
- **Factual contradiction detection**: Regex-based extraction of `"X uses/chose/decided Y"` patterns, compares predicates across memories with matching subjects
- **Decision reversal detection**: Identifies when a new DECISION contradicts an existing one via tag overlap analysis
- **Dispute resolution pipeline**: Anti-Hebbian confidence reduction, `_disputed` and `_superseded` metadata markers, and CONTRADICTS synapse creation
- **Disputed neuron deprioritization**: Retrieval pipeline reduces activation of disputed neurons by 50% and superseded neurons by 75%
- `CONTRADICTS` synapse type for linking contradictory memories
- `ConflictType`, `Conflict`, `ConflictResolution`, `ConflictReport` in new `engine/conflict_detection.py`
- `detect_conflicts()`, `resolve_conflicts()` for encode-time conflict handling
- 32 new unit tests: predicate extraction (5), predicate conflict (4), subject matching (4), tag overlap (4), helpers (4), detection integration (6), resolution (5)

### Changed

- Encoder pipeline runs conflict detection after anchor neuron creation, before fiber assembly
- Retrieval pipeline adds `_deprioritize_disputed()` step after stabilization to suppress disputed neurons
- `SynapseType` enum extended with `CONTRADICTS = "contradicts"`

## [0.11.0] - 2026-02-07

### Added

- **Activation stabilization**: Iterative dampening algorithm settles neural activations into stable patterns after spreading activation — noise floor removal, dampening (0.85x), homeostatic normalization, convergence detection (typically 2-4 iterations)
- **Multi-neuron answer reconstruction**: Strategy-based answer synthesis replacing single-neuron `reconstitute_answer()` — SINGLE mode (high-confidence top neuron), FIBER_SUMMARY mode (best fiber summary), MULTI_NEURON mode (top-5 neurons ordered by fiber pathway position)
- **Memory maturation lifecycle**: Four-stage memory model STM → Working (30min) → Episodic (4h) → Semantic (7d + spacing effect). Stage-aware decay multipliers: STM 5x, Working 2x, Episodic 1x, Semantic 0.3x
- **Spacing effect requirement**: EPISODIC → SEMANTIC promotion requires reinforcement across 3+ distinct calendar days, modeling biological spaced repetition
- **Pattern extraction**: Episodic → semantic concept formation via tag Jaccard clustering (Union-Find). Clusters of 3+ similar fibers generate CONCEPT neurons with IS_A synapses to common entities
- **MATURE consolidation strategy**: New consolidation strategy that advances maturation stages and extracts semantic patterns from mature episodic memories
- `StabilizationConfig`, `StabilizationReport`, `stabilize()` in new `engine/stabilization.py`
- `SynthesisMethod`, `ReconstructionResult`, `reconstruct_answer()` in new `engine/reconstruction.py`
- `MemoryStage`, `MaturationRecord`, `compute_stage_transition()`, `get_decay_multiplier()` in new `engine/memory_stages.py`
- `ExtractedPattern`, `ExtractionReport`, `extract_patterns()` in new `engine/pattern_extraction.py`
- `SQLiteMaturationMixin` in new `storage/sqlite_maturation.py` — maturation CRUD for SQLite backend
- Schema migration v6→v7: `memory_maturations` table with composite key (brain_id, fiber_id)
- `contributing_neurons` and `synthesis_method` fields on `RetrievalResult`
- `stages_advanced` and `patterns_extracted` fields on `ConsolidationReport`
- Maturation abstract methods on `NeuralStorage` base: `save_maturation()`, `get_maturation()`, `find_maturations()`
- 49 new unit tests: stabilization (12), reconstruction (11), memory stages (16), pattern extraction (8), plus 2 consolidation tests

### Changed

- Retrieval pipeline inserts stabilization phase after lateral inhibition and before answer reconstruction
- Answer reconstruction uses multi-strategy `reconstruct_answer()` instead of `reconstitute_answer()`
- Encoder initializes maturation record (STM stage) when creating new fibers
- Consolidation engine supports `MATURE` strategy for stage advancement and pattern extraction

## [0.10.0] - 2026-02-07

### Added

- **Formal Hebbian learning rule**: Principled weight update `Δw = η_eff * pre * post * (w_max - w)` replacing ad-hoc `weight += delta + dormancy_bonus`
- **Novelty-adaptive learning rate**: New synapses learn ~4x faster, frequently reinforced synapses stabilize toward base rate via exponential decay
- **Natural weight saturation**: `(w_max - w)` term prevents runaway weight growth — weights near ceiling barely change
- **Competitive normalization**: `normalize_outgoing_weights()` caps total outgoing weight per neuron at budget (default 5.0), implementing winner-take-most competition
- **Anti-Hebbian update**: `anti_hebbian_update()` for conflict resolution weight reduction (used in Phase 3)
- `learning_rate`, `weight_normalization_budget`, `novelty_boost_max`, `novelty_decay_rate` on `BrainConfig`
- `LearningConfig`, `WeightUpdate`, `hebbian_update`, `compute_effective_rate`, `normalize_outgoing_weights` in new `engine/learning_rule.py`
- 33 new unit tests covering learning rule, normalization, and backward compatibility

### Changed

- `Synapse.reinforce()` accepts optional `pre_activation`, `post_activation`, `now` parameters — uses formal Hebbian rule when activations provided, falls back to direct delta for backward compatibility
- `ReflexPipeline._defer_co_activated()` passes neuron activation levels to Hebbian strengthening
- `ReflexPipeline._defer_reinforce_or_create()` forwards activation levels to `reinforce()`
- Removed dormancy bonus from `Synapse.reinforce()` (novelty adaptation in learning rule replaces it)

## [0.9.6] - 2026-02-07

### Added

- **Sigmoid activation function**: Neurons now use sigmoid gating (`1/(1+e^(-6(x-0.5)))`) instead of raw clamping, producing bio-realistic nonlinear activation curves
- **Firing threshold**: Neurons only propagate signals when activation meets threshold (default 0.3), filtering borderline noise
- **Refractory period**: Cooldown prevents same neuron firing twice within a query pipeline (default 500ms), checked during spreading activation
- **Lateral inhibition**: Top-K winner-take-most competition in retrieval pipeline — top 10 neurons survive unchanged, rest suppressed by 0.7x factor
- **Homeostatic target field**: Reserved `homeostatic_target` field on NeuronState for v2 adaptive regulation
- `fired` and `in_refractory` properties on `NeuronState`
- `sigmoid_steepness`, `default_firing_threshold`, `default_refractory_ms`, `lateral_inhibition_k`, `lateral_inhibition_factor` on `BrainConfig`
- Schema migration v5→v6: four new columns on `neuron_states` table

### Changed

- `NeuronState.activate()` applies sigmoid function and accepts `now` and `sigmoid_steepness` parameters
- `NeuronState.decay()` preserves all new fields (firing_threshold, refractory_until, refractory_period_ms, homeostatic_target)
- `DecayManager.apply_decay()` uses `state.decay()` instead of manual NeuronState construction
- `ReinforcementManager.reinforce()` directly sets activation level (bypasses sigmoid for reinforcement)
- Spreading activation skips neurons in refractory cooldown
- Storage layer (SQLite + SharedStore) serializes/deserializes all new NeuronState fields

## [0.9.5] - 2026-02-07

### Added

- **Type-aware decay rates**: Different memory types now decay at biologically-inspired rates (facts: 0.02/day, todos: 0.15/day). `DEFAULT_DECAY_RATES` dict and `get_decay_rate()` helper in `memory_types.py`
- **Retrieval score breakdown**: `ScoreBreakdown` dataclass exposes confidence components (base_activation, intersection_boost, freshness_boost, frequency_boost) in `RetrievalResult` and MCP `smem_recall` response
- **SimHash near-duplicate detection**: 64-bit locality-sensitive hashing via `utils/simhash.py`. New `content_hash` field on `Neuron` model. Encoder and auto-capture use SimHash to catch paraphrased duplicates
- **Point-in-time temporal queries**: `valid_at` parameter on `smem_recall` filters fibers by temporal validity window (`time_start <= valid_at <= time_end`)
- Schema migration v4→v5: `content_hash INTEGER` column on neurons table

### Changed

- `DecayManager.apply_decay()` now uses per-neuron `state.decay_rate` instead of global rate
- `reconstitute_answer()` returns `ScoreBreakdown` as third tuple element
- `_remember()` MCP handler sets type-specific decay rates on neuron states after encoding

## [0.9.4] - 2026-02-07

### Performance

- **SQLite WAL mode** + `synchronous=NORMAL` + 8MB cache for concurrent reads and reduced I/O
- **Batch storage methods**: `get_synapses_for_neurons()`, `find_fibers_batch()`, `get_neuron_states_batch()` — single `IN()` queries replacing N sequential calls
- **Deferred write queue**: Fiber conductivity, Hebbian strengthening, and synapse writes batched after response assembly
- **Parallel anchor finding**: Entity + keyword lookups via `asyncio.gather()` instead of sequential loops
- **Batch fiber discovery**: Single junction-table query replaces 5-15 sequential `find_fibers()` calls
- **Batch subgraph extraction**: Single query replaces 20-50 sequential `get_synapses()` calls
- **BFS state prefetch**: Batch `get_neuron_states_batch()` per hop instead of individual lookups
- Target: 3-5x faster retrieval (800-4500ms → 200-800ms)

## [0.9.0] - 2026-02-06

### Added

- **Codebase indexing** (`smem_index`): Index Python files into neural graph for code-aware recall
- **Python AST extractor**: Parse functions, classes, methods, imports, constants via stdlib `ast`
- **Codebase encoder**: Map code symbols to neurons (SPATIAL/ACTION/CONCEPT/ENTITY) and synapses (CONTAINS/IS_A/RELATED_TO/CO_OCCURS)
- **Branch-aware sessions**: `smem_session` auto-detects git branch/commit/repo and stores in metadata + tags
- **Git context utility**: Detect branch, commit SHA, repo root via subprocess (zero deps)
- **CLI `smem index` command**: Index codebase from command line with `--ext`, `--status`, `--json` options
- 16 new tests for extraction, encoding, and git context

## [0.8.0]

### Added

- Initial project structure
- Core data models: Neuron, Synapse, Fiber, Brain
- In-memory storage backend using NetworkX
- Temporal extraction for Vietnamese and English
- Query parser with stimulus decomposition
- Spreading activation algorithm
- Reflex retrieval pipeline
- Memory encoder
- FastAPI server with memory and brain endpoints
- Unit and integration tests
- Docker support

## [0.1.0] - TBD

### Added

- First public release
- Core memory encoding and retrieval
- Multi-language support (English, Vietnamese)
- REST API server
- Brain export/import functionality
