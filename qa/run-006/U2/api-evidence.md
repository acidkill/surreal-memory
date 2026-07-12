# U2 Trust/Recency MCP-Surface — API QA Evidence (run-006)

**Skill:** `tonis-api-tester` (credential-real, contract-true, negative-proven)
**Date:** 2026-07-12
**Worktree:** `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
**Branch:** `feature/v290-pr2-trust-recency`  (HEAD `d970e6c` — "feat: surface trust/recency in smem_recall … + smem_source trust param")
**Backend under test:** **live SurrealDB** — `surreal_memory.storage.surrealdb.store.SurrealDBStorage` (NS `surreal_memory`, DB `default`, `http://localhost:8001`, health `200`)
**Brain:** dedicated QA brain `u2qarun006` (isolated; `my-brain.v2` untouched)

## How the surface was exercised (no mocks, worktree code only)

Per ENV RULES (no `uv run`/`uv sync`), everything ran via the pre-synced worktree venv
`.venv/bin/python`. Verified the venv loads worktree code, not the globally-installed smem:

```
$ .venv/bin/python -c "import surreal_memory; print(surreal_memory.__file__)"
/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006/src/surreal_memory/__init__.py
```

The **real MCP handler methods** that `MCPServer` dispatches the tools to were driven
in-process against the live backend (same code path as a JSON-RPC tool call, minus transport):

- `smem_source`  → `server._source(args)`   (`mcp/server.py:270`)
- `smem_recall`  → `server._recall(args)`   (`mcp/server.py:232`)
- `smem_remember`→ `server._remember(args)` (`mcp/server.py:230`)

Backend forced with `SURREAL_MEMORY_STORAGE=surrealdb` (on-disk `config.toml` says `sqlite`;
env overrides TOML). Harness asserts `type(storage).__name__ == "SurrealDBStorage"` before any
assertion — a SQLite fallback would abort as BLOCKED to avoid a false pass. Harness scripts:
`u2_harness2.py` (main pass), `u2_debug3.py`/`u2_debug4.py` (root-cause). Raw stdout saved
alongside this file (`harness2-output.txt`, `harness-output.txt`).

---

## Verdict summary

| # | Item | Verdict |
|---|------|---------|
| 1 | `smem_source` `trust` param (register/get/list/update + [0,1] reject) | **PASS** |
| 2 | `smem_recall` `score_breakdown.trust_factor` / `recency_factor` keys | **PASS** (stated contract) — see CAVEAT |
| 3 | `smem_recall` `sources[fid].trust` after linking a memory to a source | **FAIL** |

Items 2-caveat and 3 share **one root cause**: an id-normalization mismatch on the SurrealDB
backend (details at bottom).

---

## ITEM 1 — `smem_source` trust param — **PASS**

Real `_source` handler responses (live SurrealDB):

**1a. register trust=0.8 → response echoes trust**
```json
{"source_id":"dd471a60-a43f-4016-802c-dbb87172a823","name":"src-u2mark1783819288",
 "source_type":"document","status":"active","trust":0.8}
```

**1b. get → "trust": 0.8**
```json
{"source_id":"source:dd471a60_a43f_4016_802c_dbb87172a823","name":"src-u2mark1783819288",
 "source_type":"document","version":"","status":"active","trust":0.8,"file_hash":"",
 "metadata":{},"linked_neuron_count":0,"created_at":"2026-07-12T01:21:28.452223",...}
```

**1c. list → includes trust: 0.8** (matched by name to our source)
```json
[{"source_id":"source:dd471a60_a43f_4016_802c_dbb87172a823","name":"src-u2mark1783819288",
  "source_type":"document","version":"","status":"active","trust":0.8,
  "created_at":"2026-07-12T01:21:28.452223"}]
```

**1d/1e. update trust=0.5 → get shows 0.5**
```json
// update
{"updated":true,"source_id":"dd471a60-a43f-4016-802c-dbb87172a823"}
// get after update
{"source_id":"source:dd471a60_a43f_4016_802c_dbb87172a823","status":"active","trust":0.5,...}
```

**1f. register trust=1.5 → rejected with [0,1] range error** (negative path proven, A4)
```json
{"error":"trust must be in [0.0, 1.0]"}
```

All five contract points hold. (Confirms UB1 fix — `get_source`/`update` work on SurrealDB;
`update_source` accepts `trust`.)

---

## ITEM 2 — `smem_recall` score_breakdown factors — **PASS (stated contract)**

Recall of a real, matching query (`"calibration marker u2mark… provenance verification protocol"`)
that activates 8 neurons and matches fiber `8ec1e4d2_752d_…`.

**2a. trust_weight = 0.0 (brain defaults) → factors default to 1.0** (real `_recall` output):
```json
"score_breakdown":{"base_activation":1.0,"intersection_boost":0.3,"freshness_boost":0.0,
  "frequency_boost":0.0,"trust_factor":1.0,"recency_factor":1.0}
```

**2b. trust_weight = 0.5 (brain re-configured) → trust_factor active (≠ 1.0)**:
```json
"score_breakdown":{"base_activation":1.0,"intersection_boost":0.3,"freshness_boost":0.1321,
  "frequency_boost":0.0208,"trust_factor":0.85,"recency_factor":1.0}
```

`trust_factor` and `recency_factor` keys are present and float in both cases; they are `1.0`
at defaults and `trust_factor` becomes `0.85` (calibration active) when `trust_weight=0.5`.
**Stated contract: PASS.**

**How trust_weight is set (per the diff's design):** it is a `BrainConfig` field
(`core/brain.py:158` `trust_weight: float = 0.0`, also `recency_weight`=1.0, `trust_default`=0.7),
TOML-loadable via the `[brain]` section and read per-recall from `brain.config`
(`recall_handler.py:340` `ReflexPipeline(storage, brain.config)`). In this run it was set by
persisting the QA brain with `Brain.with_config(BrainConfig(trust_weight=0.5))` via
`storage.save_brain(...)` — reachable through the storage API; there is no dedicated MCP tool
to mutate a single BrainConfig field.

**⚠️ CAVEAT (secondary finding, not a contract failure):** `trust_factor=0.85` = `(1-0.5) + 0.5·0.7`,
i.e. it reflects **`trust_default` (0.7)**, NOT the linked source's real trust **(0.9)**.
`_build_trust_map` (`retrieval.py:1898`) resolves **nothing** on SurrealDB — every fiber falls
back to `trust_default` — for the same id-normalization reason that breaks Item 3 (proof below).
So trust weighting is *surfaced and active* but is currently *inert with respect to real
per-source trust* on the SurrealDB backend.

---

## ITEM 3 — `smem_recall` `sources[fid].trust` — **FAIL**

Setup via real handlers: registered source `linked-u2mark…` with `trust=0.9`, then
`smem_remember` linked to it (`source_id` arg). Remember succeeded and returned the linkage:
```json
{"success":true,"fiber_id":"8ec1e4d2-752d-46cc-bfd6-f0375b181848","source_id":"25cdf241-ca3d-47a5-be3f-d2aa3977c2ad", ...}
```

Recall of the matching query returned the fiber in `fibers_matched` **but an empty `sources` map** —
the U2 `trust` field is never emitted. Real `_recall` output (trimmed):
```json
{"confidence":1.0,"neurons_activated":8,
 "fibers_matched":["8ec1e4d2_752d_46cc_bfd6_f0375b181848","bd2de1c7_573e_4562_950a_41c72bed78b8"],
 "score_breakdown":{...},
 "sources": {}   //  <-- absent from response; no fid entry, no "trust"
}
```
(The full `_recall` response has **no** `sources` key at all — enrichment produced an empty map,
so the `if source_map:` guard at `recall_handler.py:717` never assigns `response["sources"]`.)

**The U2 line itself is correct** — the failure is that the enclosing enrichment never runs. Proof
that the link and trust are retrievable when the correct id form is used:
```
get_typed_memory('8ec1e4d2-752d-46cc-bfd6-f0375b181848')  # DASH form
   -> tm.source = 'source:25cdf241-ca3d-47a5-be3f-d2aa3977c2ad'
get_source('25cdf241-ca3d-47a5-be3f-d2aa3977c2ad')  -> trust = 0.9
```
With the dash-form fid, `sources[fid]` would be `{... , "trust": 0.9}`.

---

## Root cause (shared by Item 3 and the Item 2 caveat)

**ID-normalization mismatch on the SurrealDB backend.** Fiber ids surfaced by retrieval are
**underscore-sanitized** (SurrealDB record-id form), but typed-memory lookups match on the stored
**dash-uuid** `fiber_id` field.

Evidence (`u2_debug3.py` / `u2_debug4.py`, live SurrealDB):
```
get_typed_memory('8ec1e4d2-752d-46cc-bfd6-f0375b181848')  -> TypedMemory(... source='source:25cdf241-...')
get_typed_memory('8ec1e4d2_752d_46cc_bfd6_f0375b181848')  -> None

get_typed_memories_batch(['8ec1e4d2-752d-...'])  -> resolves (tm.source present)
get_typed_memories_batch(['8ec1e4d2_752d_...'])  -> {} (empty)

get_fiber('8ec1e4d2-752d-...').id  -> '8ec1e4d2_752d_46cc_bfd6_f0375b181848'   # Fiber carries UNDERSCORE id
```

- `get_typed_memory` matches `WHERE fiber_id = $fiber_id` on the stored field, which is the
  original **dash** uuid (`storage/surrealdb/typed_memory.py:167`).
- `result.fibers_matched` and `Fiber.id` are **underscore**-sanitized record ids.

Consequences:
1. **Item 3**: `recall_handler.py:703` loops `for fid in result.fibers_matched:` (underscore) and
   calls `get_typed_memory(fid)` → `None` for every fiber → the `sources` map stays empty → the
   U2 `"trust": src.trust` addition (`recall_handler.py:711`) is unreachable.
2. **Item 2 caveat**: `_build_trust_map` uses `fiber_ids = [f.id for f in fibers]` (underscore) →
   `get_typed_memories_batch(...)` returns `{}` → `resolve_effective_trust(None, None, trust_default)`
   → `trust_default` for every fiber → `trust_factor` reflects 0.7, never the real source 0.9.

This is a SurrealDB-backend defect. It is **pre-existing** in the source-enrichment path (U2 only
added the `trust` line inside it), but it makes the U2 Item 3 contract unobservable on live
SurrealDB, and it renders the Item 2 trust calibration inert w.r.t. real source trust.

## Suggested fix direction (not applied — QA only)

Normalize fid consistently before typed-memory lookups: either (a) have `get_typed_memory` /
`get_typed_memories_batch` match the sanitized record-id form (query the record id, or sanitize
the `fiber_id` filter the same way ids are sanitized on write), or (b) resolve fibers_matched /
`Fiber.id` back to the original dash `fiber_id` before enrichment and trust-map building. A live
regression test should assert `response["sources"][fid]["trust"]` is present after a source-linked
`smem_remember` + `smem_recall` on the SurrealDB backend.
```
