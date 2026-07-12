# U7 Battery 4 — "Spectron-demo" (Emma Oslo→Bergen) — Live SurrealDB Evidence (RE-RUN, post-fix)

- Repo/worktree: `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
- Branch: `feature/v290-release-prep` (HEAD `2ae9514` — `fix(recall): valid_at point-in-time was
  impossible for facts without event-time (U3 bug, found in U7 Spectron-demo)`)
- Live DB: `SURREALDB_URL=http://localhost:8001`, SurrealDB **3.2.0** (`curl -s
  http://localhost:8001/version` → `surrealdb-3.2.0`)
- Harness: real in-process `MCPServer()` from `surreal_memory.mcp.server` (THIS worktree's code, i.e. the
  code AFTER the fix), backed by the real `SurrealDBStorage` (`SURREAL_MEMORY_STORAGE=surrealdb`). No
  mocks anywhere in this run.
- Throwaway brain: `qa-u7b-01cb3df6`, set via `SURREAL_MEMORY_BRAIN` + `get_config(reload=True)` before
  `MCPServer()` construction. Fully deleted at the end (see Cleanup).
- Cosmetic `Schema statement failed: ... FLEXIBLE ... SCHEMAFULL` warnings on `storage.initialize()` —
  ignored per instructions (pre-existing, unrelated to this fix).
- `.venv/bin/python` only, one throwaway async script piped via `.venv/bin/python script.py` (no
  `uv`/`uv run`, no full pytest suite run).

## The fix under test

`src/surreal_memory/engine/retrieval.py::_fiber_valid_at` (lines 75-97): a fiber's event-time window
`(time_start, time_end)` is now treated as **unbounded** whenever `time_start == time_end` (a zero-width
window). Previously this pipeline-level physical-time filter required `dt == fiber.time_start` exactly
whenever a fact was written without an explicit `event_at`, which structurally could never agree with the
recall-handler's own logical-validity filter (`TypedMemory.is_valid_at`, keyed off `typed_memory.created_at`
— a later, independent clock) — see prior-run root cause below. Now only fibers with a REAL interval
(`time_start < time_end`) are filtered on physical event-time at the pipeline level; supersession
point-in-time recall is governed purely by `TypedMemory.is_valid_at()`.

## Step 1 — `smem_remember` "Emma lives in Oslo" (fact)

```python
r1 = await server.call_tool("smem_remember", {"content": "Emma lives in Oslo", "type": "fact"})
```

```json
{
  "success": true,
  "fiber_id": "fb04c906-75a3-44b4-9d75-e649f65d3aad",
  "memory_type": "fact",
  "tier": "warm",
  "neurons_created": 5,
  "message": "Remembered: Emma lives in Oslo"
}
```

`typed_memory.created_at` for the Oslo fiber (via `storage.get_typed_memory`): **`2026-07-12T05:43:05.142292`**.

**PASS.**

## Step 2 — `smem_remember` "Emma moved to Bergen" (fact) + supersession

```python
r2 = await server.call_tool("smem_remember", {"content": "Emma moved to Bergen", "type": "fact"})
```

```json
{
  "success": true,
  "fiber_id": "605f7192-f38d-49dd-97b4-f9fa9b7386db",
  "neurons_created": 4,
  "message": "Remembered: Emma moved to Bergen",
  "related_memories": [{"fiber_id": "fb04c906_...", "preview": "Emma lives in Oslo", "similarity": 0.2}]
}
```

No `conflicts_detected` key in the response (confirmed again: `"lives in"`/`"moved to"` do not match the
3 `_PREDICATE_PATTERNS` regexes in `engine/conflict_detection.py`, same known detector-coverage gap as the
prior run — unrelated to this fix).

**Supersession path used: MANUAL (`engine.supersession.supersede_typed_memory` called directly)**, per the
task's contingency for phrasing the auto-detector doesn't recognise:

```python
outcome = await supersede_typed_memory(
    storage, old_fiber_id=oslo_fiber_id, new_fiber_id=bergen_fiber_id,
    new_anchor_id=bergen_anchor, old_anchor_id=oslo_anchor,
    reason="manual:qa-u7b-spectron-rerun",
)
# SupersessionOutcome(old_fiber_id='fb04c906-...', new_fiber_id='605f7192-...', superseded=True)
```

Post-call, Oslo's `typed_memory` now carries `valid_until = "2026-07-12T05:43:05.347439"` and
`superseded_by = "605f7192-f38d-49dd-97b4-f9fa9b7386db"` — confirmed via direct storage read.

**PASS.**

## Step 3 — `smem_recall` "where does Emma live" (default) → Bergen, Oslo hard-filtered

```json
{
  "answer": "## Relevant Memories\n\n- Emma moved to Bergen\n\n## Related Information\n\n- [concept] emma lives\n- [concept] moved\n- [concept] emma moved\n- [concept] Emma lives in Oslo\n- [concept] emma\n- [concept] Emma moved to Bergen",
  "fibers_matched": ["605f7192_f38d_49dd_97b4_f9fa9b7386db"],
  "superseded_excluded_count": 1
}
```

`fibers_matched` contains **only** Bergen; `superseded_excluded_count: 1` proves the U3 hard-filter dropped
Oslo. Same behaviour as the pre-fix run (this default path was never broken).

**PASS.**

## Step 4 — `smem_recall` "where does Emma live" with `valid_at` in the valid window → **NOW PASSES**

Oslo's supersession window in this run: `valid_from = created_at = 2026-07-12T05:43:05.142292`,
`valid_until = 2026-07-12T05:43:05.347439` (≈205 ms wide — set by how fast the two `smem_remember` +
`supersede_typed_memory` calls executed in-process). Three `valid_at` values were tried:

### 4a. `valid_at = Oslo.created_at` exactly (`2026-07-12T05:43:05.142292`)

```json
{
  "answer": "## Relevant Memories\n\n- Emma lives in Oslo\n\n## Related Information\n\n- [concept] emma lives\n- [concept] moved\n- [concept] Emma lives in Oslo\n- [concept] emma moved\n- [concept] emma\n- [concept] Emma",
  "fibers_matched": ["fb04c906_75a3_44b4_9d75_e649f65d3aad"],
  "superseded_excluded_count": 1
}
```

`fibers_matched` = **Oslo only**, and Oslo is the primary answer ("Emma lives in Oslo"). **This is the
step that previously returned `fibers_matched: []` with no `superseded_excluded_count` key at all** — the
pipeline-level physical-time filter (`_fiber_valid_at`) no longer excludes the zero-width fiber, so the
candidate reaches the recall-handler's logical `TypedMemory.is_valid_at()` check, which correctly admits
it (`dt == valid_from` is valid; `dt < valid_until`).

### 4b. `valid_at` = midpoint of the Oslo validity window (`2026-07-12T05:43:05.244866`)

```json
{
  "answer": "## Relevant Memories\n\n- Emma lives in Oslo\n\n...",
  "fibers_matched": ["fb04c906_75a3_44b4_9d75_e649f65d3aad"],
  "superseded_excluded_count": 1
}
```

Same result — Oslo returned. Confirms the fix is not a one-instant fluke; any `valid_at` strictly inside
`[valid_from, valid_until)` now resolves to Oslo.

### 4c. `valid_at = Oslo.created_at + 1s` (`2026-07-12T05:43:06.142292`)

```json
{
  "answer": "## Relevant Memories\n\n- Emma moved to Bergen\n\n...",
  "fibers_matched": ["605f7192_f38d_49dd_97b4_f9fa9b7386db"],
  "superseded_excluded_count": 1
}
```

Returns **Bergen**, not Oslo. This is **correct, not a regression**: because the whole Oslo→Bergen
supersession completed in ~205 ms of real wall-clock time in this in-process script, `+1s` from Oslo's
`created_at` lands strictly *after* `valid_until` (`05:43:05.347439`) — i.e. outside Oslo's validity
window and into "current" (Bergen) territory. `TypedMemory.is_valid_at()`'s exclusive upper bound
(`dt >= valid_until → invalid`) is doing exactly what it should here. This result additionally proves the
fix is *not* over-permissive — it doesn't turn `valid_at` into a no-op that always returns the oldest fact;
the boundary is still enforced precisely once outside the window.

**VERDICT for step 4: PASS.** `valid_at` values genuinely inside the Oslo→Bergen validity window (4a, 4b)
return Oslo, as the point-in-time recall contract promises; a value past the window (4c) correctly returns
Bergen instead of leaking Oslo. The previously-reported architectural deadlock (physical-time filter vs.
logical-time filter structurally unsatisfiable together) is resolved — the two filters no longer fight
each other on facts without an explicit `event_at`.

## Step 5 — `smem_recall` "where does Emma live" with `include_superseded: true` → both appear

**Exact mode:**

```json
{
  "mode": "exact",
  "memories": [
    {
      "fiber_id": "fb04c906_75a3_44b4_9d75_e649f65d3aad",
      "content": "Emma lives in Oslo",
      "created_at": "2026-07-12T05:43:05.134532",
      "valid_until": "2026-07-12T05:43:05.347439",
      "superseded_by": "605f7192-f38d-49dd-97b4-f9fa9b7386db"
    },
    {
      "fiber_id": "605f7192_f38d_49dd_97b4_f9fa9b7386db",
      "content": "Emma moved to Bergen",
      "created_at": "2026-07-12T05:43:05.290788"
    }
  ],
  "fibers_matched": ["fb04c906_...", "605f7192_..."]
}
```

Both present; Oslo carries `valid_until` + `superseded_by`, Bergen carries neither. Confirmed also in
**associative** mode: `answer` contains both "Emma lives in Oslo" and "Emma moved to Bergen",
`fibers_matched` lists both fiber ids.

**PASS.**

## Step 6 — `smem_recall` with `trace: true` → provenance trace_get / traces

```json
{"fibers_matched": ["605f7192_..."], "superseded_excluded_count": 1, "trace_id": "ae28dc08-36f7-49f7-b6cb-45632a62ce9a"}
```

```python
r6_get = await server.call_tool("smem_provenance", {"action": "trace_get", "trace_id": "ae28dc08-..."})
```

```json
{"trace": {
  "id": "ae28dc08-36f7-49f7-b6cb-45632a62ce9a",
  "brain_id": "qa-u7b-01cb3df6",
  "query": "where does Emma live",
  "mode": "associative",
  "confidence": 1.0,
  "fiber_ids": ["605f7192_f38d_49dd_97b4_f9fa9b7386db"],
  "created_at": "2026-07-12T05:43:07.670989+00:00"
}}
```

```python
r6_traces = await server.call_tool("smem_provenance", {"action": "traces", "query_contains": "emma"})
```

```json
{"traces": [{"id": "ae28dc08-36f7-49f7-b6cb-45632a62ce9a", "query": "where does Emma live", ...}], "count": 1}
```

**PASS** (all 3 sub-checks: `trace_id` returned, `trace_get` full record, `traces` listing by
`query_contains`).

## Step 7 — `smem_uncertainty action=overview`

```json
{
  "brain": "qa-u7b-01cb3df6",
  "level": "medium",
  "counts": {"contradictions": 0, "low_evidence": 0, "superseded": 1, "expiring": 0, "drift_clusters": 0},
  "contradiction_rate": 0.0,
  "total_memories": 2,
  "scan": {"typed_scanned": 2, "typed_scan_truncated": false, "contradictions_capped": false},
  "samples": {"superseded": [{"fiber_id": "fb04c906-...", "superseded_by": "605f7192-..."}]}
}
```

`level`/`counts`/`contradiction_rate`/`scan` all present; `superseded: 1` correctly reflects the
Oslo→Bergen lineage; `drift_clusters: 0`, no raise on the real SurrealDB backend.

**PASS.**

## Cleanup

`storage.clear("qa-u7b-01cb3df6")` (neuron/neuron_state/synapse/fiber/change_log/device/merkle_hash/
typed_memory) + explicit `DELETE retrieval_trace WHERE brain_id = $bid` + `DELETE action_log WHERE
brain_id = $bid` (not covered by `clear()`) + `DELETE brain:⟨qa-u7b-01cb3df6⟩` (angle-bracket escape for
the hyphenated record literal). Verified via `SELECT count() FROM <table> WHERE brain_id = $bid GROUP ALL`
for all 10 tables touched by this run, plus a direct `SELECT * FROM brain:⟨...⟩`:

```
verify neuron: count=0
verify neuron_state: count=0
verify synapse: count=0
verify fiber: count=0
verify typed_memory: count=0
verify change_log: count=0
verify device: count=0
verify merkle_hash: count=0
verify retrieval_trace: count=0
verify action_log: count=0
verify brain record: [] (empty — brain deleted)
```

No residue left in the live DB; no real/production brain data touched. No files modified in the repo
(this evidence file only); no git commit/push performed.

## FINAL VERDICT: **PASS** (7 of 7 steps PASS)

Summary for the reviewer:

1. Steps 1, 2, 3, 5, 6, 7 all PASS against the live SurrealDB 3.2.0 with a real in-process `MCPServer`
   built from this worktree's post-fix code — remember, default-recall hard-filter, include_superseded
   (both exact and associative modes), retrieval-trace `trace_id`/`trace_get`/`traces`, and
   `smem_uncertainty overview` all behave as specified, matching the pre-fix run (these were never broken).
2. Step 2's conflict auto-detector still does not fire for "Emma lives in Oslo" / "Emma moved to Bergen"
   (known, unrelated detector-coverage gap) — supersession established via the manual
   `engine.supersession.supersede_typed_memory` path, as documented in the task contingency.
3. **Step 4 — `valid_at` point-in-time recall — now PASSES.** Commit `2ae9514`
   (`_fiber_valid_at` treating a zero-width fiber time window as unbounded) resolves the previously
   structurally-unsatisfiable dual-filter deadlock: `valid_at` values inside Oslo's validity window
   (`Oslo.created_at` exactly, and the window's midpoint) now correctly return Oslo as the primary match
   (`fibers_matched` = `[oslo_fiber]`), while a `valid_at` past the (short, ~205ms) window correctly falls
   through to Bergen — the fix is targeted, not over-permissive.
4. Cleanup fully verified — zero rows in all 10 touched tables, brain record deleted, no production/live
   data affected.

**The v2.9.0 "Spectron-demo" scenario is fully green against the live DB.**
