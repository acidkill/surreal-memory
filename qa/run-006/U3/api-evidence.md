# U3 Supersession — MCP Tool Contract Live Evidence

**Date:** 2026-07-12
**Branch:** feature/v290-pr3-supersession
**Worktree:** /home/acidkill/repos/surreal-memory/.claude/worktrees/run-006
**Live DB:** SURREALDB_URL=http://localhost:8001 (`surrealdb-3.2.0`, confirmed via `curl -s http://localhost:8001/version`)
**Method:** Real in-process `MCPServer` (`surreal_memory.mcp.server.MCPServer`) built against the live SurrealDB
via `SurrealDBStorage`. No mocks. Each item's data seeded via direct storage calls +
`engine.supersession.supersede_typed_memory` (per task instructions), then exercised through the actual
`server.call_tool(...)` dispatch so the JSON tool contract is what's asserted on. Each run used a fresh
throwaway brain (`SURREAL_MEMORY_BRAIN=qa-u3-<8 hex>`, auto-created by `get_shared_storage()`), never touching
real project memories. The `Schema statement failed: ... FLEXIBLE can only be used in SCHEMAFULL tables` lines
are the known cosmetic warning (ignored per task instructions).

---

## Item 1 — `smem_recall` supersession filtering (default hard-filter / include_superseded / escape hatch / valid_at / exact-mode lineage fields)

### Run A — brain `qa-u3-7976ba63`

Seed: two facts sharing keywords ("Emma zephyrhome record: Emma currently lives in Oslo." → superseded by
"...in Bergen."), superseded via `supersede_typed_memory(old_fiber_id=<oslo>, new_fiber_id=<bergen>, ...)`.

```
supersede outcome: SupersessionOutcome(old_fiber_id='9c276d0d-e828-4ed4-9e97-c27b74d982c1',
                                       new_fiber_id='4a46ed0b-8465-44bf-a10b-fac14ea48330',
                                       superseded=True)
```

**Call:** `server.call_tool("smem_recall", {"query": "Emma zephyrhome lives", "mode": "associative"})`
(no `include_superseded`)

```json
{
  "fibers_matched": ["4a46ed0b_8465_44bf_a10b_fac14ea48330"],
  "superseded_excluded_count": 1
}
```
→ old (superseded) fiber `9c276d0d...` is ABSENT from `fibers_matched`; only the new one (`4a46ed0b...`)
survives; `superseded_excluded_count: 1` present. **PASS.**

**Call:** `server.call_tool("smem_recall", {"query": "Emma zephyrhome lives", "mode": "associative", "include_superseded": true})`

```json
{
  "fibers_matched": ["4a46ed0b_8465_44bf_a10b_fac14ea48330", "9c276d0d_e828_4ed4_9e97_c27b74d982c1"],
  "superseded_excluded_count": null
}
```
→ superseded fiber REAPPEARS with `include_superseded: true`. **PASS.**

**Call:** `server.call_tool("smem_recall", {"query": "Emma zephyrhome lives", "mode": "exact"})` (default)

```json
{
  "mode": "exact",
  "fibers_matched": ["4a46ed0b_8465_44bf_a10b_fac14ea48330"],
  "memories": [{"fiber_id": "4a46ed0b_...", "content": "...Bergen.", "memory_type": "fact", ...}],
  "superseded_excluded_count": 1
}
```
→ same default hard-filter behaviour holds in exact mode. **PASS.**

**Call:** `server.call_tool("smem_recall", {"query": "Emma zephyrhome lives", "mode": "exact", "include_superseded": true})`

```json
{
  "mode": "exact",
  "fibers_matched": ["4a46ed0b_...", "9c276d0d_..."],
  "memories": [
    {"fiber_id": "4a46ed0b_...", "content": "...Bergen."},
    {"fiber_id": "9c276d0d_...", "content": "...Oslo.",
     "valid_until": "2026-07-12T02:42:44.362353",
     "superseded_by": "4a46ed0b-8465-44bf-a10b-fac14ea48330"}
  ]
}
```
→ exact-mode superseded item exposes `valid_until` + `superseded_by`. `valid_from` absent here because it was
never explicitly set on this particular typed_memory (field is `None` by design when unset — see Run B for the
explicit-`valid_from` case). **PASS** (partial — see Run B for full 3-field exposure).

### Run B — brain `qa-u3-638e9fa9` (valid_from exposure / escape hatch / valid_at)

Seed: old fact "Ravensplit ledger: Ravensplit HQ is currently in Rotterdam." created with `valid_from` explicitly
stamped to its own `created_at` (`tm.with_validity(valid_from=tm.created_at)`), `t_mid = utcnow()` captured, then
(after `asyncio.sleep(0.05)`) new fact "...in Antwerp." created, then superseded.

```
supersede outcome: SupersessionOutcome(old_fiber_id='09784737-4276-4fbd-9223-05102bc3431d',
                                       new_fiber_id='15ca5c0e-e80f-4a64-88fd-7f3422646f0f',
                                       superseded=True)
```

**Call:** `server.call_tool("smem_recall", {"query": "Ravensplit HQ", "mode": "exact", "include_superseded": true})`

```json
[
  {
    "fiber_id": "09784737_...",
    "content": "...Rotterdam.",
    "valid_from": "2026-07-12T02:43:11.586954",
    "valid_until": "2026-07-12T02:43:11.654160",
    "superseded_by": "15ca5c0e-e80f-4a64-88fd-7f3422646f0f"
  },
  {"fiber_id": "15ca5c0e_...", "content": "...Antwerp."}
]
```
→ superseded item exposes all three lineage fields (`valid_from`, `valid_until`, `superseded_by`). **PASS.**

**Escape hatch — `SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER=1` set in `os.environ`, NO `include_superseded`:**

**Call:** `server.call_tool("smem_recall", {"query": "Ravensplit HQ", "mode": "associative"})`

```json
{
  "fibers_matched": ["09784737_...", "15ca5c0e_..."],
  "superseded_excluded_count": null
}
```
→ superseded fact reappears via the env escape hatch alone (no `include_superseded` flag needed). **PASS.**

**After unsetting the env var, same call again (hard filter re-armed):**

```json
{
  "fibers_matched": ["15ca5c0e_..."],
  "superseded_excluded_count": 1
}
```
→ default hard-filter behaviour restored once the escape hatch is unset. **PASS** (confirms the hatch is a live
toggle, not a one-way state leak).

**`valid_at` point-in-time filter — `t_mid` (captured before the new fact even existed):**

**Call:** `server.call_tool("smem_recall", {"query": "Ravensplit HQ", "mode": "exact", "valid_at": t_mid.isoformat()})`

```json
{
  "fibers_matched": ["09784737_..."],
  "memories": [{"fiber_id": "09784737_...", "content": "...Rotterdam.", "valid_from": "...", "valid_until": "...", "superseded_by": "..."}]
}
```
→ recall at `t_mid` returns ONLY the old (then-valid) fact, not the new one (`new.valid_from` postdates `t_mid`).
**PASS.**

### Item 1 verdict: **PASS** (all 6 sub-assertions: default hard-filter + count, include_superseded reappearance,
exact-mode lineage-field exposure, escape hatch on/off, valid_at point-in-time).

---

## Item 2 — `smem_lifecycle` action=`backfill_supersession`

Brain: `qa-u3-0bf74047`. Seed: old fact "...pricing tier was Bronze." and new fact "...pricing tier is now Gold.",
each its own fiber/neuron/typed_memory (via direct storage calls, **not** `supersede_typed_memory`). Old anchor
neuron stamped `_superseded=True` (C-side only) + a `CONTRADICTS` synapse `new_anchor → old_anchor` added
directly — reproducing pre-U3 state with NO A-side (`valid_until`/`superseded_by`) lineage yet.

**Pre-check (direct storage read, not a tool call):**
```
old_tm.superseded_by: None   valid_until: None
```

**Call:** `server.call_tool("smem_lifecycle", {"action": "backfill_supersession"})`

```json
{
  "action": "backfill_supersession",
  "scanned": 1,
  "backfilled": 1,
  "already_linked": 0,
  "skipped_ambiguous": 0,
  "truncated": false,
  "brain": "qa-u3-0bf74047"
}
```
→ `backfilled: 1`. **PASS.**

**Post-check (direct storage read):**
```
old_tm.superseded_by: 9c4d1e60_57a3_45c9_9c21_68e96a8fb7b9   valid_until: 2026-07-12 02:43:44.277837
```
→ old typed_memory now carries `superseded_by` (underscore-normalized id form of the new fiber id — the
documented dash/underscore round-trip; functionally equivalent, matches `tests/unit/test_surrealdb_supersession_live.py`
precedent) and a non-null `valid_until`. **PASS.**

**Call again (idempotency):** `server.call_tool("smem_lifecycle", {"action": "backfill_supersession"})`

```json
{
  "action": "backfill_supersession",
  "scanned": 1,
  "backfilled": 0,
  "already_linked": 1,
  "skipped_ambiguous": 0,
  "truncated": false,
  "brain": "qa-u3-0bf74047"
}
```
→ `backfilled: 0`, `already_linked: 1`. **PASS** (idempotent, as specified).

### Item 2 verdict: **PASS**

---

## Item 3 — `smem_provenance` action=`trace` (A←B←C supersession chain)

Brain: `qa-u3-d32b2ca0`. Seed: three standalone facts A, B, C. Chain built with two `supersede_typed_memory` calls:
`old=A, new=B` then `old=B, new=C` (SUPERSEDES synapse direction is `new_anchor → old_anchor`, i.e.
`B_anchor → A_anchor` and `C_anchor → B_anchor`).

```
supersede A->B: SupersessionOutcome(old_fiber_id='6f1814b0-...', new_fiber_id='d0f0845b-...', superseded=True)
supersede B->C: SupersessionOutcome(old_fiber_id='d0f0845b-...', new_fiber_id='26e11333-...', superseded=True)

A anchor: 2bd28711-df1c-4930-ab6b-6ae2f947bbb4
B anchor: bcfab552-3754-4479-b512-b36fd5821aad
C anchor: 0a2d5e3c-9f39-4193-8c82-f9dffc2a3105
```

**Call:** `server.call_tool("smem_provenance", {"action": "trace", "neuron_id": "bcfab552-3754-4479-b512-b36fd5821aad"})` (B's anchor)

```json
{
  "neuron_id": "bcfab552-3754-4479-b512-b36fd5821aad",
  "provenance": [
    {
      "type": "superseded_by",
      "neuron_id": "0a2d5e3c-9f39-4193-8c82-f9dffc2a3105",
      "reason": "B->C",
      "timestamp": "2026-07-12T02:44:01.688091"
    },
    {
      "type": "supersedes",
      "neuron_id": "2bd28711-df1c-4930-ab6b-6ae2f947bbb4",
      "reason": "A->B",
      "timestamp": "2026-07-12T02:44:01.675568"
    }
  ],
  "has_source": false,
  "is_verified": false,
  "is_approved": false,
  "is_superseded": true,
  "supersedes_count": 1
}
```

Assertions:
- `"superseded_by"` entry `neuron_id` == C's anchor (`0a2d5e3c-...`). **PASS.**
- `"supersedes"` entry `neuron_id` == A's anchor (`2bd28711-...`). **PASS.**
- `is_superseded == true`. **PASS.**
- `supersedes_count == 1`. **PASS.**

### Item 3 verdict: **PASS**

---

## FINAL VERDICT: **PASS** — all 3 U3 supersession MCP tool contracts (`smem_recall`, `smem_lifecycle`
action=`backfill_supersession`, `smem_provenance` action=`trace`) verified end-to-end against the live dev
SurrealDB (v3.2.0) via real in-process `MCPServer.call_tool()` dispatch. No mocks used; each scenario ran
against its own throwaway brain (auto-created via `get_shared_storage()`, never touching production memories).
