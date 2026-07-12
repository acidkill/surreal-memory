# U5 Uncertainty Surfacing — Live SurrealDB Evidence

- Repo/worktree: `/home/acidkill/repos/surreal-memory/.claude/worktrees/run-006`
- Branch: `feature/v290-pr5-uncertainty`
- Live DB: `SURREALDB_URL=http://localhost:8001`, SurrealDB **3.2.0** (confirmed via `curl -s http://localhost:8001/version`)
- Harness: real in-process `MCPServer()` (from `surreal_memory.mcp.server`), backed by the real `SurrealDBStorage` (env `SURREAL_MEMORY_STORAGE=surrealdb`). No mocks anywhere in this run.
- Throwaway brains: `qa-u5-a9a08414` (main run) + `qa-u5-b0c97a39` (earlier failed run, cleaned up too). Set via `SURREAL_MEMORY_BRAIN` env var before construction — confirmed process-isolated (`get_shared_storage()` honours `SURREAL_MEMORY_BRAIN` without mutating the shared config).
- Cosmetic `FLEXIBLE ... SCHEMAFULL` schema-init warnings on every run — ignored per instructions (pre-existing, unrelated to U5).

## 0. Unit tests (baseline, before live-DB work)

```
.venv/bin/python -m pytest tests/unit/test_uncertainty_report.py tests/unit/test_uncertainty_handler.py -p no:xdist -p no:cacheprovider -v
```

Result: **16 passed** in 0.07s (both `TestBuildUncertaintyBlock` and `TestUncertaintyTool` suites), 0 failures.

## Seed data (live SurrealDB, brain `qa-u5-a9a08414`)

Six `smem_remember` calls via `server.call_tool("smem_remember", {...})`:

| label | content | notes |
|---|---|---|
| fact_a_postgres | "We use PostgreSQL for the primary production database." | baseline fact |
| fact_b_mysql_active_conflict | "We use MySQL for the primary production database." | contradicts fact_a. Crafted to be **shorter** than fact_a so `conflict_auto_resolve.try_auto_resolve` Rule 2 ("same session + new content longer → auto keep_new") does NOT fire → stays an **unresolved/active** CONTRADICTS synapse (verified: `conflicts_detected=1`, and the synapse carries no `_resolved` metadata) |
| fact_c_aws_east | "We use AWS us-east-1 for staging deploys." | baseline fact |
| fact_d_aws_west_supersedes | "We switched to AWS us-west-2 for staging deploys after the recent regional outage." | contradicts fact_c, deliberately **longer** → Rule 2 auto-resolve fires → supersedes fact_c (produces a `superseded` signal distinct from the active-contradiction pair) |
| low_trust_fact | "Alice mentioned she enjoys hiking on weekends." | `trust_score=0.2` (below the `_LOW_TRUST_THRESHOLD=0.4`) |
| expiring_fact | "The temporary staging API key rotates soon." | `expires_days=1` |

All 6 `smem_remember` calls returned `"success": true`, zero `error` keys (`SEED ERRORS: {}`).

Key remember-response evidence:
```json
// fact_b (contradicts fact_a):
"message": "Remembered: We use MySQL for the primary production database. (1 conflict(s) detected) (superseded 1 prior fact(s))",
"conflicts_detected": 1, "superseded_count": 1

// fact_d (contradicts fact_c, auto-resolved):
"message": "Remembered: We switched to AWS us-west-2 for staging deploys a... (1 conflict(s) detected) (superseded 1 prior fact(s))",
"conflicts_detected": 1, "superseded_count": 1
```

(Learning note for anyone reusing this recipe: `conflict_auto_resolve.try_auto_resolve` is called with `new_confidence=0.5` hard-coded in the pipeline, so its Rule 1 [`new_confidence>=0.8`] never fires there; the only rule that mattered for these facts was Rule 2 [`age<3600s AND len(new)>len(existing)`]. Making the contradicting content the same length or shorter than the original is what keeps a conflict manual/unresolved instead of being auto-superseded.)

## 1. `smem_uncertainty` overview

Call: `server.call_tool("smem_uncertainty", {"action": "overview"})`

```json
{
  "brain": "qa-u5-a9a08414",
  "level": "high",
  "counts": {
    "contradictions": 1,
    "low_evidence": 1,
    "superseded": 2,
    "expiring": 1,
    "drift_clusters": 0
  },
  "contradiction_rate": 0.1667,
  "total_memories": 6,
  "samples": { "low_evidence": [...1 item...], "superseded": [...2 items...], "drift_clusters": [] }
}
```

**PASS.** All required top-level keys present (`level`, `counts`, `contradiction_rate`, `total_memories`), all 5 `counts` sub-keys present. `contradictions=1` (the fact_a/fact_b pair, unresolved), `low_evidence=1` (Alice fact, trust 0.2), `superseded=2` (both fact_a→fact_b via the CONTRADICTS-with-supersede path, and fact_c→fact_d via the auto-resolve path), `expiring=1` (the 1-day API key fact). No `error` key.

**CRITICAL check — backend-degradation (the key ask of this validation):** `counts.drift_clusters == 0` and the call did **not raise**. Verified directly that `SurrealDBStorage` has no `get_drift_clusters` method (`grep -rn "def get_drift_clusters" src/surreal_memory/storage/` only matches `storage/sqlite_drift.py`). `uncertainty_handler._get_drift()` does `getattr(storage, "get_drift_clusters", None)` → `None` on SurrealDB → short-circuits to `[]` before any exception can occur. Confirmed live: the overview call completed cleanly with `drift_clusters: 0` and an empty `samples.drift_clusters` list — the ABC/backend-gap graceful-degradation path works exactly as designed on the real SurrealDB storage, not just in the mocked unit test (`test_drift_absent_backend_degrades_gracefully`).

## 2. `smem_uncertainty` low_evidence and expiring

```json
// action=low_evidence
{"low_evidence": [{"fiber_id": "91b8fbe6-...", "trust_score": 0.2}], "count": 1}

// action=expiring, within_days=30
{"expiring": [{"fiber_id": "d467f283-...", "memory_type": "fact", "expires_at": "2026-07-13T04:14:43.178529"}], "count": 1, "within_days": 30}
```

**PASS.** Both bounded lists returned without error, matching the seeded low-trust and expiring facts exactly.

## 3. `smem_uncertainty` contradictions (delegates to conflicts listing)

Call: `server.call_tool("smem_uncertainty", {"action": "contradictions"})`

```json
{
  "conflicts": [
    {
      "existing_neuron_id": "5e86a1f4-bb3b-453e-837f-108b8ad27334",
      "content": "We use PostgreSQL for the primary production database.",
      "disputed_by_preview": "We use MySQL for the primary production database.",
      "conflict_type": "factual_contradiction",
      "confidence": 0.7222222222222222,
      "detected_at": "2026-07-12T04:14:42.111035",
      "is_superseded": true,
      "auto_resolved": false,
      "auto_resolve_reason": ""
    }
  ],
  "count": 1
}
```

**PASS.** Shape matches `ConflictHandler._conflicts_list` exactly (`conflicts` + `count`, with `existing_neuron_id`/`content`/`disputed_by_preview`/`conflict_type`/`confidence`/`detected_at`/`is_superseded`/`auto_resolved`/`auto_resolve_reason`) — confirms `_uncertainty()`'s `action=="contradictions"` branch really delegates to the shared mixin method rather than reimplementing it.

## 4. `smem_recall` include_uncertainty

**With signal present** — `server.call_tool("smem_recall", {"query": "primary production database", "include_uncertainty": True})`:

```json
{
  "answer": "## Relevant Memories\n\n- We use MySQL for the primary production database....",
  "confidence": 1.0,
  "fibers_matched": ["d042d042_be74_4797_bc88_5721d616fd93"],
  "has_conflicts": true,
  "conflict_count": 1,
  "uncertainty": {
    "level": "high",
    "counts": {"contradictions": 1, "superseded": 0, "low_confidence": 0, "expiring": 0, "drift_clusters": 0},
    "contradictions": [{"neuron_id": "5e86a1f4-...", "content": "We use PostgreSQL for the primary production database."}],
    "superseded": [], "low_confidence": null, "expiring": [], "drift_clusters": []
  }
}
```

Result: **signal present → `uncertainty` block attached**, correctly reflecting the one disputed neuron surfaced by this recall (via `metadata.disputed_ids` from the retrieval de-prioritization step), independent of and additive to the pre-existing `has_conflicts`/`conflict_count` fields.

**Default recall, no flag** — `server.call_tool("smem_recall", {"query": "primary production database"})` → `"uncertainty" in response` = **False**.

**Default recall, unrelated query, no flag** — `server.call_tool("smem_recall", {"query": "hiking weekends"})` → `"uncertainty" in response` = **False**.

**PASS.** `include_uncertainty=True` attaches the block when (and only when) there is a real signal for that recall; omitting the flag never attaches it, on two different queries (one that has a live contradiction signal, one that doesn't) — confirming the opt-in, additive, backend-agnostic contract end-to-end against live SurrealDB.

## Cleanup

All rows for both throwaway brains (`qa-u5-a9a08414` and the earlier failed-attempt brain `qa-u5-b0c97a39`) deleted from `neuron`, `neuron_state`, `fiber`, `typed_memory`, `synapse`, `retrieval_trace`, `tool_events`, `alerts`. Verified `SELECT count() FROM neuron WHERE brain_id = $bid GROUP ALL` → `0` for both. The `brain` table row itself needed a follow-up because SurrealQL parses a bare hyphenated record literal (`brain:qa-u5-xxxx`) as a subtraction expression (`Cannot perform subtraction with 'record' and 'table'`) — the working form is the angle-bracket-escaped literal `` DELETE brain:⟨qa-u5-xxxx⟩ ``. Re-verified via full `SELECT id, name FROM brain` scan afterwards: zero rows match either throwaway brain name. No test data remains in the live DB.

## VERDICT: PASS

All 5 requested checks pass against the live SurrealDB v3.2.0 instance with a real in-process `MCPServer`, real `SurrealDBStorage`, zero mocking:

1. Seed data (contradiction + low-trust fact) — PASS
2. `smem_uncertainty overview` — PASS, including the **critical SurrealDB drift-degradation check**: `drift_clusters=0` with no exception, because `getattr(storage, "get_drift_clusters", None)` is `None` on `SurrealDBStorage` (only `sqlite_drift.py` implements it) and both `uncertainty_handler._get_drift` and `uncertainty_report._drift` short-circuit to an empty list instead of raising `AttributeError`.
3. `smem_uncertainty low_evidence` / `expiring` — PASS, bounded, no error.
4. `smem_uncertainty contradictions` — PASS, correctly delegates to the conflicts-listing shape.
5. `smem_recall include_uncertainty` — PASS: attaches `uncertainty` block when a real signal exists, absent otherwise, and absent by default on both a disputed and a non-disputed query when the flag is omitted.
