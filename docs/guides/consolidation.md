# Consolidation and upgrading to 3.12.2

Version 3.12.2 fixes recovery and database work since 3.12.1. It keeps schema
v12, the existing strategy budgets and the `3.11.0:checkpoint-v1` compatibility
epoch. Faster queries do not guarantee that every brain finishes within one
strategy budget.

## Before updating

1. Stop maintenance and wait for running consolidation calls to finish. Avoid
   replacing a package while its CLI or server processes are using it.
2. Back up the **full selected SurrealDB database**, including checkpoint and
   staging tables. A memory-only export is not a complete resumable-run backup.
   Verify restoration in an isolated environment and retain the previous package.
3. Update with the installer you already use, preserving your existing extras,
   database/brain selection, credentials and any credential-refresh wrapper.
   After publication, a pip installation can use:

   ```bash
   pip install --upgrade 'surreal-memory[surrealdb]==3.12.2'
   ```

   This example selects only the SurrealDB extra; retain any additional extras
   from your own installation. For pipx, preserve the original install spec and
   extras rather than replacing them with an unrelated minimal spec.
4. Restart MCP/server processes so they load the updated code. Check the installed
   version and `smem consolidate --help` before beginning maintenance.

## Start or resume

```bash
# Continue this brain's saved incomplete run with its saved strategy set.
smem consolidate --resume --brain default

# Start all strategies when there is no unfinished run.
smem consolidate --brain default
```

Explicit `--resume` retains the saved reference time and skips completed
strategies. It cannot be combined with an explicit `--strategy` or `--dry-run`.
A missing, completed, replaced or incompatible checkpoint is rejected. A normal
invocation can still automatically resume a compatible checkpoint; requesting
different strategies/configuration will not override the saved run.

Use one consolidation process per brain. Do not delete or edit its checkpoint,
staging rows, lease or fingerprints to force progress. The next run begins only
after the saved run has completed.

## Read the outcome

| Outcome | What it means | Next step |
|---|---|---|
| `completed`, CLI exit 0 | The requested run finished without a reported stage failure | Validate health and important recall; start a new run when needed |
| `paused`, CLI exit 1 | A strategy budget ended with committed work retained | Inspect progress, then resume the same run |
| `blocked` | Saved work, configuration or ownership checks prevent execution | Preserve the checkpoint and diagnose the mismatch |
| `failed`, `lease_lost`, `timed_out`, or reported failed stages | The run did not complete successfully | Preserve logs/state and investigate before retrying |

A stable run ID and report date are expected across resumes. Report counters are
cumulative for the saved run. A pending work unit can drain over several calls
before its committed cursor advances, so an unchanged printed cursor/count alone
does not prove a stall. Check durable phase/page progress and the actual pending
work before deciding that the run is stuck. Conversely, positive counters or
small health changes do not prove that all strategies completed.

Do not blindly repeat resumes after a new validation or connection error. A lost
WebSocket receiver does not prove that the last query failed to execute; 3.12.2
surfaces that uncertainty instead of automatically replaying it.

## Data protection and downgrade limits

- Prune, merge, lifecycle and compression can change or remove eligible graph
  records. Consolidation is maintenance, not a promise that every record remains.
- Compression protects pinned/shared neurons, preserves original backups across
  tier advancement and validates lineage before replay. Later edits are not
  overwritten merely because an old intent exists. Legacy states whose lineage
  cannot be verified stop for investigation rather than being repaired blindly.
- Fiber censuses account for SurrealDB's omission of null fields, including
  metadata, while preserving immutable page checks. Valid older staged forms
  remain supported; the previously invalid metadata-null case needs diagnosis
  rather than a fabricated hash.
- **New census pages are generally not readable by 3.12.1.** The unchanged
  compatibility epoch does not make every staging representation bidirectional.
  Do not downgrade in the middle of newly staged work; retain a tested full
  backup for rollback.

After maintenance, inspect the run's completed status and exit code, then check
important memories with representative recall queries. Health counters alone
cannot establish content integrity. `smem brain health` checks hygiene;
`smem_health()` provides broader graph diagnostics, and `smem brain recall-check`
is a sampled retrieval probe rather than a full integrity audit.

## Semantic snapshot retention

For saved runs, semantic discovery retains its frozen reference time across
bounded pair batches. A batch that reaches the pair cap pauses after its writes
are committed; `--resume` continues discovery instead of declaring the stage
complete. The cap and strategy time budget remain unchanged.

After the initial least-connected sample, deterministic ID-order candidate
windows with a one-candidate overlap cover the remaining frozen eligible
population. The report distinguishes a candidate window from the initial
resampling and explicitly marks completed frozen candidate coverage. This
coverage applies the existing similarity and top-K rules within bounded windows;
it does not promise every possible pair across different windows. A changed
frozen source fails validation before another batch is applied.

Discovery checkpoints use manifest version 5, and apply manifests use version 4.
Previous manifest versions remain readable. Older packages reject newly written
checkpoints; do not downgrade while such a run is unfinished. NumPy is required for a frozen window
larger than the pure-Python fallback limit; a missing NumPy installation stops
that window rather than silently dropping its remaining candidates.

Source validation captures a finite changefeed barrier. Its 100,000 events per
source table and 2,000,000 database events are budgets for one invocation, not
limits on the saved run's history. A large safe history pauses normally with CLI
exit 1 and a separately reported verified cursor. The next resume continues that
prefix, including the classifier's paged record-ID evidence. It does not apply
another synapse batch until the required range has been verified. Concurrent
recall statistics and new memories can be classified without changing the frozen
source; changes to that source, unknown events and missing history still fail
closed. Neither a pause nor a positive validation count proves run completion.

Validation pages are immutable snapshots of at most 1 MB, published through the
same lease-fenced checkpoint. An interrupted or unconfirmed write cannot publish
a partial transaction. The original source token, frozen reference time,
completed strategies and consolidation counters are preserved. The validation
base moves to a newer barrier only after the intervening range is proved.

```toml
[maintenance]
semantic_discovery_state_retention_days = 7
```

After `semantic_link` completes, expired discovery snapshots from finished runs
are deleted in batches of at most 500. The current run and queued/running/paused/
failed runs are protected. `0` disables this cleanup. The report includes
"Semantic discovery snapshots pruned"; a cleanup failure is reported without
turning the completed semantic-link strategy into a failure.

## Prevent read/capture feedback

`smem_recall`, `smem_context`, `smem_recap` and `smem_explain` no longer analyze
their query or returned text for new passive memories. To save new information,
use `smem_remember` or explicitly call `smem_auto(action="process", text=...)`.
The latter still honors the auto-capture settings.

This does **not** make recall read-only: normal activation, reinforcement,
retrieval traces and configured reconsolidation can still write state.
`reconsolidate=false` disables query-context reconsolidation for that call, not
every other recall write.

MCP analytics honor `[tool_memory] enabled` and blacklist prefixes. Successful
calls to the four tools above are skipped; permitted failures retain diagnostic
events. The PostToolUse hook recognizes plain and `mcp__SERVER__NAME` names and
matches blacklist prefixes against both the original and canonical name. Useful
external tool events can still be captured. This policy prevents new feedback;
it does not erase old memories or historical tool events.
