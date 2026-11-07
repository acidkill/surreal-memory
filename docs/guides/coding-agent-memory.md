# Coding memory with Codex and Claude Code

Surreal-Memory supports local Codex CLI/desktop and Claude Code on macOS/Linux.
Hooks provide automatic capture and recovery; the coding-memory skills teach agents
what merits a durable memory and how to keep a task checkpoint. Both are fail open:
a database outage must not block coding or cause a false claim that context was saved.
Native Windows compatibility is preserved, but native Windows is not a release gate.

## Install and verify

```bash
smem init --client codex   # or claude, or all
smem install-skills --client all
smem doctor
```

Without `--client`, setup detects existing client configuration directories. Explicit
selection permits creating a client configuration. Codex MCP registration uses its
native CLI; desktop shares the local Codex configuration. Skill installation uses
`~/.agents/skills/` for Codex and `~/.claude/skills/` for Claude. `CODEX_HOME` selects
Codex configuration, while the user skill discovery directory remains `~/.agents/skills/`.
Skills include their shared `references/protocol.md` in the installed tree and wheel.
Existing user edits are preserved; `--force` explicitly refreshes skill contents.

Restart the client after setup. Review and enable/trust hooks using the client's
normal controls (Codex `/hooks`); setup never bypasses trust. Managed policy can disable
user hooks even when configuration exists. `smem doctor` reports registration, missing
skills/MCP, minimal tool-tier limitations, and pending writes; registration alone does
not prove that hooks ran. The standard/full tool tiers expose `smem_checkpoint`.
Older Codex versions without enabled native hooks receive MCP and skills with an explicit
skills-only status. No legacy notify shim or automatic-compaction guarantee is provided.

Setup merges only owned smem command hooks and preserves other hooks/settings. Invalid
JSON or malformed hook structures are reported instead of replaced. Changed settings
receive a content-addressed `.smem-<hash>.bak` backup before atomic replacement.
Do not register both the Claude plugin hooks and user hooks for the same installation;
choose one registration path. Plugin hooks use `uvx --from surreal-memory smem-hook-agent`;
the published package must contain that entry point. For unreleased source testing use
the checkout installation and `smem init --client claude`. Content-idempotent writes protect repeated events.

## Durable knowledge and temporary task state

Durable memories capture decisions plus rationale/alternatives, verified traps and fixes,
explicit preferences, constraints, and reusable workflows. Use `scope="project"` and
`project_id` on `smem_remember`, batch items, `smem_recall`, and `smem_context`.
Identity defaults to `SMEM_PROJECT`, then sanitized Git origin (`host/owner/repository`),
then canonical repository path. The hook supplies the resolved ID to the agent.
Scoped filters apply before candidate limits; unrelated HOT memories, exact-fiber
lookups, and cross-brain shortcuts cannot bypass them. Unscoped legacy calls retain
their existing behavior. Existing basename-scoped hook memories remain accessible
through legacy APIs; no automatic alias broadens the new project scope.

Personal preferences apply across projects only when explicitly saved as
`type="preference", scope="global_preference"`. Include them in scoped reads with
`include_global_preferences=true`. Repository conventions never become global by inference.

Do not store transcripts, routine reads, secrets, speculative conclusions, or hidden
reasoning. Record conclusions and evidence. The skill saves knowledge immediately when
established and checkpoints meaningful milestones; an abrupt exit can lose unsaved work.

## Task checkpoint API

`smem_checkpoint(action="save"|"load", scope={...}, revision=N, state={...})` uses the
existing version-1 envelope and verified writer. It is separate from consolidation's
maintenance checkpoints. The exact scope contains `brain`, absolute canonical `project`
path, runtime `session_id`, and `agent_id` (`main` for the parent agent).

State contains `goal` (string) and string arrays `constraints`, `approvals`, `decisions`,
`completed`, `next_actions`, `blockers`, and `processes`. Put verification commands and
results in `completed`; uncertainty in `blockers`. Revalidate historical approvals and
processes against current instructions and the actual machine before using them.

Save requires a positive revision. Increment after changed state; reuse the same revision
for identical retries. Hashes are computed server-side. Conflicts/stale revisions are
rejected. A durable head points to the latest confirmed record; both fiber and typed
memory are checked before returning it. Legacy v1 checkpoints are readable by exact
scope; a saturated legacy scan fails explicitly instead of choosing an arbitrary record.
A checkpoint is limited to 100,000 bytes; oversized/sensitive/invalid state is rejected.

Load never selects the brain-wide latest session. Cross-session/client handoff requires
an explicitly chosen source scope. Concurrent local processes share cancellation-safe
per-scope locks; this is a local-client integration, not a distributed writer lease.

## Events and failures

| Event | Behavior |
|---|---|
| SessionStart / resume / SubagentStart | Restore checkpoint and selected project/personal knowledge |
| UserPromptSubmit | Retry recovery if unavailable or context was compacted |
| PreCompact | Mark context for reinjection; replay queued work and capture a bounded tail |
| PostCompact (Codex) | Mark recovery pending; never emit unsupported context fields |
| PostToolUse | Buffer redacted tool metadata locally, without database access |
| Stop / SubagentStop / SessionEnd (Codex) | Replay pending writes and capture curated conversation signals |

Claude's compact/resume SessionStart resets recovery markers. Stop recursion is ignored.
Readers handle Claude messages and Codex rollout message payloads, skip reasoning/tools,
retain incomplete JSONL lines for the next read, and bound each read to 512 KiB. On rotation
or truncation they restart; an oversized unread tail emits a diagnostic. Cursors advance
only after candidates are confirmed or placed in the retry journal. The context injection
budget is 6,000 characters (approximately 1,500 tokens); scoped knowledge uses at most
2,000 of those characters. Hooks do not run an extra model to invent task summaries.

Client hook commands always fail open. Internal deadlines are one second for tool
capture, five seconds for context loading, ten seconds overall otherwise; registered
client timeouts are slightly larger. Diagnostics go to stderr and stdout contains only
the client's hook JSON. Files are kept under `SURREAL_MEMORY_DIR` (default `~/.surrealmemory`):

- `coding-outbox.json`: mode-0600, atomically replaced pending payloads with kind, version,
  creation time (UTC Unix seconds), and stable content identity. Never raw credentials.
- `coding-context-<scope hash>.json`: cursor and context-injection marker.
- `coding-<scope hash>.lock`: per-scope cancellation-safe lock.

Default retry retention is seven days and 5 MiB, configurable through
`SMEM_RETRY_MAX_AGE_DAYS` and `SMEM_RETRY_MAX_BYTES`. Expiry is diagnosed; overflow
preserves existing queued work and reports that the new write was not queued.
The in-memory trial backend cannot confirm durable brain persistence: scoped writes remain
pending with `durable_backend_required` until SurrealDB is configured.
Only confirmed writes are acknowledged. `pending` is not `saved`; failures are never
reported as successful brain persistence. Sensitive payloads are rejected before queueing.

## Release verification

Automated tests exercise isolated agents/projects, global preferences, replay, conflicts,
malformed input, bounded/rotated transcripts, repeated compaction, preservation of user
configuration, queue bounds, and pre-connect failures. Run the repository quality gates
and live SurrealDB tests. On each supported client, verify a real session starts, a
milestone is checkpointed, compaction occurs twice, and the agent restores the correct
next action both times. Repeat with SurrealDB unavailable and after restarting it.
Desktop and macOS client behavior require native-client validation; Python fixtures do
not establish those runtime results.
