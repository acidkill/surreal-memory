---
name: coding-memory-codex
description: Preserve coding decisions, verified traps and fixes, explicit preferences, and task continuity with the Surreal-Memory MCP server. Use at task start, meaningful milestones, after compaction/resume, and before finishing work on an existing project.
---

# Coding memory for Codex

Use with Codex CLI and desktop. Native hooks restore context at SessionStart and the first UserPromptSubmit after compaction. PostCompact marks recovery pending; it does not inject context. Older versions use this skill and MCP without lifecycle guarantees.

Read [the shared protocol](references/protocol.md) before working. Apply it throughout
the task. Save valuable conclusions proactively through MCP and keep a scoped task
checkpoint. Remain fail open during memory outages; distinguish pending from saved.

Use `smem_checkpoint`, `smem_context`, `smem_recall`, `smem_remember`, and
`smem_remember_batch` from the configured Surreal-Memory MCP server. Their qualified
names depend on the client; discover the tools rather than hardcoding a transport prefix.
The standard/full tool tiers expose checkpoint support. If unavailable, report reduced
continuity and recommend a current smem server with the standard tier.
