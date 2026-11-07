---
name: coding-memory-claude
description: Preserve coding decisions, verified traps and fixes, explicit preferences, and task continuity with the Surreal-Memory MCP server. Use at task start, meaningful milestones, after compaction/resume, and before finishing work on an existing project.
---

# Coding memory for Claude

Use with Claude Code. SessionStart (including compact/resume) and UserPromptSubmit restore context. Stop fires after assistant turns, not only at session end. Never return a blocking Stop decision for memory availability.

Read [the shared protocol](references/protocol.md) before working. Apply it throughout
the task. Save valuable conclusions proactively through MCP and keep a scoped task
checkpoint. Remain fail open during memory outages; distinguish pending from saved.

Use `smem_checkpoint`, `smem_context`, `smem_recall`, `smem_remember`, and
`smem_remember_batch` from the configured Surreal-Memory MCP server. Their qualified
names depend on the client; discover the tools rather than hardcoding a transport prefix.
The standard/full tool tiers expose checkpoint support. If unavailable, report reduced
continuity and recommend a current smem server with the standard tier.
