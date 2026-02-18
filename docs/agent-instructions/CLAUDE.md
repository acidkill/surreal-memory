# NeuralMemory — Instructions for Claude Code

> Copy this section into your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` (global).

## Memory System

This workspace uses **NeuralMemory** for persistent memory across sessions.
You have access to `nmem_*` MCP tools. Use them **proactively** — do not wait for the user to ask.

### Session Start (ALWAYS do this)

```
nmem_recap()                          # Resume context from last session
nmem_context(limit=20, fresh_only=true)  # Load recent memories
nmem_session(action="get")            # Check current task/feature/progress
```

If `gap_detected: true`, run `nmem_auto(action="flush", text="<recent context>")` to recover lost content.

### During Work — REMEMBER automatically

| Event | Action |
|-------|--------|
| Decision made | `nmem_remember(content="...", type="decision", priority=7)` |
| Bug fixed | `nmem_remember(content="...", type="error", priority=7)` |
| User preference stated | `nmem_remember(content="...", type="preference", priority=6)` |
| Important fact learned | `nmem_remember(content="...", type="fact", priority=5)` |
| TODO identified | `nmem_todo(task="...", priority=6)` |
| Workflow discovered | `nmem_remember(content="...", type="workflow", priority=6)` |

### During Work — RECALL before asking

Before asking the user a question, check memory first:

```
nmem_recall(query="<topic>", depth=1)
```

Depth guide: 0=instant lookup, 1=context (default), 2=patterns, 3=deep graph traversal.

### Session End / Before Compaction

```
nmem_auto(action="process", text="<summary of session>")
nmem_session(action="set", feature="...", task="...", progress=0.8)
```

Before `/compact` or `/new`:
```
nmem_auto(action="flush", text="<recent conversation>")
```

### Project Context

```
nmem_eternal(action="save", project_name="MyProject", tech_stack=["React", "Node.js"])
nmem_eternal(action="save", decision="Use PostgreSQL", reason="Team expertise")
```

### Codebase Indexing

First time on a project:
```
nmem_index(action="scan", path="./src")
```

Then `nmem_recall(query="authentication")` finds related code through the neural graph.

### Rules

1. **Be proactive** — remember important info without being asked
2. **Check memory first** — recall before asking questions the user may have answered before
3. **Use types** — categorize memories correctly (fact/decision/error/preference/todo/workflow)
4. **Set priority** — critical=7-10, normal=5, trivial=1-3
5. **Add tags** — organize by project/topic for better retrieval
6. **Recap on start** — always call `nmem_recap()` at session beginning
