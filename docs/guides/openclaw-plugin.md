# OpenClaw Plugin Setup

NeuralMemory replaces OpenClaw's built-in memory system (`memory-core`) with a
neural graph that survives context compaction, detects contradictions, and learns
from usage patterns.

## Why Replace memory-core?

OpenClaw memory is plain Markdown — `MEMORY.md` + `memory/YYYY-MM-DD.md`.
When a session hits the context window limit, compaction summarizes older messages
and **discards** what hasn't been written to disk. Any insight the agent didn't
explicitly save is lost.

NeuralMemory stores everything in a persistent SQLite neural graph **outside** the
context window. Memories survive compaction, session restarts, and device changes.

| Feature | memory-core | NeuralMemory |
|---------|-------------|--------------|
| Storage | Markdown files | SQLite neural graph |
| Search | Vector + BM25 (needs embedding API) | Spreading activation (zero cost) |
| Compaction-safe | No — unsaved context is lost | Yes — memories live outside context |
| Conflict detection | No | Auto-detect + resolution |
| Temporal reasoning | No | Causal chains + event sequences |
| Memory lifecycle | Static (keep forever or delete) | 4-stage (STM → Working → Episodic → Semantic) |
| Cross-session | Per-workspace only | Portable SQLite brains |
| Embedding cost | ~$0.02/1K queries | $0.00 |

## Prerequisites

- **OpenClaw** installed and running (`openclaw gateway`)
- **Python 3.11+** with pip
- **Node.js 18+** with npm

## Setup (3 Steps)

### Step 1: Install packages

```bash
pip install neural-memory
npm install neuralmemory
```

### Step 2: Configure the memory slot

Edit `~/.openclaw/openclaw.json` and set the memory slot to `neuralmemory`:

```json
{
  "plugins": {
    "slots": {
      "memory": "neuralmemory"
    }
  }
}
```

This disables `memory-core` and activates NeuralMemory as the exclusive memory
provider. OpenClaw plugin slots are **exclusive** — only one plugin per slot.

### Step 3: Restart the gateway

```bash
# If running as daemon
openclaw gateway restart

# Or stop + start
openclaw gateway stop
openclaw gateway
```

### Verify

Ask your agent:

```
What memory tools do you have?
```

The agent should list `nmem_remember`, `nmem_recall`, `nmem_context`, `nmem_todo`,
`nmem_stats`, and `nmem_health`. If it mentions `memory_search` or `memory_get`
instead, the slot config is not applied — check Step 2.

## How It Works

```
OpenClaw Agent
    │
    ▼ (tool call: nmem_recall)
OpenClaw Plugin (TypeScript, in-process)
    │
    ▼ JSON-RPC over stdio
NeuralMemory MCP Server (Python subprocess)
    │
    ▼
SQLite Neural Graph (~/.neuralmemory/brains/)
```

The plugin:

1. **Starts** a Python MCP subprocess (`python -m neural_memory.mcp`) when the
   gateway boots
2. **Registers 6 tools** directly into OpenClaw's tool system
3. **Before each agent run**: queries relevant memories and injects them as context
4. **After each agent run**: auto-captures decisions, errors, and insights from
   the conversation

## Plugin Configuration

Optional config under `plugins.entries.neuralmemory.config` in `openclaw.json`:

> **Important:** The entry name **must** be `neuralmemory` (no hyphen) — this
> matches the plugin ID. Using any other name (e.g. `neural-memory`, `telegram`)
> will cause an "Invalid value" error.

```json
{
  "plugins": {
    "slots": {
      "memory": "neuralmemory"
    },
    "entries": {
      "neuralmemory": {
        "config": {
          "pythonPath": "python",
          "brain": "default",
          "autoContext": true,
          "autoCapture": true,
          "contextDepth": 1,
          "maxContextTokens": 500,
          "timeout": 30000,
          "initTimeout": 90000
        }
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `pythonPath` | `"python"` | Path to Python executable with `neural-memory` installed |
| `brain` | `"default"` | Brain name for this workspace |
| `autoContext` | `true` | Inject relevant memories before each agent run |
| `autoCapture` | `true` | Extract and store memories after each agent run |
| `contextDepth` | `1` | Recall depth: 0=instant, 1=context, 2=habit, 3=deep |
| `maxContextTokens` | `500` | Maximum tokens for auto-context injection |
| `timeout` | `30000` | MCP request timeout in milliseconds |
| `initTimeout` | `90000` | MCP initialize handshake timeout (increase if first boot is slow) |

> **Only these keys are allowed in `config`.** The schema uses
> `additionalProperties: false` — any extra key (e.g. `"enabled"`, `"url"`)
> will trigger an "Invalid value" error from OpenClaw.

## Available Tools

Once configured, the agent has access to these tools:

| Tool | Description |
|------|-------------|
| `nmem_remember` | Store a memory (fact, decision, error, preference, etc.) |
| `nmem_recall` | Query memories via spreading activation |
| `nmem_context` | Get recent memories for context |
| `nmem_todo` | Quick TODO with 30-day expiry |
| `nmem_stats` | Brain statistics |
| `nmem_health` | Brain health diagnostics |

The plugin also injects a system prompt telling the agent to use `nmem_*` tools
exclusively and **not** use `memory_search` or `memory_get` from the disabled
`memory-core` plugin.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Invalid value` in plugin config | Unknown key or wrong entry name | Only use documented config keys. Entry name must be `neuralmemory` (no hyphen) |
| `no MCP Client` | Using SKILL.md with `mcp:` block | Skills don't support MCP. Use the Plugin approach (this guide) |
| `ENOENT: python not found` | Wrong Python path | Set `pythonPath` in plugin config to your Python binary |
| `MCP process exited with code 1` | `neural-memory` not installed | Run `pip install neural-memory` |
| Agent still uses `memory_search` | Slot not configured | Set `plugins.slots.memory = "neuralmemory"` in `openclaw.json` |
| Agent uses both `nmem_*` and `memory_*` | `memory-core` still active | Check slot config — only one memory plugin can be active |
| `MCP timeout` | Slow machine or large brain | Increase `timeout` in plugin config (default: 30000ms) |
| Plugin not found | Not installed globally | Run `npm install -g @neuralmemory/openclaw-plugin` |

## Common Mistakes

### Adding unknown keys to config

```json
// WRONG — "enabled" is not a valid config key → Invalid value
{
  "plugins": {
    "enabled": true,
    "slots": { "memory": "neuralmemory" },
    "entries": {
      "neuralmemory": {
        "config": { "enabled": true }
      }
    }
  }
}

// CORRECT — only use documented keys
{
  "plugins": {
    "slots": { "memory": "neuralmemory" },
    "entries": {
      "neuralmemory": {
        "config": { "pythonPath": "python", "brain": "default" }
      }
    }
  }
}
```

The plugin schema is strict (`additionalProperties: false`). Only these config
keys are accepted: `pythonPath`, `brain`, `autoContext`, `autoCapture`,
`contextDepth`, `maxContextTokens`, `timeout`.

### Using wrong entry name

```json
// WRONG — entry name must be "neuralmemory" (the plugin ID)
{ "plugins": { "entries": { "neural-memory": { "config": {} } } } }
{ "plugins": { "entries": { "telegram": { "enabled": true } } } }

// CORRECT
{ "plugins": { "entries": { "neuralmemory": { "config": {} } } } }
```

### Using `"memory": "none"`

```json
// WRONG — disables ALL memory plugins including NeuralMemory
{ "plugins": { "slots": { "memory": "none" } } }

// CORRECT — activates NeuralMemory, disables memory-core
{ "plugins": { "slots": { "memory": "neuralmemory" } } }
```

### Using SKILL.md with `mcp:` block

```markdown
# WRONG — OpenClaw skills don't have an MCP client
---
mcp:
  neural-memory:
    command: nmem-mcp
---
```

OpenClaw skills provide instructions to the LLM but cannot spawn MCP server
processes. The plugin approach bundles its own MCP client that communicates with
the NeuralMemory Python process over stdio.

### Adding rules to AGENTS.MD

```markdown
# WRONG — AGENTS.MD rules can't disable registered tools
Do NOT use memory_search. Use nmem_recall instead.
```

AGENTS.MD is an instruction to the model, not a tool access control. The model
may still call `memory_search` if `memory-core` is registered. The correct fix
is the slot config in Step 2 — it prevents `memory-core` from loading entirely.

## Further Reading

- [Quick Start](../getting-started/quickstart.md) — Basic NeuralMemory usage
- [CLI Reference](../getting-started/cli.md) — All commands and options
- [Integration Guide](integration.md) — Setup for Claude Code, Cursor, and other editors
- [MCP Server Guide](mcp-server.md) — MCP configuration for 20+ editors
