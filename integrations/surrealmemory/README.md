# Surreal-Memory — OpenClaw Plugin

Brain-inspired persistent memory for AI agents. Stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works.

This is the **OpenClaw plugin** for [Surreal-Memory](https://github.com/acidkill/surreal-memory).

## Prerequisites

```bash
pip install surreal-memory
```

Python 3.11+ required. Verify the install:

```bash
smem-mcp --help
```

## Install

```bash
npm install -g surrealmemory
```

Or add to your OpenClaw config directly.

## OpenClaw Setup

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": {
      "memory": "surrealmemory"
    },
    "entries": {
      "surrealmemory": {
        "config": {
          "pythonPath": "python",
          "brain": "default",
          "autoContext": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

> **Important**: Setting `slots.memory = "surrealmemory"` disables the default `memory-core` plugin. Without this, agents may still use `memory_search` instead of Surreal-Memory tools.

## Tools

**v1.7.0+**: The plugin dynamically fetches **all tools** from the MCP server at startup. Whatever version of `surreal-memory` you have installed, the plugin automatically exposes every tool it provides — no plugin update needed when new tools are added.

With `surreal-memory>=4.6.0`, this includes **57 tools**:

| Category | Tools |
|----------|-------|
| **Core** | `smem_remember`, `smem_remember_batch`, `smem_recall`, `smem_context`, `smem_todo`, `smem_stats` |
| **Management** | `smem_edit`, `smem_forget`, `smem_pin`, `smem_health`, `smem_evolution`, `smem_alerts` |
| **Recall** | `smem_suggest`, `smem_narrative`, `smem_explain`, `smem_recap` |
| **Workflow** | `smem_session`, `smem_eternal`, `smem_auto`, `smem_habits`, `smem_review` |
| **Cognitive** | `smem_hypothesize`, `smem_evidence`, `smem_predict`, `smem_verify`, `smem_cognitive`, `smem_gaps`, `smem_schema` |
| **Training** | `smem_train`, `smem_train_db`, `smem_index`, `smem_import` |
| **Sync** | `smem_sync`, `smem_sync_status`, `smem_sync_config`, `smem_telegram_backup` |
| **Infra** | `smem_version`, `smem_transplant`, `smem_conflicts` |

If the MCP server is unreachable at startup, the plugin falls back to 5 core tools (remember, recall, context, stats, health) that auto-reconnect on first use.

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pythonPath` | string | `"python"` | Python executable with `surreal-memory` installed |
| `brain` | string | `"default"` | Brain name (each workspace can have its own) |
| `autoContext` | boolean | `true` | Auto-inject relevant memories before each agent run |
| `autoCapture` | boolean | `true` | Auto-extract memories after each agent run |
| `contextDepth` | integer | `1` | Recall depth: 0=instant, 1=context, 2=habits, 3=deep |
| `maxContextTokens` | integer | `500` | Max tokens for auto-context injection |
| `timeout` | integer | `30000` | MCP request timeout (ms) |

## How It Works

```
OpenClaw Agent
    |
    v
Surreal-Memory Plugin (this package)
    |  Spawns + manages lifecycle
    v
smem-mcp (Python MCP server, stdio transport)
    |
    v
~/.surrealmemory/brains/<brain>.db (SQLite)
```

The plugin spawns `smem-mcp` as a subprocess and communicates via JSON-RPC over stdio. Memories are stored in a local SQLite database.

## Troubleshooting

**Timeout on startup**: If you see `MCP timeout: initialize (30000ms)`, the Python process is slow to start. Fix:

```bash
# Pre-install to avoid cold start delays
pip install surreal-memory

# Or increase the timeout in your config
"timeout": 60000
```

**"smem-mcp not found"**: Ensure `surreal-memory` is installed in the Python environment that `pythonPath` points to.

**Schema validation errors**: Upgrade to plugin `>=1.7.0` — schemas are now normalized for strict providers (Anthropic SDK, OpenAI strict mode, Gemini). The plugin strips constraint keywords, ensures `additionalProperties: false`, and adds missing `properties` fields automatically.

## How Schema Normalization Works

The plugin normalizes MCP schemas for cross-provider compatibility:

- Strips `minimum`, `maximum`, `maxLength`, `maxItems` (rejected by some providers)
- Replaces `integer` → `number` (Gemini compatibility)
- Adds `additionalProperties: false` to all objects (OpenAI strict mode)
- Ensures every object type has a `properties` field (Anthropic SDK requirement)

This means the MCP server can use full JSON Schema features while the plugin ensures the schemas work with any LLM provider.

## Claude Code (MCP Direct)

For Claude Code users, you can skip the plugin and use MCP directly for the full toolset:

```bash
claude mcp add --scope user surreal-memory -- smem-mcp
```

## Links

- [Surreal-Memory on GitHub](https://github.com/acidkill/surreal-memory)
- [Surreal-Memory on PyPI](https://pypi.org/project/surreal-memory/)
- [Documentation](https://nhadaututheky.github.io/surreal-memory/)

## License

MIT
