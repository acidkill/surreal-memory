# MCP Server Setup — All Editors & Clients

> **Copy-paste the config for your editor and you're done.**
> No `smem init` needed — the server auto-initializes on first use.

---

## Table of Contents

- [Requirements](#requirements)
- [Claude Code (Plugin)](#claude-code-plugin--recommended)
- [Claude Code (Manual MCP)](#claude-code-manual-mcp)
- [Cursor](#cursor)
- [Windsurf (Codeium)](#windsurf-codeium)
- [VS Code](#vs-code)
- [Claude Desktop](#claude-desktop)
- [Cline](#cline)
- [Zed](#zed)
- [Google Antigravity](#google-antigravity)
- [JetBrains IDEs](#jetbrains-ides-intellij-pycharm-webstorm)
- [Gemini CLI](#gemini-cli)
- [Amazon Q Developer](#amazon-q-developer)
- [Neovim](#neovim)
- [Warp Terminal](#warp-terminal)
- [Custom / Other MCP Clients](#custom-other-mcp-clients)
- [Alternative: Python Module](#alternative-python-module-directly)
- [Alternative: Docker](#alternative-docker)
- [Environment Variables](#environment-variables)
- [Resource Usage](#resource-usage)
- [Available Tools](#available-tools)
- [Resources](#resources)
- [Agent Instructions](#agent-instructions)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python 3.11+**
- **SurrealDB ≥ 3.2.0** — the storage backend (the bundled `docker-compose.surrealdb.yml` runs it)
- **pip** or **uv** package manager

> **Optional (internal GQL fast-path):** the compose file starts SurrealDB with
> `--allow-experimental gql --allow-eval-query`, which lets `get_path` use SurrealDB 3.2's
> ISO GQL shortest-path when the server exposes it. These flags are entirely optional —
> `get_path` falls back to BFS without them.

```bash
# Install via pip
pip install surreal-memory

# Or via uv (faster)
uv pip install surreal-memory
```

> **Note:** If using `uvx` (recommended for Claude Code), you don't need to install manually — `uvx` handles it automatically.

---

## Claude Code (Plugin — Recommended)

The easiest way. One command installs everything:

```bash
/plugin marketplace add acidkill/surreal-memory
/plugin install surreal-memory@surreal-memory-marketplace
```

This auto-configures the MCP server, skills, commands, agent, and hooks.

**Done.** No further setup needed.

---

## Claude Code (CLI — Recommended for Manual Setup)

The official way to add MCP servers to Claude Code:

```bash
# Global (all projects):
claude mcp add --scope user surreal-memory -- smem-mcp

# Or with uvx (no pip install needed):
claude mcp add --scope user surreal-memory -- uvx --from surreal-memory smem-mcp

# Project-only:
claude mcp add surreal-memory -- smem-mcp
```

**Alternatively**, add to your project's `.mcp.json` manually:

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

Or if you installed via pip (no `uvx`):

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

> **Note:** Do NOT add MCP servers to `~/.claude/settings.json` or `~/.claude/mcp_servers.json` — Claude Code does not read MCP config from those files. Use `claude mcp add` or `.mcp.json`.

---

## Cursor

Add to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project):

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx (no pip install needed):**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

Restart Cursor after adding the config.

---

## Windsurf (Codeium)

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

Restart Windsurf after adding.

---

## VS Code

### With Continue Extension

Add to `~/.continue/config.json` under `mcpServers`:

```json
{
  "mcpServers": [
    {
      "name": "surreal-memory",
      "command": "smem-mcp"
    }
  ]
}
```

### With Copilot Chat (MCP support)

Add to VS Code `settings.json`:

```json
{
  "mcp": {
    "servers": {
      "surreal-memory": {
        "command": "smem-mcp"
      }
    }
  }
}
```

### VS Code Extension (GUI)

For a graphical experience, install the [Surreal-Memory VS Code Extension](https://marketplace.visualstudio.com/items?itemName=neuralmem.surrealmemory) from the marketplace.

---

## Claude Desktop

Add to `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

**Windows — full path (if `smem-mcp` not in PATH):**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "python",
      "args": ["-m", "surreal_memory.mcp"]
    }
  }
}
```

Restart Claude Desktop after adding.

---

## Cline

Add to Cline MCP settings (`cline_mcp_settings.json` in your VS Code workspace):

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp",
      "disabled": false
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"],
      "disabled": false
    }
  }
}
```

---

## Zed

Add to Zed `settings.json` (`~/.config/zed/settings.json`):

```json
{
  "language_models": {
    "mcp_servers": {
      "surreal-memory": {
        "command": "smem-mcp"
      }
    }
  }
}
```

---

## Google Antigravity

Google's AI-powered editor with built-in MCP Store.

### Option 1: MCP Store (GUI)

1. Open the **MCP Store** via the `...` dropdown at the top of the editor's agent panel
2. Browse & install servers directly
3. Authenticate if prompted

### Option 2: Custom Config (for Surreal-Memory)

1. Open MCP Store → click **"Manage MCP Servers"**
2. Click **"View raw config"**
3. Add Surreal-Memory to `mcp_config.json`:

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

4. Save and restart the editor.

> **Tip:** Antigravity also supports connecting to Surreal-Memory's FastAPI server mode. Run `smem serve` and connect via HTTP if you prefer server-side integration.

---

## JetBrains IDEs (IntelliJ, PyCharm, WebStorm)

JetBrains IDEs support MCP via the built-in AI Assistant or the JetBrains AI plugin.

Go to **Settings → Tools → AI Assistant → MCP Servers → Add**, or edit the config file directly:

- **Location**: `.idea/mcpServers.json` (project) or global settings

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

Restart the IDE after adding.

---

## Gemini CLI

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

---

## Amazon Q Developer

Add to `~/.aws/amazonq/mcp.json`:

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

**With uvx:**

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "uvx",
      "args": ["--from", "surreal-memory", "smem-mcp"]
    }
  }
}
```

---

## Neovim

With [mcp-hub.nvim](https://github.com/ravitemer/mcphub.nvim) or similar MCP plugin, add to your `mcpservers.json`:

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

Or configure in Lua:

```lua
require("mcphub").setup({
  servers = {
    ["surreal-memory"] = {
      command = "smem-mcp",
    },
  },
})
```

---

## Warp Terminal

Add to Warp's MCP config (`~/.warp/mcp.json`):

```json
{
  "mcpServers": {
    "surreal-memory": {
      "command": "smem-mcp"
    }
  }
}
```

---

## Custom / Other MCP Clients

Surreal-Memory uses **stdio transport** (JSON-RPC 2.0 over stdin/stdout). Any MCP-compatible client can connect:

```json
{
  "name": "surreal-memory",
  "transport": "stdio",
  "command": "smem-mcp"
}
```

Or with explicit Python:

```json
{
  "name": "surreal-memory",
  "transport": "stdio",
  "command": "python",
  "args": ["-m", "surreal_memory.mcp"]
}
```

---

## Alternative: Python Module Directly

If `smem-mcp` is not in your PATH, use the Python module:

```json
{
  "surreal-memory": {
    "command": "python",
    "args": ["-m", "surreal_memory.mcp"]
  }
}
```

**macOS/Linux with specific Python:**

```json
{
  "surreal-memory": {
    "command": "python3",
    "args": ["-m", "surreal_memory.mcp"]
  }
}
```

**Windows with full path:**

```json
{
  "surreal-memory": {
    "command": "C:\\Users\\YOU\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
    "args": ["-m", "surreal_memory.mcp"]
  }
}
```

---

## Alternative: Docker

```bash
docker run -i --rm -v surrealmemory:/root/.surrealmemory ghcr.io/acidkill/surreal-memory:latest smem-mcp
```

```json
{
  "surreal-memory": {
    "command": "docker",
    "args": [
      "run", "-i", "--rm",
      "-v", "surrealmemory:/root/.surrealmemory",
      "ghcr.io/acidkill/surreal-memory:latest",
      "smem-mcp"
    ]
  }
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SURREAL_MEMORY_BRAIN` | `"default"` | Brain name to use |
| `SURREAL_MEMORY_DATA_DIR` | `~/.surrealmemory` | Data directory |
| `SURREAL_MEMORY_DEBUG` | `0` | Enable debug logging (`1` to enable) |
| `MEM0_API_KEY` | — | Mem0 API key (for import) |
| `COGNEE_API_KEY` | — | Cognee API key (for import) |

**Example with custom brain:**

```json
{
  "surreal-memory": {
    "command": "smem-mcp",
    "env": {
      "SURREAL_MEMORY_BRAIN": "work"
    }
  }
}
```

---

## Resource Usage

| Metric | Value |
|--------|-------|
| **RAM (idle)** | ~12-15 MB |
| **RAM (active, small brain)** | ~30-35 MB |
| **RAM (active, large brain)** | ~55-60 MB |
| **CPU** | Near 0% when idle |
| **Disk** | ~1-50 MB per brain (SurrealDB) |
| **Startup time** | < 2 seconds |

Surreal-Memory is lightweight — it won't slow down your editor.

---

## Available Tools

**3 tools you need. 53 the agent handles automatically.**

58 tools are available, but most users only interact with three:

### Essential (You Use These)

| Tool | What You Do |
|------|-------------|
| `smem_remember` | Tell the agent to remember something — auto-detects type, tags, connections |
| `smem_recall` | Ask the agent to recall — spreading activation surfaces related memories |
| `smem_health` | Check brain health — purity score, grade (A-F), actionable fix suggestions |

### Agent-Managed (Transparent)

These tools fire automatically via MCP instructions and hooks — you don't need to call them:

| Tool | When It Fires |
|------|---------------|
| `smem_context` | Session start — loads recent context |
| `smem_session` | Tracks task/feature/progress throughout session |
| `smem_recap` | Session start — restores saved context |
| `smem_auto` | Session end — captures remaining insights |
| `smem_suggest` | During recall — autocomplete from brain |
| `smem_habits` | Periodically — suggests workflow improvements |
| `smem_stats` | On demand — brain statistics |
| `smem_tool_stats` | On demand — tool usage analytics |
| `smem_alerts` | On health check — surfaces warnings |
| `smem_situation` | On resume — one-shot snapshot (active session, recent decisions, open blockers) |
| `smem_offload` | When tool output is large — stores it as an ephemeral ref + summary |
| `smem_inflate` | On demand — restores the full content of an offloaded ref |

### Power User (Opt-In)

#### Knowledge Base

| Tool | Description |
|------|-------------|
| `smem_train` | Train brain from docs (PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV, MD) |
| `smem_train_db` | Train brain from database schema |
| `smem_index` | Index codebase for code-aware recall |
| `smem_pin` | Pin/unpin memories (pinned = permanent, skip decay) |

#### Cognitive Reasoning

| Tool | Description |
|------|-------------|
| `smem_hypothesize` | Create hypotheses with Bayesian confidence tracking |
| `smem_evidence` | Submit evidence for/against — auto-updates confidence |
| `smem_predict` | Falsifiable predictions with deadlines |
| `smem_verify` | Verify predictions correct/wrong — propagates to hypotheses |
| `smem_cognitive` | Hot index: ranked active hypotheses + predictions |
| `smem_gaps` | Knowledge gap detection and tracking |
| `smem_schema` | Schema evolution: evolve hypotheses via SUPERSEDES chain |
| `smem_explain` | Trace shortest path between two concepts |

#### Analytics & Narrative

| Tool | Description |
|------|-------------|
| `smem_evolution` | Brain evolution metrics (maturation, plasticity) |
| `smem_narrative` | Generate timeline/topic/causal narratives |
| `smem_review` | Spaced repetition reviews (Leitner box system) |
| `smem_drift` | Detect and manage semantic drift in tags |

### Admin (Maintenance)

| Tool | Description |
|------|-------------|
| `smem_edit` | Edit memory type, content, or priority |
| `smem_forget` | Soft delete (set expiry) or hard delete |
| `smem_todo` | Quick TODO with 30-day expiry |
| `smem_eternal` | Save project context, decisions, instructions |
| `smem_version` | Brain version control (snapshot, rollback, diff) |
| `smem_transplant` | Copy memories between brains |
| `smem_conflicts` | View and resolve memory conflicts |
| `smem_import` | Import from ChromaDB, Mem0, Cognee, Graphiti, LlamaIndex |
| `smem_sync` | Cloud sync: push, pull, full, or seed |
| `smem_sync_status` | Show pending changes, devices, last sync |
| `smem_sync_config` | Configure hub URL, auto-sync, conflict strategy |
| `smem_telegram_backup` | Send brain backup to Telegram |

---

## Tool Tiers

By default all 58 tools are exposed on every API turn. If you want to reduce token overhead, configure a **tool tier** in `~/.surrealmemory/config.toml`:

```toml
[tool_tier]
tier = "standard"   # minimal | standard | full
```

Or via CLI:

```bash
smem config tier --show       # show current tier
smem config tier standard     # set to standard
smem config tier full         # reset to full
```

| Tier | Tools | Est. Tokens | Savings |
|------|-------|-------------|---------|
| `full` (default) | 26 | ~3,800 | — |
| `standard` | 8 | ~1,400 | ~63% |
| `minimal` | 4 | ~700 | ~82% |

**Tier contents:**

- **minimal** — `remember`, `recall`, `context`, `recap`
- **standard** — minimal + `todo`, `session`, `auto`, `eternal`
- **full** — all 58 tools

> Hidden tools remain callable — only the schema listing changes. If the AI model already knows a tool name, it can still call it even when the tool is not exposed in `tools/list`.

---

## Resources

The MCP server provides resources for system prompts:

| Resource URI | Description |
|-------------|-------------|
| `surrealmemory://prompt/system` | Full system prompt for AI assistants |
| `surrealmemory://prompt/compact` | Compact version for token-limited contexts |

### Get MCP Config via CLI

```bash
smem mcp-config
```

### View System Prompt via CLI

```bash
smem prompt            # Full prompt
smem prompt --compact  # Compact version
smem prompt --json     # As JSON
```

---

## Agent Instructions

Copy these instructions into your project's `CLAUDE.md` (for Claude Code) or `.cursorrules` (for Cursor) to teach your AI assistant how to use Surreal-Memory proactively.

### For Claude Code

See [`docs/agent-instructions/CLAUDE.md`](../agent-instructions/CLAUDE.md) for the full template.

### For Cursor

See [`docs/agent-instructions/.cursorrules`](../agent-instructions/.cursorrules) for the full template.

### Quick Version (any editor)

```markdown
## Memory System — Surreal-Memory

This workspace uses Surreal-Memory for persistent memory.
Use smem_* MCP tools PROACTIVELY.

### Session Start (ALWAYS)
1. smem_recap() — Resume context
2. smem_context(limit=20, fresh_only=true) — Recent memories
3. smem_session(action="get") — Current task

### Auto-Remember
- Decision made → smem_remember(content="...", type="decision", priority=7)
- Bug fixed → smem_remember(content="...", type="error", priority=7)
- TODO found → smem_todo(task="...", priority=6)

### Auto-Recall
Before asking user → smem_recall(query="<topic>", depth=1)

### Session End
smem_auto(action="process", text="<session summary>")
smem_session(action="set", feature="...", progress=0.8)
```

---

## Troubleshooting

### "smem-mcp" not found

```bash
# Check if installed
pip show surreal-memory

# Check if smem-mcp is in PATH
which smem-mcp    # macOS/Linux
where smem-mcp    # Windows

# If not found, use Python module instead
python -m surreal_memory.mcp
```

### Tools not appearing in editor

1. Verify the MCP config file path is correct for your editor
2. Restart the editor completely
3. Check editor logs for MCP connection errors
4. Test manually: `echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | smem-mcp`

### Python version mismatch

```bash
# Surreal-Memory requires Python 3.11+
python --version

# If you have multiple Python versions, specify the full path
```

### Windows: encoding errors

Surreal-Memory handles Windows stdio encoding automatically. If you still see encoding issues:

```json
{
  "surreal-memory": {
    "command": "python",
    "args": ["-m", "surreal_memory.mcp"],
    "env": {
      "PYTHONIOENCODING": "utf-8"
    }
  }
}
```

### Permission denied (macOS/Linux)

```bash
chmod +x $(which smem-mcp)
```

### uvx not found

```bash
# Install uv first
pip install uv

# Or use pipx
pipx install surreal-memory
```

### Debug mode

```bash
# Run with debug logging
SURREAL_MEMORY_DEBUG=1 smem-mcp
```

### Reset to fresh state

```bash
# macOS/Linux
rm -rf ~/.surrealmemory

# Windows
rmdir /s /q %USERPROFILE%\.surrealmemory
```

---

## Quick Reference

| Editor | Config File | Config Format |
|--------|-------------|---------------|
| **Claude Code** | `claude mcp add` or `.mcp.json` | `{ "mcpServers": { ... } }` |
| **Cursor** | `~/.cursor/mcp.json` | `{ "mcpServers": { ... } }` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` | `{ "mcpServers": { ... } }` |
| **Claude Desktop** | See [path above](#claude-desktop) | `{ "mcpServers": { ... } }` |
| **VS Code (Continue)** | `~/.continue/config.json` | `{ "mcpServers": [ ... ] }` |
| **VS Code (Copilot)** | VS Code `settings.json` | `{ "mcp": { "servers": { ... } } }` |
| **Cline** | `cline_mcp_settings.json` | `{ "mcpServers": { ... } }` |
| **Zed** | `~/.config/zed/settings.json` | `{ "language_models": { "mcp_servers": { ... } } }` |
| **Antigravity** | `mcp_config.json` (via MCP Store) | `{ "mcpServers": { ... } }` |
| **JetBrains** | `.idea/mcpServers.json` | `{ "mcpServers": { ... } }` |
| **Gemini CLI** | `~/.gemini/settings.json` | `{ "mcpServers": { ... } }` |
| **Amazon Q** | `~/.aws/amazonq/mcp.json` | `{ "mcpServers": { ... } }` |
| **Neovim** | `mcpservers.json` (plugin-dependent) | `{ "mcpServers": { ... } }` |
| **Warp** | `~/.warp/mcp.json` | `{ "mcpServers": { ... } }` |

**Minimum config for any editor:**

```json
{
  "surreal-memory": {
    "command": "smem-mcp"
  }
}
```

That's it. Copy, paste, restart. Done.
