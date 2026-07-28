# CLI Guide

Guide to using the Surreal-Memory CLI with examples and common workflows.

!!! info "See also"
    For a complete auto-generated reference of all 66 commands, see the [CLI Reference](cli-reference.md).
    For MCP tool usage in Claude Code, see the [MCP Tools Reference](../api/mcp-tools.md).

## Core Commands

### smem remember

Store a memory in the brain.

```bash
smem remember "content" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--tag` | `-t` | Tags for the memory (repeatable) |
| `--type` | `-T` | Memory type (auto-detected if not specified) |
| `--priority` | `-p` | Priority 0-10 (0=lowest, 5=normal, 10=critical) |
| `--expires` | `-e` | Days until expiry |
| `--project` | `-P` | Associate with project |
| `--shared` | `-S` | Use shared/remote storage |
| `--force` | `-f` | Store even if sensitive content detected |
| `--redact` | `-r` | Auto-redact sensitive content |
| `--json` | `-j` | Output as JSON |

**Examples:**

```bash
smem remember "Fixed auth bug with null check"
smem remember "We decided to use PostgreSQL" --type decision
smem remember "Refactor auth module" --type todo --priority 7
smem remember "Meeting notes" --expires 7 --tag meeting
smem remember "Team knowledge" --shared
```

### smem recall

Query memories using spreading activation.

```bash
smem recall "query" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--depth` | `-d` | Search depth (0=instant, 1=context, 2=habit, 3=deep) |
| `--max-tokens` | `-m` | Max tokens in response (default: 500) |
| `--min-confidence` | `-c` | Minimum confidence threshold |
| `--shared` | `-S` | Use shared/remote storage |
| `--show-age` | `-a` | Show memory ages (default: true) |
| `--show-routing` | `-R` | Show query routing info |
| `--json` | `-j` | Output as JSON |

**Examples:**

```bash
smem recall "auth bug fix"
smem recall "meetings with Alice" --depth 2
smem recall "Why did the build fail?" --show-routing
smem recall "team decisions" --shared --min-confidence 0.5
```

### smem todo

Quick shortcut for TODO items.

```bash
smem todo "task" [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--priority` | `-p` | Priority 0-10 (default: 5) |
| `--project` | `-P` | Associate with project |
| `--expires` | `-e` | Days until expiry (default: 30) |
| `--tag` | `-t` | Tags (repeatable) |
| `--json` | `-j` | Output as JSON |

### smem context

Get recent memories for context injection.

```bash
smem context [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--limit` | `-l` | Number of recent memories (default: 10) |
| `--fresh-only` | | Only include memories < 30 days old |
| `--json` | `-j` | Output as JSON |

### smem list

List memories with filters.

```bash
smem list [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-T` | Filter by memory type |
| `--min-priority` | `-p` | Minimum priority |
| `--project` | `-P` | Filter by project |
| `--expired` | `-e` | Show only expired memories |
| `--include-expired` | | Include expired in results |
| `--limit` | `-l` | Maximum results (default: 20) |
| `--json` | `-j` | Output as JSON |

### smem stats

Show brain statistics.

```bash
smem stats [--json]
```

### smem check

Check content for sensitive information.

```bash
smem check "content" [--json]
```

### smem cleanup

Clean expired memories.

```bash
smem cleanup [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--expired` | `-e` | Only clean expired (default: true) |
| `--type` | `-T` | Only clean specific type |
| `--dry-run` | `-n` | Preview without deleting |
| `--force` | `-f` | Skip confirmation |

### smem consolidate

Consolidate brain memories: prune weak links, merge overlapping fibers,
advance episodic memories to semantic stage, and more.

```bash
smem consolidate [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--brain` | `-b` | Brain to consolidate (default: current) |
| `--strategy` | `-s` | Strategy to run (default: `all`) |
| `--dry-run` | `-n` | Preview without applying changes |
| `--prune-threshold` | | Synapse weight threshold for pruning (default: 0.05) |
| `--merge-overlap` | | Jaccard overlap threshold for merging (default: 0.5) |
| `--min-inactive-days` | | Minimum inactive days before pruning (default: 7.0) |

**Valid strategies:**

| Strategy | Description |
|----------|-------------|
| `prune` | Remove weak synapses and orphaned neurons |
| `merge` | Combine overlapping fibers |
| `summarize` | Create concept neurons for topic clusters |
| `mature` | Advance episodic memories to semantic stage |
| `infer` | Add inferred synapses from co-activation patterns |
| `enrich` | Enrich neurons with extracted metadata |
| `dream` | Generate synthetic bridging memories |
| `learn_habits` | Extract recurring workflow patterns |
| `dedup` | Link near-duplicates via alias edges (does not merge) |
| `semantic_link` | Add cross-domain semantic connections |
| `compress` | Compress old low-activation fibers |
| `all` | Run all strategies in dependency order (default) |

> **Note:** `mature` is a fully supported strategy. It advances episodic memories
> to the semantic stage, which improves recall quality. `smem health` may recommend
> running it when the consolidation ratio is low.

**Examples:**

```bash
smem consolidate                          # Run all strategies
smem consolidate --strategy prune         # Only prune weak links
smem consolidate -s mature                # Advance episodic memories
smem consolidate --dry-run                # Preview without changes
smem consolidate -s merge --merge-overlap 0.3
```

> **Tip:** Always use `--strategy <name>` (named flag). Positional syntax
> (`smem consolidate prune`) is not supported and will produce a helpful error.

### smem decay

Apply memory decay (Ebbinghaus forgetting curve).

```bash
smem decay [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--brain` | Brain name (default: current) |
| `--dry-run` | Preview without applying |
| `--prune-threshold` | Threshold for pruning (default: 0.01) |

---

## Brain Commands

### smem brain list

List all brains.

```bash
smem brain list [--json]
```

### smem brain create

Create a new brain.

```bash
smem brain create NAME [--use/--no-use]
```

### smem brain use

Switch to a brain.

```bash
smem brain use NAME
```

### smem brain export

Export brain to file.

```bash
smem brain export [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path |
| `--name` | `-n` | Brain name (default: current) |
| `--exclude-sensitive` | `-s` | Exclude sensitive content |

### smem brain import

Import brain from file.

```bash
smem brain import FILE [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Name for imported brain |
| `--use` | `-u` | Switch to imported brain |
| `--merge` | | Merge with existing brain |
| `--scan` | | Scan for sensitive content |

### smem brain delete

Delete a brain.

```bash
smem brain delete NAME [--force]
```

### smem brain health

Check brain health.

```bash
smem brain health [--name NAME] [--json]
```

---

## Project Commands

### smem project create

Create a project.

```bash
smem project create NAME [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--description` | `-d` | Project description |
| `--duration` | `-D` | Duration in days |
| `--tag` | `-t` | Tags (repeatable) |
| `--priority` | `-p` | Priority (default: 1.0) |

### smem project list

List projects.

```bash
smem project list [--active] [--json]
```

### smem project show

Show project details.

```bash
smem project show NAME [--json]
```

### smem project delete

Delete a project.

```bash
smem project delete NAME [--force]
```

### smem project extend

Extend project deadline.

```bash
smem project extend NAME DAYS [--json]
```

---

## Shared Mode Commands

### smem shared enable

Enable shared storage mode.

```bash
smem shared enable URL [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--api-key` | `-k` | API key for authentication |
| `--timeout` | `-t` | Request timeout in seconds |

### smem shared disable

Disable shared mode.

```bash
smem shared disable
```

### smem shared status

Show shared mode status.

```bash
smem shared status [--json]
```

### smem shared test

Test server connection.

```bash
smem shared test
```

### smem shared sync

Sync with server.

```bash
smem shared sync [--direction push|pull|both] [--json]
```

---

## Telegram Commands

### smem telegram status

Show Telegram integration configuration status.

```bash
smem telegram status [--json]
```

Shows: bot token configured (yes/no), bot name/username, chat IDs, backup-on-consolidation flag.

### smem telegram test

Send a test message to all configured Telegram chats.

```bash
smem telegram test [--json]
```

Verifies bot token and chat IDs are working. Sends a "Surreal-Memory test" message.

### smem telegram backup

Send brain .db file as backup to all configured Telegram chats.

```bash
smem telegram backup [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--brain` | `-b` | Brain name (default: current) |
| `--json` | `-j` | Output as JSON |

**Setup:**

1. Set bot token: `export SURREAL_MEMORY_TELEGRAM_BOT_TOKEN="your-bot-token"`
2. Add chat IDs to `~/.surrealmemory/config.toml`:

```toml
[telegram]
enabled = true
chat_ids = ["123456789"]
```

**Examples:**

```bash
smem telegram status               # Check config
smem telegram test                  # Verify bot works
smem telegram backup                # Backup current brain
smem telegram backup --brain work   # Backup specific brain
```

---

## Server Commands

### smem serve

Start the FastAPI server.

```bash
smem serve [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--host` | Host to bind (default: 127.0.0.1) |
| `--port` | Port to bind (default: 8000) |
| `--reload` | Enable auto-reload for development |

### smem mcp

Start MCP server for Claude integration.

```bash
smem mcp
```

### smem prompt

Show MCP system prompt.

```bash
smem prompt [--compact] [--json]
```

### smem mcp-config

Show MCP configuration.

```bash
smem mcp-config
```

### smem install-skills

Install bundled agent skills to `~/.claude/skills/`.

```bash
smem install-skills [OPTIONS]
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Overwrite existing skills with latest version |
| `--list` | `-l` | List available skills without installing |

**Examples:**

```bash
smem install-skills            # Install all skills
smem install-skills --force    # Overwrite with latest
smem install-skills --list     # Show available skills
```

---

## Memory Types

| Type | Description | Default Expiry |
|------|-------------|----------------|
| `fact` | Objective information | Never |
| `decision` | Choices made | Never |
| `preference` | User preferences | Never |
| `todo` | Action items | 30 days |
| `insight` | Learned patterns | Never |
| `context` | Situational info | 7 days |
| `instruction` | User guidelines | Never |
| `error` | Error patterns | Never |
| `workflow` | Process patterns | Never |
| `reference` | External references | Never |

## Depth Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Instant | Direct recall (who, what, where) |
| 1 | Context | Before/after context (2-3 hops) |
| 2 | Habit | Cross-time patterns |
| 3 | Deep | Full causal/emotional analysis |
