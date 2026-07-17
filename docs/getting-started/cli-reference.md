# CLI Reference

Complete reference for the `smem` command-line interface.
**80 commands** available.

!!! tip
    Run `smem --help` or `smem <command> --help` for the latest usage info.

## Table of Contents

- [Memory Operations](#memory)
  - [`smem remember`](#smem-remember)
  - [`smem recall`](#smem-recall)
  - [`smem context`](#smem-context)
  - [`smem todo`](#smem-todo)
  - [`smem q`](#smem-q)
  - [`smem a`](#smem-a)
  - [`smem last`](#smem-last)
  - [`smem today`](#smem-today)
- [Brain Management](#brain)
  - [`smem brain list`](#smem-brain-list)
  - [`smem brain use`](#smem-brain-use)
  - [`smem brain create`](#smem-brain-create)
  - [`smem brain export`](#smem-brain-export)
  - [`smem brain import`](#smem-brain-import)
  - [`smem brain delete`](#smem-brain-delete)
  - [`smem brain health`](#smem-brain-health)
  - [`smem brain transplant`](#smem-brain-transplant)
- [Information & Diagnostics](#info)
  - [`smem stats`](#smem-stats)
  - [`smem status`](#smem-status)
  - [`smem health`](#smem-health)
  - [`smem check`](#smem-check)
  - [`smem doctor`](#smem-doctor)
  - [`smem dashboard`](#smem-dashboard)
  - [`smem ui`](#smem-ui)
  - [`smem graph`](#smem-graph)
- [Training & Import/Export](#training)
  - [`smem train`](#smem-train)
  - [`smem index`](#smem-index)
  - [`smem import`](#smem-import)
  - [`smem export`](#smem-export)
- [Configuration & Setup](#config)
  - [`smem init`](#smem-init)
  - [`smem setup`](#smem-setup)
  - [`smem mcp-config`](#smem-mcp-config)
  - [`smem prompt`](#smem-prompt)
  - [`smem hooks`](#smem-hooks)
  - [`smem config preset`](#smem-config-preset)
  - [`smem config tier`](#smem-config-tier)
  - [`smem install-skills`](#smem-install-skills)
- [Server & MCP](#server)
  - [`smem serve`](#smem-serve)
  - [`smem mcp`](#smem-mcp)
- [Maintenance](#maintenance)
  - [`smem decay`](#smem-decay)
  - [`smem consolidate`](#smem-consolidate)
  - [`smem cleanup`](#smem-cleanup)
  - [`smem flush`](#smem-flush)
- [Project Management](#project)
  - [`smem project create`](#smem-project-create)
  - [`smem project list`](#smem-project-list)
  - [`smem project show`](#smem-project-show)
  - [`smem project delete`](#smem-project-delete)
  - [`smem project extend`](#smem-project-extend)
- [Advanced Features](#advanced)
  - [`smem shared enable`](#smem-shared-enable)
  - [`smem shared disable`](#smem-shared-disable)
  - [`smem shared status`](#smem-shared-status)
  - [`smem shared test`](#smem-shared-test)
  - [`smem shared sync`](#smem-shared-sync)
  - [`smem habits list`](#smem-habits-list)
  - [`smem habits show`](#smem-habits-show)
  - [`smem habits clear`](#smem-habits-clear)
  - [`smem habits status`](#smem-habits-status)
  - [`smem version create`](#smem-version-create)
  - [`smem version list`](#smem-version-list)
  - [`smem version rollback`](#smem-version-rollback)
  - [`smem version diff`](#smem-version-diff)
  - [`smem telegram status`](#smem-telegram-status)
  - [`smem telegram test`](#smem-telegram-test)
  - [`smem telegram backup`](#smem-telegram-backup)
  - [`smem list`](#smem-list)
  - [`smem update`](#smem-update)
- [Other](#other)
  - [`smem lifecycle`](#smem-lifecycle)
  - [`smem reasoning clear`](#smem-reasoning-clear)
  - [`smem reasoning mine`](#smem-reasoning-mine)
  - [`smem reasoning patterns`](#smem-reasoning-patterns)
  - [`smem reasoning status`](#smem-reasoning-status)
  - [`smem reindex`](#smem-reindex)
  - [`smem shared activate`](#smem-shared-activate)
  - [`smem storage status`](#smem-storage-status)
  - [`smem sync activate`](#smem-sync-activate)
  - [`smem sync disable`](#smem-sync-disable)
  - [`smem sync enable`](#smem-sync-enable)
  - [`smem sync status`](#smem-sync-status)
  - [`smem sync sync`](#smem-sync-sync)
  - [`smem sync test`](#smem-sync-test)
  - [`smem watch`](#smem-watch)

---

## Memory Operations {#memory}

### `smem remember`

Store a new memory (type auto-detected if not specified).

```
smem remember [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | No | `` | (positional argument) |
| `--tag / -t` | text | No | — | Tags for the memory |
| `--type / -T` | text | No | — | Memory type: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference (auto-detect... |
| `--priority / -p` | integer | No | — | Priority 0-10 (0=lowest, 5=normal, 10=critical) |
| `--expires / -e` | integer | No | — | Days until this memory expires |
| `--project / -P` | text | No | — | Associate with a project (by name) |
| `--shared / -S` | boolean | No | `False` | Use shared/remote storage for this command |
| `--force / -f` | boolean | No | `False` | Store even if sensitive content detected |
| `--redact / -r` | boolean | No | `False` | Auto-redact sensitive content before storing |
| `--timestamp / --at` | text | No | — | ISO datetime of original event (e.g. '2026-03-02T08:00:00'). Defaults to now. |
| `--ephemeral` | boolean | No | `False` | Session-scoped memory: auto-expires after 24h, never synced |
| `--stdin` | boolean | No | `False` | Read content from stdin (safe for shell-special characters) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem recall`

Query memories with intelligent routing (query type auto-detected).

```
smem recall [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | Yes | — | (positional argument) |
| `--depth / -d` | integer | No | — | Search depth (0=instant, 1=context, 2=habit, 3=deep) |
| `--max-tokens / -m` | integer | No | `500` | Max tokens in response |
| `--min-confidence / -c` | float | No | `0.0` | Minimum confidence threshold (0.0-1.0) |
| `--shared / -S` | boolean | No | `False` | Use shared/remote storage for this command |
| `--show-age / -a` | boolean | No | `True` | Show memory ages in results |
| `--show-routing / -R` | boolean | No | `False` | Show query routing info |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem context`

Get recent context (for injecting into AI conversations).

```
smem context [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--limit / -l` | integer | No | `10` | Number of recent memories |
| `--fresh-only` | boolean | No | `False` | Only include memories < 30 days old |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem todo`

Quick shortcut to add a TODO memory.

```
smem todo [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `task` | text | Yes | — | (positional argument) |
| `--priority / -p` | integer | No | `5` | Priority 0-10 (default: 5=normal, 7=high, 10=critical) |
| `--project / -P` | text | No | — | Associate with a project |
| `--expires / -e` | integer | No | — | Days until expiry (default: 30) |
| `--tag / -t` | text | No | — | Tags for the task |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem q`

Quick recall - shortcut for 'smem recall'.

```
smem q [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | Yes | — | (positional argument) |
| `-d` | integer | No | — | — |

### `smem a`

Quick add - shortcut for 'smem remember' with auto-detect.

```
smem a [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | Yes | — | (positional argument) |
| `-p` | integer | No | — | — |

### `smem last`

Show last N memories - quick view of recent activity.

```
smem last [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `-n` | integer | No | `5` | Number of memories to show |

### `smem today`

Show today's memories.

```
smem today [OPTIONS]
```

## Brain Management {#brain}

### `smem brain list`

List available brains.

```
smem brain list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem brain use`

Switch to a different brain.

```
smem brain use [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |

### `smem brain create`

Create a new brain.

```
smem brain create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--use / -u` | boolean | No | `True` | Switch to the new brain after creating |

### `smem brain export`

Export brain to JSON or markdown file.

```
smem brain export [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--output / -o` | text | No | — | Output file path |
| `--name / -n` | text | No | — | Brain name (default: current) |
| `--exclude-sensitive / -s` | boolean | No | `False` | Exclude memories with sensitive content |
| `--format / -f` | text | No | `json` | Export format: json or markdown |

### `smem brain import`

Import brain from JSON file.

```
smem brain import [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `file` | text | Yes | — | (positional argument) |
| `--name / -n` | text | No | — | Name for imported brain |
| `--use / -u` | boolean | No | `True` | Switch to imported brain |
| `--scan` | boolean | No | `True` | Scan for sensitive content before importing |

### `smem brain delete`

Delete a brain.

```
smem brain delete [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `smem brain health`

Check brain health (freshness, sensitive content).

```
smem brain health [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--name / -n` | text | No | — | Brain name (default: current) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem brain transplant`

Transplant memories from another brain into the current brain.

```
smem brain transplant [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `source` | text | Yes | — | (positional argument) |
| `--tag / -t` | text | No | — | Filter by tags |
| `--type` | text | No | — | Filter by memory types |
| `--strategy / -s` | text | No | `prefer_local` | Conflict resolution strategy |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Information & Diagnostics {#info}

### `smem stats`

Show brain statistics including freshness and memory type analysis.

```
smem stats [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem status`

Show current brain status, recent activity, and actionable suggestions.

```
smem status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem health`

Show brain health diagnostics with purity score and recommendations.

```
smem health [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem check`

Check content for sensitive information without storing.

```
smem check [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `content` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem doctor`

Run system health diagnostics.

```
smem doctor [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |
| `--fix` | boolean | No | `False` | Auto-fix available issues |
| `--dev` | boolean | No | `False` | Include source checkout and contributor tooling checks |
| `--synapse-migration` | text | No | — | Manage the synapse->RELATE migration: status \| retry \| purge-backup |

### `smem dashboard`

Show a rich dashboard with brain stats and recent activity.

```
smem dashboard [OPTIONS]
```

### `smem ui`

Interactive memory browser with rich formatting.

```
smem ui [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--type / -t` | text | No | — | Filter by memory type |
| `--search / -s` | text | No | — | Search in memory content |
| `--limit / -n` | integer | No | `20` | Number of memories to show |

### `smem graph`

Visualize neural connections as a tree graph.

```
smem graph [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `query` | text | No | — | (positional argument) |
| `--depth / -d` | integer | No | `2` | Traversal depth (1-3) |
| `--export / -e` | text | No | — | Export format: svg |
| `--output / -o` | text | No | — | Output file path (used with --export) |

## Training & Import/Export {#training}

### `smem train`

Train a brain from documentation files (markdown).

```
smem train [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `path` | text | No | `.` | (positional argument) |
| `--domain / -d` | text | No | `` | Domain tag (e.g., react, kubernetes) |
| `--brain / -b` | text | No | `` | Target brain name (default: current) |
| `--ext / -e` | text | No | — | File extensions (default: .md) |
| `--no-consolidate` | boolean | No | `False` | Skip ENRICH consolidation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem index`

Index a codebase into neural memory for code-aware recall.

```
smem index [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `path` | text | No | `.` | (positional argument) |
| `--ext / -e` | text | No | — | File extensions to index (e.g. .py) |
| `--status / -s` | boolean | No | `False` | Show indexing status instead of scanning |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem import`

Import brain from JSON file.

```
smem import [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `input_file` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Target brain name (default: from file) |
| `--merge / -m` | boolean | No | `False` | Merge with existing brain |
| `--strategy` | text | No | `prefer_local` | Conflict resolution: prefer_local, prefer_remote, prefer_recent, prefer_stronger |

### `smem export`

Export brain to JSON file for backup or sharing.

```
smem export [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `output` | text | Yes | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain to export (default: current) |

## Configuration & Setup {#config}

### `smem init`

Set up Surreal-Memory in one command.

```
smem init [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Overwrite existing config |
| `--skip-mcp` | boolean | No | `False` | Skip MCP auto-configuration |
| `--skip-skills` | boolean | No | `False` | Skip skills installation |
| `--wizard / -w` | boolean | No | `False` | Interactive setup wizard |
| `--defaults` | boolean | No | `False` | Non-interactive with all defaults |
| `--full` | boolean | No | `False` | Extended setup: embeddings, dedup, maintenance script |
| `--embeddings` | boolean | No | `False` | Set up embedding provider |
| `--skip-embeddings` | boolean | No | `False` | Skip embedding provider setup |
| `--dedup` | boolean | No | `False` | Enable dedup |

### `smem setup`

Set up optional components.

```
smem setup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `component` | text | No | `` | (positional argument) |
| `--ide` | text | No | `` | Target IDE for rules: cursor, windsurf, cline, gemini, agents |
| `--all` | boolean | No | `False` | Generate rules for all supported IDEs |
| `--force / -f` | boolean | No | `False` | Overwrite existing files |

### `smem mcp-config`

Generate MCP server configuration for Claude Code/Cursor.

```
smem mcp-config [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--with-prompt / -p` | boolean | No | `False` | Include system prompt in config |
| `--compact / -c` | boolean | No | `False` | Use compact prompt (if --with-prompt) |

### `smem prompt`

Show system prompt for AI tools.

```
smem prompt [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--compact / -c` | boolean | No | `False` | Show compact version |
| `--copy` | boolean | No | `False` | Copy to clipboard (requires pyperclip) |

### `smem hooks`

Install or manage git hooks for automatic memory capture.

```
smem hooks [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `action` | text | No | `install` | (positional argument) |
| `--path / -p` | text | No | — | Path to git repo (default: current dir) |

### `smem config preset`

Apply a configuration preset or list available presets.

```
smem config preset [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | No | `` | (positional argument) |
| `--list / -l` | boolean | No | `False` | List available presets |
| `--dry-run / -n` | boolean | No | `False` | Show changes without applying |

### `smem config tier`

Get or set the MCP tool tier to control token usage.

```
smem config tier [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | No | `` | (positional argument) |
| `--show / -s` | boolean | No | `False` | Show current tier |

### `smem install-skills`

Install Surreal-Memory skills to ~/.claude/skills/.

```
smem install-skills [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Overwrite existing skills with latest version |
| `--list / -l` | boolean | No | `False` | List available skills without installing |

## Server & MCP {#server}

### `smem serve`

Run the Surreal-Memory API server.

```
smem serve [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--host / -h` | text | No | `127.0.0.1` | Host to bind to |
| `--port / -p` | integer | No | `8000` | Port to bind to |
| `--reload / -r` | boolean | No | `False` | Enable auto-reload for development |

### `smem mcp`

Run the MCP (Model Context Protocol) server.

```
smem mcp [OPTIONS]
```

## Maintenance {#maintenance}

### `smem decay`

Apply memory decay to simulate forgetting.

```
smem decay [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | — | Brain to apply decay to |
| `--dry-run / -n` | boolean | No | `False` | Preview changes without applying |
| `--prune / -p` | float | No | `0.01` | Prune below this activation level |

### `smem consolidate`

Consolidate brain memories by pruning, merging, or summarizing.

```
smem consolidate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `strategy_positional` | text | No | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain to consolidate |
| `--strategy / -s` | text | No | `all` | Consolidation strategy. Valid values: prune, merge, summarize, mature, infer, enrich, dream, learn_habits, dedup, sem... |
| `--dry-run / -n` | boolean | No | `False` | Preview changes without applying |
| `--prune-threshold` | float | No | `0.05` | Weight threshold for pruning synapses |
| `--merge-overlap` | float | No | `0.5` | Jaccard overlap threshold for merging fibers |
| `--min-inactive-days` | float | No | `7.0` | Minimum inactive days before pruning |

### `smem cleanup`

Clean up expired or old memories.

```
smem cleanup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--expired / -e` | boolean | No | `True` | Only clean up expired memories |
| `--type / -T` | text | No | — | Only clean up specific memory type |
| `--dry-run / -n` | boolean | No | `False` | Show what would be deleted without deleting |
| `--force / -f` | boolean | No | `False` | Skip confirmation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem flush`

Emergency flush: capture memories before context is lost.

```
smem flush [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--transcript / -t` | text | No | — | Path to JSONL transcript file |
| `text` | text | No | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Project Management {#project}

### `smem project create`

Create a new project for organizing memories.

```
smem project create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--description / -d` | text | No | — | Project description |
| `--duration / -D` | integer | No | — | Duration in days (creates end date) |
| `--tag / -t` | text | No | — | Project tags |
| `--priority / -p` | float | No | `1.0` | Project priority (default: 1.0) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem project list`

List all projects.

```
smem project list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--active / -a` | boolean | No | `False` | Show only active projects |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem project show`

Show project details and its memories.

```
smem project show [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem project delete`

Delete a project (memories are preserved but unlinked).

```
smem project delete [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `smem project extend`

Extend a project's deadline.

```
smem project extend [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `days` | integer | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

## Advanced Features {#advanced}

### `smem shared enable`

Enable shared mode to connect to a remote Surreal-Memory server.

```
smem shared enable [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `server_url` | text | Yes | — | (positional argument) |
| `--api-key / -k` | text | No | — | API key for authentication |
| `--timeout / -t` | float | No | `30.0` | Request timeout in seconds |

### `smem shared disable`

Disable shared mode and use local storage.

```
smem shared disable [OPTIONS]
```

### `smem shared status`

Show shared mode status and configuration.

```
smem shared status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem shared test`

Test connection to the shared server.

```
smem shared test [OPTIONS]
```

### `smem shared sync`

Manually sync local brain with remote server.

```
smem shared sync [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--direction / -d` | text | No | `both` | Sync direction: push, pull, or both |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem habits list`

List learned workflow habits.

```
smem habits list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem habits show`

Show details of a specific learned habit.

```
smem habits show [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem habits clear`

Clear all learned habits.

```
smem habits clear [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Skip confirmation |

### `smem habits status`

Show progress toward habit detection.

```
smem habits status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem version create`

Create a version snapshot of the current brain state.

```
smem version create [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `name` | text | Yes | — | (positional argument) |
| `--description / -d` | text | No | `` | Description |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem version list`

List brain versions.

```
smem version list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--limit / -l` | integer | No | `20` | Max versions |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem version rollback`

Rollback brain to a previous version.

```
smem version rollback [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `version_id` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem version diff`

Compare two brain versions.

```
smem version diff [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `from_version` | text | Yes | — | (positional argument) |
| `to_version` | text | Yes | — | (positional argument) |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem telegram status`

Show Telegram integration status.

```
smem telegram status [OPTIONS]
```

### `smem telegram test`

Send a test message to verify configuration.

```
smem telegram test [OPTIONS]
```

### `smem telegram backup`

Send brain database file as backup to Telegram.

```
smem telegram backup [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | — | Brain name (default: active brain) |

### `smem list`

List memories with filtering by type, priority, project, and status.

```
smem list [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--type / -T` | text | No | — | Filter by memory type (fact, decision, todo, etc.) |
| `--min-priority / -p` | integer | No | — | Minimum priority (0-10) |
| `--project / -P` | text | No | — | Filter by project name |
| `--expired / -e` | boolean | No | `False` | Show only expired memories |
| `--include-expired` | boolean | No | `False` | Include expired memories in results |
| `--limit / -l` | integer | No | `20` | Maximum number of results |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem update`

Update surreal-memory to the latest version.

```
smem update [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--force / -f` | boolean | No | `False` | Force update even if already latest |
| `--check / -c` | boolean | No | `False` | Only check for updates, don't install |

## Other {#other}

### `smem lifecycle`

Manage memory lifecycle states.

```
smem lifecycle [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `action` | text | No | `status` | (positional argument) |
| `neuron_id` | text | No | — | (positional argument) |
| `--brain / -b` | text | No | — | Brain name |

### `smem reasoning clear`

Wipe staged reasoning traces for a model (privacy).

```
smem reasoning clear [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--model / -m` | text | Yes | — | Model whose staged traces to wipe |
| `--force / -f` | boolean | No | `False` | Skip confirmation |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem reasoning mine`

Mine reasoning traces from ~/.claude transcripts and distill patterns.

```
smem reasoning mine [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--backfill` | boolean | No | `False` | Scan the full history (scan_lookback_days=0) |
| `--dry-run / -n` | boolean | No | `False` | Don't write — report a no-op |
| `--models` | text | No | — | Comma-separated source models to restrict to |
| `--force / -f` | boolean | No | `False` | Run even if mining is disabled in config |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem reasoning patterns`

List learned reasoning patterns.

```
smem reasoning patterns [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--model / -m` | text | No | — | Filter by source model |
| `--category / -c` | text | No | — | Filter by category |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem reasoning status`

Show reasoning-training status: trace counts, learned patterns, per-model coverage.

```
smem reasoning status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem reindex`

Re-embed neurons for the current brain using the effective provider.

```
smem reindex [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--brain / -b` | text | No | `` | Target brain name (default: current) |
| `--dry-run` | boolean | No | `False` | Report how many would be embedded; write nothing |
| `--all` | boolean | No | `False` | Re-embed every neuron (default: only missing vectors) |
| `--batch-size` | integer | No | `64` | Neurons per embedding batch |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem shared activate`

Activate a Surreal-Memory Pro license key.

```
smem shared activate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--key / -k` | text | Yes | — | License key (NM-PRO-XXXX-XXXX-XXXX) |

### `smem storage status`

Show SurrealDB connection status and active brain.

```
smem storage status [OPTIONS]
```

### `smem sync activate`

Activate a Surreal-Memory Pro license key.

```
smem sync activate [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--key / -k` | text | Yes | — | License key (NM-PRO-XXXX-XXXX-XXXX) |

### `smem sync disable`

Disable shared mode and use local storage.

```
smem sync disable [OPTIONS]
```

### `smem sync enable`

Enable shared mode to connect to a remote Surreal-Memory server.

```
smem sync enable [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `server_url` | text | Yes | — | (positional argument) |
| `--api-key / -k` | text | No | — | API key for authentication |
| `--timeout / -t` | float | No | `30.0` | Request timeout in seconds |

### `smem sync status`

Show shared mode status and configuration.

```
smem sync status [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem sync sync`

Manually sync local brain with remote server.

```
smem sync sync [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--direction / -d` | text | No | `both` | Sync direction: push, pull, or both |
| `--json / -j` | boolean | No | `False` | Output as JSON |

### `smem sync test`

Test connection to the shared server.

```
smem sync test [OPTIONS]
```

### `smem watch`

Watch a directory and auto-ingest files into memory.

```
smem watch [OPTIONS]
```

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `directory` | text | No | `` | (positional argument) |
| `--action / -a` | text | No | `scan` | Action: scan, start, stop, status |
| `--json / -j` | boolean | No | `False` | Output as JSON |

---

*Auto-generated by `scripts/gen_cli_docs.py` from Typer app introspection — 80 commands.*
