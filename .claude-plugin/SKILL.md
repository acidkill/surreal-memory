# Neural Memory

Reflex-based memory system for AI agents — stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works.

## What It Does

Neural Memory gives AI agents persistent, associative memory across sessions. Instead of keyword search, it uses spreading activation through a neural graph — memories that fire together, wire together.

## Key Features

- **45 MCP tools** for persistent memory + cognitive reasoning
- **Spreading activation recall** — not keyword search, memories activate related memories
- **Cognitive reasoning** — hypotheses, evidence, predictions, schema evolution
- **Knowledge base training** from PDF, DOCX, PPTX, HTML, JSON, XLSX, CSV
- **Multi-device sync** with neural-aware conflict resolution
- **4 embedding providers** — Sentence Transformers, Gemini, Ollama, OpenAI
- **Retrieval pipeline** — RRF score fusion, graph expansion, Personalized PageRank
- **Session intelligence** — topic EMA tracking, LRU eviction, auto-expiry
- **React dashboard** — 7 pages: health, evolution, graph, timeline, settings
- **VS Code extension** — status bar, graph explorer, CodeLens, memory tree
- **Fernet encryption** for sensitive content
- **Brain versioning** — snapshots, rollback, export/import
- **Telegram backup** — send brain .db to chat/group/channel

## Installation

```bash
pip install neural-memory
```

Or with embeddings:

```bash
pip install neural-memory[embeddings]
```

## MCP Configuration

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "uvx",
      "args": ["--from", "neural-memory", "nmem-mcp"]
    }
  }
}
```

## Usage

Neural Memory works automatically once configured.

**RECALL** — before responding to tasks that reference past work:
- New session → `nmem_recall("current project context")`
- Past decision/event → `nmem_recall("<project> <topic>")`
- Skip for purely new, self-contained questions

**SAVE** — after completing each task, if you made a decision, fixed a bug, learned a preference, or discovered a pattern:
- `nmem_remember(content="Chose X over Y because Z", type="decision", priority=7, tags=["project", "topic"])`
- Use causal language (not flat facts). Max 1-3 sentences.
- Do NOT save ephemeral file reads, things in git history, or duplicates.

**FLUSH** — at session end:
- `nmem_auto(action="process", text="brief summary")`

## Memory Types

| Type | Use For |
|------|---------|
| fact | Stable knowledge |
| decision | "Chose X over Y because Z" |
| insight | Patterns discovered |
| error | Bugs and root causes |
| workflow | Process steps |
| preference | User preferences |
| instruction | Rules to follow |

## Links

- [GitHub](https://github.com/nhadaututtheky/neural-memory)
- [Documentation](https://nhadaututtheky.github.io/neural-memory)
- [VS Code Extension](https://marketplace.visualstudio.com/items?itemName=neuralmem.neuralmemory)
