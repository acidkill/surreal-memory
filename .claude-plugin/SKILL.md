# Neural Memory

Reflex-based memory system for AI agents — stores experiences as interconnected neurons and recalls them through spreading activation, mimicking how the human brain works.

## What It Does

Neural Memory gives AI agents persistent, associative memory across sessions. Instead of keyword search, it uses spreading activation through a neural graph — memories that fire together, wire together.

## Key Features

- **39 MCP tools** for persistent memory + cognitive reasoning
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

Neural Memory works automatically once configured. The agent should:

1. **Session start**: `nmem_recall("current project context")` to load past context
2. **After each task**: `nmem_remember("Chose X over Y because Z")` to save decisions
3. **Session end**: `nmem_auto(action="process", text="summary")` to flush context

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
