# Anthropic Discord — #mcp-servers or #show-your-work

**Message:**

Hey! I built **Surreal-Memory** — an MCP server that gives Claude Code (and other MCP clients) persistent, graph-based memory across sessions.

Instead of RAG/vector search, it uses **spreading activation on a neural graph** — memories are typed neurons connected by 41 typed synapse types (CAUSED_BY, LEADS_TO, RESOLVED_BY, CONTRADICTS, ...) and recall works by activating related concepts through the graph, not keyword matching.

**Quick setup (Claude Code):**
```
/plugin marketplace add acidkill/surreal-memory
```
Or pip + any MCP client:
```bash
pip install "surreal-memory[surrealdb]"
smem-mcp  # starts the MCP server
```

**What you get:**
- **58 MCP tools** — 3-tool core (remember / recall / health), the rest fire automatically
- **SurrealDB backend** — document + graph + vector HNSW in one database, no separate vector store
- **$0.00/query** — no LLM/embedding API calls for core encode + recall; no API keys required for core ops
- Memory lifecycle: Ebbinghaus decay, Hebbian reinforcement, sleep consolidation (ENRICH / PRUNE / MERGE / DREAM)
- All Pro-tier features free via the bundled community plugin (cone/HNSW vector search, smart merge, directional compression) — no license keys, no paywalls
- 15 memory types · single-file portable brain · JSON export/import · multi-device sync via your own Cloudflare account

**MIT · Python 3.11+ · 7200+ unit tests**

GitHub: https://github.com/acidkill/surreal-memory

*Forked from nhadaututtheky/neural-memory — inherits the neuron/synapse/spreading-activation architecture; differs by swapping SQLite for SurrealDB and unlocking all Pro features free.*

Happy to answer questions!
