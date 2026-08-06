# Surreal-Memory vs neural-memory

## A Fair Introduction

Surreal-Memory is a fork of [nhadaututtheky/neural-memory](https://github.com/nhadaututtheky/neural-memory). The neuron/synapse/fiber architecture, spreading-activation retrieval, and the MCP tool surface all originate upstream — that work deserves full credit.

This page explains what was inherited and what changed so you can make an informed choice between the two.

---

## What We Inherited

- Neurons, typed synapses, and the fiber graph model
- Spreading-activation recall (biological propagation model, not keyword/vector RAG)
- 41 explicit synapse types (`CAUSED_BY`, `LEADS_TO`, `RESOLVED_BY`, `CONTRADICTS`, …) for causal reasoning
- 15 memory types (`fact`, `decision`, `error`, `insight`, `preference`, `workflow`, `todo`, …)
- Memory lifecycle: decay (Ebbinghaus curve), Hebbian reinforcement, sleep-consolidation phases
- The 57-tool MCP surface — including the 3-tool core (`smem_remember`, `smem_recall`, `smem_health`)

---

## What We Changed

- **Storage backend replaced:** SurrealDB multi-model engine (document + graph + vector HNSW in one DB) instead of SQLite. No separate vector store, no `aiosqlite` dependency.
- **All Pro features unlocked — free.** Upstream gates cone/HNSW vector search, smart merge, and directional compression behind a paid Pro tier. In Surreal-Memory these ship as the bundled community plugin: no license key, no paywall, no tiers.
- **$0.00 per query for core ops.** No LLM or embedding API calls are required for core encode + recall. The system works offline (no API keys needed for core operations).
- **Active upstream port.** Storage-agnostic improvements from the upstream repo are ported on an ongoing basis as they reach a stable state.

---

## Side-by-Side Comparison

| Feature | neural-memory (upstream) | Surreal-Memory (fork) |
|---|---|---|
| **Storage backend** | SQLite / aiosqlite | SurrealDB (document + graph + vector) |
| **Vector search** | Pro tier only | Free — bundled community plugin (HNSW) |
| **Semantic recall** | Spreading activation | Spreading activation (inherited) |
| **Smart consolidation** | Pro tier only | Free — bundled community plugin |
| **Directional compression** | Pro tier only | Free — bundled community plugin |
| **License / pricing** | MIT; Pro features paywalled | MIT; all features free |
| **Multi-model DB** | No (single SQLite file) | Yes — SurrealDB (doc + graph + vector) |

---

## Acknowledgment

Surreal-Memory would not exist without the architecture, tooling, and ideas built by [@nhadaututtheky](https://github.com/nhadaututtheky). Thank you for open-sourcing neural-memory under MIT.

---

*Repo: [github.com/acidkill/surreal-memory](https://github.com/acidkill/surreal-memory) · PyPI: `surreal-memory` · CLI: `smem` · MIT · Python 3.11+*
