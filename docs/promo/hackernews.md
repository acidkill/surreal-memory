# Hacker News — Show HN

**Title:** Show HN: Surreal-Memory – Graph-based persistent memory for AI agents (spreading activation, not RAG)

**URL:** https://github.com/acidkill/surreal-memory

**Text (optional, for text post instead of URL post):**

Surreal-Memory is an open-source MCP server that gives AI coding agents persistent memory using a neural graph with spreading activation retrieval — instead of the usual RAG/vector search approach.

Core idea: memories are stored as typed neurons connected by 41 typed synapses (CAUSED_BY, LEADS_TO, RESOLVED_BY, CONTRADICTS, etc.). Recall works by activating seed neurons and letting activation spread through the graph, naturally surfacing related memories through association — closer to how the brain retrieves context than cosine similarity over a flat embedding store.

No embedding API calls needed for core recall — it's pure algorithmic graph traversal. Embeddings are optional for semantic (vector) search (supports Ollama, sentence-transformers, Gemini, OpenAI). Core encode + recall costs $0.00 per query.

Backed by SurrealDB's multi-model engine (document + graph + vector HNSW in one database — no separate vector store). 57 MCP tools, 7200+ tests, 67%+ CI coverage, Python 3.11+, MIT.

All Pro-tier features (HNSW vector search, smart merge, 5-tier compression, sleep consolidation) are free via the bundled community plugin — no license keys.

This is a fork of nhadaututtheky/neural-memory. It inherits the neuron/synapse/fiber architecture and spreading-activation model; it differs in backend (SurrealDB instead of SQLite) and in unlocking all Pro features for free.
