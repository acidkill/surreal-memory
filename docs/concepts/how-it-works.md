# How Surreal-Memory Works

Surreal-Memory uses a fundamentally different approach to memory retrieval than traditional search or RAG systems, powered by SurrealDB's multi-model engine (document + graph + vector).

## The Core Idea

**Human memory doesn't work like search.**

You don't query your brain with:
```sql
SELECT * FROM memories WHERE content LIKE '%Alice%' ORDER BY similarity DESC
```

Instead, thinking of "Alice" *activates* related memories - her face, your last conversation, the project you worked on together. These emerge through **association**, not **search**.

Surreal-Memory replicates this process:

```
Query: "What did Alice suggest?"
         │
         ▼
┌─────────────────────┐
│ 1. Decompose Query  │  → time hints, entities, intent
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. Find Anchors     │  → "Alice" neuron
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Spread Activation│  → activate connected neurons
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 4. Find Intersection│  → high-activation subgraph
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 5. Extract Context  │  → "Alice suggested rate limiting"
└─────────────────────┘
```

## SurrealDB Architecture

Surreal-Memory uses **SurrealDB** as its primary storage backend, leveraging all three of its query models in a single database:

### Document Store

Neurons, fibers, and synapses are stored as structured documents with typed fields, indexes, and metadata. This provides fast lookups by content, type, brain ID, and time range.

```
┌─────────────────────────────────────────┐
│ SurrealDB Document Layer                │
│                                         │
│  neuron        → content, type, hash,   │
│                  metadata, embeddings    │
│  neuron_state  → activation, decay,     │
│                  thresholds             │
│  fiber         → pathway, conductivity, │
│                  salience, tags         │
│  typed_memory  → priority, tags,        │
│                  trust, expiry          │
└─────────────────────────────────────────┘
```

### Graph Queries

Synapses are stored as SurrealDB graph edges using `RELATE`, enabling native graph traversal and path finding between neurons. A single query can traverse multi-hop relationships without multiple round-trips.

```
neuron:Alice ──connects_to──► neuron:Meeting ──connects_to──► neuron:RateLimiting
      │                            │
      └── type: DISCUSSED ────────┘── type: SUGGESTED_BY
```

Graph operations used by Surreal-Memory:
- **Neighbor lookup** - Find all neurons connected to a given neuron (in/out/both directions)
- **Path finding** - BFS shortest-path between two neurons across multi-hop synapse chains
- **Traversal with filtering** - Filter by synapse type, weight, or direction during traversal

### HNSW Vector Search

The community plugin enables SurrealDB's native HNSW (Hierarchical Navigable Small World) vector search for semantic recall. Neurons store embedding vectors, and the `vector::distance::knn()` function finds nearest neighbors by similarity.

```
Query embedding: [0.12, -0.34, 0.56, ...]
         │
         ▼
  SurrealDB KNN search
  SELECT *, vector::distance::knn() AS score
  FROM neuron WHERE embedding_vec <|10,100|> $vec
         │
         ▼
  Ranked results by semantic similarity
```

This powers **cone queries** - a retrieval strategy that combines vector similarity with graph traversal for high-precision semantic recall.

## Key Components

### Neurons

Neurons are atomic units of information:

- **Entity neurons** - People, places, things ("Alice", "coffee shop")
- **Time neurons** - Temporal references ("Tuesday 3pm", "last week")
- **Concept neurons** - Ideas, topics ("authentication", "rate limiting")
- **Action neurons** - What happened ("discussed", "decided", "fixed")
- **State neurons** - Conditions ("blocked", "completed", "urgent")

### Synapses

Synapses are typed connections between neurons:

- **Temporal** - `HAPPENED_AT`, `BEFORE`, `AFTER`
- **Causal** - `CAUSED_BY`, `LEADS_TO`, `ENABLES`
- **Associative** - `RELATED_TO`, `CO_OCCURS`
- **Semantic** - `IS_A`, `HAS_PROPERTY`, `INVOLVES`

### Fibers

Fibers are **signal pathways** through the neural graph - ordered sequences of neurons that form a coherent memory. Each fiber has a **conductivity** (0.0-1.0) that determines how well signals travel through it:

```
Fiber: "Meeting with Alice" (conductivity: 0.95)
Pathway: [Tuesday 3pm] → [Alice] → [Meeting] → [API design] → [Rate limiting]
├── [Alice] ←DISCUSSED→ [API design]
├── [Coffee shop] ←AT_LOCATION→ [Meeting]
├── [Tuesday 3pm] ←HAPPENED_AT→ [Meeting]
└── [Rate limiting] ←SUGGESTED_BY→ [Alice]
```

Frequently-accessed fibers develop higher conductivity, making their memories easier to recall - similar to how neural pathways strengthen with use in the biological brain.

## Encoding Process

When you store a memory:

```bash
nmem remember "Met Alice at coffee shop to discuss API design, she suggested rate limiting"
```

Surreal-Memory:

1. **Extracts entities** - Alice, coffee shop, API design, rate limiting
2. **Extracts temporal context** - (uses current time if not specified)
3. **Identifies relationships** - Alice DISCUSSED API design, Alice SUGGESTED rate limiting
4. **Creates neurons** - One document per entity/concept in SurrealDB
5. **Creates synapses** - Graph edges (`RELATE`) between neuron documents
6. **Bundles into fiber** - Groups everything into a coherent memory pathway

## Retrieval Process

When you query:

```bash
nmem recall "What did Alice suggest last Tuesday?"
```

Surreal-Memory (reflex mode):

1. **Parses query** - Identifies "last Tuesday" as time hint, "Alice" as entity, "suggest" as action hint
2. **Finds anchors (time-first)** - Locates time neurons first (weight 1.0), then entities (0.8), then actions (0.6)
3. **Finds fibers** - Gets fiber pathways containing anchor neurons
4. **Trail activation** - Spreads signals along fiber pathways with conductivity and time decay
5. **Co-activation detection** - Neurons reached by multiple anchor sets get binding strength boost
6. **Extracts subgraph** - Gets highest-scoring connected cluster
7. **Reinforces fibers** - Accessed fibers get conductivity boost (+0.02)
8. **Reconstructs answer** - "Alice suggested rate limiting"

## Activation Dynamics

### Reflex Mode (default)

Activation spreads along **fiber pathways** with trail decay:

```
activation(next) = current * (1 - decay) * synapse_weight * conductivity * time_factor
```

Neurons co-activated by multiple anchor sets receive Hebbian binding boost:

```
[Tuesday] ──fiber──► [Meeting] ◄──fiber── [Alice]
                         │
              co-activated (binding=1.0)
                         │
                    [BEST RESULT]
```

### Classic Mode

Distance-based decay through BFS:

```
activation(hop) = initial * decay_factor^hop
```

### Cone Queries (vector-boosted)

The community plugin adds cone queries that combine HNSW vector similarity with graph traversal:

```
Query embedding ──► KNN search ──► top-k candidates
                                            │
                                            ▼
                                   Graph traversal from
                                   each candidate neuron
                                            │
                                            ▼
                                   Merged activation scores
                                   (vector similarity * graph proximity)
```

## Depth Levels

Different queries need different exploration depths:

| Level | Name | Hops | Use Case |
|-------|------|------|----------|
| 0 | Instant | 1 | Who, what, where |
| 1 | Context | 2-3 | Before/after context |
| 2 | Habit | 4+ | Cross-time patterns |
| 3 | Deep | Full | Causal chains, emotions |

## Memory Lifecycle

Memories evolve over time:

### Decay

Unused memories weaken following the Ebbinghaus forgetting curve:

```
activation = initial * e^(-decay_rate * days)
```

### Reinforcement

Frequently accessed memories strengthen (Hebbian learning):

```
When recalled: synapse.weight += reinforcement_delta
When fiber activated: fiber.conductivity += 0.02  (capped at 1.0)
```

### Compression

Old memories can be summarized:

```
Original: [20 detailed neurons about Tuesday meeting]
Compressed: [1 summary neuron: "API design meeting with Alice"]
```

### Smart Merge

The community plugin provides embedding-based neuron consolidation. Near-duplicate neurons (cosine similarity > 0.95) are automatically detected and merged, keeping the more-accessed neuron as the canonical version.

### Directional Compression

Multi-axis semantic compression preserves entity relationships when summarizing content:
- **Summary level** - Keeps top sentences ranked by entity density
- **Essence level** - Keeps only entity-containing sentences

## Comparison with RAG

| Aspect | RAG | Surreal-Memory |
|--------|-----|----------------|
| Data model | Flat chunks | Neural graph in multi-model DB |
| Retrieval | Similarity search only | Spreading activation + vector + graph traversal |
| Storage | Separate vector DB + doc store | Single SurrealDB instance (document + graph + vector) |
| Relationships | Implicit (chunk proximity) | Explicit typed synapses (graph edges) |
| Temporal | Metadata filter | First-class neurons |
| Multi-hop | Multiple queries / LLM calls | Single graph traversal query |
| Memory lifecycle | Static | Dynamic decay/reinforce/compress |
| Semantic search | External embedding index | Native HNSW in SurrealDB |
| Path finding | Not supported | Native BFS via graph edges |
| Deduplication | Manual | Smart merge via embedding similarity |

## Smart Context Optimization

When you request context (`nmem_context`), Surreal-Memory doesn't just return the most recent memories. It uses a **5-factor composite scoring** system to select the most relevant items:

```
Score = 0.30 * activation     # How recently/actively recalled
      + 0.25 * priority       # User-assigned importance (0-10)
      + 0.20 * frequency      # How often accessed
      + 0.15 * conductivity   # Fiber signal quality
      + 0.10 * freshness      # Creation recency
```

After scoring, the pipeline:

1. **Sorts** items by composite score (highest first)
2. **Deduplicates** using SimHash fingerprints (removes near-duplicates)
3. **Allocates token budgets** proportionally to scores (higher-scored items get more tokens)
4. **Truncates** oversized items to fit their budget

This ensures you get the most relevant, diverse context within your token limit.

## Recall Pattern Learning

Surreal-Memory learns from your query patterns. When you repeatedly look up related topics in sequence (e.g., "authentication" followed by "middleware"), the system detects these co-occurrence patterns and materializes them as CONCEPT neurons connected by BEFORE synapses.

```
Session 1: recall "auth"     → recall "middleware"
Session 2: recall "jwt"      → recall "express routing"
Session 3: recall "tokens"   → recall "middleware setup"
                    ↓
           Pattern detected: auth topics → middleware topics
                    ↓
           CONCEPT("auth") ──BEFORE──► CONCEPT("middleware")
```

On subsequent recalls, Surreal-Memory suggests **related queries** by following these learned patterns, helping you discover information you frequently need together.

## Proactive Alerts

Surreal-Memory monitors brain health and creates persistent alerts when issues are detected:

- **High neuron/fiber/synapse count** — Brain needs consolidation
- **Low connectivity** — Neurons are isolated, needs enrichment
- **Expired memories** — Stale content needs cleanup
- **Stale fibers** — Unused pathways degrading

Alerts follow a lifecycle: `active → seen → acknowledged → resolved`. They're surfaced as a `pending_alerts` count in regular tool responses, and can be managed via `nmem_alerts`.

## Community Plugin

The community plugin unlocks advanced features at no cost:

- **Cone queries** - HNSW vector search via SurrealDB's native `vector::distance::knn()`
- **Smart merge** - Embedding-based neuron consolidation (deduplication)
- **Directional compression** - Multi-axis semantic preservation during summarization
- **SurrealDB storage backend** - Multi-model storage with document, graph, and vector in one database
- **Merkle delta sync** - Efficient multi-device synchronization

The plugin is auto-discovered at server startup. No license key or configuration required.

## Training from External Sources

Beyond encoding individual memories, Surreal-Memory can learn domain knowledge from structured sources.

### Database Schema Training

Surreal-Memory can learn to understand database structure by training from schema metadata:

```
SQLite Database
    │
    ▼
┌─────────────────────┐
│ SchemaIntrospector   │  Extract tables, columns, FKs, indexes
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ KnowledgeExtractor   │  Map FKs → synapse types, detect patterns
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ DBTrainer            │  Batch encode into brain neurons + synapses
└─────────────────────┘
```

**What it learns:**

- Table entities as CONCEPT neurons with semantic descriptions
- FK relationships as typed synapses (IS_A, INVOLVES, AT_LOCATION, RELATED_TO)
- Schema patterns: audit trails, soft deletes, tree hierarchies, polymorphic types, enum tables
- Join tables become direct CO_OCCURS synapses (no separate entity node)

**What it does NOT import:** Raw data rows. Only structural knowledge.

This enables queries like:
- "How are orders related to customers?" → Traces FK relationships
- "Which tables have audit trails?" → Recalls detected patterns

### Documentation Training

Surreal-Memory can train from markdown documentation files, parsing them into semantic chunks with heading hierarchy, encoding them through the NLP pipeline, and running consolidation to create domain-specific expert brains.
