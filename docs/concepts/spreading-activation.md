# Spreading Activation

The core retrieval mechanism in Surreal-Memory.

## What is Spreading Activation?

Spreading activation is a cognitive model of how memory retrieval works in the human brain. When you think of a concept, related concepts become more "active" and accessible.

**Example:** Thinking of "apple" activates:
- Fruit (category)
- Red/green (colors)
- Tree (where it grows)
- Pie (food made with it)
- iPhone (brand association)

Surreal-Memory implements this computationally:

```
Query: "apple"
    │
    ▼ activate
[apple] ─────► [fruit] ─────► [banana]
    │              │
    │              ▼
    └─────► [red] ─────► [strawberry]
```

## How It Works

Surreal-Memory supports two activation modes: **Classic** (distance-based decay) and **Reflex** (trail-based decay through fiber pathways). Reflex mode is the default in v0.6.0+.

### 1. Anchor Selection (Time-First)

The query is parsed to identify anchor neurons, with **time neurons as primary anchors**:

```bash
smem recall "What did Alice suggest about auth last Tuesday?"
```

Anchors identified (priority order):

1. `last Tuesday` (time - weight 1.0)
2. `Alice` (entity - weight 0.8)
3. `suggest` (action - weight 0.6)
4. `auth` (concept - weight 0.4)

Time-first anchoring constrains the search to temporally relevant memories before expanding to entities and concepts.

### 2. Initial Activation

Anchor neurons receive weighted initial activation:

```
[Tuesday]  = 1.0  (time anchor)
[Alice]    = 0.8  (entity anchor)
[auth]     = 0.4  (concept anchor)
```

### 3. Activation Spread

#### Classic Mode (distance-based)

Activation spreads through synapses with hop-based decay:

```python
activation(neighbor) = source_activation * synapse_weight * decay_factor
```

Example with decay_factor = 0.8:

```
Hop 0: [Alice] = 1.0
Hop 1: [Meeting with Alice] = 1.0 * 0.9 * 0.8 = 0.72
Hop 2: [JWT suggestion] = 0.72 * 0.8 * 0.8 = 0.46
Hop 3: [Auth module] = 0.46 * 0.7 * 0.8 = 0.26
```

#### Reflex Mode (trail-based) :material-new-box:{ .new }

Activation spreads **along fiber pathways** with trail decay, conductivity, and time factors:

```python
activation(next) = current_level * (1 - decay) * synapse_weight * fiber.conductivity * time_factor
```

Fibers act as signal pathways - ordered sequences of neurons that have been co-activated before. Signals travel along these established trails:

```
Fiber pathway: [Tuesday] → [Alice] → [Meeting] → [JWT] → [Auth]
                  │            │          │          │        │
Activation:     1.0          0.76       0.55       0.38    0.25
                              ↑
                    conductivity=0.95, time_factor=0.98
```

Key differences from classic mode:

- **Conductivity**: Frequently-used fibers conduct signals better (0.0-1.0)
- **Time factor**: Recently-conducted fibers transmit stronger signals
- **Pathway order**: Signals follow established neuron sequences, not arbitrary BFS

### 4. Co-Activation (Hebbian Binding) :material-new-box:{ .new }

When multiple anchor sets activate the same neuron, it receives a **co-activation boost** based on the Hebbian principle: "Neurons that fire together wire together."

```
       Tuesday anchor        Alice anchor        auth anchor
            │                     │                    │
            ▼                     ▼                    ▼
     [Meeting notes] ◄─── [JWT suggestion] ───► [Auth module]
                                │
                    co-activated by 3/3 anchors
                    binding_strength = 1.0
```

Co-activation detection finds neurons activated by multiple independent anchor sets and boosts their scores:

```python
binding_strength = co_fire_count / total_anchor_sets
```

### 5. Subgraph Extraction

The highest-scoring connected region is extracted as the result, with co-activated neurons ranked highest.

### 6. Cross-Encoder Reranking (optional) :material-new-box:{ .new }

When `[reranker].enabled = true` in `config.toml`, the pipeline over-fetches SA
candidates and reranks them with a cross-encoder that scores each `(query, memory)`
pair for relevance:

```python
blended_score = blend_weight * rerank_score + (1 - blend_weight) * activation_level
```

- Runs over HTTP against an OpenAI-compatible `/rerank` endpoint (e.g. llamastash /
  llama.cpp) or, if unset, an in-process `sentence-transformers` `CrossEncoder`.
- The activation level always contributes to the final score (`blend_weight` defaults
  to `0.7`, so activation keeps a `0.3` floor), and any reranker error falls back to
  the unmodified spreading-activation ordering — reranking never breaks recall.
- Off by default; see [Reranking](https://github.com/acidkill/surreal-memory/blob/main/README.md#reranking) for setup.

## Configuration

### Brain Config

```python
@dataclass
class BrainConfig:
    decay_rate: float = 0.1           # How fast activation decays
    reinforcement_delta: float = 0.05  # How much to strengthen on use
    activation_threshold: float = 0.2  # Minimum activation
    max_spread_hops: int = 4          # Maximum traversal depth
    max_context_tokens: int = 1500    # Max tokens in response
```

### Decay Rate

Controls how quickly activation fades:

**Classic mode** (distance-based):
```
activation = initial * decay_factor^hops
decay_factor = 1 - decay_rate
```

**Reflex mode** (trail-based):
```
activation = current * (1 - decay_rate) * synapse_weight * conductivity * time_factor
```

| decay_rate | Effect |
|------------|--------|
| 0.05 | Slow decay, wide spread |
| 0.1 | Default, balanced |
| 0.2 | Fast decay, focused |
| 0.3 | Very focused, nearby only |

### Fiber Conductivity :material-new-box:{ .new }

Fibers have a conductivity value (0.0-1.0) that affects signal transmission:

- **1.0**: Full conductivity (new or frequently-used fiber)
- **0.5**: Moderate conductivity
- **0.0**: No signal passes through

Conductivity increases by +0.02 each time a fiber is activated (capped at 1.0).

### Time Factor :material-new-box:{ .new }

Recently-conducted fibers transmit stronger signals:

```python
time_factor = max(0.1, 1.0 - (age_hours / 168))  # Decay over 7 days
```

| Age | Time Factor |
|-----|-------------|
| Just now | ~1.0 |
| 1 day | ~0.86 |
| 3 days | ~0.57 |
| 7 days | ~0.1 |

### Activation Threshold

Neurons below this level are ignored:

```python
if activation_level < activation_threshold:
    skip_neuron()
```

Lower threshold = more results, potentially noisy
Higher threshold = fewer results, more precise

### Max Spread Hops

Limits how far activation travels:

| Hops | Reach | Use Case |
|------|-------|----------|
| 1 | Direct connections | Simple lookups |
| 2-3 | Local context | Most queries |
| 4+ | Extended graph | Deep analysis |

## Depth Levels

The CLI uses depth levels to control spreading:

```bash
smem recall "query" --depth 0  # Instant: 1 hop
smem recall "query" --depth 1  # Context: 2-3 hops
smem recall "query" --depth 2  # Habit: 4+ hops
smem recall "query" --depth 3  # Deep: full traversal
```

### Auto-Detection

Without `--depth`, the system auto-detects:

| Query Pattern | Detected Depth |
|---------------|----------------|
| "What is X?" | Instant (0) |
| "What happened before X?" | Context (1) |
| "Do I usually X?" | Habit (2) |
| "Why did X happen?" | Deep (3) |

## Synapse Weight Effects

Higher-weight synapses transfer more activation:

```
High weight (0.9):  [A] ──0.9──► [B]  →  B gets 0.72 activation
Low weight (0.3):   [A] ──0.3──► [B]  →  B gets 0.24 activation
```

This naturally prioritizes:
- Strong causal links over weak associations
- Recent/reinforced paths over old/unused ones

## Multi-Anchor Convergence

The power of spreading activation shows with multiple query terms:

```
Query: "Alice auth Tuesday"

[Tuesday] ────┐  (time anchor, weight 1.0)
              │
              ▼
[Alice] ──────┤
              ▼
         [JWT meeting] ← HIGH SCORE (all 3 converge)
              ▲
              │
[auth] ───────┘
```

In reflex mode, co-activated neurons (activated by multiple anchor sets) receive a binding strength score proportional to how many anchors they were reached from. This replaces simple additive scoring with Hebbian-style binding.

## Performance Characteristics

### Hybrid Mode (default)

| Graph Size | Classic (ms) | Hybrid (ms) | Speedup | Recall |
|------------|-------------|-------------|---------|--------|
| 100 neurons | 1.6 | 0.3 | 4.6x | 77% |
| 1K neurons | 2.2 | 0.4 | 5.9x | 55% |
| 3K neurons | 3.2 | 0.4 | 7.3x | 50% |
| 5K neurons | 2.0 | 0.4 | 4.6x | 51% |

See [Benchmarks](../benchmarks.md) for full results. Regenerate with `python benchmarks/run_benchmarks.py`.

### How Hybrid Works

1. **Reflex pass** (fast): Conduct signals along fiber pathways
2. **Discovery pass** (limited BFS): Classic activation with `max_hops // 2` to find neurons outside fibers
3. **Merge**: Reflex results are primary; discovery results are dampened (0.6x) to keep fiber signals ranked higher

This gives the speed of reflex activation with the coverage of classic BFS.

## Compared to Other Approaches

### vs. Vector Similarity

| Aspect | Vector Search | Spreading Activation |
|--------|--------------|---------------------|
| Model | Geometric distance | Graph traversal |
| Relationships | Implicit | Explicit typed edges |
| Multi-hop | Multiple queries | Single traversal |
| Explanation | "Similar embedding" | "Connected via X" |

### vs. Keyword Search

| Aspect | Keyword Search | Spreading Activation |
|--------|---------------|---------------------|
| Model | Text matching | Semantic graph |
| Synonyms | Manual expansion | Automatic via links |
| Context | None | Full graph context |
| Ranking | TF-IDF / BM25 | Activation convergence |

## Reflex vs Classic Mode

You can choose which activation mode to use:

```python
# Hybrid reflex mode (default in v0.6.0+)
# Reflex trail activation + limited classic BFS for discovery
pipeline = ReflexPipeline(storage, config, use_reflex=True)

# Classic mode (distance-based, pre-v0.6.0 behavior)
pipeline = ReflexPipeline(storage, config, use_reflex=False)
```

| Aspect | Classic | Hybrid (default) |
|--------|---------|-------------------|
| Decay model | Distance-based (`decay^hops`) | Trail + limited BFS |
| Time handling | Equal weight | Time-first primary anchors |
| Fiber role | Static memory cluster | Signal pathway with conductivity |
| Multi-anchor | Additive intersection | Co-activation (Hebbian binding) |
| Recency | Not considered | Time factor boosts recent fibers |
| Speed | Baseline | 4-7x faster at scale |
| Recall | 100% (by definition) | ~50-77% of classic's neurons |

## Debugging Activation

Use `--show-routing` to see activation paths:

```bash
smem recall "auth decision" --show-routing
```

Output:
```
Query parsed: entities=[auth], intents=[decision]
Anchors: [neuron-auth-123 (w=0.8), neuron-decision-456 (w=0.4)]
Mode: reflex
Fibers: 3 containing anchor neurons
Trail activation:
  fiber-001 (conductivity=0.95, time_factor=0.92):
    neuron-auth-123 → neuron-jwt-789 (activation: 0.70)
    neuron-jwt-789 → neuron-meeting-012 (activation: 0.48)
Co-activations: neuron-jwt-789 (binding_strength=1.0, sources=2)
Result: "We decided to use JWT for authentication"
```
