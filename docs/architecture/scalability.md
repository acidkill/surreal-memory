# Scalability

Current performance characteristics and future scaling paths.

## Current Performance (v0.4.x)

### Storage Backends

| Backend | Status | Use Case |
|---------|--------|----------|
| `InMemoryStorage` | Ready | Testing, small datasets |
| `SQLiteStorage` | Ready | **Default** - Personal use |
| `SharedStorage` | Ready | Remote server connection |
| `Neo4jStorage` | Interface only | Production, large scale |

### Performance Characteristics

| Metric | SQLite Backend | Notes |
|--------|----------------|-------|
| Neurons | Up to ~100,000 | Comfortable limit |
| Query latency | 10-50ms | Typical queries |
| Memory usage | ~100MB per 10k neurons | Estimate |
| Concurrent users | 1 | SQLite limitation |

### Operation Latencies

| Operation | Typical Latency |
|-----------|-----------------|
| Encode (simple) | 10-30ms |
| Encode (complex) | 30-50ms |
| Query (depth 0) | 5-20ms |
| Query (depth 1) | 20-50ms |
| Query (depth 2) | 50-100ms |
| Query (depth 3) | 100-150ms |

## Scaling Paths

### Path 1: Neo4j Backend

For scaling beyond 100k neurons or concurrent access.

**When to use:**

- Need >100k neurons
- Need concurrent multi-user access
- Need complex graph queries (Cypher)

**Not implemented.** A Neo4j backend would subclass the `NeuralStorage` ABC in
`src/surreal_memory/storage/base.py`, the same contract the SurrealDB backend
implements:

```python
class Neo4jStorage(NeuralStorage):
    async def add_neuron(self, neuron: Neuron) -> str: ...
    async def get_neighbors(self, neuron_id: str, ...) -> list: ...
    # Full NeuralStorage interface
```

**Expected benefits:**

- Scale to millions of neurons
- Native graph traversal
- Concurrent read/write
- Cypher query language

### Path 2: Rust Extensions

For CPU-intensive operations at massive scale.

**Candidates for optimization:**

1. Spreading activation algorithm
2. Graph traversal
3. Similarity computation

**When to consider:**

- Need >1M neurons
- Need <5ms query latency
- Batch processing without LLM calls

**Approach:** PyO3 for Python bindings

```rust
// Future: src/surreal_memory_core/src/activation.rs
#[pyfunction]
fn spread_activation(
    graph: &PyGraph,
    anchors: Vec<String>,
    max_hops: usize,
) -> PyResult<HashMap<String, f64>> {
    // Rust implementation
}
```

### Path 3: Distributed Architecture

For multi-region, high-availability deployments.

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                           │
├──────────────────┬──────────────────┬───────────────────────┤
│   API Server 1   │   API Server 2   │    API Server N       │
├──────────────────┴──────────────────┴───────────────────────┤
│                    Message Queue (Redis)                     │
├──────────────────┬──────────────────┬───────────────────────┤
│  Neo4j Primary   │  Neo4j Replica   │   Neo4j Replica       │
└──────────────────┴──────────────────┴───────────────────────┘
```

**When to consider:**

- SaaS offering
- Enterprise deployment
- 99.9% uptime requirement

## Optimization Strategies

### Current Optimizations

1. **Early termination** - Stop spreading when below threshold
2. **Priority queue** - Process highest-activation neurons first
3. **Lazy loading** - Don't load full graph into memory
4. **Connection pooling** - Reuse database connections

### Future Optimizations

1. **Caching frequent paths** - Memoize common queries
2. **Batch operations** - Group database writes
3. **Async processing** - Non-blocking IO throughout
4. **Sharding** - Split brains across databases

## Memory Management

### Current Approach

- SQLite handles memory for persistent storage
- NetworkX graph for in-memory operations
- Lazy loading of neuron content

### Large Brain Handling

For brains approaching limits:

```python
# Use streaming for export
async for chunk in exporter.stream_export(brain_id):
    await write_chunk(chunk)

# Apply decay to reduce size
manager = DecayManager(prune_threshold=0.1)
await manager.apply_decay(storage)

# Archive old fibers
await archiver.compress_old_fibers(
    storage,
    older_than_days=90
)
```

## Benchmarking

### Running Benchmarks

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run benchmark suite
python scripts/benchmark.py
```

### Benchmark Areas

1. **Encoding throughput** - Memories per second
2. **Query latency** - p50, p95, p99
3. **Memory usage** - Per neuron overhead
4. **Graph traversal** - Hops per millisecond

## Capacity Planning

### Small (Personal Use)

- 1 user
- <10k neurons
- SQLite sufficient
- ~50MB disk, ~100MB RAM

### Medium (Team)

- 5-10 users
- 10k-100k neurons
- SQLite with shared server
- ~500MB disk, ~500MB RAM

### Large (Enterprise)

- 100+ users
- >100k neurons
- Neo4j recommended
- Dedicated infrastructure

## Contributing to Scalability

### Neo4j Backend

Good first contribution:

1. Implement `Neo4jStorage` class
2. Follow existing `SQLiteStorage` patterns
3. Add integration tests
4. Document deployment

### Benchmarking

Help needed:

1. Create benchmark suite
2. Test with various dataset sizes
3. Identify actual bottlenecks
4. Document findings

### Rust Extensions

Advanced:

1. Requires PyO3 experience
2. Start with spreading activation
3. Maintain Python API compatibility
4. Extensive testing required

See [CONTRIBUTING.md](../contributing.md) for guidelines.
