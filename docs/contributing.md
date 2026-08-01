# Contributing

Thank you for your interest in contributing to **surreal-memory**!

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/acidkill/surreal-memory-surrealdb-version.git
cd surreal-memory-surrealdb-version

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev + server dependencies and set up pre-commit hooks
make install-dev
# equivalent to:
#   pip install -e ".[dev,server]"
#   pre-commit install
```

### Available Commands

<!-- AUTO-GENERATED from Makefile — do not edit this table manually -->
| Command | Description |
|---------|-------------|
| `make install` | Install package (production) |
| `make install-dev` | Install with dev deps + pre-commit hooks |
| `make lint` | Run ruff linter |
| `make format` | Auto-format with ruff |
| `make format-check` | Verify formatting matches CI (no changes) |
| `make typecheck` | Run mypy type checker |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage report (fails below 67%) |
| `make security` | Run security-focused ruff rules |
| `make audit` | Preview extended rules (non-blocking) |
| `make verify` | **Full CI gate** — lint + format-check + typecheck + test-cov + security |
| `make serve` | Start development REST server (port 8000, hot-reload) |
| `make docs` | Build MkDocs site |
| `make docs-serve` | Serve docs locally |
| `make build` | Build distributable package |
| `make clean` | Remove all build/cache artifacts |
<!-- END AUTO-GENERATED -->

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
make test-cov
# equivalent to: pytest tests/ -v --cov=surreal_memory --cov-report=term-missing --cov-report=html --cov-fail-under=67

# Run specific test file
pytest tests/unit/test_neuron.py -v

# Run a single test by name
pytest tests/unit/test_encoder.py -v -k "test_encode_returns_fiber"
```

### Code Quality

```bash
# Type checking
mypy src/ --ignore-missing-imports

# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Run everything (matches CI exactly)
make verify
```

## Environment Variables

<!-- Maintained by hand. No script generates this table; when you add an env var to
     the code, add it here and to .env.example in the same change. -->
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SURREAL_MEMORY_STORAGE` | No | `sqlite` | Storage backend: `sqlite`/`memory` (test fixtures) or `surrealdb` (production backend) |
| `SURREAL_MEMORY_DIR` | No | `~/.surrealmemory/` | Data directory override |
| `SURREAL_MEMORY_BRAIN` | No | `default` | Active brain name |
| `SURREAL_MEMORY_HOST` | No | `127.0.0.1` | REST server bind host |
| `SURREAL_MEMORY_PORT` | No | `8000` | REST server port |
| `SURREAL_MEMORY_TRUSTED_NETWORKS` | No | — | Trusted CIDR ranges (comma-separated) |
| `SURREALDB_URL` | SurrealDB only | `http://localhost:8001` | SurrealDB endpoint |
| `SURREALDB_USER` | SurrealDB only | `root` | SurrealDB username |
| `SURREALDB_PASS` | SurrealDB only | `surrealmemory` | SurrealDB password |
| `SURREALDB_NS` | No | `surreal_memory` | SurrealDB namespace |
| `SURREALDB_DB` | No | `default` | SurrealDB database name |
| `SURREAL_MEMORY_HUB_URL` | Sync only | — | Cloudflare sync hub URL |
| `SURREAL_MEMORY_API_KEY` | Sync only | — | Sync hub API key |
| `SURREAL_MEMORY_SYNC_ENABLED` | No | `false` | Enable sync |
| `SURREAL_MEMORY_SYNC_AUTO` | No | `false` | Auto-sync on changes |
| `SURREAL_MEMORY_EMBEDDING_ENABLED` | No | `false` | Enable vector embeddings |
| `SURREAL_MEMORY_EMBEDDING_PROVIDER` | Embeddings only | `sentence_transformer` | `sentence_transformer`, `openai`, `openrouter`, `gemini`, `ollama`, `bge_m3`, `auto` |
| `SURREAL_MEMORY_EMBEDDING_ENDPOINT` | No | — | Base URL of a local OpenAI-compatible embedding server (e.g. llamastash bge-m3 at `http://127.0.0.1:11435/v1`). Config key `[embedding] endpoint` wins over this. |
| `SURREAL_MEMORY_EMBEDDING_MODEL` | No | — | Model name for embeddings |
| `GEMINI_API_KEY` | Gemini only | — | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI only | — | OpenAI API key |

| `CLAUDE_SESSION_ID` | No | auto | Session ID (set by Claude Code) |
<!-- END AUTO-GENERATED -->

## Code Style

We use:

- **ruff** for linting and formatting
- **mypy** for type checking
- **PEP 8** naming conventions
- **Google-style** docstrings

### Type Hints

All public functions must have type hints:

```python
# Good
def encode_memory(
    content: str,
    memory_type: MemoryType | None = None
) -> EncodingResult:
    ...

# Bad
def encode_memory(content, memory_type=None):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def query(
    self,
    query: str,
    depth: DepthLevel | None = None,
    max_tokens: int = 500
) -> RetrievalResult:
    """Query memories using spreading activation.

    Args:
        query: The query string to search for.
        depth: Search depth level (auto-detected if None).
        max_tokens: Maximum tokens in response.

    Returns:
        RetrievalResult containing context and metadata.

    Raises:
        ValueError: If query is empty.
    """
    ...
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Keep commits focused and atomic
- Write tests for new functionality
- Update documentation as needed

### 3. Run Checks

```bash
# Run the full CI gate — all checks must pass
make verify
```

### 4. Submit PR

- Write a clear description
- Reference any related issues
- Ensure CI passes

### Commit Messages

Use conventional commits:

```
feat: add decay manager for memory lifecycle
fix: handle null values in query parser
docs: update API reference
test: add tests for spreading activation
refactor: simplify neuron state management
chore: update dependencies
```

## Project Structure

```
src/surreal_memory/
├── core/          # Data structures (Neuron, Synapse, Fiber, Brain)
├── engine/        # Processing (Encoder, Pipeline, Retrieval, Compression)
├── extraction/    # NLP utilities (Parser, Temporal, etc.)
├── storage/       # Storage backends
│   ├── surrealdb/ # SurrealDB backend (primary — 163 methods across 10 mixins)
│   ├── sqlite_*.py# SQLite/InMemory backends (test fixtures only — not the production backend)
│   └── surrealdb/ # SurrealDB backend (recommended)
├── mcp/           # MCP server for Claude (58 tools, ~30 handler files)
├── server/        # FastAPI REST server + dashboard static files
├── cli/           # Command-line interface (smem / surreal-memory)
├── plugins/       # Community plugin (bypasses Pro feature gates)
├── sync/          # Cloudflare Merkle delta sync
├── hooks/         # Claude Code pre-compact / stop / post-tool-use hooks
└── utils/         # Config, simhash, safety utilities
```

## Testing Guidelines

### Unit Tests

Test individual components in isolation:

```python
# tests/unit/test_neuron.py
def test_neuron_creation():
    neuron = Neuron(
        id="test-1",
        type=NeuronType.ENTITY,
        content="Alice"
    )
    assert neuron.id == "test-1"
    assert neuron.type == NeuronType.ENTITY
```

### Integration Tests

Test component interactions:

```python
# tests/integration/test_encoding_flow.py
async def test_encode_and_retrieve():
    storage = InMemoryStorage()
    brain = Brain.create("test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    encoder = MemoryEncoder(storage, brain.config)
    await encoder.encode("Test memory")

    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query("Test")
    assert result.confidence > 0
```

### Test Fixtures

Use pytest fixtures for common setup:

```python
# tests/conftest.py
@pytest.fixture
async def storage():
    storage = InMemoryStorage()
    yield storage

@pytest.fixture
async def brain(storage):
    brain = Brain.create("test-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return brain
```

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Test coverage increases
- Bug fixes with clear reproduction steps

### Intermediate

- New CLI commands
- Storage backend optimizations
- NLP improvements

### Advanced

- Neo4j storage implementation
- Rust extensions for performance
- New retrieval algorithms

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
