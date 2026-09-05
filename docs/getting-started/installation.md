# Installation

## Requirements

- Python 3.11 or higher
- Docker + Docker Compose (for SurrealDB)
- pipx (recommended for CLI isolation)

## Quick Install

### From GitHub (recommended)

Install directly from the repository with all SurrealDB and embedding extras:

```bash
pipx install \
  "surreal-memory[surrealdb,embeddings-gemini,server] @ git+https://github.com/acidkill/surreal-memory.git"
```

### From PyPI

```bash
pip install surreal-memory[surrealdb,embeddings-gemini]
```

## SurrealDB (Required for full features)

Surreal-Memory requires a running SurrealDB instance. The included Docker Compose file starts everything in one command:

```bash
git clone https://github.com/acidkill/surreal-memory.git
cd surreal-memory
cp .env.example .env          # edit with your GEMINI_API_KEY
docker compose -f docker-compose.surrealdb.yml up -d
```

This starts:
- `SurrealDB` on port 8001 (storage backend)
- `surreal-memory` REST server + MCP server on port 8000

Dashboard at http://localhost:8000/ui, health at http://localhost:8000/health.

## Optional Extras

| Extra | Installs | When to use |
|-------|----------|-------------|
| `surrealdb` | `surrealdb>=2.0.0,<3.0.0` | **Required** for graph + vector features |
| `embeddings-gemini` | `google-genai>=1.0` | Semantic search via Gemini embeddings |
| `embeddings-openai` | `openai>=1.0` | Semantic search via OpenAI embeddings |
| `server` | FastAPI + uvicorn | REST API + web dashboard |
| `encryption` | `cryptography>=46` | Fernet encryption at rest |
| `nlp-en` | spaCy | Better entity extraction (English) |
| `extract` | pymupdf4llm + docx/pptx/xlsx | Train from documents |
| `all` | Everything above | Full feature set |

Example — install with encryption and document training:

```bash
pipx install "surreal-memory[surrealdb,embeddings-gemini,server,encryption,extract] @ git+https://github.com/acidkill/surreal-memory.git"
```

## Development Installation

For contributing to this repository or running tests:

```bash
git clone https://github.com/acidkill/surreal-memory.git
cd surreal-memory
pip install -e ".[dev,surrealdb,embeddings-gemini,server]"
```

For test infra via pipx (required for the pytest venv to find the package):

```bash
pipx install pytest
pipx inject pytest pytest-asyncio pytest-timeout numpy
pipx runpip pytest install -e .

pipx install mypy
pipx inject mypy pydantic
```

## Configure Claude Code MCP

After installing, register the MCP server with Claude Code:

```bash
claude mcp add --scope user surreal-memory -- smem-mcp
```

## Required Environment Variables

Set these before starting `smem-mcp` (add to `~/.bashrc` or `~/.zshrc`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SURREAL_MEMORY_STORAGE` | `sqlite` | Set to `surrealdb` |
| `SURREALDB_URL` | — | `http://localhost:8001` |
| `SURREALDB_USER` | — | `root` |
| `SURREALDB_PASS` | — | Your SurrealDB password. Default: `surrealmemory` |
| `SURREALDB_NS` | — | `surreal_memory` |
| `SURREALDB_DB` | — | `default` |
| `SURREALDB_AUTH_LEVEL` | `root` | Scope sign-in uses: `root`, `namespace` or `database`. A user defined `ON DATABASE` needs `database`; a root user needs `root` |
| `GEMINI_API_KEY` | — | Your Gemini API key |
| `SURREAL_MEMORY_EMBEDDING_PROVIDER` | — | `gemini` |
| `SURREAL_MEMORY_EMBEDDING_ENABLED` | `false` | Set to `true` |
| `SURREAL_MEMORY_DIR` | `~/.surrealmemory/` | Data directory |
| `SURREAL_MEMORY_BRAIN` | `default` | Active brain name |

## Verify Installation

```bash
# Check CLI
smem --version

# Run full diagnostics (checks SurrealDB, embeddings, MCP, schema)
smem doctor

# Quick smoke test
smem remember "test memory"
smem recall "test"
```

## Automated Setup

For a fully guided setup via Claude Code, see the [installation prompt](https://github.com/acidkill/surreal-memory/blob/main/INSTALL_PROMPT.md):

```
Please read INSTALL_PROMPT.md and follow the instructions to set up Surreal-Memory on this machine.
```

## Troubleshooting

### `smem: command not found`

```bash
# Reload PATH after pipx install
source ~/.bashrc

# Or run directly
python -m surreal_memory.cli --help
```

### `Connection refused` on SurrealDB

```bash
# Check container status
docker compose -f docker-compose.surrealdb.yml ps

# Restart if not running
docker compose -f docker-compose.surrealdb.yml up -d
```

### `ModuleNotFoundError: surreal_memory` in tests

The pytest venv needs a separate editable install:

```bash
pipx runpip pytest install -e .
```

### Build errors after code changes

Docker uses cached layers. Rebuild the image:

```bash
docker compose -f docker-compose.surrealdb.yml up -d --build
```
