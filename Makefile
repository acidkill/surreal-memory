.PHONY: install install-dev lint format typecheck test test-cov security audit audit-dead check clean build docs gen-docs serve

# Install package
install:
	pip install -e .

# Install with dev dependencies
# The surrealdb extra is included because the integration suite runs against a
# real SurrealDB whenever SURREALDB_URL is set -- which it is on any machine
# that actually uses smem. Without the SDK those tests fail instead of skipping.
install-dev:
	pip install -e ".[dev,server,surrealdb]"
	pre-commit install

# Run linter
lint:
	ruff check src/ tests/

# Format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run type checker
typecheck:
	mypy src/ --ignore-missing-imports

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=surreal_memory --cov-report=term-missing --cov-report=html --cov-fail-under=67

# Run security checks (S rules already in select, filtered by ignore + per-file-ignores)
security:
	ruff check src/ --select S --ignore S101,S110,S112,S311,S324
	@echo "Security scan passed."

# Preview extended rules (non-blocking audit)
audit:
	ruff check src/ tests/ --select S,A,DTZ,T20,PT,PERF,PIE,ERA --statistics || true

# Modules nothing outside tests/ can reach (ruff's F401 cannot see these)
audit-dead:
	python scripts/check_dead_modules.py

# Format check (no changes, just verify — matches CI)
format-check:
	ruff format --check src/ tests/

# Run all checks matching CI exactly (full quality gate)
verify: lint format-check typecheck test-cov security

# Legacy alias
check: verify

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Start development server
serve:
	uvicorn surreal_memory.server.app:create_app --factory --reload --port 8000

# Generate docs
# Regenerate every generated reference page (CI checks these are current)
gen-docs:
	python scripts/gen_mcp_docs.py
	python scripts/gen_cli_docs.py
	python scripts/gen_config_docs.py
	python scripts/gen_api_docs.py

docs:
	mkdocs build --strict

# Serve docs locally
docs-serve:
	mkdocs serve
