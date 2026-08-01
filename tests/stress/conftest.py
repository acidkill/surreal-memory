"""Shared fixtures for stress tests — InMemoryStorage, no mocks.

Named ``sqlite_storage`` for compatibility with the six test files that use it
as a fixture parameter; the fixture itself now runs on InMemoryStorage, which
grew the full NeuralStorage surface. Disk-contention behaviour (WAL, concurrent
file access) was never something this suite exercised — none of the stress
tests assert on it — so nothing is lost by dropping the file backing.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.storage.memory_store import InMemoryStorage


@pytest_asyncio.fixture
async def sqlite_storage() -> InMemoryStorage:
    """Create an initialized InMemoryStorage with a stress-test brain."""
    storage = InMemoryStorage()

    config = BrainConfig(
        decay_rate=0.1,
        reinforcement_delta=0.05,
        activation_threshold=0.15,
        max_spread_hops=4,
        max_context_tokens=1500,
    )
    brain = Brain.create(name="stress-test", config=config)
    await storage.save_brain(brain)
    storage.set_brain(brain.id)

    return storage


@pytest_asyncio.fixture
async def encoder(sqlite_storage: InMemoryStorage) -> MemoryEncoder:
    """Create a MemoryEncoder bound to the stress-test brain."""
    brain = await sqlite_storage.get_brain(sqlite_storage._current_brain_id)  # type: ignore[arg-type]
    assert brain is not None
    return MemoryEncoder(storage=sqlite_storage, config=brain.config)


# Sample content pools for diverse memory encoding
FACT_MEMORIES = [
    "Python 3.11 introduced TaskGroup and ExceptionGroup for structured concurrency",
    "PostgreSQL supports JSONB columns for semi-structured data storage",
    "Redis uses an event-driven single-threaded architecture for I/O operations",
    "Docker containers share the host kernel unlike virtual machines",
    "GraphQL allows clients to request exactly the fields they need",
    "Kubernetes pods are the smallest deployable units in a cluster",
    "WebSocket enables full-duplex communication over a single TCP connection",
    "JWT tokens are self-contained and do not require server-side session storage",
    "CORS headers control which origins can access an API endpoint",
    "SQLite uses WAL mode for concurrent read-write operations",
]

DECISION_MEMORIES = [
    "We decided to use FastAPI instead of Flask for the REST API",
    "The team chose PostgreSQL over MySQL for better JSON support",
    "We will use Redis for caching with a 5-minute TTL by default",
    "Authentication will use JWT with RSA-256 signing",
    "We adopted trunk-based development with short-lived feature branches",
]

ERROR_MEMORIES = [
    "ConnectionRefusedError when connecting to Redis on port 6380",
    "TypeError: expected str but got NoneType in user validation",
    "ImportError: cannot import name 'AsyncEngine' from sqlalchemy",
    "TimeoutError: API gateway returned 504 after 30 seconds",
    "PermissionError: insufficient privileges for /var/log/app.log",
]

TODO_MEMORIES = [
    "Add rate limiting to the public API endpoints",
    "Write integration tests for the payment processing flow",
    "Set up monitoring dashboards for production database metrics",
    "Migrate legacy user passwords from MD5 to bcrypt hashing",
    "Configure automated backup for the PostgreSQL database",
]

ALL_MEMORIES = FACT_MEMORIES + DECISION_MEMORIES + ERROR_MEMORIES + TODO_MEMORIES


@pytest.fixture
def memory_pool() -> list[str]:
    """Return a diverse pool of memory content strings."""
    return list(ALL_MEMORIES)
