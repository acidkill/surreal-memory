"""SurrealDB storage backend for Neural Memory.

Provides a multi-model storage backend using SurrealDB's graph, document,
and vector search capabilities in a single database.

Usage:
    from neural_memory.storage.surrealdb import SurrealDBStorage

    storage = SurrealDBStorage(
        url="http://localhost:8001",
        namespace="neural_memory",
        database="default",
        user="root",
        password="root",
    )
    await storage.initialize()
    storage.set_brain("my-brain")
"""

from neural_memory.storage.surrealdb.store import SurrealDBStorage

__all__ = ["SurrealDBStorage"]
