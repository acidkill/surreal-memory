"""SQLite -> PostgreSQL migration utility.

Exports all data from a SQLite brain database and imports it into
PostgreSQL storage. Uses the standard export_brain/import_brain
snapshot pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def migrate_sqlite_to_postgres(
    sqlite_db_path: str,
    pg_host: str = "localhost",
    pg_port: int = 5432,
    pg_database: str = "neuralmemory",
    pg_user: str = "postgres",
    pg_password: str = "",
    embedding_dim: int = 384,
    brain_name: str | None = None,
) -> dict[str, Any]:
    """Migrate a SQLite brain database to PostgreSQL.

    Args:
        sqlite_db_path: Path to the SQLite .db file.
        pg_host: PostgreSQL host.
        pg_port: PostgreSQL port.
        pg_database: PostgreSQL database name.
        pg_user: PostgreSQL user.
        pg_password: PostgreSQL password.
        embedding_dim: Embedding dimension for pgvector.
        brain_name: Specific brain to migrate (None = all brains).

    Returns:
        Migration statistics dict.
    """
    from neural_memory.storage.postgres.postgres_store import PostgreSQLStorage
    from neural_memory.storage.sqlite_store import SQLiteStorage

    # Open source SQLite
    sqlite_storage = SQLiteStorage(sqlite_db_path)
    await sqlite_storage.initialize()

    # Open target PostgreSQL
    pg_storage = PostgreSQLStorage(
        host=pg_host,
        port=pg_port,
        database=pg_database,
        user=pg_user,
        password=pg_password,
        embedding_dim=embedding_dim,
    )
    await pg_storage.initialize()

    stats: dict[str, Any] = {"brains": []}

    try:
        # Get list of brains to migrate
        if brain_name:
            brain_names = [brain_name]
        else:
            from pathlib import Path

            db_dir = Path(sqlite_db_path).parent
            brain_names = sorted(p.stem for p in db_dir.glob("*.db"))
            if not brain_names:
                brain_names = ["default"]

        for name in brain_names:
            logger.info("Migrating brain: %s", name)
            sqlite_storage.set_brain(name)

            brain = await sqlite_storage.get_brain(name)
            if brain is None:
                logger.warning("Brain '%s' not found in SQLite, skipping", name)
                continue

            # Export from SQLite
            snapshot = await sqlite_storage.export_brain(name)
            logger.info(
                "Exported brain '%s': %d neurons, %d synapses, %d fibers",
                name,
                len(snapshot.neurons),
                len(snapshot.synapses),
                len(snapshot.fibers),
            )

            # Import to PostgreSQL
            imported_id = await pg_storage.import_brain(snapshot, target_brain_id=name)
            logger.info("Imported brain '%s' to PostgreSQL as '%s'", name, imported_id)

            stats["brains"].append(
                {
                    "name": name,
                    "neurons": len(snapshot.neurons),
                    "synapses": len(snapshot.synapses),
                    "fibers": len(snapshot.fibers),
                }
            )

        stats["success"] = True
        stats["total_brains"] = len(stats["brains"])

    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        stats["success"] = False
        stats["error"] = str(e)

    finally:
        await sqlite_storage.close()
        await pg_storage.close()

    return stats
