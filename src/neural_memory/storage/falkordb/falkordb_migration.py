"""SQLite → FalkorDB migration utility.

Exports all data from a SQLite brain database and imports it into
FalkorDB graph storage. Creates proper graph nodes, edges, and indexes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def migrate_sqlite_to_falkordb(
    sqlite_db_path: str,
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_username: str | None = None,
    falkordb_password: str | None = None,
    brain_name: str | None = None,
) -> dict[str, Any]:
    """Migrate a SQLite brain database to FalkorDB.

    Args:
        sqlite_db_path: Path to the SQLite .db file
        falkordb_host: FalkorDB host
        falkordb_port: FalkorDB port
        falkordb_username: Optional FalkorDB username
        falkordb_password: Optional FalkorDB password
        brain_name: Specific brain to migrate (None = all brains)

    Returns:
        Migration statistics dict
    """
    from neural_memory.storage.falkordb.falkordb_store import FalkorDBStorage
    from neural_memory.storage.sqlite_store import SQLiteStorage

    # Open source SQLite
    sqlite_storage = SQLiteStorage(sqlite_db_path)
    await sqlite_storage.initialize()

    # Open target FalkorDB
    fdb_storage = FalkorDBStorage(
        host=falkordb_host,
        port=falkordb_port,
        username=falkordb_username,
        password=falkordb_password,
    )
    await fdb_storage.initialize()

    stats: dict[str, Any] = {"brains": []}

    try:
        # Get list of brains to migrate
        if brain_name:
            brain_names = [brain_name]
        else:
            # Discover all brains from the brains/ directory
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

            # Import to FalkorDB
            imported_id = await fdb_storage.import_brain(snapshot, target_brain_id=name)
            logger.info("Imported brain '%s' to FalkorDB as '%s'", name, imported_id)

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
        await fdb_storage.close()

    return stats
