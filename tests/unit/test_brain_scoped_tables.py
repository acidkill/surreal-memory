"""`clear()` must wipe every brain-scoped table, not a hand-maintained subset.

The list used to be spelled out inline in clear() and drifted as tables were
added, so rows in project, source, entity_refs, co_activations, depth_priors,
tool_events and the trace tables outlived their brain. Nothing else deletes by
brain, so those rows accumulated permanently. This pins the list to the schema.
"""

from __future__ import annotations

import re
from pathlib import Path

from surreal_memory.storage.surrealdb import schema as schema_module
from surreal_memory.storage.surrealdb.store import _BRAIN_SCOPED_TABLES

_BRAIN_ID_FIELD = re.compile(r"DEFINE FIELD\s+brain_id\s+ON\s+(?:TABLE\s+)?(\w+)", re.IGNORECASE)


def _tables_with_brain_id() -> set[str]:
    source = Path(schema_module.__file__).read_text(encoding="utf-8")
    return set(_BRAIN_ID_FIELD.findall(source))


def test_schema_declares_brain_scoped_tables() -> None:
    # Guards the regex itself: a rename in schema.py that breaks parsing would
    # otherwise make the comparison below vacuously pass.
    assert len(_tables_with_brain_id()) > 5


def test_clear_covers_every_brain_scoped_table() -> None:
    missing = _tables_with_brain_id() - set(_BRAIN_SCOPED_TABLES)

    assert not missing, (
        f"tables carrying brain_id but never cleared: {sorted(missing)} — "
        "add them to _BRAIN_SCOPED_TABLES or their rows will outlive the brain"
    )


def test_cleared_tables_all_exist_in_the_schema() -> None:
    source = Path(schema_module.__file__).read_text(encoding="utf-8")
    defined = set(re.findall(r"DEFINE TABLE\s+(\w+)", source))

    unknown = set(_BRAIN_SCOPED_TABLES) - defined

    assert not unknown, f"cleared tables absent from the schema: {sorted(unknown)}"
