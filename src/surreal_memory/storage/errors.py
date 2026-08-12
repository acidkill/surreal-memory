"""Backend-agnostic classification of storage errors.

Callers across the engine need to tell "the row is already there" apart from
"the write really failed" without importing either backend driver. That
question belongs to the storage layer, not to whichever engine pass happens to
ask it, so the predicate lives here and stays stdlib-only.
"""

from __future__ import annotations

__all__ = ["is_duplicate_key_error"]


def is_duplicate_key_error(exc: BaseException) -> bool:
    """True when ``exc`` means "a row with this primary key already exists".

    Deterministic edge ids turn a concurrent double-write into a primary-key
    collision, which is a benign outcome: the edge exists either way. Every
    other failure is real and must not be reported as a skipped duplicate.

    Matched by class name and message rather than by importing the backends,
    so this stays true for SurrealDB (``AlreadyExistsError``: "Database record
    `x` already exists") and SQLite (``IntegrityError``: "UNIQUE constraint
    failed") without either driver becoming a hard dependency here.
    """
    name = type(exc).__name__
    if name in {"AlreadyExistsError", "IntegrityError"}:
        return True
    text = str(exc).lower()
    return "already exists" in text or "unique constraint" in text or "duplicate key" in text
