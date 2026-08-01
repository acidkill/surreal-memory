"""Content hashing for files fed to the document trainer."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_MAX_HASH_SIZE = 2 * 1024 * 1024 * 1024


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file's content.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest of the SHA-256 hash.

    Raises:
        ValueError: If file is too large.
    """
    file_size = file_path.stat().st_size
    if file_size > _MAX_HASH_SIZE:
        raise ValueError(f"File too large to hash: {file_size} bytes (max {_MAX_HASH_SIZE})")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()
