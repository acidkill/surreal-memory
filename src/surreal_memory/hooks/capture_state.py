"""Per-session idempotency state for auto-capture hooks.

The Stop hook fires after *every* assistant turn and PreCompact fires on
each compaction; both re-read an overlapping transcript tail. When the
effective (filtered/summarized) text hasn't changed since the last
invocation, they still re-encode the same fragment or session summary —
the dominant source of memory poisoning (duplicate "Session activity"
entries accumulating turn over turn).

This module provides a deterministic, dependency-free idempotency key:
a per-session set of normalized-content hashes persisted to a small JSON
file. A capture hook skips any fragment or summary whose ``content_key``
was already saved in the same session.

Design notes:
- Fail-open: any error -> treat content as not-seen. Losing real content is
  worse than tolerating a duplicate, so errors never block a capture.
- Bounded: at most ``_MAX_HASHES_PER_SESSION`` hashes per session and
  ``_MAX_SESSIONS`` sessions are retained (FIFO trim) so the file cannot
  grow without bound.
- Atomic-ish write: write to a temp file then ``replace`` to avoid a torn
  state file if two hooks race.
- Shared key: keyed on ``CLAUDE_SESSION_ID`` so Stop and PreCompact share
  one seen-set and do not re-capture each other's writes.
- Backend-agnostic: the state file lives beside the brain config, not in
  SurrealDB/SQLite, so behavior is identical on both storage backends.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bounded retention so the state file stays small.
_MAX_HASHES_PER_SESSION = 2000
_MAX_SESSIONS = 50

_WS = re.compile(r"\s+")


def _state_path() -> Path:
    """Location of the JSON idempotency state file.

    ``Path("")`` normalizes to ``Path(".")``, which is truthy -- so the
    fallback must check the raw env string, not an already-constructed Path,
    or an unset ``SURREAL_MEMORY_DIR`` silently resolves to the process's
    current working directory instead of the user's home.
    """
    env_dir = os.environ.get("SURREAL_MEMORY_DIR", "").strip()
    data_dir = Path(env_dir) if env_dir else (Path.home() / ".surrealmemory")
    return data_dir / "capture_state.json"


def session_key(transcript_path: str | None = None) -> str:
    """Stable per-session key.

    Prefers ``CLAUDE_SESSION_ID`` (set by Claude Code); falls back to a hash
    of the transcript path, then a constant. The fallback degrades safely:
    worst case the seen-set is shared a bit too broadly within one machine.
    """
    sid = os.environ.get("CLAUDE_SESSION_ID", "").strip()
    if sid:
        return sid
    if transcript_path:
        return "tp_" + hashlib.md5(transcript_path.encode("utf-8")).hexdigest()[:16]
    return "default"


def content_key(content: str) -> str:
    """Deterministic key for a fragment: whitespace-normalized, lowercased md5.

    Re-captured fragments are byte-identical, so even a raw hash would match;
    normalization additionally collapses trivial whitespace/case variants.
    """
    norm = _WS.sub(" ", content.strip().lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def rejected_key(ck: str, min_score: int) -> str:
    """Key marking a candidate the write gate REJECTED at ``min_score``.

    Rejected candidates were never marked as seen, so every Stop/PreCompact
    re-submitted them to the gate for as long as the transcript tail carried
    them. Measured 2026-08-08: 499 auto decisions in 24h held only 134 distinct
    contents (one fragment judged 36 times), inflating the gate's denominator
    ~3.7x and making the observed accept rate look several times worse than it
    was (0.39% on rows vs ~1.4% on distinct content).

    Namespaced by the threshold **on purpose**: marking a rejection as a plain
    seen-key would silence that content for the rest of the session even if the
    threshold were lowered afterwards -- trading duplicate noise for silent
    loss, which is the worse failure. Because the key embeds the score it was
    judged against, changing ``auto_capture_min_score`` stops matching these
    keys and every previously-rejected candidate gets a fresh hearing.
    """
    return f"rej{min_score}:{ck}"


def _load_all() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.debug("capture_state: load failed (fail-open, treat as empty)", exc_info=True)
        return {}


def load_seen(skey: str) -> set[str]:
    """Return the set of content keys already captured in this session."""
    entry = _load_all().get(skey, {})
    hashes = entry.get("hashes", []) if isinstance(entry, dict) else []
    return set(hashes) if isinstance(hashes, list) else set()


def mark_seen(skey: str, new_keys: list[str]) -> None:
    """Persist newly captured content keys for this session (bounded, atomic-ish).

    Never raises -- idempotency bookkeeping must not break a capture path.
    """
    if not new_keys:
        return
    try:
        from surreal_memory.utils.timeutils import utcnow

        ts = utcnow().isoformat()

        data = _load_all()
        entry = data.get(skey, {})
        existing = entry.get("hashes", []) if isinstance(entry, dict) else []
        if not isinstance(existing, list):
            existing = []

        seen_existing = set(existing)
        combined = existing + [k for k in new_keys if k not in seen_existing]
        if len(combined) > _MAX_HASHES_PER_SESSION:
            combined = combined[-_MAX_HASHES_PER_SESSION:]
        data[skey] = {"hashes": combined, "ts": ts}

        # Bound the number of retained sessions (drop oldest by timestamp).
        if len(data) > _MAX_SESSIONS:
            ordered = sorted(
                data.items(),
                key=lambda kv: kv[1].get("ts", "") if isinstance(kv[1], dict) else "",
            )
            data = dict(ordered[-_MAX_SESSIONS:])

        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except (OSError, TypeError, ValueError):
        logger.debug("capture_state: mark_seen failed (non-fatal)", exc_info=True)
