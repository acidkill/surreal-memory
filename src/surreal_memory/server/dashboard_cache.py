"""Tiny TTL cache for the expensive dashboard reads.

The overview grade/purity (a full ``DiagnosticsEngine.analyze``) and the graph
view are inherently a few seconds each on a large brain — they aggregate over
hundreds of thousands of synapses (``GROUP BY`` scans that no index removes).
But these are *slow-moving quality/structure* views: the health grade barely
shifts between two page loads seconds apart, and neither does a 64k-node graph.

So instead of recomputing on every request, cache the result per brain for a
short window. Repeat loads (the common case — the dashboard auto-refreshes and
users navigate back and forth) are served instantly; the value is recomputed at
most once per TTL. Counts (neuron/synapse/fiber) are cheap and are NOT cached
here — the overview keeps them live.

TTL is read from ``SURREAL_MEMORY_DASHBOARD_CACHE_TTL`` (seconds); set it to 0 to
disable caching entirely (every request recomputes).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

_DEFAULT_TTL_SECONDS = 60.0


def _configured_ttl() -> float:
    raw = os.environ.get("SURREAL_MEMORY_DASHBOARD_CACHE_TTL")
    if raw is None or raw == "":
        return _DEFAULT_TTL_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_TTL_SECONDS


class TTLCache:
    """A minimal, single-process TTL cache.

    Entries expire ``ttl`` seconds after they are written. A ttl of 0 disables
    the cache (``get`` always misses, ``set`` is a no-op) so it can be turned off
    without touching call sites. The clock is injectable for deterministic tests.
    """

    def __init__(
        self,
        ttl: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = _configured_ttl() if ttl is None else max(0.0, ttl)
        self._clock = clock
        self._store: dict[str, tuple[float, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._ttl > 0

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        written_at, value = entry
        if self._clock() - written_at > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._store[key] = (self._clock(), value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
