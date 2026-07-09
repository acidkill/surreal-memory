"""Unit tests for the dashboard TTL cache (2.7.4).

The overview grade/purity and the graph payload are cached per brain for a short
window so repeat dashboard loads are instant. These pin the cache semantics:
hit within TTL, miss after expiry, disable via ttl=0 / env, and invalidation.
"""

from __future__ import annotations

import importlib

from surreal_memory.server.dashboard_cache import TTLCache


class _Clock:
    """Manual monotonic clock for deterministic expiry tests."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class TestTTLCache:
    def test_hit_within_ttl(self):
        clock = _Clock()
        cache = TTLCache(ttl=60.0, clock=clock)
        cache.set("k", {"grade": "A"})
        clock.advance(59.0)
        assert cache.get("k") == {"grade": "A"}

    def test_miss_after_expiry(self):
        clock = _Clock()
        cache = TTLCache(ttl=60.0, clock=clock)
        cache.set("k", 123)
        clock.advance(61.0)
        assert cache.get("k") is None

    def test_missing_key_returns_none(self):
        assert TTLCache(ttl=60.0).get("absent") is None

    def test_ttl_zero_disables_cache(self):
        cache = TTLCache(ttl=0.0)
        assert cache.enabled is False
        cache.set("k", "v")
        assert cache.get("k") is None

    def test_invalidate_and_clear(self):
        cache = TTLCache(ttl=60.0)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate("a")
        assert cache.get("a") is None
        assert cache.get("b") == 2
        cache.clear()
        assert cache.get("b") is None

    def test_expired_entry_is_evicted_on_read(self):
        clock = _Clock()
        cache = TTLCache(ttl=10.0, clock=clock)
        cache.set("k", "v")
        clock.advance(11.0)
        cache.get("k")  # triggers eviction
        assert "k" not in cache._store


class TestConfiguredTTL:
    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("SURREAL_MEMORY_DASHBOARD_CACHE_TTL", "5")
        mod = importlib.import_module("surreal_memory.server.dashboard_cache")
        assert mod._configured_ttl() == 5.0

    def test_env_zero_disables(self, monkeypatch):
        monkeypatch.setenv("SURREAL_MEMORY_DASHBOARD_CACHE_TTL", "0")
        assert TTLCache().enabled is False

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("SURREAL_MEMORY_DASHBOARD_CACHE_TTL", "not-a-number")
        mod = importlib.import_module("surreal_memory.server.dashboard_cache")
        assert mod._configured_ttl() == mod._DEFAULT_TTL_SECONDS
