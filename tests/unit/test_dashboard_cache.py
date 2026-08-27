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


class TestHealthReportCache:
    """The health page paid the full diagnostics cost on every single load.

    ``/api/dashboard/stats`` cached the expensive part from 2.7.4 onward, but
    ``/api/dashboard/health`` called ``DiagnosticsEngine.analyze`` directly, so
    on a large brain it paid a full multi-second analyze every time, not just when
    cold, while ``/stats`` served the identical computation from cache.
    The two now share one cached report, so they cannot disagree either.
    """

    @staticmethod
    def _reload():
        """Clear the shared cache WITHOUT reloading the module.

        importlib.reload rebinds the module's cache singletons, which other
        tests already hold references to -- it made an unrelated uncertainty
        cache test fail depending on execution order.
        """
        import surreal_memory.server.routes.dashboard_api as api

        api._GRADE_CACHE.clear()
        return api

    async def test_second_call_does_not_recompute(self, monkeypatch) -> None:
        api = self._reload()
        calls = {"n": 0}

        class _Report:
            grade = "B"
            purity_score = 71.5

        async def _fake_analyze(self, brain_id):
            calls["n"] += 1
            return _Report()

        monkeypatch.setattr(
            "surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze",
            _fake_analyze,
        )

        first = await api._cached_health_report(object(), "brain-a")
        second = await api._cached_health_report(object(), "brain-a")

        assert calls["n"] == 1, "the second load recomputed a multi-second analyze"
        assert first is second

    async def test_grade_and_health_share_one_report(self, monkeypatch) -> None:
        """Two endpoints recomputing the same analyze can also disagree."""
        api = self._reload()
        calls = {"n": 0}

        class _Report:
            grade = "C"
            purity_score = 55.0

        async def _fake_analyze(self, brain_id):
            calls["n"] += 1
            return _Report()

        monkeypatch.setattr(
            "surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze",
            _fake_analyze,
        )

        await api._cached_health_report(object(), "brain-b")
        grade, purity = await api._cached_grade_purity(object(), "brain-b")

        assert calls["n"] == 1, "grade/purity must reuse the cached health report"
        assert (grade, purity) == ("C", 55.0)

    async def test_separate_brains_do_not_share_an_entry(self, monkeypatch) -> None:
        api = self._reload()
        seen = []

        class _Report:
            grade = "A"
            purity_score = 90.0

        async def _fake_analyze(self, brain_id):
            seen.append(brain_id)
            return _Report()

        monkeypatch.setattr(
            "surreal_memory.engine.diagnostics.DiagnosticsEngine.analyze",
            _fake_analyze,
        )

        await api._cached_health_report(object(), "brain-a")
        await api._cached_health_report(object(), "brain-b")

        assert seen == ["brain-a", "brain-b"], "one brain's health served another's"


class TestNoEndpointRecomputesDiagnosticsDirectly:
    """One cached report, or four endpoints paying for it separately.

    `/health` was given a cached diagnostics report, but `/brains`,
    `/config-status` and `/storage/status` each kept calling
    `DiagnosticsEngine.analyze` directly — so the expensive analysis ran once
    per endpoint per load instead of once per brain per TTL window. Fixing them
    one at a time is what left three behind the first time, so this is a scan,
    not four separate assertions.
    """

    def test_analyze_is_only_called_through_the_cache(self) -> None:
        import ast
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src/surreal_memory/server/routes/dashboard_api.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            if node.name in {"_cached_health_report", "_cached_evolution"}:
                continue  # the two places allowed to compute an analysis

            # Track engines bound to a local first: the handlers write
            # `engine = EvolutionEngine(storage)` and then `engine.analyze(...)`,
            # so matching only the inline `Engine(...).analyze(...)` form misses
            # them entirely — which is exactly how this scan passed while
            # /evolution still recomputed on every request.
            engines = {"DiagnosticsEngine", "EvolutionEngine"}
            bound: set[str] = set()
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Assign)
                    and isinstance(inner.value, ast.Call)
                    and getattr(inner.value.func, "id", "") in engines
                ):
                    for target in inner.targets:
                        if isinstance(target, ast.Name):
                            bound.add(target.id)

            for inner in ast.walk(node):
                if not (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "analyze"
                ):
                    continue
                receiver = inner.func.value
                inline = isinstance(receiver, ast.Call) and (
                    getattr(receiver.func, "id", "") in engines
                )
                via_local = isinstance(receiver, ast.Name) and receiver.id in bound
                if inline or via_local:
                    offenders.append(f"{node.name}:{inner.lineno}")

        assert not offenders, (
            "these handlers recompute diagnostics instead of using "
            f"_cached_health_report: {offenders}"
        )


class TestHealthReportTTL:
    def test_diagnostics_ttl_is_long_enough_to_ever_hit(self) -> None:
        """A 60 s window cannot amortise a multi-second report.

        With the shared default, a dashboard visited less often than once a
        minute missed every time and paid full price on every load — the cache
        existed but only helped during continuous use.
        """
        from surreal_memory.server.routes.dashboard_api import _HEALTH_TTL_SECONDS

        assert _HEALTH_TTL_SECONDS >= 300
