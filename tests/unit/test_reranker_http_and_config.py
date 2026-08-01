"""Tests for the 2.7.0 reranker integration: HTTP (llamastash) path, the
config→BrainConfig bridge, and per-brain config persistence.

These cover the wiring that turns the reranker from dead code into a working,
config-driven feature:
  * HttpReranker scores over an OpenAI-compatible ``/rerank`` endpoint.
  * ``rerank_activations(endpoint=...)`` selects the HTTP path and blends.
  * ``RerankerConfig.endpoint`` survives dict + TOML round-trips (injection-safe).
  * ``reranker_brain_config_overrides`` / ``to_brain_config_kwargs`` map app
    config onto ``BrainConfig.reranker_*``.
  * ``store`` serialises/deserialises the full ``BrainConfig`` (so the reranker
    knobs survive a save/load round-trip instead of resetting to defaults).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.engine.reranker import (
    HttpReranker,
    _minmax,
    rerank_activations,
    reranker_available,
)
from surreal_memory.storage.sqlite_store import SQLiteStorage
from surreal_memory.storage.surrealdb.store import (
    _deserialize_brain_config,
    _serialize_brain_config,
)
from surreal_memory.unified_config import (
    BrainSettings,
    RerankerConfig,
    UnifiedConfig,
    _migrate_brain_runtime_config,
    _sanitize_toml_url,
    reranker_brain_config_overrides,
)


@dataclass
class FakeActivationResult:
    neuron_id: str
    activation_level: float
    hop_distance: int = 1
    path: list[str] | None = None
    source_anchor: str = ""

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = []


class _FakeResp:
    """Context-manager stand-in for urllib.request.urlopen()'s return."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _rerank_payload(scores_by_index: dict[int, float]) -> dict:
    return {"results": [{"index": i, "relevance_score": s} for i, s in scores_by_index.items()]}


# ---------------------------------------------------------------------------
# HttpReranker
# ---------------------------------------------------------------------------


class TestHttpReranker:
    def test_rerank_empty(self) -> None:
        r = HttpReranker(endpoint="http://x/v1", model_name="m")
        assert r.rerank("q", [], limit=5) == []

    def test_rerank_blends_and_orders(self) -> None:
        """High reranker score promotes a lower-activation candidate."""
        candidates = [
            ("n_high", "unrelated but strongly activated", 0.9),
            ("n_rel", "the actually relevant document", 0.4),
        ]
        payload = _rerank_payload({0: -5.0, 1: 8.0})  # n_high low, n_rel high

        r = HttpReranker(endpoint="http://x/v1", model_name="m", blend_weight=0.7)
        with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            results = r.rerank("relevant?", candidates, limit=2)

        assert [x.neuron_id for x in results] == ["n_rel", "n_high"]
        # n_rel blended = 0.7*1 + 0.3*0.4 = 0.82 ; n_high = 0.7*0 + 0.3*0.9 = 0.27
        assert results[0].blended_score == pytest.approx(0.82)
        assert results[1].blended_score == pytest.approx(0.27)

    def test_endpoint_trailing_slash_normalised(self) -> None:
        r = HttpReranker(endpoint="http://x/v1/", model_name="m")
        assert r._endpoint == "http://x/v1"

    def test_missing_index_maps_to_zero(self) -> None:
        candidates = [("a", "d0", 0.5), ("b", "d1", 0.5)]
        payload = _rerank_payload({0: 3.0})  # index 1 absent → -inf → 0
        r = HttpReranker(endpoint="http://x/v1", model_name="m", blend_weight=1.0)
        with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            results = r.rerank("q", candidates, limit=2)
        # blend_weight=1.0 → pure normalised score; present index wins
        assert results[0].neuron_id == "a"


class TestMinMax:
    def test_all_equal(self) -> None:
        assert _minmax([2.0, 2.0]) == [1.0, 1.0]

    def test_normalises_to_unit_range(self) -> None:
        assert _minmax([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]

    def test_missing_scores_map_to_zero(self) -> None:
        out = _minmax([float("-inf"), 1.0, 3.0])
        assert out[0] == 0.0


# ---------------------------------------------------------------------------
# rerank_activations endpoint path
# ---------------------------------------------------------------------------


class TestRerankActivationsEndpoint:
    def test_endpoint_selects_http_and_promotes_relevant(self) -> None:
        activations = {
            "n_high": FakeActivationResult("n_high", 0.9),
            "n_rel": FakeActivationResult("n_rel", 0.4),
        }
        contents = {"n_high": "unrelated", "n_rel": "relevant"}
        # candidates sorted by activation desc → n_high idx0, n_rel idx1
        payload = _rerank_payload({0: -5.0, 1: 8.0})
        with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
            result = rerank_activations(
                "relevant?",
                activations,
                contents,
                endpoint="http://127.0.0.1:11435/v1",
            )
        # n_rel now outranks n_high on blended activation level
        assert result["n_rel"].activation_level > result["n_high"].activation_level

    def test_endpoint_failure_falls_back_unchanged(self) -> None:
        activations = {"n1": FakeActivationResult("n1", 0.8)}
        contents = {"n1": "c1"}
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            result = rerank_activations(
                "q", activations, contents, endpoint="http://127.0.0.1:11435/v1"
            )
        # Reranking must never break recall
        assert result is activations

    def test_endpoint_makes_reranker_available(self) -> None:
        with patch.dict(
            "os.environ",
            {"SURREAL_MEMORY_RERANKER_ENDPOINT": "http://127.0.0.1:11435/v1"},
        ):
            assert reranker_available() is True


# ---------------------------------------------------------------------------
# RerankerConfig.endpoint — dict + TOML round-trips, injection safety
# ---------------------------------------------------------------------------


class TestRerankerConfigEndpoint:
    def test_dict_roundtrip_preserves_endpoint(self) -> None:
        cfg = RerankerConfig(enabled=True, endpoint="http://127.0.0.1:11435/v1")
        restored = RerankerConfig.from_dict(cfg.to_dict())
        assert restored.endpoint == "http://127.0.0.1:11435/v1"

    def test_default_endpoint_empty(self) -> None:
        assert RerankerConfig().endpoint == ""

    def test_toml_roundtrip_preserves_endpoint(self) -> None:
        import dataclasses

        with tempfile.TemporaryDirectory() as d:
            cfg = UnifiedConfig(data_dir=Path(d))
            cfg = dataclasses.replace(
                cfg,
                reranker=RerankerConfig(enabled=True, endpoint="http://127.0.0.1:11435/v1"),
            )
            cfg.save()
            reloaded = UnifiedConfig.load(Path(d) / "config.toml")
        assert reloaded.reranker.endpoint == "http://127.0.0.1:11435/v1"
        assert reloaded.reranker.enabled is True

    @pytest.mark.parametrize(
        "good",
        [
            "http://127.0.0.1:11435/v1",
            "https://host.example.com:8443/v1/rerank?x=1&y=2",
            "http://[::1]:11435/v1",
        ],
    )
    def test_sanitize_url_preserves_valid(self, good: str) -> None:
        assert _sanitize_toml_url(good) == good

    @pytest.mark.parametrize(
        "bad",
        ['http://x"/y', "http://x\\y", "http://x y", "http://x\nY", "a" * 300],
    )
    def test_sanitize_url_rejects_unsafe(self, bad: str) -> None:
        assert _sanitize_toml_url(bad) == ""


# ---------------------------------------------------------------------------
# Config → BrainConfig bridge
# ---------------------------------------------------------------------------


class TestRerankerBridge:
    def test_overrides_map_all_fields(self) -> None:
        rc = RerankerConfig(
            enabled=True,
            model_name="BAAI/bge-reranker-v2-m3",
            blend_weight=0.65,
            min_score=0.2,
            max_candidates=25,
            endpoint="http://127.0.0.1:11435/v1",
        )
        ov = reranker_brain_config_overrides(rc)
        assert ov == {
            "reranker_enabled": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_blend_weight": 0.65,
            "reranker_min_score": 0.2,
            "reranker_max_candidates": 25,
            "reranker_endpoint": "http://127.0.0.1:11435/v1",
        }
        # Every key is a real BrainConfig field
        bc = BrainConfig(**ov)
        assert bc.reranker_enabled is True
        assert bc.reranker_endpoint == "http://127.0.0.1:11435/v1"

    def test_to_brain_config_kwargs_includes_reranker(self) -> None:
        rc = RerankerConfig(enabled=True, endpoint="http://127.0.0.1:11435/v1")
        kwargs = BrainSettings().to_brain_config_kwargs(None, rc)
        assert kwargs["reranker_enabled"] is True
        assert kwargs["reranker_endpoint"] == "http://127.0.0.1:11435/v1"
        # Result is a valid BrainConfig
        bc = BrainConfig(**kwargs)
        assert bc.reranker_endpoint == "http://127.0.0.1:11435/v1"

    def test_to_brain_config_kwargs_without_reranker_unchanged(self) -> None:
        kwargs = BrainSettings().to_brain_config_kwargs(None)
        assert "reranker_enabled" not in kwargs


# ---------------------------------------------------------------------------
# Per-brain BrainConfig persistence (store)
# ---------------------------------------------------------------------------


class TestBrainConfigPersistence:
    def test_roundtrip_preserves_reranker(self) -> None:
        bc = BrainConfig(
            reranker_enabled=True,
            reranker_endpoint="http://127.0.0.1:11435/v1",
            reranker_blend_weight=0.65,
        )
        stored = _serialize_brain_config(bc)
        assert stored["reranker_endpoint"] == "http://127.0.0.1:11435/v1"
        restored = _deserialize_brain_config(stored)
        assert restored.reranker_enabled is True
        assert restored.reranker_endpoint == "http://127.0.0.1:11435/v1"
        assert restored.reranker_blend_weight == 0.65

    def test_legacy_metadata_config_yields_defaults(self) -> None:
        """Pre-2.7.0 rows stored metadata in the config column — deserialise to
        a default BrainConfig instead of raising."""
        legacy = {"some_metadata_key": "value", "session": "x"}
        bc = _deserialize_brain_config(legacy)
        assert bc.reranker_enabled is False
        assert isinstance(bc, BrainConfig)

    def test_empty_and_non_dict_yield_defaults(self) -> None:
        assert _deserialize_brain_config({}).reranker_enabled is False
        assert _deserialize_brain_config(None).reranker_enabled is False
        assert _deserialize_brain_config("garbage").reranker_enabled is False

    def test_unknown_keys_are_dropped(self) -> None:
        stored = _serialize_brain_config(BrainConfig(reranker_enabled=True))
        stored["a_field_removed_in_a_future_version"] = 123
        restored = _deserialize_brain_config(stored)
        assert restored.reranker_enabled is True


class TestRerankActivationsSelection:
    """The endpoint path must fire even without an in-process CrossEncoder."""

    def test_no_endpoint_no_encoder_returns_unchanged(self) -> None:
        activations = {"n1": FakeActivationResult("n1", 0.8)}
        with patch("surreal_memory.engine.reranker._check_cross_encoder", return_value=False):
            result = rerank_activations("q", activations, {"n1": "c"}, endpoint=None)
        assert result is activations

    def test_http_used_even_without_cross_encoder(self) -> None:
        activations = {
            "n_high": FakeActivationResult("n_high", 0.9),
            "n_rel": FakeActivationResult("n_rel", 0.4),
        }
        payload = _rerank_payload({0: -5.0, 1: 8.0})
        with (
            patch("surreal_memory.engine.reranker._check_cross_encoder", return_value=False),
            patch("urllib.request.urlopen", return_value=_FakeResp(payload)),
        ):
            result = rerank_activations(
                "q",
                activations,
                {"n_high": "u", "n_rel": "r"},
                endpoint="http://127.0.0.1:11435/v1",
            )
        assert result["n_rel"].activation_level > result["n_high"].activation_level


# ---------------------------------------------------------------------------
# SQLite backend persistence + config layering (real store round-trips)
# ---------------------------------------------------------------------------

ENDPOINT = "http://127.0.0.1:11435/v1"


class TestSqliteRerankerPersistence:
    """The SQLite backend must persist reranker_* too, or reranking is dead code
    there even though it works on SurrealDB."""

    @pytest.mark.asyncio
    async def test_sqlite_roundtrip_persists_reranker(self, tmp_path: Path) -> None:
        store = SQLiteStorage(tmp_path / "s.db")
        await store.initialize()
        try:
            brain = Brain.create(
                name="rr_brain",
                config=BrainConfig(
                    reranker_enabled=True,
                    reranker_endpoint=ENDPOINT,
                    reranker_blend_weight=0.65,
                ),
            )
            await store.save_brain(brain)

            got = await store.get_brain(brain.id)
            assert got is not None
            assert got.config.reranker_enabled is True
            assert got.config.reranker_endpoint == ENDPOINT
            assert got.config.reranker_blend_weight == 0.65

            by_name = await store.find_brain_by_name("rr_brain")
            assert by_name is not None
            assert by_name.config.reranker_endpoint == ENDPOINT
        finally:
            await store.close()


class TestMigrateDoesNotTouchReranker:
    """Reranker config is deployment/runtime config read from the app config at
    recall time, NOT persisted per-brain. ``_migrate_brain_runtime_config`` must
    leave the stored brain's reranker fields untouched — otherwise a reranker-off
    client (e.g. the web-UI container) would flip the flag on the shared brain for
    everyone (the reranker-flip bug)."""

    @pytest.mark.asyncio
    async def test_migrate_does_not_enable_reranker(self, tmp_path: Path) -> None:
        store = SQLiteStorage(tmp_path / "m.db")
        await store.initialize()
        try:
            brain = Brain.create(name="legacy")  # reranker off by default
            await store.save_brain(brain)

            config = replace(
                UnifiedConfig(data_dir=tmp_path),
                reranker=RerankerConfig(enabled=True, endpoint=ENDPOINT),
            )
            await _migrate_brain_runtime_config(store, brain, config)

            # The app config's reranker is NOT forced onto the stored brain.
            reloaded = await store.get_brain(brain.id)
            assert reloaded.config.reranker_enabled is False
            assert reloaded.config.reranker_endpoint == ""
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_migrate_does_not_disable_reranker(self, tmp_path: Path) -> None:
        # The exact reranker-flip scenario: the shared brain has reranker ON (set
        # by another client), and this client's app config has it OFF. Migration
        # must NOT disable it on the shared brain.
        store = SQLiteStorage(tmp_path / "n.db")
        await store.initialize()
        try:
            brain = Brain.create(
                name="shared",
                config=BrainConfig(reranker_enabled=True, reranker_endpoint=ENDPOINT),
            )
            await store.save_brain(brain)
            stored = await store.get_brain(brain.id)

            config = replace(
                UnifiedConfig(data_dir=tmp_path),
                reranker=RerankerConfig(enabled=False, endpoint=""),  # this client: off
            )
            await _migrate_brain_runtime_config(store, stored, config)

            after = await store.get_brain(brain.id)
            assert after.config.reranker_enabled is True
            assert after.config.reranker_endpoint == ENDPOINT
        finally:
            await store.close()
