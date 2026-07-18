"""Tests for unified_config.py — legacy DB migration + config sync."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from surreal_memory.cli.config import _sync_brain_to_toml
from surreal_memory.unified_config import (
    _MIN_LEGACY_DB_BYTES,
    ReasoningTrainingConfig,
    UnifiedConfig,
    _migrate_legacy_db,
    _read_current_brain_from_toml,
    _read_legacy_brain,
)


def _create_fake_db(path: Path, *, size: int = 0) -> None:
    """Create a minimal SQLite database at *path*.

    If *size* is given and larger than a bare DB, pad with extra data so
    ``stat().st_size >= size``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS neurons (id TEXT PRIMARY KEY)")
    conn.execute("INSERT OR IGNORE INTO neurons VALUES ('test-neuron-1')")
    conn.commit()
    conn.close()

    # Pad to requested size if needed.
    current = path.stat().st_size
    if size > current:
        with open(path, "ab") as f:
            f.write(b"\x00" * (size - current))


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Return a temporary Surreal-Memory data directory."""
    return tmp_path / ".surrealmemory"


def _make_config(data_dir: Path) -> UnifiedConfig:
    """Build a UnifiedConfig pointing at *data_dir* with brain='default'."""
    return UnifiedConfig(data_dir=data_dir, current_brain="default")


# ── Happy path ───────────────────────────────────────────────────


class TestMigrateLegacyDb:
    def test_copies_when_old_exists_and_new_does_not(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert new_db.exists()
        assert new_db.stat().st_size == old_db.stat().st_size

        # Old file still exists (backup).
        assert old_db.exists()

    def test_copies_wal_and_shm_if_present(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        # Create fake WAL/SHM companions.
        wal = old_db.with_name("default.db-wal")
        shm = old_db.with_name("default.db-shm")
        wal.write_bytes(b"wal-data")
        shm.write_bytes(b"shm-data")

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        brains = tmp_data_dir / "brains"
        assert (brains / "default.db-wal").read_bytes() == b"wal-data"
        assert (brains / "default.db-shm").read_bytes() == b"shm-data"

    # ── Skip conditions ──────────────────────────────────────────

    def test_skips_when_new_already_exists(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        new_db = tmp_data_dir / "brains" / "default.db"
        new_db.parent.mkdir(parents=True, exist_ok=True)
        new_db.write_text("existing")

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        # Should NOT overwrite existing new DB.
        assert new_db.read_text() == "existing"

    def test_skips_non_default_brain(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, "my-custom-brain")

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    def test_skips_small_file(self, tmp_data_dir: Path) -> None:
        """An empty-schema DB (< _MIN_LEGACY_DB_BYTES) is not migrated."""
        old_db = tmp_data_dir / "default.db"
        old_db.parent.mkdir(parents=True, exist_ok=True)
        old_db.write_bytes(b"\x00" * 4096)

        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    def test_skips_when_old_does_not_exist(self, tmp_data_dir: Path) -> None:
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        config = _make_config(tmp_data_dir)
        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    # ── Error resilience ─────────────────────────────────────────

    def test_handles_copy_error_gracefully(self, tmp_data_dir: Path) -> None:
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)

        with patch("surreal_memory.unified_config.shutil.copy2", side_effect=OSError("disk full")):
            # Should not raise — logs warning instead.
            _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert not new_db.exists()

    # ── Config brain name resolution ─────────────────────────────

    def test_uses_config_current_brain_when_none(self, tmp_data_dir: Path) -> None:
        """When brain_name is None, uses config.current_brain."""
        old_db = tmp_data_dir / "default.db"
        _create_fake_db(old_db, size=_MIN_LEGACY_DB_BYTES + 1024)

        config = _make_config(tmp_data_dir)
        assert config.current_brain == "default"

        _migrate_legacy_db(config, None)

        new_db = tmp_data_dir / "brains" / "default.db"
        assert new_db.exists()


# ── Config sync tests ────────────────────────────────────────────


def _write_toml(data_dir: Path, brain_name: str = "default") -> Path:
    """Write a minimal config.toml and return its path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    toml_path = data_dir / "config.toml"
    toml_path.write_text(
        f'version = "1.0"\ncurrent_brain = "{brain_name}"\n\n[brain]\ndecay_rate = 0.1\n',
        encoding="utf-8",
    )
    return toml_path


class TestSyncBrainToToml:
    """Tests for CLI → TOML sync via _sync_brain_to_toml."""

    def test_updates_current_brain_in_toml(self, tmp_data_dir: Path) -> None:
        _write_toml(tmp_data_dir, "default")

        _sync_brain_to_toml(tmp_data_dir, "work")

        content = (tmp_data_dir / "config.toml").read_text(encoding="utf-8")
        assert 'current_brain = "work"' in content

    def test_preserves_other_toml_content(self, tmp_data_dir: Path) -> None:
        _write_toml(tmp_data_dir, "default")

        _sync_brain_to_toml(tmp_data_dir, "work")

        content = (tmp_data_dir / "config.toml").read_text(encoding="utf-8")
        assert "decay_rate = 0.1" in content
        assert 'version = "1.0"' in content

    def test_noop_when_toml_missing(self, tmp_data_dir: Path) -> None:
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        # Should not raise
        _sync_brain_to_toml(tmp_data_dir, "work")

    def test_rejects_invalid_brain_name(self, tmp_data_dir: Path) -> None:
        _write_toml(tmp_data_dir, "default")

        _sync_brain_to_toml(tmp_data_dir, "../escape")

        # Should not have changed
        content = (tmp_data_dir / "config.toml").read_text(encoding="utf-8")
        assert 'current_brain = "default"' in content

    def test_handles_write_error_gracefully(self, tmp_data_dir: Path) -> None:
        _write_toml(tmp_data_dir, "default")

        with patch("surreal_memory.cli.config.Path.write_text", side_effect=OSError("perm")):
            # Should not raise
            _sync_brain_to_toml(tmp_data_dir, "work")


class TestReadCurrentBrainFromToml:
    """Tests for MCP-side toml reading via _read_current_brain_from_toml."""

    def test_reads_brain_name(self, tmp_data_dir: Path) -> None:
        _write_toml(tmp_data_dir, "my-brain")

        with patch(
            "surreal_memory.unified_config.get_surrealmemory_dir", return_value=tmp_data_dir
        ):
            result = _read_current_brain_from_toml()
        assert result == "my-brain"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        with patch("surreal_memory.unified_config.get_surrealmemory_dir", return_value=tmp_path):
            result = _read_current_brain_from_toml()
        assert result is None

    def test_returns_none_for_invalid_name(self, tmp_data_dir: Path) -> None:
        data_dir = tmp_data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        toml_path = data_dir / "config.toml"
        toml_path.write_text('current_brain = "../hacked"\n', encoding="utf-8")

        with patch("surreal_memory.unified_config.get_surrealmemory_dir", return_value=data_dir):
            result = _read_current_brain_from_toml()
        assert result is None


class TestEndToEndBrainSync:
    """Integration test: CLI save → TOML sync → MCP reads new brain."""

    def test_cli_save_syncs_to_toml_and_mcp_reads_it(self, tmp_data_dir: Path) -> None:
        _write_toml(tmp_data_dir, "default")

        # Simulate CLI brain switch
        _sync_brain_to_toml(tmp_data_dir, "work")

        # Simulate MCP reading the toml
        with patch(
            "surreal_memory.unified_config.get_surrealmemory_dir", return_value=tmp_data_dir
        ):
            result = _read_current_brain_from_toml()
        assert result == "work"

        # Verify the config singleton would pick it up
        config = _make_config(tmp_data_dir)
        assert config.current_brain == "default"  # old value in memory

        # After sync detection, config updates
        if result is not None and result != config.current_brain:
            config.current_brain = result
        assert config.current_brain == "work"


# ── Legacy brain migration tests ────────────────────────────────


def _write_legacy_json(data_dir: Path, brain_name: str) -> Path:
    """Write a legacy config.json with the given brain name."""
    import json

    data_dir.mkdir(parents=True, exist_ok=True)
    config_file = data_dir / "config.json"
    config_file.write_text(
        json.dumps({"current_brain": brain_name}),
        encoding="utf-8",
    )
    return config_file


class TestReadLegacyBrain:
    """Tests for _read_legacy_brain — reads current_brain from config.json."""

    def test_reads_from_same_dir(self, tmp_data_dir: Path) -> None:
        _write_legacy_json(tmp_data_dir, "myproject")
        result = _read_legacy_brain(tmp_data_dir)
        assert result == "myproject"

    def test_returns_none_for_default_brain(self, tmp_data_dir: Path) -> None:
        _write_legacy_json(tmp_data_dir, "default")
        result = _read_legacy_brain(tmp_data_dir)
        assert result is None

    def test_returns_none_when_no_json(self, tmp_data_dir: Path) -> None:
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        result = _read_legacy_brain(tmp_data_dir)
        assert result is None

    def test_reads_from_legacy_dir(self, tmp_path: Path) -> None:
        """Falls back to ~/.surreal-memory/ when data_dir has no config.json."""
        data_dir = tmp_path / ".surrealmemory"
        data_dir.mkdir(parents=True, exist_ok=True)

        legacy_dir = tmp_path / ".surreal-memory"
        _write_legacy_json(legacy_dir, "work-brain")

        with patch("surreal_memory.unified_config.Path.home", return_value=tmp_path):
            result = _read_legacy_brain(data_dir)
        assert result == "work-brain"

    def test_rejects_invalid_brain_name(self, tmp_data_dir: Path) -> None:
        _write_legacy_json(tmp_data_dir, "../escape")
        result = _read_legacy_brain(tmp_data_dir)
        assert result is None

    def test_handles_corrupt_json(self, tmp_data_dir: Path) -> None:
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        (tmp_data_dir / "config.json").write_text("not json", encoding="utf-8")
        result = _read_legacy_brain(tmp_data_dir)
        assert result is None


class TestConfigLoadMigratesBrain:
    """Tests for UnifiedConfig.load() migrating current_brain from config.json."""

    def test_migrates_brain_from_legacy_json(self, tmp_data_dir: Path) -> None:
        """When config.toml doesn't exist, load() picks up brain from config.json."""
        _write_legacy_json(tmp_data_dir, "myproject")

        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        assert config.current_brain == "myproject"
        # config.toml should now exist with the migrated brain
        toml_content = (tmp_data_dir / "config.toml").read_text(encoding="utf-8")
        assert 'current_brain = "myproject"' in toml_content

    def test_defaults_when_no_legacy_json(self, tmp_data_dir: Path) -> None:
        """When neither config.toml nor config.json exist, uses default."""
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")
        assert config.current_brain == "default"

    def test_existing_toml_not_overridden(self, tmp_data_dir: Path) -> None:
        """When config.toml already exists, config.json is NOT consulted."""
        _write_legacy_json(tmp_data_dir, "old-brain")
        _write_toml(tmp_data_dir, "toml-brain")

        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")
        assert config.current_brain == "toml-brain"


def _write_embedding_toml(
    data_dir: Path,
    *,
    enabled: bool = False,
    provider: str = "sentence_transformer",
    model: str = "all-MiniLM-L6-v2",
    threshold: float = 0.7,
) -> Path:
    """Write a config.toml with an [embedding] section and return its path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    toml_path = data_dir / "config.toml"
    toml_path.write_text(
        'version = "1.0"\n'
        'current_brain = "default"\n\n'
        "[embedding]\n"
        f"enabled = {str(enabled).lower()}\n"
        f'provider = "{provider}"\n'
        f'model = "{model}"\n'
        f"similarity_threshold = {threshold}\n",
        encoding="utf-8",
    )
    return toml_path


class TestEmbeddingEnvOverrides:
    """Tests for env vars overriding the config.toml [embedding] section."""

    def test_toml_values_used_when_no_env(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in (
            "SURREAL_MEMORY_EMBEDDING_ENABLED",
            "SURREAL_MEMORY_EMBEDDING_PROVIDER",
            "SURREAL_MEMORY_EMBEDDING_MODEL",
            "SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD",
        ):
            monkeypatch.delenv(var, raising=False)

        _write_embedding_toml(tmp_data_dir, enabled=False, provider="sentence_transformer")
        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        assert config.embedding.enabled is False
        assert config.embedding.provider == "sentence_transformer"
        assert config.embedding.model == "all-MiniLM-L6-v2"

    def test_env_overrides_toml(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The stale-default bug: toml says sentence_transformer/disabled but the
        env (set during MCP registration) says gemini/enabled — env must win.
        """
        _write_embedding_toml(tmp_data_dir, enabled=False, provider="sentence_transformer")
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_ENABLED", "true")
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_MODEL", "gemini-embedding-001")
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD", "0.55")

        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        assert config.embedding.enabled is True
        assert config.embedding.provider == "gemini"
        assert config.embedding.model == "gemini-embedding-001"
        assert config.embedding.similarity_threshold == 0.55

    def test_partial_env_keeps_other_toml_values(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_embedding_toml(
            tmp_data_dir, enabled=True, provider="openai", model="text-embedding-3-small"
        )
        for var in (
            "SURREAL_MEMORY_EMBEDDING_ENABLED",
            "SURREAL_MEMORY_EMBEDDING_MODEL",
            "SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_PROVIDER", "ollama")

        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        # Only provider overridden; enabled/model from toml stay.
        assert config.embedding.enabled is True
        assert config.embedding.provider == "ollama"
        assert config.embedding.model == "text-embedding-3-small"

    def test_empty_env_does_not_override(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_embedding_toml(tmp_data_dir, enabled=True, provider="gemini")
        monkeypatch.setenv("SURREAL_MEMORY_EMBEDDING_PROVIDER", "")
        monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENABLED", raising=False)

        config = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        assert config.embedding.enabled is True
        assert config.embedding.provider == "gemini"


class TestSqliteBackendWarning:
    """Protection: a misconfigured install must not silently land on SQLite.

    Surfaces the "unwanted SQLite brain" / data-split footgun loudly instead of
    quietly writing memories to a local SQLite brain that diverges from the
    SurrealDB the dashboard reads.
    """

    def test_warns_about_data_split_when_surreal_env_present(self, monkeypatch, caplog):
        import surreal_memory.unified_config as uc

        monkeypatch.setattr(uc, "_sqlite_backend_warned", False)
        monkeypatch.setenv("SURREALDB_URL", "http://localhost:8001")
        with caplog.at_level("WARNING"):
            uc._warn_sqlite_backend()
        assert any("split" in r.message.lower() for r in caplog.records)

    def test_emits_only_once(self, monkeypatch, caplog):
        import surreal_memory.unified_config as uc

        monkeypatch.setattr(uc, "_sqlite_backend_warned", False)
        monkeypatch.delenv("SURREALDB_URL", raising=False)
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        with caplog.at_level("WARNING"):
            uc._warn_sqlite_backend()
            uc._warn_sqlite_backend()
        sqlite_warnings = [r for r in caplog.records if "sqlite" in r.message.lower()]
        assert len(sqlite_warnings) == 1


class TestWarnMissingSurrealPass:
    def test_warns_when_storage_surrealdb_and_no_pass(self, monkeypatch, caplog):
        import surreal_memory.unified_config as uc

        monkeypatch.setattr(uc, "_missing_surreal_pass_warned", False)
        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        with caplog.at_level("WARNING"):
            uc._warn_missing_surreal_pass()
        warnings = [r for r in caplog.records if "SURREALDB_PASS" in r.message]
        assert len(warnings) == 1

    def test_no_warning_when_pass_set(self, monkeypatch, caplog):
        import surreal_memory.unified_config as uc

        monkeypatch.setattr(uc, "_missing_surreal_pass_warned", False)
        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        monkeypatch.setenv("SURREALDB_PASS", "mypassword")
        with caplog.at_level("WARNING"):
            uc._warn_missing_surreal_pass()
        warnings = [r for r in caplog.records if "SURREALDB_PASS" in r.message]
        assert len(warnings) == 0

    def test_no_warning_when_not_surrealdb_backend(self, monkeypatch, caplog):
        import surreal_memory.unified_config as uc

        monkeypatch.setattr(uc, "_missing_surreal_pass_warned", False)
        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "sqlite")
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        with caplog.at_level("WARNING"):
            uc._warn_missing_surreal_pass()
        warnings = [r for r in caplog.records if "SURREALDB_PASS" in r.message]
        assert len(warnings) == 0

    def test_emits_only_once(self, monkeypatch, caplog):
        import surreal_memory.unified_config as uc

        monkeypatch.setattr(uc, "_missing_surreal_pass_warned", False)
        monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "surrealdb")
        monkeypatch.delenv("SURREALDB_PASS", raising=False)
        with caplog.at_level("WARNING"):
            uc._warn_missing_surreal_pass()
            uc._warn_missing_surreal_pass()
        warnings = [r for r in caplog.records if "SURREALDB_PASS" in r.message]
        assert len(warnings) == 1


class TestReasoningTrainingConfig:
    """Tests for the [reasoning_training] config section (mining + injection)."""

    _ENV_VARS = (
        "SURREAL_MEMORY_REASONING_MINING",
        "SURREAL_MEMORY_REASONING_INJECTION",
        "SURREAL_MEMORY_REASONING_MODELS",
        "SURREAL_MEMORY_REASONING_INJECTION_MAP",
    )

    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in self._ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_defaults_are_opt_in_off(self) -> None:
        rt = ReasoningTrainingConfig()
        assert rt.mining_enabled is False
        assert rt.injection_enabled is False
        assert rt.mining_models == ()
        assert rt.injection_map == ()
        assert "debugging" in rt.categories
        assert "data-analysis" in rt.categories
        assert rt.min_confidence == 0.2
        assert rt.redact_secrets is True

    def test_default_max_trace_chars_is_100k(self) -> None:
        # Bumped 20_000 -> 100_000: content safety, not a real-world cut (no
        # per-scan cap left to bound total volume, so the char cap stays the
        # only safety valve — raised because real traces were being truncated).
        assert ReasoningTrainingConfig().max_trace_chars == 100_000
        assert ReasoningTrainingConfig.from_dict({}).max_trace_chars == 100_000

    def test_from_dict_ignores_legacy_max_traces_per_scan_key(self) -> None:
        # max_traces_per_scan was removed (u2): an old config.toml/dict still
        # carrying the key must load without raising, and the field must not
        # reappear on the resulting instance.
        rt = ReasoningTrainingConfig.from_dict({"max_traces_per_scan": 500, "min_confidence": 0.4})
        assert not hasattr(rt, "max_traces_per_scan")
        assert rt.min_confidence == 0.4

    def test_from_dict_to_dict_roundtrip(self) -> None:
        rt = ReasoningTrainingConfig(
            mining_enabled=True,
            injection_enabled=True,
            mining_models=("claude-fable-*",),
            injection_map=(("claude-opus-*", "claude-fable-5"),),
            min_confidence=0.5,
            max_traces_total=1234,
        )
        assert ReasoningTrainingConfig.from_dict(rt.to_dict()) == rt

    def test_from_dict_clamps_min_confidence(self) -> None:
        assert ReasoningTrainingConfig.from_dict({"min_confidence": 5}).min_confidence == 1.0
        assert ReasoningTrainingConfig.from_dict({"min_confidence": -1}).min_confidence == 0.0

    def test_from_dict_injection_map_dict_and_list_forms(self) -> None:
        from_dict = ReasoningTrainingConfig.from_dict(
            {"injection_map": {"claude-opus-*": "claude-fable-5"}}
        )
        assert from_dict.injection_map == (("claude-opus-*", "claude-fable-5"),)
        # Malformed list entries (wrong arity) are dropped; valid pairs kept.
        from_list = ReasoningTrainingConfig.from_dict({"injection_map": [["a", "b"], ["bad"]]})
        assert from_list.injection_map == (("a", "b"),)

    def test_toml_roundtrip_preserves_globs_and_injection_map(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        rt = ReasoningTrainingConfig(
            mining_enabled=True,
            injection_enabled=True,
            mining_models=("claude-fable-*", "glm-5.2"),
            injection_map=(("claude-opus-*", "claude-fable-5"),),
            min_confidence=0.35,
            scan_lookback_days=0,
        )
        UnifiedConfig(data_dir=tmp_data_dir, current_brain="default", reasoning_training=rt).save()
        reloaded = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        got = reloaded.reasoning_training
        assert got.mining_enabled is True
        assert got.injection_enabled is True
        # Globs survive save/load (would be silently dropped by _sanitize_toml_str).
        assert got.mining_models == ("claude-fable-*", "glm-5.2")
        assert got.injection_map == (("claude-opus-*", "claude-fable-5"),)
        assert got.min_confidence == 0.35
        assert got.scan_lookback_days == 0

    def test_env_overrides_toml(self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_env(monkeypatch)
        UnifiedConfig(data_dir=tmp_data_dir, current_brain="default").save()  # mining off
        monkeypatch.setenv("SURREAL_MEMORY_REASONING_MINING", "true")
        monkeypatch.setenv("SURREAL_MEMORY_REASONING_INJECTION", "1")
        monkeypatch.setenv("SURREAL_MEMORY_REASONING_MODELS", "claude-fable-*, glm-5.2")
        monkeypatch.setenv(
            "SURREAL_MEMORY_REASONING_INJECTION_MAP",
            "claude-opus-*=claude-fable-5, claude-haiku-*=glm-5.2",
        )
        cfg = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        assert cfg.reasoning_training.mining_enabled is True
        assert cfg.reasoning_training.injection_enabled is True
        assert cfg.reasoning_training.mining_models == ("claude-fable-*", "glm-5.2")
        assert cfg.reasoning_training.injection_map == (
            ("claude-opus-*", "claude-fable-5"),
            ("claude-haiku-*", "glm-5.2"),
        )

    def test_empty_env_does_not_override(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        rt = ReasoningTrainingConfig(mining_enabled=True, mining_models=("x-*",))
        UnifiedConfig(data_dir=tmp_data_dir, current_brain="default", reasoning_training=rt).save()
        monkeypatch.setenv("SURREAL_MEMORY_REASONING_MODELS", "")  # empty → ignored
        cfg = UnifiedConfig.load(config_path=tmp_data_dir / "config.toml")

        assert cfg.reasoning_training.mining_enabled is True
        assert cfg.reasoning_training.mining_models == ("x-*",)
