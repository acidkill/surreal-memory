"""Tests for LicenseConfig — tier validation, is_pro(), load/save roundtrip."""

from pathlib import Path

import pytest

from neural_memory.unified_config import LicenseConfig, UnifiedConfig


# ── LicenseConfig dataclass ─────────────────────────────────────


class TestLicenseConfig:
    def test_defaults(self) -> None:
        cfg = LicenseConfig()
        assert cfg.tier == "free"
        assert cfg.activated_at == ""
        assert cfg.expires_at == ""

    def test_from_dict_valid_tiers(self) -> None:
        for tier in ("free", "pro", "team"):
            cfg = LicenseConfig.from_dict({"tier": tier})
            assert cfg.tier == tier

    def test_from_dict_invalid_tier_falls_back_to_free(self) -> None:
        cfg = LicenseConfig.from_dict({"tier": "enterprise"})
        assert cfg.tier == "free"

    def test_from_dict_case_insensitive(self) -> None:
        cfg = LicenseConfig.from_dict({"tier": "PRO"})
        assert cfg.tier == "pro"

    def test_from_dict_empty(self) -> None:
        cfg = LicenseConfig.from_dict({})
        assert cfg.tier == "free"
        assert cfg.activated_at == ""

    def test_to_dict_roundtrip(self) -> None:
        original = LicenseConfig(tier="pro", activated_at="2026-03-24", expires_at="2027-03-24")
        restored = LicenseConfig.from_dict(original.to_dict())
        assert restored.tier == original.tier
        assert restored.activated_at == original.activated_at
        assert restored.expires_at == original.expires_at

    def test_frozen(self) -> None:
        cfg = LicenseConfig()
        with pytest.raises(AttributeError):
            cfg.tier = "pro"  # type: ignore[misc]


# ── UnifiedConfig.is_pro() ──────────────────────────────────────


class TestIsPro:
    def test_free_is_not_pro(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="free"))
        assert cfg.is_pro() is False

    def test_pro_is_pro(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="pro"))
        assert cfg.is_pro() is True

    def test_team_is_pro(self) -> None:
        cfg = UnifiedConfig(license=LicenseConfig(tier="team"))
        assert cfg.is_pro() is True

    def test_default_is_not_pro(self) -> None:
        cfg = UnifiedConfig()
        assert cfg.is_pro() is False


# ── TOML save/load roundtrip ────────────────────────────────────


class TestLicenseTomlRoundtrip:
    def test_save_and_load_preserves_license(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".neuralmemory"
        data_dir.mkdir()

        original = UnifiedConfig(
            data_dir=data_dir,
            current_brain="default",
            license=LicenseConfig(
                tier="pro",
                activated_at="2026-03-24T10:00:00",
                expires_at="2027-03-24T10:00:00",
            ),
        )
        original.save()

        # Verify TOML contains [license] section
        toml_content = (data_dir / "config.toml").read_text()
        assert "[license]" in toml_content
        assert 'tier = "pro"' in toml_content

        loaded = UnifiedConfig.load(data_dir / "config.toml")
        assert loaded.license.tier == "pro"
        assert loaded.license.activated_at == "2026-03-24T10:00:00"
        assert loaded.license.expires_at == "2027-03-24T10:00:00"
        assert loaded.is_pro() is True

    def test_load_without_license_section_defaults_to_free(self, tmp_path: Path) -> None:
        data_dir = tmp_path / ".neuralmemory"
        data_dir.mkdir()

        # Save a config, then strip the [license] section
        cfg = UnifiedConfig(data_dir=data_dir, current_brain="default")
        cfg.save()

        toml_path = data_dir / "config.toml"
        lines = toml_path.read_text().splitlines()
        filtered = []
        skip = False
        for line in lines:
            if line.strip() == "[license]":
                skip = True
                continue
            if skip and line.strip().startswith("["):
                skip = False
            if not skip:
                filtered.append(line)
        toml_path.write_text("\n".join(filtered) + "\n")

        loaded = UnifiedConfig.load(toml_path)
        assert loaded.license.tier == "free"
        assert loaded.is_pro() is False
