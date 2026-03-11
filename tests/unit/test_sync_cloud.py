"""Tests for cloud sync features: api_key, URL versioning, error handling, key masking."""

import pytest

from neural_memory.mcp.sync_handler import (
    _build_hub_url,
    _build_sync_url,
    _handle_http_error,
    _mask_key,
)
from neural_memory.unified_config import SyncConfig


class TestSyncConfig:
    """SyncConfig api_key field tests."""

    def test_api_key_default_empty(self) -> None:
        config = SyncConfig()
        assert config.api_key == ""

    def test_api_key_from_dict(self) -> None:
        config = SyncConfig.from_dict({"api_key": "nmk_abc123def456"})
        assert config.api_key == "nmk_abc123def456"

    def test_api_key_invalid_prefix_rejected(self) -> None:
        config = SyncConfig.from_dict({"api_key": "invalid_key"})
        assert config.api_key == ""

    def test_api_key_empty_string_ok(self) -> None:
        config = SyncConfig.from_dict({"api_key": ""})
        assert config.api_key == ""

    def test_api_key_in_to_dict(self) -> None:
        config = SyncConfig(api_key="nmk_test123")
        d = config.to_dict()
        assert d["api_key"] == "nmk_test123"

    def test_roundtrip_with_api_key(self) -> None:
        original = SyncConfig(
            enabled=True,
            hub_url="https://example.com",
            api_key="nmk_roundtrip123456789012345678",
            auto_sync=True,
            sync_interval_seconds=120,
            conflict_strategy="prefer_local",
        )
        restored = SyncConfig.from_dict(original.to_dict())
        assert restored == original

    def test_config_frozen(self) -> None:
        config = SyncConfig(api_key="nmk_test")
        with pytest.raises(AttributeError):
            config.api_key = "nmk_other"  # type: ignore[misc]


class TestKeyMasking:
    """API key masking for display."""

    def test_mask_valid_key(self) -> None:
        assert _mask_key("nmk_a1b2c3d4e5f6g7h8i9j0") == "nmk_a1b2c3d4****"

    def test_mask_empty(self) -> None:
        assert _mask_key("") == "(not set)"

    def test_mask_short(self) -> None:
        assert _mask_key("nmk_abc") == "(not set)"

    def test_mask_exact_12_chars(self) -> None:
        assert _mask_key("nmk_12345678") == "nmk_12345678****"


class TestUrlVersioning:
    """URL versioning for cloud vs local hub."""

    def test_cloud_url_gets_v1(self) -> None:
        url = _build_sync_url("https://sync-hub.neuralmemory.dev")
        assert url == "https://sync-hub.neuralmemory.dev/v1/hub/sync"

    def test_localhost_no_v1(self) -> None:
        url = _build_sync_url("http://localhost:8000")
        assert url == "http://localhost:8000/hub/sync"

    def test_127_no_v1(self) -> None:
        url = _build_sync_url("http://127.0.0.1:8000")
        assert url == "http://127.0.0.1:8000/hub/sync"

    def test_trailing_slash_stripped(self) -> None:
        url = _build_sync_url("https://hub.example.com/")
        assert url == "https://hub.example.com/v1/hub/sync"

    def test_build_hub_url_cloud(self) -> None:
        url = _build_hub_url("https://hub.example.com", "/auth/me")
        assert url == "https://hub.example.com/v1/auth/me"

    def test_build_hub_url_local(self) -> None:
        url = _build_hub_url("http://localhost:8000", "/health")
        assert url == "http://localhost:8000/health"


class TestHttpErrorHandling:
    """HTTP error message mapping."""

    def test_401_message(self) -> None:
        result = _handle_http_error(401)
        assert result["http_status"] == 401
        assert "API key" in result["message"]

    def test_403_message(self) -> None:
        result = _handle_http_error(403)
        assert result["http_status"] == 403
        assert "Access denied" in result["message"]

    def test_413_message(self) -> None:
        result = _handle_http_error(413)
        assert "large" in result["message"].lower()

    def test_429_message(self) -> None:
        result = _handle_http_error(429)
        assert "Rate limited" in result["message"]

    def test_unknown_status(self) -> None:
        result = _handle_http_error(502)
        assert "502" in result["message"]
        assert result["status"] == "error"
