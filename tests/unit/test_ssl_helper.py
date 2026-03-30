"""Tests for ssl_helper — certifi-based SSL context for aiohttp (#120)."""

from __future__ import annotations

import ssl
from unittest.mock import MagicMock, patch

from neural_memory.utils.ssl_helper import get_ssl_context, safe_client_session


class TestGetSslContext:
    """Test SSL context creation."""

    def setup_method(self) -> None:
        get_ssl_context.cache_clear()

    def teardown_method(self) -> None:
        get_ssl_context.cache_clear()

    def test_returns_ssl_context_when_certifi_available(self) -> None:
        """certifi installed → returns SSLContext with certifi CA bundle."""
        # certifi IS installed in test env, so default behavior should work
        result = get_ssl_context()
        assert isinstance(result, ssl.SSLContext)

    def test_result_is_cached(self) -> None:
        """Consecutive calls return the same cached object."""
        a = get_ssl_context()
        b = get_ssl_context()
        assert a is b


class TestSafeClientSession:
    """Test safe_client_session creates session with SSL connector."""

    def setup_method(self) -> None:
        get_ssl_context.cache_clear()

    def teardown_method(self) -> None:
        get_ssl_context.cache_clear()

    def test_creates_session_with_ssl_connector(self) -> None:
        fake_ctx = MagicMock(spec=ssl.SSLContext)
        with (
            patch("neural_memory.utils.ssl_helper.get_ssl_context", return_value=fake_ctx),
            patch("aiohttp.TCPConnector") as mock_connector,
            patch("aiohttp.ClientSession") as mock_session,
        ):
            result = safe_client_session(timeout=30)
            mock_connector.assert_called_once_with(ssl=fake_ctx)
            mock_session.assert_called_once_with(connector=mock_connector.return_value, timeout=30)
            assert result == mock_session.return_value

    def test_passes_extra_kwargs_to_session(self) -> None:
        with (
            patch("neural_memory.utils.ssl_helper.get_ssl_context", return_value=True),
            patch("aiohttp.TCPConnector"),
            patch("aiohttp.ClientSession") as mock_session,
        ):
            safe_client_session(headers={"X-Test": "1"})
            call_kwargs = mock_session.call_args[1]
            assert call_kwargs["headers"] == {"X-Test": "1"}
