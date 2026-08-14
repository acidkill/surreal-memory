"""Transport selection: http URLs are rewritten to ws to avoid port exhaustion.

The surrealdb SDK's HTTP transport opens a new TCP connection per RPC. On
workloads that issue tens of thousands of small queries (consolidation's
compress / semantic_link), that exhausts the ephemeral port range and every
new connection — including the SDK's own reconnect — gets RST by the kernel
("[Errno 104] Connection reset by peer", the long-standing `smem consolidate`
failure). The store therefore rewrites http(s) URLs to ws(s) so the SDK
multiplexes all RPCs over one persistent WebSocket connection.
"""

from __future__ import annotations

from surreal_memory.storage.surrealdb.store import _prefer_ws_transport


class TestPreferWsTransport:
    def test_http_rewritten_to_ws(self):
        assert _prefer_ws_transport("http://localhost:8001") == "ws://localhost:8001"

    def test_https_rewritten_to_wss(self):
        assert _prefer_ws_transport("https://db.example.com") == "wss://db.example.com"

    def test_explicit_ws_unchanged(self):
        assert _prefer_ws_transport("ws://localhost:8001") == "ws://localhost:8001"

    def test_explicit_wss_unchanged(self):
        assert _prefer_ws_transport("wss://db.example.com") == "wss://db.example.com"

    def test_non_http_scheme_unchanged(self):
        # Embedded backends (memory/surrealkv) must pass through untouched.
        assert _prefer_ws_transport("memory://") == "memory://"
        assert _prefer_ws_transport("surrealkv:///data/db") == "surrealkv:///data/db"

    def test_url_with_path_preserved(self):
        assert (
            _prefer_ws_transport("https://host/prefix")
            == "wss://host/prefix"
        )


class TestStoreUsesWsUrl:
    def test_store_rewrites_env_http_url(self, monkeypatch):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        monkeypatch.setenv("SURREALDB_URL", "http://localhost:8001")
        monkeypatch.setenv("SURREALDB_USER", "root")
        monkeypatch.setenv("SURREALDB_PASS", "surrealmemory")
        monkeypatch.setenv("SURREALDB_NS", "surreal_memory")
        monkeypatch.setenv("SURREALDB_DB", "default")

        store = SurrealDBStorage()
        assert store._url == "ws://localhost:8001"

    def test_store_respects_explicit_url_override(self, monkeypatch):
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        monkeypatch.setenv("SURREALDB_URL", "http://localhost:8001")
        store = SurrealDBStorage(url="ws://explicit:9000")
        assert store._url == "ws://explicit:9000"
