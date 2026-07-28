"""Regressions for the follow-ups collected in #98 after the PR-queue cleanup."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest


class TestBrainIdMatchesName:
    """#98/4 — the source of the brain-scope bug, not just its symptoms.

    Rows are scoped by the brain *name*, but ``Brain.create()`` defaults to a
    random uuid4. ``POST /brain`` did not pass ``brain_id``, so a brain created
    through the dashboard or API was born with an id that did not match its own
    row scope — and the next call site reaching for ``brain.id`` would reintroduce
    the bug #97 had just fixed at ~10 call sites.
    """

    def test_create_brain_endpoint_pins_id_to_name(self) -> None:
        from surreal_memory.server.routes import brain as brain_route

        source = inspect.getsource(brain_route.create_brain)
        assert "brain_id=request.name" in source, (
            "POST /brain must pin brain_id to the name; a uuid4 id does not match "
            "the row scope and silently re-opens the #97 class of bug"
        )

    def test_brain_create_still_defaults_to_uuid(self) -> None:
        """The default is fine — callers that own a scope must be explicit."""
        from surreal_memory.core.brain import Brain

        brain = Brain.create(name="scratch")
        assert brain.id != "scratch"

    def test_explicit_brain_id_wins(self) -> None:
        from surreal_memory.core.brain import Brain

        brain = Brain.create(name="scratch", brain_id="scratch")
        assert brain.id == brain.name == "scratch"


class TestSingleEnvLookup:
    """#98/1 — ``os.environ.get(X) or os.environ.get(X)`` read the same key twice.

    Functionally a no-op, so nothing was broken; it reads as if a second variable
    name was intended and got lost, which is exactly the kind of line that
    survives review indefinitely.
    """

    @pytest.mark.parametrize(
        "module_path",
        [
            "surreal_memory/cli/commands/brain.py",
            "surreal_memory/unified_config.py",
        ],
    )
    def test_no_duplicated_env_lookup(self, module_path: str) -> None:
        import surreal_memory

        root = Path(surreal_memory.__file__).resolve().parent.parent
        source = (root / module_path).read_text(encoding="utf-8")
        duplicated = (
            'os.environ.get("SURREAL_MEMORY_BRAIN") or os.environ.get("SURREAL_MEMORY_BRAIN")'
        )
        assert duplicated not in source, f"{module_path} still reads the same env key twice"


class TestBgeM3EndpointResolution:
    """#98/3 — two similarly-named env vars for "where the embedding service lives".

    The rest of the codebase reads ``SURREAL_MEMORY_EMBEDDING_ENDPOINT``; the
    BGE-M3 provider introduced ``SURREAL_MEMORY_EMBEDDING_BASE_URL``. Setting the
    wrong one yielded a silent fallback to a default URL rather than an error.
    """

    @staticmethod
    def _build(monkeypatch: Any, **env: str) -> Any:
        from surreal_memory.engine.embedding.bge_m3_embedding import BGEM3Embedding

        for key in (
            "SURREAL_MEMORY_EMBEDDING_ENDPOINT",
            "SURREAL_MEMORY_EMBEDDING_BASE_URL",
        ):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return BGEM3Embedding(api_key="test-key")

    def test_endpoint_is_honoured(self, monkeypatch: Any) -> None:
        provider = self._build(
            monkeypatch, SURREAL_MEMORY_EMBEDDING_ENDPOINT="http://127.0.0.1:9999"
        )
        assert provider._base_url == "http://127.0.0.1:9999"

    def test_base_url_still_works(self, monkeypatch: Any) -> None:
        """Anyone who configured this provider before the change keeps working."""
        provider = self._build(
            monkeypatch, SURREAL_MEMORY_EMBEDDING_BASE_URL="http://127.0.0.1:8888"
        )
        assert provider._base_url == "http://127.0.0.1:8888"

    def test_endpoint_wins_over_base_url(self, monkeypatch: Any) -> None:
        provider = self._build(
            monkeypatch,
            SURREAL_MEMORY_EMBEDDING_ENDPOINT="http://127.0.0.1:9999",
            SURREAL_MEMORY_EMBEDDING_BASE_URL="http://127.0.0.1:8888",
        )
        assert provider._base_url == "http://127.0.0.1:9999", (
            "ENDPOINT is the name the rest of the codebase uses and must take priority"
        )
