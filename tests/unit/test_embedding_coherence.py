"""An incoherent embedding configuration must be reported, not silently obeyed.

Every provider guesses a dimension for a model it does not recognise — Gemini
assumes 3072, the OpenAI-compatible providers 1536, Ollama 1024. Each guess is
reasonable in isolation and dangerous in combination: point a provider at
another provider's model name and it will confidently produce vectors of the
wrong width, which the HNSW index then rejects on every single write. Nothing
in the stack said a word about it.

Two rules close that gap:

* **A closed catalogue rejects foreign model names.** A hosted API serves a
  fixed set of models; asking it for a local model name cannot work.
* **A known model's dimension must match the configured one**, because that is
  what the vector index is built with.

The hard requirement is the *absence* of false positives: a local
OpenAI-compatible server legitimately serves arbitrary model names, and warning
about a working setup would train everyone to ignore the check.
"""

from __future__ import annotations

import pytest

from surreal_memory.engine.embedding.capability import (
    check_embedding_coherence,
    probe_embedding_capability,
)


class _Config:
    """Nested-config shape accepted by probe_embedding_capability."""

    class _Embedding:
        def __init__(self, enabled: bool, provider: str, model: str, dimension: int) -> None:
            self.enabled = enabled
            self.provider = provider
            self.model = model
            self.dimension = dimension

    def __init__(
        self, provider: str, model: str, dimension: int = 0, *, enabled: bool = True
    ) -> None:
        self.embedding = self._Embedding(enabled, provider, model, dimension)


class TestTheConfigurationThatWentUnnoticed:
    """The exact shape that survived a day of use without a single complaint."""

    def test_a_hosted_provider_pointed_at_a_local_model_is_reported(self) -> None:
        mismatch = check_embedding_coherence("gemini", "bge-m3-FP16", 1024)

        assert mismatch is not None
        assert "gemini" in mismatch.summary
        assert "bge-m3-FP16" in mismatch.summary

    def test_the_report_says_what_to_do(self) -> None:
        mismatch = check_embedding_coherence("gemini", "bge-m3-FP16", 1024)

        assert mismatch is not None
        assert mismatch.fix.strip()


class TestNoFalsePositives:
    """A check that cries wolf on a working setup is worse than no check."""

    @pytest.mark.parametrize(
        ("provider", "model", "dimension"),
        [
            # A local OpenAI-compatible server serving a BGE build under its own
            # file name — arbitrary names are the norm here, not an error.
            ("openai", "bge-m3-FP16", 1024),
            ("openai", "some-quantised-model-Q4_K_M", 768),
            ("openai", "text-embedding-3-small", 1536),
            ("openai", "text-embedding-3-large", 3072),
            ("gemini", "gemini-embedding-001", 3072),
            ("ollama", "bge-m3", 1024),
            ("ollama", "all-minilm", 384),
            ("openrouter", "openai/text-embedding-3-small", 1536),
            # Open-catalogue providers with no dimension table at all.
            ("sentence_transformer", "all-MiniLM-L6-v2", 384),
            ("bge_m3", "bge-m3", 1024),
        ],
    )
    def test_a_coherent_configuration_is_silent(
        self, provider: str, model: str, dimension: int
    ) -> None:
        assert check_embedding_coherence(provider, model, dimension) is None

    def test_dimension_zero_means_auto_and_is_not_a_conflict(self) -> None:
        """0 = derive from the provider, so there is nothing to contradict."""
        assert check_embedding_coherence("openai", "text-embedding-3-small", 0) is None

    @pytest.mark.parametrize("provider", ["auto", "", "nonsense-provider"])
    def test_providers_this_check_cannot_reason_about_are_left_alone(self, provider: str) -> None:
        assert check_embedding_coherence(provider, "whatever", 1024) is None

    def test_an_empty_model_is_left_to_the_provider_default(self) -> None:
        assert check_embedding_coherence("gemini", "", 3072) is None


class TestClosedCatalogue:
    """A hosted API cannot be asked for a model it does not serve."""

    @pytest.mark.parametrize(
        "model",
        ["bge-m3-FP16", "text-embedding-3-small", "all-MiniLM-L6-v2", "nomic-embed-text"],
    )
    def test_a_foreign_model_name_is_reported(self, model: str) -> None:
        assert check_embedding_coherence("gemini", model, 0) is not None

    def test_it_fires_even_when_no_dimension_is_configured(self) -> None:
        """The name alone is enough: the request itself would be rejected."""
        assert check_embedding_coherence("gemini", "bge-m3-FP16", 0) is not None

    def test_a_decommissioned_model_is_reported(self) -> None:
        """text-embedding-004 was removed from the catalogue; it 404s."""
        assert check_embedding_coherence("gemini", "text-embedding-004", 768) is not None


class TestDimensionConflict:
    """The configured dimension is what the vector index is built with."""

    @pytest.mark.parametrize(
        ("provider", "model", "dimension", "expected"),
        [
            ("gemini", "gemini-embedding-001", 1024, 3072),
            ("openai", "text-embedding-3-small", 1024, 1536),
            ("openai", "text-embedding-3-large", 1536, 3072),
            ("ollama", "bge-m3", 768, 1024),
            ("ollama", "all-minilm", 1024, 384),
            ("openrouter", "openai/text-embedding-3-large", 1536, 3072),
        ],
    )
    def test_a_known_model_with_the_wrong_dimension_is_reported(
        self, provider: str, model: str, dimension: int, expected: int
    ) -> None:
        mismatch = check_embedding_coherence(provider, model, dimension)

        assert mismatch is not None
        assert str(expected) in mismatch.summary
        assert str(dimension) in mismatch.summary

    def test_an_unknown_model_is_not_second_guessed(self) -> None:
        """The provider's fallback is a guess, and a guess must not raise an error.

        Only an exact catalogue hit counts as knowing the dimension — otherwise
        every local model name would be reported against a made-up number.
        """
        assert check_embedding_coherence("openai", "a-model-nobody-listed", 1024) is None


class TestProbeSurfacesIt:
    """smem_health and MCP startup read the probe, so it has to carry the news."""

    def test_a_coherent_configuration_reports_no_mismatch(self) -> None:
        result = probe_embedding_capability(_Config("ollama", "bge-m3", 1024))

        assert result.get("mismatch") is None

    def test_an_incoherent_configuration_is_reported_in_the_probe(self) -> None:
        result = probe_embedding_capability(_Config("ollama", "bge-m3", 768))

        assert result.get("mismatch")
        assert "768" in str(result["mismatch"])
        assert "768" in str(result["detail"])

    def test_disabled_embeddings_are_not_examined(self) -> None:
        result = probe_embedding_capability(_Config("gemini", "bge-m3-FP16", 1024, enabled=False))

        assert result.get("mismatch") is None


class TestTheProbeNeverRaises:
    """Its docstring promises it; a missing parent package broke the promise.

    ``find_spec("google.genai")`` imports ``google`` first, so on a machine
    without it the call raises ModuleNotFoundError instead of answering "not
    installed" — and it is `smem_health` and MCP startup that call this.
    """

    def test_a_provider_whose_parent_package_is_absent_is_reported_not_raised(self) -> None:
        result = probe_embedding_capability(_Config("gemini", "gemini-embedding-001", 3072))

        assert result["available"] in (True, False)
        assert isinstance(result["detail"], (str, type(None)))

    def test_an_unimportable_module_name_is_reported_not_raised(self, monkeypatch) -> None:
        from surreal_memory.engine.embedding import capability as capability_mod

        def _explode(_name: str) -> object:
            raise ModuleNotFoundError("No module named 'google'")

        monkeypatch.setattr(capability_mod.importlib.util, "find_spec", _explode)

        result = probe_embedding_capability(_Config("gemini", "gemini-embedding-001", 3072))

        assert result["available"] is False
        assert "not installed" in str(result["detail"])


class TestDoctorReportsIt:
    """`smem doctor` answered "ok" for a configuration that could not work."""

    def _run(self, provider: str, model: str, dimension: int) -> dict:
        from unittest.mock import MagicMock, patch

        from surreal_memory.cli.doctor import _check_embedding_provider

        config = MagicMock()
        config.embedding.enabled = True
        config.embedding.provider = provider
        config.embedding.model = model
        config.embedding.dimension = dimension

        with (
            patch("surreal_memory.unified_config.get_config", return_value=config),
            patch("surreal_memory.cli.doctor.importlib.import_module"),
        ):
            return _check_embedding_provider()

    def test_an_installed_package_with_an_impossible_model_is_not_ok(self) -> None:
        result = self._run("gemini", "bge-m3-FP16", 1024)

        assert result["status"] == "fail"
        assert "bge-m3-FP16" in result["detail"]
        assert result.get("fix", "").strip()

    def test_a_dimension_conflict_is_not_ok(self) -> None:
        result = self._run("openai", "text-embedding-3-small", 1024)

        assert result["status"] == "fail"
        assert "1536" in result["detail"]

    def test_the_working_local_setup_still_passes(self) -> None:
        """The check must not break the configuration it was written to protect."""
        result = self._run("openai", "bge-m3-FP16", 1024)

        assert result["status"] == "ok"
