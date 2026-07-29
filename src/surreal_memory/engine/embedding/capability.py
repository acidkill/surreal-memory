"""Embedding capability probe — turns the silent "embeddings configured but the
provider package is missing" failure into a visible, actionable state.

Used by ``smem_health`` (to report ``embedding`` availability) and at MCP
startup (to log a loud, actionable warning). The probe is cheap: it checks
package availability via ``importlib.util.find_spec`` and never imports heavy
models or makes API calls.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# provider name -> (import module to check, pip extra that provides it)
_PROVIDER_IMPORT: dict[str, tuple[str, str]] = {
    "gemini": ("google.genai", "embeddings-gemini"),
    "openai": ("openai", "embeddings-openai"),
    "openrouter": ("openai", "embeddings-openrouter"),
    "sentence_transformer": ("sentence_transformers", "embeddings"),
    # HTTP provider (self-hosted BGE-M3): only needs httpx, shipped in the
    # ``server`` extra. Without this entry the probe reports the configured
    # bge_m3 provider as "unknown" and logs a spurious degraded-keyword-mode
    # warning even though recall/remember embed fine via the factory.
    "bge_m3": ("httpx", "server"),
}


# Providers whose model catalogue is fixed by the vendor: a name outside it is
# not "unlisted", it is a request the API will refuse. Providers backed by an
# OpenAI-compatible server are deliberately absent — a local server serves
# whatever files it was pointed at, so arbitrary names there are normal.
_CLOSED_CATALOGUE: frozenset[str] = frozenset({"gemini"})

# Dimension a provider assumes when it meets a model it does not recognise.
# These are the values the providers themselves fall back to.
_FALLBACK_DIMENSION: dict[str, int] = {"gemini": 3072, "openai": 1536, "openrouter": 1536}


@dataclass(frozen=True)
class EmbeddingMismatch:
    """A configuration that cannot work, with the reason and the way out."""

    summary: str
    fix: str

    def __str__(self) -> str:
        return self.summary


def _find_spec_safely(module_name: str) -> object | None:
    """``importlib.util.find_spec`` that answers instead of raising.

    For a dotted name it imports the parent package first, so a missing parent
    (``google`` for ``google.genai``) raises ModuleNotFoundError rather than
    returning None — turning "is this installed?" into a crash, in the one
    function whose whole job is to report that answer calmly.
    """
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None


def _catalogue(provider: str) -> dict[str, int] | None:
    """The provider's model -> dimension table, or None if it has no fixed one."""
    try:
        if provider == "gemini":
            from surreal_memory.engine.embedding.gemini_embedding import _MODEL_DIMENSIONS

            return dict(_MODEL_DIMENSIONS)
        if provider == "openai":
            from surreal_memory.engine.embedding.openai_embedding import _MODEL_DIMENSIONS

            return dict(_MODEL_DIMENSIONS)
        if provider == "openrouter":
            from surreal_memory.engine.embedding.openrouter_embedding import _MODEL_DIMENSIONS

            return dict(_MODEL_DIMENSIONS)
        if provider == "ollama":
            from surreal_memory.engine.embedding.ollama_embedding import _MODEL_DIMENSIONS

            return dict(_MODEL_DIMENSIONS)
    except Exception:
        return None
    return None


def _known_dimension(provider: str, model: str) -> int | None:
    """The dimension this model *is* known to have — never a guess.

    Distinct from :func:`_dimension_for`, which answers "what will the provider
    probably do". Validation may only act on knowledge: calling a config broken
    because a fallback number disagreed would flag every local model name.
    """
    catalogue = _catalogue(provider)
    if not catalogue or not model:
        return None
    return catalogue.get(model)


def _dimension_for(provider: str, model: str) -> int | None:
    """Best-effort dimension lookup without loading models or calling APIs."""
    if provider not in _FALLBACK_DIMENSION:
        return None
    known = _known_dimension(provider, model)
    return known if known is not None else _FALLBACK_DIMENSION[provider]


def check_embedding_coherence(
    provider: str, model: str, dimension: int
) -> EmbeddingMismatch | None:
    """Report a provider/model/dimension combination that cannot work.

    Returns None when the configuration is coherent *or* when there is not
    enough knowledge to judge it — an unknown model on an open-catalogue
    provider is ordinary, not suspect.

    Two things are caught:

    * a model outside a hosted provider's fixed catalogue, which that API will
      refuse whatever the rest of the configuration says;
    * a catalogued model whose real dimension contradicts the configured one —
      and the configured dimension is what the vector index is built from, so
      every write would be rejected for a width the config itself asked for.

    Never raises: a diagnostic that can crash is worse than no diagnostic.
    """
    provider = (provider or "").strip()
    model = (model or "").strip()
    if not provider or not model:
        return None

    if provider in _CLOSED_CATALOGUE:
        catalogue = _catalogue(provider)
        if catalogue and model not in catalogue:
            served = ", ".join(sorted(catalogue)) or "(none)"
            return EmbeddingMismatch(
                summary=(
                    f"provider {provider!r} does not serve model {model!r} (it serves: {served})"
                ),
                fix=(
                    f"Set embedding.model to one of: {served} — or, if {model!r} is served by a"
                    " local OpenAI-compatible endpoint, set embedding.provider to 'openai' and"
                    " point SURREAL_MEMORY_EMBEDDING_ENDPOINT at it."
                ),
            )

    known = _known_dimension(provider, model)
    if known is not None and dimension and int(dimension) != known:
        return EmbeddingMismatch(
            summary=(
                f"model {model!r} produces {known}-dimensional vectors but"
                f" embedding.dimension is {int(dimension)}"
            ),
            fix=(
                f"Set embedding.dimension to {known} (or 0 to derive it). A vector index built"
                f" for {int(dimension)} dimensions rejects every write from this model."
            ),
        )
    return None


def probe_embedding_capability(config: Any) -> dict[str, Any]:
    """Report whether the configured embedding provider is usable.

    Returns a dict with: ``enabled``, ``provider``, ``model``, ``available``
    (bool) and ``detail`` (a human-readable note, including the exact install
    command when a package is missing). Never raises.
    """
    # Accept either a flat BrainConfig (embedding_enabled/provider/model) or a
    # nested unified config (config.embedding.enabled/provider/model).
    nested = getattr(config, "embedding", None)
    enabled = getattr(config, "embedding_enabled", None)
    provider = getattr(config, "embedding_provider", None)
    model = getattr(config, "embedding_model", None)
    if nested is not None and not isinstance(nested, (str, bool)):
        if enabled is None:
            enabled = getattr(nested, "enabled", False)
        if provider is None:
            provider = getattr(nested, "provider", "")
        if model is None:
            model = getattr(nested, "model", "")
    dimension_cfg = getattr(config, "embedding_dimension", None)
    if dimension_cfg is None and nested is not None and not isinstance(nested, (str, bool)):
        dimension_cfg = getattr(nested, "dimension", 0)
    enabled = bool(enabled)
    provider = (provider or "").strip()
    model = model or ""
    result: dict[str, Any] = {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "available": False,
        "dimension": None,
        "detail": None,
        # Set when the provider/model/dimension triple cannot work. Independent
        # of ``available``: the package can be installed and importable while the
        # configuration it is handed is still impossible.
        "mismatch": None,
    }

    if not enabled:
        result["detail"] = "embeddings disabled"
        return result

    try:
        mismatch = check_embedding_coherence(provider, model, int(dimension_cfg or 0))
    except Exception:  # a diagnostic must never be the thing that breaks
        mismatch = None
    if mismatch is not None:
        result["mismatch"] = mismatch.summary

    def _detail(note: str | None) -> str | None:
        """Keep the mismatch visible: later branches also write ``detail``."""
        if mismatch is None:
            return note
        full = f"{mismatch.summary} — {mismatch.fix}"
        return f"{note} | {full}" if note else full

    if provider == "ollama":
        # Local server — availability depends on the daemon, not a Python package.
        result["available"] = True
        result["detail"] = _detail("ollama (local server — ensure it is running)")
        return result

    if provider == "auto":
        # Provider is detected at runtime from whatever is installed/keyed.
        result["available"] = True
        result["detail"] = _detail("auto-detect at runtime")
        return result

    entry = _PROVIDER_IMPORT.get(provider)
    if entry is None:
        result["detail"] = _detail(f"unknown embedding provider {provider!r}")
        return result

    module_name, extra = entry
    if _find_spec_safely(module_name) is None:
        result["detail"] = _detail(
            f"{provider} provider configured but '{module_name}' is not installed — "
            f"install it with: pip install 'surreal-memory[{extra}]'"
        )
        return result

    result["available"] = True
    result["dimension"] = _dimension_for(provider, model)
    result["detail"] = _detail(result["detail"])
    return result


def warn_if_embedding_unavailable(config: Any) -> None:
    """Log a loud, actionable warning at startup if embeddings are configured
    but the provider package is missing. Recall/remember still fail-soft to
    keyword mode — this just makes the cause visible instead of silent."""
    info = probe_embedding_capability(config)
    if info["enabled"] and not info["available"]:
        logger.warning(
            "Embedding provider unavailable — running in degraded keyword mode. %s",
            info["detail"],
        )
