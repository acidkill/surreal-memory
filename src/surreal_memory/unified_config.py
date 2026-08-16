"""Unified configuration for Surreal-Memory across all tools.

This module provides a single configuration system that works across:
- CLI (smem command)
- MCP server (Claude Code, Cursor, AntiGravity)
- REST API server
- Any future integrations

Configuration is stored in ~/.surrealmemory/config.toml
Brain data is stored in ~/.surrealmemory/brains/<name>.db (SQLite)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Valid brain name: alphanumeric, hyphens, underscores, dots (no path separators)
_BRAIN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# Valid sync identifier: alphanumeric, hyphens, underscores, dots, @ (for emails)
_SYNC_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.@]*$")
_SYNC_ID_MAX_LEN = 128

# Valid TOML string value: alphanumeric, hyphens, underscores, dots, slashes, spaces
_TOML_SAFE_STRING = re.compile(r"^[a-zA-Z0-9_\-\./ ]*$")
_TOML_STR_MAX_LEN = 128
# URL charset (RFC 3986 gen-/sub-delims + unreserved + %), minus quotes,
# backslash, apostrophe and whitespace — so it can't break out of a TOML
# double-quoted basic string. Used for endpoint URLs which need ':' '?' '&' etc.
_TOML_SAFE_URL = re.compile(r"^[a-zA-Z0-9\-._~:/?#\[\]@!$&()*+,;=%]*$")
_TOML_URL_MAX_LEN = 256


def get_surrealmemory_dir() -> Path:
    """Get Surreal-Memory data directory.

    Priority:
    1. SURREAL_MEMORY_DIR environment variable
    2. ~/.surrealmemory/
    """
    env_dir = os.environ.get("SURREAL_MEMORY_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return Path.home() / ".surrealmemory"


def get_default_brain() -> str:
    """Get default brain name.

    Priority:
    1. SURREAL_MEMORY_BRAIN environment variable (validated)
    2. "default"
    """
    name = os.environ.get("SURREAL_MEMORY_BRAIN", "default")
    if not _BRAIN_NAME_PATTERN.match(name):
        return "default"
    return name


@dataclass
class AutoConfig:
    """Auto-capture configuration for MCP server."""

    enabled: bool = True
    capture_decisions: bool = True
    capture_errors: bool = True
    capture_todos: bool = True
    capture_facts: bool = True
    capture_insights: bool = True
    capture_preferences: bool = True
    min_confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "capture_decisions": self.capture_decisions,
            "capture_errors": self.capture_errors,
            "capture_todos": self.capture_todos,
            "capture_facts": self.capture_facts,
            "capture_insights": self.capture_insights,
            "capture_preferences": self.capture_preferences,
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoConfig:
        return cls(
            enabled=data.get("enabled", True),
            capture_decisions=data.get("capture_decisions", True),
            capture_errors=data.get("capture_errors", True),
            capture_todos=data.get("capture_todos", True),
            capture_facts=data.get("capture_facts", True),
            capture_insights=data.get("capture_insights", True),
            capture_preferences=data.get("capture_preferences", True),
            min_confidence=data.get("min_confidence", 0.7),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    """Settings for embedding-based semantic recall."""

    enabled: bool = False
    provider: str = "sentence_transformer"
    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    # Explicit embedding vector dimension. 0 = auto (derive from provider/model).
    # Set this when the provider/model is not in a built-in dimension table
    # (e.g. a local OpenAI-compatible server serving bge-m3 → 1024); it drives
    # the SurrealDB HNSW index dimension so the index always matches the vectors.
    dimension: int = 0
    # Base URL of an OpenAI-compatible embedding server (e.g. llamastash serving
    # bge-m3 at "http://127.0.0.1:11435/v1"). Mirrors RerankerConfig.endpoint so the
    # two halves of a local embed+rerank pair are configured the same way; before
    # this existed the endpoint could only be given as an env var, which no
    # config-driven consumer could see. Config wins, env is the fallback.
    endpoint: str = ""

    _VALID_PROVIDERS: ClassVar[tuple[str, ...]] = (
        "sentence_transformer",
        "openai",
        "openrouter",
        "gemini",
        "ollama",
        "bge_m3",
        "auto",
        "",
    )

    def __post_init__(self) -> None:
        if self.provider not in self._VALID_PROVIDERS:
            import logging

            logging.getLogger(__name__).warning(
                "Invalid embedding provider %r, falling back to disabled. Valid: %s",
                self.provider,
                self._VALID_PROVIDERS,
            )
            object.__setattr__(self, "provider", "")
            object.__setattr__(self, "enabled", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "similarity_threshold": self.similarity_threshold,
            "dimension": self.dimension,
            "endpoint": self.endpoint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSettings:
        return cls(
            enabled=bool(data.get("enabled", False)),
            provider=str(data.get("provider", "sentence_transformer")),
            model=str(data.get("model", "all-MiniLM-L6-v2")),
            similarity_threshold=float(data.get("similarity_threshold", 0.7)),
            dimension=int(data.get("dimension", 0) or 0),
            endpoint=str(data.get("endpoint", "") or "").strip(),
        )

    def resolved_endpoint(self) -> str:
        """Endpoint in effect: the configured value, else the env override.

        Same precedence as the reranker (config first, env fallback), so a
        machine-wide export cannot silently override an explicit per-brain
        setting while still working for setups that only export the env var.
        """
        import os

        return (
            self.endpoint.strip() or os.environ.get("SURREAL_MEMORY_EMBEDDING_ENDPOINT", "").strip()
        )


def _env_truthy(value: str | None) -> bool | None:
    """Parse a truthy env var. Returns None when the var is unset/empty.

    Accepts ``1/true/yes/on`` (case-insensitive) as True; everything else as
    False. Returning ``None`` lets callers distinguish "not set" from "false".
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "":
        return None
    return normalized in ("1", "true", "yes", "on")


def _load_embedding_settings(data: dict[str, Any]) -> EmbeddingSettings:
    """Build EmbeddingSettings from config.toml, letting env vars override.

    Env precedence (env wins over config.toml, config.toml wins over defaults):
        SURREAL_MEMORY_EMBEDDING_ENABLED              -> enabled (truthy parse)
        SURREAL_MEMORY_EMBEDDING_PROVIDER             -> provider
        SURREAL_MEMORY_EMBEDDING_MODEL                -> model
        SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD -> similarity_threshold
        SURREAL_MEMORY_EMBEDDING_DIMENSION            -> dimension

    Only env vars that are actually set (non-empty) override the file values.

    ``endpoint`` is deliberately NOT in that list. It follows the reranker's
    precedence instead — config first, env as the fallback — resolved at read
    time by :meth:`EmbeddingSettings.resolved_endpoint`. Overriding it here too
    would make the config value unreachable whenever the env var is exported
    machine-wide, which is exactly the case an explicit per-brain endpoint
    exists to serve.
    """
    base = EmbeddingSettings.from_dict(data)

    enabled = base.enabled
    env_enabled = _env_truthy(os.environ.get("SURREAL_MEMORY_EMBEDDING_ENABLED"))
    if env_enabled is not None:
        enabled = env_enabled

    provider = base.provider
    env_provider = os.environ.get("SURREAL_MEMORY_EMBEDDING_PROVIDER")
    if env_provider is not None and env_provider.strip():
        provider = env_provider.strip()

    model = base.model
    env_model = os.environ.get("SURREAL_MEMORY_EMBEDDING_MODEL")
    if env_model is not None and env_model.strip():
        model = env_model.strip()

    similarity_threshold = base.similarity_threshold
    env_threshold = os.environ.get("SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD")
    if env_threshold is not None and env_threshold.strip():
        try:
            similarity_threshold = float(env_threshold)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Invalid SURREAL_MEMORY_EMBEDDING_SIMILARITY_THRESHOLD=%r — ignoring",
                env_threshold,
            )

    dimension = base.dimension
    env_dimension = os.environ.get("SURREAL_MEMORY_EMBEDDING_DIMENSION")
    if env_dimension is not None and env_dimension.strip():
        try:
            dimension = int(env_dimension)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Invalid SURREAL_MEMORY_EMBEDDING_DIMENSION=%r — ignoring",
                env_dimension,
            )

    return EmbeddingSettings(
        enabled=enabled,
        provider=provider,
        model=model,
        similarity_threshold=similarity_threshold,
        dimension=dimension,
        endpoint=base.endpoint,
    )


def _load_sync_settings(data: dict[str, Any]) -> SyncConfig:
    """Build SyncConfig from config.toml, letting env vars override.

    Env precedence (env wins over config.toml, config.toml wins over defaults):
        SURREAL_MEMORY_HUB_URL       -> hub_url
        SURREAL_MEMORY_API_KEY       -> api_key
        SURREAL_MEMORY_SYNC_ENABLED  -> enabled (truthy parse)
        SURREAL_MEMORY_SYNC_AUTO     -> auto_sync (truthy parse)

    This makes a sync hub configured purely via the environment (docker-compose
    or the MCP client env) visible to the dashboard/Overview without first
    writing a ``[sync]`` block to config.toml — mirroring the embedding/storage
    env-override layers. Values are re-validated through ``SyncConfig.from_dict``
    so env-provided hub_url/api_key get the same format sanitisation.
    """
    base = SyncConfig.from_dict(data)

    hub_url = base.hub_url
    env_hub = os.environ.get("SURREAL_MEMORY_HUB_URL")
    if env_hub is not None and env_hub.strip():
        hub_url = env_hub.strip()

    api_key = base.api_key
    env_key = os.environ.get("SURREAL_MEMORY_API_KEY")
    if env_key is not None and env_key.strip():
        api_key = env_key.strip()

    enabled = base.enabled
    env_enabled = _env_truthy(os.environ.get("SURREAL_MEMORY_SYNC_ENABLED"))
    if env_enabled is not None:
        enabled = env_enabled

    auto_sync = base.auto_sync
    env_auto = _env_truthy(os.environ.get("SURREAL_MEMORY_SYNC_AUTO"))
    if env_auto is not None:
        auto_sync = env_auto

    return SyncConfig.from_dict(
        {
            "enabled": enabled,
            "hub_url": hub_url,
            "api_key": api_key,
            "auto_sync": auto_sync,
            "sync_interval_seconds": base.sync_interval_seconds,
            "conflict_strategy": base.conflict_strategy,
        }
    )


@dataclass
class BrainSettings:
    """Settings for brain behavior.

    The explicit fields are the historical keys exposed via ``[brain]`` in
    ``config.toml``. ``extras`` captures any additional ``[brain]`` keys that map
    onto ``core.brain.BrainConfig`` fields added after this class was first
    defined (e.g. ``tag_match_boost``, ``rrf_k``, …), so new BrainConfig knobs
    become config-toml-controllable without a parallel field for each one
    (issue #168). Unknown keys are filtered against BrainConfig's field set, so
    typos in ``config.toml`` do not crash brain creation.
    """

    decay_rate: float = 0.1
    reinforcement_delta: float = 0.05
    # See core.brain.BrainConfig.reinforcement_neuron_limit for why this exists
    # and why 15, not a larger jump (raised from a hardcoded, unconfigurable
    # 10; measured against a live SurrealDB, not just SQLite).
    reinforcement_neuron_limit: int = 15
    activation_threshold: float = 0.2
    max_spread_hops: int = 4
    max_context_tokens: int = 1500
    freshness_weight: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    _EXPLICIT_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "decay_rate",
            "reinforcement_delta",
            "reinforcement_neuron_limit",
            "activation_threshold",
            "max_spread_hops",
            "max_context_tokens",
            "freshness_weight",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decay_rate": self.decay_rate,
            "reinforcement_delta": self.reinforcement_delta,
            "reinforcement_neuron_limit": self.reinforcement_neuron_limit,
            "activation_threshold": self.activation_threshold,
            "max_spread_hops": self.max_spread_hops,
            "max_context_tokens": self.max_context_tokens,
            "freshness_weight": self.freshness_weight,
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BrainSettings:
        extras = {k: v for k, v in data.items() if k not in cls._EXPLICIT_KEYS}
        return cls(
            decay_rate=data.get("decay_rate", 0.1),
            reinforcement_delta=data.get("reinforcement_delta", 0.05),
            reinforcement_neuron_limit=data.get("reinforcement_neuron_limit", 15),
            activation_threshold=data.get("activation_threshold", 0.2),
            max_spread_hops=data.get("max_spread_hops", 4),
            max_context_tokens=data.get("max_context_tokens", 1500),
            freshness_weight=data.get("freshness_weight", 0.0),
            extras=extras,
        )

    def to_brain_config_kwargs(
        self,
        embedding: EmbeddingSettings | None = None,
        reranker: Any = None,
    ) -> dict[str, Any]:
        """Build kwargs for ``core.brain.BrainConfig`` from this settings instance.

        Combines the explicit BrainSettings fields, embedding-derived fields,
        reranker-derived fields (from ``config.toml [reranker]``), and any
        ``extras`` keys that match a real BrainConfig field name. Unknown extras
        are dropped so ``BrainConfig(**kwargs)`` is safe.
        """
        from surreal_memory.core.brain import BrainConfig

        valid_fields = {f.name for f in dataclasses.fields(BrainConfig)}
        kwargs: dict[str, Any] = {
            "decay_rate": self.decay_rate,
            "reinforcement_delta": self.reinforcement_delta,
            "reinforcement_neuron_limit": self.reinforcement_neuron_limit,
            "activation_threshold": self.activation_threshold,
            "max_spread_hops": self.max_spread_hops,
            "max_context_tokens": self.max_context_tokens,
            "freshness_weight": self.freshness_weight,
        }
        if embedding is not None:
            kwargs.update(
                {
                    "embedding_enabled": embedding.enabled,
                    "embedding_provider": embedding.provider,
                    "embedding_model": embedding.model,
                    "embedding_similarity_threshold": embedding.similarity_threshold,
                }
            )
        if reranker is not None:
            kwargs.update(reranker_brain_config_overrides(reranker))
        for key, value in self.extras.items():
            if key in valid_fields and key not in kwargs:
                kwargs[key] = value
        return kwargs

    def runtime_overrides(self) -> dict[str, Any]:
        """Return only ``extras`` keys that match real BrainConfig fields.

        Used by storage init to layer ``config.toml [brain]`` over a
        previously-stored brain config on upgrade (issue #168). The explicit
        fields are excluded because legacy brains may have customized them.
        """
        from surreal_memory.core.brain import BrainConfig

        valid_fields = {f.name for f in dataclasses.fields(BrainConfig)}
        return {key: value for key, value in self.extras.items() if key in valid_fields}


@dataclass
class EternalConfig:
    """Eternal context auto-save configuration."""

    enabled: bool = True
    notifications: bool = True
    auto_save_interval: int = 15
    context_warning_threshold: float = 0.8
    max_context_tokens: int = 128_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "notifications": self.notifications,
            "auto_save_interval": self.auto_save_interval,
            "context_warning_threshold": self.context_warning_threshold,
            "max_context_tokens": self.max_context_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EternalConfig:
        return cls(
            enabled=data.get("enabled", True),
            notifications=data.get("notifications", True),
            auto_save_interval=data.get("auto_save_interval", 15),
            context_warning_threshold=data.get("context_warning_threshold", 0.8),
            max_context_tokens=data.get("max_context_tokens", 128_000),
        )


@dataclass(frozen=True)
class WriteGateConfig:
    """Write-gate configuration for memory quality enforcement.

    When enabled, memories that fail quality checks are rejected before storage.
    This prevents low-quality content from degrading brain purity.

    Addresses GitHub Issue #95: write-gate to improve brain purity.
    """

    enabled: bool = False  # opt-in, backward compat (True == enforce)
    mode: str = "off"  # off | shadow | enforce. Overrides `enabled` when not "off".
    # Per-intent override for auto-captures (intent=auto ONLY; summaries and
    # turns keep `mode`). "" = inherit `mode`. Lets junk auto-captures be
    # ENFORCED while interactive writes stay in shadow — a global enforce is
    # known to false-reject real turn/summary content.
    auto_capture_mode: str = ""  # "" (inherit) | off | shadow | enforce
    min_length: int = 30  # reject content shorter than this
    min_quality_score: int = 3  # reject score below this (0-10 scale)
    auto_capture_min_score: int = 5  # stricter threshold for passive captures
    max_content_length: int = 2000  # reject wall-of-text above this
    reject_generic_filler: bool = True  # reject "done", "ok", "completed" etc.

    @property
    def effective_mode(self) -> str:
        """Resolve the operating mode. `mode` wins; otherwise fall back to the
        legacy `enabled` bool (True -> enforce, False -> off)."""
        m = (self.mode or "off").strip().lower()
        if m in ("shadow", "enforce"):
            return m
        if m == "off" and self.enabled:
            return "enforce"
        return "off"

    @property
    def effective_auto_mode(self) -> str:
        """Resolve the mode for auto-captures (intent=auto). A valid
        `auto_capture_mode` wins; otherwise inherit `effective_mode`."""
        m = (self.auto_capture_mode or "").strip().lower()
        if m in ("off", "shadow", "enforce"):
            return m
        return self.effective_mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "auto_capture_mode": self.auto_capture_mode,
            "min_length": self.min_length,
            "min_quality_score": self.min_quality_score,
            "auto_capture_min_score": self.auto_capture_min_score,
            "max_content_length": self.max_content_length,
            "reject_generic_filler": self.reject_generic_filler,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WriteGateConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            mode=str(data.get("mode", "off")),
            auto_capture_mode=str(data.get("auto_capture_mode", "")),
            min_length=int(data.get("min_length", 30)),
            min_quality_score=int(data.get("min_quality_score", 3)),
            auto_capture_min_score=int(data.get("auto_capture_min_score", 5)),
            max_content_length=int(data.get("max_content_length", 2000)),
            reject_generic_filler=bool(data.get("reject_generic_filler", True)),
        )


@dataclass(frozen=True)
class TraceConfig:
    """Retrieval-trace telemetry configuration (schema v9, opt-in).

    When enabled, a compact RetrievalTrace is persisted (fire-and-forget) for a
    sampled fraction of recalls. Neutral defaults keep tracing off entirely, so
    default recall behaviour and latency are unchanged.
    """

    enabled: bool = False  # opt-in, no traces persisted by default
    sample_rate: float = 1.0  # fraction of recalls to trace when enabled
    retention_days: int = 30  # prune traces older than this
    max_traces: int = 5000  # cap total stored traces (delete-oldest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sample_rate": self.sample_rate,
            "retention_days": self.retention_days,
            "max_traces": self.max_traces,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            sample_rate=float(data.get("sample_rate", 1.0)),
            retention_days=int(data.get("retention_days", 30)),
            max_traces=int(data.get("max_traces", 5000)),
        )


@dataclass(frozen=True)
class MaintenanceConfig:
    """Proactive brain maintenance configuration.

    Controls the health pulse system that piggybacks on remember/recall
    operations to detect brain degradation and surface maintenance hints.
    """

    enabled: bool = True
    check_interval: int = 25
    fiber_warn_threshold: int = 500
    neuron_warn_threshold: int = 2000
    synapse_warn_threshold: int = 5000
    orphan_ratio_threshold: float = 0.25
    expired_memory_warn_threshold: int = 10
    stale_fiber_ratio_threshold: float = 0.3
    stale_fiber_days: int = 90
    consolidation_ratio_threshold: float = 0.1
    auto_consolidate: bool = True
    auto_consolidate_strategies: tuple[str, ...] = ("prune", "merge", "mature", "infer")
    consolidate_cooldown_minutes: int = 30
    dream_cooldown_hours: int = 24
    expiry_cleanup_enabled: bool = True
    expiry_cleanup_interval_hours: int = 12
    expiry_cleanup_max_per_run: int = 100
    scheduled_consolidation_enabled: bool = True
    scheduled_consolidation_interval_hours: int = 24
    scheduled_consolidation_strategies: tuple[str, ...] = ("prune", "merge", "enrich")
    version_check_enabled: bool = True
    version_check_interval_hours: int = 24
    # Auto-decay in serve daemon
    decay_enabled: bool = True
    decay_interval_hours: int = 12
    # Scheduled re-index
    reindex_enabled: bool = False
    reindex_paths: tuple[str, ...] = ()
    reindex_interval_hours: int = 168  # weekly
    reindex_extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".rst",
        ".html",
        ".css",
    )
    # Notifications (webhook + health alerts)
    notifications_enabled: bool = False
    notifications_webhook_url: str = ""
    notifications_health_threshold: str = "D"  # alert at D or F
    notifications_daily_summary: bool = False
    notifications_zero_activity_alert: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "check_interval": self.check_interval,
            "fiber_warn_threshold": self.fiber_warn_threshold,
            "neuron_warn_threshold": self.neuron_warn_threshold,
            "synapse_warn_threshold": self.synapse_warn_threshold,
            "orphan_ratio_threshold": self.orphan_ratio_threshold,
            "expired_memory_warn_threshold": self.expired_memory_warn_threshold,
            "stale_fiber_ratio_threshold": self.stale_fiber_ratio_threshold,
            "stale_fiber_days": self.stale_fiber_days,
            "consolidation_ratio_threshold": self.consolidation_ratio_threshold,
            "auto_consolidate": self.auto_consolidate,
            "auto_consolidate_strategies": list(self.auto_consolidate_strategies),
            "consolidate_cooldown_minutes": self.consolidate_cooldown_minutes,
            "dream_cooldown_hours": self.dream_cooldown_hours,
            "expiry_cleanup_enabled": self.expiry_cleanup_enabled,
            "expiry_cleanup_interval_hours": self.expiry_cleanup_interval_hours,
            "expiry_cleanup_max_per_run": self.expiry_cleanup_max_per_run,
            "scheduled_consolidation_enabled": self.scheduled_consolidation_enabled,
            "scheduled_consolidation_interval_hours": self.scheduled_consolidation_interval_hours,
            "scheduled_consolidation_strategies": list(self.scheduled_consolidation_strategies),
            "version_check_enabled": self.version_check_enabled,
            "version_check_interval_hours": self.version_check_interval_hours,
            "decay_enabled": self.decay_enabled,
            "decay_interval_hours": self.decay_interval_hours,
            "reindex_enabled": self.reindex_enabled,
            "reindex_paths": list(self.reindex_paths),
            "reindex_interval_hours": self.reindex_interval_hours,
            "reindex_extensions": list(self.reindex_extensions),
            "notifications_enabled": self.notifications_enabled,
            "notifications_webhook_url": self.notifications_webhook_url,
            "notifications_health_threshold": self.notifications_health_threshold,
            "notifications_daily_summary": self.notifications_daily_summary,
            "notifications_zero_activity_alert": self.notifications_zero_activity_alert,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaintenanceConfig:
        strategies = data.get("auto_consolidate_strategies", ("prune", "merge", "mature", "infer"))
        if isinstance(strategies, list):
            strategies = tuple(strategies)
        sched_strategies = data.get(
            "scheduled_consolidation_strategies", ("prune", "merge", "enrich")
        )
        if isinstance(sched_strategies, list):
            sched_strategies = tuple(sched_strategies)
        return cls(
            enabled=data.get("enabled", True),
            check_interval=data.get("check_interval", 25),
            fiber_warn_threshold=data.get("fiber_warn_threshold", 500),
            neuron_warn_threshold=data.get("neuron_warn_threshold", 2000),
            synapse_warn_threshold=data.get("synapse_warn_threshold", 5000),
            orphan_ratio_threshold=data.get("orphan_ratio_threshold", 0.25),
            expired_memory_warn_threshold=data.get("expired_memory_warn_threshold", 10),
            stale_fiber_ratio_threshold=data.get("stale_fiber_ratio_threshold", 0.3),
            stale_fiber_days=data.get("stale_fiber_days", 90),
            consolidation_ratio_threshold=data.get("consolidation_ratio_threshold", 0.1),
            auto_consolidate=data.get("auto_consolidate", True),
            auto_consolidate_strategies=strategies,
            consolidate_cooldown_minutes=data.get("consolidate_cooldown_minutes", 30),
            dream_cooldown_hours=data.get("dream_cooldown_hours", 24),
            expiry_cleanup_enabled=data.get("expiry_cleanup_enabled", True),
            expiry_cleanup_interval_hours=data.get("expiry_cleanup_interval_hours", 12),
            expiry_cleanup_max_per_run=data.get("expiry_cleanup_max_per_run", 100),
            scheduled_consolidation_enabled=data.get("scheduled_consolidation_enabled", True),
            scheduled_consolidation_interval_hours=data.get(
                "scheduled_consolidation_interval_hours", 24
            ),
            scheduled_consolidation_strategies=sched_strategies,
            version_check_enabled=data.get("version_check_enabled", True),
            version_check_interval_hours=data.get("version_check_interval_hours", 24),
            decay_enabled=data.get("decay_enabled", True),
            decay_interval_hours=data.get("decay_interval_hours", 12),
            reindex_enabled=data.get("reindex_enabled", False),
            reindex_paths=tuple(data.get("reindex_paths", ())),
            reindex_interval_hours=data.get("reindex_interval_hours", 168),
            reindex_extensions=tuple(
                data.get(
                    "reindex_extensions",
                    (
                        ".md",
                        ".txt",
                        ".py",
                        ".js",
                        ".ts",
                        ".json",
                        ".yaml",
                        ".yml",
                        ".toml",
                        ".rst",
                        ".html",
                        ".css",
                    ),
                )
            ),
            notifications_enabled=data.get("notifications_enabled", False),
            notifications_webhook_url=data.get("notifications_webhook_url", ""),
            notifications_health_threshold=data.get("notifications_health_threshold", "D"),
            notifications_daily_summary=data.get("notifications_daily_summary", False),
            notifications_zero_activity_alert=data.get("notifications_zero_activity_alert", True),
        )


_VALID_TOOL_TIERS = frozenset({"minimal", "standard", "full"})


@dataclass(frozen=True)
class ToolTierConfig:
    """MCP tool tier configuration.

    Controls which tools are exposed via tools/list to reduce token overhead.
    Hidden tools remain callable via dispatch — only schema exposure changes.
    """

    tier: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return {"tier": self.tier}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolTierConfig:
        raw = str(data.get("tier", "full")).lower().strip()
        if raw not in _VALID_TOOL_TIERS:
            raw = "full"
        return cls(tier=raw)


@dataclass(frozen=True)
class SafetyConfig:
    """Safety and auto-redaction configuration.

    Controls automatic redaction of high-severity sensitive content
    instead of blocking the entire operation.
    """

    auto_redact_min_severity: int = 3  # Auto-redact severity 3+ by default

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_redact_min_severity": self.auto_redact_min_severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        severity = data.get("auto_redact_min_severity", 3)
        try:
            severity = max(1, min(int(severity), 3))
        except (ValueError, TypeError):
            severity = 3
        return cls(auto_redact_min_severity=severity)


@dataclass(frozen=True)
class EncryptionConfig:
    """Encryption configuration for sensitive memory content.

    When enabled, neuron content detected as sensitive (or explicitly flagged)
    is encrypted using Fernet symmetric encryption with per-brain keys.
    """

    enabled: bool = True
    auto_encrypt_sensitive: bool = True
    keys_dir: str = ""  # empty = use {data_dir}/keys/

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "auto_encrypt_sensitive": self.auto_encrypt_sensitive,
            "keys_dir": self.keys_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncryptionConfig:
        keys_dir = str(data.get("keys_dir", ""))[:256]
        return cls(
            enabled=bool(data.get("enabled", True)),
            auto_encrypt_sensitive=bool(data.get("auto_encrypt_sensitive", True)),
            keys_dir=keys_dir,
        )


@dataclass(frozen=True)
class SyncConfig:
    """Multi-device sync configuration."""

    enabled: bool = False
    hub_url: str = ""
    api_key: str = ""
    auto_sync: bool = False
    sync_interval_seconds: int = 300
    conflict_strategy: str = "prefer_recent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hub_url": self.hub_url,
            "api_key": self.api_key,
            "auto_sync": self.auto_sync,
            "sync_interval_seconds": self.sync_interval_seconds,
            "conflict_strategy": self.conflict_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SyncConfig:
        strategy = str(data.get("conflict_strategy", "prefer_recent"))
        valid_strategies = {"prefer_recent", "prefer_local", "prefer_remote", "prefer_stronger"}
        if strategy not in valid_strategies:
            strategy = "prefer_recent"
        try:
            interval = max(10, min(int(data.get("sync_interval_seconds", 300)), 86400))
        except (ValueError, TypeError):
            interval = 300
        hub_url = str(data.get("hub_url", ""))
        # Sanitize hub_url - only allow http/https URLs
        if hub_url and not hub_url.startswith(("http://", "https://")):
            hub_url = ""
        # Truncate URL to reasonable length
        hub_url = hub_url[:256]
        api_key = str(data.get("api_key", ""))
        # Validate api_key format: must start with nmk_ or be empty
        if api_key and not api_key.startswith("nmk_"):
            api_key = ""
        return cls(
            enabled=bool(data.get("enabled", False)),
            hub_url=hub_url,
            api_key=api_key,
            auto_sync=bool(data.get("auto_sync", False)),
            sync_interval_seconds=interval,
            conflict_strategy=strategy,
        )


@dataclass(frozen=True)
class DedupSettings:
    """LLM-powered deduplication settings.

    Controls the 3-tier dedup pipeline: SimHash -> Embedding -> LLM.
    All off by default to preserve zero-LLM core.
    """

    enabled: bool = False
    simhash_threshold: int = 7  # tighter: ~89% similarity (was 10 / ~85%)
    embedding_threshold: float = 0.85
    embedding_ambiguous_low: float = 0.75
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = ""
    llm_max_pairs_per_encode: int = 3
    merge_strategy: str = "keep_newer"
    max_candidates: int = 30  # wider search (was 10)
    consolidation_max_anchors: int = 2000
    """Anchors the consolidation census compares pairwise per run.

    The window rotates between runs, so this widens each pass rather than deciding
    what is ever looked at. Raising it costs CPU quadratically.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "simhash_threshold": self.simhash_threshold,
            "embedding_threshold": self.embedding_threshold,
            "embedding_ambiguous_low": self.embedding_ambiguous_low,
            "llm_enabled": self.llm_enabled,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_max_pairs_per_encode": self.llm_max_pairs_per_encode,
            "merge_strategy": self.merge_strategy,
            "max_candidates": self.max_candidates,
            "consolidation_max_anchors": self.consolidation_max_anchors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DedupSettings:
        return cls(
            enabled=bool(data.get("enabled", False)),
            simhash_threshold=int(data.get("simhash_threshold", 7)),
            embedding_threshold=float(data.get("embedding_threshold", 0.85)),
            embedding_ambiguous_low=float(data.get("embedding_ambiguous_low", 0.75)),
            llm_enabled=bool(data.get("llm_enabled", False)),
            llm_provider=str(data.get("llm_provider", "none")),
            llm_model=str(data.get("llm_model", "")),
            llm_max_pairs_per_encode=int(data.get("llm_max_pairs_per_encode", 3)),
            merge_strategy=str(data.get("merge_strategy", "keep_newer")),
            max_candidates=int(data.get("max_candidates", 30)),
            consolidation_max_anchors=int(data.get("consolidation_max_anchors", 2000)),
        )


@dataclass(frozen=True)
class Mem0SyncConfig:
    """Auto-sync configuration for Mem0 integration.

    When enabled, the MCP server auto-detects Mem0 (via MEM0_API_KEY env var
    or self_hosted flag) and syncs memories in background on startup.
    """

    enabled: bool = True
    self_hosted: bool = False
    user_id: str = ""
    agent_id: str = ""
    cooldown_minutes: int = 60
    sync_on_startup: bool = True
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": self.enabled,
            "self_hosted": self.self_hosted,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "cooldown_minutes": self.cooldown_minutes,
            "sync_on_startup": self.sync_on_startup,
        }
        if self.limit is not None:
            result["limit"] = self.limit
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mem0SyncConfig:
        user_id = _sanitize_sync_id(data.get("user_id", ""))
        agent_id = _sanitize_sync_id(data.get("agent_id", ""))
        try:
            cooldown = max(1, min(int(data.get("cooldown_minutes", 60)), 1440))
        except (ValueError, TypeError):
            cooldown = 60
        raw_limit = data.get("limit")
        try:
            limit = max(1, min(int(raw_limit), 100_000)) if raw_limit is not None else None
        except (ValueError, TypeError):
            limit = None
        return cls(
            enabled=bool(data.get("enabled", True)),
            self_hosted=bool(data.get("self_hosted", False)),
            user_id=user_id,
            agent_id=agent_id,
            cooldown_minutes=cooldown,
            sync_on_startup=bool(data.get("sync_on_startup", True)),
            limit=limit,
        )


def _sanitize_toml_str(value: str) -> str:
    """Sanitize a string value for safe TOML serialization.

    Prevents TOML injection by stripping control chars and quotes.
    Returns empty string if value contains unsafe characters.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()[:_TOML_STR_MAX_LEN]
    if not _TOML_SAFE_STRING.match(cleaned):
        return ""
    return cleaned


def _sanitize_toml_url(value: str) -> str:
    """Sanitize a URL for safe TOML serialization (double-quoted basic string).

    Unlike :func:`_sanitize_toml_str`, this permits URL characters (``:`` ``?``
    ``&`` ``%`` …) while still rejecting quotes, backslashes, whitespace and
    control characters that could break out of the string or inject TOML. Values
    that are too long or contain any unsafe character return ``""`` (rejected,
    never silently truncated to a broken URL)."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if len(cleaned) > _TOML_URL_MAX_LEN or not _TOML_SAFE_URL.match(cleaned):
        return ""
    return cleaned


_ISO_DATETIME_PATTERN = re.compile(r"^[0-9T:\-\+Z\. ]*$")


def _sanitize_iso_datetime(value: str) -> str:
    """Sanitize an ISO datetime string for TOML. Returns empty on invalid."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()[:64]
    if not _ISO_DATETIME_PATTERN.match(cleaned):
        return ""
    return cleaned


def _sanitize_sync_id(value: str) -> str:
    """Sanitize a sync identifier (user_id, agent_id).

    Strips whitespace, truncates to max length, and validates against
    allowed characters to prevent TOML injection and log injection.
    Returns empty string if invalid.
    """
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()[:_SYNC_ID_MAX_LEN]
    if not _SYNC_ID_PATTERN.match(cleaned):
        return ""
    return cleaned


_VALID_STORAGE_BACKENDS = {"surrealdb", "memory"}

SQLITE_REMOVED_ERROR = (
    "The SQLite storage backend was removed in 3.0.0. Choose one:\n"
    "  - Production: docker compose -f docker-compose.surrealdb.yml up -d, "
    "then set SURREAL_MEMORY_STORAGE=surrealdb\n"
    "  - Persistence-free trial: set SURREAL_MEMORY_STORAGE=memory\n"
    "Migrating existing data: see docs/guides/migrating-to-3.0.md\n"
    "Your existing SQLite brains at ~/.surrealmemory/brains/*.db are untouched — "
    "installing a 2.x release restores full access to them."
)


def _validate_storage_backend(value: str) -> str:
    """Validate and return storage backend, defaulting to surrealdb.

    ``"sqlite"`` is a hard error rather than a silent fallback: routing a
    misconfigured install onto an unintended backend looks like data loss
    (memories go missing because they were never written to the backend the
    user expects), which is worse than failing loudly.
    """
    if value == "sqlite":
        raise ValueError(SQLITE_REMOVED_ERROR)
    if value in _VALID_STORAGE_BACKENDS:
        return value
    logger.warning("Unknown storage_backend '%s', falling back to 'surrealdb'", value)
    return "surrealdb"


@dataclass(frozen=True)
class ToolMemoryConfig:
    """Tool memory auto-capture configuration.

    When enabled, a PostToolUse Claude Code hook captures lightweight
    metadata about every MCP tool call into a JSONL buffer. A deferred
    processing step (during consolidation) promotes patterns to neurons
    and synapses (EFFECTIVE_FOR, USED_WITH).
    """

    enabled: bool = True
    blacklist: tuple[str, ...] = ()  # Tool name prefixes to skip
    cooccurrence_window_s: int = 60  # Seconds for USED_WITH detection
    min_frequency: int = 3  # Min calls before creating a tool neuron
    max_buffer_lines: int = 10000  # Truncate JSONL buffer beyond this
    process_batch_size: int = 200  # Max events per processing cycle

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "blacklist": list(self.blacklist),
            "cooccurrence_window_s": self.cooccurrence_window_s,
            "min_frequency": self.min_frequency,
            "max_buffer_lines": self.max_buffer_lines,
            "process_batch_size": self.process_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolMemoryConfig:
        blacklist_raw = data.get("blacklist", [])
        if isinstance(blacklist_raw, (list, tuple)):
            blacklist = tuple(str(b)[:128] for b in blacklist_raw[:50])
        else:
            blacklist = ()
        try:
            window = max(1, min(int(data.get("cooccurrence_window_s", 60)), 3600))
        except (ValueError, TypeError):
            window = 60
        try:
            min_freq = max(1, min(int(data.get("min_frequency", 3)), 100))
        except (ValueError, TypeError):
            min_freq = 3
        try:
            max_buf = max(100, min(int(data.get("max_buffer_lines", 10000)), 1_000_000))
        except (ValueError, TypeError):
            max_buf = 10000
        try:
            batch = max(10, min(int(data.get("process_batch_size", 200)), 10000))
        except (ValueError, TypeError):
            batch = 200
        return cls(
            enabled=bool(data.get("enabled", False)),
            blacklist=blacklist,
            cooccurrence_window_s=window,
            min_frequency=min_freq,
            max_buffer_lines=max_buf,
            process_batch_size=batch,
        )


# Model names / injection globs may contain glob metacharacters (``*``/``?``/
# ``[]``) that the stricter _TOML_SAFE_STRING rejects. This variant allows them
# while still blocking quotes/backslashes/control chars (TOML-injection safe).
_TOML_SAFE_GLOB = re.compile(r"^[a-zA-Z0-9_\-\./ *?\[\]]*$")

# One argv part of reasoning_training.distill_llm_unload_cmd /
# distill_llm_load_cmd. Deliberately narrower than a glob: no quotes, no shell
# metacharacters, no newlines. Braces are allowed solely for the "{model}"
# placeholder. The command is executed without a shell, so this is defence in
# depth rather than the only guard.
_TOML_SAFE_ARGV = re.compile(r"^[a-zA-Z0-9_\-\./:={}]+$")
_CMD_MAX_PARTS = 16


def _sanitize_toml_glob(value: str) -> str:
    """Sanitize a model-name / glob string for safe TOML serialization."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()[:_TOML_STR_MAX_LEN]
    if not _TOML_SAFE_GLOB.match(cleaned):
        return ""
    return cleaned


_DEFAULT_REASONING_CATEGORIES: tuple[str, ...] = (
    "debugging",
    "planning",
    "implementation",
    "refactoring",
    "research",
    "verification",
    "architecture",
    "data-analysis",
)

# Model-name / glob shape for reasoning config keys — mirrors the route's
# _validate_model_name so config-level and API-level validation agree.
_REASONING_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._*?-]{1,128}$")

# Ceiling for a per-model distillation target. A guard against a runaway
# configuration, not a recommendation: how many patterns a model can actually
# yield is bounded by its trace backlog long before this. Exported because the
# dashboard's config endpoint must reject on the same number the loader clamps
# to, or the UI accepts a value that is then silently reduced on the next read.
MAX_PATTERN_TARGET = 1000


@dataclass(frozen=True)
class ReasoningTrainingConfig:
    """Reasoning-training configuration (mining reasoning traces + injection).

    Opt-in and privacy-preserving: mining and injection are both OFF by default.
    Mirrors the tool-memory pipeline but for model ``thinking`` blocks — mined
    traces are distilled into ReasoningBank patterns and optionally injected into
    other models' sessions per ``injection_map``.
    """

    mining_enabled: bool = False  # opt-in (privacy): reads no transcripts until True
    injection_enabled: bool = False  # opt-in: inject learned strategies into sessions
    # Glob patterns of source models to mine; () = all models with non-empty thinking.
    mining_models: tuple[str, ...] = ()
    # Additional Claude-Code profile roots to mine, alongside the implicit
    # ``~/.claude``. Each entry is a profile ROOT (the directory holding
    # ``projects/``), not the projects dir itself — e.g. ``~/.claude-ZAI``.
    # A second profile is how one machine runs a different vendor's models
    # (Z.AI/GLM, a work account, ...); without this the miner only ever sees
    # ``~/.claude`` and every trace from the other profile is invisible.
    # ``~`` is expanded. Note two profiles can hold a directory of the same
    # name under ``projects/`` — those traces share one ``project`` attribution
    # but stay distinguishable by ``model``, and trace_hash dedup is unaffected.
    extra_transcript_dirs: tuple[str, ...] = ()
    # target_glob -> source_model (one source per target in v1).
    injection_map: tuple[tuple[str, str], ...] = ()
    categories: tuple[str, ...] = _DEFAULT_REASONING_CATEGORIES
    min_trace_chars: int = 200
    max_trace_chars: int = 100_000
    scan_lookback_days: int = 30  # 0 = full backfill
    retention_days: int = 90
    max_traces_total: int = 20_000
    min_cluster_support: int = 3
    # Cosine above which two traces are treated as the same pattern. Embedding
    # models do not share a similarity scale, so this travels with the
    # configured embedder: under bge-m3 the median pairwise cosine of real
    # traces sits near 0.46 and the 99th percentile near 0.62, so a threshold
    # in the high 0.8s clusters almost nothing and the move-set fallback
    # silently outperforms the embedding path it is supposed to back up.
    cluster_cosine: float = 0.75
    min_confidence: float = 0.2
    min_patterns_per_category: int = 3
    injection_max_patterns: int = 5
    injection_max_chars: int = 4000
    # Rewrite distilled pattern prose with a LOCAL chat model. Needs
    # distill_llm_model plus a loopback SURREAL_MEMORY_LLM_ENDPOINT; without
    # both, distillation keeps its heuristic naming (see engine.reasoning_naming).
    distill_use_llm: bool = False
    distill_llm_model: str = ""
    # Loopback OpenAI-compatible base URL, e.g. "http://127.0.0.1:PORT/v1".
    # SURREAL_MEMORY_LLM_ENDPOINT overrides it; a non-loopback value is refused.
    distill_llm_endpoint: str = ""
    # argv (NOT a shell string) run once after distillation to unload the chat
    # model again, so it does not stay resident between runs. "{model}" is
    # replaced with distill_llm_model. Empty = leave the model loaded.
    distill_llm_unload_cmd: tuple[str, ...] = ()
    # argv (NOT a shell string) run once before the first distillation request,
    # so a model needing specific launch parameters (e.g. GPU-only, no
    # multimodal projector) does not depend on whatever an external
    # auto-starter happens to remember. Same "{model}" substitution as
    # distill_llm_unload_cmd. Empty = keep relying on implicit
    # load-on-first-request.
    distill_llm_load_cmd: tuple[str, ...] = ()
    redact_secrets: bool = True
    # Per-model distillation targets: model name -> desired pattern count (0-100).
    # An unlisted model defaults to 0 → distillation is skipped for it (the
    # preliminary Mine only DETECTS models; a UI slider raises a target to
    # actually distill). Mutable default, but the config is loaded read-only.
    pattern_targets: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mining_enabled": self.mining_enabled,
            "injection_enabled": self.injection_enabled,
            "mining_models": list(self.mining_models),
            "extra_transcript_dirs": list(self.extra_transcript_dirs),
            "injection_map": dict(self.injection_map),
            "categories": list(self.categories),
            "min_trace_chars": self.min_trace_chars,
            "max_trace_chars": self.max_trace_chars,
            "scan_lookback_days": self.scan_lookback_days,
            "retention_days": self.retention_days,
            "max_traces_total": self.max_traces_total,
            "min_cluster_support": self.min_cluster_support,
            "cluster_cosine": self.cluster_cosine,
            "min_confidence": self.min_confidence,
            "min_patterns_per_category": self.min_patterns_per_category,
            "injection_max_patterns": self.injection_max_patterns,
            "injection_max_chars": self.injection_max_chars,
            "distill_use_llm": self.distill_use_llm,
            "distill_llm_model": self.distill_llm_model,
            "distill_llm_endpoint": self.distill_llm_endpoint,
            "distill_llm_unload_cmd": list(self.distill_llm_unload_cmd),
            "distill_llm_load_cmd": list(self.distill_llm_load_cmd),
            "redact_secrets": self.redact_secrets,
            "pattern_targets": dict(self.pattern_targets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningTrainingConfig:
        def _int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(int(data.get(key, default)), hi))
            except (ValueError, TypeError):
                return default

        models_raw = data.get("mining_models", [])
        if isinstance(models_raw, (list, tuple)):
            mining_models = tuple(str(m)[:128] for m in models_raw[:100] if str(m).strip())
        else:
            mining_models = ()

        # Extra profile roots. Capped and length-limited like every other list
        # here; validity of each path is the miner's problem (a configured
        # profile that is not installed on this machine is skipped, not fatal).
        dirs_raw = data.get("extra_transcript_dirs", [])
        if isinstance(dirs_raw, (list, tuple)):
            extra_transcript_dirs = tuple(
                str(d).strip()[:4096] for d in dirs_raw[:20] if str(d).strip()
            )
        else:
            extra_transcript_dirs = ()

        map_raw = data.get("injection_map", {})
        injection_pairs: list[tuple[str, str]] = []
        if isinstance(map_raw, dict):
            for target, source in map_raw.items():
                t, s = str(target).strip(), str(source).strip()
                if t and s:
                    injection_pairs.append((t[:128], s[:128]))
        elif isinstance(map_raw, (list, tuple)):
            for item in map_raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    t, s = str(item[0]).strip(), str(item[1]).strip()
                    if t and s:
                        injection_pairs.append((t[:128], s[:128]))

        cats_raw = data.get("categories")
        if isinstance(cats_raw, (list, tuple)) and cats_raw:
            categories = tuple(str(c)[:64] for c in cats_raw[:50] if str(c).strip())
        else:
            categories = _DEFAULT_REASONING_CATEGORIES

        try:
            min_confidence = max(0.0, min(float(data.get("min_confidence", 0.2)), 1.0))
        except (ValueError, TypeError):
            min_confidence = 0.2

        # Floor at 0.05: single-linkage clustering collapses into one giant
        # component once the threshold drops near the corpus baseline, which
        # yields FEWER patterns, not more, so a near-zero value is never what
        # an operator meant.
        try:
            raw_cosine = float(data.get("cluster_cosine", 0.75))
            # NaN must not reach the clamp: min()/max() propagate it into the
            # FLOOR, which would merge every trace into one cluster instead of
            # falling back to a sane threshold.
            if raw_cosine != raw_cosine:
                raise ValueError("cluster_cosine is NaN")
            cluster_cosine = max(0.05, min(raw_cosine, 1.0))
        except (ValueError, TypeError):
            cluster_cosine = 0.75

        # Per-model pattern targets: {model: 0..100}. Keys must be model-name
        # shaped (bad keys skipped, not fatal — this loads from disk); values
        # clamp to 0..100. Legacy keys like max_patterns_per_run are ignored
        # simply by never being read here.
        targets_raw = data.get("pattern_targets", {})
        pattern_targets: dict[str, int] = {}
        if isinstance(targets_raw, dict):
            for model, count in targets_raw.items():
                name = str(model).strip()
                if not name or not _REASONING_MODEL_NAME_RE.match(name):
                    continue
                try:
                    pattern_targets[name[:128]] = max(0, min(int(count), MAX_PATTERN_TARGET))
                except (ValueError, TypeError):
                    continue

        # Unload command: argv, never a shell string. A part that is not plain
        # command syntax voids the whole command rather than being silently
        # dropped — a half-parsed teardown command is worse than none. Same
        # rule for length: a command longer than _CMD_MAX_PARTS is voided
        # outright rather than silently truncated to its first N parts, which
        # would otherwise run a shorter, functionally different command with
        # no signal that anything was dropped.
        unload_raw = data.get("distill_llm_unload_cmd", [])
        unload_cmd: tuple[str, ...] = ()
        if isinstance(unload_raw, (list, tuple)) and unload_raw:
            if len(unload_raw) > _CMD_MAX_PARTS:
                logger.warning(
                    "distill_llm_unload_cmd has %d parts, exceeding the %d-part limit;"
                    " ignoring it entirely rather than running a truncated command",
                    len(unload_raw),
                    _CMD_MAX_PARTS,
                )
            else:
                parts = [str(p).strip() for p in unload_raw]
                if all(p and _TOML_SAFE_ARGV.match(p) for p in parts):
                    unload_cmd = tuple(p[:_TOML_STR_MAX_LEN] for p in parts)

        # Load command: same argv-only and length rules as unload_cmd above.
        load_raw = data.get("distill_llm_load_cmd", [])
        load_cmd: tuple[str, ...] = ()
        if isinstance(load_raw, (list, tuple)) and load_raw:
            if len(load_raw) > _CMD_MAX_PARTS:
                logger.warning(
                    "distill_llm_load_cmd has %d parts, exceeding the %d-part limit;"
                    " ignoring it entirely rather than running a truncated command",
                    len(load_raw),
                    _CMD_MAX_PARTS,
                )
            else:
                parts = [str(p).strip() for p in load_raw]
                if all(p and _TOML_SAFE_ARGV.match(p) for p in parts):
                    load_cmd = tuple(p[:_TOML_STR_MAX_LEN] for p in parts)

        return cls(
            mining_enabled=bool(data.get("mining_enabled", False)),
            injection_enabled=bool(data.get("injection_enabled", False)),
            mining_models=mining_models,
            extra_transcript_dirs=extra_transcript_dirs,
            injection_map=tuple(injection_pairs),
            categories=categories,
            min_trace_chars=_int("min_trace_chars", 200, 0, 1_000_000),
            max_trace_chars=_int("max_trace_chars", 100_000, 1, 10_000_000),
            scan_lookback_days=_int("scan_lookback_days", 30, 0, 100_000),
            retention_days=_int("retention_days", 90, 1, 100_000),
            max_traces_total=_int("max_traces_total", 20_000, 1, 100_000_000),
            min_cluster_support=_int("min_cluster_support", 3, 1, 100_000),
            cluster_cosine=cluster_cosine,
            min_confidence=min_confidence,
            min_patterns_per_category=_int("min_patterns_per_category", 3, 1, 100_000),
            injection_max_patterns=_int("injection_max_patterns", 5, 1, 1000),
            injection_max_chars=_int("injection_max_chars", 4000, 1, 1_000_000),
            distill_use_llm=bool(data.get("distill_use_llm", False)),
            distill_llm_model=_sanitize_toml_glob(str(data.get("distill_llm_model", "") or "")),
            distill_llm_endpoint=_sanitize_toml_url(
                str(data.get("distill_llm_endpoint", "") or "")
            ),
            distill_llm_unload_cmd=unload_cmd,
            distill_llm_load_cmd=load_cmd,
            redact_secrets=bool(data.get("redact_secrets", True)),
            pattern_targets=pattern_targets,
        )


def _load_reasoning_settings(data: dict[str, Any]) -> ReasoningTrainingConfig:
    """Build ReasoningTrainingConfig from config.toml, letting env vars override.

    Env precedence (env wins over config.toml, config.toml wins over defaults):
        SURREAL_MEMORY_REASONING_MINING        -> mining_enabled (truthy parse)
        SURREAL_MEMORY_REASONING_INJECTION     -> injection_enabled (truthy parse)
        SURREAL_MEMORY_REASONING_MODELS        -> mining_models (comma-separated globs)
        SURREAL_MEMORY_REASONING_INJECTION_MAP -> injection_map ("target=source,..." pairs)
        SURREAL_MEMORY_REASONING_EXTRA_DIRS    -> extra_transcript_dirs (comma-separated
                                                  profile roots, e.g. "~/.claude-ZAI")
    """
    base = ReasoningTrainingConfig.from_dict(data)
    overrides: dict[str, Any] = {}

    env_mining = _env_truthy(os.environ.get("SURREAL_MEMORY_REASONING_MINING"))
    if env_mining is not None:
        overrides["mining_enabled"] = env_mining

    env_injection = _env_truthy(os.environ.get("SURREAL_MEMORY_REASONING_INJECTION"))
    if env_injection is not None:
        overrides["injection_enabled"] = env_injection

    env_models = os.environ.get("SURREAL_MEMORY_REASONING_MODELS")
    if env_models is not None and env_models.strip():
        overrides["mining_models"] = tuple(m.strip() for m in env_models.split(",") if m.strip())

    env_dirs = os.environ.get("SURREAL_MEMORY_REASONING_EXTRA_DIRS")
    if env_dirs is not None and env_dirs.strip():
        overrides["extra_transcript_dirs"] = tuple(
            d.strip() for d in env_dirs.split(",") if d.strip()
        )

    env_map = os.environ.get("SURREAL_MEMORY_REASONING_INJECTION_MAP")
    if env_map is not None and env_map.strip():
        pairs: list[tuple[str, str]] = []
        for chunk in env_map.split(","):
            target, sep, source = chunk.partition("=")
            if sep and target.strip() and source.strip():
                pairs.append((target.strip(), source.strip()))
        overrides["injection_map"] = tuple(pairs)

    if overrides:
        return dataclasses.replace(base, **overrides)
    return base


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram backup integration configuration.

    Bot token is read from SURREAL_MEMORY_TELEGRAM_BOT_TOKEN env var (never in config file).
    Chat IDs are stored in config.toml [telegram] section.
    """

    enabled: bool = False
    chat_ids: tuple[str, ...] = ()
    max_file_size_mb: int = 50
    backup_on_consolidation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "chat_ids": list(self.chat_ids),
            "max_file_size_mb": self.max_file_size_mb,
            "backup_on_consolidation": self.backup_on_consolidation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelegramConfig:
        raw_ids = data.get("chat_ids", [])
        if isinstance(raw_ids, (list, tuple)):
            chat_ids = tuple(str(cid).strip() for cid in raw_ids if str(cid).strip())
        else:
            chat_ids = ()
        try:
            max_size = max(1, min(int(data.get("max_file_size_mb", 50)), 2000))
        except (ValueError, TypeError):
            max_size = 50
        return cls(
            enabled=bool(data.get("enabled", False)),
            chat_ids=chat_ids,
            max_file_size_mb=max_size,
            backup_on_consolidation=bool(data.get("backup_on_consolidation", False)),
        )


@dataclass
class BudgetRetrievalConfig:
    """Token budget configuration for retrieval context allocation.

    Controls how budget-aware retrieval allocates the context window
    across candidate fibers using value-per-token ranking.
    """

    enabled: bool = True
    default_tokens: int = 4000
    system_overhead: int = 50
    per_fiber_overhead: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_tokens": self.default_tokens,
            "system_overhead": self.system_overhead,
            "per_fiber_overhead": self.per_fiber_overhead,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetRetrievalConfig:
        try:
            default_tokens = max(50, min(int(data.get("default_tokens", 4000)), 100_000))
        except (ValueError, TypeError):
            default_tokens = 4000
        try:
            system_overhead = max(0, min(int(data.get("system_overhead", 50)), 500))
        except (ValueError, TypeError):
            system_overhead = 50
        try:
            per_fiber_overhead = max(0, min(int(data.get("per_fiber_overhead", 15)), 200))
        except (ValueError, TypeError):
            per_fiber_overhead = 15
        return cls(
            enabled=bool(data.get("enabled", True)),
            default_tokens=default_tokens,
            system_overhead=system_overhead,
            per_fiber_overhead=per_fiber_overhead,
        )


@dataclass
class ResponseConfig:
    """MCP response compaction settings.

    Controls how verbose MCP tool responses are. Compact mode strips
    metadata hints, truncates lists, and shortens content previews
    to reduce token waste in agent context windows.
    """

    # Enable compact mode globally (agents can also set per-call via compact=true)
    compact_mode: bool = False

    # Max items in list fields before truncation (compact mode only)
    max_list_items: int = 10

    # Strip DX hint fields (maintenance_hint, update_hint, onboarding, etc.)
    strip_hints: bool = True

    # Max chars for content preview in list responses
    content_preview_length: int = 120

    # Auto-compact threshold: if any list in response has more items than this,
    # compact mode is applied automatically (0 = disabled)
    auto_compact_threshold: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResponseConfig:
        """Create from TOML dict."""
        return cls(
            compact_mode=bool(data.get("compact_mode", False)),
            max_list_items=int(data.get("max_list_items", 10)),
            strip_hints=bool(data.get("strip_hints", True)),
            content_preview_length=int(data.get("content_preview_length", 120)),
            auto_compact_threshold=int(data.get("auto_compact_threshold", 20)),
        )


_VALID_TIERS = frozenset({"free", "pro", "team"})


@dataclass(frozen=True)
class LicenseConfig:
    """License tier information — set via smem_sync_config(action='activate')."""

    tier: str = "free"  # "free" | "pro" | "team"
    activated_at: str = ""
    expires_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "activated_at": self.activated_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LicenseConfig:
        tier = str(data.get("tier", "free")).lower()
        if tier not in _VALID_TIERS:
            tier = "free"
        return cls(
            tier=tier,
            activated_at=_sanitize_iso_datetime(str(data.get("activated_at", ""))),
            expires_at=_sanitize_iso_datetime(str(data.get("expires_at", ""))),
        )


@dataclass(frozen=True)
class RerankerConfig:
    """Settings for optional cross-encoder reranking after spreading activation."""

    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    blend_weight: float = 0.7  # Reranker weight (SA gets 1 - this)
    min_score: float = 0.15
    max_candidates: int = 30  # Safety cap on overfetch
    # OpenAI-compatible /rerank base URL (e.g. llamastash "http://127.0.0.1:11435/v1").
    # When set, reranking is served over HTTP (no in-process torch/CrossEncoder).
    endpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "blend_weight": self.blend_weight,
            "min_score": self.min_score,
            "max_candidates": self.max_candidates,
            "endpoint": self.endpoint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RerankerConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            model_name=str(data.get("model_name", "BAAI/bge-reranker-v2-m3")),
            blend_weight=float(data.get("blend_weight", 0.7)),
            min_score=float(data.get("min_score", 0.15)),
            max_candidates=int(data.get("max_candidates", 30)),
            endpoint=str(data.get("endpoint", "")),
        )


def reranker_brain_config_overrides(reranker: RerankerConfig) -> dict[str, Any]:
    """Map ``config.toml [reranker]`` onto the ``BrainConfig.reranker_*`` fields.

    The retrieval pipeline reads reranking knobs from the per-brain
    ``BrainConfig``, so this bridges the app-level ``RerankerConfig`` onto those
    fields. Used both when creating a brain and when layering config over an
    already-stored brain (:func:`_migrate_brain_runtime_config`).
    """
    return {
        "reranker_enabled": reranker.enabled,
        "reranker_model": reranker.model_name,
        "reranker_blend_weight": reranker.blend_weight,
        "reranker_min_score": reranker.min_score,
        "reranker_max_candidates": reranker.max_candidates,
        "reranker_endpoint": reranker.endpoint,
    }


@dataclass(frozen=True)
class TierConfig:
    """Auto-tier promotion/demotion configuration (Pro feature).

    Controls automatic movement of memories between HOT/WARM/COLD tiers
    based on access patterns. Free users keep manual tiers only.
    """

    auto_enabled: bool = False  # Pro only — free users keep manual tiers
    promote_threshold: int = 5  # access_frequency >= N → WARM→HOT
    demote_inactive_days: int = 30  # no access in N days → HOT→WARM
    cold_archive_days: int = 90  # no access in N days → WARM→COLD
    max_hot_memories: int = 100  # cap HOT tier size

    def __post_init__(self) -> None:
        if self.promote_threshold < 1:
            object.__setattr__(self, "promote_threshold", 1)
        if self.demote_inactive_days < 1:
            object.__setattr__(self, "demote_inactive_days", 1)
        if self.cold_archive_days < 1:
            object.__setattr__(self, "cold_archive_days", 1)
        if self.max_hot_memories < 1:
            object.__setattr__(self, "max_hot_memories", 1)
        # Invariant: cold_archive_days must be >= demote_inactive_days
        if self.cold_archive_days < self.demote_inactive_days:
            object.__setattr__(self, "cold_archive_days", self.demote_inactive_days)

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_enabled": self.auto_enabled,
            "promote_threshold": self.promote_threshold,
            "demote_inactive_days": self.demote_inactive_days,
            "cold_archive_days": self.cold_archive_days,
            "max_hot_memories": self.max_hot_memories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierConfig:
        return cls(
            auto_enabled=bool(data.get("auto_enabled", False)),
            promote_threshold=int(data.get("promote_threshold", 5)),
            demote_inactive_days=int(data.get("demote_inactive_days", 30)),
            cold_archive_days=int(data.get("cold_archive_days", 90)),
            max_hot_memories=int(data.get("max_hot_memories", 100)),
        )


@dataclass(frozen=True)
class WatcherConfig:
    """Settings for file watcher auto-ingestion."""

    enabled: bool = False
    paths: tuple[str, ...] = ()
    extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".json",
        ".csv",
        ".xlsx",
        ".py",
        ".ts",
        ".js",
    )
    ignore_patterns: tuple[str, ...] = (
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        ".env",
    )
    debounce_seconds: float = 2.0
    max_file_size_mb: int = 10
    max_watched_dirs: int = 10
    memory_type: str = "fact"
    domain_tag: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "paths": list(self.paths),
            "extensions": list(self.extensions),
            "ignore_patterns": list(self.ignore_patterns),
            "debounce_seconds": self.debounce_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "max_watched_dirs": self.max_watched_dirs,
            "memory_type": self.memory_type,
            "domain_tag": self.domain_tag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatcherConfig:
        return cls(
            enabled=bool(data.get("enabled", False)),
            paths=tuple(data.get("paths", ())),
            extensions=tuple(data.get("extensions", cls.extensions)),
            ignore_patterns=tuple(data.get("ignore_patterns", cls.ignore_patterns)),
            debounce_seconds=float(data.get("debounce_seconds", 2.0)),
            max_file_size_mb=int(data.get("max_file_size_mb", 10)),
            max_watched_dirs=int(data.get("max_watched_dirs", 10)),
            memory_type=str(data.get("memory_type", "fact")),
            domain_tag=str(data.get("domain_tag", "")),
        )


@dataclass
class UnifiedConfig:
    """Unified configuration for Surreal-Memory.

    This configuration is shared across all tools:
    - CLI: smem commands
    - MCP: Claude Code, Cursor, AntiGravity
    - API: REST server

    Storage location: ~/.surrealmemory/config.toml
    Brain location: ~/.surrealmemory/brains/<name>.db
    """

    # Authoritative list of top-level TOML sections written by save(). Single
    # source of truth consumed by `smem doctor`'s config-freshness check — the
    # check and save() stay in lockstep, so a section reported "missing" is
    # always one that --fix (which calls save()) can actually create. save()
    # validates its emitted headers against this tuple and raises on drift.
    SECTION_NAMES: ClassVar[tuple[str, ...]] = (
        "brain",
        "embedding",
        "auto",
        "eternal",
        "maintenance",
        "safety",
        "encryption",
        "write_gate",
        "dedup",
        "tool_memory",
        "mem0_sync",
        "sync",
        "telegram",
        "license",
        "reranker",
        "tiers",
        "watcher",
        "tool_tier",
        "response",
        "trace",
        "reasoning_training",
        "cli",
    )

    # Base directory for all Surreal-Memory data
    data_dir: Path = field(default_factory=get_surrealmemory_dir)

    # Current active brain
    current_brain: str = field(default_factory=get_default_brain)

    # Brain settings
    brain: BrainSettings = field(default_factory=BrainSettings)

    # Embedding settings (cross-language recall)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)

    # Auto-capture settings for MCP
    auto: AutoConfig = field(default_factory=AutoConfig)

    # Eternal context settings
    eternal: EternalConfig = field(default_factory=EternalConfig)

    # Proactive maintenance settings
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)

    # Safety settings
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # Encryption settings
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)

    # Write gate (quality enforcement before storage)
    write_gate: WriteGateConfig = field(default_factory=WriteGateConfig)

    # Dedup settings
    dedup: DedupSettings = field(default_factory=DedupSettings)

    # MCP tool tier settings
    tool_tier: ToolTierConfig = field(default_factory=ToolTierConfig)

    # Mem0 auto-sync settings
    mem0_sync: Mem0SyncConfig = field(default_factory=Mem0SyncConfig)

    # Device identity (stable per-machine ID)
    device_id: str = ""

    # Multi-device sync settings
    sync: SyncConfig = field(default_factory=SyncConfig)

    # Storage backend: "surrealdb" (default) or "memory"
    storage_backend: str = "surrealdb"

    # Tool memory auto-capture
    tool_memory: ToolMemoryConfig = field(default_factory=ToolMemoryConfig)

    # Telegram backup integration
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    # License tier (free/pro/team)
    license: LicenseConfig = field(default_factory=LicenseConfig)

    # Cross-encoder reranking (optional)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    # Auto-tier promotion/demotion (Pro feature)
    tiers: TierConfig = field(default_factory=TierConfig)

    # File watcher auto-ingestion
    watcher: WatcherConfig = field(default_factory=WatcherConfig)

    # MCP response compaction
    response: ResponseConfig = field(default_factory=ResponseConfig)

    # Token budget retrieval
    budget: BudgetRetrievalConfig = field(default_factory=BudgetRetrievalConfig)

    # Retrieval-trace telemetry (schema v9, opt-in; neutral default = off)
    trace: TraceConfig = field(default_factory=TraceConfig)

    # Reasoning-training (mining reasoning traces + injection; opt-in, off by default)
    reasoning_training: ReasoningTrainingConfig = field(default_factory=ReasoningTrainingConfig)

    # CLI preferences
    json_output: bool = False
    default_depth: int | None = None
    default_max_tokens: int = 500

    # Metadata
    version: str = "1.0"

    def is_pro(self) -> bool:
        """Always True — Surreal-Memory is fully free.

        The paid InfinityDB/Pro chain was removed in the SurrealDB-only switch
        (commit #28), so every feature is unlocked for everyone. Kept as a
        method (rather than deleted) because callers and the dashboard still
        gate optional UI on it.
        """
        return True  # Malformed expiry → treat as perpetual

    @classmethod
    def load(cls, config_path: Path | None = None) -> UnifiedConfig:
        """Load configuration from file, or create default if doesn't exist."""
        if config_path is None:
            data_dir = get_surrealmemory_dir()
            config_path = data_dir / "config.toml"
        else:
            data_dir = config_path.parent

        if not config_path.exists():
            # Migrate current_brain from legacy config.json if available
            legacy_brain = _read_legacy_brain(data_dir)
            from surreal_memory.sync.device import get_device_id as _get_device_id

            config = cls(
                data_dir=data_dir,
                current_brain=legacy_brain or get_default_brain(),
                device_id=_get_device_id(data_dir),
                # Honor SURREAL_MEMORY_STORAGE even before a config.toml exists,
                # so a fresh process does not silently cache a sqlite singleton
                # while env says surrealdb (active brain would then read empty).
                storage_backend=_validate_storage_backend(
                    os.environ.get("SURREAL_MEMORY_STORAGE") or "surrealdb"
                ),
            )
            config.save()
            if legacy_brain:
                _logger = logging.getLogger(__name__)
                _logger.info(
                    "Migrated current_brain=%s from legacy config.json to config.toml",
                    legacy_brain,
                )
            return config

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        from surreal_memory.sync.device import get_device_id

        sync_data = data.get("sync", {})
        # device_id: prefer explicit value in [sync] section, else generate/read from file
        raw_device_id = str(data.get("device_id", "") or sync_data.get("device_id", "")).strip()
        if not raw_device_id:
            raw_device_id = get_device_id(data_dir)

        return cls(
            data_dir=data_dir,
            current_brain=(
                os.environ.get("SURREAL_MEMORY_BRAIN")
                or os.environ.get("SURREAL_MEMORY_BRAIN")
                or data.get("current_brain", get_default_brain())
            ),
            brain=BrainSettings.from_dict(data.get("brain", {})),
            embedding=_load_embedding_settings(data.get("embedding", {})),
            auto=AutoConfig.from_dict(data.get("auto", {})),
            eternal=EternalConfig.from_dict(data.get("eternal", {})),
            maintenance=MaintenanceConfig.from_dict(data.get("maintenance", {})),
            safety=SafetyConfig.from_dict(data.get("safety", {})),
            encryption=EncryptionConfig.from_dict(data.get("encryption", {})),
            write_gate=WriteGateConfig.from_dict(data.get("write_gate", {})),
            dedup=DedupSettings.from_dict(data.get("dedup", {})),
            tool_memory=ToolMemoryConfig.from_dict(data.get("tool_memory", {})),
            telegram=TelegramConfig.from_dict(data.get("telegram", {})),
            tool_tier=ToolTierConfig.from_dict(data.get("tool_tier", {})),
            mem0_sync=Mem0SyncConfig.from_dict(data.get("mem0_sync", {})),
            license=LicenseConfig.from_dict(data.get("license", {})),
            reranker=RerankerConfig.from_dict(data.get("reranker", {})),
            tiers=TierConfig.from_dict(data.get("tiers", {})),
            watcher=WatcherConfig.from_dict(data.get("watcher", {})),
            device_id=raw_device_id,
            sync=_load_sync_settings(sync_data),
            storage_backend=_validate_storage_backend(
                os.environ.get("SURREAL_MEMORY_STORAGE")
                or str(
                    data.get("storage_backend") or sync_data.get("storage_backend") or "surrealdb"
                )
            ),
            response=ResponseConfig.from_dict(data.get("response", {})),
            budget=BudgetRetrievalConfig.from_dict(data.get("budget", {})),
            trace=TraceConfig.from_dict(data.get("trace", {})),
            reasoning_training=_load_reasoning_settings(data.get("reasoning_training", {})),
            json_output=data.get("cli", {}).get("json_output", False),
            default_depth=data.get("cli", {}).get("default_depth"),
            default_max_tokens=data.get("cli", {}).get("default_max_tokens", 500),
            version=data.get("version", "1.0"),
        )

    def _reasoning_toml_lines(self) -> list[str]:
        """Render the [reasoning_training] TOML section (+ injection_map subtable)."""
        rt = self.reasoning_training
        models = ", ".join(f'"{g}"' for m in rt.mining_models if (g := _sanitize_toml_glob(m)))
        cats = ", ".join(f'"{g}"' for c in rt.categories if (g := _sanitize_toml_glob(c)))
        lines = [
            "# Reasoning-training (mining reasoning traces + injection; opt-in, off by default)",
            "[reasoning_training]",
            f"mining_enabled = {'true' if rt.mining_enabled else 'false'}",
            f"injection_enabled = {'true' if rt.injection_enabled else 'false'}",
            f"mining_models = [{models}]",
            f"categories = [{cats}]",
            f"min_trace_chars = {rt.min_trace_chars}",
            f"max_trace_chars = {rt.max_trace_chars}",
            f"scan_lookback_days = {rt.scan_lookback_days}",
            f"retention_days = {rt.retention_days}",
            f"max_traces_total = {rt.max_traces_total}",
            f"min_cluster_support = {rt.min_cluster_support}",
            f"cluster_cosine = {rt.cluster_cosine}",
            f"min_confidence = {rt.min_confidence}",
            f"min_patterns_per_category = {rt.min_patterns_per_category}",
            f"injection_max_patterns = {rt.injection_max_patterns}",
            f"injection_max_chars = {rt.injection_max_chars}",
            f"distill_use_llm = {'true' if rt.distill_use_llm else 'false'}",
            f'distill_llm_model = "{_sanitize_toml_glob(rt.distill_llm_model)}"',
            f'distill_llm_endpoint = "{_sanitize_toml_url(rt.distill_llm_endpoint)}"',
            "distill_llm_unload_cmd = ["
            + ", ".join(f'"{p}"' for p in rt.distill_llm_unload_cmd if _TOML_SAFE_ARGV.match(p))
            + "]",
            "distill_llm_load_cmd = ["
            + ", ".join(f'"{p}"' for p in rt.distill_llm_load_cmd if _TOML_SAFE_ARGV.match(p))
            + "]",
            f"redact_secrets = {'true' if rt.redact_secrets else 'false'}",
            "[reasoning_training.injection_map]",
        ]
        for target, source in rt.injection_map:
            gk = _sanitize_toml_glob(target)
            gv = _sanitize_toml_glob(source)
            if gk and gv:
                lines.append(f'"{gk}" = "{gv}"')
        # Per-model pattern targets subtable (model = 0..100). Empty is fine.
        lines.append("[reasoning_training.pattern_targets]")
        for model, count in rt.pattern_targets.items():
            gk = _sanitize_toml_glob(model)
            if gk:
                lines.append(f'"{gk}" = {int(count)}')
        return lines

    def save(self) -> None:
        """Save configuration to TOML file (atomic write via temp+rename)."""
        import tempfile

        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.data_dir / "config.toml"

        # Validate brain name before writing to prevent TOML injection
        if not _BRAIN_NAME_PATTERN.match(self.current_brain):
            raise ValueError("Invalid brain name for config save")

        # Build TOML content manually (no toml write dependency)
        lines = [
            "# Surreal-Memory Configuration",
            "# This config is shared by CLI, MCP server, and all integrations",
            "",
            f'version = "{self.version}"',
            f'current_brain = "{self.current_brain}"',
            f'storage_backend = "{_sanitize_toml_str(self.storage_backend)}"',
            "",
            "# Brain behavior settings",
            "[brain]",
            f"decay_rate = {self.brain.decay_rate}",
            f"reinforcement_delta = {self.brain.reinforcement_delta}",
            f"reinforcement_neuron_limit = {self.brain.reinforcement_neuron_limit}",
            f"activation_threshold = {self.brain.activation_threshold}",
            f"max_spread_hops = {self.brain.max_spread_hops}",
            f"max_context_tokens = {self.brain.max_context_tokens}",
            f"freshness_weight = {self.brain.freshness_weight}",
            "",
            "# Embedding settings (cross-language recall via Gemini/OpenAI/OpenRouter)",
            "[embedding]",
            f"enabled = {'true' if self.embedding.enabled else 'false'}",
            f'provider = "{self.embedding.provider}"',
            f'model = "{self.embedding.model}"',
            f"similarity_threshold = {self.embedding.similarity_threshold}",
            f"dimension = {self.embedding.dimension}",
            f'endpoint = "{_sanitize_toml_url(self.embedding.endpoint)}"',
            "",
            "# Auto-capture settings for MCP server",
            "[auto]",
            f"enabled = {'true' if self.auto.enabled else 'false'}",
            f"capture_decisions = {'true' if self.auto.capture_decisions else 'false'}",
            f"capture_errors = {'true' if self.auto.capture_errors else 'false'}",
            f"capture_todos = {'true' if self.auto.capture_todos else 'false'}",
            f"capture_facts = {'true' if self.auto.capture_facts else 'false'}",
            f"capture_insights = {'true' if self.auto.capture_insights else 'false'}",
            f"capture_preferences = {'true' if self.auto.capture_preferences else 'false'}",
            f"min_confidence = {self.auto.min_confidence}",
            "",
            "# Eternal context settings",
            "[eternal]",
            f"enabled = {'true' if self.eternal.enabled else 'false'}",
            f"notifications = {'true' if self.eternal.notifications else 'false'}",
            f"auto_save_interval = {self.eternal.auto_save_interval}",
            f"context_warning_threshold = {self.eternal.context_warning_threshold}",
            f"max_context_tokens = {self.eternal.max_context_tokens}",
            "",
            "# Proactive maintenance settings",
            "[maintenance]",
            f"enabled = {'true' if self.maintenance.enabled else 'false'}",
            f"check_interval = {self.maintenance.check_interval}",
            f"fiber_warn_threshold = {self.maintenance.fiber_warn_threshold}",
            f"neuron_warn_threshold = {self.maintenance.neuron_warn_threshold}",
            f"synapse_warn_threshold = {self.maintenance.synapse_warn_threshold}",
            f"orphan_ratio_threshold = {self.maintenance.orphan_ratio_threshold}",
            f"expired_memory_warn_threshold = {self.maintenance.expired_memory_warn_threshold}",
            f"stale_fiber_ratio_threshold = {self.maintenance.stale_fiber_ratio_threshold}",
            f"stale_fiber_days = {self.maintenance.stale_fiber_days}",
            f"auto_consolidate = {'true' if self.maintenance.auto_consolidate else 'false'}",
            f"auto_consolidate_strategies = {json.dumps(list(self.maintenance.auto_consolidate_strategies))}",
            f"consolidate_cooldown_minutes = {self.maintenance.consolidate_cooldown_minutes}",
            f"dream_cooldown_hours = {self.maintenance.dream_cooldown_hours}",
            f"expiry_cleanup_enabled = {'true' if self.maintenance.expiry_cleanup_enabled else 'false'}",
            f"expiry_cleanup_interval_hours = {self.maintenance.expiry_cleanup_interval_hours}",
            f"expiry_cleanup_max_per_run = {self.maintenance.expiry_cleanup_max_per_run}",
            f"scheduled_consolidation_enabled = {'true' if self.maintenance.scheduled_consolidation_enabled else 'false'}",
            f"scheduled_consolidation_interval_hours = {self.maintenance.scheduled_consolidation_interval_hours}",
            f"scheduled_consolidation_strategies = {json.dumps(list(self.maintenance.scheduled_consolidation_strategies))}",
            "",
            "# Safety settings",
            "[safety]",
            f"auto_redact_min_severity = {self.safety.auto_redact_min_severity}",
            "",
            "# Encryption settings",
            "[encryption]",
            f"enabled = {'true' if self.encryption.enabled else 'false'}",
            f"auto_encrypt_sensitive = {'true' if self.encryption.auto_encrypt_sensitive else 'false'}",
            f'keys_dir = "{_sanitize_toml_str(self.encryption.keys_dir)}"',
            "",
            "# Write gate (quality enforcement before storage)",
            "[write_gate]",
            f"enabled = {'true' if self.write_gate.enabled else 'false'}",
            f'mode = "{self.write_gate.mode}"',
            f'auto_capture_mode = "{self.write_gate.auto_capture_mode}"',
            f"min_length = {self.write_gate.min_length}",
            f"min_quality_score = {self.write_gate.min_quality_score}",
            f"auto_capture_min_score = {self.write_gate.auto_capture_min_score}",
            f"max_content_length = {self.write_gate.max_content_length}",
            f"reject_generic_filler = {'true' if self.write_gate.reject_generic_filler else 'false'}",
            "",
            "# Dedup settings",
            "[dedup]",
            f"enabled = {'true' if self.dedup.enabled else 'false'}",
            f"simhash_threshold = {self.dedup.simhash_threshold}",
            f"embedding_threshold = {self.dedup.embedding_threshold}",
            f"embedding_ambiguous_low = {self.dedup.embedding_ambiguous_low}",
            f"llm_enabled = {'true' if self.dedup.llm_enabled else 'false'}",
            f'llm_provider = "{_sanitize_toml_str(self.dedup.llm_provider)}"',
            f'llm_model = "{_sanitize_toml_str(self.dedup.llm_model)}"',
            f"llm_max_pairs_per_encode = {self.dedup.llm_max_pairs_per_encode}",
            f'merge_strategy = "{_sanitize_toml_str(self.dedup.merge_strategy)}"',
            f"max_candidates = {self.dedup.max_candidates}",
            "",
            "# Tool memory auto-capture",
            "[tool_memory]",
            f"enabled = {'true' if self.tool_memory.enabled else 'false'}",
            f"blacklist = [{', '.join(repr(b) for b in self.tool_memory.blacklist)}]",
            f"cooccurrence_window_s = {self.tool_memory.cooccurrence_window_s}",
            f"min_frequency = {self.tool_memory.min_frequency}",
            f"max_buffer_lines = {self.tool_memory.max_buffer_lines}",
            f"process_batch_size = {self.tool_memory.process_batch_size}",
            "",
            "# Mem0 auto-sync settings",
            "[mem0_sync]",
            f"enabled = {'true' if self.mem0_sync.enabled else 'false'}",
            f"self_hosted = {'true' if self.mem0_sync.self_hosted else 'false'}",
            f'user_id = "{_sanitize_sync_id(self.mem0_sync.user_id)}"',
            f'agent_id = "{_sanitize_sync_id(self.mem0_sync.agent_id)}"',
            f"cooldown_minutes = {self.mem0_sync.cooldown_minutes}",
            f"sync_on_startup = {'true' if self.mem0_sync.sync_on_startup else 'false'}",
        ]

        if self.mem0_sync.limit is not None:
            lines.append(f"limit = {self.mem0_sync.limit}")

        lines += [
            "",
            "# Multi-device sync settings",
            "[sync]",
            f"enabled = {'true' if self.sync.enabled else 'false'}",
            f'hub_url = "{self.sync.hub_url}"',
            f'api_key = "{_sanitize_toml_str(self.sync.api_key)}"',
            f"auto_sync = {'true' if self.sync.auto_sync else 'false'}",
            f"sync_interval_seconds = {self.sync.sync_interval_seconds}",
            f'conflict_strategy = "{self.sync.conflict_strategy}"',
            "",
            "# Telegram backup integration",
            "# Bot token: set SURREAL_MEMORY_TELEGRAM_BOT_TOKEN env var (never stored here)",
            "[telegram]",
            f"enabled = {'true' if self.telegram.enabled else 'false'}",
            f"chat_ids = [{', '.join(repr(cid) for cid in self.telegram.chat_ids)}]",
            f"max_file_size_mb = {self.telegram.max_file_size_mb}",
            f"backup_on_consolidation = {'true' if self.telegram.backup_on_consolidation else 'false'}",
            "",
            "# License tier",
            "[license]",
            f'tier = "{_sanitize_toml_str(self.license.tier)}"',
            f'activated_at = "{_sanitize_iso_datetime(self.license.activated_at)}"',
            f'expires_at = "{_sanitize_iso_datetime(self.license.expires_at)}"',
            "",
            "# Cross-encoder reranking (optional, requires pip install surreal-memory[reranker])",
            "[reranker]",
            f"enabled = {'true' if self.reranker.enabled else 'false'}",
            f'model_name = "{_sanitize_toml_str(self.reranker.model_name)}"',
            f"blend_weight = {self.reranker.blend_weight}",
            f"min_score = {self.reranker.min_score}",
            f"max_candidates = {self.reranker.max_candidates}",
            "# OpenAI-compatible /rerank base URL (e.g. llamastash); empty = in-process CrossEncoder",
            f'endpoint = "{_sanitize_toml_url(self.reranker.endpoint)}"',
            "",
            "# Auto-tier promotion/demotion (Pro feature)",
            "[tiers]",
            f"auto_enabled = {'true' if self.tiers.auto_enabled else 'false'}",
            f"promote_threshold = {self.tiers.promote_threshold}",
            f"demote_inactive_days = {self.tiers.demote_inactive_days}",
            f"cold_archive_days = {self.tiers.cold_archive_days}",
            f"max_hot_memories = {self.tiers.max_hot_memories}",
            "",
            "# File watcher auto-ingestion",
            "[watcher]",
            f"enabled = {'true' if self.watcher.enabled else 'false'}",
            f"paths = [{', '.join(repr(p) for p in self.watcher.paths)}]",
            f"extensions = [{', '.join(repr(e) for e in self.watcher.extensions)}]",
            f"ignore_patterns = [{', '.join(repr(p) for p in self.watcher.ignore_patterns)}]",
            f"debounce_seconds = {self.watcher.debounce_seconds}",
            f"max_file_size_mb = {self.watcher.max_file_size_mb}",
            f"max_watched_dirs = {self.watcher.max_watched_dirs}",
            f'memory_type = "{_sanitize_toml_str(self.watcher.memory_type)}"',
            f'domain_tag = "{_sanitize_toml_str(self.watcher.domain_tag)}"',
            "",
            "# MCP tool tier (minimal/standard/full)",
            "[tool_tier]",
            f'tier = "{self.tool_tier.tier}"',
            "",
            "# MCP response compaction",
            "[response]",
            f"compact_mode = {'true' if self.response.compact_mode else 'false'}",
            f"max_list_items = {self.response.max_list_items}",
            f"strip_hints = {'true' if self.response.strip_hints else 'false'}",
            f"content_preview_length = {self.response.content_preview_length}",
            f"auto_compact_threshold = {self.response.auto_compact_threshold}",
            "",
            "# Retrieval-trace telemetry (opt-in; off by default)",
            "[trace]",
            f"enabled = {'true' if self.trace.enabled else 'false'}",
            f"sample_rate = {self.trace.sample_rate}",
            f"retention_days = {self.trace.retention_days}",
            f"max_traces = {self.trace.max_traces}",
            "",
            *self._reasoning_toml_lines(),
            "",
            "# CLI preferences",
            "[cli]",
            f"json_output = {'true' if self.json_output else 'false'}",
            f"default_max_tokens = {self.default_max_tokens}",
        ]

        if self.default_depth is not None:
            lines.append(f"default_depth = {self.default_depth}")

        # Single source of truth: assert the [section] headers produced above
        # match SECTION_NAMES exactly. Keeps `smem doctor`'s freshness check
        # (which consumes SECTION_NAMES) in lockstep with save() — adding or
        # removing a section in one place but not the other raises here, in
        # dev/CI, instead of surfacing as a user-facing warning that --fix can
        # never satisfy. Dotted subtable headers (e.g. reasoning_training.x)
        # collapse to their top-level section.
        emitted_sections = {
            line.split("]")[0][1:].split(".")[0]
            for line in lines
            if line.startswith("[") and not line.startswith("[[")
        }
        declared = set(self.SECTION_NAMES)
        if emitted_sections != declared:
            raise RuntimeError(
                "UnifiedConfig.save() section drift: "
                f"emitted but undeclared={sorted(emitted_sections - declared)}, "
                f"declared but unemitted={sorted(declared - emitted_sections)}"
            )

        # Atomic write: write to temp file, then rename
        content = "\n".join(lines) + "\n"
        fd, tmp_path = tempfile.mkstemp(dir=str(self.data_dir), suffix=".toml.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(config_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    @property
    def brains_dir(self) -> Path:
        """Get directory where brain databases are stored."""
        return self.data_dir / "brains"

    @property
    def config_path(self) -> Path:
        """Get path to config file."""
        return self.data_dir / "config.toml"

    def get_brain_db_path(self, brain_name: str | None = None) -> Path:
        """Get path to brain SQLite database.

        Args:
            brain_name: Brain name, or use current_brain if None

        Returns:
            Path to SQLite database file

        Raises:
            ValueError: If brain name contains invalid characters
        """
        name = brain_name or self.current_brain
        if not _BRAIN_NAME_PATTERN.match(name):
            raise ValueError(
                "Invalid brain name: must contain only "
                "alphanumeric characters, hyphens, underscores, or dots"
            )
        db_path = (self.brains_dir / f"{name}.db").resolve()
        if not db_path.is_relative_to(self.brains_dir.resolve()):
            raise ValueError("Invalid brain name: path traversal detected")
        return db_path

    def list_brains(self) -> list[str]:
        """List available brains by inspecting local SQLite test-fixture files.

        Production runs on SurrealDB and should query the server directly. This
        helper only sees the local on-disk test-fixture databases.
        """
        if not self.brains_dir.exists():
            return []
        return sorted(p.stem for p in self.brains_dir.glob("*.db"))

    def switch_brain(self, brain_name: str) -> None:
        """Switch to a different brain and save config."""
        if not _BRAIN_NAME_PATTERN.match(brain_name):
            raise ValueError(
                "Invalid brain name: must contain only "
                "alphanumeric characters, hyphens, underscores, or dots"
            )
        self.current_brain = brain_name
        self.save()


# Singleton instance for easy access
_config: UnifiedConfig | None = None

# Cached storage instances keyed by db_path string
_storage_cache: dict[str, NeuralStorage] = {}
_storage_lock: asyncio.Lock | None = None


def _get_storage_lock() -> asyncio.Lock:
    """Lazy-init asyncio.Lock (must be created inside a running event loop)."""
    global _storage_lock
    if _storage_lock is None:
        _storage_lock = asyncio.Lock()
    return _storage_lock


def get_config(reload: bool = False) -> UnifiedConfig:
    """Get the unified configuration (singleton).

    Args:
        reload: Force reload from disk

    Returns:
        UnifiedConfig instance
    """
    global _config
    if _config is None or reload:
        _config = UnifiedConfig.load()
    return _config


def set_config(config: UnifiedConfig) -> None:
    """Replace the in-memory config singleton.

    Use after mutating config (e.g. license activation) so that
    REST endpoints via get_config() see the updated values.
    """
    global _config
    _config = config


def _read_legacy_brain(data_dir: Path) -> str | None:
    """Read current_brain from legacy config.json during first-time migration.

    Checks both the given data_dir and the legacy ~/.surreal-memory/ location
    for an existing config.json with a non-default brain selection.

    Returns:
        The brain name string, or ``None`` if no legacy config found
        or it uses the default brain.
    """
    # Check locations in priority order
    candidates = [data_dir / "config.json"]
    legacy_dir = Path.home() / ".surreal-memory"
    if legacy_dir != data_dir:
        candidates.append(legacy_dir / "config.json")

    for config_file in candidates:
        if not config_file.is_file():
            continue
        try:
            with open(config_file, encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("current_brain")
            if isinstance(name, str) and name != "default" and _BRAIN_NAME_PATTERN.match(name):
                return name
        except Exception:
            logger.warning(
                "Found legacy config %s but could not read it", config_file, exc_info=True
            )
            continue
    return None


def _read_current_brain_from_toml() -> str | None:
    """Read just the current_brain value from config.toml on disk.

    This is a lightweight read used by ``get_shared_storage`` to detect
    brain switches made by the CLI (which writes to config.toml via
    ``_sync_brain_to_toml``).  It avoids a full config reload.

    Returns:
        The current_brain string, or ``None`` if the file is missing
        or cannot be parsed.
    """
    toml_path = get_surrealmemory_dir() / "config.toml"
    if not toml_path.exists():
        return None
    try:
        import tomllib

        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        name = data.get("current_brain")
        if isinstance(name, str) and _BRAIN_NAME_PATTERN.match(name):
            return name
    except Exception:
        logger.debug("Could not read current_brain from config.toml", exc_info=True)
    return None


_missing_surreal_pass_warned = False


def _warn_missing_surreal_pass() -> None:
    """Emit a one-time warning when storage=surrealdb and SURREALDB_PASS is unset.

    A missing password means the server will attempt to authenticate with the
    built-in default, which will fail if the DB was started with a different
    password. Surface this before the connection attempt so the error is
    actionable.
    """
    global _missing_surreal_pass_warned
    if _missing_surreal_pass_warned:
        return
    if os.environ.get("SURREAL_MEMORY_STORAGE") != "surrealdb":
        return
    if os.environ.get("SURREALDB_PASS"):
        return
    _missing_surreal_pass_warned = True
    logging.getLogger(__name__).warning(
        "SURREALDB_PASS is not set and storage=surrealdb is active. "
        "The server will use the default password ('surrealmemory'). "
        "If your SurrealDB uses a different password, set SURREALDB_PASS "
        "in your MCP client config or run `smem doctor --fix`."
    )


async def get_shared_storage(brain_name: str | None = None) -> NeuralStorage:
    """Get storage for shared brain access.

    This is the main entry point for getting storage that works
    across CLI, MCP, and other tools. Storage instances are cached
    to avoid connection leaks.

    Respects config.storage_backend: "surrealdb" (default) or "memory".

    Args:
        brain_name: Brain name, or use config's current_brain if None

    Returns:
        NeuralStorage instance, initialized and ready to use
    """
    config = get_config()

    # When no explicit brain is requested, resolve from env var or disk.
    #
    # Priority: env var > config.toml > in-memory config
    #
    # IMPORTANT: When SURREAL_MEMORY_BRAIN / SURREAL_MEMORY_BRAIN is set, we use it
    # directly WITHOUT mutating config.current_brain. This ensures
    # process-level isolation for multi-agent setups where each Claude
    # Code session spawns its own MCP server process with a different
    # env var. Mutating the shared config object would cause cross-brain
    # contamination if the config singleton is ever shared.
    if brain_name is None:
        env_brain = os.environ.get("SURREAL_MEMORY_BRAIN")
        if env_brain:
            name = env_brain
        else:
            disk_brain = _read_current_brain_from_toml()
            if disk_brain is not None and disk_brain != config.current_brain:
                logger = logging.getLogger(__name__)
                logger.info("Brain changed on disk: %s → %s", config.current_brain, disk_brain)
                config.current_brain = disk_brain
            name = config.current_brain
    else:
        name = brain_name

    # SurrealDB backend
    if config.storage_backend == "surrealdb":
        return await _get_surrealdb_storage(config, name)

    # Non-persistent backend: everything lives in this process and is gone when
    # it exits. Opt-in only, for trying the tool out without running a database.
    if config.storage_backend == "memory":
        return await _get_memory_storage(config, name)

    # Unreachable via normal config construction (_validate_storage_backend only
    # ever produces "surrealdb"/"memory"), but a caller could bypass validation
    # by constructing UnifiedConfig directly. Fail loudly rather than silently
    # routing memories somewhere the caller didn't expect.
    raise ValueError(SQLITE_REMOVED_ERROR)


async def list_available_brains() -> list[str]:
    """List brain names from the active storage backend.

    UnifiedConfig.list_brains only inspects local sqlite fixture files, so on
    the SurrealDB backend it returns nothing and the dashboard shows zero
    brains even when the store holds data. Enumerate the SurrealDB brain table
    in that case, falling back to the sqlite-file listing otherwise.
    """
    config = get_config()
    if config.storage_backend == "surrealdb":
        try:
            storage = await get_shared_storage()
            # list_brain_names is defined on the SurrealDB backend; the base
            # NeuralStorage interface doesn't declare it (only the SurrealDB
            # path reaches here, guarded by storage_backend == "surrealdb").
            names: list[str] = await storage.list_brain_names()  # type: ignore[attr-defined]
            if names:
                return names
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to enumerate brains from SurrealDB", exc_info=True
            )
    return config.list_brains()


async def _migrate_brain_runtime_config(
    storage: NeuralStorage,
    brain: Any,
    config: UnifiedConfig,
) -> None:
    """Layer ``config.toml [brain]`` extras over an already-stored brain.config.

    Brains created on older versions store BrainConfig fields that existed at
    creation time; newer fields fall back to BrainConfig defaults on load, so a
    ``[brain]`` knob set in ``config.toml`` is silently ignored (issue #168).
    This applies any ``extras`` keys from ``BrainSettings`` to the stored brain
    config and persists the patched brain. Only ``extras`` keys are applied — the
    explicit fields are left untouched because legacy brains may carry per-brain
    customizations there. Failures are logged and swallowed — config migration
    must never break recall.
    """
    try:
        # NOTE: reranker config is intentionally NOT layered onto the brain here.
        # It is deployment/runtime config, read from the app config at recall time
        # (see ReflexPipeline). Persisting it on a shared brain let a reranker-off
        # client (e.g. the web-UI container) flip the flag for everyone on
        # connect — the reranker-flip bug. Only [brain] extras are migrated.
        overrides = config.brain.runtime_overrides()
        if not overrides:
            return

        from surreal_memory.utils.timeutils import utcnow

        current = {f.name: getattr(brain.config, f.name) for f in dataclasses.fields(brain.config)}
        diff = {k: v for k, v in overrides.items() if current.get(k) != v}
        if not diff:
            return

        patched_config = dataclasses.replace(brain.config, **diff)
        patched_brain = dataclasses.replace(brain, config=patched_config, updated_at=utcnow())
        await storage.save_brain(patched_brain)
        logger.info(
            "Brain %r config migrated from config.toml [brain] extras: %s",
            brain.name,
            sorted(diff.keys()),
        )
    except Exception:
        logger.debug(
            "Brain runtime config migration failed for %r (non-fatal)",
            getattr(brain, "name", "?"),
            exc_info=True,
        )


_memory_backend_warned = False


async def _get_memory_storage(config: UnifiedConfig, name: str) -> NeuralStorage:
    """Create or return the cached InMemoryStorage for *name*.

    Nothing is written to disk: closing the process discards every memory. This
    exists so `pip install surreal-memory` can be tried without provisioning a
    database, and so container images can run without a datastore volume.
    """
    global _memory_backend_warned
    if not _memory_backend_warned:
        _memory_backend_warned = True
        logger.warning(
            "Using the in-memory storage backend — NOTHING IS PERSISTED. "
            "Every memory is lost when this process exits. Set "
            "SURREAL_MEMORY_STORAGE=surrealdb for a durable store."
        )

    from surreal_memory.core.brain import Brain, BrainConfig
    from surreal_memory.storage.memory_store import InMemoryStorage

    cache_key = f"memory:{name}"
    lock = _get_storage_lock()
    async with lock:
        cached = _storage_cache.get(cache_key)
        if cached is not None:
            cached.set_brain(name)
            return cached

        # No initialize(): the in-memory store has no connection to open.
        storage = InMemoryStorage()

        brain = await storage.get_brain(name)
        if brain is None:
            brain = await storage.find_brain_by_name(name)
        if brain is None:
            brain_config = BrainConfig(
                **config.brain.to_brain_config_kwargs(config.embedding, config.reranker)
            )
            brain = Brain.create(name=name, config=brain_config, brain_id=name)
            await storage.save_brain(brain)

        storage.set_brain(name)
        _storage_cache[cache_key] = storage
        return storage


_surrealdb_storage: NeuralStorage | None = None


def _resolve_embedding_dim(config: UnifiedConfig) -> int:
    """Resolve the embedding vector dimension the SurrealDB HNSW index must match.

    Priority: explicit config.embedding.dimension (e.g. a local bge-m3 server →
    1024) > known Gemini model dim > 3072 default.
    """
    if config.embedding.enabled:
        if config.embedding.dimension and config.embedding.dimension > 0:
            return int(config.embedding.dimension)
        if config.embedding.model:
            from surreal_memory.engine.embedding.gemini_embedding import _MODEL_DIMENSIONS

            return _MODEL_DIMENSIONS.get(config.embedding.model, 3072)
    return 3072  # Gemini 2.0 default


async def _get_surrealdb_storage(config: UnifiedConfig, name: str) -> NeuralStorage:
    """Create or return cached SurrealDBStorage."""
    global _surrealdb_storage

    from surreal_memory.core.brain import Brain
    from surreal_memory.storage.surrealdb import SurrealDBStorage
    from surreal_memory.storage.surrealdb.connection import SurrealSettings

    # Reuse the cached instance only if its connection is still open. A prior
    # caller may have close()d it (which nulls _conn but leaves this module-level
    # reference set); returning it as-is would raise "not initialized" on first
    # use. Mirror the SQLite cache's liveness check and reinitialize otherwise.
    if _surrealdb_storage is not None and getattr(_surrealdb_storage, "_conn", None) is not None:
        _surrealdb_storage.set_brain(name)
        return _surrealdb_storage

    # Resolve the embedding vector dimension that the SurrealDB HNSW index must match.
    emb_dim = _resolve_embedding_dim(config)

    _warn_missing_surreal_pass()

    # Use SurrealSettings.from_env() as single source of truth — no duplicate
    # defaults here. SurrealDBStorage.__init__ also calls from_env() as fallback,
    # but we pass values explicitly so the caller's env is captured at this point.
    settings = SurrealSettings.from_env()
    storage = SurrealDBStorage(
        url=settings.url,
        namespace=settings.namespace,
        database=settings.database,
        user=settings.user,
        password=settings.password,
        embedding_dim=emb_dim,
    )
    await storage.initialize()

    # Ensure brain exists (idempotent).
    # Try by id first (normal case: brain_id == name), then fall back to a
    # name lookup (handles brains with UUID ids from older versions). Only
    # create when neither resolves, and create with a deterministic
    # brain_id == name so a re-run cannot insert a fresh UUID row. Without
    # both the name fallback and the explicit brain_id, every process start
    # minted a new brain:<uuid> row, leaking hundreds of duplicates.
    brain = await storage.get_brain(name)
    if brain is None:
        brain = await storage.find_brain_by_name(name)
    if brain is None:
        brain = Brain.create(name, brain_id=name)
        await storage.save_brain(brain)
    else:
        await _migrate_brain_runtime_config(storage, brain, config)

    # Brains are addressed by name in this store (neurons carry brain_id as a
    # plain string), so the brain context stays the name — never brain.id,
    # which for legacy rows is a UUID and would orphan existing neurons.
    storage.set_brain(name)
    _surrealdb_storage = storage
    logger.info("SurrealDB storage initialized for brain '%s'", name)
    return storage


async def create_isolated_storage(brain_name: str | None = None) -> NeuralStorage:
    """Create a FRESH, non-cached storage bound to *brain_name*.

    Unlike get_shared_storage, this never returns the module-global SurrealDB
    singleton, so a long-running background job (e.g. reasoning mining) can hold
    its own brain context without a concurrently-served request's set_brain()
    racing it and cross-writing into the wrong brain. The caller MUST close() the
    returned storage. SQLite already caches per brain-file (each brain a separate
    instance) and the in-memory backend is test-only, so those return the shared
    instance — only the SurrealDB global singleton needs true isolation.
    """
    config = get_config()
    name = brain_name or config.current_brain

    if config.storage_backend == "surrealdb":
        from surreal_memory.core.brain import Brain
        from surreal_memory.storage.surrealdb import SurrealDBStorage
        from surreal_memory.storage.surrealdb.connection import SurrealSettings

        _warn_missing_surreal_pass()
        settings = SurrealSettings.from_env()
        storage = SurrealDBStorage(
            url=settings.url,
            namespace=settings.namespace,
            database=settings.database,
            user=settings.user,
            password=settings.password,
            embedding_dim=_resolve_embedding_dim(config),
        )
        await storage.initialize()
        brain = await storage.get_brain(name)
        if brain is None:
            brain = await storage.find_brain_by_name(name)
        if brain is None:
            brain = Brain.create(name, brain_id=name)
            await storage.save_brain(brain)
        storage.set_brain(name)
        return storage

    # SQLite (per-brain cached) and in-memory (test-only) are not shared across
    # brains in a racy way; reuse the shared instance.
    return await get_shared_storage(name)
