"""Tests for engine/reasoning_injection.py — model resolution + prompt block.

Uses synthetic transcripts / settings.json (HOME redirected to tmp) and
InMemoryStorage seeded with pattern fibers. Covers the resolve_active_model
fallback chain, injection_map glob/default matching, per-category + max-patterns
+ char-budget selection, markdown format, and the session idempotency markers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.reasoning_injection import (
    already_injected,
    build_injection_context,
    get_reasoning_context,
    mark_injected,
    resolve_active_model,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

BRAIN = "b1"
_ENV_MODEL_VARS = ("SMEM_REASONING_TARGET_MODEL", "ANTHROPIC_MODEL")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect HOME to tmp (no settings.json) and clear model env vars."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in _ENV_MODEL_VARS:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _ucfg(tmp_path: Path, **rt_kw: object) -> UnifiedConfig:
    base: dict[str, object] = {
        "injection_enabled": True,
        "injection_max_patterns": 5,
        "injection_max_chars": 4000,
    }
    base.update(rt_kw)
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(**base),  # type: ignore[arg-type]
    )


async def _add_pattern(
    storage: InMemoryStorage,
    model: str,
    category: str,
    title: str,
    strategy: str,
    confidence: float,
    frequency: int,
) -> None:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=title)
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=title,
        tags=set(),
        metadata={
            "_reasoning_pattern": True,
            "_source_model": model,
            "_reasoning_category": category,
            "_reasoning_title": title,
            "_reasoning_strategy": strategy,
            "_reasoning_confidence": confidence,
            "_reasoning_frequency": frequency,
            "_reasoning_signature": title,
        },
    )
    await storage.add_fiber(fiber)


# ── resolve_active_model chain ───────────────────────────────────────────────


def test_resolve_from_payload_model(clean_env: Path) -> None:
    assert resolve_active_model({"model": "claude-sonnet-5-20250101"}) == "claude-sonnet-5"


def test_resolve_from_transcript_tail(clean_env: Path) -> None:
    # Transcripts are only read from under ~/.claude (spoof guard). HOME is
    # redirected to clean_env, so write the transcript there.
    tp = clean_env / ".claude" / "projects" / "x" / "t.jsonl"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(
        json.dumps({"type": "user", "message": {"content": "hi"}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {"model": "claude-haiku-4-5-20251001"}})
        + "\n",
        encoding="utf-8",
    )
    assert resolve_active_model({"transcript_path": str(tp)}) == "claude-haiku-4-5"


def test_resolve_transcript_outside_claude_rejected(clean_env: Path) -> None:
    # A transcript_path outside ~/.claude is untrusted (hook stdin is attacker-
    # controllable) and ignored; resolution falls through to "default".
    outside = clean_env / "outside.jsonl"
    outside.write_text(
        json.dumps({"type": "assistant", "message": {"model": "claude-opus-4-8"}}) + "\n",
        encoding="utf-8",
    )
    assert resolve_active_model({"transcript_path": str(outside)}) == "default"


def test_resolve_transcript_symlink_escape_rejected(clean_env: Path) -> None:
    # A symlink UNDER ~/.claude that points outside it must also be rejected:
    # resolve() canonicalizes through the symlink before the containment check,
    # so this can't be bypassed with a symlink (only a literal path prefix check
    # would be fooled). Guards against a future refactor to os.path.abspath.
    outside = clean_env / "real.jsonl"
    outside.write_text(
        json.dumps({"type": "assistant", "message": {"model": "claude-opus-4-8"}}) + "\n",
        encoding="utf-8",
    )
    claude_dir = clean_env / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    link = claude_dir / "sneaky.jsonl"
    link.symlink_to(outside)
    assert resolve_active_model({"transcript_path": str(link)}) == "default"


def test_resolve_from_env(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMEM_REASONING_TARGET_MODEL", "claude-opus-4-8")
    assert resolve_active_model({}) == "claude-opus-4-8"


def test_resolve_from_env_alias(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "sonnet")
    assert resolve_active_model({}) == "claude-sonnet-5"


def test_resolve_from_settings_alias(clean_env: Path) -> None:
    claude_dir = clean_env / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(json.dumps({"model": "opusplan"}), encoding="utf-8")
    assert resolve_active_model({}) == "claude-opus-4-8"


def test_resolve_default_when_nothing_available(clean_env: Path) -> None:
    assert resolve_active_model({}) == "default"


def test_resolve_precedence_payload_over_env(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMEM_REASONING_TARGET_MODEL", "claude-haiku-4-5")
    assert resolve_active_model({"model": "claude-opus-4-8"}) == "claude-opus-4-8"


# ── build_injection_context ──────────────────────────────────────────────────


async def test_build_block_renders_ranked(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage, "claude-fable-5", "planning", "planning: plan", "plan -> steps", 0.8, 2
    )
    await _add_pattern(
        storage, "claude-fable-5", "debugging", "debugging: verify", "verify -> check", 1.0, 3
    )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))

    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    assert block.startswith("## Reasoning strategies (learned from claude-fable-5)")
    assert "debugging: verify" in block
    assert "planning: plan" in block
    # Higher confidence*frequency ranks first.
    assert block.index("debugging: verify") < block.index("planning: plan")


async def test_injection_disabled_returns_empty(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "t", "s", 1.0, 3)
    cfg = _ucfg(
        tmp_path, injection_enabled=False, injection_map=(("claude-opus-*", "claude-fable-5"),)
    )
    assert await build_injection_context(storage, "claude-opus-4-8", cfg) == ""


async def test_no_map_match_returns_empty(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "t", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-haiku-*", "claude-fable-5"),))
    assert await build_injection_context(storage, "claude-opus-4-8", cfg) == ""


async def test_injection_map_default_key(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: x", "s", 1.0, 3)
    cfg = _ucfg(
        tmp_path,
        injection_map=(("claude-haiku-*", "nope"), ("default", "claude-fable-5")),
    )
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    assert "learned from claude-fable-5" in block


async def test_source_without_patterns_returns_empty(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "t", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-sonnet-5"),))
    assert await build_injection_context(storage, "claude-opus-4-8", cfg) == ""


async def test_max_two_per_category(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    for i in range(4):
        await _add_pattern(
            storage, "claude-fable-5", "debugging", f"debugging: {i}", "s", 1.0, 3 - i
        )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    # Only 2 of the 4 debugging patterns are injected.
    assert block.count("debugging:") == 2


async def test_injection_max_patterns_cap(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: a", "s", 1.0, 3)
    await _add_pattern(storage, "claude-fable-5", "planning", "planning: b", "s", 0.9, 3)
    cfg = _ucfg(
        tmp_path, injection_max_patterns=1, injection_map=(("claude-opus-*", "claude-fable-5"),)
    )
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    assert "1. **" in block
    assert "2. **" not in block


async def test_char_budget_limits_entries(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    long_strategy = "x" * 500
    await _add_pattern(
        storage, "claude-fable-5", "debugging", "debugging: a", long_strategy, 1.0, 3
    )
    await _add_pattern(storage, "claude-fable-5", "planning", "planning: b", long_strategy, 0.9, 3)
    cfg = _ucfg(
        tmp_path, injection_max_chars=120, injection_map=(("claude-opus-*", "claude-fable-5"),)
    )
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    # First entry always included; the second exceeds the tiny budget.
    assert "1. **" in block
    assert "2. **" not in block


async def test_build_block_warns_on_fetch_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # When the pattern-fiber fetch hits its ceiling, a model's patterns could be
    # truncated out; that must be a visible warning, not a silent empty injection.
    from surreal_memory.engine import reasoning_injection as ri

    monkeypatch.setattr(ri, "_PATTERN_FETCH_LIMIT", 2)
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: a", "s", 1.0, 3)
    await _add_pattern(storage, "claude-fable-5", "planning", "planning: b", "s", 0.9, 2)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))

    with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.reasoning_injection"):
        block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    assert "ceiling" in caplog.text
    assert block  # still renders from whatever was fetched


# ── get_reasoning_context (shared hook orchestrator) ─────────────────────────


async def test_get_reasoning_context_disabled_returns_empty(
    clean_env: Path, tmp_path: Path
) -> None:
    cfg = _ucfg(tmp_path, injection_enabled=False)
    with patch("surreal_memory.unified_config.get_config", return_value=cfg):
        assert await get_reasoning_context({"session_id": "s-disabled"}) == ""


async def test_get_reasoning_context_builds_and_marks_session(
    clean_env: Path, tmp_path: Path
) -> None:
    storage = InMemoryStorage()
    storage.set_brain("default")
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: verify", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch(
            "surreal_memory.unified_config.get_shared_storage",
            new=AsyncMock(return_value=storage),
        ),
    ):
        block = await get_reasoning_context({"session_id": "s-happy", "model": "claude-opus-4-8"})

    assert "learned from claude-fable-5" in block
    # Session marker set → the sibling UserPromptSubmit hook won't re-inject.
    assert already_injected("s-happy") is True


async def test_get_reasoning_context_skips_when_already_injected(
    clean_env: Path, tmp_path: Path
) -> None:
    mark_injected("s-dup")  # e.g. SessionStart already injected this session
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))

    async def _must_not_open(_name: str) -> object:
        raise AssertionError("storage must not open when the session is already injected")

    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch("surreal_memory.unified_config.get_shared_storage", new=_must_not_open),
    ):
        result = await get_reasoning_context({"session_id": "s-dup", "model": "claude-opus-4-8"})

    assert result == ""


# ── session idempotency markers ──────────────────────────────────────────────


def test_marker_roundtrip(clean_env: Path) -> None:
    assert already_injected("sess-1") is False
    mark_injected("sess-1")
    assert already_injected("sess-1") is True
    # Empty session id is a no-op and never "already injected".
    assert already_injected("") is False
    mark_injected("")


def test_marker_sanitizes_session_id(clean_env: Path) -> None:
    mark_injected("../evil/../id")
    marker_root = clean_env / ".surrealmemory" / "reasoning_injected"
    assert marker_root.is_dir()
    # No traversal escape — every marker stays directly under the marker root.
    assert all(p.parent == marker_root for p in marker_root.iterdir())
    assert already_injected("../evil/../id") is True


def test_marker_dir_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # SURREAL_MEMORY_DIR relocates the whole data dir; markers must follow it
    # (matches hooks/post_tool_use._get_data_dir) rather than pinning ~/.surrealmemory.
    data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(data_dir))
    mark_injected("sess-env")
    assert (data_dir / "reasoning_injected" / "sess-env").exists()
    assert already_injected("sess-env") is True
