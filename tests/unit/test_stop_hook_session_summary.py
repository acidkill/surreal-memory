"""The Stop hook's session-summary fallback must stay OFF unless asked for.

Why this file exists. ``_extract_session_summary()`` does not summarise: it takes
the last ~10 non-trivial lines of the transcript verbatim, joins them and prefixes
"Session activity: ". On prose that reads fine; on a real agent transcript it emits
harness markers and half-sentences. Measured on the production brain over 24 h,
96 of 158 write-gate rejections (61 %) came from this one path -- rows such as
``Session activity: <task-notification>`` and ``Session activity: Writing objects:``.

On a stock installation the gate is off (``WriteGateConfig`` defaults to
``enabled=False`` / ``mode="off"``), so those rows are not discarded -- they are
stored as ``CONTEXT`` memories. Where the gate is enabled it rejects them and the
fallback reliably produced telemetry noise plus the occasional truncated TODO
that scraped past the threshold and came back as session-start context. Hence the
default flip. These tests pin BOTH directions, because a default that nothing
asserts is a default that silently drifts back:

* off (the new default) -- a pattern-free transcript stores nothing, and the
  returned message says so distinguishably;
* on (explicit opt-in) -- the old behaviour still works, so #80's regression
  suite and anyone relying on it keep a supported path.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from surreal_memory.hooks import stop as stop_hook

# No "Decision:"/"Error:"/"TODO:" markers anywhere, so analyze_text_for_memories()
# finds nothing and the summary fallback is the only path that could save anything.
# Lines are >15 chars so _extract_session_summary() would produce a non-empty
# candidate if it ran -- otherwise "saved 0" would prove nothing.
_TEXT_NO_PATTERNS = (
    "Here is an overview of the staging cluster network topology for reference purposes.\n\n"
    "The quarterly release checklist includes twelve items across three teams total."
)


@pytest.fixture(autouse=True)
def _isolate_storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same pin as test_hook_capture_idempotency: a dev shell that exports
    SURREAL_MEMORY_STORAGE=surrealdb would otherwise aim these writes at the live
    brain instead of the throwaway in-memory one."""
    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "memory")
    monkeypatch.delenv("SURREAL_MEMORY_DIR", raising=False)
    monkeypatch.delenv("SURREALDB_URL", raising=False)
    monkeypatch.delenv("SURREALDB_PASS", raising=False)


@pytest.fixture
async def isolated_brain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", "/tmp")
    brain_name = "summ" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("SURREAL_MEMORY_BRAIN", brain_name)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-" + uuid.uuid4().hex[:12])

    from surreal_memory.unified_config import get_config

    get_config(reload=True)
    return brain_name


class TestSessionSummaryDefault:
    def test_default_is_off(self) -> None:
        from surreal_memory.unified_config import AutoConfig

        assert AutoConfig().capture_session_summary is False

    def test_absent_key_loads_as_off(self) -> None:
        """A config.toml written before this flag existed must not switch the
        fallback back on just because the key is missing."""
        from surreal_memory.unified_config import AutoConfig

        assert AutoConfig.from_dict({"enabled": True}).capture_session_summary is False

    def test_explicit_true_survives_round_trip(self) -> None:
        """to_dict/from_dict must carry the opt-in, or `smem doctor --fix` (which
        re-saves the whole config) would quietly reset an operator's choice."""
        from surreal_memory.unified_config import AutoConfig

        enabled = AutoConfig(capture_session_summary=True)
        assert AutoConfig.from_dict(enabled.to_dict()).capture_session_summary is True

    def test_save_emits_the_key(self, tmp_path: Path) -> None:
        """The flag must reach config.toml on disk. If save() omitted it, the
        value would live only in memory and vanish on the next load."""
        import tomllib

        from surreal_memory.unified_config import UnifiedConfig

        config = UnifiedConfig(data_dir=tmp_path)
        config.auto.capture_session_summary = True
        config.save()

        raw = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
        assert raw["auto"]["capture_session_summary"] is True


@pytest.mark.asyncio
class TestSessionSummaryBehaviour:
    async def test_off_stores_nothing_for_pattern_free_text(self, isolated_brain: str) -> None:
        result = await stop_hook.capture_text(_TEXT_NO_PATTERNS, project_name=None)

        assert result["saved"] == 0, (
            "with the fallback off, a transcript holding no memory patterns must "
            f"store nothing; got {result['memories']!r}"
        )
        message = result["message"].lower()
        assert "no memorable content" in message, (
            f"saved=0 must name the reason distinguishably, got: {result['message']!r}"
        )
        # Not a gate rejection and not a duplicate skip: nothing was ever offered.
        assert "gate" not in message and "rejected" not in message
        assert "duplicate" not in message and "idempot" not in message

    async def test_on_still_stores_the_summary(self, isolated_brain: str) -> None:
        """Positive control. Without this, `saved == 0` above would also pass if
        the hook were broken for an unrelated reason."""
        from surreal_memory.unified_config import get_config

        get_config().auto.capture_session_summary = True

        result = await stop_hook.capture_text(_TEXT_NO_PATTERNS, project_name=None)

        assert result["saved"] == 1, (
            f"opt-in must restore the old behaviour, got {result['message']!r}"
        )
        assert result["memories"][0].startswith("Session activity:")
