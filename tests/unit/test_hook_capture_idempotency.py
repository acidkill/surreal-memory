"""Behavioral regression tests for auto-capture hook idempotency (upstream #80).

Reproduces the defect confirmed manually in ~/expertP/smem-idempotency-80/
CHECKPOINTS/F1: the Stop and PreCompact hooks re-encode a session summary
(or fragment) every time they are invoked for the same session, even when
the effectively-captured text has not changed since the previous call.

Every test here runs against a real, isolated SQLite brain (a fresh tmp_path
HOME) -- never the shared prod brain. Each test uses a unique brain name and
session id to avoid cross-test collisions in unified_config's process-wide
_config / _storage_cache singletons.

Caveat: only HOME/SURREAL_MEMORY_BRAIN are overridden, not
SURREAL_MEMORY_STORAGE. If a developer's shell already has
SURREAL_MEMORY_STORAGE=surrealdb exported (for day-to-day dashboard use),
these tests resolve to the live SurrealDB backend instead of SQLite;
_cleanup_test_brain() below deletes that throwaway "idem*" brain by id in
that case so the shared DB never accumulates leftovers.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from surreal_memory.hooks import pre_compact as pre_compact_hook
from surreal_memory.hooks import stop as stop_hook

pytestmark = pytest.mark.asyncio

_TEXT_A = (
    "Here is an overview of the staging cluster network topology for reference purposes.\n\n"
    "The quarterly release checklist includes twelve items across three teams total."
)
_TEXT_B = (
    "A completely different summary about the greenhouse seedling rotation schedule.\n\n"
    "The compost bin delivery arrived near the eastern fence line yesterday afternoon."
)
# Long enough overall to pass the input firewall's 30-char floor, but every
# line is <=15 chars so _extract_session_summary()'s per-line filter drops
# all of them (summary=""), and none of the words match any memory pattern
# -- guarantees analyze_text_for_memories() finds nothing AND the stop hook's
# summary fallback also produces nothing.
_TEXT_TRIVIAL = "abcd efgh ijk\nlmno pqrs tuv\nwxyz abcd efg\nhijk lmno pqr"


async def _cleanup_test_brain(brain_name: str) -> None:
    """Best-effort: drop this test's throwaway brain if it landed on live
    SurrealDB instead of the isolated tmp_path SQLite fixture.

    The isolated_brain* fixtures only override HOME/SURREAL_MEMORY_BRAIN, not
    SURREAL_MEMORY_STORAGE. When a developer's shell already exports
    SURREAL_MEMORY_STORAGE=surrealdb (+ SURREALDB_URL/SURREALDB_PASS) for
    day-to-day dashboard use, storage_backend resolves to "surrealdb" here
    too, and every test would otherwise leave an orphaned "idem*" row behind
    on the shared DB -- the same brain-leakage failure mode already fixed for
    the other live-SurrealDB tests via cleanup_live_brains() in
    _surrealdb_live.py. SQLite brains need no cleanup: they live under the
    fixture's own tmp_path HOME, which pytest discards on its own.
    """
    from surreal_memory.unified_config import get_config, get_shared_storage

    if get_config().storage_backend != "surrealdb":
        return
    try:
        storage = await get_shared_storage()
        from tests.unit._surrealdb_live import cleanup_live_brains

        await cleanup_live_brains(storage, own_brain_id=brain_name)
    except Exception:
        pass


@pytest.fixture
async def isolated_brain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[str]:
    """Point unified_config at a throwaway HOME with a unique brain+session."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", "/tmp")
    brain_name = "idem" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("SURREAL_MEMORY_BRAIN", brain_name)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-" + uuid.uuid4().hex[:12])

    from surreal_memory.unified_config import get_config

    get_config(reload=True)
    yield brain_name
    await _cleanup_test_brain(brain_name)


@pytest.fixture
async def isolated_brain_gate_enforce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[str]:
    """Same as isolated_brain, but with the write gate forced to enforce mode
    and an impossible min_length, so every capture is rejected deterministically."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", "/tmp")
    brain_name = "idem" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("SURREAL_MEMORY_BRAIN", brain_name)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-" + uuid.uuid4().hex[:12])

    surrealmemory_dir = tmp_path / ".surrealmemory"
    surrealmemory_dir.mkdir(parents=True, exist_ok=True)
    (surrealmemory_dir / "config.toml").write_text(
        '\n[write_gate]\nmode = "enforce"\nauto_capture_mode = "enforce"\nmin_length = 999999\n',
        encoding="utf-8",
    )

    from surreal_memory.unified_config import get_config

    get_config(reload=True)
    yield brain_name
    await _cleanup_test_brain(brain_name)


class TestStopHookIdempotency:
    """Stop hook: repeated capture for the same session, unchanged content."""

    async def test_duplicate_summary_suppressed_on_second_call(self, isolated_brain: str) -> None:
        result1 = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert result1["saved"] == 1

        result2 = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert result2["saved"] == 0, (
            "second call with unchanged content must save nothing (idempotency)"
        )
        message = result2["message"].lower()
        assert "duplicate" in message or "idempot" in message, (
            f"saved=0 message must name the reason distinguishably, got: {result2['message']!r}"
        )

    async def test_new_content_same_session_still_saved(self, isolated_brain: str) -> None:
        """Criterion (5) guard: genuinely new content in the same session must
        never be suppressed by the idempotency key."""
        result1 = await stop_hook.capture_text(_TEXT_A, project_name=None)
        result2 = await stop_hook.capture_text(_TEXT_B, project_name=None)
        assert result1["saved"] == 1
        assert result2["saved"] == 1

    async def test_message_no_content(self, isolated_brain: str) -> None:
        result = await stop_hook.capture_text(_TEXT_TRIVIAL, project_name=None)
        assert result["saved"] == 0
        message = result["message"].lower()
        assert "content" in message
        assert "duplicate" not in message and "idempot" not in message
        assert "gate" not in message and "rejected" not in message

    async def test_message_gate_rejected(self, isolated_brain_gate_enforce: str) -> None:
        result = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert result["saved"] == 0
        message = result["message"].lower()
        assert "gate" in message or "rejected" in message
        assert "duplicate" not in message and "idempot" not in message

    async def test_message_duplicate_differs_from_no_content_and_gate(
        self, isolated_brain: str
    ) -> None:
        first = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert first["saved"] == 1
        second = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert second["saved"] == 0
        message = second["message"].lower()
        assert "duplicate" in message or "idempot" in message
        assert "gate" not in message and "rejected" not in message
        assert "no memorable content" not in message


class TestPreCompactHookIdempotency:
    """PreCompact hook: same idempotency contract as Stop (shared seen-set)."""

    async def test_duplicate_fragment_suppressed_on_second_call(self, isolated_brain: str) -> None:
        text = "The root cause was a missing null check in the session handler code path."
        result1 = await pre_compact_hook.flush_text(text, project_name=None)
        assert result1["saved"] == 1

        result2 = await pre_compact_hook.flush_text(text, project_name=None)
        assert result2["saved"] == 0, (
            "second flush with unchanged content must save nothing (idempotency)"
        )

    async def test_new_fragment_same_session_still_saved(self, isolated_brain: str) -> None:
        text_a = "The root cause was a missing null check in the session handler code path."
        text_b = "The workaround for this is to add a retry with exponential backoff logic."
        result1 = await pre_compact_hook.flush_text(text_a, project_name=None)
        result2 = await pre_compact_hook.flush_text(text_b, project_name=None)
        assert result1["saved"] == 1
        assert result2["saved"] == 1

    async def test_stop_and_pre_compact_share_seen_set(self, isolated_brain: str) -> None:
        """Stop and PreCompact are keyed on the same CLAUDE_SESSION_ID and must
        not re-capture each other's writes (RUNBOOK-80 prototype rationale)."""
        text = "The root cause was a missing null check in the session handler code path."
        stop_result = await stop_hook.capture_text(text, project_name=None)
        assert stop_result["saved"] == 1

        precompact_result = await pre_compact_hook.flush_text(text, project_name=None)
        assert precompact_result["saved"] == 0


class TestIdempotencyNegativePaths:
    """RUNBOOK-80 §6 negative paths: every one must give a distinguishable,
    non-crashing status -- never a silent success."""

    async def test_no_session_id_still_captures_and_dedupes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No CLAUDE_SESSION_ID -> session_key() falls back to 'default', but
        the hook must still function (capture once, suppress the repeat)."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("TMPDIR", "/tmp")
        brain_name = "idem" + uuid.uuid4().hex[:12]
        monkeypatch.setenv("SURREAL_MEMORY_BRAIN", brain_name)
        monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)

        from surreal_memory.unified_config import get_config

        get_config(reload=True)

        try:
            result1 = await stop_hook.capture_text(_TEXT_A, project_name=None)
            assert result1["saved"] == 1
            result2 = await stop_hook.capture_text(_TEXT_A, project_name=None)
            assert result2["saved"] == 0
        finally:
            await _cleanup_test_brain(brain_name)

    async def test_corrupt_idempotency_state_fails_open_not_crash(
        self, isolated_brain: str, tmp_path: Path
    ) -> None:
        """A corrupt capture_state.json must never crash or silently block a
        real capture -- fail open (treat as not-seen) per capture_state.py design."""
        state_dir = tmp_path / ".surrealmemory"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "capture_state.json").write_text("{not valid json", encoding="utf-8")

        result = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert result["saved"] == 1
        assert "error" not in result

    async def test_concurrent_calls_never_crash_or_silently_ambiguous(
        self, isolated_brain: str, tmp_path: Path
    ) -> None:
        """Two REAL hook subprocesses racing for the same session (the actual
        production shape -- separate OS processes, not two coroutines sharing
        one process's cached storage connection) must never hang or crash, and
        must never both report an ambiguous/silent result: each result is
        either a clean capture or a clean duplicate-skip, and the total saved
        count is bounded (no double-loss, no unexplained failure)."""
        import json
        import os
        import subprocess
        import sys

        proj_dir = tmp_path / ".claude" / "projects" / "race-test"
        proj_dir.mkdir(parents=True, exist_ok=True)
        transcript = proj_dir / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"role": "user", "content": _TEXT_A}) + "\n", encoding="utf-8"
        )
        hook_input = json.dumps({"transcript_path": str(transcript), "cwd": str(tmp_path)})

        # main() derives project_name from cwd's basename (no git repo here),
        # and add_typed_memory's project_id FK requires that project to already
        # exist (a separate, already-flagged bug -- task_11add155 -- unrelated
        # to idempotency). Pre-register it so this test measures the race, not
        # that unrelated FK crash.
        from surreal_memory.core.project import Project
        from surreal_memory.unified_config import get_config, get_shared_storage
        from surreal_memory.utils.timeutils import utcnow

        config = get_config()
        storage = await get_shared_storage(config.current_brain)
        brain = await storage.get_brain(config.current_brain)
        assert brain is not None
        await storage.add_project(  # type: ignore[attr-defined]
            Project(id=tmp_path.name, name=tmp_path.name, start_date=utcnow(), created_at=utcnow())
        )
        await storage.close()

        cmd = [sys.executable, "-m", "surreal_memory.hooks.stop"]
        proc1 = subprocess.Popen(  # noqa: S603 -- fixed cmd, no shell, no untrusted input
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ),
        )
        proc2 = subprocess.Popen(  # noqa: S603 -- fixed cmd, no shell, no untrusted input
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ),
        )
        try:
            _, err1 = proc1.communicate(input=hook_input, timeout=30)
            _, err2 = proc2.communicate(input=hook_input, timeout=30)
        except subprocess.TimeoutExpired:
            proc1.kill()
            proc2.kill()
            raise

        assert proc1.returncode == 0
        assert proc2.returncode == 0
        combined = err1 + err2
        captured_count = combined.count("captured 1 memories")
        # Best case: the race is caught and only one process wins (1). Worst
        # case (known, accepted limitation -- no cross-process lock on the
        # read-then-write idempotency state): both win (2). Never zero
        # (silent total loss).
        assert captured_count in (1, 2), f"unexpected capture count under race: {err1!r} {err2!r}"
        assert err1.strip() and err2.strip(), "every hook invocation must report a status"


class TestIdempotencyFuzz:
    """RUNBOOK-80 §6 fuzz/property test: >=50 iterations, fixed seed. Save
    count must increase monotonically and ONLY when genuinely new content
    (a never-before-seen decision option) is introduced."""

    async def test_random_growing_session_saves_only_on_new_content(
        self, isolated_brain: str
    ) -> None:
        import random

        rng = random.Random(1234567890)
        seen_options: set[int] = set()
        next_option = 0
        expected_total_saved = 0
        actual_total_saved = 0

        for _ in range(60):
            if seen_options and rng.random() < 0.7:
                option = rng.choice(sorted(seen_options))
                is_new = False
            else:
                option = next_option
                next_option += 1
                seen_options.add(option)
                is_new = True

            text = f"I decided to use option number {option} for this particular task."
            result = await stop_hook.capture_text(text, project_name=None)
            assert result["saved"] in (0, 1)
            actual_total_saved += result["saved"]
            if is_new:
                expected_total_saved += 1
                assert result["saved"] == 1, (
                    f"genuinely new content (option {option}) must not be suppressed"
                )
            else:
                assert result["saved"] == 0, (
                    f"repeated content (option {option}) must be suppressed, "
                    f"got message: {result['message']!r}"
                )

        assert actual_total_saved == expected_total_saved
        assert actual_total_saved == len(seen_options)
