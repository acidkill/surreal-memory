"""Behavioral regression tests for auto-capture hook idempotency (upstream #80).

The defect, confirmed manually during development: the Stop and PreCompact
hooks re-encode a session summary (or fragment) every time they are invoked
for the same session, even when the effectively-captured text has not
changed since the previous call.

Every test here runs against a real, isolated in-memory brain (a fresh tmp_path
HOME) -- never the shared prod brain. Each test uses a unique brain name and
session id to avoid cross-test collisions in unified_config's process-wide
_config / _storage_cache singletons.

That isolation is enforced by the autouse ``_isolate_storage_env`` fixture
below, not merely assumed: overriding HOME/SURREAL_MEMORY_BRAIN alone is not
enough, because UnifiedConfig.load() reads SURREAL_MEMORY_STORAGE straight
from the environment and lets it win over config.toml. See that fixture for
the full list of variables and why each one matters.
"""

from __future__ import annotations

import uuid
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


@pytest.fixture(autouse=True)
def _isolate_storage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's real environment decide a test's outcome.

    These tests assert on a throwaway in-memory brain under a tmp_path HOME. Two
    variables from a normal dev shell (``set -a; . ./.env``) silently move that
    target, so the suite passed in CI -- where nothing is exported -- and failed
    for anyone with a configured environment:

    * ``SURREAL_MEMORY_STORAGE=surrealdb`` -- UnifiedConfig.load() reads this
      directly and it outranks config.toml, so ``get_shared_storage()`` returns
      the *live* SurrealDB instead of the fixture's in-memory brain. The unique
      "idem*" brain does not exist there, so ``capture_text()`` short-circuits
      on ``{"error": "No brain configured", "saved": 0}`` and nearly every
      assertion fails as ``assert 0 == 1``. Pinned to "memory" rather than
      deleted: this file's whole contract is an isolated fixture backend.
    * ``SURREAL_MEMORY_DIR`` -- resolves the data dir ahead of ``Path.home()``
      in both UnifiedConfig and ``capture_state._state_path()``, so config.toml
      and capture_state.json are read from the developer's real directory
      instead of tmp_path (and this suite writes its state into it).

    ``SURREALDB_URL``/``SURREALDB_PASS`` are dropped too, mirroring the same
    guard in ``tests/e2e/test_api.py``: with no route to the live server, a
    future regression in the pin above fails loudly instead of quietly writing
    test brains into production.

    Autouse, so tests that build their own environment inline (rather than via
    the isolated_brain* fixtures) are covered as well; pytest instantiates
    autouse fixtures before same-scope non-autouse ones, so the pin is in place
    before any ``get_config(reload=True)`` below.
    """
    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "memory")
    monkeypatch.delenv("SURREAL_MEMORY_DIR", raising=False)
    monkeypatch.delenv("SURREALDB_URL", raising=False)
    monkeypatch.delenv("SURREALDB_PASS", raising=False)


@pytest.fixture
async def isolated_brain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point unified_config at a throwaway HOME with a unique brain+session.

    No teardown: the in-memory brain lives under this fixture's own tmp_path HOME,
    which pytest discards on its own.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", "/tmp")
    brain_name = "idem" + uuid.uuid4().hex[:12]
    monkeypatch.setenv("SURREAL_MEMORY_BRAIN", brain_name)
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-" + uuid.uuid4().hex[:12])

    from surreal_memory.unified_config import get_config

    get_config(reload=True)
    return brain_name


@pytest.fixture
async def isolated_brain_gate_enforce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
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
    return brain_name


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

        result1 = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert result1["saved"] == 1
        result2 = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert result2["saved"] == 0

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


class TestRejectedContentNotResubmitted:
    """A refusal must be remembered, not only a save.

    Until this was fixed only *successful* encodes were marked seen, so a
    fragment the gate turned down came back on every subsequent invocation and
    was judged -- and logged to ``gate_decision`` -- again. Measured on the live
    brain 2026-08-08: 499 auto decisions over 24h carried just 134 distinct
    contents, one fragment appearing 36 times between 07:30 and 17:34. Because
    the accept rate is computed over those rows, the inflated denominator made
    the gate look ~4x more hostile than it was (0.39% vs ~1.4%).
    """

    async def _count_gate_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Count every gate decision actually logged, in call order.

        ``log_gate_decision`` is what writes the ``gate_decision`` row, so it is
        the exact quantity that inflated the denominator -- assert on it rather
        than on a proxy.
        """
        from surreal_memory.engine import gate_telemetry

        calls: list[str] = []
        real = gate_telemetry.log_gate_decision

        async def counting(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            calls.append(str(kwargs.get("reason", "")))
            return await real(*args, **kwargs)  # type: ignore[arg-type]

        # The hooks do a function-local ``from ... import log_gate_decision``,
        # which resolves the attribute at call time -- so patching the module
        # attribute is enough and no import-order trickery is needed.
        monkeypatch.setattr(gate_telemetry, "log_gate_decision", counting)
        return calls

    async def test_stop_hook_does_not_rejudge_rejected_content(
        self, isolated_brain_gate_enforce: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = await self._count_gate_calls(monkeypatch)

        first = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert first["saved"] == 0
        after_first = len(calls)
        assert after_first >= 1, "the first call must actually reach the gate"

        second = await stop_hook.capture_text(_TEXT_A, project_name=None)
        assert second["saved"] == 0
        assert len(calls) == after_first, (
            "identical content already refused this session must not be judged "
            f"again; gate was invoked {len(calls) - after_first} extra time(s)"
        )

    async def test_pre_compact_honours_refusal_recorded_by_stop(
        self, isolated_brain_gate_enforce: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PreCompact must not re-judge a fragment Stop already refused.

        The earlier version of this test fed ``_TEXT_A`` straight to
        ``flush_text()``, which never reaches the gate at all for that input —
        the detector finds nothing above the emergency threshold, so the test
        passed even with the fix entirely removed (``gate_calls=0`` on both
        sides). This version forces the path: both hooks' detector is patched
        to return one controlled fragment, the gate is patched to reject it,
        and the test counts actual gate invocations — Stop must reach it
        exactly once, PreCompact not at all.
        """
        from surreal_memory.engine.quality_scorer import QualityResult

        gate_calls: list[str] = []

        def _rejecting_gate(content: str, **kwargs: object) -> QualityResult:
            gate_calls.append(content)
            return QualityResult(
                score=1, quality="low", rejected=True, rejection_reason="test refusal"
            )

        # Both hooks resolve check_write_gate via a function-local import, so
        # patching the source module covers both paths.
        monkeypatch.setattr(
            "surreal_memory.engine.quality_scorer.check_write_gate", _rejecting_gate
        )

        fragment = {
            "content": "The release manager approved the rollback plan after the incident review.",
            "confidence": 0.95,
            "priority": 5,
            "type": "decision",
        }

        def _one_fragment(*args: object, **kwargs: object) -> list[dict[str, object]]:
            return [dict(fragment)]

        monkeypatch.setattr(
            "surreal_memory.mcp.auto_capture.analyze_text_for_memories", _one_fragment
        )

        first = await stop_hook.capture_text(fragment["content"], project_name=None)
        assert first["saved"] == 0
        after_stop = len(gate_calls)
        # The stop hook gates the fragment and (after the refusal) its session
        # summary fallback, so >= 2 gate calls are expected there; what
        # matters is that the fragment itself reached the gate and was refused.
        assert after_stop >= 2, (
            f"the stop hook must reach the gate at least twice (fragment + summary), "
            f"got {after_stop}: {gate_calls}"
        )

        second = await pre_compact_hook.flush_text(fragment["content"], project_name=None)
        assert second["saved"] == 0
        assert len(gate_calls) == after_stop, (
            "PreCompact re-judged content the Stop hook had already refused: "
            f"gate was invoked {len(gate_calls) - after_stop} extra time(s)"
        )

    async def test_pre_compact_records_its_own_refusal(
        self, isolated_brain_gate_enforce: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal recording must also live in PreCompact's own reject
        branch — a fragment PreCompact refuses (Stop never saw it) must not
        be re-judged on the next PreCompact. Covers the `captured_keys.append`
        hunk in `pre_compact.py` that the Stop-side test cannot reach."""
        from surreal_memory.engine.quality_scorer import QualityResult

        gate_calls: list[str] = []

        def _rejecting_gate(content: str, **kwargs: object) -> QualityResult:
            gate_calls.append(content)
            return QualityResult(
                score=1, quality="low", rejected=True, rejection_reason="test refusal"
            )

        monkeypatch.setattr(
            "surreal_memory.engine.quality_scorer.check_write_gate", _rejecting_gate
        )

        fragment = {
            "content": "The on-call engineer drained the failed node before the midnight rollout.",
            "confidence": 0.95,
            "priority": 5,
            "type": "fact",
        }

        def _one_fragment(*args: object, **kwargs: object) -> list[dict[str, object]]:
            return [dict(fragment)]

        monkeypatch.setattr(
            "surreal_memory.mcp.auto_capture.analyze_text_for_memories", _one_fragment
        )

        first = await pre_compact_hook.flush_text(fragment["content"], project_name=None)
        assert first["saved"] == 0
        after_first = len(gate_calls)
        assert after_first >= 1, "the first PreCompact must actually reach the gate"

        second = await pre_compact_hook.flush_text(fragment["content"], project_name=None)
        assert second["saved"] == 0
        assert len(gate_calls) == after_first, (
            "PreCompact re-judged content it had already refused itself: "
            f"gate was invoked {len(gate_calls) - after_first} extra time(s)"
        )

    async def test_new_content_still_reaches_the_gate(
        self, isolated_brain_gate_enforce: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Positive control: suppression must be specific, not a blanket mute.

        Without this, a bug that silenced *everything* after the first call would
        pass both assertions above.
        """
        calls = await self._count_gate_calls(monkeypatch)

        await stop_hook.capture_text(_TEXT_A, project_name=None)
        after_first = len(calls)

        await stop_hook.capture_text(_TEXT_B, project_name=None)
        assert len(calls) > after_first, (
            "different content must still be judged -- the refusal cache is "
            "keyed on content, not on 'anything already tried'"
        )
