"""Unit tests for surreal_memory.hooks.capture_state (per-session idempotency)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surreal_memory.hooks.capture_state import (
    _MAX_HASHES_PER_SESSION,
    _MAX_SESSIONS,
    _state_path,
    content_key,
    load_seen,
    mark_seen,
    rejected_key,
    session_key,
)


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own capture_state.json, never the real one."""
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    return tmp_path


class TestStatePathFallback:
    """Regression test: Path("") normalizes to Path(".") which is truthy, so
    a naive `Path(env_var_or_empty) or fallback` never falls back -- an unset
    SURREAL_MEMORY_DIR must resolve under the user's home, never cwd."""

    def test_unset_dir_resolves_under_home_not_cwd(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_DIR", raising=False)
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        assert _state_path() == fake_home / ".surrealmemory" / "capture_state.json"

    def test_empty_string_dir_also_falls_back_to_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SURREAL_MEMORY_DIR", "")
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        assert _state_path() == fake_home / ".surrealmemory" / "capture_state.json"


class TestSessionKey:
    def test_prefers_claude_session_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SESSION_ID", "abc-123")
        assert session_key() == "abc-123"

    def test_falls_back_to_transcript_path_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
        key1 = session_key("/some/transcript.jsonl")
        key2 = session_key("/some/transcript.jsonl")
        key3 = session_key("/other/transcript.jsonl")
        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("tp_")

    def test_falls_back_to_default_constant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_SESSION_ID", raising=False)
        assert session_key(None) == "default"

    def test_empty_session_id_env_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SESSION_ID", "   ")
        assert session_key(None) == "default"


class TestContentKey:
    def test_deterministic(self) -> None:
        assert content_key("Hello world") == content_key("Hello world")

    def test_case_insensitive(self) -> None:
        assert content_key("Hello World") == content_key("hello world")

    def test_whitespace_normalized(self) -> None:
        assert content_key("hello   world\n\nfoo") == content_key("hello world foo")

    def test_different_content_different_key(self) -> None:
        assert content_key("Decision: use SQLite") != content_key("Decision: use Postgres")


class TestLoadMarkSeenRoundtrip:
    def test_empty_session_returns_empty_set(self) -> None:
        assert load_seen("nope-not-seen-yet") == set()

    def test_mark_then_load_roundtrip(self) -> None:
        skey = "session-1"
        mark_seen(skey, ["hash1", "hash2"])
        assert load_seen(skey) == {"hash1", "hash2"}

    def test_mark_seen_with_empty_list_is_noop(self, isolated_state_dir: Path) -> None:
        mark_seen("session-2", [])
        assert not (isolated_state_dir / "capture_state.json").exists()

    def test_mark_seen_accumulates_across_calls(self) -> None:
        skey = "session-3"
        mark_seen(skey, ["hash1"])
        mark_seen(skey, ["hash2"])
        assert load_seen(skey) == {"hash1", "hash2"}

    def test_sessions_are_isolated(self) -> None:
        mark_seen("session-a", ["only-in-a"])
        mark_seen("session-b", ["only-in-b"])
        assert load_seen("session-a") == {"only-in-a"}
        assert load_seen("session-b") == {"only-in-b"}

    def test_atomic_write_leaves_no_tmp_file(self, isolated_state_dir: Path) -> None:
        mark_seen("session-4", ["hash1"])
        assert not (isolated_state_dir / "capture_state.json.tmp").exists()
        assert (isolated_state_dir / "capture_state.json").exists()


class TestBoundedRetention:
    def test_hashes_per_session_bounded_fifo(self) -> None:
        skey = "session-many-hashes"
        keys = [f"h{i}" for i in range(_MAX_HASHES_PER_SESSION + 50)]
        mark_seen(skey, keys)
        seen = load_seen(skey)
        assert len(seen) == _MAX_HASHES_PER_SESSION
        # oldest entries were trimmed, newest survive
        assert "h0" not in seen
        assert keys[-1] in seen

    def test_sessions_bounded_fifo(self, isolated_state_dir: Path) -> None:
        for i in range(_MAX_SESSIONS + 10):
            mark_seen(f"sess-{i}", [f"hash-{i}"])
        data = json.loads((isolated_state_dir / "capture_state.json").read_text())
        assert len(data) == _MAX_SESSIONS
        assert "sess-0" not in data
        assert f"sess-{_MAX_SESSIONS + 9}" in data


class TestFailOpen:
    def test_corrupt_json_file_fails_open(self, isolated_state_dir: Path) -> None:
        state_file = isolated_state_dir / "capture_state.json"
        state_file.write_text("{not valid json!!", encoding="utf-8")
        assert load_seen("any-session") == set()

    def test_non_dict_json_fails_open(self, isolated_state_dir: Path) -> None:
        state_file = isolated_state_dir / "capture_state.json"
        state_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        assert load_seen("any-session") == set()

    def test_mark_seen_never_raises_on_unwritable_dir(
        self, isolated_state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point SURREAL_MEMORY_DIR at a path that cannot be created (parent is a file).
        blocker = isolated_state_dir / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setenv("SURREAL_MEMORY_DIR", str(blocker / "nested"))
        mark_seen("any-session", ["hash1"])  # must not raise


class TestRejectedKey:
    """Gate refusals are remembered too -- but only for the threshold that refused.

    Rejected candidates used to be left unmarked, so the hooks re-submitted them
    to the gate on every invocation. Measured on production 2026-08-08: 499 auto
    decisions in 24h carried only 134 distinct contents (one fragment judged 36
    times), inflating the gate's denominator ~3.7x.
    """

    def test_rejected_key_differs_from_plain_key(self) -> None:
        ck = content_key("Decision: c remains explicitly open and unresolved")
        assert rejected_key(ck, 5) != ck, (
            "a refusal must not be recorded as an acceptance -- otherwise lowering "
            "the threshold could never revive the content"
        )

    def test_rejected_key_is_threshold_scoped(self) -> None:
        ck = content_key("Insight: the harvester stalled on shard three")
        assert rejected_key(ck, 5) != rejected_key(ck, 4)

    def test_rejected_key_is_stable(self) -> None:
        ck = content_key("Error: flush_batch named the remote file from a stale counter")
        assert rejected_key(ck, 5) == rejected_key(ck, 5)

    def test_refusal_suppresses_at_same_threshold(self, isolated_state_dir: Path) -> None:
        ck = content_key("Decision: b is blocked awaiting human authorization")
        mark_seen("sess-a", [rejected_key(ck, 5)])
        seen = load_seen("sess-a")
        assert rejected_key(ck, 5) in seen

    def test_refusal_expires_when_threshold_changes(self, isolated_state_dir: Path) -> None:
        """The whole point of scoping: a lowered bar must give a second hearing.

        Marking a refusal as a plain seen-key would silence that content for the
        rest of the session even after the operator lowered the threshold --
        trading duplicate noise for silent loss, which is the worse failure.
        """
        ck = content_key("Decision: b is blocked awaiting human authorization")
        mark_seen("sess-b", [rejected_key(ck, 5)])
        seen = load_seen("sess-b")
        assert ck not in seen
        assert rejected_key(ck, 4) not in seen, (
            "after the threshold moved 5->4 the candidate must be judged again"
        )

    def test_accepted_key_survives_threshold_change(self, isolated_state_dir: Path) -> None:
        """Acceptance is unconditional -- it must not be revived by a threshold move."""
        ck = content_key("Insight: rclone token rotation confirmed on every mount")
        mark_seen("sess-c", [ck])
        seen = load_seen("sess-c")
        assert ck in seen
