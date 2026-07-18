"""Tests for engine/reasoning_miner.py — transcript scanning + ingest.

Uses synthetic JSONL transcript fixtures under a temporary ``.claude`` dir (no
real thinking text). Covers dedup, model glob filter, truncation, ``<synthetic>``
and opus (empty/denylisted) skipping, min-length gating, secret redaction,
task_context capture, model normalization, incremental scan-state, and the
async ingest path against InMemoryStorage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from surreal_memory.engine.reasoning_miner import (
    ingest_reasoning_traces,
    normalize_model,
    scan_transcripts,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

_NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _cfg(**kw: object) -> ReasoningTrainingConfig:
    base: dict[str, object] = {
        "mining_enabled": True,
        "min_trace_chars": 10,
        "max_trace_chars": 10_000,
        "scan_lookback_days": 0,
    }
    base.update(kw)
    return ReasoningTrainingConfig(**base)  # type: ignore[arg-type]


def _assistant(
    thinking: str,
    *,
    model: str = "claude-fable-5",
    session: str = "s1",
    uuid: str = "u1",
    timestamp: str = "2026-03-01T10:00:00Z",
    extra_blocks: list | None = None,
) -> dict:
    blocks: list = [{"type": "thinking", "thinking": thinking, "signature": "sig"}]
    if extra_blocks:
        blocks = extra_blocks
    return {
        "type": "assistant",
        "sessionId": session,
        "uuid": uuid,
        "cwd": "/home/x/proj",
        "timestamp": timestamp,
        "message": {"model": model, "role": "assistant", "content": blocks},
    }


def _user(text: str, *, session: str = "s1", uuid: str = "u0") -> dict:
    return {
        "type": "user",
        "sessionId": session,
        "uuid": uuid,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _write_transcript(
    claude_dir: Path, entries: list[dict], *, slug: str = "proj-a", name: str = "t.jsonl"
) -> Path:
    d = claude_dir / "projects" / slug
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return p


def _scan(claude_dir: Path, cfg: ReasoningTrainingConfig, state: Path) -> list[dict]:
    return scan_transcripts(cfg, state_path=state, claude_dir=claude_dir, now=_NOW)


def test_normalize_model_strips_date_suffix() -> None:
    assert normalize_model("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert normalize_model("claude-fable-5") == "claude-fable-5"
    assert normalize_model("  claude-sonnet-5  ") == "claude-sonnet-5"


def test_scan_extracts_thinking_with_task_context(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [
            _user("please fix the failing test in module X"),
            _assistant("restate the goal, decompose, then verify the edge cases carefully"),
        ],
    )
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert len(traces) == 1
    tr = traces[0]
    assert tr["model"] == "claude-fable-5"
    assert "verify the edge cases" in tr["content"]
    assert tr["task_context"] == "please fix the failing test in module X"
    assert tr["session_id"] == "s1"
    assert len(tr["trace_hash"]) == 64
    assert tr["created_at"] == "2026-03-01T10:00:00Z"


def test_dedup_identical_entries(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    entry = _assistant("a fairly long reasoning trace about verification", uuid="dup")
    _write_transcript(claude, [entry, entry])  # same session:uuid:block_index
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert len(traces) == 1


def test_incremental_state_skips_unchanged(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    state = tmp_path / "state.json"
    _write_transcript(claude, [_assistant("a long enough reasoning trace here")])
    assert len(_scan(claude, _cfg(), state)) == 1
    # Second scan, file unchanged → skipped entirely.
    assert _scan(claude, _cfg(), state) == []


def test_incremental_state_rescans_changed(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    state = tmp_path / "state.json"
    p = _write_transcript(claude, [_assistant("first reasoning trace long enough", uuid="a")])
    assert len(_scan(claude, _cfg(), state)) == 1
    # Append a new entry → file changes → rescan yields only the NEW trace (dedup).
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_assistant("second reasoning trace long enough", uuid="b")) + "\n")
    traces = _scan(claude, _cfg(), state)
    assert len(traces) == 1
    assert "second reasoning" in traces[0]["content"]


def test_backfill_bypasses_unchanged_skip(tmp_path: Path) -> None:
    # A normal scan skips a file whose size+mtime are unchanged since the
    # last scan (see test_incremental_state_skips_unchanged), but a backfill
    # scan re-emits its traces regardless — bypass, not state-deletion.
    claude = tmp_path / ".claude"
    state = tmp_path / "state.json"
    _write_transcript(claude, [_assistant("a long enough reasoning trace here", uuid="a")])
    assert len(_scan(claude, _cfg(), state)) == 1
    assert _scan(claude, _cfg(), state) == []  # normal 2nd scan: unchanged → skipped

    backfilled = scan_transcripts(
        _cfg(), state_path=state, claude_dir=claude, now=_NOW, backfill=True
    )
    assert len(backfilled) == 1
    assert "long enough reasoning trace" in backfilled[0]["content"]


def test_normal_scan_after_backfill_yields_nothing_new(tmp_path: Path) -> None:
    # After a backfill re-scan, the scan-state entry is rewritten just like a
    # normal scan would (never deleted) — so a THIRD, normal scan sees the
    # file as unchanged again and stays cheap.
    claude = tmp_path / ".claude"
    state = tmp_path / "state.json"
    _write_transcript(claude, [_assistant("a long enough reasoning trace here", uuid="a")])
    assert len(_scan(claude, _cfg(), state)) == 1  # 1st: normal, populates state

    backfilled = scan_transcripts(
        _cfg(), state_path=state, claude_dir=claude, now=_NOW, backfill=True
    )
    assert len(backfilled) == 1  # 2nd: backfill, bypasses skip, rewrites state

    assert _scan(claude, _cfg(), state) == []  # 3rd: normal, nothing new


def test_model_glob_filter(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [
            _assistant("fable reasoning trace long enough", model="claude-fable-5", uuid="a"),
            _assistant("sonnet reasoning trace long enough", model="claude-sonnet-5", uuid="b"),
        ],
    )
    traces = _scan(claude, _cfg(mining_models=("claude-fable-*",)), tmp_path / "state.json")
    assert [t["model"] for t in traces] == ["claude-fable-5"]


def test_synthetic_and_opus_are_skipped(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [
            _assistant("synthetic reasoning trace long enough", model="<synthetic>", uuid="a"),
            _assistant("opus reasoning trace long enough", model="claude-opus-4-8", uuid="b"),
            _assistant("fable reasoning trace long enough", model="claude-fable-5", uuid="c"),
        ],
    )
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert [t["model"] for t in traces] == ["claude-fable-5"]


def test_empty_thinking_skipped(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(claude, [_assistant("   ", uuid="a")])  # opus-style empty thinking
    assert _scan(claude, _cfg(), tmp_path / "state.json") == []


def test_min_trace_chars_gate(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(claude, [_assistant("short", uuid="a")])
    assert _scan(claude, _cfg(min_trace_chars=50), tmp_path / "state.json") == []


def test_truncation_to_max_chars(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(claude, [_assistant("x" * 500, uuid="a")])
    traces = _scan(claude, _cfg(max_trace_chars=100), tmp_path / "state.json")
    assert len(traces[0]["content"]) == 100
    assert traces[0]["content_chars"] == 100


def test_secret_redaction_before_staging(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    secret = "password = hunter2supersecret"  # noqa: S105 — synthetic test secret
    thinking = f"first I reason about the config, then note that {secret}, then continue"
    _write_transcript(claude, [_assistant(thinking, uuid="a")])

    redacted = _scan(claude, _cfg(redact_secrets=True), tmp_path / "s1.json")
    assert "hunter2supersecret" not in redacted[0]["content"]
    assert "[REDACTED]" in redacted[0]["content"]

    kept = _scan(claude, _cfg(redact_secrets=False), tmp_path / "s2.json")
    assert "hunter2supersecret" in kept[0]["content"]


def test_task_context_is_redacted(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [
            _user("my config has password = topsecret123 in it, help"),
            _assistant("reasoning about the config problem in detail here"),
        ],
    )
    traces = _scan(claude, _cfg(redact_secrets=True), tmp_path / "state.json")
    assert "topsecret123" not in traces[0]["task_context"]
    assert "[REDACTED]" in traces[0]["task_context"]


def test_content_redacts_bearer_jwt_and_vendor_keys(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123def456ghi"
    bearer_tok = "abcdefghijklmnop12345"
    openai_key = "sk-proj-ABCDEFGHIJKLMNOP1234"
    thinking = (
        f"I inspect the request; it sends Authorization: Bearer {bearer_tok} "
        f"and a jwt {jwt} and an openai key {openai_key} to the API endpoint"
    )
    _write_transcript(claude, [_assistant(thinking, uuid="a")])
    content = _scan(claude, _cfg(redact_secrets=True), tmp_path / "state.json")[0]["content"]
    assert bearer_tok not in content
    assert jwt not in content
    assert openai_key not in content
    assert "[REDACTED]" in content


def test_redaction_does_not_mangle_benign_hyphenated_prose(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    # "task-"/"disk-" contain the substring "sk-" but must NOT be redacted.
    thinking = "the task-queue-processor-v2-implementation refactors the disk-cache-layer module"
    _write_transcript(claude, [_assistant(thinking, uuid="a")])
    content = _scan(claude, _cfg(redact_secrets=True), tmp_path / "state.json")[0]["content"]
    assert "[REDACTED]" not in content
    assert "task-queue-processor-v2-implementation" in content


def test_model_normalization_applied(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [_assistant("dated model reasoning trace", model="claude-haiku-4-5-20251001", uuid="a")],
    )
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert traces[0]["model"] == "claude-haiku-4-5"


def test_files_outside_projects_glob_ignored(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    (claude / "projects").mkdir(parents=True, exist_ok=True)
    # A jsonl directly under .claude (not projects/*/) must not be scanned.
    (claude / "stray.jsonl").write_text(
        json.dumps(_assistant("stray reasoning trace long enough", uuid="a")) + "\n",
        encoding="utf-8",
    )
    assert _scan(claude, _cfg(), tmp_path / "state.json") == []


def test_nested_session_directory_discovered(tmp_path: Path) -> None:
    # projects/<project>/<session>/t.jsonl — one level deeper than the old
    # `projects/*/*.jsonl` glob covered.
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [_assistant("nested session reasoning trace long enough", uuid="n1")],
        slug="proj-a/2026-03-01-session1",
    )
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert len(traces) == 1
    assert traces[0]["project"] == "proj-a"


def test_subagent_transcript_discovered_and_attributed_to_project(tmp_path: Path) -> None:
    # projects/<project>/<session>/subagents/agent-*.jsonl — Task-tool subagent
    # transcripts must be discovered AND attributed to the top-level project,
    # not to a literal "subagents" pseudo-project.
    claude = tmp_path / ".claude"
    entry = _assistant("subagent reasoning trace long enough to mine", uuid="sa1")
    entry["isSidechain"] = True
    _write_transcript(
        claude,
        [entry],
        slug="proj-a/session1/subagents",
        name="agent-1.jsonl",
    )
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert len(traces) == 1
    assert traces[0]["project"] == "proj-a"


def test_stray_file_directly_in_projects_dir_ignored(tmp_path: Path) -> None:
    # A jsonl sitting directly in projects/ (no project subdirectory) isn't
    # associated with any project and must be skipped, even though it's a
    # deeper scan than the old glob.
    claude = tmp_path / ".claude"
    (claude / "projects").mkdir(parents=True, exist_ok=True)
    (claude / "projects" / "stray.jsonl").write_text(
        json.dumps(_assistant("stray at projects root long enough", uuid="s1")) + "\n",
        encoding="utf-8",
    )
    assert _scan(claude, _cfg(), tmp_path / "state.json") == []


def test_symlink_escape_blocked_at_depth(tmp_path: Path) -> None:
    # A symlink nested several directories deep that resolves outside
    # ~/.claude must still be rejected by the path-escape guard.
    claude = tmp_path / ".claude"
    session_dir = claude / "projects" / "proj-a" / "session1"
    session_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    real = outside / "evil.jsonl"
    real.write_text(
        json.dumps(_assistant("escaped reasoning trace long enough", uuid="e1")) + "\n",
        encoding="utf-8",
    )
    (session_dir / "evil.jsonl").symlink_to(real)
    assert _scan(claude, _cfg(), tmp_path / "state.json") == []


def test_scan_returns_all_traces_unbounded(tmp_path: Path) -> None:
    """No per-scan cap: many files' worth of traces are all returned, uncapped."""
    claude = tmp_path / ".claude"
    files = 6
    per_file = 4
    total = files * per_file
    for f in range(files):
        entries = [
            _assistant(
                f"reasoning trace file {f} number {i} long enough to qualify",
                session=f"s{f}",
                uuid=f"u{f}-{i}",
            )
            for i in range(per_file)
        ]
        _write_transcript(claude, entries, slug=f"proj-{f}")
    traces = _scan(claude, _cfg(), tmp_path / "state.json")
    assert len(traces) == total


async def test_ingest_reasoning_traces_into_storage(tmp_path: Path) -> None:
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [
            _user("do the thing"),
            _assistant("reasoning trace one long enough", model="claude-fable-5", uuid="a"),
            _assistant("reasoning trace two long enough", model="claude-sonnet-5", uuid="b"),
        ],
    )
    storage = InMemoryStorage()
    storage.set_brain("b1")
    cfg = UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=_cfg(),
    )
    result = await ingest_reasoning_traces(
        storage, "b1", cfg, claude_dir=claude, state_path=tmp_path / "state.json", now=_NOW
    )
    assert result.traces_ingested == 2
    assert result.traces_scanned == 2
    stats = await storage.get_reasoning_stats("b1")
    assert stats["total"] == 2
    assert set(stats["by_model"]) == {"claude-fable-5", "claude-sonnet-5"}


async def test_second_consecutive_backfill_ingest_is_noop_at_storage(tmp_path: Path) -> None:
    # Backfill bypasses the miner's size+mtime skip, so a 2nd backfill still
    # RE-SCANS the file and re-emits its trace — but trace_hash dedup at the
    # storage layer means nothing NEW actually gets inserted the 2nd time.
    claude = tmp_path / ".claude"
    _write_transcript(
        claude,
        [_assistant("reasoning trace long enough for backfill dedup test", uuid="a")],
    )
    storage = InMemoryStorage()
    storage.set_brain("b1")
    cfg = UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=_cfg(),
    )
    state_path = tmp_path / "state.json"

    first = await ingest_reasoning_traces(
        storage,
        "b1",
        cfg,
        claude_dir=claude,
        state_path=state_path,
        now=_NOW,
        backfill=True,
    )
    assert first.traces_ingested == 1

    second = await ingest_reasoning_traces(
        storage,
        "b1",
        cfg,
        claude_dir=claude,
        state_path=state_path,
        now=_NOW,
        backfill=True,
    )
    assert second.traces_scanned == 1  # the miner still re-emits it (bypass)
    assert second.traces_ingested == 0  # but storage-layer dedup drops it

    stats = await storage.get_reasoning_stats("b1")
    assert stats["total"] == 1  # no duplicate row was created


def _fake_unified_config(tmp_path: Path, *, mining_enabled: bool) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=_cfg(mining_enabled=mining_enabled),
    )


async def test_consolidation_skips_when_mining_disabled(tmp_path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    fake = _fake_unified_config(tmp_path, mining_enabled=False)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))
    calls = {"n": 0}

    async def _spy_ingest(*a, **k):  # pragma: no cover - must NOT be reached
        calls["n"] += 1
        from surreal_memory.engine.reasoning_miner import ReasoningIngestResult

        return ReasoningIngestResult(traces_ingested=9)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", _spy_ingest
    )

    storage = InMemoryStorage()
    storage.set_brain("b1")
    report = await ConsolidationEngine(storage).run(
        [ConsolidationStrategy.PROCESS_REASONING_TRACES]
    )
    assert report.reasoning_traces_ingested == 0
    assert calls["n"] == 0  # guard short-circuits before invoking the miner


async def test_consolidation_ingests_when_enabled(tmp_path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy
    from surreal_memory.engine.reasoning_miner import ReasoningIngestResult

    fake = _fake_unified_config(tmp_path, mining_enabled=True)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))

    async def _fake_ingest(storage, brain_id, config, **k):
        return ReasoningIngestResult(traces_ingested=3, traces_scanned=3)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", _fake_ingest
    )

    storage = InMemoryStorage()
    storage.set_brain("b1")
    report = await ConsolidationEngine(storage).run(
        [ConsolidationStrategy.PROCESS_REASONING_TRACES]
    )
    assert report.reasoning_traces_ingested == 3


async def test_consolidation_dry_run_skips_ingest(tmp_path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    fake = _fake_unified_config(tmp_path, mining_enabled=True)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))
    calls = {"n": 0}

    async def _spy_ingest(*a, **k):  # pragma: no cover - must NOT be reached
        calls["n"] += 1
        from surreal_memory.engine.reasoning_miner import ReasoningIngestResult

        return ReasoningIngestResult(traces_ingested=9)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_miner.ingest_reasoning_traces", _spy_ingest
    )

    storage = InMemoryStorage()
    storage.set_brain("b1")
    report = await ConsolidationEngine(storage).run(
        [ConsolidationStrategy.PROCESS_REASONING_TRACES], dry_run=True
    )
    assert report.reasoning_traces_ingested == 0
    assert calls["n"] == 0
