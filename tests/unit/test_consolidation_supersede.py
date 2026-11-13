"""A failed or expired lone semantic_link run no longer blocks the other strategies.

The durable progress record is one per brain. A run that died in semantic_link leaves it ``failed``
and every later run of another strategy is refused ("requested strategies differ") until someone
clears it by hand. With ``supersede_failed_semantic_link`` the next run takes the record over,
keeps an audit of what it replaced, and carries on. semantic_link is additive and idempotent
(deterministic edge ids), so abandoning its half-finished run loses nothing.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from typing import Any

import pytest

import surreal_memory.engine.consolidation_progress as progress_module
from surreal_memory.engine.consolidation import (
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationProgressError,
    ConsolidationProgressSession,
    ConsolidationResumeMismatchError,
)
from surreal_memory.utils.timeutils import utcnow
from tests.unit.test_consolidation_progress import REFERENCE_TIME, FakeProgressStorage

_ERROR = "SemanticSourceChangedError: neuron changed since the semantic source token was captured"


async def _failed_run(
    storage: FakeProgressStorage,
    strategies: list[str],
    *,
    fingerprint: str = "fp-a",
    reference_time: Any = None,
    error: str = _ERROR,
) -> str:
    session = await ConsolidationProgressSession.open(
        storage, strategies, fingerprint, reference_time or utcnow()
    )
    assert session is not None
    await session.checkpoint(
        "semantic_link", "semantic_link_discovery_similarity", cursor="7", pending=["{}"]
    )
    await session.fail(error)
    run_id = str(session.state["run_id"])
    await session.close()
    return run_id


async def _paused_run(storage: FakeProgressStorage, *, reference_time: Any) -> str:
    session = await ConsolidationProgressSession.open(
        storage, ["semantic_link"], "fp-a", reference_time
    )
    assert session is not None
    await session.checkpoint("semantic_link", "semantic_link_discovery_neurons", cursor="3")
    await session.pause()
    run_id = str(session.state["run_id"])
    await session.close()
    return run_id


@pytest.mark.asyncio
async def test_a_failed_semantic_link_run_blocks_other_strategies_without_the_flag() -> None:
    storage = FakeProgressStorage()
    await _failed_run(storage, ["semantic_link"])
    before = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError, match="requested strategies differ"):
        await ConsolidationProgressSession.open(storage, ["merge"], "fp-a", REFERENCE_TIME)

    assert storage.progress == before


@pytest.mark.asyncio
async def test_the_flag_lets_the_next_run_take_over_and_keeps_an_audit() -> None:
    storage = FakeProgressStorage()
    old_run = await _failed_run(storage, ["semantic_link"])

    session = await ConsolidationProgressSession.open(
        storage, ["merge"], "fp-a", REFERENCE_TIME, supersede_failed_semantic_link=True
    )

    assert session is not None
    assert not session.resumed
    assert session.state["run_id"] != old_run
    assert session.state["requested_strategies"] == ["merge"]
    audit = session.state["strategy_states"]["_superseded_runs"]
    assert len(audit) == 1
    assert audit[0]["run_id"] == old_run
    assert audit[0]["status"] == "failed"
    assert audit[0]["phase"] == "semantic_link_discovery_similarity"
    assert audit[0]["last_error"] == _ERROR
    assert audit[0]["reason"] == "failed_semantic_link_run_does_not_block_other_strategies"
    assert session.superseded == audit[0]
    await session.close()


@pytest.mark.asyncio
async def test_the_flag_does_not_extend_to_a_failed_run_of_several_strategies() -> None:
    storage = FakeProgressStorage()
    await _failed_run(storage, ["merge", "semantic_link"])

    with pytest.raises(ConsolidationResumeMismatchError):
        await ConsolidationProgressSession.open(
            storage, ["merge"], "fp-a", REFERENCE_TIME, supersede_failed_semantic_link=True
        )


@pytest.mark.asyncio
async def test_a_failed_semantic_link_run_retried_alone_is_resumed_not_replaced() -> None:
    storage = FakeProgressStorage()
    old_run = await _failed_run(storage, ["semantic_link"])

    session = await ConsolidationProgressSession.open(
        storage, ["semantic_link"], "fp-a", REFERENCE_TIME, supersede_failed_semantic_link=True
    )

    assert session is not None
    assert session.resumed
    assert session.state["run_id"] == old_run
    assert session.superseded is None
    await session.close()


@pytest.mark.asyncio
async def test_an_explicit_resume_never_supersedes() -> None:
    storage = FakeProgressStorage()
    old_run = await _failed_run(storage, ["semantic_link"])
    before = deepcopy(storage.progress)

    with pytest.raises(ConsolidationResumeMismatchError):
        await ConsolidationProgressSession.open(
            storage,
            ["merge"],
            "fp-a",
            REFERENCE_TIME,
            resume_run_id=old_run,
            supersede_failed_semantic_link=True,
        )

    assert storage.progress == before


def test_the_cli_rejects_resume_together_with_the_option() -> None:
    from unittest.mock import patch

    import typer

    from surreal_memory.cli.commands import tools

    with patch.object(tools, "run_async") as run_async:
        with pytest.raises(typer.Exit) as exc_info:
            tools.consolidate(resume=True, supersede_failed_semantic_link=True)

    assert exc_info.value.exit_code == 1
    run_async.assert_not_called()


@pytest.mark.asyncio
async def test_a_paused_semantic_link_run_is_not_a_failure_and_is_not_superseded() -> None:
    storage = FakeProgressStorage()
    await _paused_run(storage, reference_time=utcnow())

    with pytest.raises(ConsolidationResumeMismatchError, match="requested strategies differ"):
        await ConsolidationProgressSession.open(
            storage, ["merge"], "fp-a", utcnow(), supersede_failed_semantic_link=True
        )


@pytest.mark.asyncio
async def test_a_run_older_than_the_token_window_cannot_be_resumed_and_is_replaced() -> None:
    from surreal_memory.storage.surrealdb.semantic_source_revision import (
        SEMANTIC_SOURCE_TOKEN_MAX_AGE,
    )

    storage = FakeProgressStorage()
    old_run = await _paused_run(
        storage, reference_time=utcnow() - SEMANTIC_SOURCE_TOKEN_MAX_AGE - timedelta(minutes=1)
    )

    resumed = await ConsolidationProgressSession.open(storage, ["semantic_link"], "fp-a", utcnow())
    assert resumed is not None and resumed.resumed and resumed.superseded is None  # no flag
    await resumed.pause()
    await resumed.close()

    session = await ConsolidationProgressSession.open(
        storage, ["semantic_link"], "fp-a", utcnow(), supersede_failed_semantic_link=True
    )
    assert session is not None
    assert not session.resumed
    assert session.state["run_id"] != old_run
    assert session.superseded is not None
    assert session.superseded["reason"] == "older_than_the_source_token_window"
    await session.close()


@pytest.mark.asyncio
async def test_a_young_paused_run_is_resumed_even_with_the_flag() -> None:
    storage = FakeProgressStorage()
    old_run = await _paused_run(storage, reference_time=utcnow() - timedelta(days=2))

    session = await ConsolidationProgressSession.open(
        storage, ["semantic_link"], "fp-a", utcnow(), supersede_failed_semantic_link=True
    )

    assert session is not None and session.resumed
    assert session.state["run_id"] == old_run
    await session.close()


@pytest.mark.asyncio
async def test_a_record_left_by_another_engine_version_heals_itself_with_the_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    monkeypatch.setattr(progress_module, "CONSOLIDATION_ENGINE_VERSION", "3.11.0:checkpoint-v1")
    await _failed_run(storage, ["semantic_link"])
    monkeypatch.setattr(progress_module, "CONSOLIDATION_ENGINE_VERSION", "3.12.0:checkpoint-v2")

    with pytest.raises(ConsolidationResumeMismatchError, match="different code version"):
        await ConsolidationProgressSession.open(storage, ["semantic_link"], "fp-a", REFERENCE_TIME)
    with pytest.raises(ConsolidationResumeMismatchError):
        await ConsolidationProgressSession.open(storage, ["merge"], "fp-a", REFERENCE_TIME)

    session = await ConsolidationProgressSession.open(
        storage, ["merge"], "fp-a", REFERENCE_TIME, supersede_failed_semantic_link=True
    )
    assert session is not None and not session.resumed
    assert session.state["engine_version"] == "3.12.0:checkpoint-v2"
    await session.close()


@pytest.mark.asyncio
async def test_the_audit_keeps_the_five_most_recent_replacements_newest_first() -> None:
    from surreal_memory.storage.surrealdb.semantic_source_revision import (
        SEMANTIC_SOURCE_TOKEN_MAX_AGE,
    )

    storage = FakeProgressStorage()
    run_ids: list[str] = []
    for _ in range(7):
        # every run is born already too old to resume, so the next one replaces it
        session = await ConsolidationProgressSession.open(
            storage,
            ["semantic_link"],
            "fp-a",
            utcnow() - SEMANTIC_SOURCE_TOKEN_MAX_AGE - timedelta(minutes=1),
            supersede_failed_semantic_link=True,
        )
        assert session is not None
        run_ids.append(str(session.state["run_id"]))
        await session.pause()
        await session.close()

    audit = storage.progress["strategy_states"]["_superseded_runs"]  # type: ignore[index]
    assert [item["run_id"] for item in audit] == list(reversed(run_ids[:-1]))[:5]


@pytest.mark.asyncio
async def test_the_engine_completes_merge_after_a_failed_semantic_link_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeProgressStorage()
    failing = ConsolidationEngine(storage)

    async def die(strategy: ConsolidationStrategy, report: Any, reference: Any, dry: bool) -> None:
        await failing._checkpoint_progress("semantic_link_discovery_neurons", cursor="neuron:1")
        raise ConsolidationProgressError("neuron changed since token captured")

    monkeypatch.setattr(failing, "_run_strategy", die)
    first = await failing.run([ConsolidationStrategy.SEMANTIC_LINK])
    assert first.extra["consolidation_status"] == "failed"
    old_run = storage.progress["run_id"]  # type: ignore[index]

    blocked = ConsolidationEngine(storage)
    report = await blocked.run([ConsolidationStrategy.MERGE])
    assert report.extra["consolidation_status"] == "blocked"

    engine = ConsolidationEngine(storage, supersede_failed_semantic_link=True)
    ran: list[str] = []

    async def work(strategy: ConsolidationStrategy, report: Any, reference: Any, dry: bool) -> None:
        ran.append(strategy.value)

    monkeypatch.setattr(engine, "_run_strategy", work)
    report = await engine.run([ConsolidationStrategy.MERGE])

    assert report.extra["consolidation_status"] == "completed"
    assert ran == ["merge"]
    assert storage.progress["strategy_states"]["_superseded_runs"][0]["run_id"] == old_run  # type: ignore[index]
    assert "Superseded unfinished semantic_link run" in report.summary()
    assert old_run in report.summary()


def test_the_option_is_not_part_of_what_must_match_to_resume() -> None:
    from dataclasses import asdict

    from surreal_memory.engine.consolidation import ConsolidationConfig

    assert "supersede_failed_semantic_link" not in asdict(ConsolidationConfig())
    assert not ConsolidationEngine(FakeProgressStorage())._supersede_failed_semantic_link
    assert ConsolidationEngine(
        FakeProgressStorage(), supersede_failed_semantic_link=True
    )._supersede_failed_semantic_link


def test_the_cli_exposes_the_option() -> None:
    import typer.main

    from surreal_memory.cli.main import app

    command = typer.main.get_command(app).commands["consolidate"]  # type: ignore[attr-defined]
    options = {opt: param for param in command.params for opt in param.opts}
    assert "--supersede-failed-semantic-link" in options
    assert options["--supersede-failed-semantic-link"].default is False
