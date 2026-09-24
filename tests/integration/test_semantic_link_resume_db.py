"""Live SurrealDB regression for semantic-link replay across ID normalization.

Runs only against the explicitly opted-in loopback SMEM_TEST_SURREALDB_URL; the
fixture creates a unique database and uses the disposable test credentials.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.semantic_discovery import SemanticDiscoveryResult
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

TEST_SURREALDB_URL = os.getenv("SMEM_TEST_SURREALDB_URL")
TEST_AUTH = ("root", "root")


def _is_loopback_test_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_test_url(TEST_SURREALDB_URL),
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL",
    ),
]


class _Progress:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {"strategy_states": {"semantic_link": {}}}
        self.fail_before_apply_checkpoint = True

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return self.state["strategy_states"].setdefault(strategy, {})

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
    ) -> None:
        if phase == "semantic_link_apply" and self.fail_before_apply_checkpoint:
            self.fail_before_apply_checkpoint = False
            raise ConsolidationPausedError("simulated post-write interruption")
        self.strategy_state(strategy).update(
            phase=phase,
            cursor=cursor,
            pending=list(pending or []),
            counters=dict(counters or {}),
        )


def _engine(store: SurrealDBStorage, progress: _Progress) -> ConsolidationEngine:
    engine = ConsolidationEngine(store, ConsolidationConfig())
    engine._active_strategy = ConsolidationStrategy.SEMANTIC_LINK
    engine._progress_session = progress  # type: ignore[assignment]
    return engine


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SurrealDBStorage]:
    assert TEST_SURREALDB_URL is not None
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace="smem_ci",
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-link-resume-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def test_semantic_link_resume_after_surreal_id_normalization_is_idempotent(
    store: SurrealDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    neurons = [
        Neuron.create(
            NeuronType.CONCEPT,
            f"synthetic semantic resume {suffix}",
        )
        for suffix in ("a", "b")
    ]
    for neuron in neurons:
        await store.add_neuron(neuron)
    planned = Synapse.create(
        neurons[0].id,
        neurons[1].id,
        SynapseType.SIMILAR_TO,
        weight=0.54,
        metadata={"_semantic_discovery": True},
        synapse_id="similar_to-" + uuid.uuid4().hex,
    )
    discovery_result = SemanticDiscoveryResult(
        neurons_embedded=2,
        pairs_evaluated=1,
        synapses_created=1,
        eligible_total=2,
        synapses=[planned],
    )

    async def discover(*_args: Any, **_kwargs: Any) -> SemanticDiscoveryResult:
        return discovery_result

    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery.discover_semantic_synapses", discover
    )
    progress = _Progress()
    with pytest.raises(ConsolidationPausedError, match="post-write interruption"):
        await _engine(store, progress)._semantic_link(ConsolidationReport(), dry_run=False)

    persisted = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    expected_public_id = planned.id.replace("_", "-")
    assert [edge.id for edge in persisted] == [expected_public_id]

    report = ConsolidationReport()
    await _engine(store, progress)._semantic_link(report, dry_run=False)

    persisted_after_resume = await store.get_synapses(type=SynapseType.SIMILAR_TO)
    assert [edge.id for edge in persisted_after_resume] == [expected_public_id]
    assert report.semantic_synapses_created == 1
    assert progress.strategy_state("semantic_link")["pending"] == []
