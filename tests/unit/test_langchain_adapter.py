"""U9: LangChain adapter (SurrealMemoryRetriever + SurrealMemoryChatMessageHistory).

The whole file is skipped when langchain-core is absent (base suite stays green via the
optional extra). Storage is real InMemoryStorage injected through ``from_storage`` — the
retriever mapping runs through the REAL ReflexPipeline (the valid_at lesson: don't mock
the pipeline for a recall-backed feature); only the empty-brain fallback uses a stub.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from surreal_memory.adapters.langchain import (
    SurrealMemoryChatMessageHistory,
    SurrealMemoryRetriever,
)
from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval_types import (
    DepthLevel,
    RetrievalResult,
    Subgraph,
)
from surreal_memory.storage.memory_store import InMemoryStorage


async def _make_storage() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="lc_test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


async def _seed(storage: InMemoryStorage, content: str) -> Fiber:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=content,
    )
    await storage.add_fiber(fiber)
    return fiber


def _stub_result(answer: str) -> RetrievalResult:
    return RetrievalResult(
        answer=answer,
        confidence=0.5,
        depth_used=DepthLevel.INSTANT,
        neurons_activated=0,
        fibers_matched=[],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
        context=answer,
        latency_ms=0.0,
        tokens_used=0,
        metadata={},
        score_breakdown=None,
    )


def test_package_import_is_lazy() -> None:
    # Importing the package must not eagerly import langchain; symbols resolve on access.
    import surreal_memory.adapters as adapters

    assert hasattr(adapters, "SurrealMemoryRetriever")
    assert hasattr(adapters, "SurrealMemoryChatMessageHistory")
    with pytest.raises(AttributeError):
        _ = adapters.DoesNotExist


class TestRetriever:
    async def test_maps_fibers_to_documents(self) -> None:
        storage = await _make_storage()
        await _seed(storage, "Paris is the capital of France")
        retriever = SurrealMemoryRetriever.from_storage(storage, k=5)

        docs = await retriever.ainvoke("capital of France Paris")

        assert docs
        assert all(isinstance(d, Document) for d in docs)
        assert any("Paris" in d.page_content for d in docs)
        assert docs[0].metadata["source"] == "surreal-memory"
        assert "fiber_id" in docs[0].metadata

    async def test_k_caps_result_count(self) -> None:
        storage = await _make_storage()
        for i in range(4):
            await _seed(storage, f"Fact {i} about the capital cities of France and Europe")
        retriever = SurrealMemoryRetriever.from_storage(storage, k=2)

        docs = await retriever.ainvoke("capital cities France Europe")

        assert len(docs) <= 2

    async def test_fallback_to_answer_when_no_fibers(self) -> None:
        storage = await _make_storage()
        retriever = SurrealMemoryRetriever.from_storage(storage)

        stub = _stub_result("A synthesised answer with no backing fibers.")

        class _FakePipeline:
            def __init__(self, *_: Any, **__: Any) -> None: ...

            async def query(self, *_: Any, **__: Any) -> RetrievalResult:
                return stub

        with patch("surreal_memory.engine.retrieval.ReflexPipeline", _FakePipeline):
            docs = await retriever.ainvoke("anything")

        assert len(docs) == 1
        assert docs[0].page_content == "A synthesised answer with no backing fibers."
        assert docs[0].metadata["fallback"] is True

    async def test_invoke_within_running_loop_uses_thread_bridge(self) -> None:
        # Called with a loop already running → sync bridge must offload to a thread.
        storage = await _make_storage()
        await _seed(storage, "Rome is the capital of Italy")
        retriever = SurrealMemoryRetriever.from_storage(storage)

        docs = retriever.invoke("capital of Italy Rome")

        assert any("Rome" in d.page_content for d in docs)


def test_invoke_sync_without_running_loop() -> None:
    # No running loop → sync bridge uses asyncio.run directly.
    storage = asyncio.run(_make_storage())
    asyncio.run(_seed(storage, "Berlin is the capital of Germany"))
    retriever = SurrealMemoryRetriever.from_storage(storage)

    docs = retriever.invoke("capital of Germany Berlin")

    assert any("Berlin" in d.page_content for d in docs)


async def _noop() -> int:
    return 42


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_bridge_loop_survives_fork() -> None:
    # Regression: the shared bridge loop must be reset in a forked child. Warm it up in
    # the PARENT first (the dangerous ordering: gunicorn --preload + warmup), then fork
    # and run a sync call in the child. Without os.register_at_fork the child inherits a
    # dead loop and hangs forever.
    from surreal_memory.adapters import langchain as lc

    assert lc._run_sync(_noop()) == 42  # create the bridge loop in the parent

    pid = os.fork()
    if pid == 0:  # child
        try:
            os._exit(0 if lc._run_sync(_noop()) == 42 else 2)
        except BaseException:
            os._exit(3)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    pytest.fail("child hung after fork — bridge loop was not reset")


def test_batch_across_threads_does_not_hang() -> None:
    # Runnable.batch runs .invoke() from concurrent worker threads. The shared background
    # bridge loop must serialise them without the cross-loop deadlock a new-loop-per-call
    # bridge causes (the reviewer's HIGH #3). A short timeout guards against a regression.
    storage = asyncio.run(_make_storage())
    asyncio.run(_seed(storage, "Vienna is the capital of Austria"))
    retriever = SurrealMemoryRetriever.from_storage(storage)

    results = retriever.batch(["capital of Austria Vienna"] * 4)

    assert len(results) == 4
    assert all(any("Vienna" in d.page_content for d in docs) for docs in results)


class TestChatMessageHistory:
    def test_roundtrip_preserves_order_and_types(self) -> None:
        storage = asyncio.run(_make_storage())
        history = SurrealMemoryChatMessageHistory("s1", storage=storage)

        history.add_message(SystemMessage(content="You are helpful."))
        history.add_user_message("What is the capital of France?")
        history.add_ai_message("Paris.")

        messages = history.messages
        assert [m.content for m in messages] == [
            "You are helpful.",
            "What is the capital of France?",
            "Paris.",
        ]
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)

    def test_sessions_are_isolated(self) -> None:
        storage = asyncio.run(_make_storage())
        h1 = SurrealMemoryChatMessageHistory("s1", storage=storage)
        h2 = SurrealMemoryChatMessageHistory("s2", storage=storage)

        h1.add_user_message("in session one")
        h2.add_user_message("in session two")

        assert [m.content for m in h1.messages] == ["in session one"]
        assert [m.content for m in h2.messages] == ["in session two"]

    def test_clear_empties_the_session(self) -> None:
        storage = asyncio.run(_make_storage())
        history = SurrealMemoryChatMessageHistory("s3", storage=storage)
        history.add_user_message("temporary")
        assert history.messages

        history.clear()

        assert history.messages == []

    def test_mixed_case_session_id_roundtrips_and_isolates(self) -> None:
        # Regression: the encoder lowercases stored tags. A mixed-case session id must
        # still round-trip (the read tag is normalized the same way), and a lowercase
        # "twin" id must NOT receive this session's turns (exact lc_session metadata).
        storage = asyncio.run(_make_storage())
        mixed = SurrealMemoryChatMessageHistory("Session-ABC", storage=storage)
        twin = SurrealMemoryChatMessageHistory("session-abc", storage=storage)

        mixed.add_user_message("from the mixed-case session")
        twin.add_user_message("from the lowercase twin")

        assert [m.content for m in mixed.messages] == ["from the mixed-case session"]
        assert [m.content for m in twin.messages] == ["from the lowercase twin"]

    def test_message_order_stable_on_timestamp_tie(self) -> None:
        # Regression: turns sharing a time_start microsecond must still replay in write
        # order via the monotonic lc_seq tiebreaker (else a tight add_messages loop could
        # replay an AI answer before its human question).
        storage = asyncio.run(_make_storage())
        history = SurrealMemoryChatMessageHistory("s-order", storage=storage)
        tied = datetime(2026, 1, 1, 12, 0, 0)
        tag = history._session_tag

        async def _insert(content: str, seq: int) -> None:
            neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
            await storage.add_neuron(neuron)
            fiber = Fiber.create(
                neuron_ids={neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=neuron.id,
                summary=content,
                agent_tags={tag},
                time_start=tied,
                time_end=tied,
                metadata={
                    "lc_role": "human",
                    "lc_session": "s-order",
                    "lc_content": content,
                    "lc_seq": seq,
                },
            )
            await storage.add_fiber(fiber)

        # Insert out of order; lc_seq is the intended replay order.
        asyncio.run(_insert("third", 3))
        asyncio.run(_insert("first", 1))
        asyncio.run(_insert("second", 2))

        assert [m.content for m in history.messages] == ["first", "second", "third"]


class TestRetrieverTagField:
    def test_memory_tags_does_not_shadow_langchain_tags(self) -> None:
        # Regression: the recall-filter field must be `memory_tags`, leaving LangChain's
        # inherited `tags` (tracing/callback tags) intact and independent.
        storage = asyncio.run(_make_storage())
        retriever = SurrealMemoryRetriever.from_storage(
            storage, tags=["trace-tag"], memory_tags=["recall-tag"]
        )
        assert retriever.tags == ["trace-tag"]  # LangChain's own field, untouched
        assert retriever.memory_tags == ["recall-tag"]
