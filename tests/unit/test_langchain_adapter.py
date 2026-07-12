"""U9: LangChain adapter (SurrealMemoryRetriever + SurrealMemoryChatMessageHistory).

The whole file is skipped when langchain-core is absent (base suite stays green via the
optional extra). Storage is real InMemoryStorage injected through ``from_storage`` — the
retriever mapping runs through the REAL ReflexPipeline (the valid_at lesson: don't mock
the pipeline for a recall-backed feature); only the empty-brain fallback uses a stub.
"""

from __future__ import annotations

import asyncio
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
