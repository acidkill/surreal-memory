"""Example: a LangChain RAG chain backed by a surreal-memory brain.

Wires :class:`SurrealMemoryRetriever` into an LCEL chain and adds per-session memory via
:class:`SurrealMemoryChatMessageHistory` + ``RunnableWithMessageHistory``. The LLM call is
stubbed with a tiny fake so the file runs end-to-end with no API key — swap ``_FakeLLM``
for a real ``ChatOpenAI``/``ChatAnthropic`` in your app.

Run:  pip install surreal-memory[langchain]  &&  python examples/langchain_rag.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from surreal_memory.adapters.langchain import (
    SurrealMemoryChatMessageHistory,
    SurrealMemoryRetriever,
)
from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage


class _FakeLLM(Runnable):
    """Stand-in for a chat model — echoes the retrieved context deterministically."""

    def invoke(self, input: object, config: object | None = None, **_: object) -> BaseMessage:
        text = getattr(input, "to_string", lambda: str(input))()
        return AIMessage(content=f"(demo answer grounded in retrieved memories)\n{text[:200]}")


async def _seed(storage: InMemoryStorage, content: str) -> None:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=content)
    await storage.add_neuron(neuron)
    await storage.add_fiber(
        Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary=content,
        )
    )


async def build_storage() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="rag_demo")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    await _seed(storage, "The team chose SurrealDB as the primary datastore.")
    await _seed(storage, "The backend framework is FastAPI.")
    return storage


def _format_docs(docs: list) -> str:
    return "\n".join(f"- {d.page_content}" for d in docs)


async def main() -> None:
    storage = await build_storage()

    # In a real app: SurrealMemoryRetriever(brain_name="rag_demo", k=4) resolves storage
    # from your config. Here we inject the in-memory storage directly.
    retriever = SurrealMemoryRetriever.from_storage(storage, k=4)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using ONLY the context.\n\nContext:\n{context}"),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )

    chain: Runnable = (
        RunnablePassthrough.assign(
            context=(lambda x: x["question"]) | retriever | RunnableLambda(_format_docs)
        )
        | prompt
        | _FakeLLM()
        | StrOutputParser()
    )

    with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: SurrealMemoryChatMessageHistory(session_id, storage=storage),
        input_messages_key="question",
        history_messages_key="history",
    )

    cfg = {"configurable": {"session_id": "demo-session"}}
    print(await with_history.ainvoke({"question": "Which datastore did we pick?"}, config=cfg))
    print(await with_history.ainvoke({"question": "And the backend framework?"}, config=cfg))

    history = SurrealMemoryChatMessageHistory("demo-session", storage=storage)
    print("\nStored turns:")
    for message in history.messages:
        print(f"  {message.type}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
