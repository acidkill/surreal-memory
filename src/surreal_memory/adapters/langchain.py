"""LangChain adapter: a retriever + chat-message history backed by a surreal-memory brain.

In-process Python API (NOT the REST server). Two building blocks:

- :class:`SurrealMemoryRetriever` — a ``BaseRetriever`` that runs the reflex recall
  pipeline and maps matched fibers to LangChain ``Document`` objects.
- :class:`SurrealMemoryChatMessageHistory` — a ``BaseChatMessageHistory`` that persists
  chat turns as memories (tagged per session) and replays them verbatim.

Requires the ``langchain`` extra (``pip install surreal-memory[langchain]``). Everything
is async underneath; the sync ``BaseRetriever``/``BaseChatMessageHistory`` entry points
bridge to async via :func:`_run_sync`. Prefer the native async paths where you can.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.retrievers import BaseRetriever
    from pydantic import PrivateAttr
except ImportError as exc:  # pragma: no cover - exercised via the import-guard test
    raise ImportError(
        "The LangChain adapter requires 'langchain-core'. "
        "Install it with: pip install surreal-memory[langchain]"
    ) from exc

from surreal_memory.core.brain import BrainConfig

if TYPE_CHECKING:
    from collections.abc import Coroutine

    # Type-only: used solely in method annotations (stringised by `from __future__`), so
    # they must not sit in the runtime import guard.
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )

    from surreal_memory.core.fiber import Fiber
    from surreal_memory.storage.base import NeuralStorage

_LC_SESSION_PREFIX = "lc-session:"
_LC_TAG = "langchain"
_DEFAULT_K = 5


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine from sync code, safely whether or not a loop is running.

    No running loop → ``asyncio.run``. A loop already running (e.g. inside Jupyter or an
    async web handler) → run in a dedicated one-shot thread so we never re-enter a loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def _current_brain_config(storage: NeuralStorage) -> BrainConfig:
    """Best-effort brain config for the storage's current brain (fallback: defaults)."""
    brain_id = getattr(storage, "_current_brain_id", None) or getattr(storage, "brain_id", None)
    if brain_id:
        brain = await storage.get_brain(brain_id)
        if brain is not None:
            return brain.config
    return BrainConfig()


def _message_role(message: BaseMessage) -> str:
    """Map a LangChain message to a stored role token."""
    return {"human": "human", "ai": "ai", "system": "system"}.get(message.type, message.type)


def _message_from_stored(role: str, content: str) -> BaseMessage:
    """Rebuild a LangChain message from a stored role + verbatim content."""
    if role == "ai":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class SurrealMemoryRetriever(BaseRetriever):
    """Retrieve documents from a surreal-memory brain via the reflex recall pipeline.

    Construct with :meth:`from_storage` (dependency injection — used in tests and when you
    already hold a storage handle) or plainly (``SurrealMemoryRetriever(brain_name=...)``),
    in which case storage is resolved lazily from the shared config on first use.
    """

    brain_name: str | None = None
    depth: int = 1
    max_tokens: int = 500
    tags: list[str] | None = None
    permanent_only: bool = False
    k: int = _DEFAULT_K

    _storage: NeuralStorage | None = PrivateAttr(default=None)

    @classmethod
    def from_storage(cls, storage: NeuralStorage, **fields: Any) -> SurrealMemoryRetriever:
        """Build a retriever bound to an existing storage handle (DI)."""
        retriever = cls(**fields)
        retriever._storage = storage
        return retriever

    async def _aresolve_storage(self) -> NeuralStorage:
        if self._storage is None:
            from surreal_memory.unified_config import get_shared_storage

            self._storage = await get_shared_storage(self.brain_name)
        return self._storage

    async def _afetch(self, query: str) -> list[Document]:
        from surreal_memory.engine.retrieval import ReflexPipeline
        from surreal_memory.engine.retrieval_types import DepthLevel

        storage = await self._aresolve_storage()
        config = await _current_brain_config(storage)
        pipeline = ReflexPipeline(storage, config)
        try:
            depth = DepthLevel(self.depth)
        except ValueError:
            depth = DepthLevel.CONTEXT
        result = await pipeline.query(
            query=query,
            depth=depth,
            max_tokens=self.max_tokens,
            tags=set(self.tags) if self.tags else None,
            exclude_ephemeral=self.permanent_only,
        )

        docs: list[Document] = []
        for fiber_id in result.fibers_matched[: self.k]:
            fiber = await storage.get_fiber(fiber_id)
            if fiber is None:
                continue
            docs.append(await self._fiber_to_document(storage, fiber, result.confidence))

        if not docs and result.answer:
            # No fibers surfaced but the pipeline produced an answer — keep it as context.
            return [
                Document(
                    page_content=result.answer,
                    metadata={"source": "surreal-memory", "fallback": True},
                )
            ]
        return docs

    async def _fiber_to_document(
        self, storage: NeuralStorage, fiber: Fiber, confidence: float
    ) -> Document:
        page_content = await self._page_content(storage, fiber)
        metadata: dict[str, Any] = {
            "fiber_id": fiber.id,
            "memory_type": fiber.metadata.get("type"),
            "tags": sorted(fiber.tags),
            "salience": fiber.salience,
            "confidence": confidence,
            "source": "surreal-memory",
        }
        created_at = getattr(fiber, "created_at", None)
        if created_at is not None:
            metadata["created_at"] = created_at.isoformat()
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    async def _page_content(storage: NeuralStorage, fiber: Fiber) -> str:
        """anchor-neuron content → fiber summary → fiber essence (first non-empty)."""
        anchor_content = None
        if fiber.anchor_neuron_id:
            anchor = await storage.get_neuron(fiber.anchor_neuron_id)
            anchor_content = getattr(anchor, "content", None) if anchor is not None else None
        for candidate in (anchor_content, fiber.summary, fiber.essence):
            if candidate:
                return str(candidate)
        return ""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return cast("list[Document]", _run_sync(self._afetch(query)))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await self._afetch(query)


class SurrealMemoryChatMessageHistory(BaseChatMessageHistory):
    """Persist LangChain chat turns as surreal-memory fibers, one session per id.

    Each turn is stored via the memory encoder with tags ``{"langchain",
    "lc-session:<id>"}`` and the raw text preserved in ``metadata["lc_content"]`` (the
    encoder sanitises neuron content, so the verbatim message is kept in metadata for
    lossless replay). ``clear()`` deletes the session's fibers; orphaned neurons are left
    to the normal lifecycle/decay (documented limitation).
    """

    def __init__(
        self,
        session_id: str,
        *,
        storage: NeuralStorage | None = None,
        brain_name: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.brain_name = brain_name
        self._storage = storage

    async def _aresolve_storage(self) -> NeuralStorage:
        if self._storage is None:
            from surreal_memory.unified_config import get_shared_storage

            self._storage = await get_shared_storage(self.brain_name)
        return self._storage

    @property
    def _session_tag(self) -> str:
        return f"{_LC_SESSION_PREFIX}{self.session_id}"

    async def _aadd_message(self, message: BaseMessage) -> None:
        from surreal_memory.engine.encoder import MemoryEncoder

        storage = await self._aresolve_storage()
        config = await _current_brain_config(storage)
        encoder = MemoryEncoder(storage, config)
        content = str(message.content)
        await encoder.encode(
            content=content,
            tags={_LC_TAG, self._session_tag},
            metadata={
                "lc_role": _message_role(message),
                "lc_session": self.session_id,
                "lc_content": content,
            },
        )

    async def _aget_messages(self) -> list[BaseMessage]:
        storage = await self._aresolve_storage()
        fibers = await storage.find_fibers(tags={self._session_tag}, limit=1000)
        ordered = sorted(fibers, key=lambda f: f.time_start or f.created_at)
        messages: list[BaseMessage] = []
        for fiber in ordered:
            role = fiber.metadata.get("lc_role")
            content = fiber.metadata.get("lc_content")
            if role is None or content is None:
                continue
            messages.append(_message_from_stored(str(role), str(content)))
        return messages

    async def _aclear(self) -> None:
        storage = await self._aresolve_storage()
        fibers = await storage.find_fibers(tags={self._session_tag}, limit=1000)
        for fiber in fibers:
            await storage.delete_fiber(fiber.id)

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        # BaseChatMessageHistory annotates `messages` as a writeable attribute; we expose
        # it as a computed read-only property (each read replays from storage).
        return cast("list[BaseMessage]", _run_sync(self._aget_messages()))

    def add_message(self, message: BaseMessage) -> None:
        _run_sync(self._aadd_message(message))

    def clear(self) -> None:
        _run_sync(self._aclear())
