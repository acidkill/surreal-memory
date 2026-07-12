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
import threading
import time
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

_bridge_loop: asyncio.AbstractEventLoop | None = None
_bridge_lock = threading.Lock()


def _get_bridge_loop() -> asyncio.AbstractEventLoop:
    """A single, process-wide background event loop for the sync bridge.

    Every sync-entry coroutine runs on THIS loop (never a fresh loop per call), so
    storage's process-global ``asyncio.Lock`` objects stay bound to one loop. A
    new-loop-per-call bridge deadlocks under ``retriever.batch(...)`` — its worker
    threads each spin an independent loop and can contend on the same cached lock.
    """
    global _bridge_loop
    loop = _bridge_loop
    if loop is None:
        with _bridge_lock:
            loop = _bridge_loop
            if loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever,
                    name="surreal-memory-langchain-bridge",
                    daemon=True,
                )
                thread.start()
                _bridge_loop = loop
    return loop


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine from sync code on the shared background loop.

    Works whether or not the caller already has a running loop (the coroutine always
    executes on the dedicated bridge loop, never the caller's), and is safe under
    concurrent calls from ``Runnable.batch``'s worker threads.
    """
    return asyncio.run_coroutine_threadsafe(coro, _get_bridge_loop()).result()


def _normalize_tag(tag: str) -> str:
    """Match the encoder's default ``TagNormalizer`` (no synonyms → ``lower().strip()``).

    Storage back-ends filter tags by exact match, but ``MemoryEncoder`` lowercases every
    tag it stores. Applying the same transform on the read side keeps the session tag
    matching regardless of the ``session_id``'s original casing/whitespace.
    """
    return tag.lower().strip()


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
    # NOTE: named `memory_tags`, NOT `tags` — BaseRetriever/Runnable already defines a
    # `tags` field used for LangSmith/callback tracing tags. Shadowing it would leak these
    # recall-filter values into tracing and hijack the framework's own tag semantics.
    memory_tags: list[str] | None = None
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
            tags=set(self.memory_tags) if self.memory_tags else None,
            exclude_ephemeral=self.permanent_only,
        )

        docs: list[Document] = []
        for fiber_id in result.fibers_matched[: self.k]:
            fiber = await storage.get_fiber(fiber_id)
            if fiber is None:
                continue
            docs.append(await self._fiber_to_document(storage, fiber, result.confidence))

        if self.k > 0 and not docs and result.answer:
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
        # Normalized to the encoder's stored form so mixed-case/whitespace session ids
        # still match on read (else find_fibers' exact tag match returns nothing, and a
        # lowercase twin id would receive this session's turns).
        return _normalize_tag(f"{_LC_SESSION_PREFIX}{self.session_id}")

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
                # Monotonic write-order tiebreaker: fibers can share a time_start
                # microsecond under a tight add_messages() loop; nanosecond order keeps
                # replay stable (e.g. the human turn before its AI answer).
                "lc_seq": time.time_ns(),
            },
        )

    async def _aget_messages(self) -> list[BaseMessage]:
        storage = await self._aresolve_storage()
        fibers = await storage.find_fibers(tags={self._session_tag}, limit=1000)
        ordered = sorted(
            fibers, key=lambda f: (f.time_start or f.created_at, f.metadata.get("lc_seq", 0))
        )
        messages: list[BaseMessage] = []
        for fiber in ordered:
            # Exact session match (defence-in-depth beyond the tag) — never replay another
            # session's turns even if tag normalization ever collides.
            if fiber.metadata.get("lc_session") != self.session_id:
                continue
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
            if fiber.metadata.get("lc_session") != self.session_id:
                continue
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
