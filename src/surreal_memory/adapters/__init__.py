"""Outbound framework adapters — expose surreal-memory to external LLM frameworks.

Distinct from :mod:`surreal_memory.integration.adapters`, which is *inbound* ingestion
(pulling other memory systems IN via ``SourceAdapter``). This package points the other
way: it wraps a surreal-memory brain as building blocks for external frameworks —
currently a LangChain retriever + chat-message history.

Optional backends are import-guarded, so ``import surreal_memory.adapters`` never fails
even when LangChain is not installed. The guard fires only when you actually touch a
symbol whose backend package is missing (lazy ``__getattr__`` below), with a message
telling you which extra to install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["SurrealMemoryChatMessageHistory", "SurrealMemoryRetriever"]

if TYPE_CHECKING:
    from surreal_memory.adapters.langchain import (
        SurrealMemoryChatMessageHistory,
        SurrealMemoryRetriever,
    )


def __getattr__(name: str) -> Any:
    """Lazily import LangChain-backed symbols so the package stays import-safe."""
    if name in __all__:
        from surreal_memory.adapters import langchain

        return getattr(langchain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
