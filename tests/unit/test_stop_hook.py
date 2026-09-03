"""Tests for stop hook role filtering, memory markers, and embedding dedup."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.hooks.stop import (
    _embedding_dedup,
    _get_entry_role,
    _has_memory_markers,
    read_transcript_tail,
)


class TestGetEntryRole:
    """Tests for _get_entry_role()."""

    def test_direct_user_role(self) -> None:
        assert _get_entry_role({"role": "user", "content": "hello"}) == "user"

    def test_direct_assistant_role(self) -> None:
        assert _get_entry_role({"role": "assistant", "content": "hi"}) == "assistant"

    def test_direct_tool_role(self) -> None:
        assert _get_entry_role({"role": "tool", "content": "result"}) == "tool"

    def test_nested_message_role(self) -> None:
        entry = {"message": {"role": "assistant", "content": "nested"}}
        assert _get_entry_role(entry) == "assistant"

    def test_tool_result_type(self) -> None:
        entry = {"type": "tool_result", "content": "data"}
        assert _get_entry_role(entry) == "tool"

    def test_tool_use_type(self) -> None:
        entry = {"type": "tool_use", "name": "read", "input": {}}
        assert _get_entry_role(entry) == "tool"

    def test_tool_use_id_present(self) -> None:
        entry = {"tool_use_id": "abc123", "content": "result"}
        assert _get_entry_role(entry) == "tool"

    def test_content_list_with_tool_use(self) -> None:
        entry = {
            "content": [
                {"type": "text", "text": "Let me read that"},
                {"type": "tool_use", "name": "read", "input": {}},
            ]
        }
        assert _get_entry_role(entry) == "tool"

    def test_content_list_with_tool_result(self) -> None:
        entry = {"content": [{"type": "tool_result", "tool_use_id": "x"}]}
        assert _get_entry_role(entry) == "tool"

    def test_unknown_entry_defaults_to_user(self) -> None:
        assert _get_entry_role({"text": "some text"}) == "user"

    def test_empty_entry_defaults_to_user(self) -> None:
        assert _get_entry_role({}) == "user"

    def test_content_list_text_only_defaults_to_user(self) -> None:
        entry = {"content": [{"type": "text", "text": "just text"}]}
        assert _get_entry_role(entry) == "user"


class TestHasMemoryMarkers:
    """Tests for _has_memory_markers()."""

    def test_decision_marker(self) -> None:
        assert _has_memory_markers("We decided to use SQLite for storage")

    def test_chose_marker(self) -> None:
        assert _has_memory_markers("I chose React over Vue")

    def test_root_cause_marker(self) -> None:
        assert _has_memory_markers("The root cause was a race condition")

    def test_fixed_marker(self) -> None:
        assert _has_memory_markers("Fixed the import error in server.py")

    def test_insight_marker(self) -> None:
        assert _has_memory_markers("Turns out the config was wrong")

    def test_todo_marker(self) -> None:
        assert _has_memory_markers("TODO: add retry logic for API calls")

    def test_preference_marker(self) -> None:
        assert _has_memory_markers("I prefer using async/await everywhere")

    def test_version_marker(self) -> None:
        assert _has_memory_markers("Released v2.21.0 with cross-language support")

    def test_shipped_marker(self) -> None:
        assert _has_memory_markers("Successfully shipped the new feature")

    def test_committed_marker(self) -> None:
        assert _has_memory_markers("Committed the changes to main branch")

    def test_no_markers_generic_text(self) -> None:
        assert not _has_memory_markers("Let me read the file for you")

    def test_no_markers_tool_description(self) -> None:
        assert not _has_memory_markers("I will use the Edit tool to modify this")

    def test_no_markers_short_response(self) -> None:
        assert not _has_memory_markers("Sure, here it is")

    def test_no_markers_code_output(self) -> None:
        assert not _has_memory_markers("The function returns a list of strings")


class TestReadTranscriptTailFiltering:
    """Tests for role-based filtering in read_transcript_tail()."""

    def test_skips_tool_results(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        p = Path(str(tmp_path)) / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "What is the root cause of the bug?"}),
            json.dumps({"role": "tool", "content": "File contents: lots of code here blah blah"}),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "The root cause was a missing null check in the handler",
                }
            ),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

        result = read_transcript_tail(str(p))
        assert "root cause" in result.lower()
        assert "File contents:" not in result

    def test_skips_assistant_without_markers(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        p = Path(str(tmp_path)) / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "Can you help me with this feature?"}),
            json.dumps(
                {"role": "assistant", "content": "Let me read the file and check the code for you"}
            ),
            json.dumps(
                {
                    "role": "assistant",
                    "content": "I decided to use the factory pattern for this module",
                }
            ),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

        result = read_transcript_tail(str(p))
        # User message included
        assert "help me with this feature" in result
        # Generic assistant response filtered out
        assert "read the file and check" not in result
        # Decision-bearing assistant response included
        assert "factory pattern" in result

    def test_includes_all_user_messages(self, tmp_path: object) -> None:
        import json
        from pathlib import Path

        p = Path(str(tmp_path)) / "transcript.jsonl"
        lines = [
            json.dumps({"role": "user", "content": "First instruction about the project setup"}),
            json.dumps(
                {"role": "user", "content": "Second instruction about the API design choices"}
            ),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")

        result = read_transcript_tail(str(p))
        assert "project setup" in result
        assert "API design" in result


class TestEmbeddingDedup:
    """Tests for _embedding_dedup()."""

    @pytest.mark.asyncio
    async def test_single_item_passthrough(self) -> None:
        items = [{"content": "Decision: use SQLite", "confidence": 0.9, "type": "decision"}]
        result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_empty_list_passthrough(self) -> None:
        result = await _embedding_dedup([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_provider_returns_original(self) -> None:
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]
        with patch(
            "surreal_memory.engine.semantic_discovery._auto_detect_provider",
            side_effect=RuntimeError("no provider"),
        ):
            result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_removes_semantic_duplicates(self) -> None:
        items = [
            {"content": "Decided to use React for frontend", "confidence": 0.9, "type": "decision"},
            {"content": "Chose React for the frontend UI", "confidence": 0.8, "type": "decision"},
            {"content": "Fixed the auth bug in login", "confidence": 0.85, "type": "error"},
        ]

        mock_provider = AsyncMock()
        # Embeddings: items 0 and 1 are near-duplicates, item 2 is different
        mock_provider.embed_batch = AsyncMock(
            return_value=[[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 1.0, 0.0]]
        )

        async def mock_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        mock_provider.similarity = mock_similarity

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("ollama", "bge-m3"),
            ),
            patch(
                "surreal_memory.engine.embedding.ollama_embedding.OllamaEmbedding",
                return_value=mock_provider,
            ),
        ):
            result = await _embedding_dedup(items)

        # Should keep item 0 (higher confidence) and item 2, remove item 1
        assert len(result) == 2
        contents = [r["content"] for r in result]
        assert "Decided to use React for frontend" in contents
        assert "Fixed the auth bug in login" in contents
        assert "Chose React for the frontend UI" not in contents

    @pytest.mark.asyncio
    async def test_keeps_all_when_no_duplicates(self) -> None:
        items = [
            {"content": "Decision about database", "confidence": 0.9, "type": "decision"},
            {"content": "Error in auth module", "confidence": 0.85, "type": "error"},
            {"content": "Insight about caching", "confidence": 0.8, "type": "insight"},
        ]

        mock_provider = AsyncMock()
        mock_provider.embed_batch = AsyncMock(
            return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        async def mock_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        mock_provider.similarity = mock_similarity

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("ollama", "bge-m3"),
            ),
            patch(
                "surreal_memory.engine.embedding.ollama_embedding.OllamaEmbedding",
                return_value=mock_provider,
            ),
        ):
            result = await _embedding_dedup(items)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_skips_api_providers(self) -> None:
        """API-based providers (gemini, openai) are skipped to avoid rate limits."""
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]
        with patch(
            "surreal_memory.engine.semantic_discovery._auto_detect_provider",
            return_value=("gemini", "text-embedding-004"),
        ):
            result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_embed_failure_returns_original(self) -> None:
        """If embedding fails, return original list gracefully."""
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]

        mock_provider = AsyncMock()
        mock_provider.embed_batch = AsyncMock(side_effect=RuntimeError("model not found"))

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("ollama", "bge-m3"),
            ),
            patch(
                "surreal_memory.engine.embedding.ollama_embedding.OllamaEmbedding",
                return_value=mock_provider,
            ),
        ):
            result = await _embedding_dedup(items)
        assert result == items

    @pytest.mark.asyncio
    async def test_does_not_load_sentence_transformer_model(self) -> None:
        """The Stop hook must never instantiate the heavy sentence-transformers
        model (torch + a multi-hundred-MB download) — it runs in a fresh,
        uncached process on every session save and that is the dominant latency.

        When auto-detect resolves to sentence_transformer, the hook returns the
        items unchanged instead of loading the model.
        """
        items = [
            {"content": "Item A", "confidence": 0.8, "type": "fact"},
            {"content": "Item B", "confidence": 0.7, "type": "fact"},
        ]

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("sentence_transformer", "all-MiniLM-L6-v2"),
            ),
            patch(
                "surreal_memory.engine.embedding.sentence_transformer.SentenceTransformerEmbedding"
            ) as mock_st,
        ):
            result = await _embedding_dedup(items)

        mock_st.assert_not_called()
        assert result == items


class TestEmbeddingDedupRemoteOptIn:
    """The stop hook's dedup gate follows the same opt-in as the distiller.

    Default: only a loopback embedding endpoint is used. With
    ``reasoning_training.allow_remote_endpoints`` set, the configured remote
    endpoint qualifies too — without the opt-in the hook must keep skipping
    the provider entirely (dedup degrades to simhash, never leaks content).
    """

    ITEMS = [
        {"content": "Decision: use SQLite", "confidence": 0.9, "type": "decision"},
        {"content": "Decision: use SQ Lite", "confidence": 0.8, "type": "decision"},
    ]

    @staticmethod
    def _config(allow_remote: bool) -> object:
        from surreal_memory.unified_config import (
            EmbeddingSettings,
            ReasoningTrainingConfig,
            UnifiedConfig,
        )

        return UnifiedConfig(
            embedding=EmbeddingSettings(
                enabled=True,
                provider="openai",
                model="bge-m3",
                endpoint="https://litellm.example.com/v1",
            ),
            reasoning_training=ReasoningTrainingConfig(allow_remote_endpoints=allow_remote),
        )

    @pytest.mark.asyncio
    async def test_remote_endpoint_used_when_opt_in_is_set(self) -> None:
        mock_provider = AsyncMock()
        mock_provider.embed_batch = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
        mock_provider.similarity = AsyncMock(return_value=0.0)

        with (
            patch(
                "surreal_memory.unified_config.get_config",
                return_value=self._config(allow_remote=True),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("openai", "bge-m3"),
            ),
            patch(
                "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding",
                return_value=mock_provider,
            ) as mock_openai,
        ):
            result = await _embedding_dedup(self.ITEMS)

        mock_openai.assert_called_once()
        # The endpoint that cleared the gate is the endpoint the client gets.
        _, kwargs = mock_openai.call_args
        assert kwargs.get("base_url") == "https://litellm.example.com/v1"
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_remote_endpoint_skipped_without_the_opt_in(self) -> None:
        with (
            patch(
                "surreal_memory.unified_config.get_config",
                return_value=self._config(allow_remote=False),
            ),
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("openai", "bge-m3"),
            ),
            patch(
                "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding"
            ) as mock_openai,
        ):
            result = await _embedding_dedup(self.ITEMS)

        mock_openai.assert_not_called()
        assert result == self.ITEMS

    @pytest.mark.asyncio
    async def test_unreadable_config_falls_back_to_strict_gate(self) -> None:
        # get_config() raising must behave like the old code: env endpoint
        # only, loopback rule absolute, remote endpoint skipped.
        import os

        with (
            patch(
                "surreal_memory.unified_config.get_config",
                side_effect=RuntimeError("config unavailable"),
            ),
            patch.dict(os.environ, {"SURREAL_MEMORY_EMBEDDING_ENDPOINT": ""}, clear=False),
            patch(
                "surreal_memory.engine.semantic_discovery._auto_detect_provider",
                return_value=("openai", "bge-m3"),
            ),
            patch(
                "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding"
            ) as mock_openai,
        ):
            result = await _embedding_dedup(self.ITEMS)

        mock_openai.assert_not_called()
        assert result == self.ITEMS
