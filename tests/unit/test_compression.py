"""Unit tests for Tiered Memory Compression feature.

Covers:
- split_sentences helper
- compute_entity_density helper
- compress_tier1_extractive
- compress_tier2_entity_preserving
- compress_tier3_template
- CompressionTier enum
- CompressionConfig dataclass
- CompressionEngine.determine_target_tier
- SQLiteCompressionMixin storage (save/get/delete/stats)
- ConsolidationStrategy.COMPRESS integration
- ConsolidationReport compression fields
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.compression import (
    CompressionConfig,
    CompressionEngine,
    CompressionTier,
    compress_tier1_extractive,
    compress_tier2_entity_preserving,
    compress_tier3_template,
    compute_entity_density,
    split_sentences,
)
from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.simhash import simhash
from surreal_memory.utils.timeutils import utcnow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fiber(
    *,
    days_old: float = 0.0,
    compression_tier: int = 0,
) -> Fiber:
    """Create a minimal Fiber with a specific age and compression tier."""
    anchor_id = "anchor-1"
    created = utcnow() - timedelta(days=days_old)
    fiber = Fiber(
        id="fiber-1",
        neuron_ids={anchor_id},
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        compression_tier=compression_tier,
        created_at=created,
    )
    return fiber


# ---------------------------------------------------------------------------
# split_sentences tests
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic_split(self) -> None:
        sentences = split_sentences("Hello world. Goodbye world.")
        assert len(sentences) == 2

    def test_no_split_single(self) -> None:
        sentences = split_sentences("Hello world")
        assert len(sentences) == 1

    def test_empty_string(self) -> None:
        sentences = split_sentences("")
        assert sentences == []

    def test_question_exclamation(self) -> None:
        sentences = split_sentences("What? Yes!")
        assert len(sentences) == 2

    def test_preserves_content(self) -> None:
        text = "The cat sat. The dog ran."
        sentences = split_sentences(text)
        # Each sentence content must appear verbatim in the original text
        for sentence in sentences:
            assert sentence in text

    def test_strips_leading_trailing_whitespace(self) -> None:
        sentences = split_sentences("  Hello world.  ")
        assert len(sentences) == 1
        assert sentences[0] == "Hello world."

    def test_abbreviation_not_split(self) -> None:
        # "Dr." should not be treated as a sentence boundary
        text = "Dr. Smith is here. He is great."
        sentences = split_sentences(text)
        # Should get 2 sentences (not split at "Dr.")
        assert len(sentences) == 2

    def test_multiple_sentences(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        assert len(sentences) == 3


# ---------------------------------------------------------------------------
# compute_entity_density tests
# ---------------------------------------------------------------------------


class TestComputeEntityDensity:
    def test_no_entities_returns_zero(self) -> None:
        score = compute_entity_density("hello world", [])
        assert score == 0.0

    def test_no_matching_neurons_returns_zero(self) -> None:
        score = compute_entity_density("hello world", ["Alice", "Bob"])
        assert score == 0.0

    def test_all_match(self) -> None:
        # Every neuron content is found in the sentence
        sentence = "Alice and Bob met"
        neurons = ["Alice", "Bob"]
        score = compute_entity_density(sentence, neurons)
        assert score > 0.0

    def test_partial_match(self) -> None:
        # Only one of two neurons matches
        sentence = "Alice went shopping"
        score_one = compute_entity_density(sentence, ["Alice"])
        score_two = compute_entity_density(sentence, ["Alice", "Bob"])
        # Both positive, and one-match score >= two-match score (more neurons = lower density)
        assert score_one > 0.0
        assert score_two > 0.0

    def test_empty_sentence_returns_zero(self) -> None:
        score = compute_entity_density("", ["Alice"])
        assert score == 0.0

    def test_case_insensitive(self) -> None:
        # Neuron content "Alice" matches "alice" in sentence
        score = compute_entity_density("alice went to the store", ["Alice"])
        assert score > 0.0

    def test_result_clamped_to_one(self) -> None:
        # Even with many entities the result should not exceed 1.0
        sentence = "a"
        neurons = ["a"] * 100
        score = compute_entity_density(sentence, neurons)
        assert score <= 1.0

    def test_empty_neuron_contents_skipped(self) -> None:
        # Empty strings in neuron_contents should not count as matches
        score = compute_entity_density("hello world", ["", "", ""])
        assert score == 0.0


# ---------------------------------------------------------------------------
# compress_tier1_extractive tests
# ---------------------------------------------------------------------------


class TestCompressTier1Extractive:
    def test_keeps_top_sentences(self) -> None:
        # Sentence with entity should score higher than sentence without
        content = "Alice went to the store. The sky is blue. Bob called Alice."
        neurons = ["Alice", "Bob"]
        config = CompressionConfig(tier1_max_sentences=2, preserve_first_sentence=False)
        compressed, _ = compress_tier1_extractive(content, neurons, config)
        # Sentences mentioning Alice/Bob should survive
        assert "Alice" in compressed or "Bob" in compressed

    def test_preserves_first_sentence_true(self) -> None:
        # With preserve_first_sentence=True the first sentence always survives
        content = "Intro sentence. Alice was here. Bob was there. Charlie was around."
        neurons = ["Alice", "Bob", "Charlie"]
        config = CompressionConfig(tier1_max_sentences=1, preserve_first_sentence=True)
        compressed, _ = compress_tier1_extractive(content, neurons, config)
        assert "Intro sentence" in compressed

    def test_preserves_first_sentence_false(self) -> None:
        # With preserve_first_sentence=False the first sentence may be dropped
        # if it has low entity density and max_sentences is small
        content = "No entities here. Alice was here. Bob was there."
        neurons = ["Alice", "Bob"]
        config = CompressionConfig(tier1_max_sentences=1, preserve_first_sentence=False)
        compressed, _ = compress_tier1_extractive(content, neurons, config)
        # Should pick a sentence with Alice or Bob (higher density)
        assert "Alice" in compressed or "Bob" in compressed

    def test_max_sentences_limit(self) -> None:
        content = "One. Two. Three. Four. Five. Six."
        config = CompressionConfig(tier1_max_sentences=2, preserve_first_sentence=False)
        compressed, _ = compress_tier1_extractive(content, [], config)
        # With no entities all scores are 0; result should still be bounded
        sentences_out = [s for s in compressed.split(".") if s.strip()]
        assert len(sentences_out) <= 2

    def test_short_content_unchanged(self) -> None:
        content = "Single sentence only."
        config = CompressionConfig()
        compressed, _ = compress_tier1_extractive(content, [], config)
        assert "Single sentence only" in compressed

    def test_empty_content(self) -> None:
        compressed, entities = compress_tier1_extractive("", [], CompressionConfig())
        assert compressed == ""
        assert entities == 0

    def test_returns_string_and_int(self) -> None:
        content = "Alice is here."
        compressed, entities = compress_tier1_extractive(content, ["Alice"], CompressionConfig())
        assert isinstance(compressed, str)
        assert isinstance(entities, int)


# ---------------------------------------------------------------------------
# compress_tier2_entity_preserving tests
# ---------------------------------------------------------------------------


class TestCompressTier2EntityPreserving:
    def test_keeps_entity_sentences(self) -> None:
        content = "No entities here. Alice was seen. Nothing relevant."
        neurons = ["Alice"]
        config = CompressionConfig(preserve_first_sentence=False)
        compressed, _ = compress_tier2_entity_preserving(content, neurons, [], config)
        assert "Alice" in compressed

    def test_drops_non_entity_sentences(self) -> None:
        content = "Boring sentence one. Alice appeared. Boring sentence two."
        neurons = ["Alice"]
        config = CompressionConfig(preserve_first_sentence=False)
        compressed, _ = compress_tier2_entity_preserving(content, neurons, [], config)
        # Boring sentences should not survive when preserve_first_sentence=False
        assert "Boring sentence two" not in compressed

    def test_preserves_first_sentence_flag(self) -> None:
        content = "First sentence with no entities. Alice was here."
        neurons = ["Alice"]
        config = CompressionConfig(preserve_first_sentence=True)
        compressed, _ = compress_tier2_entity_preserving(content, neurons, [], config)
        assert "First sentence" in compressed

    def test_fallback_when_nothing_survives(self) -> None:
        # If no sentence passes the density threshold, return original content
        content = "No entities at all."
        neurons = []
        config = CompressionConfig(preserve_first_sentence=False)
        compressed, entities = compress_tier2_entity_preserving(content, neurons, [], config)
        assert compressed == content
        assert entities == 0

    def test_empty_content(self) -> None:
        compressed, entities = compress_tier2_entity_preserving("", [], [], CompressionConfig())
        assert compressed == ""
        assert entities == 0

    def test_returns_string_and_int(self) -> None:
        content = "Alice is here."
        compressed, entities = compress_tier2_entity_preserving(
            content, ["Alice"], [], CompressionConfig()
        )
        assert isinstance(compressed, str)
        assert isinstance(entities, int)


# ---------------------------------------------------------------------------
# compress_tier3_template tests
# ---------------------------------------------------------------------------


class TestCompressTier3Template:
    def test_basic_template(self) -> None:
        entities = ["Alice", "Bob"]
        relations = ["knows"]
        text, count = compress_tier3_template(entities, relations)
        assert "Alice" in text
        assert "Bob" in text
        assert "knows" in text
        assert count == 2

    def test_empty_relations_uses_default(self) -> None:
        entities = ["Alice", "Bob"]
        text, count = compress_tier3_template(entities, [])
        # Should use the default "related_to" relation
        assert "related_to" in text
        assert count == 2

    def test_empty_entities_returns_empty_string(self) -> None:
        text, count = compress_tier3_template([], [])
        assert text == ""
        assert count == 0

    def test_single_entity_returns_entity_itself(self) -> None:
        text, count = compress_tier3_template(["Alice"], [])
        assert text == "Alice"
        assert count == 1

    def test_multiple_relations(self) -> None:
        entities = ["A", "B", "C"]
        relations = ["rel1", "rel2"]
        text, count = compress_tier3_template(entities, relations)
        assert "rel1" in text
        assert "rel2" in text
        assert count == 3

    def test_semicolon_joins_triples(self) -> None:
        entities = ["X", "Y", "Z"]
        relations = ["r1", "r2"]
        text, _ = compress_tier3_template(entities, relations)
        assert ";" in text

    def test_empty_neuron_contents_filtered(self) -> None:
        # Empty strings in neuron_contents should not become entities
        entities = ["", "Alice", ""]
        text, count = compress_tier3_template(entities, [])
        # Only "Alice" is a real entity
        assert count == 1
        assert text == "Alice"


# ---------------------------------------------------------------------------
# CompressionTier enum tests
# ---------------------------------------------------------------------------


class TestCompressionTierEnum:
    def test_full_value(self) -> None:
        assert CompressionTier.FULL == 0

    def test_extractive_value(self) -> None:
        assert CompressionTier.EXTRACTIVE == 1

    def test_entity_only_value(self) -> None:
        assert CompressionTier.ENTITY_ONLY == 2

    def test_template_value(self) -> None:
        assert CompressionTier.TEMPLATE == 3

    def test_graph_only_value(self) -> None:
        assert CompressionTier.GRAPH_ONLY == 4

    def test_ordering(self) -> None:
        assert CompressionTier.FULL < CompressionTier.EXTRACTIVE
        assert CompressionTier.EXTRACTIVE < CompressionTier.ENTITY_ONLY
        assert CompressionTier.ENTITY_ONLY < CompressionTier.TEMPLATE
        assert CompressionTier.TEMPLATE < CompressionTier.GRAPH_ONLY

    def test_int_enum(self) -> None:
        assert int(CompressionTier.EXTRACTIVE) == 1


# ---------------------------------------------------------------------------
# CompressionConfig tests
# ---------------------------------------------------------------------------


class TestCompressionConfig:
    def test_default_config_tier_days(self) -> None:
        cfg = CompressionConfig()
        assert cfg.tier1_days == 7.0
        assert cfg.tier2_days == 30.0
        assert cfg.tier3_days == 90.0
        assert cfg.tier4_days == 180.0

    def test_default_config_reasonable_values(self) -> None:
        cfg = CompressionConfig()
        assert cfg.tier1_max_sentences > 0
        assert cfg.preserve_first_sentence is True
        assert cfg.tier2_min_density >= 0.0

    def test_custom_thresholds(self) -> None:
        cfg = CompressionConfig(
            tier1_days=3.0,
            tier2_days=14.0,
            tier3_days=45.0,
            tier4_days=90.0,
            tier1_max_sentences=3,
            preserve_first_sentence=False,
            tier2_min_density=0.1,
        )
        assert cfg.tier1_days == 3.0
        assert cfg.tier2_days == 14.0
        assert cfg.tier3_days == 45.0
        assert cfg.tier4_days == 90.0
        assert cfg.tier1_max_sentences == 3
        assert cfg.preserve_first_sentence is False
        assert cfg.tier2_min_density == pytest.approx(0.1)

    def test_frozen(self) -> None:
        cfg = CompressionConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.tier1_days = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CompressionEngine.determine_target_tier tests
# ---------------------------------------------------------------------------


class TestDetermineTargetTier:
    def _engine(self, config: CompressionConfig | None = None) -> CompressionEngine:
        """Build a CompressionEngine with a stub storage (not needed for determine_target_tier)."""
        from unittest.mock import MagicMock

        storage = MagicMock()
        return CompressionEngine(storage, config)

    def test_recent_fiber_full(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=1.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.FULL

    def test_just_under_tier1_full(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=6.9)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.FULL

    def test_week_old_extractive(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=15.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.EXTRACTIVE

    def test_just_at_tier1_boundary_extractive(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=7.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.EXTRACTIVE

    def test_month_old_entity_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=60.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.ENTITY_ONLY

    def test_just_at_tier2_boundary_entity_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=30.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.ENTITY_ONLY

    def test_quarter_old_template(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=120.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.TEMPLATE

    def test_just_at_tier3_boundary_template(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=90.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.TEMPLATE

    def test_old_fiber_graph_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=200.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.GRAPH_ONLY

    def test_just_at_tier4_boundary_graph_only(self) -> None:
        engine = self._engine()
        fiber = _make_fiber(days_old=180.0)
        now = utcnow()
        tier = engine.determine_target_tier(fiber, now)
        assert tier == CompressionTier.GRAPH_ONLY

    def test_custom_thresholds_respected(self) -> None:
        config = CompressionConfig(tier1_days=1.0, tier2_days=5.0, tier3_days=10.0, tier4_days=20.0)
        engine = self._engine(config)
        now = utcnow()

        assert engine.determine_target_tier(_make_fiber(days_old=0.5), now) == CompressionTier.FULL
        assert (
            engine.determine_target_tier(_make_fiber(days_old=3.0), now)
            == CompressionTier.EXTRACTIVE
        )
        assert (
            engine.determine_target_tier(_make_fiber(days_old=7.0), now)
            == CompressionTier.ENTITY_ONLY
        )
        assert (
            engine.determine_target_tier(_make_fiber(days_old=15.0), now)
            == CompressionTier.TEMPLATE
        )
        assert (
            engine.determine_target_tier(_make_fiber(days_old=25.0), now)
            == CompressionTier.GRAPH_ONLY
        )

    def test_already_compressed_skip_logic(self) -> None:
        """Fiber already at EXTRACTIVE (tier=1) → target tier 1 means no compression needed."""
        engine = self._engine()
        # Fiber is 15 days old → target EXTRACTIVE (1), fiber is already at 1
        fiber = _make_fiber(days_old=15.0, compression_tier=1)
        now = utcnow()
        target = engine.determine_target_tier(fiber, now)
        # The engine returns target = EXTRACTIVE; caller must compare with fiber.compression_tier
        assert target == CompressionTier.EXTRACTIVE
        # Caller logic: int(target) <= fiber.compression_tier → skip
        assert int(target) <= fiber.compression_tier


# ---------------------------------------------------------------------------
# Storage tests (SQLiteCompressionMixin via InMemoryStorage)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sqlite_storage(tmp_path: Path) -> InMemoryStorage:
    """InMemoryStorage backed by a temp file, brain context set."""
    store = InMemoryStorage()
    brain = Brain.create(name="test-compression-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


class TestSQLiteCompressionMixin:
    @pytest.mark.asyncio
    async def test_save_and_get_backup(self, sqlite_storage: InMemoryStorage) -> None:
        """Round-trip: save a backup then retrieve it with matching fields."""
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-abc",
            original_content="Original content text here.",
            compression_tier=1,
            original_token_count=100,
            compressed_token_count=40,
        )
        result = await sqlite_storage.get_compression_backup("fiber-abc")
        assert result is not None
        assert result["fiber_id"] == "fiber-abc"
        assert result["original_content"] == "Original content text here."
        assert result["compression_tier"] == 1
        assert result["original_token_count"] == 100
        assert result["compressed_token_count"] == 40

    @pytest.mark.asyncio
    async def test_get_nonexistent_backup_returns_none(
        self, sqlite_storage: InMemoryStorage
    ) -> None:
        result = await sqlite_storage.get_compression_backup("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_backup_returns_true(self, sqlite_storage: InMemoryStorage) -> None:
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-del",
            original_content="Some content.",
            compression_tier=2,
            original_token_count=50,
            compressed_token_count=20,
        )
        deleted = await sqlite_storage.delete_compression_backup("fiber-del")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_backup_removes_row(self, sqlite_storage: InMemoryStorage) -> None:
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-gone",
            original_content="Content.",
            compression_tier=1,
            original_token_count=10,
            compressed_token_count=5,
        )
        await sqlite_storage.delete_compression_backup("fiber-gone")
        result = await sqlite_storage.get_compression_backup("fiber-gone")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_backup_returns_false(
        self, sqlite_storage: InMemoryStorage
    ) -> None:
        deleted = await sqlite_storage.delete_compression_backup("ghost-fiber")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_compression_stats_empty(self, sqlite_storage: InMemoryStorage) -> None:
        stats = await sqlite_storage.get_compression_stats()
        assert stats["total_backups"] == 0
        assert stats["by_tier"] == {}
        assert stats["total_tokens_saved"] == 0

    @pytest.mark.asyncio
    async def test_compression_stats_counts_by_tier(self, sqlite_storage: InMemoryStorage) -> None:
        await sqlite_storage.save_compression_backup(
            fiber_id="f1",
            original_content="Content 1.",
            compression_tier=1,
            original_token_count=100,
            compressed_token_count=40,
        )
        await sqlite_storage.save_compression_backup(
            fiber_id="f2",
            original_content="Content 2.",
            compression_tier=1,
            original_token_count=80,
            compressed_token_count=30,
        )
        await sqlite_storage.save_compression_backup(
            fiber_id="f3",
            original_content="Content 3.",
            compression_tier=2,
            original_token_count=60,
            compressed_token_count=10,
        )
        stats = await sqlite_storage.get_compression_stats()
        assert stats["total_backups"] == 3
        assert stats["by_tier"][1] == 2
        assert stats["by_tier"][2] == 1
        assert stats["total_tokens_saved"] == (100 - 40) + (80 - 30) + (60 - 10)

    @pytest.mark.asyncio
    async def test_upsert_backup_replaces_existing(self, sqlite_storage: InMemoryStorage) -> None:
        """Saving twice for the same fiber_id replaces the old backup."""
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-upsert",
            original_content="Old content.",
            compression_tier=1,
            original_token_count=50,
            compressed_token_count=20,
        )
        await sqlite_storage.save_compression_backup(
            fiber_id="fiber-upsert",
            original_content="New content.",
            compression_tier=2,
            original_token_count=60,
            compressed_token_count=15,
        )
        result = await sqlite_storage.get_compression_backup("fiber-upsert")
        assert result is not None
        assert result["original_content"] == "New content."
        assert result["compression_tier"] == 2

    @pytest.mark.asyncio
    async def test_brain_isolation(self, tmp_path: Path) -> None:
        """Backups stored under brain A are not visible under brain B."""
        store = InMemoryStorage()

        brain_a = Brain.create(name="brain-a")
        brain_b = Brain.create(name="brain-b")
        await store.save_brain(brain_a)
        await store.save_brain(brain_b)

        store.set_brain(brain_a.id)
        await store.save_compression_backup(
            fiber_id="shared-fiber",
            original_content="Secret content.",
            compression_tier=1,
            original_token_count=30,
            compressed_token_count=10,
        )

        store.set_brain(brain_b.id)
        result_b = await store.get_compression_backup("shared-fiber")
        assert result_b is None

        store.set_brain(brain_a.id)
        result_a = await store.get_compression_backup("shared-fiber")
        assert result_a is not None


# ---------------------------------------------------------------------------
# Consolidation integration tests
# ---------------------------------------------------------------------------


class TestConsolidationIntegration:
    def test_compress_strategy_exists(self) -> None:
        """ConsolidationStrategy must have a COMPRESS member."""
        assert hasattr(ConsolidationStrategy, "COMPRESS")

    def test_compress_strategy_value(self) -> None:
        """COMPRESS value is 'compress'."""
        assert ConsolidationStrategy.COMPRESS == "compress"

    def test_compress_in_strategy_values(self) -> None:
        """COMPRESS appears in the set of ConsolidationStrategy values."""
        all_values = {s.value for s in ConsolidationStrategy}
        assert "compress" in all_values

    def test_report_has_fibers_compressed(self) -> None:
        report = ConsolidationReport()
        assert hasattr(report, "fibers_compressed")
        assert isinstance(report.fibers_compressed, int)

    def test_report_has_tokens_saved(self) -> None:
        report = ConsolidationReport()
        assert hasattr(report, "tokens_saved")
        assert isinstance(report.tokens_saved, int)

    def test_report_compression_fields_default_zero(self) -> None:
        report = ConsolidationReport()
        assert report.fibers_compressed == 0
        assert report.tokens_saved == 0

    def test_report_compression_fields_are_mutable(self) -> None:
        report = ConsolidationReport()
        report.fibers_compressed = 5
        report.tokens_saved = 200
        assert report.fibers_compressed == 5
        assert report.tokens_saved == 200


# ---------------------------------------------------------------------------
# Content-derived field refresh on compress / decompress / recover
# ---------------------------------------------------------------------------


async def _refresh_test_store() -> InMemoryStorage:
    """Real (non-mock) InMemoryStorage with a brain activated."""
    store = InMemoryStorage()
    brain = Brain.create(name="test-refresh-brain")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store


def _seeded_neuron(content: str, vector: list[float]) -> Neuron:
    neuron = Neuron.create(
        type=NeuronType.CONCEPT, content=content, metadata={"_embedding": list(vector)}
    )
    return dc_replace(neuron, content_hash=simhash(content))


class TestCompressionRefreshesDerivedFields:
    """Compress, decompress, and recover must not leave derived fields stale.

    Each path did ``dc_replace(neuron, content=new)`` and handed the result
    straight to ``update_neuron`` - re-saving the OLD ``content_hash`` and the
    OLD embedding vector next to the NEW content, the pattern #166 fixed for
    ``smem_edit`` but left standing here.

    The vector cannot simply be cleared: ``embedding_vec`` carries an HNSW index
    with a fixed dimension, so writing an empty array is rejected outright
    ("Incorrect vector dimension (0)") and takes the whole update with it. So
    these paths re-embed, like the edit tool - but batched, one provider call per
    compression step rather than one per neuron.
    """

    @staticmethod
    def _assert_refreshed(saved: Neuron, expected_content: str, expected_vec: list[float]) -> None:
        """The saved neuron must carry the new content's hash and its new vector."""
        assert saved.content == expected_content
        assert saved.content_hash == simhash(expected_content), (
            "content_hash must describe the content actually stored - keeping the old "
            "fingerprint feeds near-duplicate detection the SimHash of text that is gone"
        )
        assert saved.metadata["_embedding"] == expected_vec, (
            "the vector stored with the new content must describe the new content - "
            "re-saving the old one keeps the memory retrievable by what it used to say"
        )

    @pytest.mark.asyncio
    async def test_compress_graph_only_refreshes_hash_and_vector(self) -> None:
        store = await _refresh_test_store()
        orig = "the office is closed on fridays for maintenance this quarter"
        neuron = _seeded_neuron(orig, [0.1, 0.2, 0.3])
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-graph",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        fresh_vec = [9.0, 8.0, 7.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        saved = await store.get_neuron(neuron.id)
        assert saved.content == "[graph-only]"
        assert saved.content_hash == 0, (
            "the placeholder is a tombstone, not content - it must carry the "
            "no-fingerprint sentinel, not one constant SimHash shared by every "
            "graph-only anchor"
        )
        assert saved.metadata["_embedding"] == fresh_vec

    @pytest.mark.asyncio
    async def test_compress_extractive_refreshes_hash_and_vector(self) -> None:
        store = await _refresh_test_store()
        orig = (
            "Widget shipments doubled in March. The warehouse team hired five new staff. "
            "Revenue from the north region grew by twelve percent this quarter."
        )
        neuron = _seeded_neuron(orig, [0.1, 0.2, 0.3])
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-extractive",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=15.0),
        )
        await store.add_fiber(fiber)

        fresh_vec = [1.0, 1.0, 1.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            engine = CompressionEngine(store)
            result = await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)

        saved = await store.get_neuron(neuron.id)
        assert saved.content != orig, "the tier must actually have compressed the content"
        assert result.compressed_token_count < result.original_token_count
        self._assert_refreshed(saved, saved.content, fresh_vec)

    @pytest.mark.asyncio
    async def test_decompress_refreshes_hash_and_vector(self) -> None:
        store = await _refresh_test_store()
        orig = "the meeting moved to thursday at ten"
        neuron = _seeded_neuron(orig, [0.1, 0.2, 0.3])
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-decompress",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=1,
            created_at=utcnow(),
        )
        await store.add_fiber(fiber)
        await store.save_compression_backup(
            fiber_id=fiber.id,
            original_content=orig,
            compression_tier=1,
            original_token_count=10,
            compressed_token_count=5,
        )
        # Seed realistic drift: the neuron currently sits compressed, and its
        # derived fields were refreshed AGAINST THE COMPRESSED TEXT in the interim
        # (a reindex ran while the fiber was compressed). Without this the hash
        # would already equal simhash(orig) by coincidence - the field was never
        # touched since creation - and the assertion would pass on unfixed code.
        compressed = dc_replace(
            neuron,
            content="the meeting moved",
            content_hash=simhash("the meeting moved"),
            metadata={"_embedding": [4.0, 5.0, 6.0]},
        )
        await store.update_neuron(compressed)

        fresh_vec = [2.0, 2.0, 2.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            engine = CompressionEngine(store)
            ok = await engine.decompress_fiber(fiber.id)

        assert ok is True
        saved = await store.get_neuron(neuron.id)
        self._assert_refreshed(saved, orig, fresh_vec)

    @pytest.mark.asyncio
    async def test_recover_refreshes_hash_and_vector(self) -> None:
        store = await _refresh_test_store()
        orig = "quarterly numbers are final and archived"
        neuron = _seeded_neuron(orig, [0.1, 0.2, 0.3])
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-recover",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=int(CompressionTier.GRAPH_ONLY),
            created_at=utcnow(),
        )
        await store.add_fiber(fiber)
        await store.save_neuron_snapshot(
            neuron_id=neuron.id,
            brain_id=store.current_brain_id or "",
            original_content=orig,
            compressed_at=utcnow().isoformat(),
            tier=int(CompressionTier.GRAPH_ONLY),
        )
        # Same seeding rationale as the decompress test: the neuron is mid
        # GRAPH_ONLY, carrying the placeholder's hash and a vector from that phase.
        graph_only = dc_replace(
            neuron,
            content="[graph-only]",
            content_hash=simhash("[graph-only]"),
            metadata={"_embedding": [7.0, 8.0, 9.0]},
        )
        await store.update_neuron(graph_only)

        fresh_vec = [3.0, 3.0, 3.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            engine = CompressionEngine(store)
            ok = await engine.recover_fiber(fiber.id)

        assert ok is True
        saved = await store.get_neuron(neuron.id)
        self._assert_refreshed(saved, orig, fresh_vec)

    @pytest.mark.asyncio
    async def test_multi_neuron_compression_embeds_once_per_distinct_text(self) -> None:
        """A fiber's neurons must cost ONE provider round-trip - and a GRAPH_ONLY
        fiber only ONE embedding, since every neuron ends up with the same text.

        Compression runs unattended over every neuron of a fiber, under a time
        budget, in a path that already counts database round-trips as its
        dominant cost - so embedding per neuron is the thing to prevent, and
        sending N copies of the identical placeholder in the one batch is the
        residual waste after that. Identical text must map to the identical
        vector anyway, so each distinct text is embedded once and fanned out.
        (Text-to-neuron pairing for DISTINCT texts is pinned by the echo-provider
        recovery test.)
        """
        store = await _refresh_test_store()
        neurons = [
            _seeded_neuron(f"sentence number {i} about widgets", [0.1, 0.2]) for i in range(3)
        ]
        for n in neurons:
            await store.add_neuron(n)
        fiber = Fiber(
            id="f-batched",
            neuron_ids={n.id for n in neurons},
            synapse_ids=set(),
            anchor_neuron_id=neurons[0].id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        fresh_vec = [5.0, 5.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        create_provider = MagicMock(return_value=provider)
        with patch("surreal_memory.engine.semantic_discovery._create_provider", create_provider):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        assert create_provider.call_count == 1, (
            f"the provider must be resolved once per compression step, not per neuron "
            f"(got {create_provider.call_count} for {len(neurons)} neurons)"
        )
        provider.embed_batch.assert_awaited_once()
        assert provider.embed_batch.await_args.args[0] == ["[graph-only]"], (
            "three tombstoned neurons share one text - the batch must carry it once"
        )
        for n in neurons:
            saved = await store.get_neuron(n.id)
            assert saved.metadata["_embedding"] == fresh_vec, (
                "every neuron sharing the text must receive the vector embedded for it"
            )

    @pytest.mark.asyncio
    async def test_compress_leaves_a_vectorless_neuron_alone(self) -> None:
        """No vector means nothing to go stale - and no reason to reach a provider."""
        store = await _refresh_test_store()
        orig = "a plain note with no embedding at all"
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=orig)
        neuron = dc_replace(neuron, content_hash=simhash(orig))
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-no-vector",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        create_provider = MagicMock()
        with patch("surreal_memory.engine.semantic_discovery._create_provider", create_provider):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        saved = await store.get_neuron(neuron.id)
        assert saved.content == "[graph-only]"
        assert saved.content_hash == 0
        assert "_embedding" not in saved.metadata
        # An install with no vectors to refresh must not reach for an embedder.
        create_provider.assert_not_called()

    @pytest.mark.asyncio
    async def test_batched_recover_pairs_each_neuron_with_its_own_text_vector(self) -> None:
        """Every neuron must receive the vector of ITS OWN restored text.

        The batch helper embeds a list and assigns positionally, so the failure
        to guard against is a swapped assignment - distinct vectors alone would
        not catch it. The provider here derives each vector from the text it was
        given, making any misalignment visible as the wrong vector on a neuron.
        """
        store = await _refresh_test_store()
        texts = {
            "n-alpha": "alpha original text",
            "n-beta": "beta original text is deliberately longer",
            "n-gamma": "gamma",
        }
        neurons = []
        for nid in texts:
            neuron = Neuron.create(
                type=NeuronType.CONCEPT,
                content="[graph-only]",
                neuron_id=nid,
                metadata={"_embedding": [0.1, 0.1]},
            )
            neuron = dc_replace(neuron, content_hash=0)
            await store.add_neuron(neuron)
            neurons.append(neuron)
        fiber = Fiber(
            id="f-pairing",
            neuron_ids=set(texts),
            synapse_ids=set(),
            anchor_neuron_id="n-alpha",
            compression_tier=int(CompressionTier.GRAPH_ONLY),
            created_at=utcnow(),
        )
        await store.add_fiber(fiber)
        for nid, orig in texts.items():
            await store.save_neuron_snapshot(
                neuron_id=nid,
                brain_id=store.current_brain_id or "",
                original_content=orig,
                compressed_at=utcnow().isoformat(),
                tier=int(CompressionTier.GRAPH_ONLY),
            )

        provider = MagicMock()
        # Echo provider: the vector encodes the text it was derived from.
        provider.embed_batch = AsyncMock(
            side_effect=lambda batch: [[float(len(t)), 1.0] for t in batch]
        )
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            ok = await CompressionEngine(store).recover_fiber(fiber.id)

        assert ok is True
        for nid, orig in texts.items():
            saved = await store.get_neuron(nid)
            assert saved.content == orig
            assert saved.metadata["_embedding"] == [float(len(orig)), 1.0], (
                f"neuron {nid} must carry the vector derived from its own text - "
                "anything else means the batch assignment is misaligned"
            )

    @pytest.mark.asyncio
    async def test_recompression_stamps_the_sentinel_on_legacy_tombstones(self) -> None:
        """When a legacy tombstone DOES meet the compression path, it gets the sentinel.

        Tombstones written by older versions carry the SimHash of their deleted
        original text. A fiber parked at GRAPH_ONLY never re-enters compression
        (their protection is the census's content guard, not this path), but a
        PARTIALLY recovered fiber does: ``recover_fiber`` resets it to FULL while
        neurons without a snapshot keep the placeholder - the tier-0 fiber here
        models exactly that shape. Re-compression must normalise the stale hash
        to the sentinel, and must not spend a provider call doing it.
        """
        store = await _refresh_test_store()
        legacy = Neuron.create(
            type=NeuronType.CONCEPT,
            content="[graph-only]",
            metadata={"_embedding": [0.3, 0.4]},
        )
        legacy = dc_replace(legacy, content_hash=simhash("the deleted original text"))
        await store.add_neuron(legacy)
        fiber = Fiber(
            id="f-legacy",
            neuron_ids={legacy.id},
            synapse_ids=set(),
            anchor_neuron_id=legacy.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        create_provider = MagicMock()
        with patch("surreal_memory.engine.semantic_discovery._create_provider", create_provider):
            await CompressionEngine(store).compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        create_provider.assert_not_called()
        saved = await store.get_neuron(legacy.id)
        assert saved.content == "[graph-only]"
        assert saved.content_hash == 0, (
            "a legacy tombstone must not keep the fingerprint of its deleted text"
        )
        assert saved.metadata["_embedding"] == [0.3, 0.4], (
            "stamping the sentinel must not touch the vector - no re-embed on a repeat pass"
        )

    @pytest.mark.asyncio
    async def test_graph_only_anchors_are_not_paired_as_near_duplicates(self) -> None:
        """Two unrelated fibers compressed to GRAPH_ONLY must not look like duplicates.

        Every graph-only neuron ends up with the same placeholder text, so a
        SimHash of it would be one constant shared brain-wide. The consolidation
        census compares anchors by Hamming distance, so identical fingerprints
        would read as distance 0 and persist a false "duplicate of" edge between
        memories that have nothing to do with each other.
        """
        from surreal_memory.utils.simhash import is_near_duplicate

        store = await _refresh_test_store()
        saved = []
        for idx, text in enumerate(
            ["the north warehouse lease renewal terms", "a totally unrelated travel itinerary"]
        ):
            neuron = _seeded_neuron(text, [0.1, 0.2])
            await store.add_neuron(neuron)
            fiber = Fiber(
                id=f"f-dup-{idx}",
                neuron_ids={neuron.id},
                synapse_ids=set(),
                anchor_neuron_id=neuron.id,
                compression_tier=0,
                created_at=utcnow() - timedelta(days=200.0),
            )
            await store.add_fiber(fiber)
            provider = MagicMock()
            provider.embed_batch = AsyncMock(return_value=[[1.0, 1.0]])
            with patch(
                "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
            ):
                await CompressionEngine(store).compress_fiber(fiber, CompressionTier.GRAPH_ONLY)
            saved.append(await store.get_neuron(neuron.id))

        a, b = saved
        assert a.content_hash == 0 and b.content_hash == 0, (
            "placeholder content must carry the no-fingerprint sentinel that the "
            "census and the dedup pipeline both skip"
        )
        assert not is_near_duplicate(simhash("unrelated one"), 0), (
            "sanity: the sentinel must not be treated as a comparable fingerprint"
        )

    @pytest.mark.asyncio
    async def test_recompression_after_partial_recovery_does_not_re_embed_the_placeholder(
        self,
    ) -> None:
        """A re-entered GRAPH_ONLY pass must not pay for neurons already cleared.

        A fiber sitting at GRAPH_ONLY never re-enters compression (the tier
        early-return), so the realistic way already-tombstoned neurons meet
        this branch again is a PARTIAL recovery: ``recover_fiber`` resets the
        fiber to FULL when it restores at least one neuron, while neurons that
        had no snapshot keep the placeholder. When the fiber later ages back
        into GRAPH_ONLY, those neurons are already correct - re-deriving their
        fields would spend a provider call on a known result.
        """
        store = await _refresh_test_store()
        neuron = _seeded_neuron("something that will be compressed away", [0.1, 0.2])
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-repeat",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[[2.0, 2.0]])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            await CompressionEngine(store).compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        # The partial-recovery shape: fiber back at FULL, neuron still a
        # tombstone. (recover_fiber resets the tier whenever it restores any
        # neuron; this one had no snapshot, so its placeholder survived.)
        await store.delete_neuron_snapshot(neuron.id)
        recovered_fiber = dc_replace(fiber, compression_tier=int(CompressionTier.FULL))
        await store.update_fiber(recovered_fiber)

        second = MagicMock()
        with patch("surreal_memory.engine.semantic_discovery._create_provider", second):
            result = await CompressionEngine(store).compress_fiber(
                recovered_fiber, CompressionTier.GRAPH_ONLY
            )

        assert result.skipped is False, (
            "the pass must actually run - an early-returned pass proves nothing "
            "about what a re-entered one would cost"
        )
        second.assert_not_called()
        saved = await store.get_neuron(neuron.id)
        assert saved.content == "[graph-only]"
        assert saved.content_hash == 0

    @pytest.mark.asyncio
    async def test_only_the_vectored_neuron_is_embedded_and_gets_its_own_vector(self) -> None:
        """A fiber may mix neurons with and without vectors - each must be handled."""
        store = await _refresh_test_store()
        plain_a = Neuron.create(type=NeuronType.CONCEPT, content="first plain note")
        vectored = _seeded_neuron("the middle note that carries a vector", [0.1, 0.2])
        plain_b = Neuron.create(type=NeuronType.CONCEPT, content="third plain note")
        for n in (plain_a, vectored, plain_b):
            await store.add_neuron(n)
        fiber = Fiber(
            id="f-mixed",
            neuron_ids={plain_a.id, vectored.id, plain_b.id},
            synapse_ids=set(),
            anchor_neuron_id=vectored.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        fresh_vec = [4.0, 4.0]
        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[list(fresh_vec)])
        with patch(
            "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
        ):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        assert len(provider.embed_batch.await_args.args[0]) == 1, (
            "only the neuron that actually carries a vector belongs in the batch"
        )
        assert (await store.get_neuron(vectored.id)).metadata["_embedding"] == fresh_vec
        for plain in (plain_a, plain_b):
            saved = await store.get_neuron(plain.id)
            assert saved.content == "[graph-only]"
            assert saved.content_hash == 0
            assert "_embedding" not in saved.metadata

    @pytest.mark.asyncio
    async def test_short_provider_output_warns_instead_of_silently_keeping_old_vectors(
        self, caplog
    ) -> None:
        """A provider returning fewer vectors than texts must not pass silently.

        Assigning positionally over a short list would leave the tail neurons
        holding their OLD vectors with no error and no warning - a silent stale
        write, which is the failure this whole change exists to remove.
        """
        import logging

        store = await _refresh_test_store()
        old_vec = [0.1, 0.2]
        neurons = [_seeded_neuron(f"note number {i} with content", old_vec) for i in range(3)]
        for n in neurons:
            await store.add_neuron(n)
        fiber = Fiber(
            id="f-short-output",
            neuron_ids={n.id for n in neurons},
            synapse_ids=set(),
            anchor_neuron_id=neurons[0].id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        provider = MagicMock()
        provider.embed_batch = AsyncMock(return_value=[[9.0, 9.0], [8.0, 8.0]])  # 2 for 3 texts
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
            ),
            caplog.at_level(logging.WARNING, logger="surreal_memory.utils.content_refresh"),
        ):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        assert any("reindex" in r.message for r in caplog.records), (
            "a truncated provider response must be reported, not absorbed"
        )
        for n in neurons:
            assert (await store.get_neuron(n.id)).metadata["_embedding"] == old_vec, (
                "on a count mismatch no vector may be assigned - a partial write would "
                "leave some neurons silently stale and indistinguishable from success"
            )

    @pytest.mark.asyncio
    async def test_compress_warns_when_re_embedding_exceeds_its_time_budget(self, caplog) -> None:
        """The bounded-wait branch must report, not swallow."""
        import asyncio
        import logging

        store = await _refresh_test_store()
        old_vec = [0.1, 0.2]
        neuron = _seeded_neuron("a note whose re-embed will outlast the budget", old_vec)
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-timeout",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=200.0),
        )
        await store.add_fiber(fiber)

        async def _too_slow(_texts: list[str]) -> list[list[float]]:
            await asyncio.sleep(5)
            return [[1.0, 1.0]]

        provider = MagicMock()
        provider.embed_batch = _too_slow
        with (
            patch(
                "surreal_memory.engine.semantic_discovery._create_provider", return_value=provider
            ),
            patch("surreal_memory.engine.encoder._inline_embed_timeout", return_value=0.01),
            caplog.at_level(logging.WARNING, logger="surreal_memory.utils.content_refresh"),
        ):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

        saved = await store.get_neuron(neuron.id)
        assert saved.content == "[graph-only]", "a slow embedder must not block the compression"
        assert saved.content_hash == 0
        assert saved.metadata["_embedding"] == old_vec
        assert any("time budget" in r.message for r in caplog.records), (
            "exceeding the embed budget must be named as such, not folded into a generic failure"
        )

    @pytest.mark.asyncio
    async def test_compress_survives_provider_unavailable_but_warns(self, caplog) -> None:
        import logging

        store = await _refresh_test_store()
        old_vec = [0.1, 0.2, 0.3]
        orig = (
            "Widget shipments doubled in March. The warehouse team hired five new staff. "
            "Revenue from the north region grew by twelve percent this quarter."
        )
        neuron = _seeded_neuron(orig, old_vec)
        await store.add_neuron(neuron)
        fiber = Fiber(
            id="f-provider-down",
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            compression_tier=0,
            created_at=utcnow() - timedelta(days=15.0),
        )
        await store.add_fiber(fiber)

        with (
            patch(
                "surreal_memory.engine.semantic_discovery._create_provider",
                side_effect=RuntimeError("no provider"),
            ),
            caplog.at_level(logging.WARNING, logger="surreal_memory.utils.content_refresh"),
        ):
            engine = CompressionEngine(store)
            await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)

        saved = await store.get_neuron(neuron.id)
        assert saved.content != orig, "compression must not depend on embedder availability"
        assert saved.content_hash == simhash(saved.content), (
            "content_hash has no provider dependency and must still be refreshed"
        )
        assert saved.metadata["_embedding"] == old_vec, (
            "with no provider available the old vector must be kept, not dropped or guessed - "
            "an empty vector is rejected outright by the HNSW index on embedding_vec"
        )
        assert any("reindex" in r.message for r in caplog.records), (
            "a stale vector left behind must be reported loudly, with the remediation "
            "command - silence here is indistinguishable from a successful refresh"
        )
