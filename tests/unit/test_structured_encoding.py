"""Tests for structured content detection and encoding (Phase 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.extraction.structure_detector import (
    ContentFormat,
    StructuredContent,
    StructuredField,
    detect_structure,
    format_structured_output,
)

# ──────────────────── Structure Detection ────────────────────


class TestDetectJSON:
    """JSON object detection."""

    def test_simple_json(self) -> None:
        content = '{"name": "Alice", "age": 30}'
        result = detect_structure(content)
        assert result.format == ContentFormat.JSON_OBJECT
        assert result.confidence >= 0.9
        assert len(result.fields) == 2
        assert result.fields[0].name == "name"
        assert result.fields[0].value == "Alice"
        assert result.fields[1].name == "age"
        assert result.fields[1].field_type == "number"

    def test_json_with_whitespace(self) -> None:
        content = '  { "key": "value" }  '
        result = detect_structure(content)
        assert result.format == ContentFormat.JSON_OBJECT

    def test_invalid_json(self) -> None:
        content = '{"key": broken}'
        result = detect_structure(content)
        assert result.format == ContentFormat.PLAIN

    def test_json_array_not_detected(self) -> None:
        content = '["a", "b", "c"]'
        result = detect_structure(content)
        # Arrays are not JSON objects
        assert result.format != ContentFormat.JSON_OBJECT

    def test_nested_json(self) -> None:
        content = '{"user": {"name": "Bob"}, "active": true}'
        result = detect_structure(content)
        assert result.format == ContentFormat.JSON_OBJECT
        assert len(result.fields) == 2


class TestDetectCSV:
    """CSV row detection."""

    def test_single_csv_row(self) -> None:
        content = "Alice, 30, Engineer, New York"
        result = detect_structure(content)
        assert result.format == ContentFormat.CSV_ROW
        assert len(result.fields) >= 3

    def test_header_and_data(self) -> None:
        content = "Name,Age,City\nAlice,30,NYC"
        result = detect_structure(content)
        assert result.format == ContentFormat.CSV_ROW
        assert result.confidence >= 0.9
        assert result.fields[0].name == "Name"
        assert result.fields[0].value == "Alice"

    def test_tab_delimited(self) -> None:
        content = "Alice\t30\tEngineer"
        result = detect_structure(content)
        assert result.format == ContentFormat.CSV_ROW

    def test_too_few_columns(self) -> None:
        content = "just, two"
        result = detect_structure(content)
        # Two items in a comma-separated list isn't enough for CSV detection
        assert result.format == ContentFormat.PLAIN

    def test_multiline_not_csv(self) -> None:
        content = "line one\nline two\nline three"
        result = detect_structure(content)
        assert result.format != ContentFormat.CSV_ROW


class TestDetectKeyValue:
    """Key-value pair detection."""

    def test_pipe_separated(self) -> None:
        content = "Date: 2026-01-15 | Amount: 25,000,000 VND | Payee: Nguyen Van A"
        result = detect_structure(content)
        assert result.format == ContentFormat.KEY_VALUE
        assert len(result.fields) == 3
        assert result.fields[0].name == "Date"
        assert result.fields[0].field_type == "date"
        assert result.fields[1].name == "Amount"
        assert result.fields[2].name == "Payee"

    def test_line_based_kv(self) -> None:
        content = "Name: Alice\nAge: 30\nCity: New York"
        result = detect_structure(content)
        assert result.format == ContentFormat.KEY_VALUE
        assert len(result.fields) == 3

    def test_equals_separated(self) -> None:
        content = "host=localhost\nport=5432\ndb=myapp"
        result = detect_structure(content)
        assert result.format == ContentFormat.KEY_VALUE
        assert len(result.fields) == 3

    def test_single_pair_not_enough(self) -> None:
        content = "key: value"
        result = detect_structure(content)
        # Single KV pair doesn't meet minimum
        assert result.format != ContentFormat.KEY_VALUE


class TestDetectTable:
    """Markdown table row detection."""

    def test_table_row(self) -> None:
        content = "| Alice | 30 | Engineer |"
        result = detect_structure(content)
        assert result.format == ContentFormat.TABLE_ROW
        assert len(result.fields) == 3

    def test_separator_row_skipped(self) -> None:
        content = "| --- | --- | --- |"
        result = detect_structure(content)
        assert result.format != ContentFormat.TABLE_ROW

    def test_not_pipe_bordered(self) -> None:
        content = "Alice | 30 | Engineer"
        result = detect_structure(content)
        assert result.format != ContentFormat.TABLE_ROW


class TestDetectPlain:
    """Plain text fallback."""

    def test_normal_text(self) -> None:
        result = detect_structure("This is a normal sentence about memory.")
        assert result.format == ContentFormat.PLAIN
        assert not result.is_structured

    def test_empty_string(self) -> None:
        result = detect_structure("")
        assert result.format == ContentFormat.PLAIN

    def test_whitespace_only(self) -> None:
        result = detect_structure("   \n  ")
        assert result.format == ContentFormat.PLAIN


# ──────────────────── Field Type Detection ────────────────────


class TestFieldTypeDetection:
    """Verify auto-detection of field value types."""

    def test_date_iso(self) -> None:
        content = '{"date": "2026-01-15"}'
        result = detect_structure(content)
        date_field = next(f for f in result.fields if f.name == "date")
        assert date_field.field_type == "date"

    def test_currency(self) -> None:
        content = "Amount: $1,250.00 | Tax: 10%"
        result = detect_structure(content)
        amount_field = next(f for f in result.fields if f.name == "Amount")
        assert amount_field.field_type == "currency"

    def test_number(self) -> None:
        content = '{"count": "42", "ratio": "3.14"}'
        result = detect_structure(content)
        count_field = next(f for f in result.fields if f.name == "count")
        assert count_field.field_type == "number"

    def test_text_default(self) -> None:
        content = '{"name": "Alice"}'
        result = detect_structure(content)
        assert result.fields[0].field_type == "text"


# ──────────────────── Structured Output Formatting ────────────────────


class TestFormatStructuredOutput:
    """Test human-readable formatting of structured content."""

    def test_format_kv(self) -> None:
        sc = StructuredContent(
            format=ContentFormat.KEY_VALUE,
            fields=(
                StructuredField(name="Name", value="Alice"),
                StructuredField(name="Age", value="30"),
            ),
            raw="Name: Alice | Age: 30",
        )
        output = format_structured_output(sc)
        assert "Name" in output
        assert "Alice" in output
        assert ":" in output

    def test_format_csv(self) -> None:
        sc = StructuredContent(
            format=ContentFormat.CSV_ROW,
            fields=(
                StructuredField(name="Name", value="Alice"),
                StructuredField(name="Age", value="30"),
            ),
            raw="Name,Age\nAlice,30",
        )
        output = format_structured_output(sc)
        assert "Name" in output
        assert "Alice" in output
        # Should have separator line
        assert "---" in output or "-" in output

    def test_format_plain_returns_raw(self) -> None:
        sc = StructuredContent(
            format=ContentFormat.PLAIN,
            raw="Just plain text.",
        )
        assert format_structured_output(sc) == "Just plain text."

    def test_format_empty_fields_returns_raw(self) -> None:
        sc = StructuredContent(
            format=ContentFormat.KEY_VALUE,
            fields=(),
            raw="raw fallback",
        )
        assert format_structured_output(sc) == "raw fallback"


# ──────────────────── StructuredContent Dataclass ────────────────────


class TestStructuredContent:
    """StructuredContent behavior."""

    def test_is_structured(self) -> None:
        sc = StructuredContent(format=ContentFormat.JSON_OBJECT)
        assert sc.is_structured is True

    def test_is_not_structured(self) -> None:
        sc = StructuredContent(format=ContentFormat.PLAIN)
        assert sc.is_structured is False

    def test_frozen(self) -> None:
        sc = StructuredContent(format=ContentFormat.PLAIN, raw="test")
        with pytest.raises(AttributeError):
            sc.raw = "mutated"  # type: ignore[misc]


# ──────────────────── Pipeline Step Integration ────────────────────


class TestStructureDetectionStep:
    """Test StructureDetectionStep in the encoding pipeline."""

    @pytest.mark.asyncio
    async def test_structured_content_adds_metadata(self) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructureDetectionStep
        from neural_memory.utils.timeutils import utcnow

        ctx = PipelineContext(
            content='{"name": "Alice", "age": 30}',
            timestamp=utcnow(),
            metadata={},
            tags=set(),
            language="en",
        )
        step = StructureDetectionStep()
        storage = AsyncMock()
        config = MagicMock()

        result_ctx = await step.execute(ctx, storage, config)

        assert "_structure" in result_ctx.metadata
        structure = result_ctx.metadata["_structure"]
        assert structure["format"] == "json_object"
        assert len(structure["fields"]) == 2
        assert "_structured:json_object" in result_ctx.tags

    @pytest.mark.asyncio
    async def test_plain_content_no_metadata(self) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructureDetectionStep
        from neural_memory.utils.timeutils import utcnow

        ctx = PipelineContext(
            content="This is a plain text memory about something.",
            timestamp=utcnow(),
            metadata={},
            tags=set(),
            language="en",
        )
        step = StructureDetectionStep()
        storage = AsyncMock()
        config = MagicMock()

        result_ctx = await step.execute(ctx, storage, config)

        assert "_structure" not in result_ctx.metadata
        assert not any(t.startswith("_structured:") for t in result_ctx.tags)

    @pytest.mark.asyncio
    async def test_kv_pipe_separated(self) -> None:
        from neural_memory.engine.pipeline import PipelineContext
        from neural_memory.engine.pipeline_steps import StructureDetectionStep
        from neural_memory.utils.timeutils import utcnow

        ctx = PipelineContext(
            content="Date: 2026-01-15 | Amount: 25,000,000 VND | Payee: Nguyen Van A",
            timestamp=utcnow(),
            metadata={},
            tags=set(),
            language="vi",
        )
        step = StructureDetectionStep()
        storage = AsyncMock()
        config = MagicMock()

        result_ctx = await step.execute(ctx, storage, config)

        assert "_structure" in result_ctx.metadata
        assert result_ctx.metadata["_structure"]["format"] == "key_value"
        assert "_structured:key_value" in result_ctx.tags

    def test_step_name(self) -> None:
        from neural_memory.engine.pipeline_steps import StructureDetectionStep

        step = StructureDetectionStep()
        assert step.name == "structure_detection"


# ──────────────────── Retrieval Context Formatting ────────────────────


class TestRetrievalContextStructured:
    """Test structured content formatting in retrieval context."""

    def test_format_if_structured_with_structure(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        metadata = {
            "_structure": {
                "format": "key_value",
                "fields": [
                    {"name": "Name", "value": "Alice", "type": "text"},
                    {"name": "Age", "value": "30", "type": "number"},
                ],
            }
        }
        result = _format_if_structured("Name: Alice | Age: 30", metadata)
        assert "Name" in result
        assert "Alice" in result
        assert ":" in result

    def test_format_if_structured_plain(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        result = _format_if_structured("plain text", {})
        assert result == "plain text"

    def test_format_if_structured_no_fields(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        metadata = {"_structure": {"format": "key_value", "fields": []}}
        result = _format_if_structured("raw content", metadata)
        assert result == "raw content"

    def test_format_if_structured_plain_format(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        metadata = {"_structure": {"format": "plain"}}
        result = _format_if_structured("text", metadata)
        assert result == "text"


# ──────────────────── Encoder Pipeline Includes Step ────────────────────


class TestEncoderHasStructureStep:
    """Verify StructureDetectionStep is in the default pipeline."""

    def test_default_pipeline_has_structure_step(self) -> None:

        # Build default pipeline
        from neural_memory.engine.encoder import build_default_pipeline
        from neural_memory.extraction.entities import EntityExtractor
        from neural_memory.extraction.relations import RelationExtractor
        from neural_memory.extraction.sentiment import SentimentExtractor
        from neural_memory.extraction.temporal import TemporalExtractor
        from neural_memory.utils.tag_normalizer import TagNormalizer

        pipeline = build_default_pipeline(
            temporal_extractor=TemporalExtractor(),
            entity_extractor=EntityExtractor(),
            relation_extractor=RelationExtractor(),
            sentiment_extractor=SentimentExtractor(),
            tag_normalizer=TagNormalizer(),
        )

        step_names = [s.name for s in pipeline.steps]
        assert "structure_detection" in step_names

        # Structure detection should be before auto_tag
        sd_idx = step_names.index("structure_detection")
        at_idx = step_names.index("auto_tag")
        assert sd_idx < at_idx


# ──────────────────── Round-trip: detect → store → format ────────────────────


class TestRoundTrip:
    """End-to-end: detect structure, store metadata, format output."""

    def test_kv_round_trip(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        content = "Date: 2026-01-15 | Amount: 25,000,000 VND | Payee: Nguyen Van A"
        detected = detect_structure(content)
        assert detected.is_structured

        # Simulate what the pipeline stores in metadata
        stored_meta = {
            "_structure": {
                "format": detected.format.value,
                "fields": [
                    {"name": f.name, "value": f.value, "type": f.field_type}
                    for f in detected.fields
                ],
            }
        }

        # Simulate retrieval formatting
        formatted = _format_if_structured(content, stored_meta)
        assert "Date" in formatted
        assert "Amount" in formatted
        assert "Payee" in formatted

    def test_json_round_trip(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        content = '{"rate": "5.5%", "effective_date": "2026-01-01", "currency": "VND"}'
        detected = detect_structure(content)
        assert detected.format == ContentFormat.JSON_OBJECT

        stored_meta = {
            "_structure": {
                "format": detected.format.value,
                "fields": [
                    {"name": f.name, "value": f.value, "type": f.field_type}
                    for f in detected.fields
                ],
            }
        }

        formatted = _format_if_structured(content, stored_meta)
        assert "rate" in formatted
        assert "5.5%" in formatted

    def test_plain_round_trip(self) -> None:
        from neural_memory.engine.retrieval_context import _format_if_structured

        content = "The interest rate was lowered to 5.5% effective January 2026."
        detected = detect_structure(content)
        assert not detected.is_structured

        formatted = _format_if_structured(content, {})
        assert formatted == content


# ──────────────────── ContentFormat Enum ────────────────────


class TestContentFormatEnum:
    """Verify ContentFormat enum values."""

    def test_all_formats(self) -> None:
        for fmt in ("csv_row", "json_object", "key_value", "table_row", "plain"):
            assert ContentFormat(fmt).value == fmt
