"""Structured content detection — CSV, JSON, key-value, table formats.

Heuristic-based detection (no LLM). Inspects raw content and returns
a StructuredContent descriptor that downstream steps use to preserve
structure in neuron metadata.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ContentFormat(StrEnum):
    """Detected structure format of content."""

    CSV_ROW = "csv_row"
    JSON_OBJECT = "json_object"
    KEY_VALUE = "key_value"
    TABLE_ROW = "table_row"
    PLAIN = "plain"


@dataclass(frozen=True)
class StructuredField:
    """A single field extracted from structured content."""

    name: str
    value: str
    field_type: str = "text"  # "text", "number", "date", "currency"


@dataclass(frozen=True)
class StructuredContent:
    """Result of structure detection on a piece of content."""

    format: ContentFormat
    fields: tuple[StructuredField, ...] = ()
    raw: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_structured(self) -> bool:
        """Whether the content has meaningful structure."""
        return self.format != ContentFormat.PLAIN


# ── Field type detection ──────────────────────────────────────

_DATE_PATTERNS = (
    re.compile(r"^\d{4}-\d{2}-\d{2}"),  # ISO 2026-01-15
    re.compile(r"^\d{2}/\d{2}/\d{4}"),  # US/VN 15/01/2026
    re.compile(r"^\d{2}-\d{2}-\d{4}"),  # 15-01-2026
)

_CURRENCY_PATTERN = re.compile(
    r"^(?:[\$€£¥₫₹]\s*[\d,]+\.?\d*|[\d,]+\.?\d*\s*(?:VND|USD|EUR|GBP|JPY|đ|₫))$",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(r"^-?[\d,]+\.?\d*%?$")


def _detect_field_type(value: str) -> str:
    """Detect the type of a field value."""
    v = value.strip()
    if not v:
        return "text"

    # Date check
    for pat in _DATE_PATTERNS:
        if pat.match(v):
            return "date"

    # Currency check (must be before number since it also has digits)
    if _CURRENCY_PATTERN.match(v):
        return "currency"

    # Number check
    if _NUMBER_PATTERN.match(v):
        return "number"

    return "text"


# ── Format detectors ──────────────────────────────────────────

_KV_SEPARATOR = re.compile(r"\s*[:|=]\s*")
_PIPE_SEPARATOR = re.compile(r"\s*\|\s*")


def _detect_json_object(content: str) -> StructuredContent | None:
    """Try to parse content as a JSON object."""
    stripped = content.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None

    fields = tuple(
        StructuredField(
            name=str(k),
            value=str(v),
            field_type=_detect_field_type(str(v)),
        )
        for k, v in obj.items()
    )
    return StructuredContent(
        format=ContentFormat.JSON_OBJECT,
        fields=fields,
        raw=content,
        confidence=0.95,
        metadata={"key_count": len(obj)},
    )


def _detect_csv_row(content: str) -> StructuredContent | None:
    """Detect a single CSV-like row (comma or tab delimited)."""
    stripped = content.strip()

    # Must be a single line (or at most header + data)
    lines = stripped.split("\n")
    if len(lines) > 2:
        return None

    # Pick delimiter: tab > comma
    if "\t" in stripped:
        delimiter = "\t"
    elif stripped.count(",") >= 2:
        delimiter = ","
    else:
        return None

    if len(lines) == 2:
        # Header + data row
        headers = [h.strip() for h in lines[0].split(delimiter)]
        values = [v.strip() for v in lines[1].split(delimiter)]
        if len(headers) != len(values) or len(headers) < 2:
            return None
        fields = tuple(
            StructuredField(name=h, value=v, field_type=_detect_field_type(v))
            for h, v in zip(headers, values, strict=False)
        )
        confidence = 0.9
    else:
        # Single row — columns unnamed
        values = [v.strip() for v in lines[0].split(delimiter)]
        if len(values) < 3:
            return None
        fields = tuple(
            StructuredField(name=f"col_{i}", value=v, field_type=_detect_field_type(v))
            for i, v in enumerate(values)
        )
        confidence = 0.7

    # If every column is plain text (no numbers, dates, or currency), this is
    # likely prose with commas rather than actual CSV data.
    if all(f.field_type == "text" for f in fields):
        return None

    return StructuredContent(
        format=ContentFormat.CSV_ROW,
        fields=fields,
        raw=content,
        confidence=confidence,
        metadata={"column_count": len(fields), "delimiter": delimiter},
    )


def _detect_key_value(content: str) -> StructuredContent | None:
    """Detect key: value or key=value pairs (possibly pipe-separated)."""
    stripped = content.strip()

    # Try pipe-separated first: "Date: 2026 | Amount: 500 | Payee: X"
    if "|" in stripped:
        segments = _PIPE_SEPARATOR.split(stripped)
        if len(segments) >= 2:
            fields: list[StructuredField] = []
            for seg in segments:
                match = _KV_SEPARATOR.split(seg.strip(), maxsplit=1)
                if len(match) == 2:
                    fields.append(
                        StructuredField(
                            name=match[0].strip(),
                            value=match[1].strip(),
                            field_type=_detect_field_type(match[1].strip()),
                        )
                    )
            if len(fields) >= 2:
                return StructuredContent(
                    format=ContentFormat.KEY_VALUE,
                    fields=tuple(fields),
                    raw=content,
                    confidence=0.85,
                    metadata={"pair_count": len(fields), "separator": "pipe"},
                )

    # Try line-based key: value
    lines = stripped.split("\n")
    if len(lines) >= 2:
        fields_list: list[StructuredField] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = _KV_SEPARATOR.split(line, maxsplit=1)
            if len(match) == 2 and match[0].strip():
                fields_list.append(
                    StructuredField(
                        name=match[0].strip(),
                        value=match[1].strip(),
                        field_type=_detect_field_type(match[1].strip()),
                    )
                )
        if len(fields_list) >= 2:
            return StructuredContent(
                format=ContentFormat.KEY_VALUE,
                fields=tuple(fields_list),
                raw=content,
                confidence=0.8,
                metadata={"pair_count": len(fields_list), "separator": "line"},
            )

    return None


def _detect_table_row(content: str) -> StructuredContent | None:
    """Detect markdown table row: | val1 | val2 | val3 |."""
    stripped = content.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None

    # Split by pipe, ignore first/last empty
    cells = [c.strip() for c in stripped.split("|")[1:-1]]
    if len(cells) < 2:
        return None

    # Skip separator rows (----)
    if all(re.match(r"^-+$", c) for c in cells):
        return None

    fields = tuple(
        StructuredField(
            name=f"col_{i}",
            value=c,
            field_type=_detect_field_type(c),
        )
        for i, c in enumerate(cells)
    )
    return StructuredContent(
        format=ContentFormat.TABLE_ROW,
        fields=fields,
        raw=content,
        confidence=0.85,
        metadata={"column_count": len(cells)},
    )


# ── Main detector ─────────────────────────────────────────────

MAX_DETECT_CHARS = 4096  # Structure detection only meaningful on short content


def detect_structure(content: str) -> StructuredContent:
    """Detect structured content format.

    Tries detectors in order of specificity:
    1. JSON object
    2. Table row (markdown pipe syntax)
    3. CSV row (comma/tab delimited)
    4. Key-value pairs (colon/equals separated)
    5. Plain text (fallback)

    Args:
        content: Raw text content to analyze.

    Returns:
        StructuredContent with detected format and extracted fields.
    """
    if not content or not content.strip() or len(content) > MAX_DETECT_CHARS:
        return StructuredContent(format=ContentFormat.PLAIN, raw=content)

    # Try each detector in order
    for detector in (_detect_json_object, _detect_table_row, _detect_key_value, _detect_csv_row):
        result = detector(content)
        if result is not None:
            return result

    return StructuredContent(format=ContentFormat.PLAIN, raw=content)


def format_structured_output(sc: StructuredContent) -> str:
    """Format structured content for human-readable recall output.

    Args:
        sc: Detected structured content.

    Returns:
        Formatted string representation.
    """
    if not sc.is_structured or not sc.fields:
        return sc.raw

    if sc.format == ContentFormat.KEY_VALUE:
        max_name = max(len(f.name) for f in sc.fields)
        return "\n".join(f"{f.name:<{max_name}} : {f.value}" for f in sc.fields)

    if sc.format == ContentFormat.JSON_OBJECT:
        max_name = max(len(f.name) for f in sc.fields)
        return "\n".join(f"{f.name:<{max_name}} : {f.value}" for f in sc.fields)

    if sc.format in (ContentFormat.CSV_ROW, ContentFormat.TABLE_ROW):
        # Aligned columns
        headers = [f.name for f in sc.fields]
        values = [f.value for f in sc.fields]
        widths = [max(len(h), len(v)) for h, v in zip(headers, values, strict=False)]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=False))
        sep_line = " | ".join("-" * w for w in widths)
        value_line = " | ".join(v.ljust(w) for v, w in zip(values, widths, strict=False))
        return f"{header_line}\n{sep_line}\n{value_line}"

    return sc.raw
