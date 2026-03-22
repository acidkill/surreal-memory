"""Chart generator — extract data from neurons and produce chart specifications.

Supports 3 output formats:
- Vega-Lite JSON (primary — renderable by any frontend)
- Markdown table (fallback — works in text contexts)
- ASCII chart (terminal — simple vertical bar chart)

Data extraction pipeline:
1. Receive recalled neurons (from spreading activation)
2. Parse numeric values + labels from neuron content/metadata
3. Auto-detect best chart type from data shape
4. Generate chart specification
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Chart types
CHART_TYPES = ("line", "bar", "pie", "scatter", "table", "timeline")

# Patterns for extracting numeric values from content
_NUMBER_PATTERN = re.compile(
    r"(?:^|\s|=|:)\s*([+-]?\d[\d,]*\.?\d*)\s*(%|USD|VND|đ|k|K|M|B|tỷ|triệu)?",
)
_DATE_PATTERN = re.compile(
    r"\b(\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|Q[1-4][/]\d{4}|\d{1,2}[/]\d{4})\b",
)
_KV_PATTERN = re.compile(
    r"(?:^|\n)\s*[-•*]?\s*([^:=\n]{2,40})\s*[:=]\s*([+-]?\d[\d,]*\.?\d*)\s*(%|USD|VND|đ|k|K|M|B|tỷ|triệu)?",
)


@dataclass(frozen=True)
class DataPoint:
    """A single data point extracted from a neuron."""

    label: str
    value: float
    unit: str = ""
    neuron_id: str = ""
    source_id: str = ""
    date: str = ""
    category: str = ""


@dataclass(frozen=True)
class ChartSpec:
    """Generated chart specification."""

    chart_type: str
    title: str
    data_points: tuple[DataPoint, ...] = ()
    vega_lite: dict[str, Any] = field(default_factory=dict)
    markdown: str = ""
    ascii_chart: str = ""
    provenance: tuple[str, ...] = ()  # neuron IDs used


def extract_data_points(
    neurons: list[Any],
    query: str = "",
) -> list[DataPoint]:
    """Extract structured data points from recalled neurons.

    Parses key:value pairs, numeric values, and dates from neuron content.
    """
    points: list[DataPoint] = []

    for neuron in neurons:
        content = getattr(neuron, "content", "") or ""
        neuron_id = getattr(neuron, "id", "")
        source_id = getattr(neuron, "source_id", "") or ""

        # Try key:value extraction first
        kv_matches = _KV_PATTERN.findall(content)
        if kv_matches:
            for label, value_str, unit in kv_matches:
                value = _parse_number(value_str)
                if value is not None:
                    points.append(
                        DataPoint(
                            label=label.strip(),
                            value=value,
                            unit=unit,
                            neuron_id=neuron_id,
                            source_id=source_id,
                        )
                    )
            continue

        # Try date + value extraction
        dates = _DATE_PATTERN.findall(content)
        numbers = _NUMBER_PATTERN.findall(content)

        if dates and numbers:
            # Pair dates with values
            for i, date in enumerate(dates):
                if i < len(numbers):
                    value_str, unit = numbers[i]
                    value = _parse_number(value_str)
                    if value is not None:
                        points.append(
                            DataPoint(
                                label=date,
                                value=value,
                                unit=unit,
                                neuron_id=neuron_id,
                                source_id=source_id,
                                date=date,
                            )
                        )
        elif numbers:
            # Just numeric values — use content summary as label
            for value_str, unit in numbers[:5]:  # Cap at 5 per neuron
                value = _parse_number(value_str)
                if value is not None:
                    label = content[:50].strip()
                    points.append(
                        DataPoint(
                            label=label,
                            value=value,
                            unit=unit,
                            neuron_id=neuron_id,
                            source_id=source_id,
                        )
                    )

    return points


def detect_chart_type(data_points: list[DataPoint]) -> str:
    """Auto-detect best chart type from data shape."""
    if not data_points:
        return "table"

    has_dates = any(dp.date for dp in data_points)
    has_numeric = any(isinstance(dp.value, (int, float)) for dp in data_points)
    n_points = len(data_points)

    if has_dates and has_numeric:
        return "line"  # Time series → line chart
    if has_numeric and n_points <= 6:
        return "pie"
    if has_numeric and n_points > 6:
        return "bar"
    return "table"


def generate_chart(
    data_points: list[DataPoint],
    *,
    chart_type: str | None = None,
    title: str = "",
    output_format: str = "vega_lite",
) -> ChartSpec:
    """Generate a chart specification from data points.

    Args:
        data_points: Extracted data points.
        chart_type: Override chart type (auto-detect if None).
        title: Chart title.
        output_format: 'vega_lite', 'markdown_table', 'ascii', or 'all'.

    Returns:
        ChartSpec with requested format(s) filled in.
    """
    if not data_points:
        return ChartSpec(
            chart_type="table",
            title=title or "No data",
            markdown="No data points found.",
        )

    ct = chart_type if chart_type in CHART_TYPES else detect_chart_type(data_points)
    chart_title = title or _infer_title(data_points)
    provenance = tuple(dict.fromkeys(dp.neuron_id for dp in data_points if dp.neuron_id))

    vega: dict[str, Any] = {}
    md = ""
    ascii_out = ""

    if output_format in ("vega_lite", "all"):
        vega = _to_vega_lite(data_points, ct, chart_title)
    if output_format in ("markdown_table", "all"):
        md = _to_markdown(data_points, chart_title)
    if output_format in ("ascii", "all"):
        ascii_out = _to_ascii(data_points, chart_title)

    return ChartSpec(
        chart_type=ct,
        title=chart_title,
        data_points=tuple(data_points),
        vega_lite=vega,
        markdown=md,
        ascii_chart=ascii_out,
        provenance=provenance,
    )


def _parse_number(s: str) -> float | None:
    """Parse a number string, handling commas."""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _infer_title(data_points: list[DataPoint]) -> str:
    """Infer chart title from data labels."""
    units = {dp.unit for dp in data_points if dp.unit}
    unit_str = f" ({next(iter(units))})" if len(units) == 1 else ""
    return f"Data Overview{unit_str}"


def _to_vega_lite(
    data_points: list[DataPoint],
    chart_type: str,
    title: str,
) -> dict[str, Any]:
    """Generate Vega-Lite JSON spec."""
    values = [{"label": dp.label, "value": dp.value} for dp in data_points]

    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "data": {"values": values},
        "width": 400,
        "height": 250,
    }

    if chart_type == "line":
        spec["mark"] = {"type": "line", "point": True}
        spec["encoding"] = {
            "x": {"field": "label", "type": "ordinal", "title": "Period"},
            "y": {"field": "value", "type": "quantitative", "title": "Value"},
        }
    elif chart_type == "bar":
        spec["mark"] = {"type": "bar"}
        spec["encoding"] = {
            "x": {"field": "label", "type": "nominal", "title": "Category"},
            "y": {"field": "value", "type": "quantitative", "title": "Value"},
        }
    elif chart_type == "pie":
        spec["mark"] = {"type": "arc"}
        spec["encoding"] = {
            "theta": {"field": "value", "type": "quantitative"},
            "color": {"field": "label", "type": "nominal"},
        }
    elif chart_type == "scatter":
        spec["mark"] = {"type": "point"}
        spec["encoding"] = {
            "x": {"field": "label", "type": "ordinal"},
            "y": {"field": "value", "type": "quantitative"},
        }
    else:
        # table fallback — no mark
        spec["mark"] = {"type": "bar"}
        spec["encoding"] = {
            "x": {"field": "label", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        }

    return spec


def _to_markdown(data_points: list[DataPoint], title: str) -> str:
    """Generate markdown table."""
    if not data_points:
        return "No data."

    unit = data_points[0].unit
    value_header = f"Value ({unit})" if unit else "Value"

    lines = [
        f"### {title}",
        "",
        f"| Label | {value_header} |",
        "|-------|-------:|",
    ]
    for dp in data_points:
        formatted = f"{dp.value:,.2f}" if dp.value != int(dp.value) else f"{int(dp.value):,}"
        lines.append(f"| {dp.label} | {formatted} |")

    lines.append("")
    lines.append(
        f"*{len(data_points)} data points from {len({dp.neuron_id for dp in data_points})} memories*"
    )

    return "\n".join(lines)


def _to_ascii(data_points: list[DataPoint], title: str) -> str:
    """Generate simple ASCII bar chart."""
    if not data_points:
        return "No data."

    max_val = max(dp.value for dp in data_points)
    min_val = min(dp.value for dp in data_points)
    val_range = max_val - min_val if max_val != min_val else 1.0
    bar_width = 30
    max_label_len = min(max(len(dp.label) for dp in data_points), 20)

    lines = [title, "=" * (max_label_len + bar_width + 15)]

    for dp in data_points:
        label = dp.label[:20].ljust(max_label_len)
        normalized = (dp.value - min_val) / val_range if val_range else 0.5
        bar_len = max(1, int(normalized * bar_width))
        bar = "█" * bar_len
        formatted = f"{dp.value:,.1f}"
        lines.append(f"{label} │{bar} {formatted}")

    return "\n".join(lines)
