"""Preservation counts remain visible in human and machine-readable reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.engine.compression import CompressionReport
from surreal_memory.engine.consolidation import ConsolidationReport
from surreal_memory.mcp.lifecycle_handler import LifecycleHandler


@pytest.mark.parametrize("report_type", [CompressionReport, ConsolidationReport])
def test_report_defaults_and_summary(report_type: type) -> None:
    report = report_type()
    assert report.fibers_preserved == 0
    assert report.compression_counts_complete is True
    assert [line.strip() for line in report.summary().splitlines()].count(
        "Compression originals preserved: 0"
    ) == 1

    report.fibers_preserved = 7
    assert [line.strip() for line in report.summary().splitlines()].count(
        "Compression originals preserved: 7"
    ) == 1
    payload = json.loads(json.dumps(asdict(report), default=str))
    assert payload["fibers_preserved"] == 7
    assert payload["compression_counts_complete"] is True


def test_partial_consolidation_summary_marks_known_count() -> None:
    report = ConsolidationReport(fibers_preserved=3, compression_counts_complete=False)
    line = next(
        line.strip()
        for line in report.summary().splitlines()
        if "Compression originals preserved:" in line
    )
    assert line.startswith("Compression originals preserved: 3")
    assert "partial" in line.lower()
    assert "checkpoint" in line.lower()
    payload = json.loads(json.dumps(asdict(report), default=str))
    assert payload["fibers_preserved"] == 3
    assert payload["compression_counts_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False])
async def test_mcp_report_exposes_preservation_count_and_completeness(complete: bool) -> None:
    report = ConsolidationReport(fibers_preserved=3, compression_counts_complete=complete)
    delta = SimpleNamespace(report=report, to_dict=lambda: {"before": {}, "after": {}, "delta": {}})
    storage = SimpleNamespace(_current_brain_id="test-brain", brain_id="test-brain")
    handler = LifecycleHandler()
    handler.config = SimpleNamespace(is_pro=lambda: False)
    with (
        patch.object(
            handler, "get_storage", new_callable=AsyncMock, return_value=storage, create=True
        ),
        patch(
            "surreal_memory.engine.consolidation_delta.run_with_delta",
            new_callable=AsyncMock,
            return_value=delta,
        ),
    ):
        result = await handler._consolidate({"strategy": "compress"})

    payload = json.loads(json.dumps(result))
    assert payload["report"]["fibers_preserved"] == 3
    assert payload["report"]["compression_counts_complete"] is complete
    assert "Compression originals preserved: 3" in payload["summary"]
