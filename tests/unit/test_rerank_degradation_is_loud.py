"""Regression test: a failing reranker must never degrade recall *silently*.

When the reranker is enabled but unavailable (e.g. llamastash restarted and the
rerank model has not been re-loaded, which answers HTTP 501), recall used to fall
back to the raw spreading-activation ordering and only log a warning. The caller
saw results that looked exactly like reranked ones. Recall now retries once and
reports the degradation via ``on_degraded`` so it can reach the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from surreal_memory.engine.reranker import rerank_activations

ENDPOINT = "http://127.0.0.1:11435/v1"


@dataclass
class _Act:
    """Stand-in for ActivationResult (dataclasses.replace() is used on it)."""

    activation_level: float


def _activations() -> dict[str, Any]:
    return {"n1": _Act(0.8), "n2": _Act(0.5)}


def _contents() -> dict[str, str]:
    return {"n1": "alpha content", "n2": "beta content"}


class _RerankedItem:
    def __init__(self, neuron_id: str, blended_score: float) -> None:
        self.neuron_id = neuron_id
        self.blended_score = blended_score


def test_failing_reranker_reports_degradation_and_retries(monkeypatch: Any) -> None:
    """A reranker that always raises: recall keeps results, but reports it."""
    attempts: list[int] = []

    class _BrokenReranker:
        def __init__(self, **_kwargs: Any) -> None: ...

        def rerank(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            attempts.append(1)
            raise RuntimeError("HTTP Error 501: Not Implemented")

    monkeypatch.setattr(
        "surreal_memory.engine.reranker.HttpReranker", _BrokenReranker, raising=True
    )

    reasons: list[str] = []
    activations = _activations()

    result = rerank_activations(
        "query",
        activations,
        _contents(),
        endpoint=ENDPOINT,
        on_degraded=reasons.append,
    )

    # Recall is never broken: the caller still gets its activations back.
    assert result is activations
    # ...but the degradation is reported, not swallowed.
    assert reasons, "reranker failure was silent — the caller cannot tell"
    assert "501" in reasons[0]
    # ...and a transient failure gets a second chance before giving up.
    assert len(attempts) == 2, f"expected one retry, got {len(attempts)} attempt(s)"


def test_transient_failure_recovers_on_retry(monkeypatch: Any) -> None:
    """A reranker that fails once then succeeds must NOT report degradation."""

    class _FlakyReranker:
        calls = 0

        def __init__(self, **_kwargs: Any) -> None: ...

        def rerank(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            _FlakyReranker.calls += 1
            if _FlakyReranker.calls == 1:
                raise RuntimeError("model still loading")
            return [_RerankedItem("n2", 0.95), _RerankedItem("n1", 0.4)]

    monkeypatch.setattr(
        "surreal_memory.engine.reranker.HttpReranker", _FlakyReranker, raising=True
    )

    reasons: list[str] = []

    result = rerank_activations(
        "query",
        _activations(),
        _contents(),
        endpoint=ENDPOINT,
        on_degraded=reasons.append,
    )

    assert not reasons, f"retry succeeded but degradation was reported: {reasons}"
    # Reranked order won: n2 outranks n1 despite the lower initial activation.
    assert result["n2"].activation_level == 0.95


def test_no_reranker_configured_is_reported(monkeypatch: Any) -> None:
    """No endpoint and no local CrossEncoder is itself a reportable degradation."""
    monkeypatch.setattr(
        "surreal_memory.engine.reranker._check_cross_encoder", lambda: False, raising=True
    )
    monkeypatch.setattr(
        "surreal_memory.engine.reranker._rerank_endpoint", lambda: "", raising=True
    )

    reasons: list[str] = []
    activations = _activations()

    result = rerank_activations(
        "query",
        activations,
        _contents(),
        endpoint="",
        on_degraded=reasons.append,
    )

    assert result is activations
    assert reasons and "no reranker configured" in reasons[0]
