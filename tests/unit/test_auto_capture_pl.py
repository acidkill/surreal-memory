"""Tests for auto_capture: truncation fix and Polish pattern coverage.

PLAN §B2 audit finding: patterns were English-only, and the over-capture
trim cut at the FIRST sentence boundary after char 50 — amputating a causal
"...because Y" clause that usually lands past that point. Both fixed here.
"""

from __future__ import annotations

from surreal_memory.mcp.auto_capture import (
    _detect_patterns,
    analyze_text_for_memories,
)


class TestTruncationKeepsTheCausalClause:
    def test_reason_past_the_old_50_char_cutoff_survives(self) -> None:
        # The reason clause ("because it already had...") starts well past
        # char 50 but the whole capture is under 300 — must not be cut off.
        text = (
            "We decided to use PostgreSQL for the new service because it "
            "already had strong JSONB support across the team's stack."
        )
        detected = analyze_text_for_memories(text, capture_decisions=True)
        decisions = [d for d in detected if d["type"] == "decision"]
        assert decisions
        assert any("because" in d["content"] for d in decisions)

    def test_bounded_capture_with_no_early_boundary_still_bounded(self) -> None:
        # No sentence-ending punctuation anywhere near the front — the old
        # "first boundary after 50" search would never fire either, but the
        # new "last boundary before 300" search must still bound the result.
        run_on = "reason " * 80  # ~560 chars, no punctuation at all
        detected = _detect_patterns(
            f"the decision is: {run_on}",
            [r"(?:the )?decision(?: is| was)?[:\s]+(.+?)(?:\.|$)"],
            "decision",
            0.8,
            6,
            10,
        )
        assert detected
        assert len(detected[0]["content"]) <= 300

    def test_last_boundary_before_300_is_used_not_first_after_50(self) -> None:
        # Two candidate boundaries, neither a literal "." (the regex's own
        # non-greedy `.+?` only stops at "." itself, so both survive into
        # the captured text for the trim step to choose between): an early
        # "!" well after char 50 (what the OLD "first after 50" search
        # would cut at) and a later "!" still before char 300 (what the NEW
        # "last before 300" search should use instead).
        early_marker = "x" * 60 + "!"
        late_marker = "y" * 200 + "!"
        tail = "z" * 50
        text = f"the decision is: {early_marker} {late_marker} {tail}"

        detected = _detect_patterns(
            text,
            [r"(?:the )?decision(?: is| was)?[:\s]+(.+?)(?:\.|$)"],
            "decision",
            0.8,
            6,
            10,
        )
        assert detected
        captured = detected[0]["content"]
        assert len(captured) <= 300
        # The cut moved past the early "!" — a real chunk of the later
        # marker survived, proving the LAST boundary before 300 was used.
        assert "y" * 50 in captured
        # ...but content after the late boundary is still gone.
        assert "z" * 50 not in captured


class TestPolishPatterns:
    def test_decision_wybrano_zamiast(self) -> None:
        text = "Wybrano PostgreSQL zamiast MySQL bo lepsze wsparcie JSON."
        detected = analyze_text_for_memories(text, capture_decisions=True)
        assert "decision" in [d["type"] for d in detected]

    def test_decision_zdecydowano_sie_na(self) -> None:
        text = "Zdecydowaliśmy się na Redis zamiast Memcached dla cache'a."
        detected = analyze_text_for_memories(text, capture_decisions=True)
        assert "decision" in [d["type"] for d in detected]

    def test_decision_carries_same_priority_as_english(self) -> None:
        pl = analyze_text_for_memories("Wybrano PostgreSQL zamiast MySQL.", capture_decisions=True)
        en = analyze_text_for_memories("We chose PostgreSQL over MySQL.", capture_decisions=True)
        pl_decisions = [d for d in pl if d["type"] == "decision"]
        en_decisions = [d for d in en if d["type"] == "decision"]
        assert pl_decisions and en_decisions
        assert pl_decisions[0]["priority"] == en_decisions[0]["priority"]

    def test_error_przyczyna_byla(self) -> None:
        text = "Przyczyną błędu była zła konfiguracja połączeń do bazy."
        detected = analyze_text_for_memories(text, capture_errors=True)
        assert "error" in [d["type"] for d in detected]

    def test_error_naprawiono_przez(self) -> None:
        text = "Naprawiono przez dodanie retry z backoffem wykładniczym."
        detected = analyze_text_for_memories(text, capture_errors=True)
        assert "error" in [d["type"] for d in detected]

    def test_error_root_cause_polish_copula(self) -> None:
        text = "Root cause jest niejasny routing w warstwie sieciowej."
        detected = analyze_text_for_memories(text, capture_errors=True)
        assert "error" in [d["type"] for d in detected]

    def test_root_cause_english_stays_an_insight_not_an_error(self) -> None:
        # Regression guard: the Polish "root cause" error pattern must not
        # also match the plain English phrasing already covered by
        # INSIGHT_PATTERNS — it requires a Polish copula (był/była/to/jest).
        text = "The root cause was a race condition in the connection pool handler."
        detected = analyze_text_for_memories(text, capture_insights=True, capture_errors=True)
        types = [d["type"] for d in detected]
        assert "insight" in types

    def test_insight_okazalo_sie_ze(self) -> None:
        text = "Okazało się, że problem był w konfiguracji DNS."
        detected = analyze_text_for_memories(text, capture_insights=True)
        assert "insight" in [d["type"] for d in detected]

    def test_insight_wniosek(self) -> None:
        text = "Wniosek: zawsze sprawdzaj limity przed wdrożeniem na produkcję."
        detected = analyze_text_for_memories(text, capture_insights=True)
        assert "insight" in [d["type"] for d in detected]

    def test_preference_preferuje(self) -> None:
        text = "Preferuję krótkie funkcje nad długie klasy monolityczne."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        assert "preference" in [d["type"] for d in detected]

    def test_preference_zawsze_uzywaj(self) -> None:
        text = "Zawsze używaj parametryzowanych zapytań SQL w tym projekcie."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        assert "preference" in [d["type"] for d in detected]

    def test_preference_nigdy_nie(self) -> None:
        text = "Nigdy nie commituj sekretów do repozytorium, nawet testowych."
        detected = analyze_text_for_memories(text, capture_preferences=True)
        assert "preference" in [d["type"] for d in detected]

    def test_polish_diacritics_survive_lowercasing(self) -> None:
        # str.lower() is Unicode-aware, but pin it: ą/ć/ę/ł/ń/ó/ś/ź/ż must
        # not break pattern matching or get mangled in the captured content.
        text = "Zdecydowaliśmy się na wdrożenie ponieważ zwiększa wydajność."
        detected = analyze_text_for_memories(text, capture_decisions=True)
        assert "decision" in [d["type"] for d in detected]

    def test_polish_and_english_in_same_call_both_detected(self) -> None:
        text = (
            "We chose Redis for caching. Wybrano też PostgreSQL zamiast MySQL dla trwałych danych."
        )
        detected = analyze_text_for_memories(text, capture_decisions=True)
        decisions = [d for d in detected if d["type"] == "decision"]
        assert len(decisions) >= 2
