"""Query parser for decomposing queries into activation signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from surreal_memory.extraction.entities import Entity, EntityExtractor
from surreal_memory.extraction.keywords import extract_keywords
from surreal_memory.extraction.temporal import TemporalExtractor, TimeHint
from surreal_memory.utils.timeutils import utcnow


class QueryIntent(StrEnum):
    """The intent/purpose of a query."""

    ASK_WHAT = "ask_what"  # What happened?
    ASK_WHERE = "ask_where"  # Where did it happen?
    ASK_WHEN = "ask_when"  # When did it happen?
    ASK_WHO = "ask_who"  # Who was involved?
    ASK_WHY = "ask_why"  # Why did it happen?
    ASK_HOW = "ask_how"  # How did it happen?
    ASK_FEELING = "ask_feeling"  # How did I feel?
    ASK_PATTERN = "ask_pattern"  # What's the pattern?
    CONFIRM = "confirm"  # Did X happen?
    COMPARE = "compare"  # Compare X and Y
    RECALL = "recall"  # General recall
    UNKNOWN = "unknown"


class Perspective(StrEnum):
    """The perspective/framing of the query."""

    RECALL = "recall"  # Remember something
    CONFIRM = "confirm"  # Verify something
    COMPARE = "compare"  # Compare things
    ANALYZE = "analyze"  # Analyze/understand
    SUMMARIZE = "summarize"  # Get summary


@dataclass
class Stimulus:
    """
    Decomposed query signals for activation.

    A Stimulus represents all the extracted signals from a query
    that will be used to activate relevant neurons.

    Attributes:
        time_hints: Extracted time references
        keywords: Important keywords from the query
        entities: Named entities found
        intent: What the query is asking for
        perspective: How the query frames the request
        raw_query: The original query text
        language: Detected or specified language
    """

    time_hints: list[TimeHint]
    keywords: list[str]
    entities: list[Entity]
    intent: QueryIntent
    perspective: Perspective
    raw_query: str
    language: str = "auto"

    @property
    def has_time_context(self) -> bool:
        """Check if query has temporal constraints."""
        return len(self.time_hints) > 0

    @property
    def has_entities(self) -> bool:
        """Check if query mentions specific entities."""
        return len(self.entities) > 0

    @property
    def anchor_count(self) -> int:
        """Count of potential anchor points for activation."""
        return len(self.time_hints) + len(self.entities) + len(self.keywords)


class QueryParser:
    """
    Parser for decomposing queries into activation signals.

    The parser extracts:
    - Temporal references (time hints)
    - Named entities (people, places, etc.)
    - Keywords (important words)
    - Intent (what the query is asking)
    - Perspective (how the query frames the request)
    """

    # Intent detection patterns
    INTENT_PATTERNS: dict[QueryIntent, list[str]] = {
        QueryIntent.ASK_WHAT: [
            r"what",
            r"which",
            r"tell me about",
        ],
        QueryIntent.ASK_WHERE: [
            r"where",
            r"location",
            r"place",
        ],
        QueryIntent.ASK_WHEN: [
            r"when",
            r"what time",
        ],
        QueryIntent.ASK_WHO: [
            r"who",
            r"whom",
        ],
        QueryIntent.ASK_WHY: [
            r"why",
            r"reason",
            r"cause",
        ],
        QueryIntent.ASK_HOW: [
            r"how did",
            r"how was",
            r"how to",
        ],
        QueryIntent.ASK_FEELING: [
            r"how (?:did|do) (?:i|you) feel",
            r"feeling",
            r"emotion",
        ],
        QueryIntent.ASK_PATTERN: [
            r"usually",
            r"typically",
            r"pattern",
            r"often",
            r"always",
        ],
        QueryIntent.CONFIRM: [
            r"did (?:i|we|you)",
            r"was there",
            r"have (?:i|we|you)",
            r"is it true",
        ],
        QueryIntent.COMPARE: [
            r"compare",
            r"difference",
            r"versus",
            r"vs",
            r"better",
            r"worse",
        ],
    }

    def __init__(
        self,
        temporal_extractor: TemporalExtractor | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        """
        Initialize the parser.

        Args:
            temporal_extractor: Custom temporal extractor (creates default if None)
            entity_extractor: Custom entity extractor (creates default if None)
        """
        self._temporal = temporal_extractor or TemporalExtractor()
        self._entity = entity_extractor or EntityExtractor()

        # Compile intent patterns
        import re

        self._intent_compiled: dict[QueryIntent, list[re.Pattern[str]]] = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._intent_compiled[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def parse(
        self,
        query: str,
        reference_time: datetime | None = None,
        language: str = "auto",
    ) -> Stimulus:
        """
        Parse a query into a Stimulus.

        Args:
            query: The query text
            reference_time: Reference time for temporal parsing
            language: "en" or "auto"

        Returns:
            Stimulus containing all extracted signals
        """
        if reference_time is None:
            reference_time = utcnow()

        # English-only system: resolve "auto" to English
        if language == "auto":
            language = "en"

        # Extract components
        time_hints = self._temporal.extract(query, reference_time, language)
        entities = self._entity.extract(query, language)
        keywords = extract_keywords(query)

        # Detect intent
        intent = self._detect_intent(query)

        # Detect perspective
        perspective = self._detect_perspective(query, intent)

        return Stimulus(
            time_hints=time_hints,
            keywords=keywords,
            entities=entities,
            intent=intent,
            perspective=perspective,
            raw_query=query,
            language=language,
        )

    # Specificity weights: more specific intents score higher per match
    # to avoid generic intents (ASK_WHAT) shadowing specific ones (ASK_PATTERN).
    # Question-word intents (ASK_WHEN, ASK_WHERE, etc.) get a slight boost
    # over structural intents (CONFIRM) since "When did we..." is a WHEN question.
    _INTENT_SPECIFICITY: dict[QueryIntent, float] = {
        QueryIntent.ASK_FEELING: 1.5,
        QueryIntent.ASK_PATTERN: 1.3,
        QueryIntent.ASK_WHY: 1.2,
        QueryIntent.ASK_HOW: 1.2,
        QueryIntent.ASK_WHEN: 1.15,
        QueryIntent.ASK_WHERE: 1.15,
        QueryIntent.ASK_WHO: 1.15,
        QueryIntent.COMPARE: 1.2,
        QueryIntent.CONFIRM: 1.05,
    }

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the query intent using scored matching.

        Each intent's score = match_count * specificity_weight.
        The intent with highest score wins, resolving ambiguity when
        a query matches multiple intents (e.g. "What pattern..." matching
        both ASK_WHAT and ASK_PATTERN).
        """
        query_lower = query.lower()
        scores: dict[QueryIntent, float] = {}

        for intent, patterns in self._intent_compiled.items():
            match_count = sum(1 for p in patterns if p.search(query_lower))
            if match_count > 0:
                weight = self._INTENT_SPECIFICITY.get(intent, 1.0)
                scores[intent] = match_count * weight

        if not scores:
            return QueryIntent.RECALL

        return max(scores, key=lambda k: scores[k])

    def _detect_perspective(
        self,
        query: str,
        intent: QueryIntent,
    ) -> Perspective:
        """Detect the query perspective."""
        query_lower = query.lower()

        # Check for confirmation patterns
        if intent == QueryIntent.CONFIRM:
            return Perspective.CONFIRM

        # Check for comparison patterns
        if intent == QueryIntent.COMPARE:
            return Perspective.COMPARE

        # Check for summary patterns
        summary_patterns = ["summary", "summarize", "overview"]
        for pattern in summary_patterns:
            if pattern in query_lower:
                return Perspective.SUMMARIZE

        # Check for analysis patterns
        analysis_patterns = ["analyze", "understand", "explain"]
        for pattern in analysis_patterns:
            if pattern in query_lower:
                return Perspective.ANALYZE

        return Perspective.RECALL
