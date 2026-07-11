"""Lexicon-based sentiment extraction — no LLM dependency.

Extracts emotional valence, intensity, and emotion tags from text
using curated English word lexicons with negation and intensifier handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Valence(StrEnum):
    """Emotional polarity of text content."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentResult:
    """Result of sentiment extraction.

    Attributes:
        valence: Overall emotional polarity
        intensity: Strength of sentiment (0.0-1.0)
        emotion_tags: Specific emotion categories detected
        positive_count: Number of positive signals found
        negative_count: Number of negative signals found
    """

    valence: Valence
    intensity: float
    emotion_tags: frozenset[str]
    positive_count: int
    negative_count: int


# --- English lexicons ---

_POSITIVE_EN: frozenset[str] = frozenset(
    {
        # General positive
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "awesome",
        "outstanding",
        "perfect",
        "brilliant",
        "superb",
        "beautiful",
        "nice",
        "fine",
        "lovely",
        # Emotional positive
        "happy",
        "glad",
        "pleased",
        "satisfied",
        "delighted",
        "joyful",
        "excited",
        "thrilled",
        "eager",
        "enthusiastic",
        "optimistic",
        "grateful",
        "thankful",
        "proud",
        "confident",
        "hopeful",
        "relieved",
        "comfortable",
        "calm",
        "peaceful",
        # Achievement / progress
        "success",
        "successful",
        "accomplished",
        "achieved",
        "completed",
        "solved",
        "fixed",
        "resolved",
        "improved",
        "working",
        "done",
        "progress",
        "milestone",
        "breakthrough",
        "victory",
        "won",
        # Technical positive
        "clean",
        "fast",
        "efficient",
        "stable",
        "reliable",
        "robust",
        "scalable",
        "elegant",
        "optimized",
        "performant",
        "smooth",
        # Approval
        "love",
        "like",
        "enjoy",
        "appreciate",
        "recommend",
        "approve",
        "agree",
        "accept",
        "impressive",
        "remarkable",
    }
)

_NEGATIVE_EN: frozenset[str] = frozenset(
    {
        # General negative
        "bad",
        "terrible",
        "horrible",
        "awful",
        "poor",
        "worst",
        "ugly",
        "dreadful",
        "miserable",
        "pathetic",
        "lousy",
        # Emotional negative
        "frustrated",
        "annoyed",
        "irritated",
        "angry",
        "furious",
        "upset",
        "disappointed",
        "sad",
        "unhappy",
        "depressed",
        "worried",
        "anxious",
        "stressed",
        "nervous",
        "scared",
        "confused",
        "lost",
        "stuck",
        "overwhelmed",
        "exhausted",
        # Failure / problems
        "failed",
        "broken",
        "crashed",
        "error",
        "bug",
        "issue",
        "problem",
        "failure",
        "mistake",
        "wrong",
        "fault",
        "regression",
        "degraded",
        "corrupted",
        "flawed",
        # Technical negative
        "slow",
        "unstable",
        "unreliable",
        "fragile",
        "brittle",
        "bloated",
        "messy",
        "hacky",
        "spaghetti",
        "legacy",
        "deprecated",
        "outdated",
        "vulnerable",
        "insecure",
        # Disapproval
        "hate",
        "dislike",
        "reject",
        "refuse",
        "deny",
        "disagree",
        "unacceptable",
        "painful",
        "nightmare",
    }
)

# --- Negators ---

_NEGATORS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "none",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "can't",
        "cannot",
        "couldn't",
        "shouldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "without",
        "hardly",
        "barely",
        "scarcely",
    }
)

# --- Intensifiers ---

_INTENSIFIERS: frozenset[str] = frozenset(
    {
        "very",
        "extremely",
        "highly",
        "really",
        "so",
        "incredibly",
        "absolutely",
        "completely",
        "totally",
        "utterly",
        "deeply",
        "seriously",
        "terribly",
        "remarkably",
        "exceptionally",
        "particularly",
    }
)

# --- Emotion tag mapping ---

_EMOTION_MAP: dict[str, frozenset[str]] = {
    "frustration": frozenset(
        {
            "frustrated",
            "stuck",
            "annoyed",
            "irritated",
            "angry",
            "furious",
        }
    ),
    "satisfaction": frozenset(
        {
            "satisfied",
            "happy",
            "pleased",
            "accomplished",
            "glad",
            "delighted",
            "proud",
        }
    ),
    "confusion": frozenset(
        {
            "confused",
            "unclear",
            "lost",
            "puzzling",
            "puzzled",
            "bewildered",
        }
    ),
    "excitement": frozenset(
        {
            "excited",
            "eager",
            "thrilled",
            "amazing",
            "enthusiastic",
            "awesome",
            "fantastic",
            "brilliant",
        }
    ),
    "anxiety": frozenset(
        {
            "worried",
            "concerned",
            "anxious",
            "stressed",
            "nervous",
            "scared",
        }
    ),
    "relief": frozenset(
        {
            "relieved",
            "finally",
            "solved",
            "resolved",
            "fixed",
        }
    ),
    "disappointment": frozenset(
        {
            "disappointed",
            "let down",
            "underwhelming",
            "failed",
        }
    ),
}

# Build reverse lookup: word → set of emotion tags
_WORD_TO_EMOTIONS: dict[str, set[str]] = {}
for _emotion, _words in _EMOTION_MAP.items():
    for _word in _words:
        _WORD_TO_EMOTIONS.setdefault(_word, set()).add(_emotion)
del _emotion, _words, _word

# Token pattern: split on whitespace and common punctuation
_TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ỹ']+")

# Negation window: how many tokens ahead a negator affects
_NEGATION_WINDOW = 2


class SentimentExtractor:
    """Extract sentiment from text using curated English lexicons.

    Handles negation, intensifier detection, and emotion tag mapping.
    """

    def extract(self, text: str, language: str = "auto") -> SentimentResult:
        """Extract sentiment from text.

        Args:
            text: Input text to analyze
            language: Accepted for caller compatibility; ignored (English-only).

        Returns:
            SentimentResult with valence, intensity, and emotion tags
        """
        del language  # English-only; retained in signature for caller compatibility
        if not text or len(text.strip()) < 3:
            return SentimentResult(
                valence=Valence.NEUTRAL,
                intensity=0.0,
                emotion_tags=frozenset(),
                positive_count=0,
                negative_count=0,
            )

        positive_words = _POSITIVE_EN
        negative_words = _NEGATIVE_EN

        # Tokenize
        tokens = _TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return SentimentResult(
                valence=Valence.NEUTRAL,
                intensity=0.0,
                emotion_tags=frozenset(),
                positive_count=0,
                negative_count=0,
            )

        positive_count = 0
        negative_count = 0
        has_intensifier = False
        emotion_tags: set[str] = set()

        # Track negation: tokens remaining in negation window
        negation_remaining = 0

        for token in tokens:
            # Check for negator
            if token in _NEGATORS:
                negation_remaining = _NEGATION_WINDOW
                continue

            # Check for intensifier
            if token in _INTENSIFIERS:
                has_intensifier = True
                continue

            is_negated = negation_remaining > 0

            # Check positive lexicon
            if token in positive_words:
                if is_negated:
                    negative_count += 1
                else:
                    positive_count += 1

            # Check negative lexicon
            if token in negative_words:
                if is_negated:
                    positive_count += 1
                else:
                    negative_count += 1

            # Collect emotion tags
            if token in _WORD_TO_EMOTIONS:
                for etag in _WORD_TO_EMOTIONS[token]:
                    emotion_tags.add(etag)

            # Decrement negation window
            if negation_remaining > 0:
                negation_remaining -= 1

        # Compute intensity
        total_signals = positive_count + negative_count
        if total_signals == 0:
            return SentimentResult(
                valence=Valence.NEUTRAL,
                intensity=0.0,
                emotion_tags=frozenset(emotion_tags),
                positive_count=0,
                negative_count=0,
            )

        intensity = min(1.0, total_signals / max(1, len(tokens)) * 5.0)

        # Apply intensifier boost
        if has_intensifier:
            intensity = min(1.0, intensity * 1.5)

        # Determine valence
        if positive_count > negative_count:
            valence = Valence.POSITIVE
        elif negative_count > positive_count:
            valence = Valence.NEGATIVE
        else:
            valence = Valence.NEUTRAL

        return SentimentResult(
            valence=valence,
            intensity=round(intensity, 3),
            emotion_tags=frozenset(emotion_tags),
            positive_count=positive_count,
            negative_count=negative_count,
        )
