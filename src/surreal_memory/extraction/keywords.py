"""Keyword extraction from text with Vietnamese word segmentation support."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# English stop words — standard
STOP_WORDS_EN: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "what",
        "which",
        "who",
        "whom",
    }
)

# Conversational English — common contractions without apostrophes,
# filler words, and profanity that keyword extraction should skip.
# These produce noisy bigrams ("like dont", "fucking fall") when
# left in the token stream.
_STOP_WORDS_CONVERSATIONAL_EN: frozenset[str] = frozenset(
    {
        # Contractions without apostrophes (casual typing)
        "dont",
        "doesnt",
        "didnt",
        "wont",
        "wouldnt",
        "couldnt",
        "shouldnt",
        "cant",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "hadnt",
        "im",
        "ive",
        "id",
        "youre",
        "youve",
        "youll",
        "youd",
        "hes",
        "shes",
        "its",
        "were",
        "theyre",
        "theyve",
        "theyll",
        "thats",
        "theres",
        "heres",
        "whats",
        "lets",
        # Common filler / hedging
        "like",
        "really",
        "actually",
        "basically",
        "literally",
        "honestly",
        "seriously",
        "obviously",
        "suppose",
        "guess",
        "thing",
        "things",
        "something",
        "anything",
        "everything",
        "nothing",
        "kinda",
        "sorta",
        "gonna",
        "gotta",
        "wanna",
        "dunno",
        "yeah",
        "nah",
        "yep",
        "nope",
        "hey",
        "oh",
        "ugh",
        "lol",
        "lmao",
        "idk",
        "tbh",
        "imo",
        "imho",
        # Profanity (no topical signal)
        "fucking",
        "fuck",
        "shit",
        "damn",
        "hell",
        "crap",
        # Common verbs that rarely carry topical signal
        "think",
        "know",
        "want",
        "make",
        "go",
        "going",
        "come",
        "take",
        "give",
        "tell",
        "say",
        "said",
        "get",
        "got",
        "went",
        "put",
        "look",
        "looking",
    }
)

# Polish stop words — conjunctions/particles, prepositions, pronouns,
# auxiliary/copular verb forms, functional numerals, functional adverbs.
# Both diacritic and ASCII forms included: live content is frequently typed
# without diacritics (e.g. "zostaly", "reguly", "sie", "ze").
STOP_WORDS_PL: frozenset[str] = frozenset(
    {
        # spójniki i partykuły (conjunctions and particles)
        "i",
        "a",
        "o",
        "u",
        "w",
        "z",
        "ze",
        "że",
        "iz",
        "iż",
        "we",
        "oraz",
        "ale",
        "lecz",
        "lub",
        "albo",
        "ani",
        "czy",
        "bo",
        "wiec",
        "więc",
        "czyli",
        "jednak",
        "przeciez",
        "przecież",
        "tez",
        "też",
        "takze",
        "także",
        "rowniez",
        "również",
        "tylko",
        "juz",
        "już",
        "jeszcze",
        "no",
        "tak",
        "nie",
        "niech",
        "by",
        "aby",
        "zeby",
        "żeby",
        "gdyby",
        "jesli",
        "jeśli",
        "jezeli",
        "jeżeli",
        "poniewaz",
        "ponieważ",
        "gdyż",
        "gdyz",
        # przyimki (prepositions) — "pod" deliberately EXCLUDED, see collision table
        "na",
        "do",
        "od",
        "po",
        "za",
        "przy",
        "dla",
        "bez",
        "nad",
        "przez",
        "przed",
        "miedzy",
        "między",
        "wsrod",
        "wśród",
        "obok",
        "oprocz",
        "oprócz",
        "podczas",
        "wedlug",
        "według",
        # zaimki (pronouns, incl. inflections) — both diacritic and ASCII forms
        "ja",
        "ty",
        "my",
        "wy",
        "on",
        "ona",
        "ono",
        "oni",
        "one",
        "go",
        "mu",
        "ją",
        "jej",
        "jego",
        "ich",
        "im",
        "nas",
        "was",
        "nam",
        "wam",
        "mnie",
        "mi",
        "cie",
        "cię",
        # "ci" deliberately EXCLUDED: it collides with "CI" (continuous
        # integration), which is ubiquitous in this project's dev-session
        # content — exactly the ASCII-collision class this PR removes for
        # Vietnamese ("ai"/"em"). See the "pod" collision note above.
        "sie",
        "się",
        "sobie",
        "soba",
        "sobą",
        "ten",
        "ta",
        "to",
        "te",
        "tę",
        "tego",
        "tej",
        "tym",
        "tych",
        "tamten",
        "tamta",
        "tamto",
        "ktory",
        "który",
        "ktora",
        "która",
        "ktore",
        "które",
        "ktorego",
        "którego",
        "ktorej",
        "której",
        "ktorym",
        "którym",
        "ktorych",
        "których",
        "jaki",
        "jaka",
        "jakie",
        "jak",
        "co",
        "cos",
        "coś",
        "kto",
        "ktos",
        "ktoś",
        "nic",
        "nikt",
        "kazdy",
        "każdy",
        "kazda",
        "każda",
        "kazde",
        "każde",
        "wszystko",
        "wszystkie",
        "wszyscy",
        "inny",
        "inna",
        "inne",
        "swoj",
        "swój",
        "swoja",
        "swoją",
        "swoje",
        "swojej",
        "swoim",
        "moj",
        "mój",
        "moja",
        "moją",
        "moje",
        "mojej",
        "moim",
        "twoj",
        "twój",
        "twoja",
        "twoją",
        "twoje",
        "twojej",
        "twoim",
        "nasz",
        "nasza",
        "nasze",
        "naszej",
        "naszym",
        "wasz",
        "wasza",
        "wasze",
        # czasowniki posiłkowe / kopuły (auxiliary & copular verb forms)
        "jest",
        "sa",
        "są",
        "byl",
        "był",
        "byla",
        "była",
        "bylo",
        "było",
        "byly",
        "były",
        "byc",
        "być",
        "bede",
        "będę",
        "bedzie",
        "będzie",
        "beda",
        "będą",
        "zostal",
        "został",
        "zostala",
        "została",
        "zostalo",
        "zostało",
        "zostaly",
        "zostały",
        "zostac",
        "zostać",
        "zostanie",
        "ma",
        "mam",
        "masz",
        "mamy",
        "maja",
        "mają",
        "miec",
        "mieć",
        "mial",
        "miał",
        "miala",
        "miała",
        # liczebniki funkcyjne (numeral function words) — grounded in live junk
        # samples "activity dwie" / "zostaly dwie" (Phase 1 §4)
        "jeden",
        "jedna",
        "jedno",
        "dwa",
        "dwie",
        "dwoch",
        "dwóch",
        "trzy",
        "cztery",
        "piec",
        "pięć",
        "kilka",
        "wiele",
        "pare",
        "parę",
        "oba",
        "obie",
        "obu",
        # przysłówki funkcyjne (functional adverbs)
        "bardzo",
        "moze",
        "może",
        "mozna",
        "można",
        "trzeba",
        "nalezy",
        "należy",
        "wlasnie",
        "właśnie",
        "chyba",
        "raczej",
        "prawie",
        "okolo",
        "około",
        "tutaj",
        "tam",
        "gdzie",
        "kiedy",
        "gdy",
        "potem",
        "wtedy",
        "teraz",
        "dzis",
        "dziś",
        "dzisiaj",
        "wczoraj",
        "jutro",
    }
)

# Status-speak verbs common in this system's PL logs — analogous to
# _STOP_WORDS_CONVERSATIONAL_EN. Kept deliberately small and conservative.
_STOP_WORDS_CONVERSATIONAL_PL: frozenset[str] = frozenset(
    {
        "dotyczy",
        "wiem",
        "wiesz",
        "mysle",
        "myślę",
        "widze",
        "widzę",
        "sprawdze",
        "sprawdzę",
        "zrobie",
        "zrobię",
        "prosze",
        "proszę",
    }
)

# Combined stop words for backward compatibility ("auto" mode).
STOP_WORDS: frozenset[str] = (
    STOP_WORDS_EN | _STOP_WORDS_CONVERSATIONAL_EN | STOP_WORDS_PL | _STOP_WORDS_CONVERSATIONAL_PL
)

# Clause boundary: bigrams never pair words that cross one of these.
# Em-dash (—) and en-dash are both included — each sets off a parenthetical /
# separate clause the same as a comma (live-proven: "keywords.py — bigramy"
# otherwise lets the "py" filename fragment glue onto the next clause's first
# word; editors and OSes emit the en-dash just as often). Plain ASCII hyphen is
# deliberately NOT included: it's mid-word in compounds like "write-gate", which
# must still yield the useful bigram "write gate".
_CLAUSE_BOUNDARY = re.compile(r"[.,;:!?\n\r—–]+")  # noqa: RUF001 (literal en-dash intended)


def _get_stop_words(language: str, text: str) -> frozenset[str]:
    """Get appropriate stop words for the detected language."""
    if language == "en":
        return STOP_WORDS_EN | _STOP_WORDS_CONVERSATIONAL_EN
    if language == "pl":
        return (
            STOP_WORDS_PL
            | _STOP_WORDS_CONVERSATIONAL_PL
            | STOP_WORDS_EN
            | _STOP_WORDS_CONVERSATIONAL_EN
        )
    # "auto" — use all
    return STOP_WORDS


@dataclass(frozen=True)
class WeightedKeyword:
    """A keyword with an importance weight.

    Attributes:
        text: The keyword text (unigram or bi-gram)
        weight: Importance weight (0.0 - 1.5), higher = more important
    """

    text: str
    weight: float


#: Above this fraction of all-caps tokens, a text is shouted text or a
#: heading, not prose with a sparse acronym in it -- acronym rescue (below)
#: is skipped so a fully-uppercase note doesn't let every stop word back in.
#: Measured: genuine acronym usage sits at ~0.17-0.20 of tokens; fully-caps
#: text sits at ~1.0. Wide margin on both sides.
_MAX_ACRONYM_CAPS_RATIO = 0.6


def extract_weighted_keywords(
    text: str,
    min_length: int = 2,
    language: str = "auto",
) -> list[WeightedKeyword]:
    """
    Extract weighted keywords with bi-gram support.

    Scoring factors:
    - Position: earlier words score higher (1.0 → 0.5 linear decay)
    - Bi-grams: adjacent non-stop-word pairs get averaged weight * 1.2 boost

    An ALL-CAPS token (e.g. ``MA``) survives even when its lowercased form is
    a stop word (Polish ``ma``, "has"): punctuation already prevents the
    collisions that motivated removing such words from the stop list
    (``N/A`` and ``S.A.`` tokenize as two short fragments, not ``na``/``sa``),
    so only the bare uppercase form actually collides, and only that form
    is rescued. Every other token is unaffected — still lowercased, still
    filtered exactly as before.

    Args:
        text: The text to extract from
        min_length: Minimum word length for unigrams
        language: Language hint ("en", "pl", or "auto")

    Returns:
        List of WeightedKeyword sorted by weight descending
    """
    stop_words = _get_stop_words(language, text)

    # Tokenize the ORIGINAL text (not lowercased): _CLAUSE_BOUNDARY and the
    # word regex below only match punctuation/letter classes, so this yields
    # the identical clause/word boundaries as before -- only each token's own
    # casing is preserved for the acronym check that follows.
    raw_tokens: list[str] = []
    clause_of: list[int] = []
    for clause_idx, clause in enumerate(_CLAUSE_BOUNDARY.split(text)):
        for w in re.findall(r"\b[a-zA-ZÀ-ỹ]+(?:_[a-zA-ZÀ-ỹ]+)*\b", clause):
            raw_tokens.append(w)
            clause_of.append(clause_idx)

    caps_ratio = sum(1 for w in raw_tokens if w.isupper()) / len(raw_tokens) if raw_tokens else 0.0
    rescue_acronyms = caps_ratio < _MAX_ACRONYM_CAPS_RATIO

    words: list[str] = []
    for w in raw_tokens:
        lower = w.lower()
        is_stop_word = lower.replace("_", " ") in stop_words or lower in stop_words
        # Tight rule: the WHOLE token must be uppercase in the source, not
        # just capitalized -- str.isupper() already rejects "Ma"/"Na"
        # (sentence-initial capitalization), which must stay filtered.
        if is_stop_word and rescue_acronyms and w.isupper() and len(w.replace("_", "")) >= 2:
            words.append(w)
        else:
            words.append(lower)

    # Filter to content words with original position and clause id
    filtered: list[tuple[str, int, int]] = [
        (w, i, clause_of[i])
        for i, w in enumerate(words)
        if len(w.replace("_", "")) >= min_length
        and w.replace("_", " ") not in stop_words
        and w not in stop_words
    ]

    if not filtered:
        return []

    total = len(filtered)
    weighted: dict[str, float] = {}

    # Unigrams with position decay (1.0 at start → 0.5 at end)
    for idx, (word, _orig_pos, _clause_idx) in enumerate(filtered):
        position_weight = 1.0 - 0.5 * (idx / max(1, total - 1))
        # Store with underscores replaced by spaces for readability
        display_word = word.replace("_", " ")
        weighted[display_word] = max(weighted.get(display_word, 0.0), position_weight)

    # Bi-grams from adjacent non-stop words within the same clause, gap <= 2 original word positions
    for i in range(len(filtered) - 1):
        w1, p1, c1 = filtered[i]
        w2, p2, c2 = filtered[i + 1]
        if c1 == c2 and p2 - p1 <= 2:
            dw1 = w1.replace("_", " ")
            dw2 = w2.replace("_", " ")
            bigram = f"{dw1} {dw2}"
            bigram_weight = (weighted.get(dw1, 0.5) + weighted.get(dw2, 0.5)) / 2 * 1.2
            weighted[bigram] = max(weighted.get(bigram, 0.0), bigram_weight)

    results = [WeightedKeyword(text=k, weight=v) for k, v in weighted.items()]
    results.sort(key=lambda x: x.weight, reverse=True)
    return results


def extract_keywords(
    text: str,
    min_length: int = 2,
    language: str = "auto",
) -> list[str]:
    """
    Extract keywords from text, sorted by importance.

    Backward-compatible wrapper around extract_weighted_keywords().
    Returns bi-grams before unigrams, ordered by weight.

    Args:
        text: The text to extract from
        min_length: Minimum word length
        language: Language hint ("en", "pl", or "auto")

    Returns:
        List of keyword strings
    """
    weighted = extract_weighted_keywords(text, min_length, language=language)
    return [kw.text for kw in weighted]
