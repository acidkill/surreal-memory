"""Entity extraction from text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EntityType(StrEnum):
    """Types of named entities."""

    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    EVENT = "event"
    CODE = "code"
    UNKNOWN = "unknown"


class EntitySubtype(StrEnum):
    """Domain-specific entity subtypes for vertical intelligence."""

    # Financial
    FINANCIAL_METRIC = "financial_metric"  # ROE, revenue, EBITDA, P/E
    CURRENCY_AMOUNT = "currency_amount"  # $25M, €1.2B, 25000 USD
    FISCAL_PERIOD = "fiscal_period"  # Q1 2024, FY2025, H1/2024

    # Legal
    REGULATION = "regulation"  # Section 301 SOX, Article 5
    CONTRACT_CLAUSE = "contract_clause"  # Clause 5.2
    LEGAL_ENTITY = "legal_entity"  # LLC, Ltd, GmbH

    # Technical
    API_ENDPOINT = "api_endpoint"  # GET /api/v1/users, POST /webhook
    CODE_SYMBOL = "code_symbol"  # function_name(), ClassName, module.attr
    VERSION = "version"  # v2.1.0, Python 3.11, React 19


@dataclass(frozen=True)
class Entity:
    """
    A named entity extracted from text.

    Attributes:
        text: The original text of the entity
        type: The entity type
        subtype: Optional domain-specific subtype
        start: Start character position in source text
        end: End character position in source text
        confidence: Extraction confidence (0.0 - 1.0)
        raw_value: Original value string for verbatim recall
        unit: Unit of measurement if applicable (percent, USD, VND)
    """

    text: str
    type: EntityType
    start: int
    end: int
    subtype: EntitySubtype | None = None
    confidence: float = 1.0
    raw_value: str = ""
    unit: str = ""


class EntityExtractor:
    """
    Entity extractor using pattern matching.

    For production use, consider using spaCy for better entity
    recognition. This provides basic rule-based extraction as a fallback.
    """

    # Common location indicators
    LOCATION_INDICATORS: frozenset[str] = frozenset(
        {
            "at",
            "in",
            "to",
            "from",
            "restaurant",
            "office",
            "building",
            "hotel",
            "shop",
            "store",
        }
    )

    # Pre-compiled location patterns (avoid recompilation in hot loop)
    _LOCATION_PATTERNS: dict[str, re.Pattern[str]] = {
        indicator: re.compile(
            rf"\b{re.escape(indicator)}\s+([A-Z][a-zA-Z\s]+?)(?:[,.]|\s+(?:to|with|for)|$)",
            re.IGNORECASE,
        )
        for indicator in LOCATION_INDICATORS
    }

    # Pattern for capitalized words (potential entities)
    CAPITALIZED_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")

    # Code entity patterns
    # PascalCase: ReflexPipeline, MemoryEncoder (2+ capitalized segments)
    PASCAL_CASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
    # snake_case with 2+ segments: extract_keywords, activate_trail
    SNAKE_CASE_PATTERN = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*){1,})\b")
    # File paths: src/surreal_memory/server.py, config.toml
    FILE_PATH_PATTERN = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.\w+")

    # ── Domain extraction patterns ──────────────────────────────────

    # Financial metrics: ROE, EBITDA, P/E, EPS, revenue, etc.
    FINANCIAL_METRIC_PATTERN = re.compile(
        r"\b(ROE|ROA|ROI|EBITDA|EPS|P/E|P/B|NPM|GPM|CAGR|WACC|"
        r"IRR|NPV|ROIC|FCF|D/E|"
        r"revenue|profit|margin|earnings|net income|gross profit|"
        r"operating income|total assets|equity|debt)"
        r"\s*[=:≈]?\s*"
        r"([\d.,]+\s*%?|[\d.,]+\s*(?:billion|million|thousand|[BMKbmk])?\b)?",
        re.IGNORECASE,
    )

    # Currency amounts: $25M, €1.2B, ¥100K, 25000 USD
    CURRENCY_AMOUNT_PATTERN = re.compile(
        r"(?:"
        r"[\$€£¥]\s*[\d.,]+\s*(?:billion|million|thousand|[BMKbmk])?"  # $25M
        r"|[\d.,]+\s*(?:USD|EUR|GBP|JPY|VND)"  # 25000 USD
        r"|[\d.,]+\s*(?:billion|million)\s*(?:USD|EUR|GBP|VND)?"  # 1.2 billion USD
        r")",
        re.IGNORECASE,
    )

    # Fiscal periods: Q1 2024, FY2025, H1/2024
    FISCAL_PERIOD_PATTERN = re.compile(
        r"\b(?:"
        r"Q[1-4]\s*[/.]?\s*\d{4}"  # Q1 2024, Q3/2024
        r"|FY\s*\d{4}"  # FY2025
        r"|H[12]\s*[/.]?\s*\d{4}"  # H1/2024
        r"|(?:fiscal\s+)?(?:year|quarter)\s+\d{4}"  # fiscal year 2024
        r")\b",
        re.IGNORECASE,
    )

    # Legal: Section 301 SOX, Article 5, Clause 3
    REGULATION_PATTERN = re.compile(
        r"\b(?:"
        r"(?:Section|Article|Clause|Rule|Regulation)\s+\d+(?:\.\d+)*"
        r"(?:\s+(?:of\s+)?(?:the\s+)?[A-Z][A-Za-z\s]*?)?"  # Section 301 SOX
        r")\b",
        re.IGNORECASE,
    )

    # Contract clauses: Clause 5.2.1
    CONTRACT_CLAUSE_PATTERN = re.compile(
        r"\b(?:"
        r"(?:Clause|clause)\s+\d+(?:\.\d+)+"  # Clause 5.2.1
        r")\b",
        re.IGNORECASE,
    )

    # Legal entities: LLC, Ltd, GmbH, Corp
    LEGAL_ENTITY_PATTERN = re.compile(
        r"\b[A-Z][A-Za-z\s]+\s+(?:LLC|Ltd|Inc|Corp|GmbH|AG|S\.A\.|PLC|Co\.|Pty)\b",
    )

    # API endpoints: GET /api/v1/users, POST /webhook
    API_ENDPOINT_PATTERN = re.compile(
        r"\b(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+"
        r"(/[a-zA-Z0-9_/{}:.-]+)",
        re.IGNORECASE,
    )

    # Version: v2.1.0, Python 3.11, React 19, Node.js 20
    VERSION_PATTERN = re.compile(
        r"\b(?:"
        r"v\d+(?:\.\d+){1,3}(?:-[a-zA-Z0-9.]+)?"  # v2.1.0, v3.0.0-beta.1
        r"|(?:Python|Node\.?js?|React|Vue|Angular|Java|Go|Rust|Ruby|PHP|Swift|Kotlin|TypeScript|TS)"
        r"\s+\d+(?:\.\d+){0,2}"  # Python 3.11
        r")\b",
        re.IGNORECASE,
    )

    def __init__(self, use_nlp: bool = False) -> None:
        """
        Initialize the extractor.

        Args:
            use_nlp: If True, try to use spaCy (not implemented yet)
        """
        self._use_nlp = use_nlp
        self._nlp_en: Any = None

        if use_nlp:
            self._init_nlp()

    def _init_nlp(self) -> None:
        """Initialize NLP models if available."""
        # Try to load spaCy for English
        try:
            import spacy

            self._nlp_en = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pass

    def extract(
        self,
        text: str,
        language: str = "auto",
    ) -> list[Entity]:
        """
        Extract entities from text.

        Args:
            text: The text to extract from
            language: Accepted for backward compatibility; extraction is
                English-only, so this argument is ignored.

        Returns:
            List of Entity objects
        """
        entities: list[Entity] = []

        # Try NLP-based extraction first
        if self._use_nlp:
            nlp_entities = self._extract_with_nlp(text, language)
            if nlp_entities:
                return nlp_entities

        # Fall back to pattern-based extraction
        entities.extend(self._extract_domain_entities(text))
        entities.extend(self._extract_code_entities(text, entities))
        entities.extend(self._extract_capitalized_words(text, entities))
        entities.extend(self._extract_locations(text, entities))

        # Remove duplicates
        seen: set[str] = set()
        unique: list[Entity] = []
        for entity in entities:
            key = f"{entity.text.lower()}:{entity.type}"
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _extract_with_nlp(
        self,
        text: str,
        language: str,
    ) -> list[Entity] | None:
        """Try to extract using NLP models."""
        if self._nlp_en:
            doc = self._nlp_en(text)
            entities = []
            for ent in doc.ents:
                entity_type = self._map_spacy_type(ent.label_)
                if entity_type:
                    entities.append(
                        Entity(
                            text=ent.text,
                            type=entity_type,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.9,
                        )
                    )
            if entities:
                return entities

        return None

    def _map_spacy_type(self, label: str) -> EntityType | None:
        """Map spaCy NER label to EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "FAC": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
        }
        return mapping.get(label)

    def _extract_capitalized_words(
        self,
        text: str,
        existing: list[Entity],
    ) -> list[Entity]:
        """Extract capitalized words as potential entities."""
        entities = []
        existing_spans = {(e.start, e.end) for e in existing}

        for match in self.CAPITALIZED_PATTERN.finditer(text):
            # Skip if already extracted
            if (match.start(), match.end()) in existing_spans:
                continue

            word = match.group(1)

            # Skip common words
            if word.lower() in {"the", "a", "an", "i", "my", "we", "they"}:
                continue

            # Skip if at start of sentence (could be just capitalization)
            if match.start() == 0 or text[match.start() - 1] in ".!?\n":
                # Still include if it looks like a proper noun
                if len(word.split()) == 1 and len(word) < 4:
                    continue

            entities.append(
                Entity(
                    text=word,
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.5,
                )
            )

        return entities

    def _extract_code_entities(
        self,
        text: str,
        existing: list[Entity],
    ) -> list[Entity]:
        """Extract code identifiers (PascalCase, snake_case, file paths)."""
        entities: list[Entity] = []
        existing_spans = {(e.start, e.end) for e in existing}
        existing_texts = {e.text.lower() for e in existing}

        # PascalCase (e.g., ReflexPipeline, MemoryEncoder)
        for match in self.PASCAL_CASE_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            if match.group(1).lower() in existing_texts:
                continue
            entities.append(
                Entity(
                    text=match.group(1),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                )
            )

        # snake_case (e.g., extract_keywords, activate_trail)
        for match in self.SNAKE_CASE_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            word = match.group(1)
            if word.lower() in existing_texts:
                continue
            # Skip common non-code snake_case (e.g., stop words joined)
            if len(word) < 5:
                continue
            entities.append(
                Entity(
                    text=word,
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8,
                )
            )

        # File paths (e.g., src/surreal_memory/server.py)
        for match in self.FILE_PATH_PATTERN.finditer(text):
            if (match.start(), match.end()) in existing_spans:
                continue
            entities.append(
                Entity(
                    text=match.group(0),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                )
            )

        return entities

    def _extract_domain_entities(self, text: str) -> list[Entity]:
        """Extract financial, legal, and technical domain entities."""
        entities: list[Entity] = []

        # Financial metrics (ROE = 12.8%, revenue = 500 million)
        for match in self.FINANCIAL_METRIC_PATTERN.finditer(text):
            raw_val = match.group(2) or ""
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.FINANCIAL_METRIC,
                    confidence=0.85,
                    raw_value=raw_val.strip(),
                    unit=_detect_unit(raw_val) if raw_val else "",
                )
            )

        # Currency amounts ($25M, 1.2 billion USD)
        for match in self.CURRENCY_AMOUNT_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.CURRENCY_AMOUNT,
                    confidence=0.9,
                    raw_value=match.group(0).strip(),
                    unit=_detect_currency(match.group(0)),
                )
            )

        # Fiscal periods (Q1 2024, FY2025)
        for match in self.FISCAL_PERIOD_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.FISCAL_PERIOD,
                    confidence=0.9,
                    raw_value=match.group(0).strip(),
                )
            )

        # Regulations (Section 301 SOX, Article 5)
        for match in self.REGULATION_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.REGULATION,
                    confidence=0.85,
                    raw_value=match.group(0).strip(),
                )
            )

        # Contract clauses (Clause 5.2.1)
        for match in self.CONTRACT_CLAUSE_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.UNKNOWN,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.CONTRACT_CLAUSE,
                    confidence=0.85,
                    raw_value=match.group(0).strip(),
                )
            )

        # Legal entities (XYZ LLC, ABC Corp)
        for match in self.LEGAL_ENTITY_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.ORGANIZATION,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.LEGAL_ENTITY,
                    confidence=0.8,
                    raw_value=match.group(0).strip(),
                )
            )

        # API endpoints (GET /api/v1/users)
        for match in self.API_ENDPOINT_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.API_ENDPOINT,
                    confidence=0.9,
                    raw_value=match.group(1),
                )
            )

        # Versions (v2.1.0, Python 3.11)
        for match in self.VERSION_PATTERN.finditer(text):
            entities.append(
                Entity(
                    text=match.group(0).strip(),
                    type=EntityType.CODE,
                    start=match.start(),
                    end=match.end(),
                    subtype=EntitySubtype.VERSION,
                    confidence=0.9,
                    raw_value=match.group(0).strip(),
                )
            )

        return entities

    def _extract_locations(
        self,
        text: str,
        existing: list[Entity],
    ) -> list[Entity]:
        """Extract locations based on context indicators."""
        entities = []
        existing_texts = {e.text.lower() for e in existing}

        # Find words after location indicators (pre-compiled patterns)
        for pattern in self._LOCATION_PATTERNS.values():
            for match in pattern.finditer(text):
                location = match.group(1).strip()

                if location.lower() in existing_texts:
                    continue

                if len(location) < 2:
                    continue

                entities.append(
                    Entity(
                        text=location,
                        type=EntityType.LOCATION,
                        start=match.start(1),
                        end=match.start(1) + len(location),
                        confidence=0.7,
                    )
                )

        return entities


# ── Module-level helpers for domain extraction ─────────────────────


def _detect_unit(value: str) -> str:
    """Detect unit from a financial value string."""
    v = value.strip().lower()
    if "%" in v:
        return "percent"
    if any(w in v for w in ("billion", "b")):
        return "billion"
    if any(w in v for w in ("million", "m")):
        return "million"
    if any(w in v for w in ("thousand", "k")):
        return "thousand"
    return ""


def _detect_currency(text: str) -> str:
    """Detect currency from a currency amount string."""
    t = text.strip()
    if t.startswith("$") or "USD" in t.upper():
        return "USD"
    if t.startswith("€") or "EUR" in t.upper():
        return "EUR"
    if t.startswith("£") or "GBP" in t.upper():
        return "GBP"
    if t.startswith("¥") or "JPY" in t.upper():
        return "JPY"
    if "VND" in t.upper():
        return "VND"
    return ""
