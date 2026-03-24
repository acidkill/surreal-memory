"""Smart query expansion — synonym, abbreviation, and cross-language expansion.

Expands query keywords at retrieval time so that keyword anchor search
finds more relevant neurons without requiring embedding. All maps are
bidirectional and case-insensitive.
"""

from __future__ import annotations

# ── Synonym groups ────────────────────────────────────────────────
# Each frozenset is a bidirectional synonym group: looking up any member
# returns all others.  Add new groups freely.

_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    # English — general
    frozenset({"cost", "expense", "spending", "expenditure"}),
    frozenset({"revenue", "income", "earnings", "sales"}),
    frozenset({"error", "bug", "issue", "failure", "fault"}),
    frozenset({"auth", "authentication", "authorization", "login"}),
    frozenset({"deploy", "deployment", "release", "ship"}),
    frozenset({"config", "configuration", "settings", "preferences"}),
    frozenset({"database", "db", "datastore"}),
    frozenset({"api", "endpoint", "route"}),
    frozenset({"test", "testing", "spec", "unittest"}),
    frozenset({"perf", "performance", "speed", "latency"}),
    frozenset({"user", "account", "profile"}),
    frozenset({"log", "logging", "logger"}),
    frozenset({"cache", "caching", "memoize"}),
    frozenset({"retry", "retries", "backoff"}),
    frozenset({"queue", "job", "task", "worker"}),
    # English — financial
    frozenset({"profit", "earnings", "margin", "gain"}),
    frozenset({"loss", "deficit", "shortfall"}),
    frozenset({"asset", "holding", "portfolio"}),
    frozenset({"liability", "debt", "obligation"}),
    # Vietnamese — general
    frozenset({"chi phí", "phí", "giá", "chi phi"}),
    frozenset({"doanh thu", "thu nhập", "doanh_thu", "thu_nhap"}),
    frozenset({"lỗi", "sự cố", "su co", "loi"}),
    frozenset({"triển khai", "trien khai"}),
    frozenset({"cấu hình", "cau hinh"}),
    frozenset({"người dùng", "tài khoản", "nguoi dung", "tai khoan"}),
    frozenset({"lợi nhuận", "lãi", "loi nhuan"}),
    frozenset({"thua lỗ", "lỗ", "thua lo"}),
)

# Build reverse lookup: word → frozenset (computed once at import)
SYNONYM_MAP: dict[str, frozenset[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        SYNONYM_MAP[_word.lower()] = _group


# ── Abbreviation map ─────────────────────────────────────────────
# Abbreviation → full form.  Reverse lookup is also supported.

ABBREVIATION_MAP: dict[str, str] = {
    "roe": "return on equity",
    "roa": "return on assets",
    "eps": "earnings per share",
    "pe": "price to earnings",
    "api": "application programming interface",
    "db": "database",
    "ui": "user interface",
    "ux": "user experience",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "pr": "pull request",
    "mr": "merge request",
    "jwt": "json web token",
    "sql": "structured query language",
    "css": "cascading style sheets",
    "html": "hypertext markup language",
    "sdk": "software development kit",
    "cli": "command line interface",
    "orm": "object relational mapping",
    "crud": "create read update delete",
    "ebitda": "earnings before interest taxes depreciation amortization",
    "npm": "node package manager",
    "mcp": "model context protocol",
}

# Reverse abbreviation lookup: full form → abbreviation
_REVERSE_ABBREVIATION: dict[str, str] = {_full: _abbr for _abbr, _full in ABBREVIATION_MAP.items()}


# ── Cross-language map ───────────────────────────────────────────
# Bidirectional pairs: EN ↔ VI keyword hints for non-embedding recall.

_CROSS_LANG_PAIRS: tuple[tuple[str, str], ...] = (
    ("cost", "chi phí"),
    ("error", "lỗi"),
    ("deploy", "triển khai"),
    ("revenue", "doanh thu"),
    ("decision", "quyết định"),
    ("pattern", "mẫu"),
    ("workflow", "quy trình"),
    ("profit", "lợi nhuận"),
    ("loss", "thua lỗ"),
    ("user", "người dùng"),
    ("config", "cấu hình"),
    ("test", "kiểm thử"),
    ("database", "cơ sở dữ liệu"),
    ("account", "tài khoản"),
    ("memory", "bộ nhớ"),
)

# Build bidirectional lookup: word → list of cross-language equivalents
CROSS_LANG_MAP: dict[str, list[str]] = {}
for _en, _vi in _CROSS_LANG_PAIRS:
    CROSS_LANG_MAP.setdefault(_en.lower(), []).append(_vi.lower())
    CROSS_LANG_MAP.setdefault(_vi.lower(), []).append(_en.lower())


# ── Public API ───────────────────────────────────────────────────


def expand_terms(
    keywords: list[str],
    *,
    enable_synonyms: bool = True,
    enable_abbreviations: bool = True,
    enable_cross_language: bool = True,
    max_per_term: int = 5,
    custom_synonyms: dict[str, list[str]] | None = None,
) -> list[str]:
    """Expand keywords with synonyms, abbreviations, and cross-language hints.

    Returns a flat deduplicated list of expanded keywords (original + expansions).
    Never mutates the input list.

    Args:
        keywords: Original keywords to expand.
        enable_synonyms: Enable synonym lookup.
        enable_abbreviations: Enable abbreviation expansion.
        enable_cross_language: Enable cross-language hints.
        max_per_term: Max expansions per original term.
        custom_synonyms: Optional user-provided synonym groups.

    Returns:
        Flat list of unique keywords (originals + expansions), lowercased.
    """
    result: list[str] = []
    seen: set[str] = set()

    # Build custom synonym lookup if provided
    custom_map: dict[str, list[str]] = {}
    if custom_synonyms:
        for key, synonyms in custom_synonyms.items():
            all_terms = [key.lower()] + [s.lower() for s in synonyms]
            for term in all_terms:
                custom_map[term] = [t for t in all_terms if t != term]

    for kw in keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower or kw_lower in seen:
            continue

        # Always include the original
        result.append(kw_lower)
        seen.add(kw_lower)

        expansions: list[str] = []

        # Vietnamese compound: space ↔ underscore
        if " " in kw_lower:
            underscore_variant = kw_lower.replace(" ", "_")
            expansions.append(underscore_variant)
        elif "_" in kw_lower:
            space_variant = kw_lower.replace("_", " ")
            expansions.append(space_variant)

        # Synonym expansion
        if enable_synonyms:
            group = SYNONYM_MAP.get(kw_lower)
            if group:
                for synonym in group:
                    if synonym.lower() != kw_lower:
                        expansions.append(synonym.lower())

            # Custom synonyms
            if kw_lower in custom_map:
                expansions.extend(custom_map[kw_lower])

        # Abbreviation expansion
        if enable_abbreviations:
            # Abbreviation → full form
            full = ABBREVIATION_MAP.get(kw_lower)
            if full:
                expansions.append(full)

            # Full form → abbreviation (reverse)
            abbr = _REVERSE_ABBREVIATION.get(kw_lower)
            if abbr:
                expansions.append(abbr)

        # Cross-language expansion
        if enable_cross_language:
            cross = CROSS_LANG_MAP.get(kw_lower)
            if cross:
                expansions.extend(cross)

        # Deduplicate and cap
        added = 0
        for exp in expansions:
            exp_lower = exp.lower().strip()
            if exp_lower and exp_lower not in seen and added < max_per_term:
                result.append(exp_lower)
                seen.add(exp_lower)
                added += 1

    return result
