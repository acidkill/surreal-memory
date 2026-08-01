#!/usr/bin/env python3
"""Generate the configuration reference from unified_config.py.

Usage:
    python scripts/gen_config_docs.py              # write to docs/reference/config.md
    python scripts/gen_config_docs.py --check      # exit 1 if the page is stale
    python scripts/gen_config_docs.py --stdout     # print instead of writing

Field descriptions come from the trailing ``#`` comment on the declaration, so
documenting a setting means commenting it where it is defined.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import io
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from surreal_memory import unified_config as uc  # noqa: E402

OUTPUT = ROOT / "docs" / "reference" / "config.md"

#: Read but owned by the operating system, not by surreal-memory.
_ENV_SKIP = frozenset({"APPDATA", "HOME", "PATH", "USER", "USERNAME", "USERPROFILE"})

_ENV_PATTERN = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["']([A-Z][A-Z0-9_]*)["']"""
    r"""|os\.environ\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]"""
)


def _field_comments(cls: type) -> dict[str, str]:
    """Map field name -> trailing ``#`` comment on its declaration."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}

    comments: dict[str, str] = {}
    pending: str | None = None
    try:
        for tok_type, text, start, _end, line in tokenize.generate_tokens(
            io.StringIO(source).readline
        ):
            if tok_type == tokenize.NEWLINE:
                pending = None
            elif tok_type == tokenize.NAME and pending is None:
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                # A field declaration looks like `name: type = default`
                if stripped.startswith(f"{text}:") and start[1] == indent:
                    pending = text
            elif tok_type == tokenize.COMMENT and pending:
                comments[pending] = text.lstrip("#").strip()
                pending = None
    except tokenize.TokenError:
        pass
    return comments


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell."""
    return text.replace("|", r"\|")


def _portable(value: Path) -> str:
    """Render a path without the machine it was generated on.

    Defaults derived from the home directory would otherwise bake one
    developer's absolute path into the published page — and make --check fail
    everywhere else, including CI.
    """
    home = Path.home()
    try:
        return f"~/{value.relative_to(home).as_posix()}"
    except ValueError:
        return value.as_posix()


def _render_default(field: dataclasses.Field[Any]) -> str:
    if field.default is not dataclasses.MISSING:
        value: Any = field.default
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            value = field.default_factory()  # type: ignore[misc]
        except Exception:
            return "—"
    else:
        return "—"

    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, Path):
        return f"`{_cell(_portable(value))}`"
    if isinstance(value, str):
        return f"`{_cell(value)}`" if value else '`""`'
    if isinstance(value, (list, tuple, set, frozenset)):
        return "`[]`" if not value else f"`{_cell(str(list(value)))}`"
    if isinstance(value, dict):
        return "`{}`" if not value else f"`{_cell(str(value))}`"
    return f"`{_cell(str(value))}`"


def _render_type(field: dataclasses.Field[Any]) -> str:
    raw = (
        field.type
        if isinstance(field.type, str)
        else getattr(field.type, "__name__", str(field.type))
    )
    return f"`{_cell(str(raw))}`"


def _section_doc(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.split("\n\n", 1)[0].replace("\n", " ").strip()


def _is_section(field: dataclasses.Field[Any]) -> type | None:
    if field.default_factory is dataclasses.MISSING:  # type: ignore[misc]
        return None
    try:
        value = field.default_factory()  # type: ignore[misc]
    except Exception:
        return None
    return type(value) if dataclasses.is_dataclass(value) else None


def _sections() -> list[tuple[str, type]]:
    out: list[tuple[str, type]] = []
    for field in dataclasses.fields(uc.UnifiedConfig):
        cls = _is_section(field)
        if cls is not None:
            out.append((field.name, cls))
    return out


def _field_table(cls: type, skip_sections: bool = False) -> list[str]:
    comments = _field_comments(cls)
    lines = ["| Setting | Type | Default | Description |", "|---|---|---|---|"]
    for field in dataclasses.fields(cls):
        if field.name.startswith("_"):
            continue
        if skip_sections and _is_section(field) is not None:
            continue
        lines.append(
            f"| `{field.name}` | {_render_type(field)} | "
            f"{_render_default(field)} | {comments.get(field.name, '')} |"
        )
    return lines


def _env_vars() -> dict[str, list[str]]:
    """Map env var name -> source files that read it."""
    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in _ENV_PATTERN.finditer(text):
            name = match.group(1) or match.group(2)
            if not name or name in _ENV_SKIP:
                continue
            found.setdefault(name, set()).add(path.relative_to(ROOT).as_posix())
    return {name: sorted(paths) for name, paths in sorted(found.items())}


def generate() -> str:
    sections = _sections()
    env = _env_vars()

    lines: list[str] = [
        "# Configuration Reference",
        "",
        "Every setting in `~/.surrealmemory/config.toml`, generated from the dataclasses",
        "in `unified_config.py`. Unknown keys are ignored on load, so a file written by",
        "an older version keeps working.",
        "",
        "Run `smem init` to write a file with the current defaults.",
        "",
        "## Top level",
        "",
        "Keys that sit outside any `[section]`.",
        "",
        *_field_table(uc.UnifiedConfig, skip_sections=True),
        "",
    ]

    for name, cls in sections:
        lines.append(f"## `[{name}]`")
        lines.append("")
        doc = _section_doc(cls)
        if doc:
            lines.extend([doc, ""])
        lines.extend(_field_table(cls))
        lines.append("")

    lines.extend(
        [
            "## Environment variables",
            "",
            "Read straight from the environment. Where the same setting exists in both",
            "places, the environment wins.",
            "",
            "| Variable | Read in |",
            "|---|---|",
        ]
    )
    for name, paths in env.items():
        shown = ", ".join(f"`{p}`" for p in paths[:3])
        if len(paths) > 3:
            shown += f", +{len(paths) - 3} more"
        lines.append(f"| `{name}` | {shown} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "*Auto-generated by `scripts/gen_config_docs.py` from `unified_config.py` — "
            f"{len(sections)} sections, {len(env)} environment variables.*",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the configuration reference")
    parser.add_argument("--check", action="store_true", help="Check if the page is up-to-date")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout")
    args = parser.parse_args()

    content = generate()
    rel = OUTPUT.relative_to(ROOT)

    if args.stdout:
        print(content)
        return

    if args.check:
        if not OUTPUT.exists():
            print(f"  {rel} does not exist — generate with: python scripts/gen_config_docs.py")
            sys.exit(1)
        if OUTPUT.read_text(encoding="utf-8") != content:
            print(f"  {rel} is STALE — regenerate with: python scripts/gen_config_docs.py")
            sys.exit(1)
        print(f"  {rel} is up-to-date ({len(content)} chars)")
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Generated {rel} ({len(content)} chars, {len(_sections())} sections)")


if __name__ == "__main__":
    main()
