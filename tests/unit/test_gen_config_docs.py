"""The generated configuration reference must not depend on the machine.

Defaults are computed by calling the dataclass factories, and some of them
derive from the home directory. Rendering those verbatim would publish one
developer's absolute paths and make `--check` fail on every other machine,
including CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gen_config_docs.py"


@pytest.fixture(scope="module")
def gen_config_docs():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("gen_config_docs", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_config_docs"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("gen_config_docs", None)


def test_home_directory_never_reaches_the_page(gen_config_docs) -> None:  # type: ignore[no-untyped-def]
    content = gen_config_docs.generate()

    assert Path.home().as_posix() not in content


def test_paths_under_home_render_with_a_tilde(gen_config_docs) -> None:  # type: ignore[no-untyped-def]
    rendered = gen_config_docs._portable(Path.home() / ".surrealmemory" / "brains")

    assert rendered == "~/.surrealmemory/brains"


def test_paths_outside_home_are_left_alone(gen_config_docs) -> None:  # type: ignore[no-untyped-def]
    assert gen_config_docs._portable(Path("/etc/surrealmemory")) == "/etc/surrealmemory"


def test_pipes_are_escaped_so_tables_survive(gen_config_docs) -> None:  # type: ignore[no-untyped-def]
    # `int | None` in a type annotation would otherwise split the table cell.
    assert gen_config_docs._cell("int | None") == r"int \| None"


def test_every_config_section_is_documented(gen_config_docs) -> None:  # type: ignore[no-untyped-def]
    import dataclasses

    from surreal_memory.unified_config import UnifiedConfig

    content = gen_config_docs.generate()
    expected = {
        f.name
        for f in dataclasses.fields(UnifiedConfig)
        if gen_config_docs._is_section(f) is not None
    }

    missing = {name for name in expected if f"## `[{name}]`" not in content}

    assert not missing, f"config sections absent from the reference: {sorted(missing)}"
