"""Regression guard: surrealdb SDK stubs must not shadow an installed SDK.

Several unit-test modules stub ``sys.modules["surrealdb"]`` with a MagicMock
so ``store.py`` imports without the optional SDK. Before the try-import guard
(``except ImportError``) they keyed on ``"surrealdb" not in sys.modules``,
which conflates *not yet imported* with *not installed*: in a full-suite run
the stub was installed at collection time and every live (SURREALDB_URL) test
ERRORed at fixture setup with ``TypeError: object MagicMock can't be used in
'await' expression`` — while the same file passed solo.

Pytest imports every test module during collection before any test runs, so
by the time this test executes, each stub-installing module has already had
its chance to pollute ``sys.modules``. If the SDK is installed, whatever sits
there must be the real package (or nothing).
"""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest


def test_stub_modules_do_not_shadow_installed_sdk() -> None:
    try:
        importlib.metadata.distribution("surrealdb")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("surrealdb SDK not installed — stubs are legitimate here")

    mod = sys.modules.get("surrealdb")
    if mod is None:
        return  # nobody imported or stubbed it — fine
    assert isinstance(mod, types.ModuleType) and mod.__spec__ is not None, (
        "sys.modules['surrealdb'] is a test stub, but the real SDK is installed. "
        "Some test module stubbed the SDK unconditionally (use the "
        "try/except ImportError guard) — this breaks every live SURREALDB_URL "
        "test that runs later in the session."
    )
