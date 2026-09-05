"""Regression: `run_update_check_background()` respects a hermetic-mode env var.

The daemon thread that `smem`'s CLI kicks off on almost every invocation calls
`urlopen(PYPI_URL, timeout=3)` against pypi.org. `#110`/`#121` closed the
write side of that path (`_isolated_home_dir` in `tests/conftest.py`
redirects `$HOME` for the whole session); this closes the network side, so a
pytest run inside a sandboxed or offline environment doesn't sit through per-test
3-second `OSError` swallows or make silent live PyPI requests.

Contract: `SURREAL_MEMORY_NO_UPDATE_CHECK` follows the repo's canonical
env-var truthiness (`_env_truthy` in `unified_config.py`: `1/true/yes/on`,
case-insensitive, everything else false). A truthy value makes
`run_update_check_background()` a no-op *before* it starts the thread; any
other value — including `off`, `no`, `0`, `false`, arbitrary strings —
lets the check run. The env var is set to `"1"` session-wide in
`tests/conftest.py::_isolated_home_dir`, alongside the `$HOME` redirect —
same fixture, same rationale.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from surreal_memory.cli import update_check


class TestUpdateCheckEnvGate:
    """Guards the "no PyPI calls under pytest" contract."""

    def test_env_var_short_circuits_before_thread_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With SURREAL_MEMORY_NO_UPDATE_CHECK=1, no daemon thread is created."""
        monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", "1")
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_not_called()

    def test_env_var_missing_still_starts_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the env var, the daemon thread does start — positive control
        that the short-circuit is specifically the env var, not something else."""
        # `_isolated_home_dir` (session, autouse) sets this to "1" for the whole
        # test session; explicitly clear it for this one test to prove the
        # short-circuit is env-gated rather than a permanent no-op.
        monkeypatch.delenv("SURREAL_MEMORY_NO_UPDATE_CHECK", raising=False)
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_called_once()
        # `daemon=True` in the call so a leaked thread never blocks interpreter shutdown.
        _args, kwargs = thread_ctor.call_args
        assert kwargs.get("daemon") is True

    @pytest.mark.parametrize(
        "falsy", ["", "0", "false", "no", "FALSE", "No", "off", "OFF", "disabled", "anything"]
    )
    def test_falsy_values_do_not_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch, falsy: str
    ) -> None:
        """Everything except `1/true/yes/on` counts as "not set" — including
        `off`/`disabled`, which under an inverted convention would have
        silently disabled the update check."""
        monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", falsy)
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_called_once()

    @pytest.mark.parametrize("truthy", ["1", "yes", "true", "TRUE", "on", "YES", "On"])
    def test_truthy_values_short_circuit(
        self, monkeypatch: pytest.MonkeyPatch, truthy: str
    ) -> None:
        """Exactly the repo's canonical truthy set, case-insensitive — the
        same `_env_truthy` convention as `SURREAL_MEMORY_EMBEDDING_ENABLED`
        and friends."""
        monkeypatch.setenv("SURREAL_MEMORY_NO_UPDATE_CHECK", truthy)
        with patch("surreal_memory.cli.update_check.threading.Thread") as thread_ctor:
            update_check.run_update_check_background()
        thread_ctor.assert_not_called()
