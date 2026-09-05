"""The compression pass logs a failed brain pre-fetch instead of degrading in silence.

``CompressionEngine.run`` fetches the brain once per pass because the derived-field
refresh needs its embedding config. The fetch is deliberately fail-soft — the refresh
helper looks the brain up per fiber when it is missing — but a silent fallback hides
both the loss of the optimisation and the real cause of any downstream warning.
"""

from typing import Any

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.engine.compression import CompressionEngine
from surreal_memory.storage.memory_store import InMemoryStorage

_PREFETCH_WARNING = "Brain pre-fetch failed"


async def _storage_with_brain() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="prefetch-brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


@pytest.mark.asyncio
async def test_failed_brain_prefetch_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    storage = await _storage_with_brain()

    async def _raise(_brain_id: str) -> Any:
        raise RuntimeError("storage unavailable")

    storage.get_brain = _raise  # type: ignore[method-assign]

    with caplog.at_level("WARNING", logger="surreal_memory.engine.compression"):
        report = await CompressionEngine(storage).run()

    assert report is not None, "the pass must still complete; the fallback is fail-soft"
    warnings = [r for r in caplog.records if _PREFETCH_WARNING in r.getMessage()]
    assert warnings, (
        f"expected a {_PREFETCH_WARNING!r} warning; got: {[r.getMessage() for r in caplog.records]}"
    )
    assert warnings[0].exc_info is not None, "the cause must travel with the warning"

    await storage.close()


@pytest.mark.asyncio
async def test_successful_brain_prefetch_stays_quiet(caplog: pytest.LogCaptureFixture) -> None:
    """Positive control: a pass that fetches its brain must not warn."""
    storage = await _storage_with_brain()

    with caplog.at_level("WARNING", logger="surreal_memory.engine.compression"):
        await CompressionEngine(storage).run()

    assert not [r for r in caplog.records if _PREFETCH_WARNING in r.getMessage()]

    await storage.close()
