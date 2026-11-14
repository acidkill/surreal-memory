"""Check the production image offline; assertions are its acceptance checks."""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
from html.parser import HTMLParser
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class AssetLinks(HTMLParser):
    """Collect the dashboard entry point's local JavaScript and CSS bundles."""

    def __init__(self) -> None:
        super().__init__()
        self.paths: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if tag in {"script", "link"} and key in {"src", "href"} and value:
                if value.startswith("/assets/"):
                    self.paths.add(value)


async def check_similarity_backend() -> None:
    """Observe the actual durable selector, using only two synthetic vectors."""
    from surreal_memory.core.brain import BrainConfig
    from surreal_memory.core.neuron import Neuron, NeuronType
    from surreal_memory.engine.semantic_discovery import discover_semantic_synapses

    neurons = [
        Neuron.create(
            NeuronType.CONCEPT,
            f"image smoke {index}",
            neuron_id=f"smoke-{index}",
            metadata={"_embedding": [1.0, index * 0.01]},
        )
        for index in range(2)
    ]
    storage = SimpleNamespace(
        current_brain_id="image-smoke",
        find_neurons_after_id=AsyncMock(
            side_effect=lambda cursor, **_options: neurons if cursor is None else []
        ),
        get_synapses_after_id=AsyncMock(return_value=[]),
        get_synapse_degrees=AsyncMock(return_value={}),
        find_existing_synapse_pairs=AsyncMock(return_value=set()),
        capture_semantic_source_token=AsyncMock(return_value="image-smoke-source"),
        assert_semantic_source_unchanged=AsyncMock(),
        save_semantic_discovery_state=AsyncMock(),
    )
    with patch(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        return_value=(True, "openai", "image-smoke"),
    ):
        result = await discover_semantic_synapses(
            storage, BrainConfig(embedding_enabled=True), checkpoint=AsyncMock()
        )
    assert result.synapses_created == 1
    backend = storage.save_semantic_discovery_state.call_args.args[2]["legacy_checkpoint"][
        "similarity_backend"
    ]
    assert version("numpy") == "2.5.4", version("numpy")
    assert backend == "numpy_float32:2.5.4", backend
    from surreal_memory import __version__

    assert version("surreal-memory") == __version__
    print(f"Installed package {__version__}; selected similarity backend {backend}")


def main() -> None:
    """Exercise installed extras, the CLI and dashboard without a live backend."""
    assert os.getuid() != 0, "production image must run as a non-root user"
    for module in ("surrealdb", "openai", "numpy", "uvicorn"):
        importlib.import_module(module)
    asyncio.run(check_similarity_backend())
    smem = shutil.which("smem")
    assert smem, "installed CLI is missing"
    subprocess.run([smem, "consolidate", "--help"], check=True, capture_output=True)

    from fastapi.testclient import TestClient

    from surreal_memory.server.app import app

    # No context manager: deliberately skip lifespan, which initializes the database.
    client = TestClient(app)
    response = client.get("/ui")
    assert response.status_code == 200, response.text
    assert "text/html" in response.headers["content-type"]
    parser = AssetLinks()
    parser.feed(response.text)
    assert parser.paths, "dashboard HTML has no built assets"
    for path in sorted(parser.paths):
        asset = client.get(path)
        assert asset.status_code == 200, path
        assert asset.content, path
        assert "text/html" not in asset.headers["content-type"], path
    print(f"Production image smoke passed: extras, consolidate CLI, UI, {len(parser.paths)} assets")


if __name__ == "__main__":
    main()
