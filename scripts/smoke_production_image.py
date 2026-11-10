"""Check the production image offline; assertions are its acceptance checks."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from html.parser import HTMLParser


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


def main() -> None:
    """Exercise installed extras, the CLI and dashboard without a live backend."""
    assert os.getuid() != 0, "production image must run as a non-root user"
    for module in ("surrealdb", "openai", "numpy", "uvicorn"):
        importlib.import_module(module)
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
