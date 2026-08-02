"""Vite manifest resolution.

These exist because a resolver that read only the entry's own ``css`` key
shipped a completely unstyled site while every other test passed: the HTML was
correct, the routes were correct, and the stylesheet simply was not linked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from dentist_ai.web import templating
from dentist_ai.web.templating import THEME_INIT_CSP_HASH, THEME_INIT_SCRIPT, ViteAssets

MANIFEST = {
    "_shared.hash.js": {
        "file": "assets/shared.hash.js",
        "css": ["assets/tokens.hash.css"],
    },
    "_deep.hash.js": {
        "file": "assets/deep.hash.js",
        "imports": ["_shared.hash.js"],
        "css": ["assets/deep.hash.css"],
    },
    "src/entries/page.ts": {
        "file": "assets/page.hash.js",
        "isEntry": True,
        "imports": ["_deep.hash.js"],
        "css": ["assets/page.hash.css"],
    },
    "src/entries/bare.ts": {"file": "assets/bare.hash.js", "isEntry": True},
}


@pytest.fixture
def assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ViteAssets:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    monkeypatch.setattr(templating, "MANIFEST_PATH", manifest_path)
    return ViteAssets(cache=False)


def test_collects_css_from_transitive_imports(assets: ViteAssets) -> None:
    """Shared-chunk CSS is only reachable through the import graph."""
    html = str(assets.script("src/entries/page.ts"))

    assert "assets/tokens.hash.css" in html, "design tokens live in a shared chunk"
    assert "assets/deep.hash.css" in html
    assert "assets/page.hash.css" in html


def test_stylesheets_are_ordered_dependency_first(assets: ViteAssets) -> None:
    """Cascade order must match what the bundler assumed."""
    html = str(assets.script("src/entries/page.ts"))

    assert html.index("tokens.hash.css") < html.index("deep.hash.css")
    assert html.index("deep.hash.css") < html.index("page.hash.css")


def test_script_tag_points_at_the_entry(assets: ViteAssets) -> None:
    html = str(assets.script("src/entries/page.ts"))
    assert '<script type="module" src="/static/dist/assets/page.hash.js" defer>' in html


def test_preload_includes_shared_chunks(assets: ViteAssets) -> None:
    html = str(assets.preload("src/entries/page.ts"))
    assert "assets/shared.hash.js" in html
    assert "assets/page.hash.js" in html


def test_entry_without_imports_still_resolves(assets: ViteAssets) -> None:
    html = str(assets.script("src/entries/bare.ts"))
    assert "assets/bare.hash.js" in html


def test_unknown_entry_renders_nothing(assets: ViteAssets) -> None:
    assert str(assets.script("src/entries/missing.ts")) == ""


def test_missing_manifest_degrades_quietly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(templating, "MANIFEST_PATH", tmp_path / "absent.json")
    assert str(ViteAssets(cache=False).script("src/entries/page.ts")) == ""


def test_theme_script_hash_is_derived_from_the_source() -> None:
    import base64
    import hashlib

    digest = hashlib.sha256(THEME_INIT_SCRIPT.encode("utf-8")).digest()
    assert f"'sha256-{base64.b64encode(digest).decode('ascii')}'" == THEME_INIT_CSP_HASH


async def test_rendered_pages_link_a_stylesheet(client: AsyncClient) -> None:
    """End-to-end guard: every entry point must ship its styles.

    Asserted against the real build output, so a change to the bundler's
    chunking strategy cannot quietly unstyle the site.
    """
    for path in ("/", "/login", "/pricing"):
        body = (await client.get(path)).text
        assert 'rel="stylesheet"' in body, f"{path} rendered without any stylesheet"
