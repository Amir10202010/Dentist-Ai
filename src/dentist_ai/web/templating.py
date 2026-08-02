"""Jinja environment and Vite asset resolution.

Vite content-hashes every asset. Reading its manifest lets templates emit
``/static/dist/app.a1b2c3.js`` with a one-year immutable cache header: the
filename is the cache key.
"""

from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Final, TypedDict

from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from dentist_ai.core.config import Settings
from dentist_ai.core.logging import get_logger

log = get_logger(__name__)

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
TEMPLATES_DIR: Final[Path] = PACKAGE_ROOT / "templates"
STATIC_DIR: Final[Path] = PACKAGE_ROOT / "static"
DIST_DIR: Final[Path] = STATIC_DIR / "dist"
MANIFEST_PATH: Final[Path] = DIST_DIR / ".vite" / "manifest.json"

#: Applies the stored theme before first paint, so a dark-mode user never sees
#: a white flash. It must run synchronously in <head>, which means it cannot
#: be a deferred module — hence an inline script, allow-listed in the CSP by
#: hash rather than by opening the policy up with 'unsafe-inline'.
THEME_INIT_SCRIPT: Final[str] = (
    "(function(){try{var t=localStorage.getItem('dentist-ai:theme');"
    "if(t==='dark'||t==='light'){document.documentElement.dataset.theme=t}}"
    "catch(e){}document.documentElement.classList.remove('no-js')})()"
)


def _csp_hash(script: str) -> str:
    digest = hashlib.sha256(script.encode("utf-8")).digest()
    return f"'sha256-{base64.b64encode(digest).decode('ascii')}'"


#: The exact CSP source expression that authorises the script above. Any edit
#: to the script changes this automatically — they cannot drift apart.
THEME_INIT_CSP_HASH: Final[str] = _csp_hash(THEME_INIT_SCRIPT)


class ManifestChunk(TypedDict, total=False):
    """The subset of Vite's manifest entry shape that we consume."""

    file: str
    css: list[str]
    imports: list[str]
    isEntry: bool


Manifest = dict[str, ManifestChunk]


class ViteAssets:
    """Resolves entry names to hashed URLs, with a dev-friendly fallback."""

    def __init__(self, *, cache: bool) -> None:
        self._cache = cache
        self._manifest: Manifest | None = None

    def _load(self) -> Manifest:
        if self._manifest is not None and self._cache:
            return self._manifest
        if not MANIFEST_PATH.is_file():
            # Missing manifest means the frontend has not been built. Warn
            # loudly but keep serving: the HTML is still readable without it.
            log.warning("vite_manifest_missing", path=str(MANIFEST_PATH), hint="run `make build`")
            self._manifest = {}
            return self._manifest
        with MANIFEST_PATH.open(encoding="utf-8") as handle:
            loaded: Manifest = json.load(handle)
        self._manifest = loaded
        return loaded

    def _walk(self, entry: str) -> tuple[list[str], list[str]]:
        """Collect stylesheets and JS chunks for ``entry`` and its imports.

        Vite lists only a chunk's *own* CSS under its manifest key. Styles that
        Rollup hoisted into a shared chunk — which is where our design tokens
        end up, since every entry imports them — are reachable only by walking
        ``imports`` transitively. Reading the entry alone silently ships a page
        with no stylesheet at all.
        """
        manifest = self._load()
        stylesheets: list[str] = []
        scripts: list[str] = []
        seen: set[str] = set()

        def visit(key: str) -> None:
            if key in seen:
                return
            seen.add(key)
            chunk = manifest.get(key)
            if chunk is None:
                return
            # Depth-first so a dependency's CSS is ordered before the importer's,
            # matching the cascade the bundler assumed.
            for imported in chunk.get("imports", []):
                visit(imported)
            for css in chunk.get("css", []):
                if css not in stylesheets:
                    stylesheets.append(css)
            file = chunk.get("file", "")
            if file and not file.endswith(".css") and file not in scripts:
                scripts.append(file)

        visit(entry)
        return stylesheets, scripts

    def script(self, entry: str) -> Markup:
        stylesheets, scripts = self._walk(entry)
        if not scripts and not stylesheets:
            return Markup("")

        tags = [f'<link rel="stylesheet" href="/static/dist/{css}">' for css in stylesheets]
        # The entry is the last script collected; shared chunks it imports are
        # fetched by the module loader itself.
        chunk = self._load().get(entry)
        if chunk is not None:
            tags.append(
                f'<script type="module" src="/static/dist/{chunk.get("file", "")}" defer></script>'
            )
        # All values originate from our own build manifest, never user input.
        return Markup("\n".join(tags))  # noqa: S704

    def preload(self, entry: str) -> Markup:
        """Preload the entry and its static imports.

        Without this the browser discovers shared chunks only after parsing the
        entry, serialising two round-trips that could have been one.
        """
        _, scripts = self._walk(entry)
        return Markup(  # noqa: S704
            "\n".join(f'<link rel="modulepreload" href="/static/dist/{file}">' for file in scripts)
        )


@lru_cache(maxsize=1)
def _assets_for(cache: bool) -> ViteAssets:
    return ViteAssets(cache=cache)


def build_templates(settings: Settings) -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    env = templates.env
    env.auto_reload = settings.debug
    env.trim_blocks = True
    env.lstrip_blocks = True

    assets = _assets_for(not settings.debug)
    env.globals["vite_script"] = assets.script
    env.globals["vite_preload"] = assets.preload
    env.globals["settings"] = settings
    env.globals["theme_init_script"] = Markup(THEME_INIT_SCRIPT)  # noqa: S704 - constant
    env.filters["confidence"] = _format_confidence
    return templates


def _format_confidence(value: float) -> str:
    return f"{value * 100:.0f}%"
