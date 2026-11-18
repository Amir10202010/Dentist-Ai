"""Vercel entry point: a read-mostly public demo of Dentist-AI.

Vercel functions have a read-only filesystem except ``/tmp``, no persistent
disk and no long-running process. So the demo ships a pre-seeded SQLite
database and its scan storage in ``vercel/seed/``, copies them into ``/tmp``
on a cold start, and runs with the stub detector. Anything a visitor changes
lives only as long as that function instance; the next cold start is a clean
demo clinic again. The production setup is still ``render.yaml`` / Docker.
"""

from __future__ import annotations

import mimetypes
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "vercel" / "seed"
RUNTIME = Path("/tmp/dentist-ai")

sys.path.insert(0, str(ROOT / "src"))

# The function image has no /etc/mime.types, and with `nosniff` set a
# stylesheet served as text/html is refused by the browser.
for _type, _ext in (
    ("text/css", ".css"),
    ("text/javascript", ".js"),
    ("application/json", ".json"),
    ("image/svg+xml", ".svg"),
    ("font/woff2", ".woff2"),
    ("image/webp", ".webp"),
    ("application/manifest+json", ".webmanifest"),
):
    mimetypes.add_type(_type, _ext)

if not (RUNTIME / "dentist_ai.sqlite3").exists():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SEED / "dentist_ai.sqlite3", RUNTIME / "dentist_ai.sqlite3")
    shutil.copytree(SEED / "storage", RUNTIME / "storage", dirs_exist_ok=True)

os.environ.setdefault("DENTIST_AI__ENVIRONMENT", "production")
os.environ.setdefault("DENTIST_AI__DATABASE__URL", f"sqlite+aiosqlite:///{RUNTIME / 'dentist_ai.sqlite3'}")
os.environ.setdefault("DENTIST_AI__STORAGE__ROOT", str(RUNTIME / "storage"))
os.environ.setdefault("DENTIST_AI__ML__BACKEND", "stub")
os.environ.setdefault("DENTIST_AI__ML__WARM_UP_ON_STARTUP", "false")
os.environ.setdefault("DENTIST_AI__LOG_FORMAT", "json")
os.environ.setdefault("DENTIST_AI__SECURITY__TRUSTED_HOSTS", '["*"]')

from dentist_ai.main import app as _app  # noqa: E402

_PREFIX = "/api/index"


async def app(scope, receive, send):  # type: ignore[no-untyped-def]
    """Undo Vercel's rewrite so the app sees the path the browser asked for."""
    if scope["type"] in ("http", "websocket"):
        scope = dict(scope)
        path = scope.get("path", "/")
        if path.startswith(_PREFIX):
            path = path[len(_PREFIX):] or "/"
        scope["path"] = path
        scope["raw_path"] = path.encode()
        scope["root_path"] = ""
        # Vercel compresses at the edge. A response the app already gzipped
        # comes back out of Vercel's bridge with its content-type reset to
        # text/html, which `nosniff` then makes browsers reject.
        scope["headers"] = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"accept-encoding"]
    await _app(scope, receive, send)


__all__ = ["app"]
