"""Server-rendered pages, security headers and SEO surface."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

MARKETING_PATHS = ["/", "/about", "/pricing", "/contact", "/privacy"]
APP_PATHS = ["/app", "/app/studies", "/app/patients", "/app/settings"]


@pytest.mark.parametrize("path", MARKETING_PATHS)
async def test_marketing_pages_render(client: AsyncClient, path: str) -> None:
    response = await client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")

    body = response.text
    assert "<title>" in body
    assert 'name="description"' in body
    # Server-rendered content, not an empty shell waiting on JavaScript.
    assert "Dentist-AI" in body
    assert "<h1" in body or "<h2" in body


@pytest.mark.parametrize("path", APP_PATHS)
async def test_app_pages_redirect_when_anonymous(client: AsyncClient, path: str) -> None:
    response = await client.get(path)
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/login?next=")


@pytest.mark.parametrize("path", APP_PATHS)
async def test_app_pages_render_when_authenticated(authed_client: AsyncClient, path: str) -> None:
    response = await authed_client.get(path)
    assert response.status_code == 200
    body = response.text
    assert 'name="csrf-token"' in body
    # Patient data must not be indexable.
    assert 'name="robots" content="noindex, nofollow"' in body
    assert "Клиника Смайл" in body


async def test_study_detail_page_renders(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    upload = await authed_client.post(
        "/api/v1/studies", files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")}
    )
    public_id = upload.json()["publicId"]

    response = await authed_client.get(f"/app/studies/{public_id}")
    assert response.status_code == 200
    assert f'data-study-id="{public_id}"' in response.text


async def test_login_page_redirects_when_signed_in(authed_client: AsyncClient) -> None:
    for path in ("/login", "/register"):
        response = await authed_client.get(path)
        assert response.status_code == 303
        assert response.headers["location"] == "/app"


async def test_security_headers_present(client: AsyncClient) -> None:
    headers = (await client.get("/")).headers

    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in headers["Permissions-Policy"]

    csp = headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    # The inline theme script is allowed by hash, never by 'unsafe-inline'.
    assert "'unsafe-inline'" not in csp.split("style-src")[0]
    assert "sha256-" in csp


async def test_inline_script_hash_matches_rendered_markup(client: AsyncClient) -> None:
    """The CSP hash must authorise the exact bytes we render.

    If these drift, every browser silently blocks the theme script and dark
    mode flashes on load — a failure no server-side test would otherwise catch.
    """
    import base64
    import hashlib
    import re

    response = await client.get("/")
    csp = response.headers["Content-Security-Policy"]

    match = re.search(r"<script>(.*?)</script>", response.text, re.DOTALL)
    assert match is not None
    rendered = match.group(1)

    digest = hashlib.sha256(rendered.encode("utf-8")).digest()
    expected = f"'sha256-{base64.b64encode(digest).decode('ascii')}'"
    assert expected in csp


async def test_robots_blocks_app_and_api(client: AsyncClient) -> None:
    response = await client.get("/robots.txt")
    assert response.status_code == 200
    # Non-production deployments must not be indexed at all.
    assert "Disallow: /" in response.text


async def test_sitemap_lists_public_pages(client: AsyncClient) -> None:
    response = await client.get("/sitemap.xml")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "<loc>http://testserver/pricing</loc>" in response.text
    # Authenticated areas must never appear in the sitemap.
    assert "/app" not in response.text


async def test_healthz_and_readyz(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


async def test_request_id_is_echoed(client: AsyncClient) -> None:
    response = await client.get("/", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["X-Request-ID"] == "trace-me-123"
