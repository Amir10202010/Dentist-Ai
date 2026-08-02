"""Authentication, session and CSRF behaviour."""

from __future__ import annotations

import re

import pytest
from httpx import ASGITransport, AsyncClient

from dentist_ai.core.config import Settings
from dentist_ai.core.security import SessionService
from dentist_ai.main import create_app
from tests.conftest import REGISTRATION, bootstrap_csrf, register_and_login


async def test_register_creates_organization_and_signs_in(client: AsyncClient) -> None:
    token = await bootstrap_csrf(client)
    response = await client.post(
        "/api/v1/auth/register", json=REGISTRATION, headers={"X-CSRF-Token": token}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["user"]["email"] == "aigul@clinic.kz"
    assert body["user"]["role"] == "owner"
    assert body["user"]["organization"]["name"] == "Клиника Смайл"
    # Cyrillic clinic names must still yield a usable ASCII slug.
    assert body["user"]["organization"]["slug"] == "klinika-smail"
    assert client.cookies.get("dentist_ai_session")


async def test_register_rejects_duplicate_email(client: AsyncClient) -> None:
    token = await register_and_login(client)
    response = await client.post(
        "/api/v1/auth/register", json=REGISTRATION, headers={"X-CSRF-Token": token}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "email_taken"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("password", "short"),
        ("password", "password123"),
        ("email", "not-an-email"),
        ("fullName", "x"),
    ],
)
async def test_register_validation(client: AsyncClient, field: str, value: str) -> None:
    token = await bootstrap_csrf(client)
    payload = {**REGISTRATION, field: value}
    if field == "password":
        payload["passwordConfirm"] = value

    response = await client.post(
        "/api/v1/auth/register", json=payload, headers={"X-CSRF-Token": token}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_failed"


async def test_register_rejects_mismatched_confirmation(client: AsyncClient) -> None:
    token = await bootstrap_csrf(client)
    response = await client.post(
        "/api/v1/auth/register",
        json={**REGISTRATION, "passwordConfirm": "something-else-entirely"},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 422


async def test_login_succeeds_and_rotates_session(client: AsyncClient) -> None:
    await register_and_login(client)
    before = client.cookies.get("dentist_ai_session")

    token = await bootstrap_csrf(client)
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": REGISTRATION["email"], "password": REGISTRATION["password"]},
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 200
    # Session fixation defence: a fresh id on every authentication.
    assert client.cookies.get("dentist_ai_session") != before


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("aigul@clinic.kz", "wrong-password-entirely"),
        ("nobody@clinic.kz", "correct-horse-battery"),
    ],
)
async def test_login_failure_is_indistinguishable(
    client: AsyncClient, email: str, password: str
) -> None:
    """Wrong password and unknown account must return the identical error.

    Any difference here is a user-enumeration oracle.
    """
    await register_and_login(client)
    token = await bootstrap_csrf(client)

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "invalid_credentials"
    assert response.json()["title"] == "Неверный email или пароль."


async def test_password_is_never_stored_in_plaintext(
    client: AsyncClient, settings: Settings
) -> None:
    import sqlite3
    from contextlib import closing

    await register_and_login(client)
    path = settings.database.url.removeprefix("sqlite+aiosqlite:///")
    # `with sqlite3.connect(...)` manages the *transaction*, not the
    # connection — closing() is what actually releases the handle.
    with closing(sqlite3.connect(path)) as connection:
        stored = connection.execute("SELECT password_hash FROM users").fetchone()[0]

    assert REGISTRATION["password"] not in stored
    assert stored.startswith("$argon2id$")


async def test_me_requires_authentication(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


async def test_logout_clears_session(authed_client: AsyncClient) -> None:
    assert (await authed_client.get("/api/v1/auth/me")).status_code == 200

    logout = await authed_client.post("/api/v1/auth/logout")
    assert logout.status_code == 200

    assert (await authed_client.get("/api/v1/auth/me")).status_code == 401


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------
async def test_write_without_csrf_token_is_rejected(authed_client: AsyncClient) -> None:
    del authed_client.headers["X-CSRF-Token"]
    response = await authed_client.post("/api/v1/patients", json={"fullName": "Тест"})
    assert response.status_code == 403
    assert response.json()["code"] == "csrf_failed"


async def test_csrf_cookie_alone_does_not_authorise(authed_client: AsyncClient) -> None:
    """The core double-submit invariant.

    A cross-site form post carries cookies automatically. If the CSRF cookie
    were accepted as proof on its own, the protection would be worthless.
    """
    del authed_client.headers["X-CSRF-Token"]
    assert authed_client.cookies.get("dentist_ai_csrf")

    response = await authed_client.post("/api/v1/patients", json={"fullName": "Тест"})
    assert response.status_code == 403


async def test_token_from_another_session_is_rejected(
    client: AsyncClient, settings: Settings
) -> None:
    await register_and_login(client)

    # A second, unrelated app instance mints a token bound to its own session.
    other_app = create_app(settings)
    async with (
        AsyncClient(
            transport=ASGITransport(app=other_app),
            base_url="http://testserver",
            headers={"Origin": "http://testserver"},
        ) as other,
        other_app.router.lifespan_context(other_app),
    ):
        foreign_token = (await other.get("/api/v1/auth/csrf")).json()["csrfToken"]

    response = await client.post(
        "/api/v1/patients",
        json={"fullName": "Тест"},
        headers={"X-CSRF-Token": foreign_token},
    )
    assert response.status_code == 403


async def test_cross_origin_request_is_rejected(authed_client: AsyncClient) -> None:
    response = await authed_client.post(
        "/api/v1/patients",
        json={"fullName": "Тест"},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "csrf_failed"


async def test_safe_methods_need_no_token(authed_client: AsyncClient) -> None:
    del authed_client.headers["X-CSRF-Token"]
    assert (await authed_client.get("/api/v1/patients")).status_code == 200


def _meta_csrf_token(html: str) -> str:
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    assert match is not None, "the page rendered no CSRF meta tag"
    return match.group(1)


async def test_stale_session_cookie_does_not_block_registration(
    client: AsyncClient, settings: Settings
) -> None:
    """A signed session whose user is gone must not wedge the auth forms.

    Resetting the database leaves real browsers holding a perfectly valid
    session cookie for a user that no longer exists. Minting the page token
    against the *user* while the middleware verified against the *cookie* put
    those visitors in a dead end: every submit answered "Сессия устарела", and
    the refresh it advises re-sends the same cookie, so nothing ever changed.
    """
    sessions = SessionService(settings)
    orphan_cookie, _ = sessions.issue(user_id=99_999)
    client.cookies.set("dentist_ai_session", orphan_cookie)

    page = await client.get("/register")
    assert page.status_code == 200

    response = await client.post(
        "/api/v1/auth/register",
        json=REGISTRATION,
        headers={"X-CSRF-Token": _meta_csrf_token(page.text)},
    )
    assert response.status_code == 201, response.text


async def test_stale_session_cookie_does_not_block_login(
    client: AsyncClient, settings: Settings
) -> None:
    """The same dead end reached from the sign-in form."""
    await register_and_login(client)
    client.cookies.clear()

    sessions = SessionService(settings)
    orphan_cookie, _ = sessions.issue(user_id=99_999)
    client.cookies.set("dentist_ai_session", orphan_cookie)

    page = await client.get("/login")
    assert page.status_code == 200

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": REGISTRATION["email"], "password": REGISTRATION["password"]},
        headers={"X-CSRF-Token": _meta_csrf_token(page.text)},
    )
    assert response.status_code == 200, response.text
