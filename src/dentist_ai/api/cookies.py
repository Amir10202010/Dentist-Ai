"""Session and CSRF cookie handling.

Flags are derived from the environment: ``Secure`` is on wherever it matters
and off on ``http://localhost``, where it would silently break sign-in.
"""

from __future__ import annotations

from fastapi import Request, Response

from dentist_ai.core.config import Settings
from dentist_ai.core.security import SessionService, anonymous_session_id


def csrf_session_id(request: Request, sessions: SessionService, settings: Settings) -> str:
    """The identity a CSRF token is bound to, derived from the request alone.

    Minting and verification have to agree on this, so both go through here.
    It depends only on the session *cookie*, never on whether that session
    still resolves to a live user: gating on the user would make a page mint an
    anonymous token while ``CSRFMiddleware`` verified against the
    cookie's real session id. Every POST would then fail as "session expired",
    and refreshing — the one thing the message tells you to do — cannot help,
    because the stale cookie comes back with each request.
    """
    session_data = sessions.load(request.cookies.get(settings.security.session_cookie_name))
    if session_data is not None:
        return session_data.session_id
    return anonymous_session_id(request.headers.get("host", ""))


def set_session_cookies(
    response: Response,
    user_id: int,
    sessions: SessionService,
    settings: Settings,
) -> str:
    """Issue a fresh session + CSRF pair. Returns the CSRF token.

    Always mints a **new** session id rather than reusing whatever the client
    presented — that is what closes session fixation, where an attacker plants
    a known id before the victim authenticates.
    """
    security = settings.security
    cookie_value, session_data = sessions.issue(user_id)
    csrf_token = sessions.issue_csrf(session_data.session_id)

    response.set_cookie(
        security.session_cookie_name,
        cookie_value,
        max_age=security.session_max_age_seconds,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        path="/",
    )
    # Readable by JavaScript: the frontend echoes it back in the
    # CSRF header. It is not a credential — possession alone proves nothing
    # without the session cookie it is bound to.
    response.set_cookie(
        security.csrf_cookie_name,
        csrf_token,
        max_age=security.session_max_age_seconds,
        httponly=False,
        secure=settings.is_production,
        samesite="lax",
        path="/",
    )
    return csrf_token


def clear_session_cookies(response: Response, settings: Settings) -> None:
    for name in (settings.security.session_cookie_name, settings.security.csrf_cookie_name):
        response.delete_cookie(
            name,
            path="/",
            httponly=name == settings.security.session_cookie_name,
            secure=settings.is_production,
            samesite="lax",
        )
