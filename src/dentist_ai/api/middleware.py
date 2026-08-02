"""HTTP middleware: request context, security headers, CSRF."""

from __future__ import annotations

import secrets
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from dentist_ai.api.cookies import csrf_session_id
from dentist_ai.core.config import Settings
from dentist_ai.core.errors import CSRFError, problem_response
from dentist_ai.core.logging import bind_request_context, clear_request_context, get_logger
from dentist_ai.core.security import SessionService
from dentist_ai.web.templating import THEME_INIT_CSP_HASH

log = get_logger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to the logger, and time the response."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or secrets.token_hex(8)
        request.state.request_id = request_id

        clear_request_context()
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            request.state.duration_ms = duration_ms

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["Server-Timing"] = f"app;dur={duration_ms}"

        # Health checks and static assets would otherwise drown the log.
        if not request.url.path.startswith(("/static/", "/healthz")):
            log.info("request", status=response.status_code, duration_ms=duration_ms)
        clear_request_context()
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline hardening headers.

    The CSP is nonce-free and script-src 'self' because the app ships no
    inline scripts — templates reference hashed bundles only. That is
    deliberate: it makes XSS via injected ``<script>`` unexploitable rather
    than merely unlikely.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings
        self._csp = "; ".join(
            [
                "default-src 'self'",
                "base-uri 'self'",
                "form-action 'self'",
                "frame-ancestors 'none'",
                "object-src 'none'",
                # Hash-pinned rather than 'unsafe-inline': the one inline
                # script we ship is the pre-paint theme setter, and its hash is
                # derived from the source at import time.
                f"script-src 'self' {THEME_INIT_CSP_HASH}",
                # Inline styles remain necessary for per-element overlay
                # positioning computed from finding coordinates.
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: blob:",
                "font-src 'self'",
                "connect-src 'self'",
                "manifest-src 'self'",
                "upgrade-insecure-requests" if settings.is_production else "",
            ]
        ).strip("; ")

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        response = await call_next(request)
        headers = response.headers
        headers.setdefault("Content-Security-Policy", self._csp)
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), interest-cohort=()",
        )
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        if self._settings.is_production:
            headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains; preload",
            )
        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """Two independent CSRF defences on every state-changing request.

    1. **Origin check.** ``Origin``/``Referer`` must match the served host.
    2. **Signed token in a custom header.** The token is bound to the session
       id, so one minted for a different session is rejected.

    The token is read from a *header only*, never from the CSRF cookie: a
    cross-site form post would carry that cookie automatically, so accepting
    it as proof would defeat the entire mechanism. The cookie exists solely so
    the frontend can read its own token — the browser's inability to set a
    custom header cross-origin is what actually blocks the attack.

    Enforced centrally rather than per-route, so a newly added POST endpoint is
    protected by default instead of by remembering a decorator.
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings,
        sessions: SessionService,
        *,
        exempt_paths: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(app)
        self._settings = settings
        self._sessions = sessions
        self._exempt = exempt_paths

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        if request.method in SAFE_METHODS or request.url.path in self._exempt:
            return await call_next(request)

        if not self._origin_allowed(request):
            log.warning("csrf_origin_rejected", path=request.url.path)
            return problem_response(CSRFError())

        session_id = csrf_session_id(request, self._sessions, self._settings)

        token = request.headers.get(self._settings.security.csrf_header_name)
        if not self._sessions.verify_csrf(token, session_id):
            log.warning("csrf_token_rejected", path=request.url.path)
            return problem_response(CSRFError())

        return await call_next(request)

    def _origin_allowed(self, request: Request) -> bool:
        host = request.headers.get("host")
        origin = request.headers.get("origin")
        if origin is not None:
            return bool(host) and urlparse(origin).netloc == host

        # Some clients omit Origin on same-origin requests; fall back to
        # Referer, and accept its absence only outside production.
        referer = request.headers.get("referer")
        if referer is not None:
            return bool(host) and urlparse(referer).netloc == host
        return not self._settings.is_production
