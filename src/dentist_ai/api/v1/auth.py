"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from dentist_ai.api.cookies import clear_session_cookies, csrf_session_id, set_session_cookies
from dentist_ai.api.deps import (
    AppSettings,
    AuditDep,
    AuthDep,
    CurrentUser,
    LoginRateLimit,
    RegisterRateLimit,
    RequestCtx,
    Sessions,
    get_rate_limiter,
)
from dentist_ai.db.models import User
from dentist_ai.schemas.auth import (
    LoginRequest,
    OrganizationResponse,
    RegisterRequest,
    SessionResponse,
    UserResponse,
)
from dentist_ai.schemas.common import OkResponse
from dentist_ai.services.audit import AuditAction

router = APIRouter(prefix="/auth", tags=["auth"])


def present_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        initials=user.initials,
        role=user.role,
        locale=user.locale,
        last_login_at=user.last_login_at,
        organization=OrganizationResponse(
            id=user.organization.id,
            name=user.organization.name,
            slug=user.organization.slug,
        ),
    )


@router.post(
    "/register",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RegisterRateLimit],
    summary="Create a clinic account and sign in",
)
async def register(
    payload: RegisterRequest,
    response: Response,
    auth: AuthDep,
    sessions: Sessions,
    settings: AppSettings,
    context: RequestCtx,
) -> SessionResponse:
    user = await auth.register(payload, context)
    # Sign the new owner straight in — a "now go log in" round-trip after
    # registration is friction with no security benefit.
    csrf_token = set_session_cookies(response, user.id, sessions, settings)
    return SessionResponse(user=present_user(user), csrf_token=csrf_token)


@router.post(
    "/login",
    response_model=SessionResponse,
    dependencies=[LoginRateLimit],
    summary="Sign in",
)
async def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
    auth: AuthDep,
    sessions: Sessions,
    settings: AppSettings,
    context: RequestCtx,
) -> SessionResponse:
    user = await auth.authenticate(payload, context)

    # A successful login clears the attacker's budget, so a legitimate user who
    # fat-fingered their password a few times is not locked out afterwards.
    await get_rate_limiter(request).reset(f"login:ip:{context.ip_address}")

    csrf_token = set_session_cookies(response, user.id, sessions, settings)
    return SessionResponse(user=present_user(user), csrf_token=csrf_token)


@router.post("/logout", response_model=OkResponse, summary="Sign out")
async def logout(
    response: Response,
    settings: AppSettings,
    user: CurrentUser,
    audit: AuditDep,
    context: RequestCtx,
) -> OkResponse:
    await audit.record(
        action=AuditAction.LOGOUT,
        organization_id=user.organization_id,
        actor_id=user.id,
        resource_type="user",
        resource_id=user.id,
        context=context,
    )
    clear_session_cookies(response, settings)
    return OkResponse()


@router.get("/me", response_model=UserResponse, summary="Current user")
async def me(user: CurrentUser) -> UserResponse:
    return present_user(user)


class CsrfResponse(OkResponse):
    csrf_token: str


@router.get("/csrf", response_model=CsrfResponse, summary="Bootstrap a CSRF token")
async def csrf(
    request: Request,
    response: Response,
    sessions: Sessions,
    settings: AppSettings,
) -> CsrfResponse:
    """Issue a CSRF token for the current (possibly anonymous) session.

    Handing this out freely is safe: the token alone grants nothing. What
    stops a cross-origin attacker is that a browser will not let their page
    attach a custom header to a request at our origin — plus the Origin check
    in ``CSRFMiddleware``.
    """
    token = sessions.issue_csrf(csrf_session_id(request, sessions, settings))
    response.set_cookie(
        settings.security.csrf_cookie_name,
        token,
        max_age=settings.security.session_max_age_seconds,
        httponly=False,
        secure=settings.is_production,
        samesite="lax",
        path="/",
    )
    return CsrfResponse(csrf_token=token)
