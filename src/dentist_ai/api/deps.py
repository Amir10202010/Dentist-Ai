"""FastAPI dependencies — the composition root for every request."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Annotated

from fastapi import Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from dentist_ai.core.config import RateLimitRule, Settings
from dentist_ai.core.errors import AuthenticationError, RateLimitedError
from dentist_ai.core.ratelimit import RateLimiter
from dentist_ai.core.security import PasswordService, SessionService
from dentist_ai.db.models import User
from dentist_ai.db.session import session_scope
from dentist_ai.ml.detector import Detector
from dentist_ai.ml.pipeline import ModelRegistry
from dentist_ai.services.analytics import AnalyticsService
from dentist_ai.services.assistant import AssistantService
from dentist_ai.services.audit import AuditService, RequestContext
from dentist_ai.services.auth import AuthService
from dentist_ai.services.notifications import NotificationService
from dentist_ai.services.patients import PatientService
from dentist_ai.services.planning import PlanningService
from dentist_ai.services.scans import ScanService
from dentist_ai.services.storage import ImageStorage, MeshStorage, VolumeStorage
from dentist_ai.services.studies import StudyService
from dentist_ai.services.treatment import TreatmentService
from dentist_ai.services.volumes import VolumeService

# --------------------------------------------------------------------------
# Singletons stashed on app.state during startup
# --------------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_sessions(request: Request) -> SessionService:
    sessions: SessionService = request.app.state.sessions
    return sessions


def get_passwords(request: Request) -> PasswordService:
    passwords: PasswordService = request.app.state.passwords
    return passwords


def get_storage(request: Request) -> ImageStorage:
    storage: ImageStorage = request.app.state.storage
    return storage


def get_mesh_storage(request: Request) -> MeshStorage:
    storage: MeshStorage = request.app.state.mesh_storage
    return storage


def get_volume_storage(request: Request) -> VolumeStorage:
    storage: VolumeStorage = request.app.state.volume_storage
    return storage


def get_model_registry(request: Request) -> ModelRegistry:
    registry: ModelRegistry = request.app.state.model_registry
    return registry


def get_detector(request: Request) -> Detector:
    detector: Detector = request.app.state.detector
    return detector


def get_rate_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    """One transactional session per request; commits on success."""
    async with session_scope(request.app.state.session_factory) as session:
        yield session


type DbSession = Annotated[AsyncSession, Depends(get_db)]
type AppSettings = Annotated[Settings, Depends(get_settings)]
type Sessions = Annotated[SessionService, Depends(get_sessions)]
type StorageDep = Annotated[ImageStorage, Depends(get_storage)]
type MeshStorageDep = Annotated[MeshStorage, Depends(get_mesh_storage)]
type VolumeStorageDep = Annotated[VolumeStorage, Depends(get_volume_storage)]
type ModelRegistryDep = Annotated[ModelRegistry, Depends(get_model_registry)]
type DetectorDep = Annotated[Detector, Depends(get_detector)]


# --------------------------------------------------------------------------
# Request-scoped values
# --------------------------------------------------------------------------
def get_request_context(request: Request) -> RequestContext:
    return RequestContext(
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


def _client_ip(request: Request) -> str | None:
    """Resolve the client IP behind a reverse proxy.

    Only the *last* hop in ``X-Forwarded-For`` is trustworthy, and only when
    the app is deployed behind a proxy that overwrites it. Uvicorn's
    ``--proxy-headers`` populates ``request.client`` correctly, so that is the
    primary source and the header is a fallback.
    """
    if request.client is not None:
        return request.client.host
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return None


type RequestCtx = Annotated[RequestContext, Depends(get_request_context)]


# --------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------
def get_audit_service(session: DbSession) -> AuditService:
    return AuditService(session)


type AuditDep = Annotated[AuditService, Depends(get_audit_service)]


def get_auth_service(
    session: DbSession,
    passwords: Annotated[PasswordService, Depends(get_passwords)],
    audit: AuditDep,
) -> AuthService:
    return AuthService(session, passwords, audit)


def get_patient_service(session: DbSession, audit: AuditDep) -> PatientService:
    return PatientService(session, audit)


def get_study_service(
    session: DbSession,
    storage: StorageDep,
    detector: DetectorDep,
    audit: AuditDep,
) -> StudyService:
    return StudyService(session, storage, detector, audit)


def get_scan_service(
    session: DbSession,
    storage: MeshStorageDep,
    audit: AuditDep,
) -> ScanService:
    return ScanService(session, storage, audit)


def get_treatment_service(session: DbSession, audit: AuditDep) -> TreatmentService:
    return TreatmentService(session, audit)


def get_analytics_service(session: DbSession) -> AnalyticsService:
    return AnalyticsService(session)


def get_notification_service(session: DbSession, audit: AuditDep) -> NotificationService:
    return NotificationService(session, audit)


type NotificationDep = Annotated[NotificationService, Depends(get_notification_service)]


def get_volume_service(
    session: DbSession,
    storage: VolumeStorageDep,
    registry: ModelRegistryDep,
    audit: AuditDep,
    notifications: NotificationDep,
) -> VolumeService:
    return VolumeService(session, storage, registry, audit, notifications)


type AuthDep = Annotated[AuthService, Depends(get_auth_service)]
type PatientDep = Annotated[PatientService, Depends(get_patient_service)]
type StudyDep = Annotated[StudyService, Depends(get_study_service)]
type ScanDep = Annotated[ScanService, Depends(get_scan_service)]
type TreatmentDep = Annotated[TreatmentService, Depends(get_treatment_service)]
def get_planning_service(session: DbSession, audit: AuditDep) -> PlanningService:
    return PlanningService(session, audit)


def get_assistant_service(session: DbSession, audit: AuditDep) -> AssistantService:
    return AssistantService(session, audit)


type AnalyticsDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
type VolumeDep = Annotated[VolumeService, Depends(get_volume_service)]
type PlanningDep = Annotated[PlanningService, Depends(get_planning_service)]
type AssistantDep = Annotated[AssistantService, Depends(get_assistant_service)]


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------
async def get_optional_user(
    request: Request,
    settings: AppSettings,
    sessions: Sessions,
    auth: AuthDep,
) -> User | None:
    """Resolve the signed-in user, or ``None``. Never raises."""
    cached = getattr(request.state, "user", None)
    if isinstance(cached, User):
        return cached

    session_data = sessions.load(request.cookies.get(settings.security.session_cookie_name))
    if session_data is None:
        return None

    user = await auth.get_active_user(session_data.user_id)
    request.state.user = user
    request.state.session_data = session_data
    return user


type OptionalUser = Annotated[User | None, Depends(get_optional_user)]


async def get_current_user(user: OptionalUser) -> User:
    if user is None:
        raise AuthenticationError
    return user


type CurrentUser = Annotated[User, Depends(get_current_user)]


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
def rate_limit(
    bucket: str,
    rule_for: Callable[[Settings], RateLimitRule],
    *,
    per_user: bool = False,
) -> Callable[..., Coroutine[None, None, None]]:
    """Dependency factory applying a named rate-limit bucket.

    Keyed by user id when authenticated (so a shared clinic NAT does not lock
    out an entire practice) and by IP otherwise.
    """

    async def _guard(
        request: Request,
        response: Response,
        settings: AppSettings,
        limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
        user: OptionalUser,
    ) -> None:
        identity = (
            f"user:{user.id}" if per_user and user is not None else f"ip:{_client_ip(request)}"
        )
        rule = rule_for(settings)
        decision = await limiter.check(f"{bucket}:{identity}", rule)

        response.headers["X-RateLimit-Limit"] = str(rule.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)

        if not decision.allowed:
            raise RateLimitedError(decision.retry_after_seconds)

    return _guard


LoginRateLimit = Depends(rate_limit("login", lambda s: s.security.login_rule))
RegisterRateLimit = Depends(rate_limit("register", lambda s: s.security.register_rule))
UploadRateLimit = Depends(rate_limit("upload", lambda s: s.security.upload_rule, per_user=True))
ApiRateLimit = Depends(rate_limit("api", lambda s: s.security.api_rule, per_user=True))
