"""Application factory.

Every singleton is constructed here and attached to ``app.state``, so tests
can build an app with different pieces without patching module globals.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dentist_ai.api.middleware import (
    CSRFMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from dentist_ai.api.v1.router import api_router
from dentist_ai.core.config import Settings, get_settings
from dentist_ai.core.errors import register_exception_handlers
from dentist_ai.core.logging import configure_logging, get_logger
from dentist_ai.core.ratelimit import InMemoryRateLimiter
from dentist_ai.core.security import PasswordService, SessionService
from dentist_ai.db.session import create_engine, create_session_factory
from dentist_ai.ml.cbct import build_registry
from dentist_ai.ml.factory import build_detector
from dentist_ai.services.storage import ImageStorage, MeshStorage, VolumeStorage
from dentist_ai.web.routes import LoginRequiredError
from dentist_ai.web.routes import router as web_router
from dentist_ai.web.templating import STATIC_DIR

log = get_logger(__name__)

#: Login and registration are the only unsafe endpoints reachable without a
#: session; they are protected by the origin check plus the anonymous CSRF
#: token, not exempted.
CSRF_EXEMPT_PATHS: frozenset[str] = frozenset({"/healthz", "/readyz"})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.sessions = SessionService(settings)
        app.state.passwords = PasswordService(settings)
        app.state.storage = ImageStorage(settings.storage)
        app.state.mesh_storage = MeshStorage(settings.storage)
        app.state.volume_storage = VolumeStorage(settings.storage)
        app.state.rate_limiter = InMemoryRateLimiter()
        app.state.detector = build_detector(settings)
        # The CBCT pipelines hold no state and no weights, so one registry per
        # process is enough and building it costs nothing.
        app.state.model_registry = build_registry()

        if settings.ml.warm_up_on_startup:
            try:
                await app.state.detector.warm_up()
            except Exception:
                # A cold model must not stop the web tier from serving; the
                # first inference request will surface a clean 503 instead.
                log.exception("detector_warm_up_failed")

        log.info(
            "application_started",
            environment=settings.environment,
            detector=settings.ml.backend,
            pipelines=app.state.model_registry.names(),
        )
        try:
            yield
        finally:
            shutdown = getattr(app.state.detector, "shutdown", None)
            if callable(shutdown):
                shutdown()
            await engine.dispose()
            log.info("application_stopped")

    app = FastAPI(
        title="Dentist-AI",
        version="1.0.0",
        summary="AI-assisted analysis of dental radiographs.",
        lifespan=lifespan,
        # Interactive docs expose the full data model; fine for development,
        # not something to publish alongside a patient database.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    _install_middleware(app, settings)
    register_exception_handlers(app)

    @app.exception_handler(LoginRequiredError)
    async def _redirect_to_login(_: Request, exc: LoginRequiredError) -> RedirectResponse:
        return RedirectResponse(f"/login?next={quote(exc.next_path)}", status_code=303)

    app.include_router(api_router)
    app.include_router(web_router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: is the process up? Touches nothing else."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> dict[str, str]:
        """Readiness: can we actually serve? Verifies the database round-trips."""
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ready", "detector": settings.ml.backend}

    return app


def _install_middleware(app: FastAPI, settings: Settings) -> None:
    # Starlette runs middleware in reverse registration order, so the last one
    # added is outermost. Request context must wrap everything to give every
    # log line — including those from failures deeper in the stack — an id.
    app.add_middleware(
        CSRFMiddleware,
        settings=settings,
        sessions=SessionService(settings),
        exempt_paths=CSRF_EXEMPT_PATHS,
    )
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    if settings.is_production:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.security.trusted_hosts)
    app.add_middleware(RequestContextMiddleware)


app = create_app()
