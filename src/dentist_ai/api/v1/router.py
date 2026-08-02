"""v1 API surface."""

from __future__ import annotations

from fastapi import APIRouter

from dentist_ai.api.v1 import (
    analytics,
    assistant,
    auth,
    collaboration,
    library,
    patients,
    planning,
    scans,
    search,
    settings,
    studies,
    timeline,
    treatment,
    volumes,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(patients.router)
api_router.include_router(studies.router)
api_router.include_router(scans.router)
api_router.include_router(volumes.router)
api_router.include_router(planning.router)
api_router.include_router(assistant.router)
api_router.include_router(search.router)
api_router.include_router(library.router)
api_router.include_router(collaboration.router)
api_router.include_router(timeline.router)
api_router.include_router(treatment.router)
api_router.include_router(analytics.router)
api_router.include_router(settings.router)
