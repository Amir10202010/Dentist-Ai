"""Dashboard metrics and the finding taxonomy reference."""

from __future__ import annotations

from fastapi import APIRouter, Response

from dentist_ai.api.deps import AnalyticsDep, ApiRateLimit, CurrentUser
from dentist_ai.ml.taxonomy import FINDING_CLASSES
from dentist_ai.schemas.clinical import DashboardResponse
from dentist_ai.schemas.common import ApiModel

router = APIRouter(tags=["analytics"])


class TaxonomyEntry(ApiModel):
    class_id: int
    key: str
    label: str
    category: str
    severity: str
    needs_attention: bool


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    dependencies=[ApiRateLimit],
    summary="Clinic dashboard metrics",
)
async def dashboard(
    user: CurrentUser,
    analytics: AnalyticsDep,
    response: Response,
) -> DashboardResponse:
    # Short private cache: the dashboard is expensive relative to its rate of
    # change, and a 30s window collapses a page refresh storm into one query.
    response.headers["Cache-Control"] = "private, max-age=30"
    return await analytics.dashboard(
        organization_id=user.organization_id,
        locale=user.locale,
        # Eagerly loaded with the user, so reading it costs no extra query.
        timezone=user.organization.timezone,
    )


@router.get(
    "/taxonomy",
    response_model=list[TaxonomyEntry],
    summary="All detectable finding classes",
)
async def taxonomy(user: CurrentUser, response: Response) -> list[TaxonomyEntry]:
    # Immutable for the lifetime of a deployment.
    response.headers["Cache-Control"] = "private, max-age=3600"
    return [
        TaxonomyEntry(
            class_id=item.class_id,
            key=item.key,
            label=item.label(user.locale),
            category=item.category.value,
            severity=item.severity.value,
            needs_attention=item.needs_attention,
        )
        for item in FINDING_CLASSES
    ]
