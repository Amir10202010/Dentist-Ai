"""Workspace search: one endpoint over four kinds of record."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from dentist_ai.api.deps import ApiRateLimit, CurrentUser, DbSession
from dentist_ai.ml.taxonomy import Severity
from dentist_ai.schemas.search import SearchKind, SearchResults
from dentist_ai.services.search import SearchFilters, SearchService

router = APIRouter(prefix="/search", tags=["search"], dependencies=[ApiRateLimit])

#: Per kind, not in total. A search that returned fifty of everything would be
#: a page nobody reads; the client asks for more of the one group it wants.
MAX_PAGE_SIZE = 50


def get_search_service(session: DbSession) -> SearchService:
    return SearchService(session)


type SearchDep = Annotated[SearchService, Depends(get_search_service)]


@router.get(
    "",
    response_model=SearchResults,
    summary="Search patients, radiographs, CBCT volumes and the case library",
)
async def search(
    user: CurrentUser,
    search_service: SearchDep,
    q: Annotated[str | None, Query(max_length=120, description="Free text")] = None,
    kind: Annotated[SearchKind | None, Query(description="Restrict to one kind")] = None,
    severity: Annotated[Severity | None, Query()] = None,
    finding: Annotated[str | None, Query(max_length=64, description="Finding class key")] = None,
    date_from: Annotated[date | None, Query()] = None,
    date_to: Annotated[date | None, Query(description="Inclusive")] = None,
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    doctor: Annotated[int | None, Query(description="Uploaded by this user id")] = None,
    patient_id: Annotated[int | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 10,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SearchResults:
    return await search_service.search(
        SearchFilters(
            query=q,
            kind=kind,
            severity=severity,
            finding=finding,
            date_from=date_from,
            date_to=date_to,
            min_confidence=min_confidence,
            doctor_id=doctor,
            patient_id=patient_id,
            limit=limit,
            offset=offset,
        ),
        organization_id=user.organization_id,
        locale=user.locale,
    )
