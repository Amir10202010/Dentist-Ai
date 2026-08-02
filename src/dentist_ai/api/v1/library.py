"""Case library CRUD.

``/tags`` is declared before ``/{public_id}``: FastAPI matches in declaration
order, so the reverse would make the tag list resolve as a case whose public id
happens to be "tags".
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from dentist_ai.api.deps import ApiRateLimit, AuditDep, CurrentUser, DbSession, RequestCtx
from dentist_ai.db.models import CaseEntry
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale
from dentist_ai.schemas.common import OkResponse, Page, PageMeta
from dentist_ai.schemas.library import (
    CaseCreateRequest,
    CaseFinding,
    CaseListItemResponse,
    CaseResponse,
    CaseTagResponse,
    CaseUpdateRequest,
)
from dentist_ai.services.library import LibraryService, describe_finding, split_tokens

router = APIRouter(prefix="/library", tags=["library"], dependencies=[ApiRateLimit])

MAX_PAGE_SIZE = 100


def get_library_service(session: DbSession, audit: AuditDep) -> LibraryService:
    return LibraryService(session, audit)


type LibraryDep = Annotated[LibraryService, Depends(get_library_service)]


def _present_row(entry: CaseEntry, locale: Locale = DEFAULT_LOCALE) -> CaseListItemResponse:
    return CaseListItemResponse(
        public_id=entry.public_id,
        title=entry.title,
        summary=entry.summary,
        tags=split_tokens(entry.tags),
        findings=_present_findings(entry.finding_keys, locale),
        created_at=entry.created_at,
        created_by_name=entry.created_by.full_name if entry.created_by else None,
        href=f"/app/library/{entry.public_id}",
    )


def _present(entry: CaseEntry, locale: Locale = DEFAULT_LOCALE) -> CaseResponse:
    row = _present_row(entry, locale)
    return CaseResponse(
        **row.model_dump(),
        diagnosis=entry.diagnosis,
        treatment=entry.treatment,
        outcome=entry.outcome,
        patient_id=entry.patient_id,
        study_public_id=entry.study_public_id,
        volume_public_id=entry.volume_public_id,
        updated_at=entry.updated_at,
    )


def _present_findings(raw: str, locale: Locale) -> list[CaseFinding]:
    return [
        CaseFinding(key=item.key, label=item.label, severity=item.severity)
        for item in (describe_finding(key, locale) for key in split_tokens(raw))
    ]


@router.get("", response_model=Page[CaseListItemResponse], summary="List case-library entries")
async def list_cases(
    user: CurrentUser,
    library: LibraryDep,
    q: Annotated[str | None, Query(max_length=120)] = None,
    tag: Annotated[str | None, Query(max_length=40)] = None,
    finding: Annotated[str | None, Query(max_length=64, description="Finding class key")] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[CaseListItemResponse]:
    entries, total = await library.list_entries(
        organization_id=user.organization_id,
        query=q,
        tag=tag,
        finding=finding,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[_present_row(entry, user.locale) for entry in entries],
        meta=PageMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(entries) < total,
        ),
    )


@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Publish a case, optionally seeded from a study or a volume",
)
async def create_case(
    payload: CaseCreateRequest,
    user: CurrentUser,
    library: LibraryDep,
    context: RequestCtx,
) -> CaseResponse:
    entry = await library.create(payload, actor=user, context=context)
    return _present(entry, user.locale)


@router.get(
    "/tags",
    response_model=list[CaseTagResponse],
    summary="Distinct tags with counts, for the filter chips",
)
async def list_tags(user: CurrentUser, library: LibraryDep) -> list[CaseTagResponse]:
    counts = await library.tag_counts(organization_id=user.organization_id)
    return [CaseTagResponse(tag=item.tag, count=item.count) for item in counts]


@router.get("/{public_id}", response_model=CaseResponse, summary="Get a case")
async def get_case(public_id: str, user: CurrentUser, library: LibraryDep) -> CaseResponse:
    entry = await library.get(public_id, organization_id=user.organization_id)
    return _present(entry, user.locale)


@router.put("/{public_id}", response_model=CaseResponse, summary="Replace a case")
async def update_case(
    public_id: str,
    payload: CaseUpdateRequest,
    user: CurrentUser,
    library: LibraryDep,
    context: RequestCtx,
) -> CaseResponse:
    entry = await library.update(public_id, payload, actor=user, context=context)
    return _present(entry, user.locale)


@router.delete("/{public_id}", response_model=OkResponse, summary="Delete a case")
async def delete_case(
    public_id: str,
    user: CurrentUser,
    library: LibraryDep,
    context: RequestCtx,
) -> OkResponse:
    await library.delete(public_id, actor=user, context=context)
    return OkResponse()
