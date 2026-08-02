"""Radiograph upload, analysis results and the authorised image route."""

from __future__ import annotations

import csv
import io
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from dentist_ai.api.deps import (
    ApiRateLimit,
    AuditDep,
    CurrentUser,
    RequestCtx,
    StorageDep,
    StudyDep,
)
from dentist_ai.api.presenters import present_finding, present_study, present_study_row
from dentist_ai.clinical.report import build_report
from dentist_ai.core.errors import NotFoundError
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import StudyStatus
from dentist_ai.schemas.clinical import (
    FindingResponse,
    FindingReviewRequest,
    FindingToothRequest,
    StudyListItemResponse,
    StudyReportResponse,
    StudyResponse,
    StudyUpdateRequest,
)
from dentist_ai.schemas.common import OkResponse, Page, PageMeta
from dentist_ai.services.audit import AuditAction

router = APIRouter(prefix="/studies", tags=["studies"])

MAX_PAGE_SIZE = 60
#: Studies are immutable once analysed, and the URL is unguessable, so a long
#: private cache is safe and removes a round-trip on every viewer interaction.
_IMAGE_CACHE_CONTROL = "private, max-age=86400, immutable"


@router.get("", response_model=Page[StudyListItemResponse], dependencies=[ApiRateLimit])
async def list_studies(
    user: CurrentUser,
    studies: StudyDep,
    patient_id: Annotated[int | None, Query()] = None,
    status_filter: Annotated[StudyStatus | None, Query(alias="status")] = None,
    q: Annotated[str | None, Query(max_length=120)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 24,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[StudyListItemResponse]:
    rows, total = await studies.list_studies(
        organization_id=user.organization_id,
        patient_id=patient_id,
        status=status_filter,
        query=q,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[present_study_row(row, user.locale) for row in rows],
        meta=PageMeta(total=total, limit=limit, offset=offset, has_more=offset + len(rows) < total),
    )


@router.post(
    "",
    response_model=StudyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a radiograph and analyse it",
)
async def upload_study(
    user: CurrentUser,
    studies: StudyDep,
    context: RequestCtx,
    file: Annotated[UploadFile, File(description="JPEG, PNG, WebP, BMP or TIFF")],
    patient_id: Annotated[int | None, Form()] = None,
) -> StudyResponse:
    study = await studies.upload_and_analyze(
        file, actor=user, context=context, patient_id=patient_id
    )
    return present_study(study, user.locale)


@router.get("/{public_id}", response_model=StudyResponse, dependencies=[ApiRateLimit])
async def get_study(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
    context: RequestCtx,
) -> StudyResponse:
    study = await studies.get_detailed(
        public_id, organization_id=user.organization_id, actor=user, context=context
    )
    return present_study(study, user.locale)


@router.patch("/{public_id}", response_model=StudyResponse, dependencies=[ApiRateLimit])
async def update_study(
    public_id: str,
    payload: StudyUpdateRequest,
    user: CurrentUser,
    studies: StudyDep,
    context: RequestCtx,
) -> StudyResponse:
    await studies.update(
        public_id,
        actor=user,
        context=context,
        patient_id=payload.patient_id,
        notes=payload.notes,
    )
    study = await studies.get_detailed(public_id, organization_id=user.organization_id)
    return present_study(study, user.locale)


@router.post("/{public_id}/reanalyze", response_model=StudyResponse, dependencies=[ApiRateLimit])
async def reanalyze_study(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
) -> StudyResponse:
    study = await studies.get_detailed(public_id, organization_id=user.organization_id)
    await studies.reanalyze(study)
    refreshed = await studies.get_detailed(public_id, organization_id=user.organization_id)
    return present_study(refreshed, user.locale)


@router.delete("/{public_id}", response_model=OkResponse, dependencies=[ApiRateLimit])
async def delete_study(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
    context: RequestCtx,
) -> OkResponse:
    await studies.delete(public_id, actor=user, context=context)
    return OkResponse()


@router.patch(
    "/{public_id}/findings/{finding_id}",
    response_model=FindingResponse,
    dependencies=[ApiRateLimit],
    summary="Confirm or reject a detection",
)
async def review_finding(
    public_id: str,
    finding_id: int,
    payload: FindingReviewRequest,
    user: CurrentUser,
    studies: StudyDep,
    context: RequestCtx,
) -> FindingResponse:
    finding = await studies.review_finding(
        public_id, finding_id, payload.review, actor=user, context=context
    )
    return present_finding(finding, user.locale)


@router.put(
    "/{public_id}/findings/{finding_id}/tooth",
    response_model=FindingResponse,
    dependencies=[ApiRateLimit],
    summary="Correct the FDI number of a detection",
)
async def set_finding_tooth(
    public_id: str,
    finding_id: int,
    payload: FindingToothRequest,
    user: CurrentUser,
    studies: StudyDep,
    context: RequestCtx,
) -> FindingResponse:
    finding = await studies.set_finding_tooth(
        public_id, finding_id, payload.tooth_number, actor=user, context=context
    )
    return present_finding(finding, user.locale)


@router.get(
    "/{public_id}/report",
    response_model=StudyReportResponse,
    dependencies=[ApiRateLimit],
    summary="Odontogram, per-tooth findings and protocol recommendations",
)
async def get_report(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
) -> StudyReportResponse:
    study = await studies.get_detailed(public_id, organization_id=user.organization_id)
    presented = present_study(study, user.locale)
    return build_report(
        study_public_id=study.public_id,
        patient_name=presented.patient.full_name if presented.patient else None,
        findings=presented.findings,
        generated_at=utcnow(),
        locale=user.locale,
    )


# --------------------------------------------------------------------------
# Image delivery — the only path out of private storage
# --------------------------------------------------------------------------
@router.get("/{public_id}/image", summary="Full-resolution radiograph")
async def get_image(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
    audit: AuditDep,
    context: RequestCtx,
    storage: StorageDep,
) -> FileResponse:
    study = await studies.get_for_image_access(public_id, organization_id=user.organization_id)
    path = storage.master_path(study.content_hash)
    if not path.is_file():
        raise NotFoundError("Файл снимка недоступен.")

    await audit.record(
        action=AuditAction.STUDY_IMAGE_ACCESSED,
        organization_id=user.organization_id,
        actor_id=user.id,
        resource_type="study",
        resource_id=public_id,
        context=context,
    )
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": _IMAGE_CACHE_CONTROL},
    )


@router.get("/{public_id}/thumbnail", summary="Thumbnail")
async def get_thumbnail(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
    storage: StorageDep,
) -> FileResponse:
    study = await studies.get_for_image_access(public_id, organization_id=user.organization_id)
    path = storage.thumbnail_path(study.content_hash)
    if not path.is_file():
        path = storage.master_path(study.content_hash)
    if not path.is_file():
        raise NotFoundError("Файл снимка недоступен.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": _IMAGE_CACHE_CONTROL},
    )


@router.get("/{public_id}/export.csv", summary="Export findings as CSV")
async def export_findings(
    public_id: str,
    user: CurrentUser,
    studies: StudyDep,
    audit: AuditDep,
    context: RequestCtx,
) -> StreamingResponse:
    study = await studies.get_detailed(public_id, organization_id=user.organization_id)
    presented = present_study(study, user.locale)

    buffer = io.StringIO()
    # BOM so Excel opens Cyrillic UTF-8 correctly — without it the most common
    # consumer of this file renders every label as mojibake.
    buffer.write("﻿")
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(
        ["Снимок", "Пациент", "Зуб", "Находка", "Категория", "Важность", "Уверенность", "Статус"]
    )
    for finding in presented.findings:
        writer.writerow(
            [
                study.public_id,
                presented.patient.full_name if presented.patient else "—",
                finding.tooth_number or "—",
                finding.label,
                finding.category.value,
                finding.severity.value,
                f"{finding.confidence:.1%}",
                finding.review.value,
            ]
        )

    await audit.record(
        action=AuditAction.STUDY_EXPORTED,
        organization_id=user.organization_id,
        actor_id=user.id,
        resource_type="study",
        resource_id=public_id,
        context=context,
    )

    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="study-{public_id}.csv"',
        },
    )


@router.head("/{public_id}/image", include_in_schema=False)
async def head_image(public_id: str, user: CurrentUser, studies: StudyDep) -> Response:
    await studies.get_for_image_access(public_id, organization_id=user.organization_id)
    return Response(status_code=status.HTTP_200_OK, media_type="image/jpeg")
