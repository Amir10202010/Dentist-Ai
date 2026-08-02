"""3D scan upload, metadata and mesh delivery."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from fastapi.responses import FileResponse

from dentist_ai.api.deps import (
    ApiRateLimit,
    AuditDep,
    CurrentUser,
    MeshStorageDep,
    RequestCtx,
    ScanDep,
    UploadRateLimit,
)
from dentist_ai.api.presenters import present_scan
from dentist_ai.core.errors import NotFoundError
from dentist_ai.db.models import ScanArch, ScanKind
from dentist_ai.schemas.common import OkResponse, Page, PageMeta
from dentist_ai.schemas.scans import ScanResponse, ScanUpdateRequest
from dentist_ai.services.audit import AuditAction

router = APIRouter(prefix="/scans", tags=["scans"])

MAX_PAGE_SIZE = 100
#: The stored mesh is immutable and the URL is unguessable, so a long private
#: cache removes a multi-megabyte download on every revisit.
_MESH_CACHE_CONTROL = "private, max-age=86400, immutable"


@router.get("", response_model=Page[ScanResponse], dependencies=[ApiRateLimit])
async def list_scans(
    user: CurrentUser,
    scans: ScanDep,
    patient_id: Annotated[int, Query(description="Scans always belong to a patient.")],
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ScanResponse]:
    rows, total = await scans.list_for_patient(
        patient_id,
        organization_id=user.organization_id,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[present_scan(scan, user.locale) for scan in rows],
        meta=PageMeta(total=total, limit=limit, offset=offset, has_more=offset + len(rows) < total),
    )


@router.post(
    "",
    response_model=ScanResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[UploadRateLimit],
    summary="Upload a 3D scan (STL, PLY or OBJ)",
)
async def upload_scan(
    user: CurrentUser,
    scans: ScanDep,
    context: RequestCtx,
    file: Annotated[UploadFile, File(description="STL, PLY or OBJ, binary or ASCII")],
    patient_id: Annotated[int, Form()],
    kind: Annotated[ScanKind, Form()] = ScanKind.INTRAORAL,
    arch: Annotated[ScanArch, Form()] = ScanArch.BOTH,
    captured_on: Annotated[date | None, Form()] = None,
    notes: Annotated[str | None, Form(max_length=4000)] = None,
) -> ScanResponse:
    scan = await scans.upload(
        file,
        actor=user,
        context=context,
        patient_id=patient_id,
        kind=kind,
        arch=arch,
        captured_on=captured_on,
        notes=notes,
    )
    return present_scan(scan, user.locale)


@router.get("/{public_id}", response_model=ScanResponse, dependencies=[ApiRateLimit])
async def get_scan(
    public_id: str,
    user: CurrentUser,
    scans: ScanDep,
    context: RequestCtx,
) -> ScanResponse:
    scan = await scans.get(
        public_id, organization_id=user.organization_id, actor=user, context=context
    )
    return present_scan(scan, user.locale)


@router.patch("/{public_id}", response_model=ScanResponse, dependencies=[ApiRateLimit])
async def update_scan(
    public_id: str,
    payload: ScanUpdateRequest,
    user: CurrentUser,
    scans: ScanDep,
    context: RequestCtx,
) -> ScanResponse:
    scan = await scans.update(
        public_id,
        actor=user,
        context=context,
        kind=payload.kind,
        arch=payload.arch,
        captured_on=payload.captured_on,
        notes=payload.notes,
    )
    return present_scan(scan, user.locale)


@router.delete("/{public_id}", response_model=OkResponse, dependencies=[ApiRateLimit])
async def delete_scan(
    public_id: str,
    user: CurrentUser,
    scans: ScanDep,
    context: RequestCtx,
) -> OkResponse:
    await scans.delete(public_id, actor=user, context=context)
    return OkResponse()


@router.get("/{public_id}/mesh", summary="Canonical binary STL")
async def get_mesh(
    public_id: str,
    user: CurrentUser,
    scans: ScanDep,
    storage: MeshStorageDep,
    audit: AuditDep,
    context: RequestCtx,
) -> FileResponse:
    scan = await scans.get(public_id, organization_id=user.organization_id)
    path = storage.path(scan.content_hash)
    if not path.is_file():
        raise NotFoundError("Файл модели недоступен.")

    await audit.record(
        action=AuditAction.SCAN_VIEWED,
        organization_id=user.organization_id,
        actor_id=user.id,
        resource_type="scan",
        resource_id=public_id,
        context=context,
    )
    return FileResponse(
        path,
        media_type="model/stl",
        headers={"Cache-Control": _MESH_CACHE_CONTROL},
    )
