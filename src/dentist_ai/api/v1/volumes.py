"""CBCT upload, analysis, voxel delivery, measurements and annotations.

The one endpoint worth reading carefully is ``GET /{id}/voxels``. It serves a
canonical ``DVOL`` payload — up to 16 MB — and it is the hot path for the
viewer, so it carries a long private cache and an ``ETag``. The bytes are
immutable and content-addressed, so a conditional request can be answered
without touching the disk at all.

It is also the endpoint that hands a browser a patient's anatomy, so it checks
organisation membership and writes an audit row exactly like the radiograph and
mesh endpoints do.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from dentist_ai.api.deps import (
    ApiRateLimit,
    AuditDep,
    CurrentUser,
    RequestCtx,
    UploadRateLimit,
    VolumeDep,
    VolumeStorageDep,
)
from dentist_ai.api.presenters import (
    present_annotation,
    present_measurement,
    present_volume,
    present_volume_finding,
    present_volume_row,
)
from dentist_ai.core.errors import NotFoundError
from dentist_ai.db.models import VolumeFieldOfView
from dentist_ai.ml.cbct import build_registry
from dentist_ai.schemas.common import OkResponse, Page, PageMeta
from dentist_ai.schemas.volumes import (
    AnnotationCreateRequest,
    AnnotationResponse,
    FindingReviewRequest,
    MeasurementCreateRequest,
    MeasurementResponse,
    ReanalyseRequest,
    VolumeFindingResponse,
    VolumeListItemResponse,
    VolumeResponse,
    VolumeUpdateRequest,
)
from dentist_ai.services.audit import AuditAction

router = APIRouter(prefix="/volumes", tags=["volumes"])

MAX_PAGE_SIZE = 100
#: The payload is derived from immutable, content-addressed bytes, so a day of
#: private cache removes a multi-megabyte transfer on every revisit to a case.
_VOXEL_CACHE_CONTROL = "private, max-age=86400, immutable"
_PREVIEW_CACHE_CONTROL = "private, max-age=86400, immutable"
_PREVIEW_PLANES = frozenset({"axial", "coronal", "sagittal"})


@router.get("", response_model=Page[VolumeListItemResponse], dependencies=[ApiRateLimit])
async def list_volumes(
    user: CurrentUser,
    volumes: VolumeDep,
    patient_id: Annotated[int | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[VolumeListItemResponse]:
    rows, total = await volumes.list_for_organization(
        organization_id=user.organization_id,
        patient_id=patient_id,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[present_volume_row(row, user.locale) for row in rows],
        meta=PageMeta(total=total, limit=limit, offset=offset, has_more=offset + len(rows) < total),
    )


@router.post(
    "",
    response_model=VolumeResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[UploadRateLimit],
    summary="Upload a CBCT study (DICOM series in a ZIP, multi-frame DICOM, or NIfTI)",
)
async def upload_volume(
    user: CurrentUser,
    volumes: VolumeDep,
    context: RequestCtx,
    file: Annotated[UploadFile, File(description="ZIP of DICOM, .dcm, .nii or .nii.gz")],
    patient_id: Annotated[int, Form()],
    field_of_view: Annotated[VolumeFieldOfView, Form()] = VolumeFieldOfView.BOTH_JAWS,
    captured_on: Annotated[date | None, Form()] = None,
    notes: Annotated[str | None, Form(max_length=4000)] = None,
    pipeline: Annotated[str | None, Form(max_length=64)] = None,
) -> VolumeResponse:
    record = await volumes.upload(
        file,
        actor=user,
        context=context,
        patient_id=patient_id,
        field_of_view=field_of_view,
        captured_on=captured_on,
        notes=notes,
        pipeline_name=pipeline,
    )
    return present_volume(record, user.locale)


@router.get("/pipelines", dependencies=[ApiRateLimit], summary="Registered analysis pipelines")
async def list_pipelines(user: CurrentUser) -> list[dict[str, object]]:
    """What the deployment can run, and which stages each pipeline contains.

    Exposed because "which model produced this" is only answerable if the set
    of models is inspectable. The viewer's attribution panel reads it.
    """
    _ = user
    registry = build_registry()
    return [
        {
            "name": pipeline.name,
            "version": pipeline.version,
            "stages": [
                {
                    "name": stage.name,
                    "kind": stage.kind.value,
                    "kindLabel": stage.kind_label,
                    "version": stage.version,
                }
                for stage in pipeline.describe()
            ],
        }
        for pipeline in registry.all()
    ]


@router.get("/{public_id}", response_model=VolumeResponse, dependencies=[ApiRateLimit])
async def get_volume(
    public_id: str,
    user: CurrentUser,
    volumes: VolumeDep,
    context: RequestCtx,
) -> VolumeResponse:
    record = await volumes.get(
        public_id, organization_id=user.organization_id, actor=user, context=context
    )
    return present_volume(record, user.locale)


@router.get("/{public_id}/voxels", summary="Canonical DVOL payload for the viewer")
async def get_voxels(
    public_id: str,
    request: Request,
    user: CurrentUser,
    volumes: VolumeDep,
    audit: AuditDep,
    context: RequestCtx,
) -> Response:
    record = await volumes.get(public_id, organization_id=user.organization_id)

    # The content hash is a strong validator by construction, so a revisit can
    # be answered without reading 16 MB off disk.
    etag = f'"{record.content_hash}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": etag})

    payload = await volumes.voxels(record)
    await audit.record(
        action=AuditAction.VOLUME_VOXELS_ACCESSED,
        organization_id=user.organization_id,
        actor_id=user.id,
        resource_type="volume",
        resource_id=public_id,
        context=context,
    )
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Cache-Control": _VOXEL_CACHE_CONTROL, "ETag": etag},
    )


@router.get("/{public_id}/preview/{plane}", summary="Mid-plane JPEG preview")
async def get_preview(
    public_id: str,
    plane: str,
    user: CurrentUser,
    volumes: VolumeDep,
    storage: VolumeStorageDep,
) -> FileResponse:
    if plane not in _PREVIEW_PLANES:
        raise NotFoundError("Неизвестная плоскость просмотра.")

    record = await volumes.get(public_id, organization_id=user.organization_id)
    path = storage.preview_path(record.content_hash, plane)
    if not path.is_file():
        raise NotFoundError("Превью недоступно.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": _PREVIEW_CACHE_CONTROL},
    )


@router.patch("/{public_id}", response_model=VolumeResponse, dependencies=[ApiRateLimit])
async def update_volume(
    public_id: str,
    payload: VolumeUpdateRequest,
    user: CurrentUser,
    volumes: VolumeDep,
    context: RequestCtx,
) -> VolumeResponse:
    record = await volumes.update(
        public_id,
        actor=user,
        context=context,
        field_of_view=payload.field_of_view,
        captured_on=payload.captured_on,
        notes=payload.notes,
    )
    return present_volume(record, user.locale)


@router.post(
    "/{public_id}/analyse",
    response_model=VolumeResponse,
    dependencies=[UploadRateLimit],
    summary="Re-run the analysis pipeline over a stored volume",
)
async def reanalyse_volume(
    public_id: str,
    payload: ReanalyseRequest,
    user: CurrentUser,
    volumes: VolumeDep,
    context: RequestCtx,
) -> VolumeResponse:
    record = await volumes.reanalyse(
        public_id, actor=user, context=context, pipeline_name=payload.pipeline
    )
    return present_volume(record, user.locale)


@router.delete("/{public_id}", response_model=OkResponse, dependencies=[ApiRateLimit])
async def delete_volume(
    public_id: str,
    user: CurrentUser,
    volumes: VolumeDep,
    context: RequestCtx,
) -> OkResponse:
    await volumes.delete(public_id, actor=user, context=context)
    return OkResponse()


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------
@router.patch(
    "/{public_id}/findings/{finding_id}",
    response_model=VolumeFindingResponse,
    dependencies=[ApiRateLimit],
)
async def review_finding(
    public_id: str,
    finding_id: int,
    payload: FindingReviewRequest,
    user: CurrentUser,
    volumes: VolumeDep,
    context: RequestCtx,
) -> VolumeFindingResponse:
    finding = await volumes.review_finding(
        public_id,
        finding_id,
        actor=user,
        context=context,
        review=payload.review.value,
    )
    return present_volume_finding(finding, user.locale)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
@router.get(
    "/{public_id}/measurements",
    response_model=list[MeasurementResponse],
    dependencies=[ApiRateLimit],
)
async def list_measurements(
    public_id: str, user: CurrentUser, volumes: VolumeDep
) -> list[MeasurementResponse]:
    record = await volumes.get(public_id, organization_id=user.organization_id)
    return [present_measurement(item) for item in record.measurements]


@router.post(
    "/{public_id}/measurements",
    response_model=MeasurementResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[ApiRateLimit],
)
async def add_measurement(
    public_id: str,
    payload: MeasurementCreateRequest,
    user: CurrentUser,
    volumes: VolumeDep,
) -> MeasurementResponse:
    measurement = await volumes.add_measurement(
        public_id,
        actor=user,
        kind=payload.kind,
        plane=payload.plane,
        points=payload.points,
        label=payload.label,
        notes=payload.notes,
    )
    return present_measurement(measurement)


@router.delete(
    "/{public_id}/measurements/{measurement_id}",
    response_model=OkResponse,
    dependencies=[ApiRateLimit],
)
async def delete_measurement(
    public_id: str,
    measurement_id: int,
    user: CurrentUser,
    volumes: VolumeDep,
) -> OkResponse:
    await volumes.delete_measurement(public_id, measurement_id, actor=user)
    return OkResponse()


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
@router.get(
    "/{public_id}/annotations",
    response_model=list[AnnotationResponse],
    dependencies=[ApiRateLimit],
)
async def list_annotations(
    public_id: str, user: CurrentUser, volumes: VolumeDep
) -> list[AnnotationResponse]:
    record = await volumes.get(public_id, organization_id=user.organization_id)
    return [present_annotation(item) for item in record.annotations]


@router.post(
    "/{public_id}/annotations",
    response_model=AnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[ApiRateLimit],
)
async def add_annotation(
    public_id: str,
    payload: AnnotationCreateRequest,
    user: CurrentUser,
    volumes: VolumeDep,
) -> AnnotationResponse:
    annotation = await volumes.add_annotation(
        public_id,
        actor=user,
        kind=payload.kind,
        plane=payload.plane,
        position=(payload.x, payload.y, payload.z),
        title=payload.title,
        body=payload.body,
        finding_id=payload.volume_finding_id,
    )
    return present_annotation(annotation)


@router.delete(
    "/{public_id}/annotations/{annotation_id}",
    response_model=OkResponse,
    dependencies=[ApiRateLimit],
)
async def delete_annotation(
    public_id: str,
    annotation_id: int,
    user: CurrentUser,
    volumes: VolumeDep,
) -> OkResponse:
    await volumes.delete_annotation(public_id, annotation_id, actor=user)
    return OkResponse()
