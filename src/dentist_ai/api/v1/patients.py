"""Patient CRUD."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status

from dentist_ai.api.deps import (
    ApiRateLimit,
    CurrentUser,
    PatientDep,
    RequestCtx,
    ScanDep,
    StudyDep,
    TreatmentDep,
)
from dentist_ai.api.presenters import (
    build_timeline,
    present_patient,
    present_patient_row,
    present_plan,
    present_scan,
    present_study_row,
)
from dentist_ai.schemas.clinical import (
    PatientCreateRequest,
    PatientResponse,
    PatientSummaryResponse,
    PatientUpdateRequest,
)
from dentist_ai.schemas.common import OkResponse, Page, PageMeta
from dentist_ai.schemas.overview import PatientOverviewResponse
from dentist_ai.services.patients import PatientRow

router = APIRouter(prefix="/patients", tags=["patients"], dependencies=[ApiRateLimit])

MAX_PAGE_SIZE = 100
#: Enough history for a patient page without turning it into a paginated view.
MAX_TIMELINE_ROWS = 100


@router.get("", response_model=Page[PatientSummaryResponse], summary="List patients")
async def list_patients(
    user: CurrentUser,
    patients: PatientDep,
    q: Annotated[str | None, Query(max_length=120, description="Name, phone or chart no.")] = None,
    include_archived: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[PatientSummaryResponse]:
    rows, total = await patients.list_patients(
        organization_id=user.organization_id,
        query=q,
        include_archived=include_archived,
        limit=limit,
        offset=offset,
    )
    return Page(
        items=[
            present_patient_row(row, scan_count=row.scan_count, open_plan_items=row.open_plan_items)
            for row in rows
        ],
        meta=PageMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(rows) < total,
        ),
    )


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a patient",
)
async def create_patient(
    payload: PatientCreateRequest,
    user: CurrentUser,
    patients: PatientDep,
    context: RequestCtx,
) -> PatientResponse:
    patient = await patients.create(payload, actor=user, context=context)
    return present_patient(patient)


@router.get("/{patient_id}", response_model=PatientResponse, summary="Get a patient")
async def get_patient(
    patient_id: int,
    user: CurrentUser,
    patients: PatientDep,
    context: RequestCtx,
) -> PatientResponse:
    patient = await patients.get(
        patient_id, organization_id=user.organization_id, actor=user, context=context
    )
    return present_patient(patient)


@router.put("/{patient_id}", response_model=PatientResponse, summary="Update a patient")
async def update_patient(
    patient_id: int,
    payload: PatientUpdateRequest,
    user: CurrentUser,
    patients: PatientDep,
    context: RequestCtx,
) -> PatientResponse:
    patient = await patients.update(patient_id, payload, actor=user, context=context)
    return present_patient(patient)


@router.delete("/{patient_id}", response_model=OkResponse, summary="Archive a patient")
async def archive_patient(
    patient_id: int,
    user: CurrentUser,
    patients: PatientDep,
    context: RequestCtx,
) -> OkResponse:
    await patients.archive(patient_id, actor=user, context=context)
    return OkResponse()


@router.post(
    "/{patient_id}/restore", response_model=PatientResponse, summary="Restore an archived patient"
)
async def restore_patient(
    patient_id: int,
    user: CurrentUser,
    patients: PatientDep,
) -> PatientResponse:
    patient = await patients.restore(patient_id, actor=user)
    return present_patient(patient)


@router.get(
    "/{patient_id}/overview",
    response_model=PatientOverviewResponse,
    summary="Studies, scans, plans and a merged timeline",
)
async def patient_overview(
    patient_id: int,
    user: CurrentUser,
    patients: PatientDep,
    studies: StudyDep,
    scans: ScanDep,
    treatment: TreatmentDep,
    context: RequestCtx,
) -> PatientOverviewResponse:
    patient = await patients.get(
        patient_id, organization_id=user.organization_id, actor=user, context=context
    )
    study_rows, study_total = await studies.list_studies(
        organization_id=user.organization_id, patient_id=patient_id, limit=MAX_TIMELINE_ROWS
    )
    scan_rows, scan_total = await scans.list_for_patient(
        patient_id, organization_id=user.organization_id, limit=MAX_TIMELINE_ROWS
    )
    plans = await treatment.list_for_patient(patient_id, organization_id=user.organization_id)

    presented_studies = [present_study_row(row, user.locale) for row in study_rows]
    presented_scans = [present_scan(scan, user.locale) for scan in scan_rows]
    presented_plans = [present_plan(plan, user.locale) for plan in plans]

    summary = present_patient_row(
        PatientRow(
            patient=patient,
            study_count=study_total,
            last_study_at=study_rows[0].study.created_at if study_rows else None,
        ),
        scan_count=scan_total,
        open_plan_items=sum(plan.open_count for plan in presented_plans),
    )

    return PatientOverviewResponse(
        patient=summary,
        studies=presented_studies,
        scans=presented_scans,
        plans=presented_plans,
        timeline=build_timeline(
            patient, presented_studies, presented_scans, presented_plans, user.locale
        ),
    )
