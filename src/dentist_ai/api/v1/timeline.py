"""The merged patient history, plus the notes and appointments it owns."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from dentist_ai.api.deps import ApiRateLimit, AuditDep, CurrentUser, DbSession, RequestCtx
from dentist_ai.db.models import Appointment, PatientNote
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale
from dentist_ai.schemas.common import OkResponse
from dentist_ai.schemas.timeline import (
    AppointmentCreateRequest,
    AppointmentResponse,
    AppointmentUpdateRequest,
    NoteCreateRequest,
    PatientNoteResponse,
    PatientTimelineResponse,
    TimelineEventResponse,
    TimelineKindCount,
    TimelineMetricResponse,
    TimelineSummaryResponse,
    appointment_status_label,
    note_kind_label,
    timeline_kind_label,
)
from dentist_ai.services.timeline import (
    DEFAULT_LIMIT,
    TimelineItem,
    TimelineService,
    TimelineSummary,
)


# The composition root for this feature. It sits beside the routes rather than
# in `api/deps.py` only because the feature is self-contained; nothing else
# constructs a `TimelineService`.
def get_timeline_service(session: DbSession, audit: AuditDep) -> TimelineService:
    return TimelineService(session, audit)


type TimelineDep = Annotated[TimelineService, Depends(get_timeline_service)]

router = APIRouter(prefix="/timeline", tags=["timeline"], dependencies=[ApiRateLimit])


@router.get(
    "/{patient_id}",
    response_model=PatientTimelineResponse,
    summary="Everything that has happened to one patient",
)
async def get_timeline(
    patient_id: int,
    user: CurrentUser,
    timeline: TimelineDep,
    limit: Annotated[int, Query(ge=1, le=DEFAULT_LIMIT)] = DEFAULT_LIMIT,
) -> PatientTimelineResponse:
    result = await timeline.build(
        patient_id,
        organization_id=user.organization_id,
        locale=user.locale,
        limit=limit,
    )
    return PatientTimelineResponse(
        patient_id=result.patient.id,
        patient_name=result.patient.full_name,
        summary=_present_summary(result.summary, user.locale),
        entries=[_present_entry(entry) for entry in result.entries],
    )


@router.post(
    "/{patient_id}/notes",
    response_model=PatientNoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a note to the patient's record",
)
async def add_note(
    patient_id: int,
    payload: NoteCreateRequest,
    user: CurrentUser,
    timeline: TimelineDep,
    context: RequestCtx,
) -> PatientNoteResponse:
    note = await timeline.add_note(
        patient_id,
        actor=user,
        context=context,
        kind=payload.kind,
        body=payload.body,
    )
    return _present_note(note, author_name=user.full_name, locale=user.locale)


@router.delete("/{patient_id}/notes/{note_id}", response_model=OkResponse, summary="Delete a note")
async def delete_note(
    patient_id: int,
    note_id: int,
    user: CurrentUser,
    timeline: TimelineDep,
    context: RequestCtx,
) -> OkResponse:
    await timeline.delete_note(patient_id, note_id, actor=user, context=context)
    return OkResponse()


@router.get(
    "/{patient_id}/appointments",
    response_model=list[AppointmentResponse],
    summary="Booked and past visits",
)
async def list_appointments(
    patient_id: int,
    user: CurrentUser,
    timeline: TimelineDep,
) -> list[AppointmentResponse]:
    rows = await timeline.list_appointments(patient_id, organization_id=user.organization_id)
    return [_present_appointment(row, user.locale) for row in rows]


@router.post(
    "/{patient_id}/appointments",
    response_model=AppointmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Book a visit",
)
async def add_appointment(
    patient_id: int,
    payload: AppointmentCreateRequest,
    user: CurrentUser,
    timeline: TimelineDep,
    context: RequestCtx,
) -> AppointmentResponse:
    appointment = await timeline.add_appointment(
        patient_id,
        actor=user,
        context=context,
        title=payload.title,
        starts_at=payload.starts_at,
        duration_minutes=payload.duration_minutes,
        plan_item_id=payload.plan_item_id,
        notes=payload.notes,
    )
    return _present_appointment(appointment, user.locale)


@router.patch(
    "/{patient_id}/appointments/{appointment_id}",
    response_model=AppointmentResponse,
    summary="Reschedule, confirm or cancel a visit",
)
async def update_appointment(
    patient_id: int,
    appointment_id: int,
    payload: AppointmentUpdateRequest,
    user: CurrentUser,
    timeline: TimelineDep,
    context: RequestCtx,
) -> AppointmentResponse:
    appointment = await timeline.update_appointment(
        patient_id,
        appointment_id,
        actor=user,
        context=context,
        title=payload.title,
        starts_at=payload.starts_at,
        duration_minutes=payload.duration_minutes,
        status=payload.status,
        notes=payload.notes,
    )
    return _present_appointment(appointment, user.locale)


@router.delete(
    "/{patient_id}/appointments/{appointment_id}",
    response_model=OkResponse,
    summary="Delete a visit",
)
async def delete_appointment(
    patient_id: int,
    appointment_id: int,
    user: CurrentUser,
    timeline: TimelineDep,
    context: RequestCtx,
) -> OkResponse:
    await timeline.delete_appointment(patient_id, appointment_id, actor=user, context=context)
    return OkResponse()


def _present_entry(entry: TimelineItem) -> TimelineEventResponse:
    return TimelineEventResponse(
        kind=entry.kind,
        at=entry.at,
        title=entry.title,
        subtitle=entry.subtitle,
        href=entry.href,
        icon=entry.icon,
        severity=entry.severity,
        metric=(
            None
            if entry.metric is None
            else TimelineMetricResponse(
                label=entry.metric.label, value=entry.metric.value, unit=entry.metric.unit
            )
        ),
    )


def _present_summary(
    summary: TimelineSummary, locale: Locale = DEFAULT_LOCALE
) -> TimelineSummaryResponse:
    return TimelineSummaryResponse(
        total=summary.total,
        counts=[
            TimelineKindCount(kind=kind, label=timeline_kind_label(kind, locale), count=count)
            for kind, count in sorted(summary.counts.items(), key=lambda pair: pair[0].value)
        ],
        first_at=summary.first_at,
        last_at=summary.last_at,
        next_appointment_at=summary.next_appointment_at,
        open_plan_items=summary.open_plan_items,
    )


def _present_note(
    note: PatientNote, *, author_name: str | None, locale: Locale = DEFAULT_LOCALE
) -> PatientNoteResponse:
    return PatientNoteResponse(
        id=note.id,
        kind=note.kind,
        kind_label=note_kind_label(note.kind, locale),
        body=note.body,
        author_id=note.author_id,
        author_name=author_name,
        created_at=note.created_at,
    )


def _present_appointment(
    appointment: Appointment, locale: Locale = DEFAULT_LOCALE
) -> AppointmentResponse:
    return AppointmentResponse(
        id=appointment.id,
        title=appointment.title,
        starts_at=appointment.starts_at,
        duration_minutes=appointment.duration_minutes,
        status=appointment.status,
        status_label=appointment_status_label(appointment.status, locale),
        notes=appointment.notes,
        plan_item_id=appointment.plan_item_id,
        created_at=appointment.created_at,
        created_by_name=(appointment.created_by.full_name if appointment.created_by else None),
    )
