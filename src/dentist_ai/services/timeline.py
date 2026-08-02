"""One patient's history, merged from every table that records something.

Seven sources — radiographs, CBCT volumes, surface scans, plan steps, notes,
appointments and library publications — each fetched in a single round-trip and
merged in Python. A SQL ``UNION`` over them would need every source projected
onto one column list, and the per-kind derivations that make an entry readable
(a study's worst severity, a procedure's label) live in the taxonomy and the
protocol table rather than in the database. What the union would buy is one
query instead of seven, on a page that renders once per patient visit.

The merge is what the feature *is*: a clinician asking "what has happened to
this person" should not have to read four screens and hold the ordering in
their head.

Two properties are load-bearing:

**Every instant is timezone-aware UTC.** Guaranteed by the ``UtcDateTime``
column type rather than re-checked here — read its docstring for why a naive
datetime is a real bug on SQLite and not a cosmetic one. Sorting a mix of aware
and naive values raises; sorting them after silently coercing would reorder the
history instead.

**A booked visit is not history.** ``Appointment.starts_at`` is routinely in
the future, so it sorts to the top of a reverse-chronological list and would
otherwise become the patient's "last activity". The summary bounds itself on
what has already happened and reports the next visit separately.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.clinical import protocols
from dentist_ai.clinical.labels import plan_item_status_label, scan_kind_label
from dentist_ai.core.errors import NotFoundError, PermissionDeniedError
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import (
    Appointment,
    AppointmentStatus,
    CaseEntry,
    NoteKind,
    Patient,
    PatientNote,
    PlanItemStatus,
    Scan3D,
    Study,
    TreatmentPlan,
    TreatmentPlanItem,
    User,
    Volume,
)
from dentist_ai.ml.cbct_taxonomy import by_key as volume_by_key
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale, Severity, by_id
from dentist_ai.schemas.timeline import (
    TimelineEventKind,
    appointment_status_label,
    note_kind_label,
)
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext

#: Enough history for a screen without turning it into a paginated view; the
#: summary still counts everything, so a truncated list never lies about how
#: much there is.
DEFAULT_LIMIT: Final[int] = 200

_OPEN_ITEM_STATUSES: Final[tuple[PlanItemStatus, ...]] = (
    PlanItemStatus.PROPOSED,
    PlanItemStatus.ACCEPTED,
    PlanItemStatus.SCHEDULED,
    PlanItemStatus.IN_PROGRESS,
)

_WORDING: Final[dict[str, dict[Locale, str]]] = {
    "patient_created": {"ru": "Карта заведена", "en": "Record created", "kk": "Карта ашылды"},
    "study": {"ru": "Загружен снимок", "en": "Radiograph uploaded", "kk": "Түсірілім жүктелді"},
    "volume": {"ru": "Загружена КЛКТ", "en": "CBCT uploaded", "kk": "КЛКТ жүктелді"},
    "case": {"ru": "Случай опубликован", "en": "Case published", "kk": "Жағдай жарияланды"},
    "attention": {
        "ru": "Требуют внимания",
        "en": "Needing attention",
        "kk": "Назар аударуды қажет етеді",
    },
    "quality": {"ru": "Качество съёмки", "en": "Acquisition quality", "kk": "Түсірілім сапасы"},
    "visits": {"ru": "Визитов", "en": "Visits", "kk": "Қабылдаулар"},
    "duration": {"ru": "Длительность", "en": "Duration", "kk": "Ұзақтығы"},
    "minutes": {"ru": "мин", "en": "min", "kk": "мин"},
    "triangles": {"ru": "Полигонов", "en": "Triangles", "kk": "Полигондар"},
}


def _wording(key: str, locale: Locale) -> str:
    table = _WORDING[key]
    return table.get(locale) or table[DEFAULT_LOCALE]


@dataclass(frozen=True, slots=True)
class TimelineMetric:
    label: str
    value: float
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class TimelineItem:
    kind: TimelineEventKind
    at: datetime
    title: str
    icon: str
    subtitle: str | None = None
    href: str | None = None
    severity: Severity | None = None
    metric: TimelineMetric | None = None


@dataclass(frozen=True, slots=True)
class TimelineSummary:
    total: int
    counts: dict[TimelineEventKind, int]
    first_at: datetime | None
    last_at: datetime | None
    next_appointment_at: datetime | None
    open_plan_items: int


@dataclass(frozen=True, slots=True)
class PatientTimeline:
    patient: Patient
    entries: list[TimelineItem]
    summary: TimelineSummary


class TimelineService:
    def __init__(self, session: AsyncSession, audit: AuditService) -> None:
        self._session = session
        self._audit = audit

    # -- the merged history -----------------------------------------------
    async def build(
        self,
        patient_id: int,
        *,
        organization_id: int,
        locale: Locale = DEFAULT_LOCALE,
        limit: int = DEFAULT_LIMIT,
    ) -> PatientTimeline:
        patient = await self.get_patient(patient_id, organization_id=organization_id)

        plan_items, open_plan_items = await self._plan_items(patient_id, organization_id, locale)
        entries: list[TimelineItem] = [
            TimelineItem(
                kind=TimelineEventKind.PATIENT_CREATED,
                at=patient.created_at,
                title=_wording("patient_created", locale),
                icon="user-plus",
                href=f"/app/patients/{patient.id}",
            ),
            *await self._studies(patient_id, organization_id, locale),
            *await self._volumes(patient_id, organization_id, locale),
            *await self._scans(patient_id, organization_id, locale),
            *plan_items,
            *await self._notes(patient_id, organization_id, locale),
            *await self._appointments(patient_id, organization_id, locale),
            *await self._cases(patient_id, organization_id, locale),
        ]
        entries.sort(key=lambda entry: entry.at, reverse=True)

        # Summarised before truncation: the header strip states how much
        # history exists, not how much of it fitted on the page.
        summary = _summarise(entries, open_plan_items=open_plan_items, now=utcnow())
        return PatientTimeline(patient=patient, entries=entries[:limit], summary=summary)

    async def _studies(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> list[TimelineItem]:
        rows = await self._session.scalars(
            select(Study)
            # One follow-up query for every study's findings, not one per
            # study: severity and the attention count come from the taxonomy,
            # which SQL cannot reach.
            .options(selectinload(Study.findings))
            .where(Study.organization_id == organization_id, Study.patient_id == patient_id)
        )
        items: list[TimelineItem] = []
        for study in rows.unique().all():
            classes = [by_id(finding.class_id) for finding in study.findings]
            attention = sum(1 for item in classes if item.needs_attention)
            items.append(
                TimelineItem(
                    kind=TimelineEventKind.STUDY,
                    at=study.created_at,
                    title=_wording("study", locale),
                    subtitle=study.original_filename,
                    href=f"/app/studies/{study.public_id}",
                    icon="scan",
                    severity=_worst(item.severity for item in classes),
                    metric=TimelineMetric(
                        label=_wording("attention", locale), value=float(attention)
                    ),
                )
            )
        return items

    async def _volumes(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> list[TimelineItem]:
        rows = await self._session.scalars(
            select(Volume)
            .options(selectinload(Volume.findings))
            .where(Volume.organization_id == organization_id, Volume.patient_id == patient_id)
        )
        items: list[TimelineItem] = []
        for volume in rows.unique().all():
            classes = [volume_by_key(finding.class_key) for finding in volume.findings]
            items.append(
                TimelineItem(
                    kind=TimelineEventKind.VOLUME,
                    at=volume.created_at,
                    title=_wording("volume", locale),
                    subtitle=volume.original_filename,
                    href=f"/app/volumes/{volume.public_id}",
                    icon="cube-transparent",
                    severity=_worst(item.severity for item in classes),
                    metric=(
                        None
                        if volume.quality_score is None
                        else TimelineMetric(
                            label=_wording("quality", locale),
                            value=round(volume.quality_score, 3),
                        )
                    ),
                )
            )
        return items

    async def _scans(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> list[TimelineItem]:
        rows = await self._session.scalars(
            select(Scan3D).where(
                Scan3D.organization_id == organization_id, Scan3D.patient_id == patient_id
            )
        )
        return [
            TimelineItem(
                kind=TimelineEventKind.SCAN,
                at=scan.created_at,
                title=scan_kind_label(scan.kind, locale),
                subtitle=scan.original_filename,
                href=f"/app/scans/{scan.public_id}",
                icon="cube",
                metric=TimelineMetric(
                    label=_wording("triangles", locale), value=float(scan.triangle_count)
                ),
            )
            for scan in rows.all()
        ]

    async def _plan_items(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> tuple[list[TimelineItem], int]:
        """Plan steps, and how many of them are still open.

        The open count rides back with the rows rather than being counted by a
        second query: the summary needs exactly the number this loop already
        walks past.
        """
        rows = await self._session.execute(
            select(TreatmentPlanItem, TreatmentPlan)
            .join(TreatmentPlan, TreatmentPlanItem.plan_id == TreatmentPlan.id)
            .where(
                TreatmentPlan.organization_id == organization_id,
                TreatmentPlan.patient_id == patient_id,
            )
        )
        items: list[TimelineItem] = []
        open_count = 0
        for item, plan in rows.all():
            if item.status in _OPEN_ITEM_STATUSES:
                open_count += 1
            procedure = protocols.by_code(item.procedure_code)
            tooth = f" · {item.tooth_number}" if item.tooth_number else ""
            label = procedure.label(locale) if procedure else item.procedure_code
            items.append(
                TimelineItem(
                    # A completed step is dated by its completion; an open one
                    # only exists as of the plan it was written into.
                    kind=TimelineEventKind.PLAN_ITEM,
                    at=item.completed_at or plan.created_at,
                    title=f"{label}{tooth}",
                    subtitle=f"{plan.title} · {plan_item_status_label(item.status, locale)}",
                    icon="clipboard",
                    metric=TimelineMetric(
                        label=_wording("visits", locale), value=float(item.estimated_visits)
                    ),
                )
            )
        return items, open_count

    async def _notes(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> list[TimelineItem]:
        rows = await self._session.scalars(
            select(PatientNote)
            .options(selectinload(PatientNote.author))
            .where(
                PatientNote.organization_id == organization_id,
                PatientNote.patient_id == patient_id,
            )
        )
        return [
            TimelineItem(
                kind=TimelineEventKind.NOTE,
                at=note.created_at,
                title=note_kind_label(note.kind, locale),
                subtitle=note.body,
                icon="note",
            )
            for note in rows.all()
        ]

    async def _appointments(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> list[TimelineItem]:
        rows = await self._session.scalars(
            select(Appointment).where(
                Appointment.organization_id == organization_id,
                Appointment.patient_id == patient_id,
            )
        )
        return [
            TimelineItem(
                kind=TimelineEventKind.APPOINTMENT,
                at=appointment.starts_at,
                title=appointment.title,
                subtitle=appointment_status_label(appointment.status, locale),
                icon="calendar",
                metric=TimelineMetric(
                    label=_wording("duration", locale),
                    value=float(appointment.duration_minutes),
                    unit=_wording("minutes", locale),
                ),
            )
            for appointment in rows.all()
        ]

    async def _cases(
        self, patient_id: int, organization_id: int, locale: Locale
    ) -> list[TimelineItem]:
        rows = await self._session.scalars(
            select(CaseEntry).where(
                CaseEntry.organization_id == organization_id,
                CaseEntry.patient_id == patient_id,
            )
        )
        return [
            TimelineItem(
                kind=TimelineEventKind.CASE,
                at=entry.created_at,
                title=_wording("case", locale),
                subtitle=entry.title,
                icon="book",
            )
            for entry in rows.all()
        ]

    # -- notes -------------------------------------------------------------
    async def add_note(
        self,
        patient_id: int,
        *,
        actor: User,
        context: RequestContext,
        kind: NoteKind,
        body: str,
    ) -> PatientNote:
        await self.get_patient(patient_id, organization_id=actor.organization_id)
        note = PatientNote(
            organization_id=actor.organization_id,
            patient_id=patient_id,
            author_id=actor.id,
            kind=kind,
            body=body,
        )
        self._session.add(note)
        await self._session.flush()
        await self._record_change(actor, context, "patient_note", note.id, kind.value)
        return note

    async def delete_note(
        self, patient_id: int, note_id: int, *, actor: User, context: RequestContext
    ) -> None:
        note = await self._session.scalar(
            select(PatientNote).where(
                PatientNote.id == note_id,
                PatientNote.patient_id == patient_id,
                PatientNote.organization_id == actor.organization_id,
            )
        )
        if note is None:
            raise NotFoundError("Заметка не найдена.")
        _assert_owns(
            note.author_id, actor, "Удалить заметку может только автор или владелец клиники."
        )

        await self._session.delete(note)
        await self._record_change(actor, context, "patient_note", note_id, "deleted")

    # -- appointments ------------------------------------------------------
    async def add_appointment(
        self,
        patient_id: int,
        *,
        actor: User,
        context: RequestContext,
        title: str,
        starts_at: datetime,
        duration_minutes: int,
        plan_item_id: int | None = None,
        notes: str | None = None,
    ) -> Appointment:
        await self.get_patient(patient_id, organization_id=actor.organization_id)
        if plan_item_id is not None:
            await self._assert_plan_item_for(plan_item_id, patient_id, actor.organization_id)

        appointment = Appointment(
            organization_id=actor.organization_id,
            patient_id=patient_id,
            created_by_id=actor.id,
            plan_item_id=plan_item_id,
            title=title,
            starts_at=starts_at,
            duration_minutes=duration_minutes,
            status=AppointmentStatus.SCHEDULED,
            notes=notes,
        )
        self._session.add(appointment)
        await self._session.flush()
        await self._record_change(actor, context, "appointment", appointment.id, "scheduled")
        return await self._get_appointment(appointment.id, patient_id, actor.organization_id)

    async def update_appointment(
        self,
        patient_id: int,
        appointment_id: int,
        *,
        actor: User,
        context: RequestContext,
        title: str | None = None,
        starts_at: datetime | None = None,
        duration_minutes: int | None = None,
        status: AppointmentStatus | None = None,
        notes: str | None = None,
    ) -> Appointment:
        appointment = await self._get_appointment(appointment_id, patient_id, actor.organization_id)
        # Reception confirms and cancels bookings, so this is open to every
        # member; only removing the row is restricted.
        if title is not None:
            appointment.title = title
        if starts_at is not None:
            appointment.starts_at = starts_at
        if duration_minutes is not None:
            appointment.duration_minutes = duration_minutes
        if status is not None:
            appointment.status = status
        if notes is not None:
            appointment.notes = notes or None

        await self._record_change(
            actor, context, "appointment", appointment_id, (status or appointment.status).value
        )
        return appointment

    async def delete_appointment(
        self, patient_id: int, appointment_id: int, *, actor: User, context: RequestContext
    ) -> None:
        appointment = await self._get_appointment(appointment_id, patient_id, actor.organization_id)
        _assert_owns(
            appointment.created_by_id,
            actor,
            "Удалить приём может только тот, кто его создал, или владелец клиники.",
        )
        await self._session.delete(appointment)
        await self._record_change(actor, context, "appointment", appointment_id, "deleted")

    async def list_appointments(
        self, patient_id: int, *, organization_id: int
    ) -> list[Appointment]:
        await self.get_patient(patient_id, organization_id=organization_id)
        rows = await self._session.scalars(
            select(Appointment)
            .options(selectinload(Appointment.created_by))
            .where(
                Appointment.organization_id == organization_id,
                Appointment.patient_id == patient_id,
            )
            .order_by(Appointment.starts_at.desc(), Appointment.id.desc())
        )
        return list(rows.all())

    # -- helpers -----------------------------------------------------------
    async def get_patient(self, patient_id: int, *, organization_id: int) -> Patient:
        patient = await self._session.scalar(
            select(Patient).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if patient is None:
            # 404 rather than 403 for another clinic's chart: the response must
            # not confirm that the record exists.
            raise NotFoundError("Пациент не найден.")
        return patient

    async def _get_appointment(
        self, appointment_id: int, patient_id: int, organization_id: int
    ) -> Appointment:
        appointment = await self._session.scalar(
            select(Appointment)
            .options(selectinload(Appointment.created_by))
            .execution_options(populate_existing=True)
            .where(
                Appointment.id == appointment_id,
                Appointment.patient_id == patient_id,
                Appointment.organization_id == organization_id,
            )
        )
        if appointment is None:
            raise NotFoundError("Приём не найден.")
        return appointment

    async def _assert_plan_item_for(
        self, plan_item_id: int, patient_id: int, organization_id: int
    ) -> None:
        exists = await self._session.scalar(
            select(TreatmentPlanItem.id)
            .join(TreatmentPlan, TreatmentPlanItem.plan_id == TreatmentPlan.id)
            .where(
                TreatmentPlanItem.id == plan_item_id,
                TreatmentPlan.patient_id == patient_id,
                TreatmentPlan.organization_id == organization_id,
            )
        )
        if exists is None:
            raise NotFoundError("Этап плана не найден.")

    async def _record_change(
        self,
        actor: User,
        context: RequestContext,
        resource_type: str,
        resource_id: int,
        detail: str,
    ) -> None:
        """Audit a timeline write.

        Filed under ``PATIENT_UPDATED``: a note or a booking is a change to the
        patient's record, and ``AuditAction`` has no entry of its own for
        either. ``resource_type`` still names what actually changed, so the
        trail does not lose the distinction.
        """
        await self._audit.record(
            action=AuditAction.PATIENT_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type=resource_type,
            resource_id=resource_id,
            context=context,
            detail=detail,
        )


def _assert_owns(author_id: int | None, actor: User, message: str) -> None:
    """Author-or-owner, in the shape of ``UserRole.can_delete_patients``.

    A coarse role property rather than a permission table: an owner can clear
    anything in their clinic, everyone else can clear only what they wrote.
    """
    if author_id != actor.id and not actor.role.can_manage_members:
        raise PermissionDeniedError(message)


def _worst(severities: Iterable[Severity]) -> Severity | None:
    return min(severities, key=lambda severity: severity.rank, default=None)


def _summarise(
    entries: list[TimelineItem], *, open_plan_items: int, now: datetime
) -> TimelineSummary:
    counts: Counter[TimelineEventKind] = Counter(entry.kind for entry in entries)
    happened = [entry.at for entry in entries if entry.at <= now]
    upcoming = sorted(
        entry.at
        for entry in entries
        if entry.kind is TimelineEventKind.APPOINTMENT and entry.at > now
    )
    return TimelineSummary(
        total=len(entries),
        counts=dict(counts),
        first_at=min(happened, default=None),
        last_at=max(happened, default=None),
        next_appointment_at=upcoming[0] if upcoming else None,
        open_plan_items=open_plan_items,
    )
