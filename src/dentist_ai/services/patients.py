"""Patient records, always scoped to the caller's organisation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import ColumnElement, Select, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from dentist_ai.core.errors import ConflictError, NotFoundError
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import (
    Patient,
    PlanItemStatus,
    Scan3D,
    Study,
    TreatmentPlan,
    TreatmentPlanItem,
    User,
)
from dentist_ai.schemas.clinical import PatientCreateRequest, PatientUpdateRequest
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext

#: Below this, a digit run is more likely a chart number fragment than a
#: phone number, and matching on it would return noise.
_MIN_PHONE_DIGITS = 3

_OPEN_ITEM_STATUSES = (
    PlanItemStatus.PROPOSED,
    PlanItemStatus.ACCEPTED,
    PlanItemStatus.SCHEDULED,
    PlanItemStatus.IN_PROGRESS,
)


@dataclass(frozen=True, slots=True)
class PatientRow:
    patient: Patient
    study_count: int
    last_study_at: datetime | None
    scan_count: int = 0
    open_plan_items: int = 0


class PatientService:
    def __init__(self, session: AsyncSession, audit: AuditService) -> None:
        self._session = session
        self._audit = audit

    async def list_patients(
        self,
        *,
        organization_id: int,
        query: str | None = None,
        include_archived: bool = False,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[PatientRow], int]:
        """Paginated search. Returns the page plus the unpaginated total."""
        base = self._scoped(organization_id, include_archived=include_archived)
        if query:
            base = base.where(self._search_predicate(query))

        total = await self._session.scalar(select(func.count()).select_from(base.subquery()))

        # Every per-row count is aggregated in the same round-trip; one query
        # per row would make the table cost scale with the page size.
        studies = (
            select(
                Study.patient_id.label("patient_id"),
                func.count(Study.id).label("study_count"),
                func.max(Study.created_at).label("last_study_at"),
            )
            .where(Study.organization_id == organization_id)
            .group_by(Study.patient_id)
            .subquery()
        )
        scans = (
            select(
                Scan3D.patient_id.label("patient_id"),
                func.count(Scan3D.id).label("scan_count"),
            )
            .where(Scan3D.organization_id == organization_id)
            .group_by(Scan3D.patient_id)
            .subquery()
        )
        plan_items = (
            select(
                TreatmentPlan.patient_id.label("patient_id"),
                func.count(TreatmentPlanItem.id).label("open_items"),
            )
            .join(TreatmentPlanItem, TreatmentPlanItem.plan_id == TreatmentPlan.id)
            .where(
                TreatmentPlan.organization_id == organization_id,
                TreatmentPlanItem.status.in_(_OPEN_ITEM_STATUSES),
            )
            .group_by(TreatmentPlan.patient_id)
            .subquery()
        )

        rows = await self._session.execute(
            base.add_columns(
                func.coalesce(studies.c.study_count, 0),
                studies.c.last_study_at,
                func.coalesce(scans.c.scan_count, 0),
                func.coalesce(plan_items.c.open_items, 0),
            )
            .outerjoin(studies, studies.c.patient_id == Patient.id)
            .outerjoin(scans, scans.c.patient_id == Patient.id)
            .outerjoin(plan_items, plan_items.c.patient_id == Patient.id)
            .order_by(Patient.created_at.desc(), Patient.id.desc())
            .limit(limit)
            .offset(offset)
        )

        return (
            [
                PatientRow(patient, count, last, scan_total, open_items)
                for patient, count, last, scan_total, open_items in rows.all()
            ],
            int(total or 0),
        )

    async def get(
        self,
        patient_id: int,
        *,
        organization_id: int,
        actor: User | None = None,
        context: RequestContext | None = None,
    ) -> Patient:
        patient = await self._session.scalar(
            select(Patient).where(
                Patient.id == patient_id,
                Patient.organization_id == organization_id,
            )
        )
        if patient is None:
            # A 404 rather than a 403 for cross-tenant ids: the response must
            # not confirm that some other clinic's record exists.
            raise NotFoundError("Пациент не найден.")

        if actor is not None:
            await self._audit.record(
                action=AuditAction.PATIENT_VIEWED,
                organization_id=organization_id,
                actor_id=actor.id,
                resource_type="patient",
                resource_id=patient.id,
                context=context,
            )
        return patient

    async def create(
        self,
        payload: PatientCreateRequest,
        *,
        actor: User,
        context: RequestContext,
    ) -> Patient:
        patient = Patient(
            organization_id=actor.organization_id,
            full_name=payload.full_name,
            phone=payload.phone,
            email=payload.email,
            date_of_birth=payload.date_of_birth,
            sex=payload.sex,
            medical_record_number=payload.medical_record_number,
            notes=payload.notes,
        )
        patient.refresh_search_text()
        self._session.add(patient)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConflictError("Пациент с таким номером карты уже существует.") from exc

        await self._audit.record(
            action=AuditAction.PATIENT_CREATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="patient",
            resource_id=patient.id,
            context=context,
        )
        return patient

    async def update(
        self,
        patient_id: int,
        payload: PatientUpdateRequest,
        *,
        actor: User,
        context: RequestContext,
    ) -> Patient:
        patient = await self.get(patient_id, organization_id=actor.organization_id)
        patient.full_name = payload.full_name
        patient.phone = payload.phone
        patient.email = payload.email
        patient.date_of_birth = payload.date_of_birth
        patient.sex = payload.sex
        patient.medical_record_number = payload.medical_record_number
        patient.notes = payload.notes
        patient.refresh_search_text()

        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            raise ConflictError("Пациент с таким номером карты уже существует.") from exc

        await self._audit.record(
            action=AuditAction.PATIENT_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="patient",
            resource_id=patient.id,
            context=context,
        )
        return patient

    async def archive(self, patient_id: int, *, actor: User, context: RequestContext) -> Patient:
        """Soft delete: keeps studies and the audit trail intact."""
        patient = await self.get(patient_id, organization_id=actor.organization_id)
        patient.archived_at = utcnow()
        await self._audit.record(
            action=AuditAction.PATIENT_ARCHIVED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="patient",
            resource_id=patient.id,
            context=context,
        )
        return patient

    async def restore(self, patient_id: int, *, actor: User) -> Patient:
        patient = await self.get(patient_id, organization_id=actor.organization_id)
        patient.archived_at = None
        return patient

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _scoped(organization_id: int, *, include_archived: bool) -> Select[tuple[Patient]]:
        statement = select(Patient).where(Patient.organization_id == organization_id)
        if not include_archived:
            statement = statement.where(Patient.archived_at.is_(None))
        return statement

    @staticmethod
    def _search_predicate(query: str) -> ColumnElement[bool]:
        """Match against the pre-folded ``search_text`` column.

        Folding happens in Python on write (see ``Patient.refresh_search_text``):
        SQL ``lower()`` is ASCII-only on SQLite and never matches a Cyrillic
        name there.

        A leading wildcard cannot use the btree index, so this scans within the
        organisation. The upgrade path is a pg_trgm GIN index — see
        docs/ARCHITECTURE.md.
        """
        # Digits-only input is a phone search: strip formatting so
        # "+7 701" matches a number stored as "+77011111111".
        needle = query.strip().lower()
        compact = "".join(char for char in needle if char.isdigit())
        patterns = [f"%{needle}%"]
        if len(compact) >= _MIN_PHONE_DIGITS:
            patterns.append(f"%{compact}%")
        return or_(*(Patient.search_text.like(pattern) for pattern in patterns))
