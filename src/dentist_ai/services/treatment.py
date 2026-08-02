"""Treatment plans: the path from a set of findings to agreed work."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.clinical.charting import is_valid
from dentist_ai.clinical.protocols import Priority, by_code, procedures_for
from dentist_ai.core.errors import NotFoundError, PermissionDeniedError, ValidationError
from dentist_ai.core.ids import generate_public_id
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import (
    Finding,
    FindingReview,
    Patient,
    PlanItemStatus,
    PlanStatus,
    Study,
    TreatmentPlan,
    TreatmentPlanItem,
    User,
)
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext

#: A proposal drawn from a low-confidence detection is noise. Findings a
#: clinician has confirmed bypass this: their judgement outranks the score.
MIN_PROPOSAL_CONFIDENCE = 0.5

_OPEN_STATUSES = (
    PlanItemStatus.PROPOSED,
    PlanItemStatus.ACCEPTED,
    PlanItemStatus.SCHEDULED,
    PlanItemStatus.IN_PROGRESS,
)


@dataclass(frozen=True, slots=True)
class ItemDraft:
    procedure_code: str
    tooth_number: int | None
    priority: Priority
    estimated_visits: int
    estimated_minutes: int
    notes: str | None = None
    source_finding_id: int | None = None
    source_study_public_id: str | None = None


class TreatmentService:
    def __init__(self, session: AsyncSession, audit: AuditService) -> None:
        self._session = session
        self._audit = audit

    # -- reads ------------------------------------------------------------
    async def get(self, public_id: str, *, organization_id: int) -> TreatmentPlan:
        plan = await self._session.scalar(
            select(TreatmentPlan)
            .options(
                selectinload(TreatmentPlan.items),
                selectinload(TreatmentPlan.options),
                selectinload(TreatmentPlan.patient),
                selectinload(TreatmentPlan.created_by),
            )
            # Without this, a plan already in the identity map keeps the
            # `items` collection it was loaded with, so steps added earlier in
            # the same request are missing from the response.
            .execution_options(populate_existing=True)
            .where(
                TreatmentPlan.public_id == public_id,
                TreatmentPlan.organization_id == organization_id,
            )
        )
        if plan is None:
            raise NotFoundError("План лечения не найден.")
        return plan

    async def list_for_patient(
        self, patient_id: int, *, organization_id: int
    ) -> list[TreatmentPlan]:
        rows = await self._session.scalars(
            select(TreatmentPlan)
            .options(
                selectinload(TreatmentPlan.items),
                selectinload(TreatmentPlan.options),
                selectinload(TreatmentPlan.created_by),
            )
            .where(
                TreatmentPlan.organization_id == organization_id,
                TreatmentPlan.patient_id == patient_id,
            )
            .order_by(TreatmentPlan.created_at.desc(), TreatmentPlan.id.desc())
        )
        return list(rows.unique().all())

    # -- plans ------------------------------------------------------------
    async def create(
        self,
        *,
        actor: User,
        context: RequestContext,
        patient_id: int,
        title: str,
        notes: str | None = None,
    ) -> TreatmentPlan:
        await self._assert_patient_in_org(patient_id, actor.organization_id)

        plan = TreatmentPlan(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            patient_id=patient_id,
            created_by_id=actor.id,
            title=title,
            status=PlanStatus.DRAFT,
            notes=notes,
        )
        self._session.add(plan)
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.PLAN_CREATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="treatment_plan",
            resource_id=plan.public_id,
            context=context,
        )
        return await self.get(plan.public_id, organization_id=actor.organization_id)

    async def update(
        self,
        public_id: str,
        *,
        actor: User,
        context: RequestContext,
        title: str,
        status: PlanStatus,
        notes: str | None,
    ) -> TreatmentPlan:
        self._assert_can_plan(actor)
        plan = await self.get(public_id, organization_id=actor.organization_id)
        plan.title = title
        plan.status = status
        plan.notes = notes

        await self._audit.record(
            action=AuditAction.PLAN_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="treatment_plan",
            resource_id=plan.public_id,
            context=context,
            detail=status.value,
        )
        return plan

    async def delete(self, public_id: str, *, actor: User, context: RequestContext) -> None:
        self._assert_can_plan(actor)
        plan = await self.get(public_id, organization_id=actor.organization_id)
        await self._session.delete(plan)
        await self._audit.record(
            action=AuditAction.PLAN_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="treatment_plan",
            resource_id=public_id,
            context=context,
            detail="deleted",
        )

    # -- proposals --------------------------------------------------------
    async def propose_from_study(
        self,
        study_public_id: str,
        *,
        actor: User,
        context: RequestContext,
        plan_public_id: str | None = None,
    ) -> TreatmentPlan:
        """Draft plan items from a study's findings and append them.

        Rejected findings are ignored outright. Everything else contributes
        the procedures its class is associated with in ``clinical.protocols``,
        skipping anything the plan already covers for the same tooth.
        """
        self._assert_can_plan(actor)

        study = await self._session.scalar(
            select(Study)
            .options(selectinload(Study.findings), selectinload(Study.patient))
            .where(
                Study.public_id == study_public_id,
                Study.organization_id == actor.organization_id,
            )
        )
        if study is None:
            raise NotFoundError("Снимок не найден.")
        if study.patient_id is None:
            raise ValidationError("Привяжите снимок к пациенту, прежде чем составлять план.")

        plan = (
            await self.get(plan_public_id, organization_id=actor.organization_id)
            if plan_public_id
            else await self._open_plan_for(study.patient_id, actor=actor, context=context)
        )
        if plan.patient_id != study.patient_id:
            raise ValidationError("План принадлежит другому пациенту.")

        drafts = draft_items(study.findings, study_public_id=study.public_id)
        await self._append(plan, drafts, actor=actor, context=context)
        return await self.get(plan.public_id, organization_id=actor.organization_id)

    async def _open_plan_for(
        self, patient_id: int, *, actor: User, context: RequestContext
    ) -> TreatmentPlan:
        existing = await self._session.scalar(
            select(TreatmentPlan)
            .options(selectinload(TreatmentPlan.items))
            .where(
                TreatmentPlan.organization_id == actor.organization_id,
                TreatmentPlan.patient_id == patient_id,
                TreatmentPlan.status.in_((PlanStatus.DRAFT, PlanStatus.ACTIVE)),
            )
            .order_by(TreatmentPlan.created_at.desc())
        )
        if existing is not None:
            return existing
        return await self.create(
            actor=actor,
            context=context,
            patient_id=patient_id,
            title="План лечения",
        )

    async def _append(
        self,
        plan: TreatmentPlan,
        drafts: list[ItemDraft],
        *,
        actor: User,
        context: RequestContext,
    ) -> None:
        covered = {
            (item.procedure_code, item.tooth_number)
            for item in plan.items
            if item.status in _OPEN_STATUSES
        }
        position = max((item.position for item in plan.items), default=-1)

        added = 0
        for draft in drafts:
            key = (draft.procedure_code, draft.tooth_number)
            if key in covered:
                continue
            covered.add(key)
            position += 1
            added += 1
            self._session.add(
                TreatmentPlanItem(
                    plan_id=plan.id,
                    procedure_code=draft.procedure_code,
                    tooth_number=draft.tooth_number,
                    priority=draft.priority.value,
                    estimated_visits=draft.estimated_visits,
                    estimated_minutes=draft.estimated_minutes,
                    status=PlanItemStatus.PROPOSED,
                    position=position,
                    notes=draft.notes,
                    source_finding_id=draft.source_finding_id,
                    source_study_public_id=draft.source_study_public_id,
                )
            )

        await self._session.flush()
        if added:
            await self._audit.record(
                action=AuditAction.PLAN_ITEM_ADDED,
                organization_id=actor.organization_id,
                actor_id=actor.id,
                resource_type="treatment_plan",
                resource_id=plan.public_id,
                context=context,
                detail=f"{added} proposed",
            )

    # -- items ------------------------------------------------------------
    async def add_item(
        self,
        plan_public_id: str,
        *,
        actor: User,
        context: RequestContext,
        procedure_code: str,
        tooth_number: int | None,
        notes: str | None,
    ) -> TreatmentPlanItem:
        self._assert_can_plan(actor)
        procedure = by_code(procedure_code)
        if procedure is None:
            raise ValidationError("Неизвестная процедура.")
        if tooth_number is not None and not is_valid(tooth_number):
            raise ValidationError("Номер зуба должен быть в нотации FDI.")

        plan = await self.get(plan_public_id, organization_id=actor.organization_id)
        item = TreatmentPlanItem(
            plan_id=plan.id,
            procedure_code=procedure.code,
            tooth_number=tooth_number,
            priority=procedure.priority.value,
            estimated_visits=procedure.visits,
            estimated_minutes=procedure.minutes,
            status=PlanItemStatus.ACCEPTED,
            position=max((existing.position for existing in plan.items), default=-1) + 1,
            notes=notes,
        )
        self._session.add(item)
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.PLAN_ITEM_ADDED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="treatment_plan",
            resource_id=plan.public_id,
            context=context,
            detail=procedure.code,
        )
        return item

    async def update_item(
        self,
        plan_public_id: str,
        item_id: int,
        *,
        actor: User,
        context: RequestContext,
        status: PlanItemStatus,
        tooth_number: int | None,
        scheduled_for: date | None,
        estimated_visits: int,
        estimated_minutes: int,
        notes: str | None,
    ) -> TreatmentPlanItem:
        self._assert_can_plan(actor)
        if tooth_number is not None and not is_valid(tooth_number):
            raise ValidationError("Номер зуба должен быть в нотации FDI.")

        item = await self._get_item(plan_public_id, item_id, actor.organization_id)
        item.status = status
        item.tooth_number = tooth_number
        item.scheduled_for = scheduled_for
        item.estimated_visits = estimated_visits
        item.estimated_minutes = estimated_minutes
        item.notes = notes
        item.completed_at = utcnow() if status is PlanItemStatus.DONE else None

        await self._audit.record(
            action=AuditAction.PLAN_ITEM_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="treatment_plan_item",
            resource_id=item.id,
            context=context,
            detail=status.value,
        )
        return item

    async def remove_item(
        self,
        plan_public_id: str,
        item_id: int,
        *,
        actor: User,
        context: RequestContext,
    ) -> None:
        self._assert_can_plan(actor)
        item = await self._get_item(plan_public_id, item_id, actor.organization_id)
        await self._session.delete(item)
        await self._audit.record(
            action=AuditAction.PLAN_ITEM_REMOVED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="treatment_plan_item",
            resource_id=item_id,
            context=context,
        )

    # -- helpers ----------------------------------------------------------
    async def _get_item(
        self, plan_public_id: str, item_id: int, organization_id: int
    ) -> TreatmentPlanItem:
        item = await self._session.scalar(
            select(TreatmentPlanItem)
            .join(TreatmentPlan, TreatmentPlanItem.plan_id == TreatmentPlan.id)
            .where(
                TreatmentPlanItem.id == item_id,
                TreatmentPlan.public_id == plan_public_id,
                TreatmentPlan.organization_id == organization_id,
            )
        )
        if item is None:
            raise NotFoundError("Этап плана не найден.")
        return item

    async def _assert_patient_in_org(self, patient_id: int, organization_id: int) -> None:
        exists = await self._session.scalar(
            select(Patient.id).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if exists is None:
            raise NotFoundError("Пациент не найден.")

    @staticmethod
    def _assert_can_plan(actor: User) -> None:
        if not actor.role.can_review_findings:
            raise PermissionDeniedError("Планировать лечение может только врач.")


def draft_items(findings: list[Finding], *, study_public_id: str) -> list[ItemDraft]:
    """Turn a study's findings into plan proposals, most urgent first."""
    drafts: list[ItemDraft] = []
    for finding in findings:
        if finding.review is FindingReview.REJECTED:
            continue
        if (
            finding.review is not FindingReview.CONFIRMED
            and finding.confidence < MIN_PROPOSAL_CONFIDENCE
        ):
            continue

        for procedure in procedures_for(finding.class_key):
            drafts.append(
                ItemDraft(
                    procedure_code=procedure.code,
                    tooth_number=finding.tooth_number,
                    priority=procedure.priority,
                    estimated_visits=procedure.visits,
                    estimated_minutes=procedure.minutes,
                    source_finding_id=finding.id,
                    source_study_public_id=study_public_id,
                )
            )

    drafts.sort(key=lambda draft: (draft.priority.rank, draft.tooth_number or 99))
    return drafts
