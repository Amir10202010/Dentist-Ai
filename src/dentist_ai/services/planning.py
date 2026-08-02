"""Persisting a proposed plan, and accepting one of its options.

The planner in ``clinical/treatment_planner.py`` is pure — it takes findings
and returns a proposal. This module is the part that reads the findings out of
the database and writes the result back, and it exists separately for the
reason the whole ``clinical/`` package exists: the decision logic has to be
readable without a session in scope.

The one behaviour worth stating is what "generate" means here. It writes a
**draft** plan with its options attached and no items. Nothing is scheduled,
nothing is priced, and nothing appears in the patient's open work until a
clinician picks an option — at which point that option's steps become plan
items and the plan becomes active. A generated plan that silently entered the
schedule would be the product making a treatment decision, which is the one
thing it must never do.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.clinical import protocols, treatment_planner
from dentist_ai.clinical.treatment_planner import (
    FindingSource,
    PlannedFinding,
    ProposedPlan,
)
from dentist_ai.core.errors import ConflictError, NotFoundError, ValidationError
from dentist_ai.core.ids import generate_public_id
from dentist_ai.core.logging import get_logger
from dentist_ai.db.models import (
    Finding,
    FindingReview,
    Patient,
    PlanItemStatus,
    PlanOrigin,
    PlanStatus,
    Study,
    TreatmentApproach,
    TreatmentOption,
    TreatmentPlan,
    TreatmentPlanItem,
    User,
    Volume,
    VolumeFinding,
)
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext

log = get_logger(__name__)


class PlanningService:
    def __init__(self, session: AsyncSession, audit: AuditService) -> None:
        self._session = session
        self._audit = audit

    async def generate(
        self,
        patient_id: int,
        *,
        actor: User,
        context: RequestContext,
        volume_public_id: str | None = None,
        study_public_id: str | None = None,
    ) -> TreatmentPlan:
        """Propose a plan for a patient from their reviewed findings.

        Draws on *both* examinations by default, because that is what a
        clinician does: the CBCT places a lesion and the panoramic history
        shows what has already been treated. Narrowing to one record is
        available for the case where a plan should rest on a single study.
        """
        patient = await self._patient(patient_id, actor.organization_id)
        radiograph, volume = await self._findings_for(
            patient.id,
            actor.organization_id,
            volume_public_id=volume_public_id,
            study_public_id=study_public_id,
        )
        if not radiograph and not volume:
            raise ValidationError(
                "Нет находок, на основании которых можно составить план. "
                "Загрузите снимок или КЛКТ и дождитесь анализа."
            )

        quality = await self._quality_for(volume_public_id, actor.organization_id)
        proposal = treatment_planner.propose(
            [*radiograph, *volume], locale=actor.locale, quality_score=quality
        )
        if proposal is None:
            raise ValidationError(
                "Найденные изменения не требуют лечения — план не составлен. "
                "Все находки носят анатомический или регистрационный характер."
            )

        plan = await self._persist(
            proposal,
            patient=patient,
            actor=actor,
            volume_public_id=volume_public_id,
            study_public_id=study_public_id,
        )
        await self._audit.record(
            action=AuditAction.PLAN_CREATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="plan",
            resource_id=plan.public_id,
            context=context,
            detail=f"generated:{len(proposal.options)} options",
        )
        log.info(
            "plan_generated",
            plan=plan.public_id,
            patient=patient.id,
            options=len(proposal.options),
            weeks=proposal.estimated_weeks,
        )
        return await self.get(plan.public_id, organization_id=actor.organization_id)

    async def accept_option(
        self,
        public_id: str,
        approach: TreatmentApproach,
        *,
        actor: User,
        context: RequestContext,
    ) -> TreatmentPlan:
        """Turn one option's steps into the plan's actual work.

        This is the moment a proposal becomes a commitment, so it is explicit
        and it is audited. Re-accepting is refused rather than silently
        replacing the items: a clinician who has already edited the schedule
        would lose that work, and "start again" is a delete plus a generate.
        """
        plan = await self.get(public_id, organization_id=actor.organization_id)
        if plan.items:
            raise ConflictError(
                "В плане уже есть этапы. Удалите план и сформируйте заново, "
                "чтобы выбрать другой вариант."
            )

        option = next((item for item in plan.options if item.approach is approach), None)
        if option is None:
            raise NotFoundError("Такого варианта лечения в плане нет.")

        for position, (code, tooth) in enumerate(_decode_steps(option.procedure_codes)):
            procedure = protocols.by_code(code)
            if procedure is None:
                continue
            self._session.add(
                TreatmentPlanItem(
                    plan_id=plan.id,
                    procedure_code=procedure.code,
                    tooth_number=tooth,
                    # Copied from the protocol rather than read back later, so
                    # an agreed plan is not silently re-priced by a later edit
                    # to the table.
                    priority=procedure.priority.value,
                    estimated_visits=procedure.visits,
                    estimated_minutes=procedure.minutes,
                    status=PlanItemStatus.PROPOSED,
                    position=position,
                )
            )

        for item in plan.options:
            item.is_selected = item.approach is approach
        plan.status = PlanStatus.ACTIVE
        plan.complexity = option.complexity
        plan.estimated_weeks = option.estimated_weeks
        await self._session.flush()

        # The identity map still holds the collection loaded before the insert,
        # so a re-read returns the plan with no items and the caller sees an
        # empty plan it has just filled. Expiring forces the reload below to
        # fetch them.
        self._session.expire(plan, ["items", "options"])

        await self._audit.record(
            action=AuditAction.PLAN_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="plan",
            resource_id=plan.public_id,
            context=context,
            detail=f"accepted:{approach.value}",
        )
        return await self.get(public_id, organization_id=actor.organization_id)

    async def get(self, public_id: str, *, organization_id: int) -> TreatmentPlan:
        plan = await self._session.scalar(
            select(TreatmentPlan)
            .options(
                selectinload(TreatmentPlan.items),
                selectinload(TreatmentPlan.options),
                selectinload(TreatmentPlan.patient),
                selectinload(TreatmentPlan.created_by),
            )
            .where(
                TreatmentPlan.public_id == public_id,
                TreatmentPlan.organization_id == organization_id,
            )
        )
        if plan is None:
            raise NotFoundError("План лечения не найден.")
        return plan

    # -- internals --------------------------------------------------------
    async def _persist(
        self,
        proposal: ProposedPlan,
        *,
        patient: Patient,
        actor: User,
        volume_public_id: str | None,
        study_public_id: str | None,
    ) -> TreatmentPlan:
        source_volume_id: int | None = None
        if volume_public_id is not None:
            source_volume_id = await self._session.scalar(
                select(Volume.id).where(
                    Volume.public_id == volume_public_id,
                    Volume.organization_id == actor.organization_id,
                )
            )

        plan = TreatmentPlan(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            patient_id=patient.id,
            created_by_id=actor.id,
            title=proposal.title,
            # A draft until a clinician picks an option. Nothing is scheduled
            # and nothing counts as open work in the meantime.
            status=PlanStatus.DRAFT,
            origin=PlanOrigin.GENERATED,
            complexity=proposal.complexity,
            estimated_weeks=proposal.estimated_weeks,
            risks="\n".join(proposal.risks) or None,
            follow_up=proposal.follow_up,
            rationale=proposal.rationale,
            source_volume_id=source_volume_id,
            source_study_public_id=study_public_id,
        )
        self._session.add(plan)
        await self._session.flush()

        for position, option in enumerate(proposal.options):
            self._session.add(
                TreatmentOption(
                    plan_id=plan.id,
                    position=position,
                    title=option.title,
                    approach=option.approach,
                    summary=option.summary,
                    priority=option.priority.value,
                    complexity=option.complexity,
                    estimated_visits=option.visits,
                    estimated_minutes=option.minutes,
                    estimated_weeks=option.weeks,
                    benefits=option.benefits,
                    risks=option.risks,
                    procedure_codes=option.encoded_steps(),
                    is_selected=False,
                )
            )
        await self._session.flush()
        return plan

    async def _patient(self, patient_id: int, organization_id: int) -> Patient:
        patient = await self._session.scalar(
            select(Patient).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if patient is None:
            raise NotFoundError("Пациент не найден.")
        return patient

    async def _findings_for(
        self,
        patient_id: int,
        organization_id: int,
        *,
        volume_public_id: str | None,
        study_public_id: str | None,
    ) -> tuple[list[PlannedFinding], list[PlannedFinding]]:
        """Load findings for the planner, excluding anything a clinician rejected.

        A rejected finding is a clinician saying the model was wrong, so
        planning work from it would be the product overruling them.
        """
        radiograph_query = (
            select(Finding)
            .join(Study, Study.id == Finding.study_id)
            .where(
                Study.organization_id == organization_id,
                Study.patient_id == patient_id,
                Finding.review != FindingReview.REJECTED,
            )
        )
        if study_public_id is not None:
            radiograph_query = radiograph_query.where(Study.public_id == study_public_id)

        volume_query = (
            select(VolumeFinding)
            .join(Volume, Volume.id == VolumeFinding.volume_id)
            .where(
                Volume.organization_id == organization_id,
                Volume.patient_id == patient_id,
                VolumeFinding.review != FindingReview.REJECTED,
            )
        )
        if volume_public_id is not None:
            volume_query = volume_query.where(Volume.public_id == volume_public_id)

        radiograph_rows = list(await self._session.scalars(radiograph_query))
        volume_rows = list(await self._session.scalars(volume_query))

        return (
            [
                PlannedFinding(
                    class_key=row.class_key,
                    source=FindingSource.RADIOGRAPH,
                    confidence=row.confidence,
                    tooth_number=row.tooth_number,
                    confirmed=row.review is FindingReview.CONFIRMED,
                )
                for row in radiograph_rows
            ],
            [
                PlannedFinding(
                    class_key=row.class_key,
                    source=FindingSource.CBCT,
                    confidence=row.confidence,
                    tooth_number=row.tooth_number,
                    region=row.region,
                    extent_mm=row.extent_mm,
                    confirmed=row.review is FindingReview.CONFIRMED,
                )
                for row in volume_rows
            ],
        )

    async def _quality_for(
        self, volume_public_id: str | None, organization_id: int
    ) -> float | None:
        if volume_public_id is None:
            return None
        return await self._session.scalar(
            select(Volume.quality_score).where(
                Volume.public_id == volume_public_id,
                Volume.organization_id == organization_id,
            )
        )


def _decode_steps(encoded: str) -> list[tuple[str, int | None]]:
    """Read back the ``code:tooth`` pairs an option was stored with.

    Tolerates the bare-code form an earlier build wrote, so a plan generated
    before this change still accepts rather than silently producing nothing.
    """
    steps: list[tuple[str, int | None]] = []
    for chunk in encoded.split(","):
        if not chunk:
            continue
        code, _, tooth = chunk.partition(":")
        steps.append((code, int(tooth) if tooth.isdigit() else None))
    return steps
