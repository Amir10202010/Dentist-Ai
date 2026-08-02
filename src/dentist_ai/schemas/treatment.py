"""Treatment plan payloads."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field, computed_field

from dentist_ai.clinical.protocols import Priority, ProcedureCategory
from dentist_ai.db.models import (
    PlanComplexity,
    PlanItemStatus,
    PlanOrigin,
    PlanStatus,
    TreatmentApproach,
)
from dentist_ai.schemas.common import ApiModel


class ProcedureOption(ApiModel):
    """An entry from the protocol table, for the "add step" picker."""

    code: str
    label: str
    category: ProcedureCategory
    category_label: str
    priority: Priority
    priority_label: str
    visits: int
    minutes: int


class PlanItemResponse(ApiModel):
    id: int
    procedure_code: str
    procedure_label: str
    category: ProcedureCategory
    category_label: str
    tooth_number: int | None
    tooth_name: str | None
    priority: Priority
    priority_label: str
    status: PlanItemStatus
    status_label: str
    estimated_visits: int
    estimated_minutes: int
    scheduled_for: date | None
    completed_at: datetime | None
    notes: str | None
    source_finding_id: int | None
    source_study_public_id: str | None


class TreatmentOptionResponse(ApiModel):
    """One defensible way to treat the same case.

    The three options are alternatives, not tiers: the conservative one is the
    right answer for a patient who wants the problem treated and nothing more,
    so the UI shows them side by side rather than as an upsell ladder.
    """

    position: int
    title: str
    approach: TreatmentApproach
    approach_label: str
    summary: str
    priority: Priority
    priority_label: str
    complexity: PlanComplexity
    complexity_label: str
    estimated_visits: int
    estimated_minutes: int
    #: Calendar weeks including healing, which is not the same as chair time.
    estimated_weeks: int
    benefits: str
    risks: str
    procedure_codes: list[str]
    is_selected: bool


class PlanResponse(ApiModel):
    public_id: str
    patient_id: int
    patient_name: str | None
    title: str
    status: PlanStatus
    status_label: str
    notes: str | None
    created_at: datetime
    created_by_name: str | None
    items: list[PlanItemResponse] = Field(default_factory=list)

    #: Whether a clinician assembled this plan or the planner proposed it. The
    #: UI must not present a generated draft as an agreed course of treatment.
    origin: PlanOrigin = PlanOrigin.MANUAL
    complexity: PlanComplexity | None = None
    complexity_label: str | None = None
    estimated_weeks: int | None = None
    risks: str | None = None
    follow_up: str | None = None
    rationale: str | None = None
    options: list[TreatmentOptionResponse] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def open_count(self) -> int:
        return sum(1 for item in self.items if item.status.is_open)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def done_count(self) -> int:
        return sum(1 for item in self.items if item.status is PlanItemStatus.DONE)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_visits(self) -> int:
        return sum(item.estimated_visits for item in self.items if item.status.is_open)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_minutes(self) -> int:
        return sum(
            item.estimated_visits * item.estimated_minutes
            for item in self.items
            if item.status.is_open
        )


class PlanCreateRequest(ApiModel):
    patient_id: int
    title: str = Field(default="План лечения", min_length=1, max_length=160)
    notes: str | None = Field(default=None, max_length=4000)


class PlanUpdateRequest(ApiModel):
    title: str = Field(min_length=1, max_length=160)
    status: PlanStatus
    notes: str | None = Field(default=None, max_length=4000)


class PlanItemCreateRequest(ApiModel):
    procedure_code: str = Field(max_length=48)
    tooth_number: int | None = Field(default=None, ge=11, le=48)
    notes: str | None = Field(default=None, max_length=2000)


class PlanItemUpdateRequest(ApiModel):
    status: PlanItemStatus
    tooth_number: int | None = Field(default=None, ge=11, le=48)
    scheduled_for: date | None = None
    estimated_visits: int = Field(default=1, ge=1, le=40)
    estimated_minutes: int = Field(default=60, ge=5, le=600)
    notes: str | None = Field(default=None, max_length=2000)


class ProposeFromStudyRequest(ApiModel):
    study_public_id: str = Field(max_length=26)
    plan_public_id: str | None = Field(default=None, max_length=26)


class GeneratePlanRequest(ApiModel):
    patient_id: int
    #: Narrow the plan to one record. Omitted, it draws on every reviewed
    #: finding the patient has.
    volume_public_id: str | None = Field(default=None, max_length=26)
    study_public_id: str | None = Field(default=None, max_length=26)


class AcceptOptionRequest(ApiModel):
    approach: TreatmentApproach
