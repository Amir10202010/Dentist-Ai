"""Everything the patient page needs, in one response."""

from __future__ import annotations

from pydantic import Field

from dentist_ai.schemas.clinical import (
    PatientSummaryResponse,
    StudyListItemResponse,
    TimelineEntry,
)
from dentist_ai.schemas.common import ApiModel
from dentist_ai.schemas.scans import ScanResponse
from dentist_ai.schemas.treatment import PlanResponse


class PatientOverviewResponse(ApiModel):
    patient: PatientSummaryResponse
    studies: list[StudyListItemResponse] = Field(default_factory=list)
    scans: list[ScanResponse] = Field(default_factory=list)
    plans: list[PlanResponse] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
