"""Patient, study and finding payloads."""

from __future__ import annotations

import enum
from datetime import UTC, date, datetime

from pydantic import Field, computed_field, field_validator

from dentist_ai.db.models import FindingReview, Sex, StudyStatus
from dentist_ai.ml.taxonomy import Category, Severity
from dentist_ai.schemas.common import ApiModel


def _today() -> date:
    """Today in UTC. ``date.today()`` would read the container's local clock."""
    return datetime.now(UTC).date()


# --------------------------------------------------------------------------
# Patients
# --------------------------------------------------------------------------
class PatientCreateRequest(ApiModel):
    full_name: str = Field(min_length=2, max_length=160)
    phone: str | None = Field(default=None, max_length=32)
    email: str | None = Field(default=None, max_length=320)
    date_of_birth: date | None = None
    sex: Sex = Sex.UNSPECIFIED
    medical_record_number: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("full_name")
    @classmethod
    def _collapse(cls, value: str) -> str:
        collapsed = " ".join(value.split())
        if not collapsed:
            raise ValueError("Укажите ФИО пациента")
        return collapsed

    @field_validator("phone")
    @classmethod
    def _normalise_phone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        digits = "".join(char for char in value if char.isdigit() or char == "+")
        return digits or None

    @field_validator("date_of_birth")
    @classmethod
    def _not_in_future(cls, value: date | None) -> date | None:
        if value is not None and value > _today():
            raise ValueError("Дата рождения не может быть в будущем")
        return value


class PatientUpdateRequest(PatientCreateRequest):
    """Same shape as create; PUT semantics (full replacement)."""


class PatientResponse(ApiModel):
    id: int
    full_name: str
    phone: str | None
    email: str | None
    date_of_birth: date | None
    sex: Sex
    medical_record_number: str | None
    notes: str | None
    created_at: datetime
    archived_at: datetime | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def age(self) -> int | None:
        if self.date_of_birth is None:
            return None
        today = _today()
        had_birthday = (today.month, today.day) >= (
            self.date_of_birth.month,
            self.date_of_birth.day,
        )
        return today.year - self.date_of_birth.year - (0 if had_birthday else 1)


class PatientSummaryResponse(PatientResponse):
    study_count: int = 0
    last_study_at: datetime | None = None
    scan_count: int = 0
    open_plan_items: int = 0


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------
class BoundingBox(ApiModel):
    """Normalised to [0, 1] against the stored master image."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class FindingResponse(ApiModel):
    id: int
    class_id: int
    class_key: str
    label: str
    category: Category
    severity: Severity
    #: Localised severity name. Sent by the server rather than mapped in the
    #: client, so translations live in exactly one place.
    severity_label: str
    confidence: float
    box: BoundingBox
    #: FDI number, estimated on ingest unless a clinician has corrected it.
    tooth_number: int | None = None
    tooth_name: str | None = None
    tooth_confirmed: bool = False
    review: FindingReview
    reviewed_at: datetime | None


class FindingReviewRequest(ApiModel):
    review: FindingReview


class FindingToothRequest(ApiModel):
    """``null`` detaches the finding from any tooth."""

    tooth_number: int | None = Field(default=None, ge=11, le=48)


class CategoryCount(ApiModel):
    category: Category
    label: str
    count: int


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------
class StudyResponse(ApiModel):
    public_id: str
    status: StudyStatus
    original_filename: str
    width: int
    height: int
    byte_size: int
    model_version: str | None
    inference_ms: int | None
    failure_reason: str | None
    notes: str | None
    created_at: datetime
    analyzed_at: datetime | None
    patient: PatientResponse | None
    uploaded_by_name: str | None

    image_url: str
    thumbnail_url: str

    findings: list[FindingResponse] = Field(default_factory=list)
    category_counts: list[CategoryCount] = Field(default_factory=list)
    attention_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def top_confidence(self) -> float | None:
        return max((item.confidence for item in self.findings), default=None)


class StudyListItemResponse(ApiModel):
    public_id: str
    status: StudyStatus
    original_filename: str
    created_at: datetime
    thumbnail_url: str
    patient_id: int | None
    patient_name: str | None
    finding_count: int
    attention_count: int
    top_severity: Severity | None
    top_severity_label: str | None


class StudyUpdateRequest(ApiModel):
    patient_id: int | None = None
    notes: str | None = Field(default=None, max_length=4000)


# --------------------------------------------------------------------------
# Odontogram and report
# --------------------------------------------------------------------------
class ToothCell(ApiModel):
    """One tooth on the odontogram."""

    tooth_number: int
    #: Worst severity among this tooth's pathology findings, if any.
    severity: Severity | None = None
    finding_count: int = 0
    has_restoration: bool = False
    is_missing: bool = False


class ToothGroup(ApiModel):
    tooth_number: int
    tooth_name: str
    findings: list[FindingResponse]


class Recommendation(ApiModel):
    """A protocol entry a finding maps to. Not model output, not advice."""

    procedure_code: str
    label: str
    category: str
    category_label: str
    priority: str
    priority_label: str
    tooth_number: int | None
    #: The finding this came from, worded for a human.
    reason: str
    source_finding_id: int | None


class StudyReportResponse(ApiModel):
    study_public_id: str
    generated_at: datetime
    patient_name: str | None
    #: Counts, worded. Assembled from the numbers below, never from a model.
    summary: str
    finding_count: int
    attention_count: int
    reviewed_count: int
    affected_teeth: int
    chart: list[ToothCell]
    teeth: list[ToothGroup]
    regional: list[FindingResponse]
    recommendations: list[Recommendation]
    disclaimer: str


# --------------------------------------------------------------------------
# Patient timeline
# --------------------------------------------------------------------------
class TimelineKind(enum.StrEnum):
    PATIENT_CREATED = "patient_created"
    STUDY = "study"
    SCAN = "scan"
    PLAN_ITEM = "plan_item"


class TimelineEntry(ApiModel):
    kind: TimelineKind
    at: datetime
    title: str
    subtitle: str | None = None
    href: str | None = None
    icon: str
    severity: Severity | None = None


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
class Tone(enum.StrEnum):
    """How a piece of derived information should read.

    Not the same axis as :class:`Severity`: a finding's severity is clinical,
    a tone is editorial. "The review queue is empty" is ``POSITIVE`` and has
    no clinical severity at all.
    """

    POSITIVE = "positive"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TimeSeriesPoint(ApiModel):
    date: date
    value: float


class LabelledCount(ApiModel):
    key: str
    label: str
    count: int
    severity: Severity | None = None


class MetricDelta(ApiModel):
    """One period measured against the one before it."""

    current: int
    previous: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def change(self) -> float | None:
        """Ratio change, or ``None`` when there is no baseline.

        Growth from zero is not a percentage. Reporting "+∞%" — or the more
        common "+100%" — for the first study a clinic ever uploads is a number
        the reader cannot act on, so the client renders the absolute count
        instead.
        """
        if self.previous == 0:
            return None
        return round((self.current - self.previous) / self.previous, 4)


class ReviewQueueItem(ApiModel):
    """A study carrying pathology findings no clinician has adjudicated yet.

    The dashboard's primary call to action, ordered by severity and then by how
    long it has been waiting.
    """

    public_id: str
    patient_name: str | None
    original_filename: str
    created_at: datetime
    #: Unreviewed findings in the "pathology" category, i.e. the ones that
    #: actually require a decision.
    pending_count: int
    top_severity: Severity
    top_severity_label: str
    top_finding_label: str
    top_confidence: float


class ActivityItem(ApiModel):
    """One audit event, rendered for humans.

    Sourced from the same append-only trail that satisfies the compliance
    requirement, so the feed cannot drift from the record.
    """

    id: int
    action: str
    actor_name: str | None
    summary: str
    icon: str
    tone: Tone
    resource_type: str
    resource_id: str | None
    created_at: datetime


class Insight(ApiModel):
    """An observation the clinic would otherwise have to go looking for.

    Every insight is derived from the tenant's own rows by an explicit rule in
    ``services/analytics.py``. None of them are model output, and none of them
    are advice — they point at a number and, where there is one, at the screen
    that acts on it.
    """

    key: str
    tone: Tone
    icon: str
    title: str
    body: str
    #: The single number the insight is about, pre-formatted. Rendered large.
    metric: str | None = None
    action_label: str | None = None
    action_href: str | None = None


class PipelineStatus(ApiModel):
    """Whether analysis is currently keeping up."""

    pending: int
    processing: int
    completed_today: int
    #: Failures in the last seven days. Older ones are not actionable.
    failed_recent: int


class ReviewStats(ApiModel):
    """How the clinic and the model agree.

    ``agreement_rate`` is over *adjudicated* findings only — unreviewed ones
    are not evidence either way, and folding them in would make the number
    drift down simply because the clinic got busy.
    """

    confirmed: int
    rejected: int
    unreviewed: int
    agreement_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DashboardResponse(ApiModel):
    generated_at: datetime

    total_patients: int
    new_patients_this_week: int
    total_studies: int
    studies_this_week: int
    findings_needing_attention: int
    average_inference_ms: int | None
    reviewed_share: float = Field(ge=0.0, le=1.0)
    studies_over_time: list[TimeSeriesPoint]
    top_findings: list[LabelledCount]
    category_breakdown: list[LabelledCount]

    studies_delta: MetricDelta
    patients_delta: MetricDelta
    attention_delta: MetricDelta

    #: Studies awaiting adjudication, capped for the widget.
    review_queue: list[ReviewQueueItem]
    #: Total studies in that state, which is usually larger than the list.
    review_queue_total: int
    #: Unreviewed pathology findings across the whole clinic.
    pending_findings: int
    oldest_pending_at: datetime | None

    activity: list[ActivityItem]
    insights: list[Insight]
    pipeline: PipelineStatus
    review_stats: ReviewStats
