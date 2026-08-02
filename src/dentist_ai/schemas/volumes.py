"""CBCT payloads.

The one shape worth explaining is :class:`VolumeGeometryResponse`. The viewer
needs the voxel grid, the millimetre spacing and the Hounsfield mapping
*before* it downloads 16 MB of voxels — it sizes its textures, lays out the
three MPR panes and builds the window slider from them. So geometry rides on
the metadata response and the voxels come from a separate endpoint the browser
can cache for a day.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field, computed_field

from dentist_ai.db.models import (
    AnnotationKind,
    FindingReview,
    MeasurementKind,
    StudyStatus,
    ViewPlane,
    VolumeFieldOfView,
    VolumeFormat,
)
from dentist_ai.ml.cbct_taxonomy import VolumeCategory
from dentist_ai.ml.taxonomy import Severity
from dentist_ai.schemas.common import ApiModel


class VolumeGeometryResponse(ApiModel):
    """Everything needed to interpret the voxel payload."""

    width: int
    height: int
    depth: int
    #: Millimetres per voxel, ``(x, y, z)``.
    spacing: tuple[float, float, float]
    #: ``hounsfield = stored * huSlope + huIntercept``.
    hu_slope: float
    hu_intercept: float
    #: Default window in stored 0-255 units, from the scanner where it wrote one.
    window_center: float
    window_width: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def physical_size(self) -> tuple[float, float, float]:
        """Field of view in millimetres. Drives the viewer's scale bar."""
        return (
            round(self.width * self.spacing[0], 1),
            round(self.height * self.spacing[1], 1),
            round(self.depth * self.spacing[2], 1),
        )


class BoundingBox3DResponse(ApiModel):
    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float


class VolumeFindingResponse(ApiModel):
    id: int
    class_key: str
    label: str
    category: VolumeCategory
    category_label: str
    severity: Severity
    severity_label: str
    confidence: float
    box: BoundingBox3DResponse
    region: str
    region_label: str
    tooth_number: int | None
    tooth_name: str | None
    tooth_confirmed: bool
    extent_mm: float | None
    mean_density: float | None
    #: Why the pipeline says this — the taxonomy's fixed explanation, frozen at
    #: analysis time so a printed report keeps reading the way it was signed.
    rationale: str
    next_steps: str
    #: Which stage produced it, for the model-attribution panel.
    produced_by: str
    #: Findings that must be presented as needing specialist confirmation and
    #: never as a diagnosis. Set by the presenter from the taxonomy, so a
    #: client cannot forget to check it.
    requires_confirmation: bool
    review: FindingReview
    reviewed_at: datetime | None


class MeasurementResponse(ApiModel):
    id: int
    kind: MeasurementKind
    plane: ViewPlane
    label: str
    #: Normalised ``[x, y, z]`` triples, so a measurement stays attached to the
    #: anatomy rather than to a resolution.
    points: list[list[float]]
    value: float
    unit: str
    notes: str | None
    created_at: datetime
    created_by_name: str | None


class AnnotationResponse(ApiModel):
    id: int
    kind: AnnotationKind
    plane: ViewPlane
    x: float
    y: float
    z: float | None
    title: str
    body: str | None
    volume_finding_id: int | None
    created_at: datetime
    created_by_name: str | None


class QualityResponse(ApiModel):
    """The QC stage's verdict on the acquisition."""

    score: float
    label: str
    notes: list[str]


class StageRecordResponse(ApiModel):
    name: str
    kind: str
    kind_label: str
    version: str
    status: str
    ms: int
    summary: str


class AiRunResponse(ApiModel):
    public_id: str
    pipeline_name: str
    pipeline_version: str
    status: StudyStatus
    total_ms: int
    finding_count: int
    failure_reason: str | None
    stages: list[StageRecordResponse]
    created_at: datetime
    triggered_by_name: str | None


class VolumeCategoryCount(ApiModel):
    category: VolumeCategory
    label: str
    count: int


class VolumeResponse(ApiModel):
    public_id: str
    patient_id: int
    patient_name: str | None
    original_filename: str
    source_format: VolumeFormat
    field_of_view: VolumeFieldOfView
    field_of_view_label: str
    status: StudyStatus
    failure_reason: str | None
    byte_size: int
    source_slice_count: int
    geometry: VolumeGeometryResponse
    quality: QualityResponse | None
    pipeline_version: str | None
    analysis_ms: int | None
    analyzed_at: datetime | None
    captured_on: date | None
    notes: str | None
    created_at: datetime
    uploaded_by_name: str | None
    voxels_url: str
    preview_url: str
    page_url: str
    findings: list[VolumeFindingResponse]
    category_counts: list[VolumeCategoryCount]
    attention_count: int
    finding_count: int
    measurements: list[MeasurementResponse]
    annotations: list[AnnotationResponse]


class VolumeListItemResponse(ApiModel):
    public_id: str
    patient_id: int
    patient_name: str | None
    original_filename: str
    field_of_view: VolumeFieldOfView
    field_of_view_label: str
    status: StudyStatus
    created_at: datetime
    captured_on: date | None
    preview_url: str
    page_url: str
    finding_count: int
    attention_count: int
    top_severity: Severity | None
    top_severity_label: str | None
    quality_score: float | None
    voxel_count: int


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class VolumeUpdateRequest(ApiModel):
    field_of_view: VolumeFieldOfView = VolumeFieldOfView.BOTH_JAWS
    captured_on: date | None = None
    notes: str | None = Field(default=None, max_length=4000)


class FindingReviewRequest(ApiModel):
    review: FindingReview


class FindingToothRequest(ApiModel):
    tooth_number: int | None = Field(default=None, ge=11, le=48)


class MeasurementCreateRequest(ApiModel):
    kind: MeasurementKind
    plane: ViewPlane = ViewPlane.AXIAL
    #: Two points for a distance, three for an angle, one for a density probe.
    #: The exact count is checked in the service, which owns the rule.
    points: list[list[float]] = Field(min_length=1, max_length=3)
    label: str = Field(default="", max_length=120)
    notes: str | None = Field(default=None, max_length=2000)


class AnnotationCreateRequest(ApiModel):
    kind: AnnotationKind = AnnotationKind.MARKER
    plane: ViewPlane = ViewPlane.AXIAL
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    z: float = Field(ge=0.0, le=1.0)
    title: str = Field(min_length=1, max_length=160)
    body: str | None = Field(default=None, max_length=4000)
    volume_finding_id: int | None = None


class ReanalyseRequest(ApiModel):
    #: Name of a registered pipeline, or ``None`` for the default.
    pipeline: str | None = Field(default=None, max_length=64)
