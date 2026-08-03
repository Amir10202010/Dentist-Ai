"""ORM → wire-model mapping.

Kept out of both the routers and the services, so changing the JSON shape
never means touching business logic and templates render from the same
objects the JSON API returns.
"""

from __future__ import annotations

import json
from collections import Counter

from dentist_ai.clinical import charting, protocols, treatment_planner
from dentist_ai.clinical.labels import (
    field_of_view_label,
    plan_item_status_label,
    plan_status_label,
    quality_label,
    scan_arch_label,
    scan_kind_label,
)
from dentist_ai.core.text import plural_ru
from dentist_ai.db.models import (
    AiRun,
    Annotation,
    Finding,
    Measurement,
    Patient,
    Scan3D,
    Study,
    TreatmentOption,
    TreatmentPlan,
    TreatmentPlanItem,
    Volume,
    VolumeFinding,
)
from dentist_ai.ml.cbct_taxonomy import Region, VolumeCategory, region_label, volume_category_label
from dentist_ai.ml.cbct_taxonomy import by_key as volume_by_key
from dentist_ai.ml.taxonomy import (
    CATEGORY_LABELS,
    DEFAULT_LOCALE,
    SEVERITY_LABELS,
    Category,
    by_key,
)
from dentist_ai.schemas.clinical import (
    BoundingBox,
    CategoryCount,
    FindingResponse,
    PatientResponse,
    PatientSummaryResponse,
    StudyListItemResponse,
    StudyResponse,
    TimelineEntry,
    TimelineKind,
)
from dentist_ai.schemas.scans import ScanBounds, ScanResponse
from dentist_ai.schemas.treatment import (
    PlanItemResponse,
    PlanResponse,
    ProcedureOption,
    TreatmentOptionResponse,
)
from dentist_ai.schemas.volumes import (
    AiRunResponse,
    AnnotationResponse,
    BoundingBox3DResponse,
    MeasurementResponse,
    QualityResponse,
    StageRecordResponse,
    VolumeCategoryCount,
    VolumeFindingResponse,
    VolumeGeometryResponse,
    VolumeListItemResponse,
    VolumeResponse,
)
from dentist_ai.services.patients import PatientRow
from dentist_ai.services.studies import StudyListRow
from dentist_ai.services.volumes import VolumeListRow

_UNKNOWN_PROCEDURE = "—"
#: A stored measurement point is an ``[x, y, z]`` triple; anything else is a
#: row written by an incompatible build and is skipped rather than trusted.
_COORDINATES_PER_POINT = 3


def present_patient(patient: Patient) -> PatientResponse:
    return PatientResponse.model_validate(patient)


def present_patient_row(
    row: PatientRow,
    *,
    scan_count: int = 0,
    open_plan_items: int = 0,
) -> PatientSummaryResponse:
    # `model_validate` reads the ORM object; the aggregate columns are not on
    # it, so they are layered on with `model_copy`. (`model_validate` has no
    # `update` parameter — passing one raises at runtime.)
    summary = PatientSummaryResponse.model_validate(row.patient)
    return summary.model_copy(
        update={
            "study_count": row.study_count,
            "last_study_at": row.last_study_at,
            "scan_count": scan_count,
            "open_plan_items": open_plan_items,
        }
    )


def present_finding(finding: Finding, locale: str = DEFAULT_LOCALE) -> FindingResponse:
    taxonomy = by_key(finding.class_key)
    return FindingResponse(
        id=finding.id,
        class_id=finding.class_id,
        class_key=finding.class_key,
        label=taxonomy.label(locale),
        category=taxonomy.category,
        severity=taxonomy.severity,
        severity_label=SEVERITY_LABELS[taxonomy.severity].get(
            locale, SEVERITY_LABELS[taxonomy.severity][DEFAULT_LOCALE]
        ),
        confidence=round(finding.confidence, 4),
        box=BoundingBox(x=finding.x, y=finding.y, width=finding.width, height=finding.height),
        tooth_number=finding.tooth_number,
        tooth_name=(
            None
            if finding.tooth_number is None
            else charting.tooth_name(finding.tooth_number, locale)
        ),
        tooth_confirmed=finding.tooth_confirmed,
        review=finding.review,
        reviewed_at=finding.reviewed_at,
    )


def present_study(study: Study, locale: str = DEFAULT_LOCALE) -> StudyResponse:
    findings = [present_finding(item, locale) for item in study.findings]
    # Severity first, then confidence: triage order, not model order.
    findings.sort(key=lambda item: (item.severity.rank, -item.confidence))

    counts: Counter[Category] = Counter(item.category for item in findings)
    category_counts = [
        CategoryCount(
            category=category,
            label=CATEGORY_LABELS[category].get(locale, CATEGORY_LABELS[category]["ru"]),
            count=counts[category],
        )
        for category in Category
        if counts[category]
    ]

    return StudyResponse(
        public_id=study.public_id,
        status=study.status,
        original_filename=study.original_filename,
        width=study.width,
        height=study.height,
        byte_size=study.byte_size,
        model_version=study.model_version,
        inference_ms=study.inference_ms,
        failure_reason=study.failure_reason,
        notes=study.notes,
        created_at=study.created_at,
        analyzed_at=study.analyzed_at,
        patient=present_patient(study.patient) if study.patient else None,
        uploaded_by_name=study.uploaded_by.full_name if study.uploaded_by else None,
        image_url=study_image_url(study.public_id),
        thumbnail_url=study_thumbnail_url(study.public_id),
        findings=findings,
        category_counts=category_counts,
        attention_count=counts[Category.PATHOLOGY],
    )


def present_study_row(row: StudyListRow, locale: str = DEFAULT_LOCALE) -> StudyListItemResponse:
    return StudyListItemResponse(
        public_id=row.study.public_id,
        status=row.study.status,
        original_filename=row.study.original_filename,
        created_at=row.study.created_at,
        thumbnail_url=study_thumbnail_url(row.study.public_id),
        patient_id=row.study.patient_id,
        patient_name=row.patient_name,
        finding_count=row.finding_count,
        attention_count=row.attention_count,
        top_severity=row.top_severity,
        top_severity_label=(
            None
            if row.top_severity is None
            else SEVERITY_LABELS[row.top_severity].get(
                locale, SEVERITY_LABELS[row.top_severity][DEFAULT_LOCALE]
            )
        ),
    )


def present_scan(scan: Scan3D, locale: str = DEFAULT_LOCALE) -> ScanResponse:
    return ScanResponse(
        public_id=scan.public_id,
        patient_id=scan.patient_id,
        patient_name=scan.patient.full_name if scan.patient else None,
        original_filename=scan.original_filename,
        source_format=scan.source_format,
        kind=scan.kind,
        kind_label=scan_kind_label(scan.kind, locale),
        arch=scan.arch,
        arch_label=scan_arch_label(scan.arch, locale),
        triangle_count=scan.triangle_count,
        byte_size=scan.byte_size,
        bounds=ScanBounds(
            min=(scan.min_x, scan.min_y, scan.min_z),
            max=(scan.max_x, scan.max_y, scan.max_z),
        ),
        captured_on=scan.captured_on,
        notes=scan.notes,
        created_at=scan.created_at,
        uploaded_by_name=scan.uploaded_by.full_name if scan.uploaded_by else None,
        mesh_url=scan_mesh_url(scan.public_id),
        page_url=f"/app/scans/{scan.public_id}",
    )


def present_plan_item(item: TreatmentPlanItem, locale: str = DEFAULT_LOCALE) -> PlanItemResponse:
    procedure = protocols.by_code(item.procedure_code)
    priority = protocols.Priority(item.priority)
    category = procedure.category if procedure else protocols.ProcedureCategory.DIAGNOSTICS
    return PlanItemResponse(
        id=item.id,
        procedure_code=item.procedure_code,
        procedure_label=procedure.label(locale) if procedure else _UNKNOWN_PROCEDURE,
        category=category,
        category_label=protocols.category_label(category, locale),
        tooth_number=item.tooth_number,
        tooth_name=(
            None if item.tooth_number is None else charting.tooth_name(item.tooth_number, locale)
        ),
        priority=priority,
        priority_label=protocols.priority_label(priority, locale),
        status=item.status,
        status_label=plan_item_status_label(item.status, locale),
        estimated_visits=item.estimated_visits,
        estimated_minutes=item.estimated_minutes,
        scheduled_for=item.scheduled_for,
        completed_at=item.completed_at,
        notes=item.notes,
        source_finding_id=item.source_finding_id,
        source_study_public_id=item.source_study_public_id,
    )


def present_plan(plan: TreatmentPlan, locale: str = DEFAULT_LOCALE) -> PlanResponse:
    items = [present_plan_item(item, locale) for item in plan.items]
    items.sort(key=lambda item: (item.priority.rank, item.tooth_number or 99, item.id))
    return PlanResponse(
        public_id=plan.public_id,
        patient_id=plan.patient_id,
        patient_name=plan.patient.full_name if plan.patient else None,
        title=plan.title,
        status=plan.status,
        status_label=plan_status_label(plan.status, locale),
        notes=plan.notes,
        created_at=plan.created_at,
        created_by_name=plan.created_by.full_name if plan.created_by else None,
        items=items,
        origin=plan.origin,
        complexity=plan.complexity,
        complexity_label=(
            None
            if plan.complexity is None
            else treatment_planner.complexity_label(plan.complexity, locale)
        ),
        estimated_weeks=plan.estimated_weeks,
        risks=plan.risks,
        follow_up=plan.follow_up,
        rationale=plan.rationale,
        options=[present_treatment_option(item, locale) for item in plan.options],
    )


def present_treatment_option(
    option: TreatmentOption, locale: str = DEFAULT_LOCALE
) -> TreatmentOptionResponse:
    priority = protocols.Priority(option.priority)
    return TreatmentOptionResponse(
        position=option.position,
        title=option.title,
        approach=option.approach,
        approach_label=treatment_planner.approach_label(option.approach, locale),
        summary=option.summary,
        priority=priority,
        priority_label=protocols.priority_label(priority, locale),
        complexity=option.complexity,
        complexity_label=treatment_planner.complexity_label(option.complexity, locale),
        estimated_visits=option.estimated_visits,
        estimated_minutes=option.estimated_minutes,
        estimated_weeks=option.estimated_weeks,
        benefits=option.benefits,
        risks=option.risks,
        procedure_codes=[
            chunk.partition(":")[0] for chunk in option.procedure_codes.split(",") if chunk
        ],
        is_selected=option.is_selected,
    )


def present_procedure_catalogue(locale: str = DEFAULT_LOCALE) -> list[ProcedureOption]:
    return [
        ProcedureOption(
            code=procedure.code,
            label=procedure.label(locale),
            category=procedure.category,
            category_label=protocols.category_label(procedure.category, locale),
            priority=procedure.priority,
            priority_label=protocols.priority_label(procedure.priority, locale),
            visits=procedure.visits,
            minutes=procedure.minutes,
        )
        for procedure in protocols.PROCEDURES
    ]


def build_timeline(
    patient: Patient,
    studies: list[StudyListItemResponse],
    scans: list[ScanResponse],
    plans: list[PlanResponse],
    locale: str = DEFAULT_LOCALE,
) -> list[TimelineEntry]:
    """Merge everything that happened to a patient into one dated list."""
    entries: list[TimelineEntry] = [
        TimelineEntry(
            kind=TimelineKind.PATIENT_CREATED,
            at=patient.created_at,
            title=_wording("patient_created", locale),
            icon="user-plus",
        )
    ]

    for study in studies:
        attention = study.attention_count
        entries.append(
            TimelineEntry(
                kind=TimelineKind.STUDY,
                at=study.created_at,
                title=_wording("study", locale),
                subtitle=_findings_subtitle(study.finding_count, attention, locale),
                href=f"/app/studies/{study.public_id}",
                icon="scan",
                severity=study.top_severity,
            )
        )

    for scan in scans:
        entries.append(
            TimelineEntry(
                kind=TimelineKind.SCAN,
                at=scan.created_at,
                title=scan.kind_label,
                subtitle=f"{scan.arch_label} · {scan.triangle_count:,} △".replace(",", " "),
                href=scan.page_url,
                icon="cube",
            )
        )

    for plan in plans:
        for item in plan.items:
            at = item.completed_at or plan.created_at
            tooth = f" · {item.tooth_number}" if item.tooth_number else ""
            entries.append(
                TimelineEntry(
                    kind=TimelineKind.PLAN_ITEM,
                    at=at,
                    title=f"{item.procedure_label}{tooth}",
                    subtitle=f"{plan.title} · {item.status_label}",
                    icon="clipboard",
                )
            )

    entries.sort(key=lambda entry: entry.at, reverse=True)
    return entries


def _findings_subtitle(total: int, attention: int, locale: str) -> str:
    if locale == "en":
        return f"{total} findings, {attention} needing attention"
    if locale == "kk":
        return f"{total} белгі, {attention} назар аударуды қажет етеді"
    findings_word = plural_ru(total, "находка", "находки", "находок")
    return f"{total} {findings_word}, требуют внимания: {attention}"


def _wording(key: str, locale: str) -> str:
    table = {
        "patient_created": {"ru": "Карта заведена", "en": "Record created", "kk": "Карта ашылды"},
        "study": {"ru": "Загружен снимок", "en": "Radiograph uploaded", "kk": "Түсірілім жүктелді"},
    }[key]
    return table.get(locale) or table[DEFAULT_LOCALE]


def study_image_url(public_id: str) -> str:
    return f"/api/v1/studies/{public_id}/image"


def study_thumbnail_url(public_id: str) -> str:
    return f"/api/v1/studies/{public_id}/thumbnail"


def scan_mesh_url(public_id: str) -> str:
    return f"/api/v1/scans/{public_id}/mesh"


def volume_voxels_url(public_id: str) -> str:
    return f"/api/v1/volumes/{public_id}/voxels"


def volume_preview_url(public_id: str, plane: str = "axial") -> str:
    return f"/api/v1/volumes/{public_id}/preview/{plane}"


# ---------------------------------------------------------------------------
# CBCT
# ---------------------------------------------------------------------------
def present_volume_finding(
    finding: VolumeFinding, locale: str = DEFAULT_LOCALE
) -> VolumeFindingResponse:
    taxonomy = volume_by_key(finding.class_key)
    region = _region_or_default(finding.region)
    return VolumeFindingResponse(
        id=finding.id,
        class_key=finding.class_key,
        label=taxonomy.label(locale),
        category=taxonomy.category,
        category_label=volume_category_label(taxonomy.category, locale),
        severity=taxonomy.severity,
        severity_label=SEVERITY_LABELS[taxonomy.severity].get(
            locale, SEVERITY_LABELS[taxonomy.severity][DEFAULT_LOCALE]
        ),
        confidence=round(finding.confidence, 4),
        box=BoundingBox3DResponse(
            x=finding.x,
            y=finding.y,
            z=finding.z,
            width=finding.width,
            height=finding.height,
            depth=finding.depth,
        ),
        region=finding.region,
        region_label=region_label(region, locale),
        tooth_number=finding.tooth_number,
        tooth_name=(
            None
            if finding.tooth_number is None
            else charting.tooth_name(finding.tooth_number, locale)
        ),
        tooth_confirmed=finding.tooth_confirmed,
        extent_mm=finding.extent_mm,
        mean_density=finding.mean_density,
        rationale=finding.rationale or taxonomy.why(locale),
        next_steps=finding.next_steps or taxonomy.what_next(locale),
        produced_by=finding.produced_by,
        # Read from the taxonomy rather than stored on the row: this is a
        # policy about how a class may be presented, and a stored copy would
        # let a finding written before the policy tightened escape it.
        requires_confirmation=taxonomy.requires_confirmation,
        review=finding.review,
        reviewed_at=finding.reviewed_at,
    )


def present_measurement(measurement: Measurement) -> MeasurementResponse:
    return MeasurementResponse(
        id=measurement.id,
        kind=measurement.kind,
        plane=measurement.plane,
        label=measurement.label,
        points=_decode_points(measurement.points),
        value=measurement.value,
        unit=measurement.unit,
        notes=measurement.notes,
        created_at=measurement.created_at,
        created_by_name=(measurement.created_by.full_name if measurement.created_by else None),
    )


def present_annotation(annotation: Annotation) -> AnnotationResponse:
    return AnnotationResponse(
        id=annotation.id,
        kind=annotation.kind,
        plane=annotation.plane,
        x=annotation.x,
        y=annotation.y,
        z=annotation.z,
        title=annotation.title,
        body=annotation.body,
        volume_finding_id=annotation.volume_finding_id,
        created_at=annotation.created_at,
        created_by_name=annotation.created_by.full_name if annotation.created_by else None,
    )


def present_volume_geometry(record: Volume) -> VolumeGeometryResponse:
    return VolumeGeometryResponse(
        width=record.width,
        height=record.height,
        depth=record.depth,
        spacing=(record.spacing_x, record.spacing_y, record.spacing_z),
        hu_slope=record.hu_slope,
        hu_intercept=record.hu_intercept,
        window_center=record.window_center,
        window_width=record.window_width,
    )


def present_volume(record: Volume, locale: str = DEFAULT_LOCALE) -> VolumeResponse:
    findings = [present_volume_finding(item, locale) for item in record.findings]
    # Severity first, then confidence: triage order, not model order.
    findings.sort(key=lambda item: (item.severity.rank, -item.confidence))

    counts: Counter[VolumeCategory] = Counter(item.category for item in findings)
    category_counts = [
        VolumeCategoryCount(
            category=category,
            label=volume_category_label(category, locale),
            count=counts[category],
        )
        for category in VolumeCategory
        if counts[category]
    ]

    return VolumeResponse(
        public_id=record.public_id,
        patient_id=record.patient_id,
        patient_name=record.patient.full_name if record.patient else None,
        original_filename=record.original_filename,
        source_format=record.source_format,
        field_of_view=record.field_of_view,
        field_of_view_label=field_of_view_label(record.field_of_view, locale),
        status=record.status,
        failure_reason=record.failure_reason,
        byte_size=record.byte_size,
        source_slice_count=record.source_slice_count,
        geometry=present_volume_geometry(record),
        quality=(
            None
            if record.quality_score is None
            else QualityResponse(
                score=round(record.quality_score, 3),
                label=quality_label(record.quality_score, locale),
                notes=[],
            )
        ),
        pipeline_version=record.pipeline_version,
        analysis_ms=record.analysis_ms,
        analyzed_at=record.analyzed_at,
        captured_on=record.captured_on,
        notes=record.notes,
        created_at=record.created_at,
        uploaded_by_name=record.uploaded_by.full_name if record.uploaded_by else None,
        voxels_url=volume_voxels_url(record.public_id),
        preview_url=volume_preview_url(record.public_id),
        page_url=f"/app/volumes/{record.public_id}",
        findings=findings,
        category_counts=category_counts,
        attention_count=sum(
            1 for item in record.findings if volume_by_key(item.class_key).needs_attention
        ),
        finding_count=len(findings),
        measurements=[present_measurement(item) for item in record.measurements],
        annotations=[present_annotation(item) for item in record.annotations],
    )


def present_volume_row(row: VolumeListRow, locale: str = DEFAULT_LOCALE) -> VolumeListItemResponse:
    record = row.volume
    return VolumeListItemResponse(
        public_id=record.public_id,
        patient_id=record.patient_id,
        patient_name=record.patient.full_name if record.patient else None,
        original_filename=record.original_filename,
        field_of_view=record.field_of_view,
        field_of_view_label=field_of_view_label(record.field_of_view, locale),
        status=record.status,
        created_at=record.created_at,
        captured_on=record.captured_on,
        preview_url=volume_preview_url(record.public_id),
        page_url=f"/app/volumes/{record.public_id}",
        finding_count=row.finding_count,
        attention_count=row.attention_count,
        top_severity=row.top_severity,
        top_severity_label=(
            None
            if row.top_severity is None
            else SEVERITY_LABELS[row.top_severity].get(
                locale, SEVERITY_LABELS[row.top_severity][DEFAULT_LOCALE]
            )
        ),
        quality_score=(None if record.quality_score is None else round(record.quality_score, 3)),
        voxel_count=record.width * record.height * record.depth,
    )


def present_ai_run(run: AiRun) -> AiRunResponse:
    return AiRunResponse(
        public_id=run.public_id,
        pipeline_name=run.pipeline_name,
        pipeline_version=run.pipeline_version,
        status=run.status,
        total_ms=run.total_ms,
        finding_count=run.finding_count,
        failure_reason=run.failure_reason,
        stages=[StageRecordResponse.model_validate(item) for item in _decode_stages(run.stages)],
        created_at=run.created_at,
        triggered_by_name=run.triggered_by.full_name if run.triggered_by else None,
    )


def _region_or_default(value: str) -> Region:
    """Tolerate a region written by a newer pipeline than this build knows."""
    try:
        return Region(value)
    except ValueError:
        return Region.FULL_VOLUME


def _decode_points(raw: str) -> list[list[float]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        [float(value) for value in point]
        for point in parsed
        if isinstance(point, list) and len(point) == _COORDINATES_PER_POINT
    ]


def _decode_stages(raw: str) -> list[dict[str, object]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]
