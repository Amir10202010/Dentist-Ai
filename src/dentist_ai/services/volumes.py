"""CBCT studies: upload, analysis, findings, measurements and annotations.

The volumetric counterpart of :mod:`dentist_ai.services.studies`, and it makes
the same trade in the same place: the pipeline runs inline, inside the upload
request, because a synchronous result is a better experience than an upload
that returns "pending" and a poll loop.

It is a closer call here than it is for a radiograph. Decoding a 400-slice
DICOM series and running six analysis stages is a couple of seconds on a
clinic's hardware against roughly one for a radiograph, and it happens under a
request rather than in a worker. The ``Volume.status`` column already models
``pending/processing/completed/failed``, so moving to a queue stays a routing
change rather than a migration — the trigger to do it is a field of view large
enough to push the request past its timeout, not a general preference for
asynchrony.

What the module does not do is let an analysis failure lose the scan. Ingest
and analysis are separate transactions in effect: the volume row is written
and flushed before the pipeline runs, so a stage that throws leaves a stored,
viewable scan marked ``failed`` with a reason, not a 500 and a discarded
upload.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.datastructures import UploadFile

from dentist_ai.clinical import charting
from dentist_ai.core.errors import NotFoundError, PermissionDeniedError, ValidationError
from dentist_ai.core.ids import generate_public_id
from dentist_ai.core.logging import get_logger
from dentist_ai.core.text import safe_display_name
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import (
    AiRun,
    Annotation,
    AnnotationKind,
    FindingReview,
    Measurement,
    MeasurementKind,
    NotificationKind,
    NotificationTone,
    Patient,
    StudyStatus,
    User,
    ViewPlane,
    Volume,
    VolumeFieldOfView,
    VolumeFinding,
)
from dentist_ai.ml.cbct_taxonomy import by_key
from dentist_ai.ml.pipeline import ModelRegistry, Pipeline, RunRecord, VolumeInput
from dentist_ai.ml.taxonomy import Severity
from dentist_ai.services import volume as volume_codec
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext
from dentist_ai.services.notifications import NotificationService
from dentist_ai.services.storage import VolumeStorage

log = get_logger(__name__)

#: Measurement kinds and how many points each needs. A distance is two ends,
#: an angle is three (vertex in the middle), a density probe is one.
_POINTS_REQUIRED: dict[MeasurementKind, int] = {
    MeasurementKind.DISTANCE: 2,
    MeasurementKind.ANGLE: 3,
    MeasurementKind.DENSITY: 1,
}
#: Below this a measurement arm has no direction, so an angle is undefined.
_MIN_ARM_MM: Final[float] = 1e-6
#: Every measurement point is a triple of normalised volume coordinates.
_COORDINATES_PER_POINT: Final[int] = 3

_UNITS: dict[MeasurementKind, str] = {
    MeasurementKind.DISTANCE: "мм",
    MeasurementKind.ANGLE: "°",
    MeasurementKind.DENSITY: "HU",
}


@dataclass(frozen=True, slots=True)
class VolumeListRow:
    """A volume plus the counts a list view needs, without loading findings."""

    volume: Volume
    finding_count: int
    attention_count: int
    top_severity: Severity | None


class VolumeService:
    def __init__(
        self,
        session: AsyncSession,
        storage: VolumeStorage,
        registry: ModelRegistry,
        audit: AuditService,
        notifications: NotificationService,
    ) -> None:
        self._session = session
        self._storage = storage
        self._registry = registry
        self._audit = audit
        self._notifications = notifications

    # -- ingest -----------------------------------------------------------
    async def upload(
        self,
        upload: UploadFile,
        *,
        actor: User,
        context: RequestContext,
        patient_id: int,
        field_of_view: VolumeFieldOfView,
        captured_on: date | None,
        notes: str | None,
        pipeline_name: str | None = None,
    ) -> Volume:
        patient = await self._patient_in_org(patient_id, actor.organization_id)
        stored = await self._storage.save_upload(upload)

        record = Volume(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            patient_id=patient_id,
            uploaded_by_id=actor.id,
            original_filename=safe_display_name(upload.filename, fallback="cbct.dcm"),
            source_format=stored.source_format,
            content_hash=stored.content_hash,
            byte_size=stored.byte_size,
            width=stored.width,
            height=stored.height,
            depth=stored.depth,
            spacing_x=stored.spacing[0],
            spacing_y=stored.spacing[1],
            spacing_z=stored.spacing[2],
            hu_slope=stored.hu_slope,
            hu_intercept=stored.hu_intercept,
            window_center=stored.window_center,
            window_width=stored.window_width,
            source_slice_count=stored.source_slice_count,
            field_of_view=field_of_view,
            captured_on=captured_on,
            notes=notes,
            status=StudyStatus.PROCESSING,
        )
        self._session.add(record)
        # Flushed before analysis so a failing stage leaves a stored, viewable
        # scan rather than discarding a 300 MB upload the clinic just made.
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.VOLUME_UPLOADED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="volume",
            resource_id=record.public_id,
            context=context,
        )

        await self.analyse(record, actor=actor, patient=patient, pipeline_name=pipeline_name)
        return await self.get(record.public_id, organization_id=actor.organization_id)

    async def analyse(
        self,
        record: Volume,
        *,
        actor: User,
        patient: Patient | None = None,
        pipeline_name: str | None = None,
    ) -> RunRecord | None:
        """Run the pipeline over a stored volume and persist what it found."""
        try:
            pipeline = self._registry.get(pipeline_name)
        except KeyError as exc:
            raise ValidationError(f"Неизвестный конвейер анализа: {pipeline_name}") from exc

        path = self._storage.path(record.content_hash)
        if not path.is_file():
            record.status = StudyStatus.FAILED
            record.failure_reason = "Файл тома недоступен."
            return None

        try:
            # Reading 16 MB and running six stages is CPU-bound; off-loop it so
            # one analysis does not stall every other request on the worker.
            payload = await asyncio.to_thread(path.read_bytes)
            run = await asyncio.to_thread(_run_pipeline, pipeline, payload, record, patient)
        except Exception as exc:
            log.exception("volume_analysis_failed", volume=record.public_id)
            record.status = StudyStatus.FAILED
            record.failure_reason = f"{type(exc).__name__}: {exc}"[:255]
            return None

        await self._replace_findings(record, run)

        record.status = StudyStatus.COMPLETED if run.succeeded else StudyStatus.FAILED
        record.failure_reason = (
            None if run.succeeded else "; ".join(item.summary for item in run.failed_stages)[:255]
        )
        record.pipeline_version = f"{run.pipeline_name}@{run.pipeline_version}"
        record.analysis_ms = run.total_ms
        record.analyzed_at = utcnow()
        record.quality_score = run.quality.score if run.quality else None

        self._session.add(
            AiRun(
                public_id=generate_public_id(),
                organization_id=record.organization_id,
                resource_type="volume",
                resource_id=record.public_id,
                triggered_by_id=actor.id,
                pipeline_name=run.pipeline_name,
                pipeline_version=run.pipeline_version,
                status=record.status,
                total_ms=run.total_ms,
                finding_count=len(run.detections),
                failure_reason=record.failure_reason,
                stages=json.dumps(
                    [
                        {
                            "name": stage.name,
                            "kind": stage.kind.value,
                            "kindLabel": stage.kind_label,
                            "version": stage.version,
                            "status": stage.status.value,
                            "ms": stage.duration_ms,
                            "summary": stage.summary,
                        }
                        for stage in run.stages
                    ],
                    ensure_ascii=False,
                ),
            )
        )
        await self._session.flush()
        await self._announce(record, run)

        log.info(
            "volume_analysed",
            volume=record.public_id,
            findings=len(run.detections),
            ms=run.total_ms,
            quality=record.quality_score,
        )
        return run

    async def reanalyse(
        self,
        public_id: str,
        *,
        actor: User,
        context: RequestContext,
        pipeline_name: str | None = None,
    ) -> Volume:
        record = await self.get(public_id, organization_id=actor.organization_id)
        record.status = StudyStatus.PROCESSING
        await self.analyse(record, actor=actor, pipeline_name=pipeline_name)
        await self._audit.record(
            action=AuditAction.VOLUME_ANALYSED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="volume",
            resource_id=public_id,
            context=context,
        )
        return await self.get(public_id, organization_id=actor.organization_id)

    async def _replace_findings(self, record: Volume, run: RunRecord) -> None:
        """Swap in the new findings, preserving clinician corrections.

        A re-run must not silently discard review decisions. A finding of the
        same class in the same place is treated as the same finding, so a
        confirmation survives a pipeline upgrade; anything else is new.
        """
        previous = list(
            await self._session.scalars(
                select(VolumeFinding).where(VolumeFinding.volume_id == record.id)
            )
        )
        # Identity across runs is "same class, same place to within a
        # hundredth of the volume". Two decimal places is about a millimetre
        # on a dental field of view — tight enough that two different lesions
        # never collide, loose enough that a pipeline whose boxes shifted
        # slightly still recognises its own previous finding.
        carried_by_key = {
            _identity(item.class_key, item.x, item.y, item.z): item for item in previous
        }

        for item in previous:
            await self._session.delete(item)
        await self._session.flush()

        for detection in run.detections:
            taxonomy = by_key(detection.class_key)
            box = detection.box
            estimated_tooth = (
                charting.estimate_tooth_3d(
                    x=box.center[0],
                    y=box.center[1],
                    z=box.center[2],
                    occlusal_z=run.landmarks.get("occlusal_z", 0.5),
                    midline_x=run.landmarks.get("midline_x", 0.5),
                    arch_center_y=run.landmarks.get("arch_center_y", 0.5),
                )
                if taxonomy.tooth_level
                else None
            )
            carried = carried_by_key.get(_identity(detection.class_key, box.x, box.y, box.z))

            finding = VolumeFinding(
                volume_id=record.id,
                class_key=detection.class_key,
                confidence=detection.confidence,
                x=box.x,
                y=box.y,
                z=box.z,
                width=box.width,
                height=box.height,
                depth=box.depth,
                region=detection.region.value,
                tooth_number=estimated_tooth,
                extent_mm=detection.extent_mm,
                mean_density=detection.mean_density,
                rationale=taxonomy.why(),
                next_steps=taxonomy.what_next(),
                produced_by=detection.produced_by,
            )
            if carried is not None:
                # A clinician's adjudication and hand-corrected tooth number
                # outlive the run that prompted them; a re-analysis is allowed
                # to change the model's opinion, never the reviewer's.
                finding.review = carried.review
                finding.reviewed_by_id = carried.reviewed_by_id
                finding.reviewed_at = carried.reviewed_at
                if carried.tooth_confirmed:
                    finding.tooth_number = carried.tooth_number
                    finding.tooth_confirmed = True
            self._session.add(finding)

        await self._session.flush()

    async def _announce(self, record: Volume, run: RunRecord) -> None:
        """Tell the uploader what came back, and flag anything critical."""
        if record.uploaded_by_id is None:
            return

        critical = [
            item for item in run.detections if by_key(item.class_key).severity is Severity.CRITICAL
        ]
        href = f"/app/volumes/{record.public_id}"

        if critical:
            labels = ", ".join(sorted({by_key(item.class_key).label() for item in critical}))
            await self._notifications.push(
                organization_id=record.organization_id,
                user_id=record.uploaded_by_id,
                kind=NotificationKind.CRITICAL_FINDING,
                tone=NotificationTone.CRITICAL,
                title="Обнаружены критические находки",
                body=f"{record.original_filename}: {labels}. Требуется просмотр врачом.",
                href=href,
            )
            return

        await self._notifications.push(
            organization_id=record.organization_id,
            user_id=record.uploaded_by_id,
            kind=(
                NotificationKind.ANALYSIS_COMPLETED
                if run.succeeded
                else NotificationKind.ANALYSIS_FAILED
            ),
            tone=NotificationTone.POSITIVE if run.succeeded else NotificationTone.WARNING,
            title="Анализ КЛКТ завершён" if run.succeeded else "Анализ КЛКТ завершён с ошибками",
            body=f"{record.original_filename}: найдено находок — {len(run.detections)}.",
            href=href,
        )

    # -- reads ------------------------------------------------------------
    async def get(
        self,
        public_id: str,
        *,
        organization_id: int,
        actor: User | None = None,
        context: RequestContext | None = None,
    ) -> Volume:
        record = await self._session.scalar(
            select(Volume)
            .options(
                selectinload(Volume.patient),
                selectinload(Volume.uploaded_by),
                selectinload(Volume.findings),
                selectinload(Volume.measurements).selectinload(Measurement.created_by),
                selectinload(Volume.annotations).selectinload(Annotation.created_by),
            )
            .where(Volume.public_id == public_id, Volume.organization_id == organization_id)
        )
        if record is None:
            raise NotFoundError("КЛКТ-исследование не найдено.")

        if actor is not None:
            await self._audit.record(
                action=AuditAction.VOLUME_VIEWED,
                organization_id=organization_id,
                actor_id=actor.id,
                resource_type="volume",
                resource_id=record.public_id,
                context=context,
            )
        return record

    async def list_for_organization(
        self,
        *,
        organization_id: int,
        patient_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[VolumeListRow], int]:
        base = select(Volume).where(Volume.organization_id == organization_id)
        if patient_id is not None:
            base = base.where(Volume.patient_id == patient_id)

        total = await self._session.scalar(select(func.count()).select_from(base.subquery()))
        rows = await self._session.scalars(
            base.options(
                selectinload(Volume.patient),
                selectinload(Volume.uploaded_by),
                selectinload(Volume.findings),
            )
            .order_by(Volume.created_at.desc(), Volume.id.desc())
            .limit(limit)
            .offset(offset)
        )

        listed: list[VolumeListRow] = []
        for record in rows.unique().all():
            classes = [by_key(item.class_key) for item in record.findings]
            attention = sum(1 for item in classes if item.needs_attention)
            top = min((item.severity for item in classes), key=lambda s: s.rank, default=None)
            listed.append(
                VolumeListRow(
                    volume=record,
                    finding_count=len(record.findings),
                    attention_count=attention,
                    top_severity=top,
                )
            )
        return listed, int(total or 0)

    async def voxels(self, record: Volume) -> bytes:
        """The canonical ``DVOL`` payload for the viewer."""
        path = self._storage.path(record.content_hash)
        if not path.is_file():
            raise NotFoundError("Файл тома недоступен.")
        return await asyncio.to_thread(path.read_bytes)

    # -- edits ------------------------------------------------------------
    async def update(
        self,
        public_id: str,
        *,
        actor: User,
        context: RequestContext,
        field_of_view: VolumeFieldOfView,
        captured_on: date | None,
        notes: str | None,
    ) -> Volume:
        record = await self.get(public_id, organization_id=actor.organization_id)
        record.field_of_view = field_of_view
        record.captured_on = captured_on
        record.notes = notes

        await self._audit.record(
            action=AuditAction.VOLUME_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="volume",
            resource_id=public_id,
            context=context,
        )
        return record

    async def delete(self, public_id: str, *, actor: User, context: RequestContext) -> None:
        if not actor.role.can_delete_patients:
            raise PermissionDeniedError("Недостаточно прав для удаления исследования.")

        record = await self.get(public_id, organization_id=actor.organization_id)
        content_hash = record.content_hash
        await self._session.delete(record)
        await self._session.flush()

        # Content-addressed storage deduplicates, so the bytes only go when the
        # last row referencing them does.
        still_referenced = await self._session.scalar(
            select(func.count()).select_from(Volume).where(Volume.content_hash == content_hash)
        )
        if not still_referenced:
            self._storage.delete(content_hash)

        await self._audit.record(
            action=AuditAction.VOLUME_DELETED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="volume",
            resource_id=public_id,
            context=context,
        )

    async def review_finding(
        self,
        public_id: str,
        finding_id: int,
        *,
        actor: User,
        context: RequestContext,
        review: str,
    ) -> VolumeFinding:
        if not actor.role.can_review_findings:
            raise PermissionDeniedError("Недостаточно прав для оценки находок.")

        record = await self.get(public_id, organization_id=actor.organization_id)
        finding = next((item for item in record.findings if item.id == finding_id), None)
        if finding is None:
            raise NotFoundError("Находка не найдена.")

        finding.review = FindingReview(review)
        finding.reviewed_by_id = actor.id
        finding.reviewed_at = utcnow()

        await self._audit.record(
            action=AuditAction.FINDING_REVIEWED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="volume_finding",
            resource_id=str(finding_id),
            context=context,
            detail=f"{finding.class_key} -> {review}",
        )
        return finding

    # -- measurements -----------------------------------------------------
    async def add_measurement(
        self,
        public_id: str,
        *,
        actor: User,
        kind: MeasurementKind,
        plane: ViewPlane,
        points: list[list[float]],
        label: str,
        notes: str | None,
    ) -> Measurement:
        record = await self.get(public_id, organization_id=actor.organization_id)
        required = _POINTS_REQUIRED[kind]
        if len(points) != required:
            raise ValidationError(
                f"Для измерения «{kind.value}» нужно {required} точк(и), получено {len(points)}."
            )
        if any(len(point) != _COORDINATES_PER_POINT for point in points):
            raise ValidationError("Каждая точка измерения — это три координаты.")

        value = _measure(kind, points, record)
        measurement = Measurement(
            organization_id=actor.organization_id,
            volume_id=record.id,
            created_by_id=actor.id,
            kind=kind,
            plane=plane,
            label=label[:120],
            points=json.dumps(points),
            value=value,
            unit=_UNITS[kind],
            notes=notes,
        )
        self._session.add(measurement)
        await self._session.flush()
        return measurement

    async def delete_measurement(self, public_id: str, measurement_id: int, *, actor: User) -> None:
        record = await self.get(public_id, organization_id=actor.organization_id)
        measurement = next(
            (item for item in record.measurements if item.id == measurement_id), None
        )
        if measurement is None:
            raise NotFoundError("Измерение не найдено.")
        await self._session.delete(measurement)

    # -- annotations ------------------------------------------------------
    async def add_annotation(
        self,
        public_id: str,
        *,
        actor: User,
        kind: AnnotationKind,
        plane: ViewPlane,
        position: tuple[float, float, float],
        title: str,
        body: str | None,
        finding_id: int | None,
    ) -> Annotation:
        record = await self.get(public_id, organization_id=actor.organization_id)
        annotation = Annotation(
            organization_id=actor.organization_id,
            volume_id=record.id,
            created_by_id=actor.id,
            kind=kind,
            plane=plane,
            x=position[0],
            y=position[1],
            z=position[2],
            title=title[:160],
            body=body,
            volume_finding_id=finding_id,
        )
        self._session.add(annotation)
        await self._session.flush()
        return annotation

    async def delete_annotation(self, public_id: str, annotation_id: int, *, actor: User) -> None:
        record = await self.get(public_id, organization_id=actor.organization_id)
        annotation = next((item for item in record.annotations if item.id == annotation_id), None)
        if annotation is None:
            raise NotFoundError("Аннотация не найдена.")
        await self._session.delete(annotation)

    # -- internals --------------------------------------------------------
    async def _patient_in_org(self, patient_id: int, organization_id: int) -> Patient:
        patient = await self._session.scalar(
            select(Patient).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if patient is None:
            raise NotFoundError("Пациент не найден.")
        return patient


def _identity(class_key: str, x: float, y: float, z: float) -> tuple[str, float, float, float]:
    return (class_key, round(x, 2), round(y, 2), round(z, 2))


def _run_pipeline(
    pipeline: Pipeline,
    payload: bytes,
    record: Volume,
    patient: Patient | None,
) -> RunRecord:
    """Decode the stored volume and run the analysis. Executed off-loop."""
    header = volume_codec.decode_header(payload)
    return pipeline.run(
        VolumeInput(
            voxels=volume_codec.decode_voxels(payload, header),
            spacing=header.spacing,
            hu_slope=header.hu_slope,
            hu_intercept=header.hu_intercept,
            field_of_view=record.field_of_view.value,
            patient_age=_age_of(patient),
        )
    )


def _age_of(patient: Patient | None) -> int | None:
    if patient is None or patient.date_of_birth is None:
        return None
    today = utcnow().date()
    born = patient.date_of_birth
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _measure(kind: MeasurementKind, points: list[list[float]], record: Volume) -> float:
    """Convert normalised viewer coordinates into a physical measurement.

    The conversion is the reason measurements are stored normalised: the
    millimetre value depends on the volume's spacing, and holding both means a
    stored measurement can be checked against the geometry it was taken on.
    """
    scale = (
        record.width * record.spacing_x,
        record.height * record.spacing_y,
        record.depth * record.spacing_z,
    )

    def to_mm(point: list[float]) -> tuple[float, float, float]:
        return (point[0] * scale[0], point[1] * scale[1], point[2] * scale[2])

    if kind is MeasurementKind.DENSITY:
        # A density probe has no geometry; the sampled value comes from the
        # client, which already has the voxels loaded.
        return 0.0

    if kind is MeasurementKind.DISTANCE:
        first, second = to_mm(points[0]), to_mm(points[1])
        # `math.sqrt` rather than `** 0.5`: exponentiation on floats is typed
        # as possibly returning a complex number, which it cannot here.
        squared = sum((a - b) ** 2 for a, b in zip(first, second, strict=True))
        return round(math.sqrt(squared), 2)

    vertex = to_mm(points[1])
    arm_a = [a - b for a, b in zip(to_mm(points[0]), vertex, strict=True)]
    arm_b = [a - b for a, b in zip(to_mm(points[2]), vertex, strict=True)]
    length_a = math.sqrt(sum(value**2 for value in arm_a))
    length_b = math.sqrt(sum(value**2 for value in arm_b))
    if length_a < _MIN_ARM_MM or length_b < _MIN_ARM_MM:
        return 0.0
    cosine = sum(a * b for a, b in zip(arm_a, arm_b, strict=True)) / (length_a * length_b)
    degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return round(float(degrees), 1)


__all__ = ["VolumeListRow", "VolumeService"]
