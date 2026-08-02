"""Radiograph analysis: upload, inference, findings, review."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
from sqlalchemy import Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.datastructures import UploadFile

from dentist_ai.clinical.charting import estimate_tooth, is_valid
from dentist_ai.core.errors import (
    InferenceError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from dentist_ai.core.ids import generate_public_id
from dentist_ai.core.logging import get_logger
from dentist_ai.core.text import safe_display_name
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import Finding, FindingReview, Patient, Study, StudyStatus, User
from dentist_ai.ml.detector import Detector
from dentist_ai.ml.taxonomy import Severity, by_id
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext
from dentist_ai.services.storage import ImageStorage

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StudyListRow:
    study: Study
    patient_name: str | None
    finding_count: int
    attention_count: int
    top_severity: Severity | None


class StudyService:
    def __init__(
        self,
        session: AsyncSession,
        storage: ImageStorage,
        detector: Detector,
        audit: AuditService,
    ) -> None:
        self._session = session
        self._storage = storage
        self._detector = detector
        self._audit = audit

    # -- ingestion --------------------------------------------------------
    async def upload_and_analyze(
        self,
        upload: UploadFile,
        *,
        actor: User,
        context: RequestContext,
        patient_id: int | None = None,
    ) -> Study:
        """Store the image, run inference, persist structured findings.

        Inference runs inline: on CPU it is a couple of seconds, and a
        synchronous result is a far better experience than polling. The
        ``status`` column exists so moving to a queue later is a routing
        change, not a schema migration.
        """
        if patient_id is not None:
            await self._assert_patient_in_org(patient_id, actor.organization_id)

        stored = await self._storage.save_upload(upload)

        study = Study(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            patient_id=patient_id,
            uploaded_by_id=actor.id,
            original_filename=safe_display_name(upload.filename, fallback="snapshot.jpg"),
            content_hash=stored.content_hash,
            content_type=stored.content_type,
            byte_size=stored.byte_size,
            width=stored.width,
            height=stored.height,
            status=StudyStatus.PROCESSING,
        )
        self._session.add(study)
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.STUDY_UPLOADED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="study",
            resource_id=study.public_id,
            context=context,
        )

        try:
            await self._run_inference(study)
        except InferenceError as exc:
            # The upload itself succeeded and is worth keeping — the clinician
            # can retry analysis without re-scanning the patient.
            study.status = StudyStatus.FAILED
            study.failure_reason = exc.message
            log.warning("inference_failed", study=study.public_id, reason=exc.message)
            raise

        # Re-fetch with relationships eagerly loaded. The in-memory instance
        # has unloaded `patient` / `uploaded_by` proxies, and touching those
        # during response serialisation would attempt lazy IO outside the
        # async context (MissingGreenlet).
        return await self._fetch_detailed(study.public_id, actor.organization_id)

    async def reanalyze(self, study: Study) -> Study:
        await self._session.execute(delete(Finding).where(Finding.study_id == study.id))
        study.status = StudyStatus.PROCESSING
        study.failure_reason = None
        await self._run_inference(study)
        return study

    async def _run_inference(self, study: Study) -> None:
        path = self._storage.master_path(study.content_hash)
        if not path.is_file():
            raise InferenceError("Файл снимка недоступен.")

        with Image.open(path) as image:
            image.load()
            result = await self._detector.detect(image)

        for detection in result.detections:
            taxonomy = by_id(detection.class_id)
            self._session.add(
                Finding(
                    study_id=study.id,
                    class_id=detection.class_id,
                    class_key=taxonomy.key,
                    confidence=detection.confidence,
                    x=detection.x,
                    y=detection.y,
                    width=detection.width,
                    height=detection.height,
                    tooth_number=estimate_tooth(
                        taxonomy,
                        x=detection.x,
                        y=detection.y,
                        width=detection.width,
                        height=detection.height,
                    ),
                )
            )

        study.status = StudyStatus.COMPLETED
        study.model_version = result.model_version
        study.inference_ms = result.duration_ms
        study.analyzed_at = utcnow()
        await self._session.flush()

        log.info(
            "study_analyzed",
            study=study.public_id,
            findings=len(result.detections),
            duration_ms=result.duration_ms,
        )

    # -- reads ------------------------------------------------------------
    async def _fetch_detailed(self, public_id: str, organization_id: int) -> Study:
        """Load a study with every relationship the presenter touches."""
        await self._session.flush()
        study = await self._session.scalar(
            select(Study)
            .options(
                selectinload(Study.findings),
                selectinload(Study.patient),
                selectinload(Study.uploaded_by),
            )
            .where(Study.public_id == public_id, Study.organization_id == organization_id)
        )
        if study is None:
            raise NotFoundError("Снимок не найден.")
        return study

    async def get_detailed(
        self,
        public_id: str,
        *,
        organization_id: int,
        actor: User | None = None,
        context: RequestContext | None = None,
    ) -> Study:
        study = await self._fetch_detailed(public_id, organization_id)

        if actor is not None:
            await self._audit.record(
                action=AuditAction.STUDY_VIEWED,
                organization_id=organization_id,
                actor_id=actor.id,
                resource_type="study",
                resource_id=study.public_id,
                context=context,
            )
        return study

    async def get_for_image_access(self, public_id: str, *, organization_id: int) -> Study:
        """Minimal lookup used by the authorised image route (hot path)."""
        study = await self._session.scalar(
            select(Study).where(
                Study.public_id == public_id, Study.organization_id == organization_id
            )
        )
        if study is None:
            raise NotFoundError("Снимок не найден.")
        return study

    async def list_studies(
        self,
        *,
        organization_id: int,
        patient_id: int | None = None,
        status: StudyStatus | None = None,
        query: str | None = None,
        limit: int = 24,
        offset: int = 0,
    ) -> tuple[list[StudyListRow], int]:
        base: Select[tuple[Study]] = select(Study).where(Study.organization_id == organization_id)
        if patient_id is not None:
            base = base.where(Study.patient_id == patient_id)
        if status is not None:
            base = base.where(Study.status == status)
        if query:
            pattern = f"%{query.strip().lower()}%"
            base = base.outerjoin(Patient, Study.patient_id == Patient.id).where(
                func.lower(func.coalesce(Patient.full_name, "")).like(pattern)
                | func.lower(Study.original_filename).like(pattern)
                | (Study.public_id == query.strip().upper())
            )

        total = await self._session.scalar(select(func.count()).select_from(base.subquery()))

        rows = await self._session.execute(
            base.options(selectinload(Study.patient), selectinload(Study.findings))
            .order_by(Study.created_at.desc(), Study.id.desc())
            .limit(limit)
            .offset(offset)
        )

        result: list[StudyListRow] = []
        for study in rows.scalars().unique().all():
            severities = [by_id(item.class_id) for item in study.findings]
            attention = [item for item in severities if item.needs_attention]
            result.append(
                StudyListRow(
                    study=study,
                    patient_name=study.patient.full_name if study.patient else None,
                    finding_count=len(study.findings),
                    attention_count=len(attention),
                    top_severity=min(
                        (item.severity for item in severities),
                        key=lambda severity: severity.rank,
                        default=None,
                    ),
                )
            )
        return result, int(total or 0)

    # -- mutations --------------------------------------------------------
    async def update(
        self,
        public_id: str,
        *,
        actor: User,
        context: RequestContext,
        patient_id: int | None,
        notes: str | None,
    ) -> Study:
        study = await self.get_for_image_access(public_id, organization_id=actor.organization_id)
        if patient_id is not None:
            await self._assert_patient_in_org(patient_id, actor.organization_id)
        study.patient_id = patient_id
        study.notes = notes
        await self._audit.record(
            action=AuditAction.STUDY_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="study",
            resource_id=study.public_id,
            context=context,
        )
        return study

    async def set_finding_tooth(
        self,
        public_id: str,
        finding_id: int,
        tooth_number: int | None,
        *,
        actor: User,
        context: RequestContext,
    ) -> Finding:
        """Correct the estimated FDI number, or clear it."""
        if not actor.role.can_review_findings:
            raise PermissionDeniedError("Только врач может изменять номер зуба.")
        if tooth_number is not None and not is_valid(tooth_number):
            raise ValidationError(
                "Номер зуба должен быть в нотации FDI: 11-18, 21-28, 31-38, 41-48."
            )

        finding = await self._get_finding(public_id, finding_id, actor.organization_id)
        finding.tooth_number = tooth_number
        finding.tooth_confirmed = True

        await self._audit.record(
            action=AuditAction.FINDING_RECHARTED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="finding",
            resource_id=finding.id,
            context=context,
            detail=str(tooth_number) if tooth_number is not None else "cleared",
        )
        return finding

    async def review_finding(
        self,
        public_id: str,
        finding_id: int,
        review: FindingReview,
        *,
        actor: User,
        context: RequestContext,
    ) -> Finding:
        if not actor.role.can_review_findings:
            raise PermissionDeniedError("Только врач может подтверждать находки.")

        finding = await self._get_finding(public_id, finding_id, actor.organization_id)
        finding.review = review
        finding.reviewed_by_id = actor.id
        finding.reviewed_at = utcnow() if review is not FindingReview.UNREVIEWED else None

        await self._audit.record(
            action=AuditAction.FINDING_REVIEWED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="finding",
            resource_id=finding.id,
            context=context,
            detail=review.value,
        )
        return finding

    async def delete(self, public_id: str, *, actor: User, context: RequestContext) -> None:
        if not actor.role.can_delete_patients:
            raise PermissionDeniedError("Недостаточно прав для удаления снимка.")

        study = await self.get_for_image_access(public_id, organization_id=actor.organization_id)
        content_hash = study.content_hash
        await self._session.delete(study)
        await self._session.flush()

        # Content-addressed storage is shared: only reclaim the blob once no
        # study anywhere still references it.
        still_referenced = await self._session.scalar(
            select(func.count()).select_from(Study).where(Study.content_hash == content_hash)
        )
        if not still_referenced:
            self._storage.delete(content_hash)

        await self._audit.record(
            action=AuditAction.STUDY_DELETED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="study",
            resource_id=public_id,
            context=context,
        )

    # -- helpers ----------------------------------------------------------
    async def _get_finding(self, public_id: str, finding_id: int, organization_id: int) -> Finding:
        finding = await self._session.scalar(
            select(Finding)
            .join(Study, Finding.study_id == Study.id)
            .where(
                Finding.id == finding_id,
                Study.public_id == public_id,
                Study.organization_id == organization_id,
            )
        )
        if finding is None:
            raise NotFoundError("Находка не найдена.")
        return finding

    async def _assert_patient_in_org(self, patient_id: int, organization_id: int) -> None:
        exists = await self._session.scalar(
            select(Patient.id).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if exists is None:
            raise NotFoundError("Пациент не найден.")
