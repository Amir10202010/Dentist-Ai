"""3D scans: upload, listing, metadata, deletion."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.datastructures import UploadFile

from dentist_ai.core.errors import NotFoundError, PermissionDeniedError
from dentist_ai.core.ids import generate_public_id
from dentist_ai.core.logging import get_logger
from dentist_ai.core.text import safe_display_name
from dentist_ai.db.models import Patient, Scan3D, ScanArch, ScanKind, User
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext
from dentist_ai.services.storage import MeshStorage

log = get_logger(__name__)


class ScanService:
    def __init__(
        self,
        session: AsyncSession,
        storage: MeshStorage,
        audit: AuditService,
    ) -> None:
        self._session = session
        self._storage = storage
        self._audit = audit

    async def upload(
        self,
        upload: UploadFile,
        *,
        actor: User,
        context: RequestContext,
        patient_id: int,
        kind: ScanKind,
        arch: ScanArch,
        captured_on: date | None,
        notes: str | None,
    ) -> Scan3D:
        await self._assert_patient_in_org(patient_id, actor.organization_id)
        stored = await self._storage.save_upload(upload)

        scan = Scan3D(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            patient_id=patient_id,
            uploaded_by_id=actor.id,
            original_filename=safe_display_name(upload.filename, fallback="scan.stl"),
            source_format=stored.source_format,
            content_hash=stored.content_hash,
            byte_size=stored.byte_size,
            triangle_count=stored.triangle_count,
            kind=kind,
            arch=arch,
            min_x=stored.bounds_min[0],
            min_y=stored.bounds_min[1],
            min_z=stored.bounds_min[2],
            max_x=stored.bounds_max[0],
            max_y=stored.bounds_max[1],
            max_z=stored.bounds_max[2],
            captured_on=captured_on,
            notes=notes,
        )
        self._session.add(scan)
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.SCAN_UPLOADED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="scan",
            resource_id=scan.public_id,
            context=context,
        )
        log.info(
            "scan_uploaded",
            scan=scan.public_id,
            triangles=scan.triangle_count,
            source=scan.source_format.value,
        )
        return await self.get(scan.public_id, organization_id=actor.organization_id)

    async def get(
        self,
        public_id: str,
        *,
        organization_id: int,
        actor: User | None = None,
        context: RequestContext | None = None,
    ) -> Scan3D:
        scan = await self._session.scalar(
            select(Scan3D)
            .options(selectinload(Scan3D.patient), selectinload(Scan3D.uploaded_by))
            .where(Scan3D.public_id == public_id, Scan3D.organization_id == organization_id)
        )
        if scan is None:
            raise NotFoundError("3D-модель не найдена.")

        if actor is not None:
            await self._audit.record(
                action=AuditAction.SCAN_VIEWED,
                organization_id=organization_id,
                actor_id=actor.id,
                resource_type="scan",
                resource_id=scan.public_id,
                context=context,
            )
        return scan

    async def list_for_patient(
        self,
        patient_id: int,
        *,
        organization_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Scan3D], int]:
        base = select(Scan3D).where(
            Scan3D.organization_id == organization_id, Scan3D.patient_id == patient_id
        )
        total = await self._session.scalar(select(func.count()).select_from(base.subquery()))
        rows = await self._session.scalars(
            base.options(selectinload(Scan3D.uploaded_by), selectinload(Scan3D.patient))
            .order_by(Scan3D.created_at.desc(), Scan3D.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows.unique().all()), int(total or 0)

    async def update(
        self,
        public_id: str,
        *,
        actor: User,
        context: RequestContext,
        kind: ScanKind,
        arch: ScanArch,
        captured_on: date | None,
        notes: str | None,
    ) -> Scan3D:
        scan = await self.get(public_id, organization_id=actor.organization_id)
        scan.kind = kind
        scan.arch = arch
        scan.captured_on = captured_on
        scan.notes = notes

        await self._audit.record(
            action=AuditAction.SCAN_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="scan",
            resource_id=scan.public_id,
            context=context,
        )
        return scan

    async def delete(self, public_id: str, *, actor: User, context: RequestContext) -> None:
        if not actor.role.can_delete_patients:
            raise PermissionDeniedError("Недостаточно прав для удаления 3D-модели.")

        scan = await self.get(public_id, organization_id=actor.organization_id)
        content_hash = scan.content_hash
        await self._session.delete(scan)
        await self._session.flush()

        still_referenced = await self._session.scalar(
            select(func.count()).select_from(Scan3D).where(Scan3D.content_hash == content_hash)
        )
        if not still_referenced:
            self._storage.delete(content_hash)

        await self._audit.record(
            action=AuditAction.SCAN_DELETED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="scan",
            resource_id=public_id,
            context=context,
        )

    async def _assert_patient_in_org(self, patient_id: int, organization_id: int) -> None:
        exists = await self._session.scalar(
            select(Patient.id).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if exists is None:
            raise NotFoundError("Пациент не найден.")
