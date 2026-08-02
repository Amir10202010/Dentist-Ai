#!/usr/bin/env python3
"""Seed a demo clinic.

Gives a fresh clone something to look at without hand-clicking through the
UI. Idempotent: re-running tops the data up rather than duplicating it.

    make seed
"""

from __future__ import annotations

import asyncio
import io
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Importable whether invoked as `python scripts/seed.py` or `make seed`.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from dentist_ai.clinical import charting  # noqa: E402
from dentist_ai.clinical.charting import estimate_tooth  # noqa: E402
from dentist_ai.clinical.protocols import by_code  # noqa: E402
from dentist_ai.core.config import get_settings  # noqa: E402
from dentist_ai.core.ids import generate_public_id  # noqa: E402
from dentist_ai.core.security import PasswordService  # noqa: E402
from dentist_ai.db.models import (  # noqa: E402
    AiRun,
    AuditEvent,
    Finding,
    FindingReview,
    Organization,
    Patient,
    PlanItemStatus,
    PlanStatus,
    Scan3D,
    ScanArch,
    ScanKind,
    Sex,
    Study,
    StudyStatus,
    TreatmentPlan,
    TreatmentPlanItem,
    User,
    UserRole,
    Volume,
    VolumeFieldOfView,
    VolumeFinding,
)
from dentist_ai.db.session import (  # noqa: E402
    create_engine,
    create_session_factory,
    session_scope,
)
from dentist_ai.ml.cbct import build_registry  # noqa: E402
from dentist_ai.ml.cbct_taxonomy import by_key as volume_by_key  # noqa: E402
from dentist_ai.ml.detector import Detector  # noqa: E402
from dentist_ai.ml.factory import build_detector  # noqa: E402
from dentist_ai.ml.pipeline import ModelRegistry, VolumeInput  # noqa: E402
from dentist_ai.ml.taxonomy import by_id  # noqa: E402
from dentist_ai.services import volume as volume_codec  # noqa: E402
from dentist_ai.services.audit import AuditAction  # noqa: E402
from dentist_ai.services.storage import (  # noqa: E402
    ImageStorage,
    MeshStorage,
    VolumeStorage,
)
from scripts.generate_assets import render_radiograph  # noqa: E402
from scripts.synthetic_arch import arch_stl_bytes  # noqa: E402
from scripts.synthetic_cbct import build_preset  # noqa: E402

DEMO_EMAIL = "demo@dentist-ai.app"
DEMO_PASSWORD = "demo-clinic-2026"  # noqa: S105 - documented demo credential

PATIENTS = [
    ("Иванов Иван Петрович", "+77015554433", "1985-03-12", Sex.MALE, "A-1024"),
    ("Петрова Мария Сергеевна", "+77027778899", "1992-11-04", Sex.FEMALE, "A-1025"),
    ("Сидоров Алексей Юрьевич", "+77031112233", "1978-07-23", Sex.MALE, "A-1026"),
    ("Абенова Асель Маратовна", "+77054445566", "2001-01-30", Sex.FEMALE, "A-1027"),
    ("Ким Виктор Александрович", "+77089990011", "1966-09-18", Sex.MALE, "A-1028"),
]


def _audit(
    session: AsyncSession,
    *,
    organization_id: int,
    actor_id: int,
    action: AuditAction,
    resource_type: str,
    resource_id: str | int,
    created_at: datetime,
) -> None:
    """Write an audit row directly.

    The service takes its timestamp from the clock, which is right everywhere
    except here: a seeded study dated three weeks ago needs an audit row dated
    three weeks ago, or the activity feed shows a month of history arriving in
    the same second.
    """
    session.add(
        AuditEvent(
            organization_id=organization_id,
            actor_id=actor_id,
            action=action.value,
            resource_type=resource_type,
            resource_id=str(resource_id),
            created_at=created_at,
        )
    )


def synthetic_radiograph(seed: int) -> bytes:
    """A different-looking panoramic per study, so the grid is not uniform."""
    image = render_radiograph(size=(1000, 750), seed=seed)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=84)
    return buffer.getvalue()


async def main() -> None:  # noqa: PLR0915
    settings = get_settings()
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    storage = ImageStorage(settings.storage)
    mesh_storage = MeshStorage(settings.storage)
    volume_storage = VolumeStorage(settings.storage)
    registry = build_registry()
    detector = build_detector(settings)
    passwords = PasswordService(settings)

    async with session_scope(factory) as session:
        existing = await session.scalar(select(User).where(User.email == DEMO_EMAIL))
        if existing is not None:
            print(f"Demo account already present: {DEMO_EMAIL}")
            organization_id = existing.organization_id
            actor_id = existing.id
        else:
            organization = Organization(name="Демо-клиника", slug="demo-clinic")
            owner = User(
                organization=organization,
                email=DEMO_EMAIL,
                full_name="Айгуль Сагиндикова",
                password_hash=passwords.hash(DEMO_PASSWORD),
                role=UserRole.OWNER,
            )
            session.add_all([organization, owner])
            await session.flush()
            organization_id = organization.id
            actor_id = owner.id
            print(f"✓ Created clinic and owner {DEMO_EMAIL}")

        created_patients: list[Patient] = []
        #: Only the ones this run actually inserted, so re-seeding does not
        #: append a second set of audit rows for patients that already existed.
        new_patients: list[Patient] = []
        for name, phone, dob, sex, mrn in PATIENTS:
            found = await session.scalar(
                select(Patient).where(
                    Patient.organization_id == organization_id,
                    Patient.medical_record_number == mrn,
                )
            )
            if found is not None:
                created_patients.append(found)
                continue

            patient = Patient(
                organization_id=organization_id,
                full_name=name,
                phone=phone,
                date_of_birth=datetime.strptime(dob, "%Y-%m-%d").replace(tzinfo=UTC).date(),
                sex=sex,
                medical_record_number=mrn,
            )
            patient.refresh_search_text()
            session.add(patient)
            created_patients.append(patient)
            new_patients.append(patient)
        await session.flush()

        for patient in new_patients:
            _audit(
                session,
                organization_id=organization_id,
                actor_id=actor_id,
                action=AuditAction.PATIENT_CREATED,
                resource_id=patient.id,
                resource_type="patient",
                created_at=datetime.now(UTC),
            )

        existing_studies = await session.scalar(
            select(Study).where(Study.organization_id == organization_id)
        )
        if existing_studies is not None:
            print("Studies already seeded — skipping.")
            await engine.dispose()
            return

        await _seed_studies(
            session,
            storage=storage,
            detector=detector,
            organization_id=organization_id,
            actor_id=actor_id,
            patients=created_patients,
        )
        scan_count = await _seed_scans(
            session,
            storage=mesh_storage,
            organization_id=organization_id,
            actor_id=actor_id,
            patients=created_patients,
        )
        volume_count, finding_count = await _seed_volumes(
            session,
            storage=volume_storage,
            registry=registry,
            organization_id=organization_id,
            actor_id=actor_id,
            patients=created_patients,
        )
        plan_count = await _seed_plans(
            session,
            organization_id=organization_id,
            actor_id=actor_id,
            patients=created_patients,
        )

        print(f"✓ Seeded {len(created_patients)} patients with studies")
        print(f"✓ Seeded {scan_count} 3D scans and {plan_count} treatment plans")
        print(f"✓ Seeded {volume_count} CBCT studies with {finding_count} findings")

    await engine.dispose()
    print("\nSign in at http://127.0.0.1:8000/login")
    print(f"  email:    {DEMO_EMAIL}")
    print(f"  password: {DEMO_PASSWORD}")


async def _seed_studies(
    session: AsyncSession,
    *,
    storage: ImageStorage,
    detector: Detector,
    organization_id: int,
    actor_id: int,
    patients: list[Patient],
) -> None:
    """One to three backdated, analysed studies per patient."""
    rng = random.Random(2026)
    for index, patient in enumerate(patients):
        for offset in range(rng.randint(1, 3)):
            raw = synthetic_radiograph(index * 10 + offset)
            stored = await storage.store_bytes(raw)

            with Image.open(storage.master_path(stored.content_hash)) as image:
                image.load()
                result = await detector.detect(image)

            # Backdate so the dashboard trend chart has a real shape.
            created = datetime.now(UTC) - timedelta(days=rng.randint(0, 27), hours=offset)
            study = Study(
                public_id=generate_public_id(),
                organization_id=organization_id,
                patient_id=patient.id,
                original_filename=f"ОПТГ_{patient.medical_record_number}_{offset + 1}.jpg",
                content_hash=stored.content_hash,
                content_type=stored.content_type,
                byte_size=stored.byte_size,
                width=stored.width,
                height=stored.height,
                status=StudyStatus.COMPLETED,
                model_version=result.model_version,
                inference_ms=result.duration_ms,
                analyzed_at=created,
                created_at=created,
                updated_at=created,
            )
            session.add(study)
            await session.flush()

            # The activity feed reads back out of the audit trail, so a
            # seeded clinic has to leave the same footprints a real upload
            # would — backdated to when the study says it happened.
            _audit(
                session,
                organization_id=organization_id,
                actor_id=actor_id,
                action=AuditAction.STUDY_UPLOADED,
                resource_id=study.public_id,
                resource_type="study",
                created_at=created,
            )

            for detection in result.detections:
                taxonomy = by_id(detection.class_id)
                session.add(
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
                        # A realistic mix of reviewed and untouched findings.
                        review=rng.choice(
                            [
                                FindingReview.UNREVIEWED,
                                FindingReview.UNREVIEWED,
                                FindingReview.CONFIRMED,
                                FindingReview.REJECTED,
                            ]
                        ),
                    )
                )


#: Which phantom each patient gets, and how the scan was framed. Chosen to
#: spread the demo across the finding taxonomy rather than to flatter it: one
#: clean scan, one with a degraded acquisition, and pathology in between.
_CBCT_CASES: Final[tuple[tuple[str, VolumeFieldOfView], ...]] = (
    ("periapical", VolumeFieldOfView.BOTH_JAWS),
    ("cyst", VolumeFieldOfView.BOTH_JAWS),
    ("implant-site", VolumeFieldOfView.MANDIBLE),
    ("restored", VolumeFieldOfView.BOTH_JAWS),
    ("periodontal", VolumeFieldOfView.BOTH_JAWS),
    ("healthy", VolumeFieldOfView.BOTH_JAWS),
    ("poor-quality", VolumeFieldOfView.BOTH_JAWS),
)


async def _seed_volumes(
    session: AsyncSession,
    *,
    storage: VolumeStorage,
    registry: ModelRegistry,
    organization_id: int,
    actor_id: int,
    patients: list[Patient],
) -> tuple[int, int]:
    """One analysed CBCT per patient, from a synthetic phantom.

    The pipeline is run for real rather than the findings being fabricated, so
    the demo data exercises the same code a clinic's upload does — and so a
    regression in the analysis shows up in `make seed` rather than only in
    production.
    """
    rng = random.Random(90210)
    pipeline = registry.get()
    volumes = 0
    findings = 0

    for index, patient in enumerate(patients):
        preset, field_of_view = _CBCT_CASES[index % len(_CBCT_CASES)]
        payload = build_preset(preset, seed=index + 41)
        stored = await storage.store_bytes(payload)

        record = Volume(
            public_id=generate_public_id(),
            organization_id=organization_id,
            patient_id=patient.id,
            uploaded_by_id=actor_id,
            original_filename=f"{patient.medical_record_number}_CBCT.nii",
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
            captured_on=(datetime.now(UTC) - timedelta(days=rng.randint(2, 90))).date(),
            status=StudyStatus.PROCESSING,
        )
        session.add(record)
        await session.flush()

        # Read the voxels back out of the canonical container rather than
        # keeping the decoded array, so the seed exercises the same round trip
        # a real analysis does.
        canonical = await asyncio.to_thread(storage.path(stored.content_hash).read_bytes)
        header = volume_codec.decode_header(canonical)
        run = pipeline.run(
            VolumeInput(
                voxels=volume_codec.decode_voxels(canonical, header),
                spacing=stored.spacing,
                hu_slope=stored.hu_slope,
                hu_intercept=stored.hu_intercept,
                field_of_view=field_of_view.value,
            )
        )

        for detection in run.detections:
            taxonomy = volume_by_key(detection.class_key)
            tooth = (
                charting.estimate_tooth_3d(
                    x=detection.box.center[0],
                    y=detection.box.center[1],
                    z=detection.box.center[2],
                    occlusal_z=run.landmarks.get("occlusal_z", 0.5),
                    midline_x=run.landmarks.get("midline_x", 0.5),
                    arch_center_y=run.landmarks.get("arch_center_y", 0.5),
                )
                if taxonomy.tooth_level
                else None
            )
            session.add(
                VolumeFinding(
                    volume_id=record.id,
                    class_key=detection.class_key,
                    confidence=detection.confidence,
                    x=detection.box.x,
                    y=detection.box.y,
                    z=detection.box.z,
                    width=detection.box.width,
                    height=detection.box.height,
                    depth=detection.box.depth,
                    region=detection.region.value,
                    tooth_number=tooth,
                    extent_mm=detection.extent_mm,
                    mean_density=detection.mean_density,
                    rationale=taxonomy.why(),
                    next_steps=taxonomy.what_next(),
                    produced_by=detection.produced_by,
                )
            )
            findings += 1

        record.status = StudyStatus.COMPLETED if run.succeeded else StudyStatus.FAILED
        record.pipeline_version = f"{run.pipeline_name}@{run.pipeline_version}"
        record.analysis_ms = run.total_ms
        record.analyzed_at = datetime.now(UTC)
        record.quality_score = run.quality.score if run.quality else None

        session.add(
            AiRun(
                public_id=generate_public_id(),
                organization_id=organization_id,
                resource_type="volume",
                resource_id=record.public_id,
                triggered_by_id=actor_id,
                pipeline_name=run.pipeline_name,
                pipeline_version=run.pipeline_version,
                status=record.status,
                total_ms=run.total_ms,
                finding_count=len(run.detections),
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
        volumes += 1

    await session.flush()
    return volumes, findings


async def _seed_scans(
    session: AsyncSession,
    *,
    storage: MeshStorage,
    organization_id: int,
    actor_id: int,
    patients: list[Patient],
) -> int:
    """An upper and a lower arch for the first few patients."""
    rng = random.Random(4711)
    created = 0
    for index, patient in enumerate(patients[:3]):
        for upper in (True, False):
            stored = await storage.store_bytes(arch_stl_bytes(upper=upper, seed=index * 3.7))
            captured = datetime.now(UTC) - timedelta(days=rng.randint(1, 20))
            session.add(
                Scan3D(
                    public_id=generate_public_id(),
                    organization_id=organization_id,
                    patient_id=patient.id,
                    uploaded_by_id=actor_id,
                    original_filename=(
                        f"{patient.medical_record_number}_{'upper' if upper else 'lower'}.stl"
                    ),
                    source_format=stored.source_format,
                    content_hash=stored.content_hash,
                    byte_size=stored.byte_size,
                    triangle_count=stored.triangle_count,
                    kind=ScanKind.INTRAORAL,
                    arch=ScanArch.UPPER if upper else ScanArch.LOWER,
                    min_x=stored.bounds_min[0],
                    min_y=stored.bounds_min[1],
                    min_z=stored.bounds_min[2],
                    max_x=stored.bounds_max[0],
                    max_y=stored.bounds_max[1],
                    max_z=stored.bounds_max[2],
                    captured_on=captured.date(),
                    created_at=captured,
                    updated_at=captured,
                )
            )
            created += 1
    await session.flush()
    return created


async def _seed_plans(
    session: AsyncSession,
    *,
    organization_id: int,
    actor_id: int,
    patients: list[Patient],
) -> int:
    """One plan for the first two patients, part-way through treatment."""
    steps = [
        ("caries_restoration", 36, PlanItemStatus.DONE),
        ("root_canal", 24, PlanItemStatus.IN_PROGRESS),
        ("periodontal_therapy", None, PlanItemStatus.ACCEPTED),
        ("prosthetic_plan", 46, PlanItemStatus.PROPOSED),
    ]
    created = 0
    for patient in patients[:2]:
        plan = TreatmentPlan(
            public_id=generate_public_id(),
            organization_id=organization_id,
            patient_id=patient.id,
            created_by_id=actor_id,
            title="План лечения",
            status=PlanStatus.ACTIVE,
        )
        session.add(plan)
        await session.flush()

        for position, (code, tooth, status) in enumerate(steps):
            procedure = by_code(code)
            if procedure is None:
                continue
            session.add(
                TreatmentPlanItem(
                    plan_id=plan.id,
                    procedure_code=procedure.code,
                    tooth_number=tooth,
                    priority=procedure.priority.value,
                    estimated_visits=procedure.visits,
                    estimated_minutes=procedure.minutes,
                    status=status,
                    position=position,
                    completed_at=(
                        datetime.now(UTC) - timedelta(days=6)
                        if status is PlanItemStatus.DONE
                        else None
                    ),
                )
            )
        created += 1
    await session.flush()
    return created


if __name__ == "__main__":
    asyncio.run(main())
