"""Persistence model.

Every patient-bearing row is scoped to an ``Organization``; that scoping is
enforced in the service layer and backed by composite indexes, so a
tenant-leaking query is both hard to write and slow if you manage it.
"""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dentist_ai.db.base import Base, BigIntPk, TimestampMixin, UtcDateTime


class UserRole(enum.StrEnum):
    OWNER = "owner"
    DENTIST = "dentist"
    ASSISTANT = "assistant"

    @property
    def can_manage_members(self) -> bool:
        return self is UserRole.OWNER

    @property
    def can_delete_patients(self) -> bool:
        return self in (UserRole.OWNER, UserRole.DENTIST)

    @property
    def can_review_findings(self) -> bool:
        return self in (UserRole.OWNER, UserRole.DENTIST)


class StudyStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FindingReview(enum.StrEnum):
    """Clinician adjudication of a model detection."""

    UNREVIEWED = "unreviewed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class Sex(enum.StrEnum):
    MALE = "male"
    FEMALE = "female"
    UNSPECIFIED = "unspecified"


class MeshFormat(enum.StrEnum):
    """Format a 3D scan arrived in. Storage always holds binary STL."""

    STL = "stl"
    PLY = "ply"
    OBJ = "obj"


class ScanKind(enum.StrEnum):
    INTRAORAL = "intraoral"
    PLASTER_MODEL = "plaster_model"
    CBCT_SURFACE = "cbct_surface"
    RESTORATION_DESIGN = "restoration_design"


class ScanArch(enum.StrEnum):
    UPPER = "upper"
    LOWER = "lower"
    BOTH = "both"


class PlanStatus(enum.StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlanItemStatus(enum.StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    DECLINED = "declined"

    @property
    def is_open(self) -> bool:
        return self not in (PlanItemStatus.DONE, PlanItemStatus.DECLINED)


class VolumeFormat(enum.StrEnum):
    """Format a CBCT study arrived in. Storage always holds canonical DVOL."""

    DICOM = "dicom"
    NIFTI = "nifti"


class VolumeFieldOfView(enum.StrEnum):
    """What the scanner was aimed at, which decides what may be reported.

    A 5×5 cm implant-site volume does not contain the condyles, so a TMJ
    finding on one would be an artefact of the analysis rather than an
    observation. The pipeline reads this to bound its own claims.
    """

    FULL_HEAD = "full_head"
    BOTH_JAWS = "both_jaws"
    MAXILLA = "maxilla"
    MANDIBLE = "mandible"
    TMJ = "tmj"
    SINUS = "sinus"
    IMPLANT_SITE = "implant_site"


class MeasurementKind(enum.StrEnum):
    DISTANCE = "distance"
    ANGLE = "angle"
    DENSITY = "density"


class ViewPlane(enum.StrEnum):
    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"
    VOLUME = "volume"


class AnnotationKind(enum.StrEnum):
    MARKER = "marker"
    REGION = "region"
    QUESTION = "question"


class NotificationKind(enum.StrEnum):
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    UPLOAD_RECEIVED = "upload_received"
    REPORT_READY = "report_ready"
    CRITICAL_FINDING = "critical_finding"
    REVIEW_ASSIGNED = "review_assigned"
    COMMENT_ADDED = "comment_added"
    FOLLOW_UP_DUE = "follow_up_due"


class NotificationTone(enum.StrEnum):
    """Editorial register of a notification, not clinical severity."""

    POSITIVE = "positive"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AssignmentStatus(enum.StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    DECLINED = "declined"

    @property
    def is_open(self) -> bool:
        return self in (AssignmentStatus.PENDING, AssignmentStatus.ACCEPTED)


class AppointmentStatus(enum.StrEnum):
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class NoteKind(enum.StrEnum):
    CLINICAL = "clinical"
    ADMINISTRATIVE = "administrative"
    FOLLOW_UP = "follow_up"


class PlanOrigin(enum.StrEnum):
    MANUAL = "manual"
    GENERATED = "generated"


class PlanComplexity(enum.StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class TreatmentApproach(enum.StrEnum):
    """How aggressive a generated option is, at equal clinical validity."""

    CONSERVATIVE = "conservative"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class AssistantRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), nullable=False, unique=True, index=True)
    locale: Mapped[str] = mapped_column(String(5), nullable=False, default="ru")
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="Asia/Almaty")

    members: Mapped[list[User]] = relationship(back_populates="organization")
    patients: Mapped[list[Patient]] = relationship(back_populates="organization")


class User(Base, TimestampMixin):
    __tablename__ = "users"
    __table_args__ = (
        # Case-insensitivity is handled by normalising to lower-case on write,
        # which keeps the index usable on both Postgres and SQLite.
        UniqueConstraint("email", name="uq_users_email"),
        Index("ix_users_organization_id_role", "organization_id", "role"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, length=20), nullable=False, default=UserRole.OWNER
    )
    locale: Mapped[str] = mapped_column(String(5), nullable=False, default="ru")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    organization: Mapped[Organization] = relationship(back_populates="members")

    @property
    def initials(self) -> str:
        parts = [part for part in self.full_name.split() if part]
        return "".join(part[0].upper() for part in parts[:2]) or "?"


class Patient(Base, TimestampMixin):
    __tablename__ = "patients"
    __table_args__ = (
        # Tenant-first ordering: every list query filters by organisation and
        # then sorts, so this index serves both halves.
        Index("ix_patients_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_patients_organization_id_full_name", "organization_id", "full_name"),
        Index("ix_patients_organization_id_search_text", "organization_id", "search_text"),
        UniqueConstraint("organization_id", "medical_record_number", name="uq_patients_org_mrn"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    #: Clinic-assigned chart number. Optional, unique within the organisation.
    medical_record_number: Mapped[str | None] = mapped_column(String(64))
    phone: Mapped[str | None] = mapped_column(String(32), index=True)
    email: Mapped[str | None] = mapped_column(String(320))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    sex: Mapped[Sex] = mapped_column(
        Enum(Sex, native_enum=False, length=16), nullable=False, default=Sex.UNSPECIFIED
    )
    notes: Mapped[str | None] = mapped_column(Text)
    archived_at: Mapped[datetime | None] = mapped_column(UtcDateTime, index=True)

    #: Lower-cased name + phone + chart number, maintained on write by
    #: :meth:`refresh_search_text`.
    #:
    #: Searching via ``lower(full_name) LIKE …`` looks equivalent but is not:
    #: SQLite's ``lower()`` only folds ASCII, so "Иванов" never matches
    #: "иванов" there while working correctly on Postgres. Folding in Python
    #: gives one behaviour on every backend — and collapses three OR'd
    #: predicates into a single indexed column.
    search_text: Mapped[str] = mapped_column(String(560), nullable=False, default="")

    organization: Mapped[Organization] = relationship(back_populates="patients")
    studies: Mapped[list[Study]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    scans: Mapped[list[Scan3D]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    volumes: Mapped[list[Volume]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    treatment_plans: Mapped[list[TreatmentPlan]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    notes_log: Mapped[list[PatientNote]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    appointments: Mapped[list[Appointment]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )

    def refresh_search_text(self) -> None:
        """Recompute the denormalised search column. Call after any write."""
        parts = (self.full_name, self.phone, self.medical_record_number, self.email)
        self.search_text = " ".join(part for part in parts if part).lower()[:560]


class Study(Base, TimestampMixin):
    """A single uploaded radiograph and its analysis run."""

    __tablename__ = "studies"
    __table_args__ = (
        Index("ix_studies_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_studies_patient_id_created_at", "patient_id", "created_at"),
        Index("ix_studies_organization_id_status", "organization_id", "status"),
        CheckConstraint("width > 0 AND height > 0", name="study_positive_dimensions"),
        CheckConstraint("byte_size > 0", name="study_positive_size"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    uploaded_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    #: SHA-256 of the normalised image bytes; doubles as the storage key and as
    #: a cheap duplicate-upload check.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)

    status: Mapped[StudyStatus] = mapped_column(
        Enum(StudyStatus, native_enum=False, length=20),
        nullable=False,
        default=StudyStatus.PENDING,
    )
    failure_reason: Mapped[str | None] = mapped_column(String(255))
    model_version: Mapped[str | None] = mapped_column(String(64))
    inference_ms: Mapped[int | None] = mapped_column(Integer)
    analyzed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    notes: Mapped[str | None] = mapped_column(Text)

    patient: Mapped[Patient | None] = relationship(back_populates="studies")
    uploaded_by: Mapped[User | None] = relationship()
    findings: Mapped[list[Finding]] = relationship(
        back_populates="study",
        cascade="all, delete-orphan",
        order_by="Finding.confidence.desc()",
    )


class Finding(Base, TimestampMixin):
    """One detection: a class, a confidence and a normalised bounding box.

    Coordinates are normalised to ``[0, 1]`` against the stored master image,
    so overlays render at any display size and re-encoding the master can
    never invalidate them.
    """

    __tablename__ = "findings"
    __table_args__ = (
        Index("ix_findings_study_id_class_id", "study_id", "class_id"),
        Index("ix_findings_study_id_confidence", "study_id", "confidence"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="finding_confidence_range"),
        CheckConstraint(
            "x >= 0.0 AND y >= 0.0 AND width > 0.0 AND height > 0.0 "
            "AND x + width <= 1.0001 AND y + height <= 1.0001",
            name="finding_box_within_bounds",
        ),
        CheckConstraint(
            "tooth_number IS NULL OR ("
            "tooth_number / 10 BETWEEN 1 AND 4 AND tooth_number % 10 BETWEEN 1 AND 8)",
            name="finding_tooth_number_fdi",
        ),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    study_id: Mapped[int] = mapped_column(
        ForeignKey("studies.id", ondelete="CASCADE"), nullable=False
    )
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Stable slug from the taxonomy. Persisted alongside ``class_id`` so a
    #: future re-ordering of model outputs cannot silently relabel history.
    class_key: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)

    #: FDI number, estimated from the box position on ingest and correctable
    #: by the clinician. ``None`` for regional findings that belong to no
    #: single tooth.
    tooth_number: Mapped[int | None] = mapped_column(Integer)
    #: Set once a clinician has changed the number by hand, so a re-run of the
    #: estimate leaves their correction alone.
    tooth_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    review: Mapped[FindingReview] = mapped_column(
        Enum(FindingReview, native_enum=False, length=20),
        nullable=False,
        default=FindingReview.UNREVIEWED,
    )
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    study: Mapped[Study] = relationship(back_populates="findings")


class Scan3D(Base, TimestampMixin):
    """An intraoral scan, plaster-model scan or CBCT-derived surface.

    Whatever the file arrived as, storage holds one canonical binary STL
    addressed by the hash of those bytes, exactly like a radiograph.
    """

    __tablename__ = "scans_3d"
    __table_args__ = (
        Index("ix_scans_3d_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_scans_3d_patient_id_created_at", "patient_id", "created_at"),
        CheckConstraint("triangle_count > 0", name="scan_positive_triangles"),
        CheckConstraint("byte_size > 0", name="scan_positive_size"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    uploaded_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_format: Mapped[MeshFormat] = mapped_column(
        Enum(MeshFormat, native_enum=False, length=8), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    triangle_count: Mapped[int] = mapped_column(Integer, nullable=False)

    kind: Mapped[ScanKind] = mapped_column(
        Enum(ScanKind, native_enum=False, length=24), nullable=False, default=ScanKind.INTRAORAL
    )
    arch: Mapped[ScanArch] = mapped_column(
        Enum(ScanArch, native_enum=False, length=8), nullable=False, default=ScanArch.BOTH
    )

    #: Axis-aligned bounds in the file's own units, which for every dental
    #: scanner in practice are millimetres. The viewer needs them to frame the
    #: model, and the size they imply is a cheap sanity check on the upload.
    min_x: Mapped[float] = mapped_column(Float, nullable=False)
    min_y: Mapped[float] = mapped_column(Float, nullable=False)
    min_z: Mapped[float] = mapped_column(Float, nullable=False)
    max_x: Mapped[float] = mapped_column(Float, nullable=False)
    max_y: Mapped[float] = mapped_column(Float, nullable=False)
    max_z: Mapped[float] = mapped_column(Float, nullable=False)

    captured_on: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)

    patient: Mapped[Patient] = relationship(back_populates="scans")
    uploaded_by: Mapped[User | None] = relationship()


class Volume(Base, TimestampMixin):
    """A CBCT reconstruction and its analysis run.

    The volumetric counterpart of :class:`Study`. Whatever the study arrived as
    — a zip of DICOM instances, one multi-frame file, a NIfTI from a research
    pipeline — storage holds one canonical ``DVOL`` addressed by the hash of
    those bytes, and the geometry below describes what is in it.
    """

    __tablename__ = "volumes"
    __table_args__ = (
        Index("ix_volumes_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_volumes_patient_id_created_at", "patient_id", "created_at"),
        Index("ix_volumes_organization_id_status", "organization_id", "status"),
        CheckConstraint(
            "width > 0 AND height > 0 AND depth > 0", name="volume_positive_dimensions"
        ),
        CheckConstraint(
            "spacing_x > 0 AND spacing_y > 0 AND spacing_z > 0", name="volume_positive_spacing"
        ),
        CheckConstraint("byte_size > 0", name="volume_positive_size"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    uploaded_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_format: Mapped[VolumeFormat] = mapped_column(
        Enum(VolumeFormat, native_enum=False, length=12), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)

    #: Voxel counts of the *stored* volume, after ingest decimation.
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Millimetres per stored voxel. Every measurement the viewer reports is
    #: this multiplied by a voxel count, so it is the calibration of the whole
    #: feature and is written once, on ingest.
    spacing_x: Mapped[float] = mapped_column(Float, nullable=False)
    spacing_y: Mapped[float] = mapped_column(Float, nullable=False)
    spacing_z: Mapped[float] = mapped_column(Float, nullable=False)

    #: ``modality_units = stored * hu_slope + hu_intercept``, so an 8-bit
    #: sample can still be reported in Hounsfield units.
    hu_slope: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    hu_intercept: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    window_center: Mapped[float] = mapped_column(Float, nullable=False, default=127.5)
    window_width: Mapped[float] = mapped_column(Float, nullable=False, default=255.0)
    #: Slices the scanner produced, before decimation.
    source_slice_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    field_of_view: Mapped[VolumeFieldOfView] = mapped_column(
        Enum(VolumeFieldOfView, native_enum=False, length=16),
        nullable=False,
        default=VolumeFieldOfView.BOTH_JAWS,
    )
    captured_on: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str | None] = mapped_column(Text)

    status: Mapped[StudyStatus] = mapped_column(
        Enum(StudyStatus, native_enum=False, length=20),
        nullable=False,
        default=StudyStatus.PENDING,
    )
    failure_reason: Mapped[str | None] = mapped_column(String(255))
    pipeline_version: Mapped[str | None] = mapped_column(String(64))
    analysis_ms: Mapped[int | None] = mapped_column(Integer)
    analyzed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    #: Acquisition quality in ``[0, 1]``, from the pipeline's QC stage. Low
    #: scores suppress nothing, but every finding is presented beside it.
    quality_score: Mapped[float | None] = mapped_column(Float)

    patient: Mapped[Patient] = relationship(back_populates="volumes")
    uploaded_by: Mapped[User | None] = relationship()
    findings: Mapped[list[VolumeFinding]] = relationship(
        back_populates="volume",
        cascade="all, delete-orphan",
        order_by="VolumeFinding.confidence.desc()",
    )
    measurements: Mapped[list[Measurement]] = relationship(
        back_populates="volume", cascade="all, delete-orphan"
    )
    annotations: Mapped[list[Annotation]] = relationship(
        back_populates="volume", cascade="all, delete-orphan"
    )


class VolumeFinding(Base, TimestampMixin):
    """One volumetric detection: a class, a confidence and a normalised box.

    The box is a rectangular prism normalised to ``[0, 1]`` on each axis
    against the stored volume, for the same reason the 2D box is normalised:
    re-ingesting at a different decimation factor must not invalidate stored
    findings.

    ``rationale`` and ``next_steps`` are copied from the taxonomy at analysis
    time rather than looked up on read, so a report printed today still reads
    the way it did when the clinician signed it.
    """

    __tablename__ = "volume_findings"
    __table_args__ = (
        Index("ix_volume_findings_volume_id_class_key", "volume_id", "class_key"),
        Index("ix_volume_findings_volume_id_confidence", "volume_id", "confidence"),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="volume_finding_confidence_range"
        ),
        CheckConstraint(
            "x >= 0.0 AND y >= 0.0 AND z >= 0.0 "
            "AND width > 0.0 AND height > 0.0 AND depth > 0.0 "
            "AND x + width <= 1.0001 AND y + height <= 1.0001 AND z + depth <= 1.0001",
            name="volume_finding_box_within_bounds",
        ),
        CheckConstraint(
            "tooth_number IS NULL OR ("
            "tooth_number / 10 BETWEEN 1 AND 4 AND tooth_number % 10 BETWEEN 1 AND 8)",
            name="volume_finding_tooth_number_fdi",
        ),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    volume_id: Mapped[int] = mapped_column(
        ForeignKey("volumes.id", ondelete="CASCADE"), nullable=False
    )
    #: Stable key from ``ml/cbct_taxonomy.py``.
    class_key: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    depth: Mapped[float] = mapped_column(Float, nullable=False)

    region: Mapped[str] = mapped_column(String(32), nullable=False, default="full_volume")
    tooth_number: Mapped[int | None] = mapped_column(Integer)
    tooth_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: Longest extent of the finding in millimetres, where the class is one a
    #: measurement means something for. ``None`` for classes that are not.
    extent_mm: Mapped[float | None] = mapped_column(Float)
    #: Mean stored intensity inside the box, converted to modality units.
    mean_density: Mapped[float | None] = mapped_column(Float)

    #: Frozen copies of the taxonomy's explanation and follow-through.
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    next_steps: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Which pipeline stage produced it, for the model-attribution panel.
    produced_by: Mapped[str] = mapped_column(String(48), nullable=False, default="detection")

    review: Mapped[FindingReview] = mapped_column(
        Enum(FindingReview, native_enum=False, length=20),
        nullable=False,
        default=FindingReview.UNREVIEWED,
    )
    reviewed_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    reviewed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    volume: Mapped[Volume] = relationship(back_populates="findings")


class Measurement(Base, TimestampMixin):
    """A distance, angle or density sample a clinician took in the viewer.

    Points are stored normalised to the volume rather than in millimetres, so
    they stay attached to the anatomy; the value in ``value``/``unit`` is
    computed once against the volume's spacing and kept, so a stored
    measurement never silently changes.
    """

    __tablename__ = "measurements"
    __table_args__ = (
        Index("ix_measurements_volume_id_created_at", "volume_id", "created_at"),
        CheckConstraint("value >= 0.0", name="measurement_non_negative"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    volume_id: Mapped[int] = mapped_column(
        ForeignKey("volumes.id", ondelete="CASCADE"), nullable=False
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    kind: Mapped[MeasurementKind] = mapped_column(
        Enum(MeasurementKind, native_enum=False, length=16), nullable=False
    )
    plane: Mapped[ViewPlane] = mapped_column(
        Enum(ViewPlane, native_enum=False, length=12), nullable=False, default=ViewPlane.AXIAL
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    #: JSON array of ``[x, y, z]`` triples in normalised volume coordinates.
    points: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(8), nullable=False, default="мм")
    notes: Mapped[str | None] = mapped_column(Text)

    volume: Mapped[Volume] = relationship(back_populates="measurements")
    created_by: Mapped[User | None] = relationship()


class Annotation(Base, TimestampMixin):
    """A clinician's note pinned to a point in a volume or on a radiograph."""

    __tablename__ = "annotations"
    __table_args__ = (
        Index("ix_annotations_volume_id_created_at", "volume_id", "created_at"),
        Index("ix_annotations_study_id_created_at", "study_id", "created_at"),
        CheckConstraint(
            "(volume_id IS NOT NULL) <> (study_id IS NOT NULL)",
            name="annotation_exactly_one_target",
        ),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    volume_id: Mapped[int | None] = mapped_column(ForeignKey("volumes.id", ondelete="CASCADE"))
    study_id: Mapped[int | None] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"))
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    kind: Mapped[AnnotationKind] = mapped_column(
        Enum(AnnotationKind, native_enum=False, length=16),
        nullable=False,
        default=AnnotationKind.MARKER,
    )
    plane: Mapped[ViewPlane] = mapped_column(
        Enum(ViewPlane, native_enum=False, length=12), nullable=False, default=ViewPlane.AXIAL
    )
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    #: ``None`` on a radiograph, which has no third axis.
    z: Mapped[float | None] = mapped_column(Float)

    title: Mapped[str] = mapped_column(String(160), nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    #: The detection this annotation comments on, when it comments on one.
    volume_finding_id: Mapped[int | None] = mapped_column(
        ForeignKey("volume_findings.id", ondelete="SET NULL")
    )

    volume: Mapped[Volume | None] = relationship(back_populates="annotations")
    created_by: Mapped[User | None] = relationship()


class AiRun(Base, TimestampMixin):
    """One execution of the analysis pipeline over one resource.

    Kept because "the AI found a cyst" is not reviewable without knowing which
    models ran, in what order, on which version of the taxonomy, and what each
    of them took. ``stages`` is the serialised per-stage record.
    """

    __tablename__ = "ai_runs"
    __table_args__ = (
        Index("ix_ai_runs_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_ai_runs_resource_type_resource_id", "resource_type", "resource_id"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    #: ``volume`` or ``study``. Not a foreign key: a run outlives the resource
    #: it describes, exactly like an audit row.
    resource_type: Mapped[str] = mapped_column(String(24), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(26), nullable=False)
    triggered_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    pipeline_name: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[StudyStatus] = mapped_column(
        Enum(StudyStatus, native_enum=False, length=20),
        nullable=False,
        default=StudyStatus.COMPLETED,
    )
    total_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    finding_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_reason: Mapped[str | None] = mapped_column(String(255))
    #: JSON array of stage records: name, kind, version, status, ms, summary.
    stages: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    triggered_by: Mapped[User | None] = relationship()


class TreatmentPlan(Base, TimestampMixin):
    """An ordered set of proposed and agreed work for one patient."""

    __tablename__ = "treatment_plans"
    __table_args__ = (
        Index("ix_treatment_plans_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_treatment_plans_patient_id_status", "patient_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    title: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[PlanStatus] = mapped_column(
        Enum(PlanStatus, native_enum=False, length=16), nullable=False, default=PlanStatus.DRAFT
    )
    notes: Mapped[str | None] = mapped_column(Text)

    #: Whether a clinician assembled this plan or the planner proposed it. A
    #: generated plan is a draft until someone accepts it; the distinction is
    #: kept on the row because it changes what the UI is allowed to imply.
    origin: Mapped[PlanOrigin] = mapped_column(
        Enum(PlanOrigin, native_enum=False, length=12), nullable=False, default=PlanOrigin.MANUAL
    )
    complexity: Mapped[PlanComplexity | None] = mapped_column(
        Enum(PlanComplexity, native_enum=False, length=12)
    )
    #: Calendar span, distinct from chair time: an implant plan is two hours of
    #: work spread over four months of healing.
    estimated_weeks: Mapped[int | None] = mapped_column(Integer)
    risks: Mapped[str | None] = mapped_column(Text)
    follow_up: Mapped[str | None] = mapped_column(Text)
    rationale: Mapped[str | None] = mapped_column(Text)
    source_volume_id: Mapped[int | None] = mapped_column(
        ForeignKey("volumes.id", ondelete="SET NULL")
    )
    source_study_public_id: Mapped[str | None] = mapped_column(String(26))

    patient: Mapped[Patient] = relationship(back_populates="treatment_plans")
    created_by: Mapped[User | None] = relationship()
    items: Mapped[list[TreatmentPlanItem]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="TreatmentPlanItem.position",
    )
    options: Mapped[list[TreatmentOption]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="TreatmentOption.position",
    )


class TreatmentPlanItem(Base, TimestampMixin):
    """One step of a plan: a procedure, optionally on a specific tooth."""

    __tablename__ = "treatment_plan_items"
    __table_args__ = (
        Index("ix_treatment_plan_items_plan_id_position", "plan_id", "position"),
        Index("ix_treatment_plan_items_plan_id_status", "plan_id", "status"),
        CheckConstraint(
            "tooth_number IS NULL OR ("
            "tooth_number / 10 BETWEEN 1 AND 4 AND tooth_number % 10 BETWEEN 1 AND 8)",
            name="plan_item_tooth_number_fdi",
        ),
        CheckConstraint("estimated_visits > 0", name="plan_item_positive_visits"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("treatment_plans.id", ondelete="CASCADE"), nullable=False
    )
    #: Procedure code from ``clinical.protocols``. Stored as a string so a
    #: plan written today keeps its meaning when the table gains entries.
    procedure_code: Mapped[str] = mapped_column(String(48), nullable=False)
    tooth_number: Mapped[int | None] = mapped_column(Integer)
    #: Copied from the protocol at creation rather than looked up on read, so
    #: an agreed plan is not silently re-priced by a later edit to the table.
    priority: Mapped[str] = mapped_column(String(16), nullable=False)
    estimated_visits: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    estimated_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)

    status: Mapped[PlanItemStatus] = mapped_column(
        Enum(PlanItemStatus, native_enum=False, length=16),
        nullable=False,
        default=PlanItemStatus.PROPOSED,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scheduled_for: Mapped[date | None] = mapped_column(Date)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    notes: Mapped[str | None] = mapped_column(Text)

    #: Which detection suggested this step, when one did. Cleared rather than
    #: cascaded if the study is deleted: the agreed work outlives its evidence.
    source_finding_id: Mapped[int | None] = mapped_column(
        ForeignKey("findings.id", ondelete="SET NULL")
    )
    source_study_public_id: Mapped[str | None] = mapped_column(String(26))

    plan: Mapped[TreatmentPlan] = relationship(back_populates="items")


class AuditEvent(Base):
    """Append-only trail of access to patient data.

    No ``updated_at`` and no delete cascade: audit rows outlive the records
    they describe.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_audit_events_actor_id_created_at", "actor_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(BigIntPk, nullable=False)
    actor_id: Mapped[int | None] = mapped_column(BigIntPk)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)


class TreatmentOption(Base, TimestampMixin):
    """One way of treating the same case, alongside the others.

    A plan holds the work; an option holds the *choice*. Three options on one
    plan are three defensible answers to the same findings — conservative,
    standard, comprehensive — and the row records what each costs the patient
    in time, risk and money so the conversation is not conducted from memory.
    """

    __tablename__ = "treatment_options"
    __table_args__ = (
        Index("ix_treatment_options_plan_id_position", "plan_id", "position"),
        CheckConstraint("estimated_visits > 0", name="option_positive_visits"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("treatment_plans.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    title: Mapped[str] = mapped_column(String(160), nullable=False)
    approach: Mapped[TreatmentApproach] = mapped_column(
        Enum(TreatmentApproach, native_enum=False, length=16),
        nullable=False,
        default=TreatmentApproach.STANDARD,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    priority: Mapped[str] = mapped_column(String(16), nullable=False, default="routine")
    complexity: Mapped[PlanComplexity] = mapped_column(
        Enum(PlanComplexity, native_enum=False, length=12),
        nullable=False,
        default=PlanComplexity.MODERATE,
    )
    estimated_visits: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    estimated_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    estimated_weeks: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    benefits: Mapped[str] = mapped_column(Text, nullable=False, default="")
    risks: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Procedure codes, comma-separated, in the order they would be performed.
    procedure_codes: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    is_selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    plan: Mapped[TreatmentPlan] = relationship(back_populates="options")


class PatientNote(Base, TimestampMixin):
    """A dated clinical or administrative note on a patient's record."""

    __tablename__ = "patient_notes"
    __table_args__ = (
        Index("ix_patient_notes_patient_id_created_at", "patient_id", "created_at"),
        Index("ix_patient_notes_organization_id_created_at", "organization_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    kind: Mapped[NoteKind] = mapped_column(
        Enum(NoteKind, native_enum=False, length=16), nullable=False, default=NoteKind.CLINICAL
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="notes_log")
    author: Mapped[User | None] = relationship()


class Appointment(Base, TimestampMixin):
    """A booked visit, optionally realising one step of a treatment plan."""

    __tablename__ = "appointments"
    __table_args__ = (
        Index("ix_appointments_organization_id_starts_at", "organization_id", "starts_at"),
        Index("ix_appointments_patient_id_starts_at", "patient_id", "starts_at"),
        CheckConstraint("duration_minutes > 0", name="appointment_positive_duration"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), nullable=False
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    #: Cleared rather than cascaded: a visit that happened is a fact about the
    #: patient even after the plan step it came from is removed.
    plan_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("treatment_plan_items.id", ondelete="SET NULL")
    )

    title: Mapped[str] = mapped_column(String(160), nullable=False)
    starts_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    status: Mapped[AppointmentStatus] = mapped_column(
        Enum(AppointmentStatus, native_enum=False, length=16),
        nullable=False,
        default=AppointmentStatus.SCHEDULED,
    )
    notes: Mapped[str | None] = mapped_column(Text)

    patient: Mapped[Patient] = relationship(back_populates="appointments")
    created_by: Mapped[User | None] = relationship()


class Comment(Base, TimestampMixin):
    """A threaded discussion attached to any resource in the workspace.

    ``resource_type``/``resource_id`` rather than one nullable foreign key per
    kind: the alternative is a column per resource and a CHECK constraint that
    grows every time the product gains a screen.
    """

    __tablename__ = "comments"
    __table_args__ = (
        Index("ix_comments_resource_type_resource_id", "resource_type", "resource_id"),
        Index("ix_comments_organization_id_created_at", "organization_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    author_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    resource_type: Mapped[str] = mapped_column(String(24), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("comments.id", ondelete="CASCADE"))

    body: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    resolved_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    author: Mapped[User | None] = relationship(foreign_keys=[author_id])
    replies: Mapped[list[Comment]] = relationship(
        back_populates="parent", cascade="all, delete-orphan"
    )
    # `remote_side` names the parent end of the self-reference. The reference
    # is to this class's own `id` column, which the linter reads as shadowing
    # the builtin at the point of use.
    parent: Mapped[Comment | None] = relationship(
        back_populates="replies", remote_side="Comment.id"
    )


class ReviewAssignment(Base, TimestampMixin):
    """A request that a named colleague look at a case.

    Sharing inside the clinic rather than a public link. A URL that grants
    access to a patient record without a session is a liability the product
    does not need, so a case is shared *to a user*, and the audit trail keeps
    naming a person.
    """

    __tablename__ = "review_assignments"
    __table_args__ = (
        Index("ix_review_assignments_assignee_id_status", "assignee_id", "status"),
        Index("ix_review_assignments_organization_id_created_at", "organization_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    resource_type: Mapped[str] = mapped_column(String(24), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(32), nullable=False)
    assignee_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    assigned_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    status: Mapped[AssignmentStatus] = mapped_column(
        Enum(AssignmentStatus, native_enum=False, length=16),
        nullable=False,
        default=AssignmentStatus.PENDING,
    )
    due_on: Mapped[date | None] = mapped_column(Date)
    note: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    assignee: Mapped[User] = relationship(foreign_keys=[assignee_id])
    assigned_by: Mapped[User | None] = relationship(foreign_keys=[assigned_by_id])


class Notification(Base):
    """One entry in a user's notification centre.

    No ``updated_at``: the only mutation is marking it read, which has its own
    timestamp and is more useful than a generic one.
    """

    __tablename__ = "notifications"
    __table_args__ = (
        Index("ix_notifications_user_id_created_at", "user_id", "created_at"),
        # Serves the unread badge, which is the most frequent query in the app.
        Index("ix_notifications_user_id_read_at", "user_id", "read_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[NotificationKind] = mapped_column(
        Enum(NotificationKind, native_enum=False, length=24), nullable=False
    )
    tone: Mapped[NotificationTone] = mapped_column(
        Enum(NotificationTone, native_enum=False, length=12),
        nullable=False,
        default=NotificationTone.INFO,
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    body: Mapped[str | None] = mapped_column(Text)
    href: Mapped[str | None] = mapped_column(String(255))
    read_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)

    user: Mapped[User] = relationship()


class CaseEntry(Base, TimestampMixin):
    """A teaching case: what was seen, what was done, how it turned out.

    Deliberately a copy rather than a view over the live record. A library
    entry is a statement about a finished case; letting it re-read a plan that
    has since been edited would make yesterday's teaching material change
    overnight.
    """

    __tablename__ = "case_entries"
    __table_args__ = (
        Index("ix_case_entries_organization_id_created_at", "organization_id", "created_at"),
        Index("ix_case_entries_organization_id_search_text", "organization_id", "search_text"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    title: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    diagnosis: Mapped[str] = mapped_column(Text, nullable=False, default="")
    treatment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    outcome: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Comma-separated finding keys, so a case can be found by what is in it.
    finding_keys: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    tags: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    #: Lower-cased title + summary + diagnosis + tags, on the same reasoning as
    #: ``Patient.search_text``.
    search_text: Mapped[str] = mapped_column(String(1024), nullable=False, default="")

    #: Kept as public ids rather than foreign keys: a library entry survives
    #: the deletion of the study it was written from.
    study_public_id: Mapped[str | None] = mapped_column(String(26))
    volume_public_id: Mapped[str | None] = mapped_column(String(26))
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id", ondelete="SET NULL"))

    created_by: Mapped[User | None] = relationship()

    def refresh_search_text(self) -> None:
        parts = (self.title, self.summary, self.diagnosis, self.treatment, self.tags)
        self.search_text = " ".join(part for part in parts if part).lower()[:1024]


class AssistantThread(Base, TimestampMixin):
    """A conversation about one case."""

    __tablename__ = "assistant_threads"
    __table_args__ = (
        Index("ix_assistant_threads_user_id_created_at", "user_id", "created_at"),
        Index("ix_assistant_threads_organization_id_created_at", "organization_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(26), nullable=False, unique=True, index=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    volume_id: Mapped[int | None] = mapped_column(ForeignKey("volumes.id", ondelete="CASCADE"))
    study_public_id: Mapped[str | None] = mapped_column(String(26))
    title: Mapped[str] = mapped_column(String(160), nullable=False, default="Новый разговор")

    messages: Mapped[list[AssistantMessage]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="AssistantMessage.id",
    )


class AssistantMessage(Base):
    """One turn of an assistant conversation.

    ``citations`` is the serialised list of records the answer was built from.
    An assistant that cannot show its working is not usable on patient data,
    so an answer without citations is a bug rather than a style choice.
    """

    __tablename__ = "assistant_messages"
    __table_args__ = (Index("ix_assistant_messages_thread_id_id", "thread_id", "id"),)

    id: Mapped[int] = mapped_column(BigIntPk, primary_key=True)
    thread_id: Mapped[int] = mapped_column(
        ForeignKey("assistant_threads.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[AssistantRole] = mapped_column(
        Enum(AssistantRole, native_enum=False, length=12), nullable=False
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    #: Which question shape the router matched, for evaluating coverage.
    intent: Mapped[str | None] = mapped_column(String(48))
    citations: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, nullable=False, index=True)

    thread: Mapped[AssistantThread] = relationship(back_populates="messages")
