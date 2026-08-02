"""The analysis engine: several models, run in order, over one volume.

A single detector was enough for a radiograph — one image in, boxes out. A
CBCT reconstruction is not that shape. Deciding whether a lucency at a root
apex is a lesion needs to know where bone ends and air begins, which needs the
volume segmented; deciding whether the *finding* is trustworthy needs to know
whether the patient moved during the scan. Those are separate questions, they
fail separately, and they will be replaced by trained networks on separate
schedules.

So this module is a small orchestrator rather than a model:

* A :class:`Stage` is anything with a name, a version and a ``run``. It reads
  the volume and whatever earlier stages wrote into :class:`PipelineState`,
  and writes its own results back.
* A :class:`Pipeline` runs stages in order, times each one, and — importantly
  — isolates their failures. A segmentation model that throws marks its own
  stage failed and lets the rest of the pipeline continue with whatever it
  did produce. The alternative, one exception discarding an eight-second
  analysis, is the wrong trade for clinical software.
* Every run is described by a :class:`RunRecord`, which is what gets persisted
  and shown in the UI. "The AI found a cyst" is not reviewable; "the detection
  stage, v3, found it in 240 ms on a volume the QC stage scored 0.82" is.

Replacing a stage with a trained network is implementing this protocol and
changing one line in :mod:`dentist_ai.ml.cbct`. Nothing above this module
knows how many stages there are or what any of them does.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from dentist_ai.core.logging import get_logger
from dentist_ai.ml import volumetrics
from dentist_ai.ml.cbct_taxonomy import Region

log = get_logger(__name__)

VoxelArray = npt.NDArray[np.uint8]


class StageKind(enum.StrEnum):
    """What a stage contributes, independent of how it is implemented."""

    QUALITY_CONTROL = "quality_control"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    REPORT = "report"
    TREATMENT = "treatment"


STAGE_KIND_LABELS: dict[StageKind, str] = {
    StageKind.QUALITY_CONTROL: "Контроль качества",
    StageKind.SEGMENTATION: "Сегментация",
    StageKind.DETECTION: "Детекция",
    StageKind.CLASSIFICATION: "Классификация",
    StageKind.REPORT: "Синтез заключения",
    StageKind.TREATMENT: "Планирование лечения",
}


class StageStatus(enum.StrEnum):
    OK = "ok"
    #: Ran, but had nothing to do — a TMJ stage on an implant-site volume.
    SKIPPED = "skipped"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VolumeInput:
    """Everything a stage is allowed to see about the scan.

    Deliberately free of database objects. A stage takes voxels and geometry;
    giving it an ORM row would let a model reach the patient's name, and would
    make the pipeline untestable without a session.
    """

    #: ``[z, y, x]`` in stored 8-bit units.
    voxels: VoxelArray
    #: Millimetres per voxel, in ``(x, y, z)`` order.
    spacing: tuple[float, float, float]
    #: ``modality = stored * hu_slope + hu_intercept``.
    hu_slope: float
    hu_intercept: float
    #: What the scanner was aimed at, as a lower-case string matching
    #: ``VolumeFieldOfView``. Bounds what a stage may claim to have seen.
    field_of_view: str
    #: Age in years, when the record has a date of birth. A radiolucency at an
    #: unerupted third molar means something different at 14 and at 45.
    patient_age: int | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        depth, height, width = self.voxels.shape
        return int(depth), int(height), int(width)

    @property
    def voxel_volume_mm3(self) -> float:
        return self.spacing[0] * self.spacing[1] * self.spacing[2]

    def to_hu(self, stored: float) -> float:
        return stored * self.hu_slope + self.hu_intercept

    def from_hu(self, hounsfield: float) -> float:
        return (hounsfield - self.hu_intercept) / self.hu_slope


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    """A prism normalised to ``[0, 1]`` on each axis of the stored volume."""

    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float

    def clamped(self) -> BoundingBox3D:
        """Clip into the unit cube.

        The database has a CHECK constraint that would reject a box a voxel
        outside the volume; normalising here rather than discovering it at
        INSERT time keeps the failure in the layer that can fix it.
        """
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        z = min(max(self.z, 0.0), 1.0)
        return BoundingBox3D(
            x=x,
            y=y,
            z=z,
            width=min(max(self.width, 1e-4), 1.0 - x),
            height=min(max(self.height, 1e-4), 1.0 - y),
            depth=min(max(self.depth, 1e-4), 1.0 - z),
        )

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
            self.z + self.depth / 2,
        )


def box_from_grid(
    bounds: tuple[int, int, int, int, int, int],
    shape: tuple[int, int, int],
) -> BoundingBox3D:
    """Analysis-grid bounds ``(z0, z1, y0, y1, x0, x1)`` to a normalised box.

    Stages work on a coarse grid; findings are stored against the volume.
    Normalising here is what keeps the two from having to agree on a
    resolution — re-ingesting a scan at a different decimation factor moves
    every grid index and no stored box.
    """
    z0, z1, y0, y1, x0, x1 = bounds
    depth, height, width = shape
    return BoundingBox3D(
        x=x0 / max(width, 1),
        y=y0 / max(height, 1),
        z=z0 / max(depth, 1),
        width=(x1 - x0) / max(width, 1),
        height=(y1 - y0) / max(height, 1),
        depth=(z1 - z0) / max(depth, 1),
    ).clamped()


@dataclass(frozen=True, slots=True)
class Candidate:
    """A region the detection stage thinks is worth classifying.

    Untyped on purpose: detection answers "something is here and here is what
    it looks like", classification answers "this is what it is". Keeping them
    apart is what lets either be swapped for a trained model without the other
    changing.
    """

    box: BoundingBox3D
    #: Strength of the detection signal, before a class is assigned.
    salience: float
    #: Measured properties the classifier reasons over: mean density in HU,
    #: volume in mm³, elongation, and how enclosed by bone the region is.
    features: dict[str, float]
    region: Region


@dataclass(frozen=True, slots=True)
class VolumeDetection:
    """A classified finding, ready to be persisted."""

    class_key: str
    confidence: float
    box: BoundingBox3D
    region: Region
    produced_by: str
    tooth_number: int | None = None
    extent_mm: float | None = None
    mean_density: float | None = None


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """What the QC stage concluded about the acquisition itself."""

    #: Overall usability in ``[0, 1]``. Never suppresses findings; it is shown
    #: beside them so a reader can weigh them.
    score: float
    noise: float
    motion: float
    metal: float
    coverage: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageRecord:
    """What one stage did, for the run log."""

    name: str
    kind: StageKind
    version: str
    status: StageStatus
    duration_ms: int
    summary: str

    @property
    def kind_label(self) -> str:
        return STAGE_KIND_LABELS[self.kind]


@dataclass(slots=True)
class PipelineState:
    """The accumulator stages read from and write to.

    Mutable, and passed to every stage — which is the point. A classification
    stage needs the segmentation stage's masks, and threading them through
    return values would fix the stage order into the type system.
    """

    quality: QualityAssessment | None = None
    #: The volume resampled onto the analysis grid. Written by the first stage
    #: that needs it and reused by the rest: four stages each block-averaging
    #: 16 M voxels would spend most of the run doing the same arithmetic.
    grid: volumetrics.Grid | None = None
    #: Per-voxel tissue class from the segmentation stage, on the coarse
    #: analysis grid rather than at full resolution.
    tissue: npt.NDArray[np.uint8] | None = None
    #: Landmarks the segmentation stage located, normalised to ``[0, 1]``:
    #: occlusal plane height, arch centre, jaw split, and so on.
    landmarks: dict[str, float] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    detections: list[VolumeDetection] = field(default_factory=list)
    #: Procedure codes the treatment stage derived, in the order it would
    #: perform them. Consumed by the planner, which turns them into a plan a
    #: clinician can accept or discard.
    recommendations: list[str] = field(default_factory=list)
    #: Free-form notes a stage wants to surface in the run log.
    notes: list[str] = field(default_factory=list)


def ensure_grid(volume: VolumeInput, state: PipelineState) -> volumetrics.Grid:
    """The analysis grid, computed once per run.

    Stages call this rather than resampling themselves, so a pipeline built
    from a different set of stages still pays for the resample exactly once —
    and so a stage remains runnable on its own in a test, where nothing has
    populated the state yet.
    """
    if state.grid is None:
        state.grid = volumetrics.to_grid(
            volume.voxels,
            volume.spacing,
            hu_slope=volume.hu_slope,
            hu_intercept=volume.hu_intercept,
        )
    return state.grid


@runtime_checkable
class Stage(Protocol):
    """Anything that can contribute to an analysis."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> StageKind: ...

    @property
    def version(self) -> str: ...

    def applies_to(self, volume: VolumeInput) -> bool:
        """Whether this stage has anything to say about this volume."""
        ...

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        """Do the work, mutating ``state``. Returns a one-line summary."""
        ...


@dataclass(frozen=True, slots=True)
class RunRecord:
    """The complete result of one pipeline execution."""

    pipeline_name: str
    pipeline_version: str
    stages: tuple[StageRecord, ...]
    detections: tuple[VolumeDetection, ...]
    quality: QualityAssessment | None
    landmarks: dict[str, float]
    total_ms: int

    @property
    def failed_stages(self) -> tuple[StageRecord, ...]:
        return tuple(item for item in self.stages if item.status is StageStatus.FAILED)

    @property
    def succeeded(self) -> bool:
        """Whether enough ran for the result to be worth showing.

        A pipeline is useful as long as classification produced something; a
        failed treatment-recommendation stage costs the user a panel, not the
        analysis.
        """
        blocking = {StageKind.DETECTION, StageKind.CLASSIFICATION}
        return not any(item.kind in blocking for item in self.failed_stages)


class Pipeline:
    """An ordered set of stages, run over one volume."""

    def __init__(self, name: str, version: str, stages: tuple[Stage, ...]) -> None:
        self._name = name
        self._version = version
        self._stages = stages

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    def describe(self) -> tuple[StageRecord, ...]:
        """The pipeline's shape without running it, for the model registry UI."""
        return tuple(
            StageRecord(
                name=stage.name,
                kind=stage.kind,
                version=stage.version,
                status=StageStatus.SKIPPED,
                duration_ms=0,
                summary="",
            )
            for stage in self._stages
        )

    def run(self, volume: VolumeInput) -> RunRecord:
        state = PipelineState()
        ensure_grid(volume, state)
        records: list[StageRecord] = []
        started = time.perf_counter()

        for stage in self._stages:
            records.append(self._run_stage(stage, volume, state))

        return RunRecord(
            pipeline_name=self._name,
            pipeline_version=self._version,
            stages=tuple(records),
            detections=tuple(state.detections),
            quality=state.quality,
            landmarks=dict(state.landmarks),
            total_ms=int((time.perf_counter() - started) * 1000),
        )

    def _run_stage(self, stage: Stage, volume: VolumeInput, state: PipelineState) -> StageRecord:
        started = time.perf_counter()

        def record(status: StageStatus, summary: str) -> StageRecord:
            return StageRecord(
                name=stage.name,
                kind=stage.kind,
                version=stage.version,
                status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
                summary=summary,
            )

        if not stage.applies_to(volume):
            return record(StageStatus.SKIPPED, "Не применимо к этому объёму")

        try:
            summary = stage.run(volume, state)
        except Exception as exc:
            # Logged with the stage name so a recurring failure is greppable,
            # and reported to the user as a degraded run rather than an error
            # page over an analysis that mostly worked.
            log.exception("pipeline_stage_failed", stage=stage.name, version=stage.version)
            return record(StageStatus.FAILED, f"{type(exc).__name__}: {exc}"[:200])

        return record(StageStatus.OK, summary)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    """Named pipelines, so a deployment can carry more than one.

    A clinic doing implantology and a clinic doing orthodontics want different
    stages emphasised, and a research deployment wants a pipeline with an
    experimental stage in it that production does not have.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Pipeline] = {}
        self._default: str | None = None

    def register(self, pipeline: Pipeline, *, default: bool = False) -> None:
        self._pipelines[pipeline.name] = pipeline
        if default or self._default is None:
            self._default = pipeline.name

    def get(self, name: str | None = None) -> Pipeline:
        key = name or self._default
        if key is None or key not in self._pipelines:
            msg = f"No pipeline registered under {key!r}"
            raise KeyError(msg)
        return self._pipelines[key]

    def names(self) -> tuple[str, ...]:
        return tuple(self._pipelines)

    def all(self) -> tuple[Pipeline, ...]:
        return tuple(self._pipelines.values())
