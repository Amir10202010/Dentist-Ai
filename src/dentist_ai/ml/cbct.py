"""Composition root for CBCT analysis.

The one place that knows which stages exist and in what order. Swapping the
heuristic classifier for a trained network is a change to
:data:`_DEFAULT_STAGES`; nothing above this module is aware of the difference.

Two pipelines ship, and they differ in what they are permitted to say rather
than in how well they see. ``dental-cbct`` reads the scan and derives the care
pathway from it. ``radiology-review`` runs the identical reading and stops
before the treatment stage: a radiology service reports findings and does not
propose treatment, and the cleanest way to guarantee that is to not run the
stage that would.

## What the default backend is, and is not

There is no trained volumetric network in this repository, and the pipeline
does not pretend otherwise. The stages are real image analysis — Otsu
thresholding, connected components, morphological enclosure tests, crest-height
profiling — applied by explicit clinical rules. Findings are genuinely derived
from the voxels rather than seeded from a hash, which is what makes the viewer,
the report and the assistant honest end to end.

What that buys, and what it costs, is worth stating plainly. It buys a system
that runs on a clinic's existing hardware in a couple of seconds, whose every
decision is auditable, and whose failure modes are legible. It costs
sensitivity on exactly the findings a convolutional network is best at — early
caries, subtle resorption, a hairline fracture. The interface says so, the
confidences are capped accordingly, and the classes that must never be read as
a diagnosis are capped hardest.
"""

from __future__ import annotations

from typing import Final

from dentist_ai.ml.pipeline import ModelRegistry, Pipeline, Stage
from dentist_ai.ml.stages.classification import FindingClassificationStage
from dentist_ai.ml.stages.detection import LesionDetectionStage
from dentist_ai.ml.stages.quality import QualityControlStage
from dentist_ai.ml.stages.segmentation import AnatomySegmentationStage
from dentist_ai.ml.stages.synthesis import (
    ReportSynthesisStage,
    TreatmentRecommendationStage,
)

#: Bumped when a change alters what the pipeline reports, so a stored run can
#: be compared against the version that produced it.
PIPELINE_VERSION: Final[str] = "1.4.0"

DEFAULT_PIPELINE: Final[str] = "dental-cbct"
RADIOLOGY_PIPELINE: Final[str] = "radiology-review"


def _reading_stages() -> tuple[Stage, ...]:
    """Everything up to and including a consolidated finding list.

    The order is a dependency order, not a preference: segmentation needs the
    grid, detection needs the tissue map, classification needs the landmarks,
    and synthesis needs all three.
    """
    return (
        QualityControlStage(),
        AnatomySegmentationStage(),
        LesionDetectionStage(),
        FindingClassificationStage(),
        ReportSynthesisStage(),
    )


def build_registry() -> ModelRegistry:
    """Construct the registry a process serves from."""
    registry = ModelRegistry()
    registry.register(
        Pipeline(
            DEFAULT_PIPELINE,
            PIPELINE_VERSION,
            (*_reading_stages(), TreatmentRecommendationStage()),
        ),
        default=True,
    )
    registry.register(Pipeline(RADIOLOGY_PIPELINE, PIPELINE_VERSION, _reading_stages()))
    return registry
