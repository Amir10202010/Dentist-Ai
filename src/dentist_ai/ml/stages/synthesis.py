"""The last two stages: make the finding list readable, then act on it.

Everything upstream is generous — detection favours sensitivity, and several
rules can fire on overlapping regions. What reaches a clinician has to be the
opposite: one entry per real thing, ordered by what needs attention first, with
confidences that already account for how good the scan was.

:class:`ReportSynthesisStage` does that consolidation. :class:`TreatmentRecommendationStage`
then turns the surviving findings into an ordered set of procedure codes, which
is the input the treatment planner builds a plan from. It stops at codes
deliberately: the planner is in ``clinical/``, where a clinician can audit it,
and a stage in ``ml/`` proposing treatment directly would put that decision
inside the model layer.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from dentist_ai.ml.cbct_taxonomy import VolumeCategory, by_key
from dentist_ai.ml.pipeline import (
    BoundingBox3D,
    PipelineState,
    StageKind,
    VolumeDetection,
    VolumeInput,
)
from dentist_ai.ml.taxonomy import Severity

#: Overlap above which two findings of the same class are the same thing seen
#: twice. Generous, because two rules firing on one lesion produce boxes that
#: agree closely.
_MERGE_IOU: Final[float] = 0.34
#: Ceiling on reported findings. A CBCT of a restored mouth can produce
#: hundreds of dense bodies; past this the list stops being read.
_MAX_FINDINGS: Final[int] = 48
#: How far a poor-quality scan is allowed to pull confidence down. At quality
#: 0, a finding keeps 55% of the confidence it would have had on a clean scan.
_QUALITY_FLOOR: Final[float] = 0.55


class ReportSynthesisStage:
    """Deduplicates, calibrates against scan quality, and orders for triage."""

    name = "report-synthesis"
    kind = StageKind.REPORT
    version = "1.4.0"

    def applies_to(self, volume: VolumeInput) -> bool:
        return True

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        before = len(state.detections)
        merged = _merge_overlapping(state.detections)
        calibrated = [self._calibrate(detection, state) for detection in merged]

        calibrated.sort(key=lambda item: (by_key(item.class_key).severity.rank, -item.confidence))
        state.detections = calibrated[:_MAX_FINDINGS]

        attention = sum(1 for item in state.detections if by_key(item.class_key).needs_attention)
        state.notes.append(
            f"Итог: {len(state.detections)} находок, из них требующих внимания {attention}."
        )
        return f"{before} → {len(state.detections)} после слияния · внимание: {attention}"

    @staticmethod
    def _calibrate(detection: VolumeDetection, state: PipelineState) -> VolumeDetection:
        """Scale confidence by how readable the scan was.

        Quality findings are exempt: the confidence that a scan has motion
        artefact should not be reduced because the scan has motion artefact.
        """
        quality = state.quality
        taxonomy = by_key(detection.class_key)
        if quality is None or taxonomy.category is VolumeCategory.QUALITY:
            return detection

        factor = _QUALITY_FLOOR + (1.0 - _QUALITY_FLOOR) * quality.score
        return VolumeDetection(
            class_key=detection.class_key,
            confidence=round(float(np.clip(detection.confidence * factor, 0.01, 1.0)), 4),
            box=detection.box,
            region=detection.region,
            produced_by=detection.produced_by,
            tooth_number=detection.tooth_number,
            extent_mm=detection.extent_mm,
            mean_density=detection.mean_density,
        )


class TreatmentRecommendationStage:
    """Derives the procedure codes the findings imply, in the order to do them."""

    name = "care-pathway"
    kind = StageKind.TREATMENT
    version = "1.3.0"

    def applies_to(self, volume: VolumeInput) -> bool:
        return True

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        if not state.detections:
            state.recommendations = []
            return "находок нет — план не требуется"

        # Urgency first, and within it the order the taxonomy lists, which is
        # diagnosis before intervention: a CBCT referral precedes the surgery
        # it informs.
        ordered: list[tuple[int, str]] = []
        for detection in state.detections:
            taxonomy = by_key(detection.class_key)
            rank = taxonomy.severity.rank
            ordered.extend((rank, code) for code in taxonomy.procedures)

        ordered.sort(key=lambda item: item[0])

        seen: set[str] = set()
        codes: list[str] = []
        for _rank, code in ordered:
            if code not in seen:
                seen.add(code)
                codes.append(code)

        state.recommendations = codes
        urgent = sum(
            1
            for detection in state.detections
            if by_key(detection.class_key).severity in (Severity.CRITICAL, Severity.HIGH)
        )
        return f"{len(codes)} процедур(ы) · срочных находок: {urgent}"


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
def _merge_overlapping(detections: list[VolumeDetection]) -> list[VolumeDetection]:
    """Keep the most confident of each cluster of same-class overlapping boxes.

    Class-aware on purpose. A dense body inside a lucency is two true findings
    — a root filling and the lesion around it — and suppressing one because
    the boxes overlap would delete the clinically interesting half.
    """
    kept: list[VolumeDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            other.class_key == detection.class_key and _iou(other.box, detection.box) > _MERGE_IOU
            for other in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def _iou(first: BoundingBox3D, second: BoundingBox3D) -> float:
    overlap_x = _overlap(first.x, first.width, second.x, second.width)
    overlap_y = _overlap(first.y, first.height, second.y, second.height)
    overlap_z = _overlap(first.z, first.depth, second.z, second.depth)
    intersection = overlap_x * overlap_y * overlap_z
    if intersection <= 0:
        return 0.0

    union = (
        first.width * first.height * first.depth
        + second.width * second.height * second.depth
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _overlap(a_start: float, a_size: float, b_start: float, b_size: float) -> float:
    return max(0.0, min(a_start + a_size, b_start + b_size) - max(a_start, b_start))
