"""Classification: what each detected region actually is.

Detection measured regions; this stage names them. It does so with an explicit
rule per finding class rather than a black box, and that is a deliberate
choice rather than a placeholder for one.

Each rule has two parts, and separating them is the whole design:

**A gate** — the conditions that are *definitional*. A dental implant is buried
in bone below the alveolar crest. A region sitting at the occlusal plane with
air above it is a crown, and no amount of being the right size and density
makes it a fixture. Gates are boolean and they veto: fail one and the class is
not considered at all.

**A score** — the conditions that are *evidential*. Given that a region is
buried in bone, how implant-like are its size, shape and radiopacity? These are
blended into a confidence.

Folding both into one weighted average — the obvious way to write this — is
what makes heuristic classifiers untrustworthy: enough weak positive evidence
outvotes a hard anatomical impossibility, and every false positive worth the
name comes from that. Keeping them apart also produces a better explanation.
"Buried in bone, below the crest, 220 mm³, cylindrical" is a sentence a
clinician can check.

Density carries more of the work here than shape does, because it is the one
measurement that separates the pairs geometry cannot. A cyst and a marrow space
are the same size and shape; one is fluid and the other is bone. A lesion and a
maxillary sinus are both radiolucent; one is water density and the other is
air. Those two facts remove most of the false positives this stage would
otherwise produce.

The scores are conservative by construction: nothing exceeds 0.9, and the
classes that must never be read as a diagnosis are capped far lower. When a
trained network replaces this stage it implements the same protocol and emits
the same :class:`VolumeDetection`; everything downstream is unaffected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from dentist_ai.ml import volumetrics
from dentist_ai.ml.cbct_taxonomy import Region, by_key
from dentist_ai.ml.pipeline import (
    BoundingBox3D,
    Candidate,
    PipelineState,
    StageKind,
    VolumeDetection,
    VolumeInput,
)
from dentist_ai.ml.stages.segmentation import BONE, DENSE, region_for

Features = dict[str, float]

#: A class must beat this to be reported at all. Everything below it is a
#: region the classifier does not recognise, which is a legitimate answer.
_SCORE_FLOOR: Final[float] = 0.45
#: Ceilings by class. Findings that can only ever be a referral — a mass, a
#: fracture line — are capped well below the rest, so the interface can never
#: present one as settled.
_CEILINGS: Final[dict[str, float]] = {
    "suspicious_mass": 0.52,
    "root_fracture": 0.58,
    "abscess": 0.72,
    "odontogenic_infection": 0.78,
    "caries_3d": 0.70,
}
_DEFAULT_CEILING: Final[float] = 0.90

#: Density window, in Hounsfield units, for the contents of a pathological
#: radiolucency. The upper bound excludes trabecular marrow, which is
#: geometrically identical to a small lesion and several hundred units denser.
#: The lower bound excludes air — which is what a maxillary sinus and the
#: airway are: both radiolucent, neither a lesion.
_LESION_HU_LOW: Final[float] = -120.0
_LESION_HU_HIGH: Final[float] = 260.0
#: A genuine fluid-filled lesion is uniformly lucent. A partial-volume artefact
#: at a cortical boundary averages low while still containing bone-density
#: voxels, which a ceiling on the peak catches and a ceiling on the mean does
#: not.
_LESION_PEAK_HU: Final[float] = 780.0

#: Crest depression, in millimetres below the arch's own median crest, that
#: counts as bone loss.
_BONE_LOSS_MM: Final[float] = 2.6
#: Share of arch sites that must show loss before it is called generalised
#: rather than localised. Thirty per cent is the boundary the 2017 World
#: Workshop classification draws, so the number is clinical rather than tuned.
_GENERALISED_SHARE: Final[float] = 0.30
#: Below the generalised share but above this, the loss is reported as
#: localised instead of suppressed.
_LOCALISED_SHARE: Final[float] = 0.10
#: Left-right bone volume difference that reads as a real asymmetry rather
#: than as the patient being a few degrees off-centre in the chair.
_ASYMMETRY_FLOOR: Final[float] = 0.075
#: Condylar volume difference between sides that warrants a TMJ finding.
_CONDYLE_DIFFERENCE: Final[float] = 0.18
#: Most instances of one class in a single report. Six entries reading
#: "odontogenic infection" is never a true reading of one scan; it is one rule
#: firing repeatedly on the same anatomy, and truncating is more honest than
#: presenting it as six separate diseases.
_PER_CLASS_LIMIT: Final[int] = 4


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------
def _band(value: float, low: float, high: float, *, slack: float | None = None) -> float:
    """1 inside ``[low, high]``, falling linearly to 0 over ``slack`` outside."""
    span = max(high - low, 1e-6)
    reach = slack if slack is not None else span * 0.6
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - distance / max(reach, 1e-6))


def _above(value: float, threshold: float, *, slack: float) -> float:
    if value >= threshold:
        return 1.0
    return max(0.0, 1.0 - (threshold - value) / max(slack, 1e-6))


def _below(value: float, threshold: float, *, slack: float) -> float:
    if value <= threshold:
        return 1.0
    return max(0.0, 1.0 - (value - threshold) / max(slack, 1e-6))


def _blend(*terms: tuple[float, float]) -> float:
    """Weighted mean of ``(weight, score)`` pairs."""
    total = sum(weight for weight, _ in terms)
    if total <= 0:
        return 0.0
    return sum(weight * score for weight, score in terms) / total


def _is_lesion_density(features: Features) -> bool:
    """Whether the contents could be a fluid or soft-tissue lesion at all."""
    mean = features.get("mean_hu", 0.0)
    peak = features.get("max_hu", 0.0)
    return _LESION_HU_LOW <= mean <= _LESION_HU_HIGH and peak <= _LESION_PEAK_HU


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Rule:
    class_key: str
    #: Necessary conditions. Failing any one rules the class out entirely.
    gate: Callable[[Features], bool]
    #: Evidential conditions, blended into a confidence.
    score: Callable[[Features], float]


# -- lucencies --------------------------------------------------------------
def _gate_cyst(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        and features["volume_mm3"] >= 250.0
        and features["enclosure"] >= 0.70
        # A cyst expands; it does not track along a canal. Anything this
        # elongated is a vessel, a canal or a marrow channel.
        and features["elongation"] <= 3.0
    )


def _score_cyst(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 350.0, 9000.0, slack=900.0)),
        (3.0, _above(features["enclosure"], 0.78, slack=0.2)),
        (2.0, _above(features["fill"], 0.30, slack=0.24)),
        (2.0, _below(features["elongation"], 2.0, slack=1.2)),
    )


def _gate_apical_lesion(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        and 25.0 <= features["volume_mm3"] <= 420.0
        and features["enclosure"] >= 0.70
        and features["elongation"] <= 3.2
        # Periapical means at an apex, which is within a root's length of the
        # occlusal plane. A lucency up in the ramus is something else.
        and features["distance_to_occlusal"] <= 0.26
    )


def _score_apical_lesion(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 40.0, 320.0, slack=160.0)),
        (2.5, _above(features["enclosure"], 0.80, slack=0.2)),
        (2.0, _below(features["elongation"], 2.2, slack=1.2)),
        (2.5, _below(features["distance_to_occlusal"], 0.18, slack=0.12)),
        (2.0, _above(features["fill"], 0.25, slack=0.2)),
    )


def _gate_infection(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        and features["volume_mm3"] >= 80.0
        # Neither fully contained nor open: inflammation spreading through
        # bone leaves a partly intact boundary.
        and 0.55 <= features["enclosure"] <= 0.85
        and features["distance_to_occlusal"] <= 0.30
        and features["elongation"] <= 4.0
    )


def _score_infection(features: Features) -> float:
    return _blend(
        (2.5, _band(features["volume_mm3"], 120.0, 1400.0, slack=600.0)),
        (2.5, _band(features["enclosure"], 0.58, 0.78, slack=0.12)),
        (2.0, _below(features["fill"], 0.5, slack=0.26)),
        (2.0, _below(features["distance_to_occlusal"], 0.22, slack=0.14)),
    )


def _gate_abscess(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        and features["volume_mm3"] >= 110.0
        # A breached cortical plate: less enclosed than an intact lesion, but
        # still recognisably intraosseous.
        and 0.45 <= features["enclosure"] < 0.60
        and features["distance_to_occlusal"] <= 0.30
    )


def _score_abscess(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 150.0, 2200.0, slack=800.0)),
        (3.0, _band(features["enclosure"], 0.46, 0.58, slack=0.1)),
        (2.0, _below(features["fill"], 0.44, slack=0.24)),
    )


def _gate_suspicious_mass(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        and features["volume_mm3"] >= 1600.0
        # Indistinct margins: it fills its own bounding box poorly, which is
        # what an infiltrative border looks like once measured.
        and features["fill"] <= 0.38
        and features["enclosure"] <= 0.80
    )


def _score_suspicious_mass(features: Features) -> float:
    return _blend(
        (3.0, _above(features["volume_mm3"], 2400.0, slack=1400.0)),
        (3.0, _below(features["fill"], 0.30, slack=0.16)),
        (2.0, _band(features["enclosure"], 0.5, 0.72, slack=0.16)),
    )


def _gate_canal(features: Features) -> bool:
    return (
        features["upper"] < 0.5
        and features["elongation"] >= 3.5
        and features["enclosure"] >= 0.65
        and features["extent_mm"] >= 8.0
        # Neurovascular content: neither bone nor air.
        and -200.0 <= features["mean_hu"] <= 500.0
    )


def _score_canal(features: Features) -> float:
    return _blend(
        (4.0, _above(features["elongation"], 5.0, slack=2.5)),
        (3.0, _above(features["extent_mm"], 18.0, slack=12.0)),
        (2.0, _above(features["enclosure"], 0.8, slack=0.2)),
    )


def _gate_root_fracture(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        # Thin, long, wholly inside mineralised tissue, and at root level.
        and features["elongation"] >= 4.5
        and 8.0 <= features["volume_mm3"] <= 90.0
        and features["enclosure"] >= 0.85
        and features["distance_to_occlusal"] <= 0.20
        and features["extent_mm"] >= 4.0
    )


def _score_root_fracture(features: Features) -> float:
    return _blend(
        (4.0, _above(features["elongation"], 6.0, slack=3.0)),
        (3.0, _below(features["volume_mm3"], 45.0, slack=45.0)),
        (2.0, _above(features["enclosure"], 0.92, slack=0.1)),
    )


def _gate_caries(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        # Within the crown: at the occlusal plane, small, and surrounded by
        # hard tissue on every side.
        and features["volume_mm3"] <= 60.0
        and features["distance_to_occlusal"] <= 0.07
        and features["enclosure"] >= 0.85
        and features["elongation"] <= 2.5
    )


def _score_caries(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 6.0, 45.0, slack=25.0)),
        (3.0, _below(features["distance_to_occlusal"], 0.04, slack=0.04)),
        (2.0, _below(features["elongation"], 1.8, slack=0.9)),
    )


def _gate_bone_loss(features: Features) -> bool:
    return (
        _is_lesion_density(features)
        # An angular defect opens toward the crest, so it is bounded on fewer
        # sides than a contained lesion and sits right at the margin.
        and 0.40 <= features["enclosure"] < 0.70
        and features["distance_to_occlusal"] <= 0.12
        and features["volume_mm3"] >= 40.0
        # A defect is a pocket, not a plane. Without this the soft-tissue
        # sheet in the occlusal gap — thin, wide and at exactly the right
        # height — reads as bone loss on every scan.
        and features["elongation"] <= 6.0
    )


def _score_bone_loss(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 60.0, 900.0, slack=400.0)),
        (3.0, _band(features["enclosure"], 0.45, 0.66, slack=0.12)),
        (3.0, _below(features["distance_to_occlusal"], 0.08, slack=0.06)),
    )


# -- dense bodies -----------------------------------------------------------
def _gate_implant(features: Features) -> bool:
    return (
        # Buried in bone, and below the crest. This pair is what separates a
        # fixture from a crown: both are dense and similarly sized, only one
        # has bone on every side and sits away from the occlusal plane.
        features["enclosure"] >= 0.55
        and features["distance_to_occlusal"] >= 0.06
        and 45.0 <= features["volume_mm3"] <= 900.0
        and 1.6 <= features["elongation"] <= 6.0
        and features["extent_mm"] >= 6.0
    )


def _score_implant(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 80.0, 600.0, slack=320.0)),
        (3.0, _band(features["elongation"], 2.2, 5.0, slack=1.4)),
        (2.0, _above(features["enclosure"], 0.8, slack=0.25)),
        (2.0, _above(features["max_stored"], 210.0, slack=60.0)),
    )


def _gate_root_filling(features: Features) -> bool:
    return (
        # A thin line inside a root: enclosed by dentine, apical to the crown,
        # and long enough to be a canal rather than a cusp tip.
        features["enclosure"] >= 0.75
        and features["distance_to_occlusal"] >= 0.06
        and 3.0 <= features["volume_mm3"] <= 80.0
        and features["elongation"] >= 2.5
        and features["extent_mm"] >= 5.0
    )


def _score_root_filling(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 5.0, 50.0, slack=40.0)),
        (3.0, _band(features["elongation"], 3.0, 9.0, slack=2.5)),
        (2.0, _above(features["extent_mm"], 8.0, slack=4.0)),
    )


def _gate_impacted(features: Features) -> bool:
    return (
        # An unerupted crown: dense, wholly within bone, well away from the
        # occlusal plane, and in the posterior-lateral segment where third
        # molars are.
        features["enclosure"] >= 0.75
        and features["volume_mm3"] >= 110.0
        and features["distance_to_occlusal"] >= 0.10
        and features.get("posterior", 0.0) > 0.5
        and features.get("lateral", 0.0) > 0.5
        and features["extent_mm"] >= 6.0
    )


def _score_impacted(features: Features) -> float:
    return _blend(
        (3.0, _band(features["volume_mm3"], 150.0, 1500.0, slack=600.0)),
        (3.0, _above(features["enclosure"], 0.88, slack=0.14)),
        (2.0, _above(features["distance_to_occlusal"], 0.14, slack=0.08)),
    )


_LUCENCY_RULES: Final[tuple[_Rule, ...]] = (
    _Rule("cyst", _gate_cyst, _score_cyst),
    _Rule("apical_lesion", _gate_apical_lesion, _score_apical_lesion),
    _Rule("odontogenic_infection", _gate_infection, _score_infection),
    _Rule("abscess", _gate_abscess, _score_abscess),
    _Rule("suspicious_mass", _gate_suspicious_mass, _score_suspicious_mass),
    _Rule("mandibular_canal", _gate_canal, _score_canal),
    _Rule("root_fracture", _gate_root_fracture, _score_root_fracture),
    _Rule("caries_3d", _gate_caries, _score_caries),
    _Rule("bone_loss_3d", _gate_bone_loss, _score_bone_loss),
)

_DENSE_RULES: Final[tuple[_Rule, ...]] = (
    _Rule("implant", _gate_implant, _score_implant),
    _Rule("root_canal_filling", _gate_root_filling, _score_root_filling),
    _Rule("impacted_third_molar", _gate_impacted, _score_impacted),
)

#: Families whose meaning is fixed by how they were detected, so they bypass
#: the rule contest entirely.
_DIRECT_CLASSES: Final[dict[str, str]] = {
    "arch_gap": "missing_tooth",
    "sinus": "sinus_proximity",
    "condyle": "tmj_abnormality",
}


class FindingClassificationStage:
    """Assigns a taxonomy class and a confidence to every candidate."""

    name = "finding-classifier"
    kind = StageKind.CLASSIFICATION
    version = "4.0.0"

    def applies_to(self, volume: VolumeInput) -> bool:
        return True

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        landmarks = state.landmarks
        classified: list[VolumeDetection] = []

        for candidate in state.candidates:
            detection = self._classify(candidate, landmarks)
            if detection is not None:
                classified.append(detection)

        capped = _cap_per_class(classified)
        state.detections.extend(capped)

        structural = self._structural(state)
        state.detections.extend(structural)

        return (
            f"классифицировано {len(capped)} из {len(state.candidates)} кандидатов · "
            f"структурных {len(structural)} · "
            f"не распознано {len(state.candidates) - len(classified)}"
        )

    # -- per-candidate ----------------------------------------------------
    def _classify(
        self, candidate: Candidate, landmarks: dict[str, float]
    ) -> VolumeDetection | None:
        features = dict(candidate.features)
        family = next(
            (key.removeprefix("family_") for key in features if key.startswith("family_")),
            "",
        )

        direct = _DIRECT_CLASSES.get(family)
        if direct is not None:
            return self._direct(candidate, features, direct)

        rules = {"lucency": _LUCENCY_RULES, "dense": _DENSE_RULES}.get(family, ())
        if not rules:
            return None

        self._augment(features, landmarks)
        best_key, best_score = "", 0.0
        for rule in rules:
            if not rule.gate(features):
                continue
            score = rule.score(features)
            if score > best_score:
                best_key, best_score = rule.class_key, score

        if best_score < _SCORE_FLOOR or not best_key:
            return None

        ceiling = _CEILINGS.get(best_key, _DEFAULT_CEILING)
        # Salience carries the detector's confidence that the region is real;
        # the rule score carries the classifier's confidence in the label.
        # Both have to hold for the finding to be worth reporting.
        confidence = float(np.clip(best_score * (0.65 + 0.35 * candidate.salience), 0.0, ceiling))

        return VolumeDetection(
            class_key=best_key,
            confidence=round(confidence, 4),
            box=candidate.box.clamped(),
            region=candidate.region,
            produced_by=self.name,
            extent_mm=(
                round(features["extent_mm"], 1)
                if by_key(best_key).measurable and features.get("extent_mm")
                else None
            ),
            mean_density=round(features.get("mean_hu", 0.0), 1),
        )

    def _direct(
        self, candidate: Candidate, features: Features, class_key: str
    ) -> VolumeDetection | None:
        """Families with exactly one meaning, so no rule contest is needed."""
        if class_key == "tmj_abnormality":
            difference = features.get("condyle_difference", 0.0)
            if difference < _CONDYLE_DIFFERENCE:
                return None
            confidence = float(np.clip(0.35 + difference * 1.4, 0.35, 0.8))
        elif class_key == "sinus_proximity":
            residual = features.get("residual_bone_mm", 0.0)
            confidence = float(np.clip(0.55 + (8.0 - residual) * 0.05, 0.5, 0.9))
        else:
            confidence = float(np.clip(0.45 + candidate.salience * 0.45, 0.4, 0.88))

        return VolumeDetection(
            class_key=class_key,
            confidence=round(confidence, 4),
            box=candidate.box.clamped(),
            region=candidate.region,
            produced_by=self.name,
            extent_mm=round(features["extent_mm"], 1) if features.get("extent_mm") else None,
        )

    @staticmethod
    def _augment(features: Features, landmarks: dict[str, float]) -> None:
        """Add the positional features the rules ask for.

        Kept out of the detector because they depend on landmarks, and a
        detector that has to know where the midline is cannot be swapped for
        one that works on raw voxels.
        """
        arch_y = landmarks.get("arch_center_y", 0.5)
        midline = landmarks.get("midline_x", 0.5)
        features["posterior"] = 1.0 if features.get("y_rel", 0.5) > arch_y + 0.04 else 0.0
        features["lateral"] = 1.0 if abs(features.get("x_rel", 0.5) - midline) > 0.18 else 0.0

    # -- whole-volume findings -------------------------------------------
    def _structural(self, state: PipelineState) -> list[VolumeDetection]:
        """Findings that are properties of the whole scan, not of one region."""
        findings: list[VolumeDetection] = []
        landmarks = state.landmarks

        asymmetry = landmarks.get("asymmetry", 0.0)
        if asymmetry > _ASYMMETRY_FLOOR:
            findings.append(
                VolumeDetection(
                    class_key="jaw_asymmetry",
                    confidence=round(float(np.clip(0.3 + asymmetry * 3.0, 0.3, 0.82)), 4),
                    box=_skeleton_box(landmarks),
                    region=Region.FULL_VOLUME,
                    produced_by=self.name,
                    extent_mm=round(asymmetry * 100, 1),
                )
            )

        findings.extend(self._crest_findings(state))
        return findings

    def _crest_findings(self, state: PipelineState) -> list[VolumeDetection]:
        """Alveolar crest height across the arch, column by column.

        The crest is the highest mineralised cell in each column of the
        mandible. Comparing every column against the arch's own median makes
        the measurement relative to this patient rather than to a population
        norm — which is the only comparison a single uncalibrated CBCT
        supports.
        """
        tissue = state.tissue
        grid = state.grid
        if tissue is None or grid is None:
            return []

        depth, _height, width = grid.shape
        landmarks = state.landmarks
        occlusal_index = int(landmarks.get("occlusal_z", 0.5) * depth)
        if occlusal_index < 4:
            return []

        # Bone, and specifically not tooth. Taking the highest mineralised
        # cell per column measures the crown standing in it, which is why the
        # obvious version of this check reports a healthy crest on a mouth
        # that has lost half its support.
        thresholds = volumetrics.tissue_thresholds(grid)
        values = grid.values[:occlusal_index]
        mandible = (tissue[:occlusal_index] >= BONE) & (values < thresholds.tooth)
        if not mandible.any():
            return []

        # Highest mineralised row per column, as an index into the slab.
        occupied = mandible.any(axis=1)
        crest = np.where(
            occupied.any(axis=0),
            occupied.shape[0] - 1 - np.argmax(occupied[::-1], axis=0),
            -1,
        ).astype(np.float32)

        valid = crest >= 0
        if int(valid.sum()) < max(8, width // 6):
            return []

        # The reference is the patient's own *intact* bone level, which is a
        # high percentile of the crest heights rather than their median. With
        # a median, an arch that has lost bone across most of its length pulls
        # its own baseline down and measures as healthy — the more advanced
        # the disease, the less of it is detected.
        reference = float(np.percentile(crest[valid], 75))
        deficit_mm = (reference - crest) * grid.spacing[2]
        affected = valid & (deficit_mm > _BONE_LOSS_MM)
        share = float(affected.sum()) / float(valid.sum())

        findings: list[VolumeDetection] = []
        if share >= _LOCALISED_SHARE:
            # Same measurement, two readings. Loss across most of the arch is
            # a disease of the periodontium; loss confined to a few sites is a
            # defect, and the two carry different treatment.
            generalised = share >= _GENERALISED_SHARE
            findings.append(
                VolumeDetection(
                    class_key="periodontal_disease" if generalised else "bone_loss_3d",
                    confidence=round(float(np.clip(0.3 + share * 0.9, 0.3, 0.85)), 4),
                    box=_crest_box(landmarks),
                    region=Region.FULL_VOLUME,
                    produced_by=self.name,
                    extent_mm=round(float(np.max(deficit_mm[valid])), 1),
                )
            )

        irregularity = self._arch_irregularity(tissue, occlusal_index)
        if irregularity > 0.55:
            findings.append(
                VolumeDetection(
                    class_key="orthodontic_anomaly",
                    confidence=round(float(np.clip(irregularity * 0.8, 0.3, 0.72)), 4),
                    box=_crest_box(landmarks),
                    region=Region.FULL_VOLUME,
                    produced_by=self.name,
                )
            )

        return findings

    @staticmethod
    def _arch_irregularity(tissue: volumetrics.TissueArray, occlusal_index: int) -> float:
        """How unevenly the teeth are spaced along the arch.

        A well-aligned dentition produces a regular alternation of dense
        (tooth) and less-dense (interdental) columns. Crowding and rotation
        break that rhythm, which shows up as a high coefficient of variation
        in the widths of the dense runs.
        """
        slab_start = max(0, occlusal_index - 6)
        slab = tissue[slab_start : occlusal_index + 1]
        if slab.size == 0:
            return 0.0

        dense_columns = (slab >= DENSE).any(axis=(0, 1))
        widths: list[int] = []
        run = 0
        for occupied in dense_columns:
            if occupied:
                run += 1
            elif run:
                widths.append(run)
                run = 0
        if run:
            widths.append(run)

        if len(widths) < 4:
            return 0.0
        array = np.asarray(widths, dtype=np.float64)
        mean = float(array.mean())
        if mean <= 0:
            return 0.0
        return float(np.clip(array.std() / mean, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cap_per_class(detections: list[VolumeDetection]) -> list[VolumeDetection]:
    """Keep the most confident instances of each class and discard the tail."""
    counts: dict[str, int] = {}
    kept: list[VolumeDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        seen = counts.get(detection.class_key, 0)
        if seen >= _PER_CLASS_LIMIT:
            continue
        counts[detection.class_key] = seen + 1
        kept.append(detection)
    return kept


def _skeleton_box(landmarks: dict[str, float]) -> BoundingBox3D:
    """The bounds of the mineralised skeleton, for whole-volume findings."""
    x0 = landmarks.get("bone_x_min", 0.1)
    y0 = landmarks.get("bone_y_min", 0.1)
    z0 = landmarks.get("bone_z_min", 0.1)
    return BoundingBox3D(
        x=x0,
        y=y0,
        z=z0,
        width=max(landmarks.get("bone_x_max", 0.9) - x0, 0.1),
        height=max(landmarks.get("bone_y_max", 0.9) - y0, 0.1),
        depth=max(landmarks.get("bone_z_max", 0.9) - z0, 0.1),
    ).clamped()


def _crest_box(landmarks: dict[str, float]) -> BoundingBox3D:
    """A slab around the alveolar crests, for findings about them."""
    x0 = landmarks.get("bone_x_min", 0.1)
    y0 = landmarks.get("bone_y_min", 0.1)
    return BoundingBox3D(
        x=x0,
        y=y0,
        z=max(landmarks.get("occlusal_z", 0.5) - 0.16, 0.0),
        width=max(landmarks.get("bone_x_max", 0.9) - x0, 0.1),
        height=max(landmarks.get("bone_y_max", 0.9) - y0, 0.1),
        depth=0.22,
    ).clamped()


def resolve_region(box: BoundingBox3D, landmarks: dict[str, float]) -> Region:
    """Public helper for callers that need the region of an edited box."""
    return Region(region_for(box.center, landmarks))
