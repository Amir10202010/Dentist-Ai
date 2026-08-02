"""Quality control: is this reconstruction worth reading?

Runs first, and its output qualifies everything after it. A CBCT taken while
the patient swallowed is not a slightly worse CBCT — it is one where a
"lucency" may be a registration seam, and where a millimetre measurement for
implant planning is not defensible.

The stage does not gate the pipeline. Suppressing findings on a poor scan
would hide the one thing a clinician most needs to see, and a clinician can
weigh a finding against a stated quality score perfectly well. What it does
instead is publish the score, emit the artefacts as findings in their own
right, and let the synthesis stage discount confidence in proportion.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from dentist_ai.ml import volumetrics
from dentist_ai.ml.cbct_taxonomy import Region
from dentist_ai.ml.pipeline import (
    BoundingBox3D,
    PipelineState,
    QualityAssessment,
    StageKind,
    VolumeDetection,
    VolumeInput,
    box_from_grid,
    ensure_grid,
)

#: Above this fraction of very dense voxels, metal is dominating the
#: reconstruction rather than merely present in it.
_METAL_SIGNIFICANT: Final[float] = 0.004
#: Through-plane to in-plane sharpness ratio. Slices misregistered by patient
#: movement make the through-plane difference jump relative to the in-plane
#: one, which stays crisp because each slice is internally consistent.
_MOTION_RATIO_CLEAN: Final[float] = 1.15
_MOTION_RATIO_SEVERE: Final[float] = 2.4
#: Noise, as a fraction of the volume's own air-to-bone contrast. Expressed as
#: a ratio rather than in absolute units because ingest rescales every volume
#: onto 0-255 against its own percentiles, so an absolute standard deviation
#: means something different in every scan.
_NOISE_CLEAN: Final[float] = 0.05
_NOISE_SEVERE: Final[float] = 0.28


class QualityControlStage:
    """Measures noise, motion, metal artefact and field-of-view coverage."""

    name = "acquisition-qc"
    kind = StageKind.QUALITY_CONTROL
    version = "1.2.0"

    def applies_to(self, volume: VolumeInput) -> bool:
        return volume.voxels.size > 0

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        grid = ensure_grid(volume, state)
        thresholds = volumetrics.tissue_thresholds(grid)

        noise = self._noise(grid, thresholds)
        motion = self._motion(grid)
        metal, metal_fraction = self._metal(grid, thresholds)
        coverage, truncated = self._coverage(grid, thresholds.air_soft)

        # Weighted because the failure modes are not equally disqualifying:
        # motion invalidates measurement, metal invalidates a region, noise
        # only lowers confidence everywhere.
        score = float(
            np.clip(
                1.0 - (0.42 * motion + 0.30 * noise + 0.16 * metal + 0.12 * (1.0 - coverage)),
                0.05,
                1.0,
            )
        )

        notes: list[str] = []
        if truncated:
            notes.append(
                "Объект выходит за границы поля зрения — часть анатомии не реконструирована."
            )
        if motion > 0.45:
            notes.append("Признаки движения пациента: измерения проводить с осторожностью.")
        if metal_fraction > _METAL_SIGNIFICANT:
            notes.append(f"Металлические реставрации занимают {metal_fraction * 100:.1f}% объёма.")

        state.quality = QualityAssessment(
            score=score,
            noise=noise,
            motion=motion,
            metal=metal,
            coverage=coverage,
            notes=tuple(notes),
        )

        self._emit_artefacts(grid, thresholds, state, metal_fraction=metal_fraction, motion=motion)

        return (
            f"score {score:.2f} · шум {noise:.2f} · движение {motion:.2f} · "
            f"металл {metal:.2f} · охват {coverage:.2f}"
        )

    # -- measurements -----------------------------------------------------
    @staticmethod
    def _noise(grid: volumetrics.Grid, thresholds: volumetrics.TissueThresholds) -> float:
        """Spread inside air, relative to this volume's own tissue contrast.

        Air is the only region whose true value is constant, so its spread in
        the reconstruction is what the scanner and the reconstruction added.
        Dividing by the air-to-bone separation makes the result comparable
        between scans that were rescaled differently on ingest.
        """
        air = grid.values[grid.values < thresholds.air_soft]
        if air.size < 64:
            return 0.5
        contrast = max(thresholds.soft_bone - thresholds.air_soft, 1e-3)
        ratio = float(np.std(air)) / contrast
        span = _NOISE_SEVERE - _NOISE_CLEAN
        return float(np.clip((ratio - _NOISE_CLEAN) / span, 0.0, 1.0))

    @staticmethod
    def _motion(grid: volumetrics.Grid) -> float:
        """Through-plane sharpness relative to in-plane, normalised."""
        through = volumetrics.gradient_energy(grid.values, axis=0)
        in_plane = (
            volumetrics.gradient_energy(grid.values, axis=1)
            + volumetrics.gradient_energy(grid.values, axis=2)
        ) / 2
        if in_plane < 1e-6:
            return 0.5
        ratio = through / in_plane
        span = _MOTION_RATIO_SEVERE - _MOTION_RATIO_CLEAN
        return float(np.clip((ratio - _MOTION_RATIO_CLEAN) / span, 0.0, 1.0))

    @staticmethod
    def _metal(
        grid: volumetrics.Grid, thresholds: volumetrics.TissueThresholds
    ) -> tuple[float, float]:
        """Severity in ``[0, 1]`` and the raw fraction of very dense voxels."""
        cut = max(thresholds.dense, 235.0)
        fraction = float((grid.values >= cut).mean())
        # A crown is a few tenths of a percent; a full-arch bridge is percent.
        severity = float(np.clip(fraction / (_METAL_SIGNIFICANT * 4), 0.0, 1.0))
        return severity, fraction

    @staticmethod
    def _coverage(grid: volumetrics.Grid, air_threshold: float) -> tuple[float, bool]:
        """How much of the field of view holds the patient, and whether it is cut off."""
        tissue = grid.values > air_threshold
        occupancy = float(tissue.mean())

        # Tissue reaching a face of the volume means the anatomy continues
        # outside it: the reconstruction is truncated, and anything measured
        # near that face is measured against an edge rather than an anatomy.
        faces = (
            tissue[0].mean(),
            tissue[-1].mean(),
            tissue[:, 0].mean(),
            tissue[:, -1].mean(),
            tissue[:, :, 0].mean(),
            tissue[:, :, -1].mean(),
        )
        truncated = any(float(value) > 0.12 for value in faces)

        # A dental CBCT is mostly air by design; 25-55% occupancy is normal,
        # and both a nearly empty and a nearly full volume are suspect.
        ideal = float(np.clip(1.0 - abs(occupancy - 0.38) / 0.38, 0.0, 1.0))
        return (ideal * (0.75 if truncated else 1.0), truncated)

    # -- findings ---------------------------------------------------------
    def _emit_artefacts(
        self,
        grid: volumetrics.Grid,
        thresholds: volumetrics.TissueThresholds,
        state: PipelineState,
        *,
        metal_fraction: float,
        motion: float,
    ) -> None:
        """Report artefacts as findings, not just as a score.

        A clinician needs to know *where* the metal is, because that is the
        region whose reading is unreliable — a number in a corner of the
        screen cannot convey that.
        """
        if metal_fraction > _METAL_SIGNIFICANT:
            dense_mask = grid.values >= max(thresholds.dense, 235.0)
            bounds = volumetrics.bounding_box_of(dense_mask)
            if bounds is not None:
                state.detections.append(
                    VolumeDetection(
                        class_key="metal_artifact",
                        confidence=float(np.clip(0.55 + metal_fraction * 40, 0.5, 0.97)),
                        box=box_from_grid(bounds, grid.shape),
                        region=Region.FULL_VOLUME,
                        produced_by=self.name,
                    )
                )

        if motion > 0.5:
            state.detections.append(
                VolumeDetection(
                    class_key="motion_artifact",
                    confidence=float(np.clip(0.4 + motion * 0.55, 0.4, 0.95)),
                    box=BoundingBox3D(0.05, 0.05, 0.05, 0.9, 0.9, 0.9),
                    region=Region.FULL_VOLUME,
                    produced_by=self.name,
                )
            )
