"""Segmentation: where the anatomy is, before asking what is wrong with it.

Every later stage needs this. "A lucency at the apex" is only meaningful once
something has decided which voxels are bone, where the roots point, and which
jaw they belong to; without that a detector finds the airway and reports it as
a cyst.

The output is two things:

* a **tissue map** — air, soft tissue, bone, dense material — on the analysis
  grid, which is what the detection stage searches inside;
* a set of **landmarks** in normalised coordinates: the occlusal plane, the
  dental midline, the arch centre, the extent of each jaw. Those are what turn
  a box at ``z = 0.31`` into "right posterior mandible".

The thresholds are derived from the volume rather than fixed, because CBCT
grey values are not calibrated: the same patient on two scanners produces two
different numbers for the same cortical bone. Otsu on this volume's own
histogram is a stable answer where a hard-coded Hounsfield cut-off is not.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from dentist_ai.ml import volumetrics
from dentist_ai.ml.pipeline import PipelineState, StageKind, VolumeInput, ensure_grid

#: Tissue codes written into ``PipelineState.tissue``.
AIR: Final[int] = 0
SOFT: Final[int] = 1
BONE: Final[int] = 2
DENSE: Final[int] = 3

#: Field-of-view values whose acquisitions genuinely contain both jaws. A
#: narrower volume gets landmarks for the jaw it holds and no occlusal split.
_BOTH_JAW_VIEWS: Final[frozenset[str]] = frozenset({"full_head", "both_jaws"})


class AnatomySegmentationStage:
    """Separates tissue classes and locates the jaw landmarks."""

    name = "anatomy-segmentation"
    kind = StageKind.SEGMENTATION
    version = "2.1.0"

    def applies_to(self, volume: VolumeInput) -> bool:
        depth, height, width = volume.shape
        return min(depth, height, width) >= 8

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        grid = ensure_grid(volume, state)
        thresholds = volumetrics.tissue_thresholds(grid)

        tissue = np.full(grid.shape, AIR, dtype=np.uint8)
        tissue[grid.values > thresholds.air_soft] = SOFT
        tissue[grid.values > thresholds.soft_bone] = BONE
        tissue[grid.values >= thresholds.dense] = DENSE
        state.tissue = tissue

        mineralised = tissue >= BONE
        landmarks = self._landmarks(grid, mineralised, volume)
        landmarks.update(
            {
                "threshold_air_soft": thresholds.air_soft,
                "threshold_soft_bone": thresholds.soft_bone,
                "threshold_dense": thresholds.dense,
                "grid_depth": float(grid.shape[0]),
                "grid_height": float(grid.shape[1]),
                "grid_width": float(grid.shape[2]),
                "grid_spacing_x": grid.spacing[0],
                "grid_spacing_y": grid.spacing[1],
                "grid_spacing_z": grid.spacing[2],
            }
        )
        state.landmarks.update(landmarks)

        bone_fraction = float(mineralised.mean())
        return (
            f"кость {bone_fraction * 100:.1f}% объёма · "
            f"окклюзия z={landmarks['occlusal_z']:.2f} · "
            f"средняя линия x={landmarks['midline_x']:.2f}"
        )

    # -- landmarks --------------------------------------------------------
    def _landmarks(
        self,
        grid: volumetrics.Grid,
        mineralised: volumetrics.BoolArray,
        volume: VolumeInput,
    ) -> dict[str, float]:
        depth, height, width = grid.shape

        occlusal_index = self._occlusal_index(grid, mineralised, volume)
        occlusal_z = occlusal_index / max(depth, 1)

        # The midline is the centre of mass of mineralised tissue across the
        # left-right axis. It is not assumed to be the middle of the volume:
        # patients are rarely centred in the field of view.
        column_profile = volumetrics.axis_profile(mineralised, axis=2)
        midline_x = self._centre_of_mass(column_profile) / max(width, 1)

        row_profile = volumetrics.axis_profile(mineralised, axis=1)
        arch_center_y = self._centre_of_mass(row_profile) / max(height, 1)

        bounds = volumetrics.bounding_box_of(mineralised)
        if bounds is None:
            extent = {
                "bone_z_min": 0.0,
                "bone_z_max": 1.0,
                "bone_y_min": 0.0,
                "bone_y_max": 1.0,
                "bone_x_min": 0.0,
                "bone_x_max": 1.0,
            }
        else:
            z0, z1, y0, y1, x0, x1 = bounds
            extent = {
                "bone_z_min": z0 / depth,
                "bone_z_max": z1 / depth,
                "bone_y_min": y0 / height,
                "bone_y_max": y1 / height,
                "bone_x_min": x0 / width,
                "bone_x_max": x1 / width,
            }

        left = mineralised[:, :, int(midline_x * width) :]
        right = mineralised[:, :, : int(midline_x * width)]
        left_volume = float(left.sum()) * grid.cell_volume_mm3
        right_volume = float(right.sum()) * grid.cell_volume_mm3
        total = left_volume + right_volume
        asymmetry = abs(left_volume - right_volume) / total if total > 0 else 0.0

        mandible = mineralised[:occlusal_index]
        maxilla = mineralised[occlusal_index:]

        return {
            "occlusal_z": float(occlusal_z),
            "midline_x": float(midline_x),
            "arch_center_y": float(arch_center_y),
            "bone_volume_mm3": float(total),
            "left_bone_mm3": left_volume,
            "right_bone_mm3": right_volume,
            "asymmetry": float(asymmetry),
            "mandible_bone_mm3": float(mandible.sum()) * grid.cell_volume_mm3,
            "maxilla_bone_mm3": float(maxilla.sum()) * grid.cell_volume_mm3,
            **extent,
        }

    def _occlusal_index(
        self,
        grid: volumetrics.Grid,
        mineralised: volumetrics.BoolArray,
        volume: VolumeInput,
    ) -> int:
        """Locate the plane where the arches meet.

        Found from the *teeth* rather than from the bone. The obvious approach
        — look for the gap between the two bands of alveolar bone along the
        superior-inferior axis — fails on real anatomy, because the crowns
        occupy that gap and the ramus keeps the profile high well above it.

        Enamel does not have that problem. It is the densest tissue in the
        volume, it exists only in crowns, and crowns of both arches meet at
        the occlusal plane. The centre of mass of the densest class is
        therefore the plane itself, and it degrades gracefully: in a partially
        edentulous mouth it shifts toward the remaining teeth, which is still
        the right answer.
        """
        depth = grid.shape[0]
        thresholds = volumetrics.tissue_thresholds(grid)
        dense_profile = volumetrics.axis_profile(grid.values >= thresholds.dense, axis=0)

        # A fully edentulous scan, or one whose densest tissue is a single
        # metal restoration, gives too little to locate a plane from.
        if float(dense_profile.sum()) > 0.02 and int((dense_profile > 0).sum()) >= 3:
            return round(self._centre_of_mass(dense_profile))

        bone_profile = volumetrics.smooth(volumetrics.axis_profile(mineralised, axis=0))
        if volume.field_of_view in _BOTH_JAW_VIEWS and bone_profile.size >= 8:
            return volumetrics.largest_valley(bone_profile)
        return min(round(self._centre_of_mass(bone_profile)), depth - 1)

    @staticmethod
    def _centre_of_mass(profile: volumetrics.FloatArray) -> float:
        total = float(profile.sum())
        if total <= 0:
            return profile.size / 2
        weights = np.arange(profile.size, dtype=np.float64)
        return float((profile.astype(np.float64) * weights).sum() / total)


def region_for(
    center: tuple[float, float, float],
    landmarks: dict[str, float],
) -> str:
    """Name the anatomical region a normalised ``(x, y, z)`` point falls in.

    An estimate, and presented as one. It rests on the head being upright and
    roughly centred, which is what a positioned CBCT gives and what a rotated
    one does not — so the value is editable in the UI, exactly like the FDI
    number a panoramic finding gets.
    """
    x, y, z = center
    occlusal = landmarks.get("occlusal_z", 0.5)
    midline = landmarks.get("midline_x", 0.5)
    arch_y = landmarks.get("arch_center_y", 0.5)

    upper = z >= occlusal
    # Anterior of the arch centre by a clear margin is the front segment;
    # the six anterior teeth span roughly the front third of the arch.
    if y < arch_y - 0.12:
        return "anterior_maxilla" if upper else "anterior_mandible"

    left = x >= midline
    if upper:
        return "maxilla_left" if left else "maxilla_right"
    return "mandible_left" if left else "mandible_right"
