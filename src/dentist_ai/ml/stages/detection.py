"""Detection: which regions of the volume are worth a second look.

This stage answers "something is here", never "this is a cyst". It measures
regions and hands them on; :mod:`.classification` decides what they are.
Keeping the two apart is what allows either to be replaced by a trained model
without the other changing, and it means a detector tuned for sensitivity can
be paired with a classifier tuned for specificity — which is the combination
clinical imaging actually wants.

Six families of candidate are produced, each from a different property of the
reconstruction:

``lucency``
    Low-density regions enclosed by bone. Cysts, apical lesions, infections
    and the marrow spaces that mimic them.
``dense``
    Compact high-density bodies. Implants, root fillings, posts, restorations.
``canal``
    Elongated low-density structures below the occlusal plane — the inferior
    alveolar canal, which every mandibular surgical plan has to respect.
``arch_gap``
    Runs of the arch with no tooth-density above the crest. Missing teeth.
``sinus``
    Air spaces above the occlusal plane, measured against the bone beneath
    them. Residual ridge height is the number that decides an implant plan.
``condyle``
    The superior-posterior-lateral bodies, measured left against right.

The costly part is connected-component labelling, which runs once for the
low-density mask and once for the dense one; everything else is profiles and
bounding boxes over an array of a few hundred thousand cells.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from dentist_ai.ml import volumetrics
from dentist_ai.ml.cbct_taxonomy import Region
from dentist_ai.ml.pipeline import (
    BoundingBox3D,
    Candidate,
    PipelineState,
    StageKind,
    VolumeInput,
    box_from_grid,
    ensure_grid,
)
from dentist_ai.ml.stages.segmentation import BONE, DENSE, region_for

#: A lesion smaller than this is below what a CBCT at clinical resolution can
#: distinguish from a marrow space, and reporting it would be noise.
_MIN_LESION_MM3: Final[float] = 12.0
#: Above this a "lesion" is the airway, the nasal cavity or the field of view
#: itself rather than a lesion.
_MAX_LESION_MM3: Final[float] = 24_000.0
#: Fraction of a region's surface that must abut mineralised tissue before it
#: counts as enclosed rather than as part of the outside world.
_ENCLOSURE_FLOOR: Final[float] = 0.45
#: Smallest dense body worth reporting — below this it is a speck of noise or
#: the tip of a fissure sealant.
_MIN_DENSE_MM3: Final[float] = 4.0
#: Residual bone under a sinus below which proximity matters for implants.
_SINUS_PROXIMITY_MM: Final[float] = 8.0
#: Shortest edentulous span worth reporting. A premolar is about 7 mm wide, so
#: anything narrower is an interdental space rather than a missing tooth.
_MIN_EDENTULOUS_MM: Final[float] = 6.5
#: Angular bins in the arch sweep. Roughly two bins per tooth across a 180°
#: horseshoe, which resolves a single missing tooth without letting one noisy
#: bin open a gap.
_ARCH_BINS: Final[int] = 64
#: Candidates kept per family, most salient first. A CBCT of a heavily
#: restored mouth has hundreds of dense bodies; a report listing all of them
#: is a report nobody reads.
_PER_FAMILY_LIMIT: Final[int] = 14


class LesionDetectionStage:
    """Finds candidate regions and measures them."""

    name = "volumetric-detection"
    kind = StageKind.DETECTION
    version = "3.0.0"

    def applies_to(self, volume: VolumeInput) -> bool:
        depth, height, width = volume.shape
        return min(depth, height, width) >= 8

    def run(self, volume: VolumeInput, state: PipelineState) -> str:
        grid = ensure_grid(volume, state)
        tissue = state.tissue
        if tissue is None:
            # Segmentation failed or was skipped. Detection can still run on a
            # locally derived tissue map; the regional labels will be coarser.
            thresholds = volumetrics.tissue_thresholds(grid)
            tissue = np.where(
                grid.values >= thresholds.dense,
                np.uint8(DENSE),
                np.where(grid.values > thresholds.soft_bone, np.uint8(BONE), np.uint8(0)),
            )

        mineralised = tissue >= BONE
        thresholds = volumetrics.tissue_thresholds(grid)
        landmarks = state.landmarks

        counts: dict[str, int] = {}
        for family, produced in (
            ("lucency", self._lucencies(grid, mineralised, thresholds, landmarks, volume)),
            ("dense", self._dense_bodies(grid, tissue, landmarks, volume)),
            ("arch_gap", self._arch_gaps(grid, tissue, landmarks)),
            ("sinus", self._sinus_proximity(grid, mineralised, landmarks)),
            ("condyle", self._condyles(grid, mineralised, landmarks, volume)),
        ):
            kept = produced[:_PER_FAMILY_LIMIT]
            counts[family] = len(kept)
            state.candidates.extend(kept)

        summary = " · ".join(f"{family}: {count}" for family, count in counts.items() if count)
        return summary or "кандидатов не найдено"

    # -- families ---------------------------------------------------------
    def _lucencies(
        self,
        grid: volumetrics.Grid,
        mineralised: volumetrics.BoolArray,
        thresholds: volumetrics.TissueThresholds,
        landmarks: dict[str, float],
        volume: VolumeInput,
    ) -> list[Candidate]:
        """Fluid-density pockets sitting inside bone.

        Two decisions define this search.

        *Inside* is decided by asking, along each axis, whether there is
        mineralised tissue on both sides of the cell. Requiring two of the
        three axes to bracket it admits a lesion that has eroded one cortical
        plate — precisely the aggressive lesion worth finding — while still
        excluding the airway, which is open along its length.

        *Lucent* is decided at the soft-tissue ceiling rather than at the bone
        boundary. The difference is the whole search: everything below the
        bone boundary includes trabecular marrow, and a lesion sitting inside
        a marrow channel then labels as one continuous 40 mm structure instead
        of as the focal object it is.
        """
        interior = self._interior_mask(mineralised)
        lucent = interior & (grid.values < thresholds.soft_ceiling)
        if not lucent.any():
            return []

        labels, count = volumetrics.label_components(lucent)
        components = volumetrics.measure_components(
            labels, count, grid, enclosing=mineralised, min_cells=4
        )

        candidates: list[Candidate] = []
        for component in components:
            if not _MIN_LESION_MM3 <= component.volume_mm3 <= _MAX_LESION_MM3:
                continue
            if component.enclosure < _ENCLOSURE_FLOOR:
                continue
            # A region one cell thick in any direction is an interface, not a
            # volume: the partial-volume film between two tissues, or the
            # sheet of soft tissue in the occlusal gap. A real lesion has
            # extent in all three directions.
            if min(component.size) < 2:
                continue

            box = box_from_grid(component.bounds, grid.shape)
            fill = component.volume_mm3 / max(_box_volume_mm3(component, grid), 1e-6)
            # A rounded, well-enclosed, medium-sized void is the signature of
            # a cyst; the salience blends those three into one number so the
            # classifier receives candidates in a sensible order.
            salience = float(
                np.clip(
                    0.32
                    + 0.30 * component.enclosure
                    + 0.22 * fill
                    + 0.16 * min(component.volume_mm3 / 900.0, 1.0),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                Candidate(
                    box=box,
                    salience=salience,
                    features=_features(
                        component,
                        grid,
                        box,
                        landmarks,
                        volume,
                        family="lucency",
                        fill=fill,
                    ),
                    region=_region_enum(box, landmarks),
                )
            )

        candidates.sort(key=lambda item: item.salience, reverse=True)
        return candidates

    def _dense_bodies(
        self,
        grid: volumetrics.Grid,
        tissue: volumetrics.TissueArray,
        landmarks: dict[str, float],
        volume: VolumeInput,
    ) -> list[Candidate]:
        """Compact very-high-density regions: dental work, not anatomy."""
        dense = tissue >= DENSE
        if not dense.any():
            return []

        labels, count = volumetrics.label_components(dense)
        # Enclosure is measured against bone *excluding* the dense class, so a
        # crown — whose neighbours are air and soft tissue above the crest —
        # scores near zero while a fixture buried in the ridge scores high.
        # Measuring against all mineralised tissue would score both alike,
        # since a dense body is itself mineralised.
        components = volumetrics.measure_components(
            labels,
            count,
            grid,
            enclosing=(tissue == BONE),
            # Enamel against bone is a sharp edge, so no slack is wanted here.
            # With it, a crown reads as surrounded by bone and the one property
            # separating it from a fixture disappears.
            tolerate_partial_volume=False,
            min_cells=2,
        )

        candidates: list[Candidate] = []
        for component in components:
            if component.volume_mm3 < _MIN_DENSE_MM3:
                continue
            box = box_from_grid(component.bounds, grid.shape)
            fill = component.volume_mm3 / max(_box_volume_mm3(component, grid), 1e-6)
            salience = float(
                np.clip(0.45 + 0.35 * fill + 0.2 * min(component.max_value / 255, 1), 0, 1)
            )
            candidates.append(
                Candidate(
                    box=box,
                    salience=salience,
                    features=_features(
                        component, grid, box, landmarks, volume, family="dense", fill=fill
                    ),
                    region=_region_enum(box, landmarks),
                )
            )

        candidates.sort(key=lambda item: item.features["volume_mm3"], reverse=True)
        return candidates

    def _arch_gaps(
        self,
        grid: volumetrics.Grid,
        tissue: volumetrics.TissueArray,
        landmarks: dict[str, float],
    ) -> list[Candidate]:
        """Stretches of the arch with no tooth above the alveolar crest.

        Sampled **around the arch**, in polar coordinates about its centre,
        rather than along the left-right axis.

        Projecting onto x is the obvious implementation and it is wrong for
        exactly the teeth that matter. The arch runs left-to-right across the
        front, so a projection resolves the incisors well — but it turns
        posteriorly, and by the molars it runs front-to-back, so several teeth
        collapse into one column and a missing molar leaves no gap to find. A
        polar sweep follows the curve, so a span reads the same whether it is
        anterior or posterior.

        The angular width of a tooth is roughly constant around the arch, which
        also means one threshold in millimetres of arc works everywhere.
        """
        depth, height, width = grid.shape
        occlusal = landmarks.get("occlusal_z", 0.5)
        centre_x = landmarks.get("midline_x", 0.5) * width
        # The polar origin sits behind the arch, not at the bone centroid: the
        # dentition is a horseshoe, and sweeping from its own centre of mass
        # would put the origin inside the curve for the anterior teeth and
        # outside it for the molars.
        centre_y = landmarks.get("bone_y_max", 0.9) * height
        results: list[Candidate] = []

        for upper, (z0, z1) in (
            (False, (max(0, int((occlusal - 0.18) * depth)), int(occlusal * depth))),
            (True, (int(occlusal * depth), min(depth, int((occlusal + 0.18) * depth)))),
        ):
            if z1 - z0 < 2:
                continue
            slab = tissue[z0:z1]
            # Peak tissue class per column: DENSE where a tooth or restoration
            # stands, BONE where only the ridge remains.
            has_tooth = (slab >= DENSE).any(axis=0)
            has_ridge = (slab >= BONE).any(axis=0)
            if not has_ridge.any():
                continue

            tooth_by_angle, ridge_by_angle, radius_by_angle = _polar_profiles(
                has_tooth, has_ridge, centre_x=centre_x, centre_y=centre_y
            )
            occupancy = volumetrics.smooth(tooth_by_angle, window=3)

            # An angular bin whose arc length is the width of a premolar. The
            # radius varies around the arch, so the conversion uses each run's
            # own mean radius rather than a single figure.
            for start, end in _runs_below(occupancy, threshold=0.34, min_length=2):
                # A gap at either end of the sweep is the sweep leaving the
                # dentition, not a missing tooth.
                if start == 0 or end >= _ARCH_BINS:
                    continue
                # No ridge beneath means the ray left the bone entirely.
                if float(ridge_by_angle[start:end].mean()) < 0.6:
                    continue

                mean_radius = float(radius_by_angle[start:end].mean())
                if mean_radius <= 0:
                    continue
                arc_mm = (
                    (end - start) / _ARCH_BINS * np.pi * mean_radius * grid.spacing[0]
                )
                if arc_mm < _MIN_EDENTULOUS_MM:
                    continue

                box = _arc_box(
                    start,
                    end,
                    mean_radius,
                    centre_x=centre_x,
                    centre_y=centre_y,
                    z0=z0,
                    z1=z1,
                    shape=grid.shape,
                )
                results.append(
                    Candidate(
                        box=box,
                        salience=float(np.clip(0.4 + arc_mm / 40.0, 0.3, 0.9)),
                        features={
                            "family_arch_gap": 1.0,
                            "volume_mm3": 0.0,
                            "extent_mm": float(arc_mm),
                            "span_mm": float(arc_mm),
                            "elongation": 1.0,
                            "enclosure": 0.0,
                            "fill": 0.0,
                            "mean_hu": 0.0,
                            "max_hu": 0.0,
                            "max_stored": 0.0,
                            "upper": 1.0 if upper else 0.0,
                            "z_rel": box.center[2],
                            "y_rel": box.center[1],
                            "x_rel": box.center[0],
                            "distance_to_occlusal": abs(box.center[2] - occlusal),
                            "side": 1.0
                            if box.center[0] >= landmarks.get("midline_x", 0.5)
                            else 0.0,
                        },
                        region=_region_enum(box, landmarks),
                    )
                )

        results.sort(key=lambda item: item.features["span_mm"], reverse=True)
        return results

    def _sinus_proximity(
        self,
        grid: volumetrics.Grid,
        mineralised: volumetrics.BoolArray,
        landmarks: dict[str, float],
    ) -> list[Candidate]:
        """Residual bone height beneath each maxillary air space.

        The number an implantologist opens a CBCT to get. Measured by walking
        down from the floor of each sinus and counting mineralised cells until
        the ridge ends.
        """
        depth, _height, _width = grid.shape
        occlusal = landmarks.get("occlusal_z", 0.5)
        z_start = int(occlusal * depth)
        if depth - z_start < 4:
            return []

        interior = self._interior_mask(mineralised)
        sinus_mask = np.zeros(grid.shape, dtype=bool)
        sinus_mask[z_start:] = interior[z_start:] & ~mineralised[z_start:]
        if not sinus_mask.any():
            return []

        labels, count = volumetrics.label_components(sinus_mask)
        components = volumetrics.measure_components(
            labels, count, grid, enclosing=mineralised, min_cells=20
        )

        results: list[Candidate] = []
        for component in components:
            # The maxillary sinuses are the two largest enclosed air spaces
            # above the occlusal plane; anything small is a marrow space.
            if component.volume_mm3 < 600.0:
                continue

            z0, _z1, y0, y1, x0, x1 = component.bounds
            beneath = mineralised[max(z0 - 12, 0) : z0, y0:y1, x0:x1]
            if beneath.size == 0:
                continue
            # Mean bone thickness under the footprint, in millimetres.
            residual_mm = float(beneath.sum(axis=0).mean()) * grid.spacing[2]
            if residual_mm > _SINUS_PROXIMITY_MM:
                continue

            box = box_from_grid((max(z0 - 12, 0), z0 + 1, y0, y1, x0, x1), grid.shape)
            results.append(
                Candidate(
                    box=box,
                    salience=float(np.clip(1.0 - residual_mm / _SINUS_PROXIMITY_MM, 0.25, 0.95)),
                    features={
                        "family_sinus": 1.0,
                        "volume_mm3": component.volume_mm3,
                        "extent_mm": float(residual_mm),
                        "residual_bone_mm": float(residual_mm),
                        "elongation": component.elongation,
                        "enclosure": component.enclosure,
                        "fill": 0.0,
                        "mean_hu": 0.0,
                        "max_stored": 0.0,
                        "upper": 1.0,
                        "z_rel": box.center[2],
                        "y_rel": box.center[1],
                        "x_rel": box.center[0],
                        "distance_to_occlusal": abs(box.center[2] - occlusal),
                        "side": 1.0 if box.center[0] >= landmarks.get("midline_x", 0.5) else 0.0,
                    },
                    region=(
                        Region.MAXILLARY_SINUS_LEFT
                        if box.center[0] >= landmarks.get("midline_x", 0.5)
                        else Region.MAXILLARY_SINUS_RIGHT
                    ),
                )
            )

        results.sort(key=lambda item: item.salience, reverse=True)
        return results[:2]

    def _condyles(
        self,
        grid: volumetrics.Grid,
        mineralised: volumetrics.BoolArray,
        landmarks: dict[str, float],
        volume: VolumeInput,
    ) -> list[Candidate]:
        """The mandibular condyles, compared with each other.

        Only attempted when the field of view plausibly contains them. On an
        implant-site volume the superior-posterior corner is soft tissue, and
        measuring it would manufacture a TMJ finding out of nothing.
        """
        if volume.field_of_view not in {"full_head", "tmj", "both_jaws"}:
            return []

        depth, height, width = grid.shape
        occlusal = landmarks.get("occlusal_z", 0.5)
        midline = landmarks.get("midline_x", 0.5)
        z0 = int(min(0.96, occlusal + 0.20) * depth)
        y0 = int(landmarks.get("arch_center_y", 0.5) * height)
        if depth - z0 < 3 or height - y0 < 3:
            return []

        posterior_superior = np.zeros(grid.shape, dtype=bool)
        posterior_superior[z0:, y0:, :] = mineralised[z0:, y0:, :]
        split = int(midline * width)

        measured: list[tuple[bool, float, tuple[int, int, int, int, int, int]]] = []
        for is_left, region_mask in (
            (False, posterior_superior[:, :, :split]),
            (True, posterior_superior[:, :, split:]),
        ):
            bounds = volumetrics.bounding_box_of(region_mask)
            if bounds is None:
                continue
            offset = 0 if not is_left else split
            adjusted = (
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
                bounds[4] + offset,
                bounds[5] + offset,
            )
            measured.append((is_left, float(region_mask.sum()) * grid.cell_volume_mm3, adjusted))

        if len(measured) != 2:
            return []

        volumes = [item[1] for item in measured]
        total = sum(volumes)
        difference = abs(volumes[0] - volumes[1]) / total if total > 0 else 0.0

        results: list[Candidate] = []
        for is_left, condyle_volume, bounds in measured:
            box = box_from_grid(bounds, grid.shape)
            results.append(
                Candidate(
                    box=box,
                    salience=float(np.clip(difference * 2.2, 0.0, 0.9)),
                    features={
                        "family_condyle": 1.0,
                        "volume_mm3": condyle_volume,
                        "extent_mm": 0.0,
                        "condyle_difference": float(difference),
                        "elongation": 1.0,
                        "enclosure": 0.0,
                        "fill": 0.0,
                        "mean_hu": 0.0,
                        "max_stored": 0.0,
                        "upper": 1.0,
                        "z_rel": box.center[2],
                        "y_rel": box.center[1],
                        "x_rel": box.center[0],
                        "distance_to_occlusal": abs(box.center[2] - occlusal),
                        "side": 1.0 if is_left else 0.0,
                    },
                    region=Region.TMJ_LEFT if is_left else Region.TMJ_RIGHT,
                )
            )
        return results

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _interior_mask(mineralised: volumetrics.BoolArray) -> volumetrics.BoolArray:
        """Cells with bone on both sides along at least two axes."""
        bracketed = np.zeros(mineralised.shape, dtype=np.int8)
        for axis in range(3):
            before = np.maximum.accumulate(mineralised, axis=axis)
            after = np.flip(
                np.maximum.accumulate(np.flip(mineralised, axis=axis), axis=axis), axis=axis
            )
            bracketed += (before & after).astype(np.int8)
        return bracketed >= 2


def _box_volume_mm3(component: volumetrics.Component, grid: volumetrics.Grid) -> float:
    size_z, size_y, size_x = component.size
    return size_x * grid.spacing[0] * size_y * grid.spacing[1] * size_z * grid.spacing[2]


def _features(
    component: volumetrics.Component,
    grid: volumetrics.Grid,
    box: BoundingBox3D,
    landmarks: dict[str, float],
    volume: VolumeInput,
    *,
    family: str,
    fill: float,
) -> dict[str, float]:
    occlusal = landmarks.get("occlusal_z", 0.5)
    midline = landmarks.get("midline_x", 0.5)
    x_rel, y_rel, z_rel = box.center
    _ = grid
    return {
        f"family_{family}": 1.0,
        "volume_mm3": component.volume_mm3,
        "extent_mm": component.extent_mm,
        "elongation": component.elongation,
        "enclosure": component.enclosure,
        "fill": float(fill),
        "mean_stored": component.mean_value,
        "max_stored": component.max_value,
        # The single most useful discriminator in the whole classifier. A
        # cystic lesion is fluid — near 0 HU — while the trabecular marrow it
        # otherwise resembles in shape, size and enclosure reads several
        # hundred HU higher. Without this the pipeline reports every marrow
        # space in a healthy mandible as a lesion.
        "mean_hu": volume.to_hu(component.mean_value),
        "max_hu": volume.to_hu(component.max_value),
        "upper": 1.0 if z_rel >= occlusal else 0.0,
        "z_rel": z_rel,
        "y_rel": y_rel,
        "x_rel": x_rel,
        "distance_to_occlusal": abs(z_rel - occlusal),
        "side": 1.0 if x_rel >= midline else 0.0,
    }


def _region_enum(box: BoundingBox3D, landmarks: dict[str, float]) -> Region:
    return Region(region_for(box.center, landmarks))


def _polar_profiles(
    has_tooth: volumetrics.BoolArray,
    has_ridge: volumetrics.BoolArray,
    *,
    centre_x: float,
    centre_y: float,
) -> tuple[volumetrics.FloatArray, volumetrics.FloatArray, volumetrics.FloatArray]:
    """Sweep the arch angularly, returning tooth, ridge and radius per bin.

    Each cell of the two masks is assigned to the angular bin it falls in, and
    the bins are reduced independently. Doing it by binning the cells rather
    than by casting rays keeps the cost to two passes over the slab and avoids
    the sampling artefacts a ray walk produces near the origin.
    """
    height, width = has_ridge.shape
    rows, columns = np.nonzero(has_ridge)
    tooth_profile = np.zeros(_ARCH_BINS, dtype=np.float32)
    ridge_profile = np.zeros(_ARCH_BINS, dtype=np.float32)
    radius_profile = np.zeros(_ARCH_BINS, dtype=np.float32)
    if rows.size == 0:
        return tooth_profile, ridge_profile, radius_profile

    offset_x = columns.astype(np.float64) - centre_x
    offset_y = rows.astype(np.float64) - centre_y
    # Angle measured from the patient's right, sweeping forward and round to
    # the left, so bin 0 is the right molar region and the last bin the left.
    angles = np.arctan2(-offset_y, offset_x)
    bins = np.clip(((np.pi - angles) / np.pi * _ARCH_BINS).astype(np.int64), 0, _ARCH_BINS - 1)
    radii = np.hypot(offset_x, offset_y)

    counts = np.bincount(bins, minlength=_ARCH_BINS).astype(np.float64)
    occupied = counts > 0
    ridge_profile[occupied] = 1.0

    tooth_hits = np.bincount(
        bins, weights=has_tooth[rows, columns].astype(np.float64), minlength=_ARCH_BINS
    )
    # A bin counts as toothed if any cell in it is dense: a crown occupies only
    # part of the radial extent the bin covers.
    tooth_profile[tooth_hits > 0] = 1.0

    radius_sums = np.bincount(bins, weights=radii, minlength=_ARCH_BINS)
    np.divide(radius_sums, counts, out=radius_profile, where=occupied, casting="unsafe")
    _ = height, width
    return tooth_profile, ridge_profile, radius_profile


def _arc_box(
    start: int,
    end: int,
    mean_radius: float,
    *,
    centre_x: float,
    centre_y: float,
    z0: int,
    z1: int,
    shape: tuple[int, int, int],
) -> BoundingBox3D:
    """Bounding prism of an angular run, back in normalised volume coordinates."""
    depth, height, width = shape
    angles = np.pi - (np.linspace(start, end, 8) / _ARCH_BINS) * np.pi
    xs = centre_x + mean_radius * np.cos(angles)
    ys = centre_y - mean_radius * np.sin(angles)

    x0 = float(np.min(xs)) / width
    x1 = float(np.max(xs)) / width
    y0 = float(np.min(ys)) / height
    y1 = float(np.max(ys)) / height
    return BoundingBox3D(
        x=x0,
        y=y0,
        z=z0 / depth,
        # A span of one bin is still a real gap; give it a minimum footprint so
        # the overlay is visible and the CHECK constraint is satisfied.
        width=max(x1 - x0, 0.02),
        height=max(y1 - y0, 0.02),
        depth=max((z1 - z0) / depth, 0.02),
    ).clamped()


def _runs_below(
    profile: volumetrics.FloatArray, *, threshold: float, min_length: int
) -> list[tuple[int, int]]:
    """Index ranges where ``profile`` stays under ``threshold``."""
    below = profile < threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(below):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                runs.append((start, index))
            start = None
    if start is not None and below.size - start >= min_length:
        runs.append((start, int(below.size)))
    return runs
