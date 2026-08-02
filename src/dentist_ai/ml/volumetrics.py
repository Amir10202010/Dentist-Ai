"""Numerical primitives the analysis stages are built from.

Everything here is plain NumPy. That is a constraint, not an accident: the
application's install is Pillow and NumPy, and a clinic running the stub
detector should not need a 900 MB scientific stack to open a CBCT. SciPy would
supply half of this module, and the day a trained backend arrives it can be
added as an optional dependency alongside torch — but the default deployment
keeps working with what is here.

Two things shape the implementations.

**Everything runs on a coarse grid.** Lesion-scale structure is millimetres
across; a 64³ analysis grid resolves it at roughly 2 mm per cell, which is
enough to find a lucency and far cheaper than working at 256³. Boxes are
carried in normalised coordinates, so a finding located on the coarse grid
lands correctly on the full-resolution volume in the viewer.

**Nothing loops over voxels in Python.** Connected components are found by
iterated label propagation rather than a recursive flood fill, because a pure
Python flood fill over a quarter of a million cells is seconds, and the same
work as a handful of vectorised array operations is milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
#: Coordinates and accumulators computed with ``np.bincount``, which promotes
#: to double precision regardless of the input dtype.
WideArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]
#: The per-cell tissue class written by the segmentation stage.
TissueArray = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]

#: Longest edge of the analysis grid, chosen so a typical 100 mm dental field
#: of view is analysed at roughly one cell per millimetre.
#:
#: This is the single most consequential number in the analysis. Below it the
#: measurements stop meaning anything clinically: at 2 mm cells a periapical
#: lesion is two cells across, its "elongation" and "roundness" are quantisation
#: noise, and the classifier ends up reasoning about artefacts of the grid. At
#: 1 mm a lesion is 3-8 cells, an implant is 4 × 10, and the shape features are
#: measurements rather than rounding.
#:
#: The cost is connected-component labelling, which is superlinear in cell
#: count — roughly 300 ms per pass here against 40 ms at half the resolution.
#: Three passes per analysis is a second of CPU, which is the right trade for
#: findings that are about the anatomy instead of about the sampling.
ANALYSIS_GRID: Final[int] = 112
#: Ceiling on label-propagation sweeps. A compact blob converges in about its
#: own radius; the cap bounds the pathological case of a long thin structure
#: without changing the answer for anything clinically interesting.
_MAX_PROPAGATION_SWEEPS: Final[int] = 96
#: Tolerance for float comparisons on intensities.
_EPSILON: Final[float] = 1e-6
#: Fewer samples than this and a histogram-derived threshold is guesswork.
_MIN_HISTOGRAM_SAMPLES: Final[int] = 64
#: Shortest profile a valley search can say anything about.
_MIN_PROFILE_POINTS: Final[int] = 8
#: Narrowest interior window worth searching within that profile.
_MIN_VALLEY_WIDTH: Final[int] = 3
#: Shortest smoothing window that averages anything.
_MIN_SMOOTH_WINDOW: Final[int] = 2
#: An axis shorter than this has no first difference.
_MIN_AXIS_LENGTH: Final[int] = 2


@dataclass(frozen=True, slots=True)
class Grid:
    """A volume resampled onto the analysis grid, plus how to get back."""

    values: FloatArray
    #: Millimetres per grid cell, ``(x, y, z)``.
    spacing: tuple[float, float, float]
    #: Cells per axis, ``(depth, height, width)``.
    shape: tuple[int, int, int]
    #: ``modality_units = stored * hu_slope + hu_intercept``. Carried on the
    #: grid so thresholding can reason in Hounsfield units, which are a
    #: physical fact, rather than only in the stored levels, which are an
    #: artefact of how ingest rescaled this particular volume.
    hu_slope: float = 1.0
    hu_intercept: float = 0.0

    @property
    def cell_volume_mm3(self) -> float:
        return self.spacing[0] * self.spacing[1] * self.spacing[2]

    def from_hu(self, hounsfield: float) -> float:
        """A Hounsfield value expressed in this grid's stored levels."""
        return (hounsfield - self.hu_intercept) / max(self.hu_slope, 1e-6)

    def normalise(self, z: float, y: float, x: float) -> tuple[float, float, float]:
        """Grid indices to ``[0, 1]`` volume coordinates, in ``(x, y, z)``."""
        depth, height, width = self.shape
        return (x / max(width, 1), y / max(height, 1), z / max(depth, 1))


def to_grid(
    voxels: npt.NDArray[np.uint8],
    spacing: tuple[float, float, float],
    *,
    target: int = ANALYSIS_GRID,
    hu_slope: float = 1.0,
    hu_intercept: float = 0.0,
) -> Grid:
    """Block-average onto the analysis grid.

    Averaging rather than sampling, for the reason ingest decimates the same
    way: a reconstruction is noisy and dropping cells aliases that noise into
    exactly the low-density pockets the detection stage is looking for.
    """
    depth, height, width = (int(value) for value in voxels.shape)
    factor_z = max(1, -(-depth // target))
    factor_y = max(1, -(-height // target))
    factor_x = max(1, -(-width // target))

    cropped = voxels[
        : (depth // factor_z) * factor_z,
        : (height // factor_y) * factor_y,
        : (width // factor_x) * factor_x,
    ]
    reduced = (
        cropped.reshape(
            cropped.shape[0] // factor_z,
            factor_z,
            cropped.shape[1] // factor_y,
            factor_y,
            cropped.shape[2] // factor_x,
            factor_x,
        )
        .mean(axis=(1, 3, 5), dtype=np.float32)
        .astype(np.float32)
    )

    grid_depth, grid_height, grid_width = (int(value) for value in reduced.shape)
    return Grid(
        values=np.ascontiguousarray(reduced),
        spacing=(
            spacing[0] * factor_x,
            spacing[1] * factor_y,
            spacing[2] * factor_z,
        ),
        shape=(grid_depth, grid_height, grid_width),
        hu_slope=hu_slope,
        hu_intercept=hu_intercept,
    )


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------
def otsu(values: FloatArray, *, bins: int = 128) -> float:
    """Otsu's threshold: the split that minimises within-class variance.

    Computed on a histogram rather than on the samples, so the cost is a
    function of ``bins`` and not of the volume size.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    low = float(finite.min())
    high = float(finite.max())
    if high - low < _EPSILON:
        return low

    counts, edges = np.histogram(finite, bins=bins, range=(low, high))
    weights = counts.astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return low

    centres = (edges[:-1] + edges[1:]) / 2
    weight_below = np.cumsum(weights)
    weight_above = total - weight_below
    # Guard the tails: a bin where one side is empty has no between-class
    # variance to speak of and would otherwise divide by zero.
    valid = (weight_below > 0) & (weight_above > 0)
    if not valid.any():
        return float(np.median(finite))

    sum_below = np.cumsum(weights * centres)
    sum_total = sum_below[-1]
    mean_below = np.divide(sum_below, weight_below, out=np.zeros_like(sum_below), where=valid)
    mean_above = np.divide(
        sum_total - sum_below, weight_above, out=np.zeros_like(sum_below), where=valid
    )
    between = weight_below * weight_above * (mean_below - mean_above) ** 2
    between[~valid] = -1.0
    return float(centres[int(np.argmax(between))])


@dataclass(frozen=True, slots=True)
class TissueThresholds:
    """Where the boundaries between tissue classes fall, in stored units."""

    air_soft: float
    soft_bone: float
    dense: float
    #: Upper bound of fluid and soft-tissue density — the level a cyst, a
    #: granuloma or the contents of a canal sit below, and trabecular bone
    #: sits above.
    #:
    #: Distinct from ``soft_bone``, and the distinction matters more than any
    #: other threshold in the system. Searching for lucencies below
    #: ``soft_bone`` finds every marrow space in the mandible and merges each
    #: real lesion into the marrow channel it sits inside, so a periapical
    #: lesion is reported as 40 mm of canal. Searching below this level finds
    #: the lesion as the focal, well-defined object it is.
    soft_ceiling: float
    #: Boundary between bone and tooth substance. Dentine is denser than
    #: cortical bone, which is what makes an alveolar crest measurable at all:
    #: taking the highest mineralised cell in a column finds the crown of the
    #: tooth standing in it, and reports a normal crest on a mouth that has
    #: lost half its support.
    tooth: float


#: Upper bound of fluid and soft-tissue density, in Hounsfield units. Water is
#: 0, muscle about 50, and trabecular bone starts several hundred above this,
#: so the boundary is a physical fact rather than a tuned constant.
_SOFT_TISSUE_CEILING_HU: Final[float] = 300.0


def _soft_ceiling(grid: Grid, air_soft: float, boundary: float) -> float:
    """Where fluid stops and mineralised tissue starts, in stored levels.

    Anchored in Hounsfield units, because that is a property of tissue rather
    than of this volume's histogram. Deriving it purely from the histogram —
    as a fraction of the way from the air boundary to the bone boundary —
    looks equivalent and is not: the fraction moves whenever the *mix* of
    tissue in the field of view changes, so restoring two crowns shifts the
    threshold that decides whether a lesion elsewhere in the jaw is found.

    The Otsu-derived range is kept as a clamp, for the scanner whose grey
    values are not Hounsfield units at all.
    """
    span = max(boundary - air_soft, 1e-6)
    anchored = grid.from_hu(_SOFT_TISSUE_CEILING_HU)
    return float(np.clip(anchored, air_soft + 0.15 * span, air_soft + 0.60 * span))


def tissue_thresholds(grid: Grid) -> TissueThresholds:
    """Split the volume into air, soft tissue, bone and dense material.

    Two applications of Otsu rather than one: the first separates the field of
    view — mostly air — from the patient, and the second, applied only to what
    is left, separates soft tissue from mineralised tissue. Running one global
    threshold instead puts the boundary in the middle of the air peak, because
    air is the majority class in every CBCT.
    """
    values = grid.values
    air_soft = otsu(values)

    tissue = values[values > air_soft]
    soft_bone = (
        otsu(tissue)
        if tissue.size > _MIN_HISTOGRAM_SAMPLES
        else air_soft + (255.0 - air_soft) * 0.45
    )

    mineralised = values[values > soft_bone]
    # Enamel, cortical bone and metal all live above the bone threshold; the
    # high percentile picks out the restorations among them.
    dense = (
        float(np.percentile(mineralised, 96))
        if mineralised.size > _MIN_HISTOGRAM_SAMPLES
        else 255.0
    )

    boundary = float(max(soft_bone, air_soft + 1.0))
    return TissueThresholds(
        air_soft=float(air_soft),
        soft_bone=boundary,
        dense=float(max(dense, soft_bone + 1.0)),
        # Placed at 45% of the way from the air boundary to the bone boundary,
        # which for a dental CBCT lands a little above muscle and well below
        # trabecular bone. A fraction rather than a fixed Hounsfield value
        # because cone-beam grey levels are not calibrated: the same cortical
        # plate reads differently on two machines, but its position relative
        # to that machine's own air and bone peaks is stable.
        soft_ceiling=_soft_ceiling(grid, air_soft, boundary),
        tooth=float(boundary + 0.5 * (max(dense, boundary + 1.0) - boundary)),
    )


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------
def label_components(mask: BoolArray) -> tuple[IntArray, int]:
    """Label 6-connected regions of a boolean volume.

    Each cell starts as its own label and repeatedly takes the maximum of
    itself and its six neighbours; a connected region converges to the largest
    label it contains. Sweeps alternate direction so information travels both
    ways along an axis, which roughly halves the number needed.
    """
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), 0

    labels = np.where(
        mask,
        np.arange(1, mask.size + 1, dtype=np.int32).reshape(mask.shape),
        np.int32(0),
    )

    for sweep in range(_MAX_PROPAGATION_SWEEPS):
        previous = labels
        working = labels if sweep % 2 == 0 else labels[::-1, ::-1, ::-1]
        neighbours = working.copy()
        neighbours[1:, :, :] = np.maximum(neighbours[1:, :, :], working[:-1, :, :])
        neighbours[:, 1:, :] = np.maximum(neighbours[:, 1:, :], working[:, :-1, :])
        neighbours[:, :, 1:] = np.maximum(neighbours[:, :, 1:], working[:, :, :-1])
        neighbours[:-1, :, :] = np.maximum(neighbours[:-1, :, :], working[1:, :, :])
        neighbours[:, :-1, :] = np.maximum(neighbours[:, :-1, :], working[:, 1:, :])
        neighbours[:, :, :-1] = np.maximum(neighbours[:, :, :-1], working[:, :, 1:])

        labels = neighbours if sweep % 2 == 0 else neighbours[::-1, ::-1, ::-1]
        labels = np.where(mask, labels, np.int32(0))
        if np.array_equal(labels, previous):
            break

    # Compact the sparse label values to 1..n so callers can index by label.
    present = np.unique(labels)
    present = present[present > 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[present] = np.arange(1, present.size + 1, dtype=np.int32)
    return remap[labels], int(present.size)


@dataclass(frozen=True, slots=True)
class Component:
    """A connected region and the measurements a classifier reasons over."""

    label: int
    cell_count: int
    #: Inclusive-exclusive bounds on the analysis grid.
    bounds: tuple[int, int, int, int, int, int]
    #: Centre of mass in grid indices, ``(z, y, x)``.
    centroid: tuple[float, float, float]
    mean_value: float
    max_value: float
    volume_mm3: float
    #: Longest bounding-box edge, in millimetres.
    extent_mm: float
    #: Ratio of the longest bounding-box edge to the shortest. 1 is a cube,
    #: large values are the tubular structures — canals, roots, wires.
    elongation: float
    #: Fraction of the region's boundary cells that touch mineralised tissue.
    #: A cyst is enclosed by bone; the airway is not.
    enclosure: float

    @property
    def size(self) -> tuple[int, int, int]:
        z0, z1, y0, y1, x0, x1 = self.bounds
        return (z1 - z0, y1 - y0, x1 - x0)


def measure_components(
    labels: IntArray,
    count: int,
    grid: Grid,
    *,
    enclosing: BoolArray | None = None,
    tolerate_partial_volume: bool = True,
    min_cells: int = 6,
) -> list[Component]:
    """Measure every labelled region, discarding ones too small to be real."""
    if count == 0:
        return []

    flat_labels = labels.reshape(-1)
    flat_values = grid.values.reshape(-1)
    occupied = flat_labels > 0
    if not occupied.any():
        return []

    indices = np.nonzero(occupied)[0]
    owners = flat_labels[indices]
    _depth, height, width = grid.shape
    z_index = (indices // (height * width)).astype(np.float64)
    y_index = ((indices // width) % height).astype(np.float64)
    x_index = (indices % width).astype(np.float64)
    values = flat_values[indices].astype(np.float64)

    size = count + 1
    counts = np.bincount(owners, minlength=size)
    sum_z = np.bincount(owners, weights=z_index, minlength=size)
    sum_y = np.bincount(owners, weights=y_index, minlength=size)
    sum_x = np.bincount(owners, weights=x_index, minlength=size)
    sum_v = np.bincount(owners, weights=values, minlength=size)
    max_v = np.zeros(size, dtype=np.float64)
    np.maximum.at(max_v, owners, values)

    min_z = _extreme(owners, z_index, size, largest=False)
    max_z = _extreme(owners, z_index, size, largest=True)
    min_y = _extreme(owners, y_index, size, largest=False)
    max_y = _extreme(owners, y_index, size, largest=True)
    min_x = _extreme(owners, x_index, size, largest=False)
    max_x = _extreme(owners, x_index, size, largest=True)

    enclosure = _enclosure_ratios(
        labels, count, enclosing, tolerate_partial_volume=tolerate_partial_volume
    )
    spacing_x, spacing_y, spacing_z = grid.spacing

    components: list[Component] = []
    for label in range(1, size):
        cells = int(counts[label])
        if cells < min_cells:
            continue

        extents_mm = (
            (max_z[label] - min_z[label] + 1) * spacing_z,
            (max_y[label] - min_y[label] + 1) * spacing_y,
            (max_x[label] - min_x[label] + 1) * spacing_x,
        )
        longest = max(extents_mm)
        shortest = max(min(extents_mm), 1e-6)

        components.append(
            Component(
                label=label,
                cell_count=cells,
                bounds=(
                    int(min_z[label]),
                    int(max_z[label]) + 1,
                    int(min_y[label]),
                    int(max_y[label]) + 1,
                    int(min_x[label]),
                    int(max_x[label]) + 1,
                ),
                centroid=(
                    float(sum_z[label] / cells),
                    float(sum_y[label] / cells),
                    float(sum_x[label] / cells),
                ),
                mean_value=float(sum_v[label] / cells),
                max_value=float(max_v[label]),
                volume_mm3=cells * grid.cell_volume_mm3,
                extent_mm=float(longest),
                elongation=float(longest / shortest),
                enclosure=float(enclosure[label]),
            )
        )

    components.sort(key=lambda item: item.volume_mm3, reverse=True)
    return components


def _extreme(owners: IntArray, coords: WideArray, size: int, *, largest: bool) -> WideArray:
    out = np.full(size, -np.inf if largest else np.inf, dtype=np.float64)
    if largest:
        np.maximum.at(out, owners, coords)
    else:
        np.minimum.at(out, owners, coords)
    out[~np.isfinite(out)] = 0.0
    return out


def dilate(mask: BoolArray) -> BoolArray:
    """Grow a mask by one cell in each of the six axis directions."""
    grown = mask.copy()
    grown[1:, :, :] |= mask[:-1, :, :]
    grown[:-1, :, :] |= mask[1:, :, :]
    grown[:, 1:, :] |= mask[:, :-1, :]
    grown[:, :-1, :] |= mask[:, 1:, :]
    grown[:, :, 1:] |= mask[:, :, :-1]
    grown[:, :, :-1] |= mask[:, :, 1:]
    return grown


def _enclosure_ratios(
    labels: IntArray,
    count: int,
    enclosing: BoolArray | None,
    *,
    tolerate_partial_volume: bool,
) -> WideArray:
    """For each label, the fraction of its surface that abuts ``enclosing``.

    Computed for all labels at once by comparing shifted label arrays, so the
    cost does not scale with the number of regions.

    ``tolerate_partial_volume`` dilates the enclosing mask by one cell first,
    and choosing it correctly is what makes the metric mean anything.

    Turn it **on** for low-contrast boundaries. At the rim of a fluid lesion,
    cells that are half fluid and half bone average below the bone threshold,
    so testing immediate neighbours finds that transitional layer rather than
    the bone behind it — and a lesion sitting entirely within the mandible
    measures as barely enclosed. One cell of slack measures the anatomy
    instead of the sampling, while a genuine cortical breach, which is many
    cells wide, stays clearly visible.

    Turn it **off** for high-contrast boundaries. Enamel against bone has no
    meaningful transitional layer, and the slack would instead reach across
    the gap between a crown and the root beneath it — making a crown, whose
    defining property is that it is *not* surrounded by bone, measure as fully
    enclosed and become indistinguishable from an implant.
    """
    ratios = np.zeros(count + 1, dtype=np.float64)
    if enclosing is None or count == 0:
        return ratios
    if tolerate_partial_volume:
        enclosing = dilate(enclosing)

    boundary_total = np.zeros(count + 1, dtype=np.float64)
    boundary_hit = np.zeros(count + 1, dtype=np.float64)

    for axis in range(3):
        for shift in (1, -1):
            shifted_labels = np.roll(labels, shift, axis=axis)
            shifted_enclosing = np.roll(enclosing, shift, axis=axis)
            # A boundary cell is one whose neighbour belongs to a different
            # region; the neighbour being mineralised is what "enclosed" means.
            outward = (labels > 0) & (shifted_labels != labels)
            owners = labels[outward]
            if owners.size == 0:
                continue
            np.add.at(boundary_total, owners, 1.0)
            np.add.at(boundary_hit, owners, shifted_enclosing[outward].astype(np.float64))

    np.divide(
        boundary_hit,
        boundary_total,
        out=ratios,
        where=boundary_total > 0,
    )
    return ratios


# ---------------------------------------------------------------------------
# Profiles and landmarks
# ---------------------------------------------------------------------------
def axis_profile(mask: BoolArray, axis: int) -> FloatArray:
    """Fraction of cells set, per index along ``axis``."""
    others = tuple(index for index in range(3) if index != axis)
    return mask.mean(axis=others, dtype=np.float32)


def largest_valley(profile: FloatArray, *, margin: float = 0.2) -> int:
    """Index of the lowest point between the profile's two dominant peaks.

    Used to find the occlusal plane: mineralised tissue forms two bands along
    the superior-inferior axis — the mandibular and maxillary arches — and the
    gap between them is where the jaws meet. The margin keeps the search away
    from the ends, where the profile falls off simply because the field of
    view does.
    """
    length = profile.size
    if length < _MIN_PROFILE_POINTS:
        return length // 2

    low = int(length * margin)
    high = int(length * (1.0 - margin))
    if high - low < _MIN_VALLEY_WIDTH:
        return length // 2

    interior = profile[low:high]
    peak = int(np.argmax(profile))
    # Split the search at the global peak so the valley found is between two
    # bands rather than on the far shoulder of a single one.
    if low < peak < high - 1:
        left = profile[low:peak]
        right = profile[peak:high]
        return (
            low + int(np.argmin(left))
            if left.size and float(left.min()) <= float(right.min())
            else peak + int(np.argmin(right))
        )
    return low + int(np.argmin(interior))


def bounding_box_of(mask: BoolArray) -> tuple[int, int, int, int, int, int] | None:
    """Tight bounds of a mask as ``(z0, z1, y0, y1, x0, x1)``, or ``None``."""
    if not mask.any():
        return None
    z_any = mask.any(axis=(1, 2))
    y_any = mask.any(axis=(0, 2))
    x_any = mask.any(axis=(0, 1))
    z_indices = np.nonzero(z_any)[0]
    y_indices = np.nonzero(y_any)[0]
    x_indices = np.nonzero(x_any)[0]
    return (
        int(z_indices[0]),
        int(z_indices[-1]) + 1,
        int(y_indices[0]),
        int(y_indices[-1]) + 1,
        int(x_indices[0]),
        int(x_indices[-1]) + 1,
    )


def smooth(profile: FloatArray, window: int = 5) -> FloatArray:
    """Moving average with edge padding, for peak-finding on noisy profiles."""
    if profile.size <= window or window < _MIN_SMOOTH_WINDOW:
        return profile
    pad = window // 2
    padded = np.pad(profile, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")[: profile.size].astype(np.float32)


def gradient_energy(values: FloatArray, axis: int) -> float:
    """Mean absolute first difference along ``axis``.

    A proxy for how sharp the volume is in that direction. Comparing the
    through-plane value against the in-plane one is how the QC stage detects
    motion: patient movement blurs across slices while leaving each slice
    individually crisp.
    """
    if values.shape[axis] < _MIN_AXIS_LENGTH:
        return 0.0
    return float(np.abs(np.diff(values, axis=axis)).mean())
