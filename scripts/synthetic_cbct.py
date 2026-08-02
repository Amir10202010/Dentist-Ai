"""Build a synthetic dental CBCT volume.

A demo of a CBCT product needs a CBCT, and a real one is a patient. This
module builds a phantom instead: two parabolic alveolar arches, teeth with
crowns and roots, maxillary sinuses, an inferior alveolar canal, and whichever
pathology the caller asks for.

It is a phantom, not a simulation. Densities are in the right Hounsfield
neighbourhood and the geometry is anatomically arranged, which is enough to
exercise every stage of the analysis pipeline honestly — the segmentation
really does have to find the occlusal plane, and the detector really does have
to find the lesion among the marrow spaces. It is not enough to evaluate
against, and nothing here should be mistaken for validation data.

Written out as NIfTI-1 because it is the one volumetric format that is a
header and a block of voxels, so the round trip through
``services/volume.py`` exercises a real parser rather than a private one.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

VolumeArray = npt.NDArray[np.int16]
#: The working buffer during rendering, before quantisation to int16.
FloatVolume = npt.NDArray[np.float32]

# Hounsfield units, roughly where a cone-beam reconstruction puts each tissue.
HU_AIR: Final[int] = -1000
HU_SOFT: Final[int] = 60
HU_MARROW: Final[int] = 700
HU_CORTICAL: Final[int] = 1100
HU_DENTINE: Final[int] = 1500
HU_ENAMEL: Final[int] = 2400
HU_METAL: Final[int] = 3000

#: FDI numbers along each quadrant, central incisor outwards.
_QUADRANT_TEETH: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(slots=True)
class Pathology:
    """What to place in the phantom beyond normal anatomy."""

    #: FDI numbers to leave out of the arch entirely.
    missing_teeth: tuple[int, ...] = ()
    #: FDI numbers to give a periapical radiolucency.
    apical_lesions: tuple[int, ...] = ()
    #: FDI numbers to fit with a titanium fixture instead of a natural tooth.
    implants: tuple[int, ...] = ()
    #: FDI numbers whose canals are obturated.
    root_fillings: tuple[int, ...] = ()
    #: Radius in millimetres of a cyst in the posterior mandible, if any.
    cyst_radius_mm: float = 0.0
    #: Millimetres to drop the crest by, across the posterior segments.
    bone_loss_mm: float = 0.0
    #: Place an unerupted third molar in the mandible.
    impacted_third_molar: bool = False
    #: Simulate patient movement as a lateral shift accumulating over slices.
    motion_mm: float = 0.0
    #: Extra noise beyond the baseline, in Hounsfield units.
    extra_noise: float = 0.0


@dataclass(slots=True)
class Phantom:
    """Geometry of the phantom, in millimetres."""

    #: Voxel counts, ``(depth, height, width)`` — z, y, x.
    shape: tuple[int, int, int] = (180, 200, 200)
    spacing: tuple[float, float, float] = (0.5, 0.5, 0.5)
    seed: int = 0
    pathology: Pathology = field(default_factory=Pathology)

    @property
    def depth(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[2]

    def mm_to_voxels(self, millimetres: float, axis: int) -> float:
        """Axis 0 is z, 1 is y, 2 is x — matching ``shape``."""
        spacing = (self.spacing[2], self.spacing[1], self.spacing[0])[axis]
        return millimetres / spacing


def build(phantom: Phantom) -> VolumeArray:
    """Render the phantom to a Hounsfield-unit volume."""
    rng = np.random.default_rng(phantom.seed)
    # Built in float32 so the stamping primitives can blend, then quantised to
    # int16 at the end — which is the precision a real reconstruction carries.
    volume: FloatVolume = np.full(phantom.shape, HU_AIR, dtype=np.float32)

    _add_head(volume, phantom)
    occlusal_z = phantom.depth * 0.5

    _add_arch(volume, phantom, occlusal_z=occlusal_z, upper=False)
    _add_arch(volume, phantom, occlusal_z=occlusal_z, upper=True)
    _add_sinuses(volume, phantom, occlusal_z=occlusal_z)
    _add_mandibular_canal(volume, phantom, occlusal_z=occlusal_z)
    _add_teeth(volume, phantom, occlusal_z=occlusal_z, rng=rng)
    _add_pathology(volume, phantom, occlusal_z=occlusal_z, rng=rng)

    noise = 28.0 + phantom.pathology.extra_noise
    volume += rng.normal(0.0, noise, size=volume.shape).astype(np.float32)

    if phantom.pathology.motion_mm > 0:
        volume = _apply_motion(volume, phantom)

    return np.clip(volume, -1024, 3200).astype(np.int16)


# ---------------------------------------------------------------------------
# Anatomy
# ---------------------------------------------------------------------------
def _add_head(volume: npt.NDArray[np.float32], phantom: Phantom) -> None:
    """Soft tissue envelope, so the volume is not a skeleton floating in air."""
    z, y, x = _grids(phantom)
    centre_z = phantom.depth * 0.5
    centre_y = phantom.height * 0.52
    centre_x = phantom.width * 0.5

    inside = (
        ((x - centre_x) / (phantom.width * 0.42)) ** 2
        + ((y - centre_y) / (phantom.height * 0.40)) ** 2
        + ((z - centre_z) / (phantom.depth * 0.46)) ** 2
    ) <= 1.0
    volume[inside] = HU_SOFT


def _arch_points(phantom: Phantom, *, upper: bool) -> list[tuple[float, float]]:
    """Sample the alveolar arch as a parabola in the axial plane.

    The maxillary arch is a little wider and a little further forward than the
    mandibular one, which is what gives a normal overjet.
    """
    half_width = phantom.width * (0.30 if upper else 0.285)
    depth_scale = phantom.height * (0.30 if upper else 0.29)
    front_y = phantom.height * (0.26 if upper else 0.275)
    centre_x = phantom.width * 0.5

    points: list[tuple[float, float]] = []
    for index in range(160):
        t = -1.0 + 2.0 * index / 159
        points.append((centre_x + t * half_width, front_y + depth_scale * t * t))
    return points


def _add_arch(
    volume: npt.NDArray[np.float32],
    phantom: Phantom,
    *,
    occlusal_z: float,
    upper: bool,
) -> None:
    """The alveolar bone: a cortical shell around a marrow core."""
    height_mm = 26.0
    height_voxels = phantom.mm_to_voxels(height_mm, axis=0)
    z_low = occlusal_z + 2 if upper else occlusal_z - height_voxels - 2
    z_high = occlusal_z + height_voxels + 2 if upper else occlusal_z - 2

    radius = phantom.mm_to_voxels(5.4, axis=2)
    loss = phantom.pathology.bone_loss_mm

    for index, (x, y) in enumerate(_arch_points(phantom, upper=upper)):
        # Posterior segments recede first in periodontal disease, so the crest
        # drop is applied away from the midline.
        # Posterior segments recede; the anterior ones are spared. A smooth
        # gradient from midline to molar would be a uniform recession, which
        # is neither what periodontitis looks like nor something a
        # crest-height comparison against the same arch can detect.
        posterior = 1.0 if abs(index / 159 - 0.5) * 2 > 0.45 else 0.0
        drop = phantom.mm_to_voxels(loss * posterior, axis=0) if loss > 0 else 0.0
        low = z_low + (drop if upper else 0.0)
        high = z_high - (drop if not upper else 0.0)

        _stamp_column(volume, phantom, x, y, low, high, radius, HU_CORTICAL)
        _stamp_column(volume, phantom, x, y, low + 1.5, high - 1.5, radius * 0.62, HU_MARROW)


def _add_sinuses(volume: npt.NDArray[np.float32], phantom: Phantom, *, occlusal_z: float) -> None:
    """Two air spaces above the posterior maxilla."""
    z, y, x = _grids(phantom)
    floor = occlusal_z + phantom.mm_to_voxels(16.0, axis=0)

    for side in (-1, 1):
        centre_x = phantom.width * (0.5 + side * 0.20)
        centre_y = phantom.height * 0.46
        centre_z = floor + phantom.mm_to_voxels(11.0, axis=0)
        inside = (
            ((x - centre_x) / phantom.mm_to_voxels(13.0, axis=2)) ** 2
            + ((y - centre_y) / phantom.mm_to_voxels(15.0, axis=1)) ** 2
            + ((z - centre_z) / phantom.mm_to_voxels(12.0, axis=0)) ** 2
        ) <= 1.0
        volume[inside] = HU_AIR


def _add_mandibular_canal(
    volume: npt.NDArray[np.float32], phantom: Phantom, *, occlusal_z: float
) -> None:
    """The inferior alveolar canal, running below the mandibular roots."""
    canal_z = occlusal_z - phantom.mm_to_voxels(24.0, axis=0)
    radius = phantom.mm_to_voxels(1.5, axis=2)

    for x, y in _arch_points(phantom, upper=False):
        # The canal sits buccal to the arch curve and a little deeper. Its
        # contents are the neurovascular bundle — fat and vessels, near soft
        # tissue density, not the several hundred units of the marrow it runs
        # through, which is exactly why it reads as a lucency.
        _stamp_sphere(
            volume,
            phantom,
            (canal_z, y + phantom.mm_to_voxels(1.5, axis=1), x),
            (radius, radius, radius),
            HU_SOFT,
        )


def _tooth_positions(phantom: Phantom, *, upper: bool) -> dict[int, tuple[float, float]]:
    """FDI number to axial position along the arch.

    Quadrant 1 is the patient's upper right, 2 upper left, 3 lower left and 4
    lower right — and the patient's right is low ``x``, matching the DICOM
    convention the analysis stages assume.
    """
    points = _arch_points(phantom, upper=upper)
    positions: dict[int, tuple[float, float]] = {}
    right_quadrant, left_quadrant = (1, 2) if upper else (4, 3)

    for order, tooth in enumerate(_QUADRANT_TEETH):
        # Teeth crowd toward the midline; sampling the parabola linearly in
        # index puts them at plausible relative spacings.
        offset = 0.06 + order * 0.115
        left_index = min(len(points) - 1, int((0.5 + offset / 2) * (len(points) - 1)))
        right_index = max(0, int((0.5 - offset / 2) * (len(points) - 1)))
        positions[left_quadrant * 10 + tooth] = points[left_index]
        positions[right_quadrant * 10 + tooth] = points[right_index]
    return positions


def _add_teeth(
    volume: npt.NDArray[np.float32],
    phantom: Phantom,
    *,
    occlusal_z: float,
    rng: np.random.Generator,
) -> None:
    pathology = phantom.pathology

    for upper in (False, True):
        direction = 1 if upper else -1
        for tooth, (x, y) in _tooth_positions(phantom, upper=upper).items():
            if tooth in pathology.missing_teeth:
                continue
            if tooth in pathology.implants:
                _add_implant(volume, phantom, x, y, occlusal_z, direction)
                continue

            molar = tooth % 10 >= 6
            # Sized so adjacent crowns approach contact without fusing into
            # one radiopaque mass at 1 mm sampling. Real crowns do contact,
            # and where they do, thresholding alone cannot separate them —
            # which is a real limit of this pipeline, not of the phantom.
            crown_radius = phantom.mm_to_voxels(3.9 if molar else 3.0, axis=2)
            crown_z = occlusal_z + direction * phantom.mm_to_voxels(4.0, axis=0)
            _stamp_sphere(
                volume,
                phantom,
                (crown_z, y, x),
                (phantom.mm_to_voxels(4.2, axis=0), crown_radius, crown_radius),
                HU_ENAMEL + rng.normal(0, 60),
            )

            root_top = occlusal_z + direction * phantom.mm_to_voxels(6.0, axis=0)
            root_tip = occlusal_z + direction * phantom.mm_to_voxels(15.0, axis=0)
            _stamp_column(
                volume,
                phantom,
                x,
                y,
                min(root_top, root_tip),
                max(root_top, root_tip),
                phantom.mm_to_voxels(2.4 if molar else 1.9, axis=2),
                HU_DENTINE,
            )

            if tooth in pathology.root_fillings:
                _stamp_column(
                    volume,
                    phantom,
                    x,
                    y,
                    min(root_top, root_tip) + 1,
                    max(root_top, root_tip) - 1,
                    phantom.mm_to_voxels(0.7, axis=2),
                    HU_METAL,
                )


def _add_implant(
    volume: npt.NDArray[np.float32],
    phantom: Phantom,
    x: float,
    y: float,
    occlusal_z: float,
    direction: int,
) -> None:
    """A titanium fixture: a dense cylinder in bone under a dense abutment."""
    body_top = occlusal_z + direction * phantom.mm_to_voxels(2.0, axis=0)
    body_end = occlusal_z + direction * phantom.mm_to_voxels(12.0, axis=0)
    _stamp_column(
        volume,
        phantom,
        x,
        y,
        min(body_top, body_end),
        max(body_top, body_end),
        phantom.mm_to_voxels(2.0, axis=2),
        HU_METAL,
    )


def _add_pathology(
    volume: npt.NDArray[np.float32],
    phantom: Phantom,
    *,
    occlusal_z: float,
    rng: np.random.Generator,
) -> None:
    pathology = phantom.pathology

    for tooth in pathology.apical_lesions:
        upper = tooth // 10 in (1, 2)
        positions = _tooth_positions(phantom, upper=upper)
        if tooth not in positions:
            continue
        x, y = positions[tooth]
        direction = 1 if upper else -1
        apex_z = occlusal_z + direction * phantom.mm_to_voxels(17.0, axis=0)
        radius = phantom.mm_to_voxels(rng.uniform(2.6, 3.6), axis=2)
        _stamp_sphere(volume, phantom, (apex_z, y, x), (radius, radius, radius), HU_SOFT)

    if pathology.cyst_radius_mm > 0:
        radius = phantom.mm_to_voxels(pathology.cyst_radius_mm, axis=2)
        # On the arch, not beside it. A sphere placed at an arbitrary point in
        # the mandibular half of the volume lands in soft tissue, where it is
        # not a cyst and the detector is right to ignore it.
        centre_x, centre_y = _arch_points(phantom, upper=False)[24]
        centre_z = occlusal_z - phantom.mm_to_voxels(12.0, axis=0)
        _stamp_sphere(
            volume, phantom, (centre_z, centre_y, centre_x), (radius, radius, radius), HU_SOFT * 0.6
        )

    if pathology.impacted_third_molar:
        points = _arch_points(phantom, upper=False)
        x, y = points[-6]
        crown_z = occlusal_z - phantom.mm_to_voxels(11.0, axis=0)
        radius = phantom.mm_to_voxels(4.4, axis=2)
        _stamp_sphere(
            volume,
            phantom,
            (crown_z, y + phantom.mm_to_voxels(3.0, axis=1), x),
            (radius * 0.9, radius, radius),
            HU_ENAMEL,
        )


def _apply_motion(volume: FloatVolume, phantom: Phantom) -> FloatVolume:
    """Shift each slice a little further than the last.

    A crude but faithful model of the artefact that matters: every slice stays
    internally sharp while the stack stops being registered, which is exactly
    the signal the QC stage measures.
    """
    shifted = np.empty_like(volume)
    amplitude = phantom.mm_to_voxels(phantom.pathology.motion_mm, axis=2)
    rng = np.random.default_rng(phantom.seed + 991)
    # A swallow is a step, not a drift: the stack is registered, then it is
    # not. Superimposing per-slice jitter on that step reproduces both halves
    # of what the QC stage looks for.
    step_at = volume.shape[0] // 2
    for index in range(volume.shape[0]):
        drift = (amplitude if index >= step_at else 0.0) + rng.normal(0, amplitude / 3)
        offset = round(float(drift))
        shifted[index] = np.roll(volume[index], offset, axis=1)
    return shifted


# ---------------------------------------------------------------------------
# Stamping primitives
# ---------------------------------------------------------------------------
def _grids(
    phantom: Phantom,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    return np.meshgrid(
        np.arange(phantom.depth, dtype=np.float32),
        np.arange(phantom.height, dtype=np.float32),
        np.arange(phantom.width, dtype=np.float32),
        indexing="ij",
    )


def _stamp_sphere(
    volume: npt.NDArray[np.float32],
    phantom: Phantom,
    centre: tuple[float, float, float],
    radii: tuple[float, float, float],
    value: float,
) -> None:
    """Write an ellipsoid, working on a local window rather than the volume.

    A full-volume mesh grid per stamp would be 7 M floats each time, and the
    phantom places a few hundred stamps.
    """
    bounds = []
    for axis, (centre_value, radius) in enumerate(zip(centre, radii, strict=True)):
        limit = volume.shape[axis]
        low = max(0, int(np.floor(centre_value - radius)) - 1)
        high = min(limit, int(np.ceil(centre_value + radius)) + 2)
        if high <= low:
            return
        bounds.append((low, high))

    (z0, z1), (y0, y1), (x0, x1) = bounds
    local_z, local_y, local_x = np.meshgrid(
        np.arange(z0, z1, dtype=np.float32),
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )
    inside = (
        ((local_z - centre[0]) / max(radii[0], 1e-3)) ** 2
        + ((local_y - centre[1]) / max(radii[1], 1e-3)) ** 2
        + ((local_x - centre[2]) / max(radii[2], 1e-3)) ** 2
    ) <= 1.0

    window = volume[z0:z1, y0:y1, x0:x1]
    window[inside] = value
    _ = phantom  # geometry already resolved by the caller


def _stamp_column(
    volume: npt.NDArray[np.float32],
    phantom: Phantom,
    x: float,
    y: float,
    z_low: float,
    z_high: float,
    radius: float,
    value: float,
) -> None:
    """Write a vertical cylinder of circular cross-section."""
    z0 = max(0, int(np.floor(z_low)))
    z1 = min(volume.shape[0], int(np.ceil(z_high)) + 1)
    if z1 <= z0:
        return

    y0 = max(0, int(np.floor(y - radius)) - 1)
    y1 = min(volume.shape[1], int(np.ceil(y + radius)) + 2)
    x0 = max(0, int(np.floor(x - radius)) - 1)
    x1 = min(volume.shape[2], int(np.ceil(x + radius)) + 2)
    if y1 <= y0 or x1 <= x0:
        return

    local_y, local_x = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )
    inside = ((local_y - y) ** 2 + (local_x - x) ** 2) <= radius**2

    window = volume[z0:z1, y0:y1, x0:x1]
    window[:, inside] = value
    _ = phantom


# ---------------------------------------------------------------------------
# NIfTI output
# ---------------------------------------------------------------------------
def to_nifti(volume: VolumeArray, spacing: tuple[float, float, float]) -> bytes:
    """Serialise as single-file NIfTI-1 (``n+1``), little-endian int16."""
    depth, height, width = volume.shape
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, width, height, depth, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 4)  # DT_INT16
    struct.pack_into("<h", header, 72, 16)  # bitpix
    struct.pack_into("<8f", header, 76, 1.0, spacing[0], spacing[1], spacing[2], 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", header, 108, 352.0)  # vox_offset
    struct.pack_into("<f", header, 112, 1.0)  # scl_slope
    struct.pack_into("<f", header, 116, 0.0)  # scl_inter
    header[344:348] = b"n+1\x00"
    # Four bytes of extension flags sit between the header and the voxels,
    # which is what makes vox_offset 352 rather than 348.
    return bytes(header) + b"\x00\x00\x00\x00" + volume.astype("<i2").tobytes()


#: Ready-made cases, used by the seed script so the demo clinic has a spread
#: of findings rather than five copies of one scan.
PRESETS: Final[dict[str, Pathology]] = {
    "healthy": Pathology(),
    "periapical": Pathology(apical_lesions=(36, 46), root_fillings=(36,)),
    "implant-site": Pathology(missing_teeth=(36, 37), bone_loss_mm=1.4),
    "restored": Pathology(implants=(46,), root_fillings=(16, 26), apical_lesions=(16,)),
    "cyst": Pathology(cyst_radius_mm=6.5, impacted_third_molar=True),
    "periodontal": Pathology(bone_loss_mm=4.2, missing_teeth=(18, 28, 38, 48)),
    "poor-quality": Pathology(motion_mm=3.5, extra_noise=45.0, apical_lesions=(46,)),
}


def build_preset(name: str, *, seed: int = 0, shape: tuple[int, int, int] | None = None) -> bytes:
    """Render a named preset straight to NIfTI bytes."""
    phantom = Phantom(seed=seed, pathology=PRESETS[name])
    if shape is not None:
        phantom.shape = shape
    return to_nifti(build(phantom), phantom.spacing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to write the .nii file")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="periapical")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = build_preset(args.preset, seed=args.seed)
    args.output.write_bytes(payload)
    print(f"→ {args.output} ({len(payload) / 1_048_576:.1f} MB, preset {args.preset})")


if __name__ == "__main__":
    main()
