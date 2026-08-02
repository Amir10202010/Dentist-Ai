"""Reading CBCT volumes and normalising them to one canonical format.

The 3D equivalent of :mod:`dentist_ai.services.mesh`, and it exists for the
same reason. Scanners export a CBCT study as a directory of a few hundred
DICOM files, sometimes as one multi-frame DICOM, sometimes as NIfTI from a
research pipeline — three encodings, two byte orders, signed and unsigned
pixels, and a rescale that turns stored integers into Hounsfield units.

All of it is decoded here and re-emitted as ``DVOL``: a fixed 64-byte header
followed by 8-bit voxels in slice-major order. The browser therefore needs one
parser instead of three, a file that claims to be a volume but is not gets
rejected at the door, and the content hash is a function of the voxels rather
than of whichever console wrote the study.

Two decisions inside that are worth stating.

**Eight bits, not sixteen.** A CBCT reconstruction carries 12 bits of real
signal, but the viewer's job is windowing, and a window maps a range onto 256
display levels no matter how many bits went in. Storing 8 halves the transfer
and lets WebGL sample the volume as ``R8`` with no conversion. The linear map
back to Hounsfield units is kept in the header, so a voxel readout still
reports HU rather than a display level.

**Decimated on ingest, not on read.** A 0.2 mm full-arch reconstruction is
600×600×600 voxels — 216 MB, more than a browser will hold as a 3D texture.
Ingest block-averages by an integer factor per axis until every axis fits the
configured ceiling, which is a low-pass filter followed by a resample rather
than nearest-neighbour throwing detail away.

Compressed transfer syntaxes (JPEG, JPEG 2000, RLE) are rejected with a
message that names the problem. Decoding them needs a codec this project does
not depend on, and silently producing a black volume would be worse.
"""

from __future__ import annotations

import gzip
import io
import struct
import zipfile
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from PIL import Image

from dentist_ai.core.errors import UnsupportedMediaTypeError
from dentist_ai.core.logging import get_logger
from dentist_ai.db.models import VolumeFormat

log = get_logger(__name__)

VoxelArray = npt.NDArray[np.uint8]
ScalarArray = npt.NDArray[np.float32]

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
#: Longest edge kept after decimation. 256³ is 16 MB of voxels, which uploads
#: to a WebGL2 3D texture on every GPU this product targets, including the
#: integrated ones in a clinic's reception PC.
DEFAULT_MAX_DIMENSION: Final[int] = 256
#: Hard ceiling on decoded voxels before decimation, as a decompression-bomb
#: guard: a 200-byte gzip header can otherwise claim a 40 GB volume.
MAX_DECODED_VOXELS: Final[int] = 1_400_000_000
#: A CBCT series is a few hundred slices. Ten thousand means a zip of an
#: entire PACS export, not one study.
MAX_SERIES_SLICES: Final[int] = 4_096
MAX_ZIP_MEMBERS: Final[int] = 20_000
#: Total uncompressed bytes admitted from an archive.
MAX_ZIP_BYTES: Final[int] = 2 * 1024 * 1024 * 1024
#: Below this there is no volume, whatever the extension says.
MIN_VOLUME_BYTES: Final[int] = 512
#: Tolerance for float comparisons on intensities and spacings.
_EPSILON: Final[float] = 1e-6
#: A slice pitch below this is a rounding artefact, not a real spacing.
_MIN_SLICE_PITCH_MM: Final[float] = 1e-4
#: Shortest direction cosine pair that defines a usable slice normal.
_MIN_NORMAL_LENGTH: Final[float] = 1e-9
#: Fewer than this and it is a radiograph or a scout view, not a volume.
MIN_SLICES: Final[int] = 4

#: Percentiles that define the stored intensity range. The low tail is air,
#: which is most of the field of view in a CBCT, so clipping it at 0.5 rather
#: than at the true minimum keeps the useful range from collapsing into a few
#: display levels. The high tail is metal restorations and their streaks.
_WINDOW_LOW_PERCENTILE: Final[float] = 0.5
_WINDOW_HIGH_PERCENTILE: Final[float] = 99.5

# ---------------------------------------------------------------------------
# Canonical container
# ---------------------------------------------------------------------------
_MAGIC: Final[bytes] = b"DVOL0001"
_HEADER_SIZE: Final[int] = 64
_HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<8sIII fff ff ff B3x I")

_FORMAT_CODES: Final[dict[VolumeFormat, int]] = {
    VolumeFormat.DICOM: 1,
    VolumeFormat.NIFTI: 2,
}
_FORMAT_BY_CODE: Final[dict[int, VolumeFormat]] = {
    code: value for value, code in _FORMAT_CODES.items()
}


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """A decoded volume, ready to be written out.

    ``voxels`` is indexed ``[z, y, x]`` — slice, row, column — which is the
    order every source format stores and the order the canonical file keeps,
    so writing it is a ``tobytes()`` with no transpose.
    """

    voxels: VoxelArray
    #: Millimetres per voxel along x, y and z, after decimation.
    spacing: tuple[float, float, float]
    source_format: VolumeFormat
    #: ``modality_units = stored * hu_slope + hu_intercept``. Lets a voxel
    #: readout report Hounsfield units from an 8-bit sample.
    hu_slope: float
    hu_intercept: float
    #: Default window, in stored 0-255 units. Taken from the DICOM header when
    #: the console wrote one, otherwise the full stored range.
    window_center: float
    window_width: float
    #: Slices in the source series, before decimation. Reported so the study
    #: metadata reflects what the scanner produced, not what we kept.
    source_slice_count: int

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(depth, height, width)``."""
        depth, height, width = self.voxels.shape
        return int(depth), int(height), int(width)

    @property
    def physical_size(self) -> tuple[float, float, float]:
        """Field of view in millimetres, along x, y and z."""
        depth, height, width = self.shape
        return (
            width * self.spacing[0],
            height * self.spacing[1],
            depth * self.spacing[2],
        )


@dataclass(frozen=True, slots=True)
class VolumeHeader:
    """The canonical header, read back without touching the voxels."""

    width: int
    height: int
    depth: int
    spacing: tuple[float, float, float]
    hu_slope: float
    hu_intercept: float
    window_center: float
    window_width: float
    source_format: VolumeFormat


def encode_canonical(geometry: VolumeGeometry) -> bytes:
    """Serialise a volume as ``DVOL``: fixed header, then raw 8-bit voxels."""
    depth, height, width = geometry.shape
    header = _HEADER_STRUCT.pack(
        _MAGIC,
        width,
        height,
        depth,
        geometry.spacing[0],
        geometry.spacing[1],
        geometry.spacing[2],
        geometry.hu_slope,
        geometry.hu_intercept,
        geometry.window_center,
        geometry.window_width,
        _FORMAT_CODES[geometry.source_format],
        0,
    )
    return header.ljust(_HEADER_SIZE, b"\x00") + geometry.voxels.tobytes()


def decode_header(raw: bytes) -> VolumeHeader:
    """Read a canonical header. Raises if the bytes are not one."""
    if len(raw) < _HEADER_SIZE or not raw.startswith(_MAGIC):
        raise UnsupportedMediaTypeError("Файл не является каноническим томом.")
    (
        _,
        width,
        height,
        depth,
        spacing_x,
        spacing_y,
        spacing_z,
        slope,
        intercept,
        window_center,
        window_width,
        format_code,
        _flags,
    ) = _HEADER_STRUCT.unpack_from(raw, 0)
    return VolumeHeader(
        width=int(width),
        height=int(height),
        depth=int(depth),
        spacing=(float(spacing_x), float(spacing_y), float(spacing_z)),
        hu_slope=float(slope),
        hu_intercept=float(intercept),
        window_center=float(window_center),
        window_width=float(window_width),
        source_format=_FORMAT_BY_CODE.get(int(format_code), VolumeFormat.DICOM),
    )


def decode_voxels(raw: bytes, header: VolumeHeader) -> VoxelArray:
    """Read the voxels out of a canonical payload, as a ``[z, y, x]`` view.

    A view rather than a copy: the payload is up to 16 MB and the analysis only
    reads it, so ``frombuffer`` hands the pipeline the bytes it already has.
    """
    expected = header.depth * header.height * header.width
    available = len(raw) - _HEADER_SIZE
    if available < expected:
        raise UnsupportedMediaTypeError("Файл тома обрывается на середине.")
    return np.frombuffer(raw, dtype=np.uint8, count=expected, offset=_HEADER_SIZE).reshape(
        header.depth, header.height, header.width
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse(raw: bytes, *, max_dimension: int = DEFAULT_MAX_DIMENSION) -> VolumeGeometry:
    """Decode a CBCT upload, rejecting anything that is not a volume."""
    if len(raw) < MIN_VOLUME_BYTES:
        raise UnsupportedMediaTypeError("Файл слишком мал для объёмного снимка.")

    if raw[:2] == b"PK":
        slices = _read_zip(raw)
        return _finish(_assemble_dicom(slices), max_dimension=max_dimension)

    if raw[:2] == b"\x1f\x8b":
        return _finish(_parse_nifti(_gunzip(raw)), max_dimension=max_dimension)

    if _looks_like_nifti(raw):
        return _finish(_parse_nifti(raw), max_dimension=max_dimension)

    if _looks_like_dicom(raw):
        return _finish(_assemble_dicom([_parse_dicom(raw)]), max_dimension=max_dimension)

    raise UnsupportedMediaTypeError(
        "Не удалось распознать объёмный снимок. Поддерживаются ZIP с DICOM-серией, "
        "многокадровый DICOM и NIfTI (.nii, .nii.gz)."
    )


def middle_slices(geometry: VolumeGeometry) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Mid axial, coronal and sagittal planes, windowed for a preview.

    Each is returned at its true anisotropic aspect corrected to square pixels,
    so a thumbnail of a 0.3 × 0.3 × 0.6 mm volume is not vertically squashed.
    """
    depth, height, width = geometry.shape
    spacing_x, spacing_y, spacing_z = geometry.spacing

    axial = geometry.voxels[depth // 2, :, :]
    coronal = geometry.voxels[:, height // 2, :]
    sagittal = geometry.voxels[:, :, width // 2]

    return (
        _to_preview(axial, width * spacing_x, height * spacing_y),
        _to_preview(coronal, width * spacing_x, depth * spacing_z),
        _to_preview(sagittal, height * spacing_y, depth * spacing_z),
    )


def _to_preview(plane: VoxelArray, physical_width: float, physical_height: float) -> Image.Image:
    image = Image.fromarray(np.ascontiguousarray(plane), mode="L")
    if physical_width <= 0 or physical_height <= 0:
        return image
    # Coronal and sagittal planes are indexed by slice along one axis, so their
    # pixels are as tall as the slice pitch. Rescaling to the physical aspect
    # is what makes a preview measurable by eye.
    longest = max(image.width, image.height)
    scale = longest / max(physical_width, physical_height)
    target = (
        max(1, round(physical_width * scale)),
        max(1, round(physical_height * scale)),
    )
    return image.resize(target, Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _RawVolume:
    """Decoded modality-unit samples, before decimation and quantisation."""

    samples: ScalarArray
    spacing: tuple[float, float, float]
    source_format: VolumeFormat
    #: Console-authored window in modality units, when the file carried one.
    window: tuple[float, float] | None


def _finish(volume: _RawVolume, *, max_dimension: int) -> VolumeGeometry:
    source_slice_count = int(volume.samples.shape[0])
    samples, spacing = _decimate(volume.samples, volume.spacing, max_dimension)

    low, high = _intensity_range(samples)
    slope = (high - low) / 255.0
    stored = np.clip((samples - low) / slope, 0.0, 255.0).astype(np.uint8)

    if volume.window is not None:
        center = (volume.window[0] - low) / slope
        width = max(volume.window[1] / slope, 1.0)
    else:
        center, width = 127.5, 255.0

    return VolumeGeometry(
        voxels=np.ascontiguousarray(stored),
        spacing=spacing,
        source_format=volume.source_format,
        hu_slope=slope,
        hu_intercept=low,
        window_center=float(np.clip(center, 0.0, 255.0)),
        window_width=float(np.clip(width, 1.0, 512.0)),
        source_slice_count=source_slice_count,
    )


def _intensity_range(samples: ScalarArray) -> tuple[float, float]:
    """Robust low and high bounds for the stored range."""
    # Percentiles over a 16 M-voxel array cost more than the accuracy is worth;
    # a strided sample of ~2 M voxels lands within a fraction of a percent.
    flat = samples.reshape(-1)
    step = max(1, flat.size // 2_000_000)
    sampled = flat[::step]
    low = float(np.percentile(sampled, _WINDOW_LOW_PERCENTILE))
    high = float(np.percentile(sampled, _WINDOW_HIGH_PERCENTILE))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < _EPSILON:
        # A flat or degenerate reconstruction: fall back to true extrema so
        # the division below cannot produce infinities.
        low = float(np.min(samples))
        high = float(np.max(samples))
    if high - low < _EPSILON:
        high = low + 1.0
    return low, high


def _decimate(
    samples: ScalarArray,
    spacing: tuple[float, float, float],
    max_dimension: int,
) -> tuple[ScalarArray, tuple[float, float, float]]:
    """Block-average each axis by an integer factor until it fits the ceiling.

    Averaging rather than sampling: a CBCT is noisy, and dropping three voxels
    in four aliases that noise into the thing a clinician is looking for.
    """
    depth, height, width = (int(value) for value in samples.shape)
    factor_z = max(1, -(-depth // max_dimension))
    factor_y = max(1, -(-height // max_dimension))
    factor_x = max(1, -(-width // max_dimension))

    if factor_x == 1 and factor_y == 1 and factor_z == 1:
        return samples, spacing

    # Crop the remainder rather than padding: a partial edge block would be
    # averaged against zeros and show up as a dark rim.
    cropped = samples[
        : (depth // factor_z) * factor_z,
        : (height // factor_y) * factor_y,
        : (width // factor_x) * factor_x,
    ]
    reduced = cropped.reshape(
        cropped.shape[0] // factor_z,
        factor_z,
        cropped.shape[1] // factor_y,
        factor_y,
        cropped.shape[2] // factor_x,
        factor_x,
    ).mean(axis=(1, 3, 5), dtype=np.float32)

    return (
        np.ascontiguousarray(reduced, dtype=np.float32),
        (
            spacing[0] * factor_x,
            spacing[1] * factor_y,
            spacing[2] * factor_z,
        ),
    )


# ---------------------------------------------------------------------------
# Archives
# ---------------------------------------------------------------------------
def _gunzip(raw: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
            return handle.read(MAX_ZIP_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise UnsupportedMediaTypeError("Не удалось распаковать .gz-файл.") from exc


def _read_zip(raw: bytes) -> list[_DicomSlice]:
    """Extract every DICOM image in an archive, ignoring the rest.

    A scanner export routinely carries DICOMDIR, viewer executables and a
    readme alongside the series; anything that does not parse as an image is
    skipped rather than failing the upload.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise UnsupportedMediaTypeError("Архив повреждён.") from exc

    with archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) > MAX_ZIP_MEMBERS:
            raise UnsupportedMediaTypeError(
                f"В архиве {len(members):,} файлов при пределе {MAX_ZIP_MEMBERS:,}."
            )
        if sum(item.file_size for item in members) > MAX_ZIP_BYTES:
            raise UnsupportedMediaTypeError("Распакованный архив слишком велик.")

        slices: list[_DicomSlice] = []
        #: Why the first unreadable instance failed. Kept because the common
        #: real-world case is an archive where *every* instance is rejected for
        #: the same reason — a compressed transfer syntax — and reporting only
        #: "no readable slices" would hide the one fact that tells the clinic
        #: how to fix its export.
        first_failure: UnsupportedMediaTypeError | None = None

        for member in sorted(members, key=lambda item: item.filename):
            if member.file_size < MIN_VOLUME_BYTES:
                continue
            payload = archive.read(member)
            if not _looks_like_dicom(payload):
                continue
            try:
                slices.append(_parse_dicom(payload))
            except UnsupportedMediaTypeError as exc:
                # One unreadable instance must not sink a 400-slice series.
                if first_failure is None:
                    first_failure = exc
                continue
            if len(slices) > MAX_SERIES_SLICES:
                raise UnsupportedMediaTypeError(
                    f"В серии больше {MAX_SERIES_SLICES:,} срезов. Загрузите одно исследование."
                )

    if not slices:
        if first_failure is not None:
            raise first_failure
        raise UnsupportedMediaTypeError("В архиве нет читаемых DICOM-срезов.")
    return slices


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------
_DICM_MAGIC: Final[bytes] = b"DICM"
_PREAMBLE: Final[int] = 128

_TAG_TRANSFER_SYNTAX: Final[tuple[int, int]] = (0x0002, 0x0010)
_TAG_SERIES_UID: Final[tuple[int, int]] = (0x0020, 0x000E)
_TAG_INSTANCE_NUMBER: Final[tuple[int, int]] = (0x0020, 0x0013)
_TAG_POSITION: Final[tuple[int, int]] = (0x0020, 0x0032)
_TAG_ORIENTATION: Final[tuple[int, int]] = (0x0020, 0x0037)
_TAG_SAMPLES_PER_PIXEL: Final[tuple[int, int]] = (0x0028, 0x0002)
_TAG_NUMBER_OF_FRAMES: Final[tuple[int, int]] = (0x0028, 0x0008)
_TAG_ROWS: Final[tuple[int, int]] = (0x0028, 0x0010)
_TAG_COLUMNS: Final[tuple[int, int]] = (0x0028, 0x0011)
_TAG_PIXEL_SPACING: Final[tuple[int, int]] = (0x0028, 0x0030)
_TAG_BITS_ALLOCATED: Final[tuple[int, int]] = (0x0028, 0x0100)
_TAG_PIXEL_REPRESENTATION: Final[tuple[int, int]] = (0x0028, 0x0103)
_TAG_WINDOW_CENTER: Final[tuple[int, int]] = (0x0028, 0x1050)
_TAG_WINDOW_WIDTH: Final[tuple[int, int]] = (0x0028, 0x1051)
_TAG_RESCALE_INTERCEPT: Final[tuple[int, int]] = (0x0028, 0x1052)
_TAG_RESCALE_SLOPE: Final[tuple[int, int]] = (0x0028, 0x1053)
_TAG_SLICE_THICKNESS: Final[tuple[int, int]] = (0x0018, 0x0050)
_TAG_SPACING_BETWEEN_SLICES: Final[tuple[int, int]] = (0x0018, 0x0088)
_TAG_PIXEL_DATA: Final[tuple[int, int]] = (0x7FE0, 0x0010)

#: Group of the item and delimiter tags, which are always encoded implicitly
#: even inside an explicit-VR stream.
_DELIMITER_GROUP: Final[int] = 0xFFFE
#: Group of the file-meta elements, which precede the dataset proper.
_META_GROUP: Final[int] = 0x0002
#: Smallest element header: a tag plus a 32-bit length.
_MIN_ELEMENT_HEADER: Final[int] = 8
#: Value counts for the multi-valued elements the parser reads.
_POSITION_VALUES: Final[int] = 3
_ORIENTATION_VALUES: Final[int] = 6
_PIXEL_SPACING_VALUES: Final[int] = 2
#: Width of a binary US value.
_US_BYTES: Final[int] = 2

_ITEM: Final[tuple[int, int]] = (0xFFFE, 0xE000)
_ITEM_DELIMITER: Final[tuple[int, int]] = (0xFFFE, 0xE00D)
_SEQUENCE_DELIMITER: Final[tuple[int, int]] = (0xFFFE, 0xE0DD)
_UNDEFINED_LENGTH: Final[int] = 0xFFFFFFFF

#: The value representation of every tag we read, so the parser never needs a
#: full data dictionary: implicit-VR datasets carry no VR on the wire, and for
#: our tags this table is the authority in both encodings.
_TAG_VR: Final[dict[tuple[int, int], str]] = {
    _TAG_TRANSFER_SYNTAX: "UI",
    _TAG_SERIES_UID: "UI",
    _TAG_INSTANCE_NUMBER: "IS",
    _TAG_POSITION: "DS",
    _TAG_ORIENTATION: "DS",
    _TAG_SAMPLES_PER_PIXEL: "US",
    _TAG_NUMBER_OF_FRAMES: "IS",
    _TAG_ROWS: "US",
    _TAG_COLUMNS: "US",
    _TAG_PIXEL_SPACING: "DS",
    _TAG_BITS_ALLOCATED: "US",
    _TAG_PIXEL_REPRESENTATION: "US",
    _TAG_WINDOW_CENTER: "DS",
    _TAG_WINDOW_WIDTH: "DS",
    _TAG_RESCALE_INTERCEPT: "DS",
    _TAG_RESCALE_SLOPE: "DS",
    _TAG_SLICE_THICKNESS: "DS",
    _TAG_SPACING_BETWEEN_SLICES: "DS",
    _TAG_PIXEL_DATA: "OW",
}

_VALUE_REPRESENTATIONS: Final[frozenset[str]] = frozenset(
    {
        "AE",
        "AS",
        "AT",
        "CS",
        "DA",
        "DS",
        "DT",
        "FD",
        "FL",
        "IS",
        "LO",
        "LT",
        "OB",
        "OD",
        "OF",
        "OL",
        "OV",
        "OW",
        "PN",
        "SH",
        "SL",
        "SQ",
        "SS",
        "ST",
        "SV",
        "TM",
        "UC",
        "UI",
        "UL",
        "UN",
        "UR",
        "US",
        "UT",
        "UV",
    }
)
#: VRs whose explicit-VR header carries a 32-bit length after two reserved
#: bytes, rather than the usual 16-bit one.
_LONG_HEADER_VRS: Final[frozenset[str]] = frozenset(
    {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "SV", "UC", "UN", "UR", "UT", "UV"}
)

_IMPLICIT_VR_LE: Final[str] = "1.2.840.10008.1.2"
_EXPLICIT_VR_LE: Final[str] = "1.2.840.10008.1.2.1"
_DEFLATED_EXPLICIT_VR_LE: Final[str] = "1.2.840.10008.1.2.1.99"
_EXPLICIT_VR_BE: Final[str] = "1.2.840.10008.1.2.2"

_UNCOMPRESSED_SYNTAXES: Final[frozenset[str]] = frozenset(
    {_IMPLICIT_VR_LE, _EXPLICIT_VR_LE, _DEFLATED_EXPLICIT_VR_LE, _EXPLICIT_VR_BE}
)


@dataclass(frozen=True, slots=True)
class _DicomSlice:
    """One decoded instance: pixels in modality units plus what places them."""

    pixels: ScalarArray
    series_uid: str
    instance_number: int
    position: tuple[float, float, float] | None
    normal: tuple[float, float, float] | None
    pixel_spacing: tuple[float, float]
    slice_spacing: float | None
    window: tuple[float, float] | None

    @property
    def frames(self) -> int:
        return int(self.pixels.shape[0])


def _looks_like_dicom(raw: bytes) -> bool:
    if raw[_PREAMBLE : _PREAMBLE + 4] == _DICM_MAGIC:
        return True
    # Files written without the Part 10 preamble start straight at the first
    # element, which for any image is group 0x0008.
    return len(raw) >= _MIN_ELEMENT_HEADER and struct.unpack_from("<H", raw, 0)[0] in (
        _META_GROUP,
        0x0008,
    )


def _parse_dicom(raw: bytes) -> _DicomSlice:
    offset = _PREAMBLE + 4 if raw[_PREAMBLE : _PREAMBLE + 4] == _DICM_MAGIC else 0

    transfer_syntax = _EXPLICIT_VR_LE
    if offset > 0 or struct.unpack_from("<H", raw, offset)[0] == _META_GROUP:
        meta = _collect(raw, offset, explicit=True, endian="<", stop_group=0x0002)
        transfer_syntax = _text(meta.get(_TAG_TRANSFER_SYNTAX, b"")) or _EXPLICIT_VR_LE
        offset = meta.end_offset

    if transfer_syntax not in _UNCOMPRESSED_SYNTAXES:
        raise UnsupportedMediaTypeError(
            "DICOM со сжатием изображения (JPEG, JPEG 2000, RLE) не поддерживается. "
            "Экспортируйте серию без компрессии или в формате NIfTI."
        )

    body = raw[offset:]
    if transfer_syntax == _DEFLATED_EXPLICIT_VR_LE:
        body = _inflate(body)
        offset = 0
    else:
        offset = 0

    explicit = transfer_syntax != _IMPLICIT_VR_LE
    endian = ">" if transfer_syntax == _EXPLICIT_VR_BE else "<"
    elements = _collect(body, offset, explicit=explicit, endian=endian)

    return _build_slice(elements, endian=endian)


def _inflate(body: bytes) -> bytes:
    import zlib  # noqa: PLC0415 - only deflated DICOM needs it

    try:
        return zlib.decompressobj(-zlib.MAX_WBITS).decompress(body, MAX_ZIP_BYTES)
    except zlib.error as exc:
        raise UnsupportedMediaTypeError("Не удалось распаковать сжатый DICOM.") from exc


class _Elements(dict[tuple[int, int], bytes]):
    """Top-level elements of one dataset, plus where parsing stopped."""

    end_offset: int = 0


def _collect(
    data: bytes,
    offset: int,
    *,
    explicit: bool,
    endian: str,
    stop_group: int | None = None,
) -> _Elements:
    """Walk a dataset, keeping the elements we need and skipping sequences.

    Only top-level elements are collected. Nested ones are skipped wholesale
    rather than merged, because a tag such as Rows appearing inside a
    Referenced Image Sequence describes some other image entirely.
    """
    elements = _Elements()
    while offset + _MIN_ELEMENT_HEADER <= len(data):
        tag, vr, length, header = _read_header(data, offset, explicit=explicit, endian=endian)
        if stop_group is not None and tag[0] != stop_group:
            break
        offset += header

        if vr == "SQ" or length == _UNDEFINED_LENGTH:
            if tag == _TAG_PIXEL_DATA:
                raise UnsupportedMediaTypeError(
                    "Пиксельные данные DICOM сжаты и не поддерживаются. "
                    "Экспортируйте серию без компрессии."
                )
            offset = (
                _skip_sequence(data, offset, explicit=explicit, endian=endian)
                if length == _UNDEFINED_LENGTH
                else offset + length
            )
            continue

        if tag in _TAG_VR:
            elements[tag] = data[offset : offset + length]
            if tag == _TAG_PIXEL_DATA:
                offset += length
                break

        offset += length

    elements.end_offset = offset
    return elements


def _read_header(
    data: bytes, offset: int, *, explicit: bool, endian: str
) -> tuple[tuple[int, int], str, int, int]:
    """Return ``(tag, vr, length, header_size)`` for the element at ``offset``.

    In an explicit-VR dataset the two bytes after the tag are the VR — unless
    the element is an item or a delimiter, which are always encoded implicitly
    even inside an explicit stream. Sniffing for a known VR distinguishes the
    two without a separate code path per nesting level.
    """
    group, element = struct.unpack_from(f"{endian}HH", data, offset)
    tag = (int(group), int(element))

    if group != _DELIMITER_GROUP and explicit:
        candidate = data[offset + 4 : offset + 6].decode("ascii", errors="replace")
        if candidate in _VALUE_REPRESENTATIONS:
            if candidate in _LONG_HEADER_VRS:
                if offset + 12 > len(data):
                    return tag, candidate, 0, len(data) - offset
                length = struct.unpack_from(f"{endian}I", data, offset + 8)[0]
                return tag, candidate, int(length), 12
            length = struct.unpack_from(f"{endian}H", data, offset + 6)[0]
            return tag, candidate, int(length), 8

    length = struct.unpack_from(f"{endian}I", data, offset + 4)[0]
    return tag, _TAG_VR.get(tag, ""), int(length), 8


def _skip_sequence(data: bytes, offset: int, *, explicit: bool, endian: str) -> int:
    """Advance past an undefined-length sequence, including nested ones."""
    depth = 1
    while offset + _MIN_ELEMENT_HEADER <= len(data) and depth > 0:
        tag, _vr, length, header = _read_header(data, offset, explicit=explicit, endian=endian)
        offset += header
        if tag in (_SEQUENCE_DELIMITER, _ITEM_DELIMITER):
            depth -= 1
        elif length == _UNDEFINED_LENGTH:
            # An undefined-length item or a nested sequence: its own delimiter
            # closes it, so descend rather than skip.
            depth += 1
        elif tag != _ITEM:
            offset += length
        else:
            offset += length
    return offset


def _build_slice(elements: _Elements, *, endian: str) -> _DicomSlice:
    rows = _first_int(elements.get(_TAG_ROWS), "US", endian)
    columns = _first_int(elements.get(_TAG_COLUMNS), "US", endian)
    if not rows or not columns:
        raise UnsupportedMediaTypeError("В DICOM-файле нет размеров изображения.")

    samples_per_pixel = _first_int(elements.get(_TAG_SAMPLES_PER_PIXEL), "US", endian) or 1
    if samples_per_pixel != 1:
        raise UnsupportedMediaTypeError("Цветные DICOM-изображения не являются томограммой.")

    bits = _first_int(elements.get(_TAG_BITS_ALLOCATED), "US", endian) or 16
    signed = bool(_first_int(elements.get(_TAG_PIXEL_REPRESENTATION), "US", endian))
    frames = _first_int(elements.get(_TAG_NUMBER_OF_FRAMES), "IS", endian) or 1

    payload = elements.get(_TAG_PIXEL_DATA)
    if not payload:
        raise UnsupportedMediaTypeError("В DICOM-файле нет пиксельных данных.")

    pixels = _decode_pixels(
        payload,
        rows=rows,
        columns=columns,
        frames=frames,
        bits=bits,
        signed=signed,
        endian=endian,
    )

    slope = _first_float(elements.get(_TAG_RESCALE_SLOPE), endian) or 1.0
    intercept = _first_float(elements.get(_TAG_RESCALE_INTERCEPT), endian) or 0.0
    modality = pixels.astype(np.float32) * np.float32(slope) + np.float32(intercept)

    spacing = _floats(elements.get(_TAG_PIXEL_SPACING), endian)
    # DICOM writes PixelSpacing as row spacing then column spacing — y before x.
    pixel_spacing = (
        (float(spacing[1]), float(spacing[0]))
        if len(spacing) >= _PIXEL_SPACING_VALUES
        else (1.0, 1.0)
    )

    slice_spacing = _first_float(elements.get(_TAG_SPACING_BETWEEN_SLICES), endian) or _first_float(
        elements.get(_TAG_SLICE_THICKNESS), endian
    )

    center = _first_float(elements.get(_TAG_WINDOW_CENTER), endian)
    width = _first_float(elements.get(_TAG_WINDOW_WIDTH), endian)

    position = _floats(elements.get(_TAG_POSITION), endian)
    orientation = _floats(elements.get(_TAG_ORIENTATION), endian)

    return _DicomSlice(
        pixels=modality,
        series_uid=_text(elements.get(_TAG_SERIES_UID, b"")),
        instance_number=_first_int(elements.get(_TAG_INSTANCE_NUMBER), "IS", endian) or 0,
        position=(
            (float(position[0]), float(position[1]), float(position[2]))
            if len(position) >= _POSITION_VALUES
            else None
        ),
        normal=_normal_from(orientation),
        pixel_spacing=pixel_spacing,
        slice_spacing=slice_spacing if slice_spacing and slice_spacing > 0 else None,
        window=(center, width) if center is not None and width and width > 0 else None,
    )


def _decode_pixels(
    payload: bytes,
    *,
    rows: int,
    columns: int,
    frames: int,
    bits: int,
    signed: bool,
    endian: str,
) -> npt.NDArray[np.int32]:
    if bits not in (8, 16):
        raise UnsupportedMediaTypeError(f"Глубина {bits} бит на воксель не поддерживается.")
    if rows * columns * frames > MAX_DECODED_VOXELS:
        raise UnsupportedMediaTypeError("Объём снимка превышает допустимый предел.")

    kind = "i" if signed else "u"
    dtype = np.dtype(f"{endian}{kind}{bits // 8}")
    count = rows * columns * frames
    if len(payload) < count * dtype.itemsize:
        raise UnsupportedMediaTypeError("Пиксельные данные DICOM обрываются на середине.")

    flat = np.frombuffer(payload, dtype=dtype, count=count)
    return flat.reshape(frames, rows, columns).astype(np.int32)


def _normal_from(orientation: list[float]) -> tuple[float, float, float] | None:
    """Slice normal as the cross product of the two in-plane direction cosines."""
    if len(orientation) < _ORIENTATION_VALUES:
        return None
    row = np.array(orientation[:3], dtype=np.float64)
    column = np.array(orientation[3:6], dtype=np.float64)
    normal = np.cross(row, column)
    length = float(np.linalg.norm(normal))
    if length < _MIN_NORMAL_LENGTH:
        return None
    normal = normal / length
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def _assemble_dicom(slices: list[_DicomSlice]) -> _RawVolume:
    """Order a series into a volume and derive its slice pitch."""
    if not slices:
        raise UnsupportedMediaTypeError("Не найдено ни одного DICOM-среза.")

    # A multi-frame instance already is the volume; a series is one frame per
    # file. Anything in between is a mixed export we decline to guess at.
    if len(slices) == 1 and slices[0].frames >= MIN_SLICES:
        single = slices[0]
        pitch = single.slice_spacing or single.pixel_spacing[1]
        return _RawVolume(
            samples=np.ascontiguousarray(single.pixels, dtype=np.float32),
            spacing=(single.pixel_spacing[0], single.pixel_spacing[1], pitch),
            source_format=VolumeFormat.DICOM,
            window=single.window,
        )

    series = _dominant_series(slices)
    if len(series) < MIN_SLICES:
        raise UnsupportedMediaTypeError(
            f"В серии {len(series)} срез(ов) — это рентгенограмма, а не объёмный снимок. "
            "Загрузите её в раздел «Снимки»."
        )

    ordered, pitch = _order_series(series)
    first = ordered[0]
    shapes = {item.pixels.shape[1:] for item in ordered}
    if len(shapes) != 1:
        raise UnsupportedMediaTypeError("Срезы серии имеют разный размер.")

    samples = np.concatenate([item.pixels for item in ordered], axis=0)
    return _RawVolume(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        spacing=(first.pixel_spacing[0], first.pixel_spacing[1], pitch),
        source_format=VolumeFormat.DICOM,
        window=first.window,
    )


def _dominant_series(slices: list[_DicomSlice]) -> list[_DicomSlice]:
    """Pick the largest series in the archive.

    A CBCT export often ships the reconstruction alongside a scout view and a
    handful of secondary captures. The reconstruction is always the one with
    the most instances.
    """
    grouped: dict[str, list[_DicomSlice]] = {}
    for item in slices:
        grouped.setdefault(item.series_uid, []).append(item)
    chosen = max(grouped.values(), key=len)
    if len(grouped) > 1:
        log.info("dicom_series_selected", series=len(grouped), slices=len(chosen))
    return chosen


def _order_series(series: list[_DicomSlice]) -> tuple[list[_DicomSlice], float]:
    """Sort slices along the acquisition axis and measure the pitch between them.

    Instance number is a fallback, not the primary key: a series reconstructed
    twice, or exported by a console that renumbers, can carry instance numbers
    that do not follow geometry. The position projected onto the slice normal
    is the physical truth.
    """
    normal = next((item.normal for item in series if item.normal is not None), (0.0, 0.0, 1.0))
    axis = np.array(normal, dtype=np.float64)

    positioned = [item for item in series if item.position is not None]
    if len(positioned) == len(series):
        offsets = [float(np.dot(np.array(item.position), axis)) for item in positioned]
        order = sorted(range(len(series)), key=lambda index: offsets[index])
        ordered = [series[index] for index in order]
        sorted_offsets = [offsets[index] for index in order]
        gaps = np.diff(sorted_offsets)
        pitch = float(np.median(np.abs(gaps))) if gaps.size else 0.0
        if pitch > _MIN_SLICE_PITCH_MM:
            return ordered, pitch

    ordered = sorted(series, key=lambda item: item.instance_number)
    fallback = next(
        (item.slice_spacing for item in ordered if item.slice_spacing is not None),
        None,
    )
    return ordered, fallback or ordered[0].pixel_spacing[1]


# ---------------------------------------------------------------------------
# DICOM value decoding
# ---------------------------------------------------------------------------
def _text(raw: bytes) -> str:
    return raw.decode("ascii", errors="replace").strip("\x00 \t\r\n")


def _floats(raw: bytes | None, endian: str) -> list[float]:
    """Decode a numeric element as a list, tolerating either encoding."""
    if not raw:
        return []
    text = _text(raw)
    if text and all(char in "0123456789+-.eE\\ " for char in text):
        values: list[float] = []
        for part in text.split("\\"):
            token = part.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                return []
        return values
    # Binary FL/FD, which some consoles use where the standard says DS.
    if len(raw) % 8 == 0:
        return [float(value) for value in np.frombuffer(raw, dtype=f"{endian}f8")]
    if len(raw) % 4 == 0:
        return [float(value) for value in np.frombuffer(raw, dtype=f"{endian}f4")]
    return []


def _first_float(raw: bytes | None, endian: str) -> float | None:
    values = _floats(raw, endian)
    return values[0] if values else None


def _first_int(raw: bytes | None, vr: str, endian: str) -> int | None:
    if not raw:
        return None
    if vr == "US":
        if len(raw) < _US_BYTES:
            return None
        return int(struct.unpack_from(f"{endian}H", raw, 0)[0])
    values = _floats(raw, endian)
    return int(values[0]) if values else None


# ---------------------------------------------------------------------------
# NIfTI-1
# ---------------------------------------------------------------------------
_NIFTI_HEADER_SIZE: Final[int] = 348
#: A NIfTI with fewer than three dimensions is an image, not a volume.
_NIFTI_MIN_DIMENSIONS: Final[int] = 3
#: Narrower than this in plane and the file is a profile, not a slice.
_MIN_IN_PLANE_VOXELS: Final[int] = 2
_NIFTI_MAGICS: Final[frozenset[bytes]] = frozenset({b"n+1\x00", b"ni1\x00"})

#: NIfTI datatype code -> numpy character code, for the types a reconstruction
#: is actually written in.
_NIFTI_TYPES: Final[dict[int, str]] = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    256: "i1",
    512: "u2",
    768: "u4",
}


def _looks_like_nifti(raw: bytes) -> bool:
    if len(raw) < _NIFTI_HEADER_SIZE:
        return False
    return raw[344:348] in _NIFTI_MAGICS


def _parse_nifti(raw: bytes) -> _RawVolume:
    if not _looks_like_nifti(raw):
        raise UnsupportedMediaTypeError("Файл не является NIfTI-изображением.")
    if raw[344:348] == b"ni1\x00":
        raise UnsupportedMediaTypeError(
            "NIfTI в двух файлах (.hdr/.img) не поддерживается. Сохраните как один .nii."
        )

    # sizeof_hdr is 348 in the file's own byte order; a swapped value is how
    # the format tells a reader it was written big-endian.
    endian = "<" if struct.unpack_from("<i", raw, 0)[0] == _NIFTI_HEADER_SIZE else ">"
    if struct.unpack_from(f"{endian}i", raw, 0)[0] != _NIFTI_HEADER_SIZE:
        raise UnsupportedMediaTypeError("Повреждённый заголовок NIfTI.")

    dims = struct.unpack_from(f"{endian}8h", raw, 40)
    datatype = struct.unpack_from(f"{endian}h", raw, 70)[0]
    pixdim = struct.unpack_from(f"{endian}8f", raw, 76)
    vox_offset = int(struct.unpack_from(f"{endian}f", raw, 108)[0])
    scl_slope = float(struct.unpack_from(f"{endian}f", raw, 112)[0])
    scl_inter = float(struct.unpack_from(f"{endian}f", raw, 116)[0])

    if dims[0] < _NIFTI_MIN_DIMENSIONS:
        raise UnsupportedMediaTypeError("NIfTI-файл двумерный, а не объёмный.")
    width, height, depth = (int(dims[1]), int(dims[2]), int(dims[3]))
    if min(width, height) < _MIN_IN_PLANE_VOXELS or depth < MIN_SLICES:
        raise UnsupportedMediaTypeError(f"В файле {depth} срез(ов) — это не объёмный снимок.")
    if width * height * depth > MAX_DECODED_VOXELS:
        raise UnsupportedMediaTypeError("Объём снимка превышает допустимый предел.")

    code = _NIFTI_TYPES.get(int(datatype))
    if code is None:
        raise UnsupportedMediaTypeError(f"Тип данных NIfTI {datatype} не поддерживается.")

    dtype = np.dtype(f"{endian}{code}")
    count = width * height * depth
    start = max(vox_offset, _NIFTI_HEADER_SIZE)
    if len(raw) < start + count * dtype.itemsize:
        raise UnsupportedMediaTypeError("Данные NIfTI обрываются на середине.")

    # NIfTI stores x fastest, so a C-order read with reversed dimensions
    # lands directly on the [z, y, x] layout the canonical format uses.
    voxels = np.frombuffer(raw, dtype=dtype, count=count, offset=start).reshape(
        depth, height, width
    )
    samples = voxels.astype(np.float32)
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        samples = samples * np.float32(scl_slope or 1.0) + np.float32(scl_inter)

    spacing = tuple(max(float(value), 1e-3) for value in pixdim[1:4])
    return _RawVolume(
        samples=np.ascontiguousarray(samples),
        spacing=(spacing[0], spacing[1], spacing[2]),
        source_format=VolumeFormat.NIFTI,
        window=None,
    )
