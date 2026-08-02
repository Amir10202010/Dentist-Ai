"""Volumetric decoding: DICOM, NIfTI and the canonical container.

The parsers are hand-written against the formats' own specifications, so these
tests build files byte by byte rather than round-tripping through a library
that would share the parser's assumptions.
"""

from __future__ import annotations

import io
import struct
import zipfile

import numpy as np
import pytest

from dentist_ai.core.errors import UnsupportedMediaTypeError
from dentist_ai.db.models import VolumeFormat
from dentist_ai.services import volume as codec

# ---------------------------------------------------------------------------
# Fixtures: files written to spec
# ---------------------------------------------------------------------------
_IMPLICIT_VR_LE = "1.2.840.10008.1.2"
_EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
_JPEG_BASELINE = "1.2.840.10008.1.2.4.50"


def _ds(text: str) -> bytes:
    """DICOM string values are even-length, padded with a space."""
    raw = text.encode("ascii")
    return raw + (b" " if len(raw) % 2 else b"")


def _explicit(group: int, element: int, vr: bytes, payload: bytes) -> bytes:
    if vr in (b"OB", b"OW", b"SQ", b"UN", b"UT"):
        return struct.pack("<HH2sHI", group, element, vr, 0, len(payload)) + payload
    return struct.pack("<HH2sH", group, element, vr, len(payload)) + payload


def _implicit(group: int, element: int, payload: bytes) -> bytes:
    return struct.pack("<HHI", group, element, len(payload)) + payload


def dicom_slice(
    index: int,
    *,
    rows: int = 32,
    columns: int = 32,
    transfer_syntax: str = _EXPLICIT_VR_LE,
    series: str = "1.2.3.4",
    fill: int | None = None,
) -> bytes:
    """One CT instance, positioned along z at 0.5 mm intervals."""
    pixels = (
        np.full((rows, columns), fill, dtype=np.uint16)
        if fill is not None
        else (np.random.default_rng(index).random((rows, columns)) * 800).astype(np.uint16)
    )

    def write(group: int, element: int, vr: bytes, payload: bytes) -> bytes:
        """Encode one element in whichever VR mode the syntax specifies."""
        if transfer_syntax == _IMPLICIT_VR_LE:
            return _implicit(group, element, payload)
        return _explicit(group, element, vr, payload)

    body = b"".join(
        [
            write(0x0008, 0x0018, b"UI", _ds(f"1.2.3.4.{index}")),
            write(0x0018, 0x0050, b"DS", _ds("0.5")),
            write(0x0020, 0x000E, b"UI", _ds(series)),
            write(0x0020, 0x0013, b"IS", _ds(str(index))),
            write(0x0020, 0x0032, b"DS", _ds(f"0.0\\0.0\\{index * 0.5}")),
            write(0x0020, 0x0037, b"DS", _ds("1\\0\\0\\0\\1\\0")),
            write(0x0028, 0x0002, b"US", struct.pack("<H", 1)),
            write(0x0028, 0x0010, b"US", struct.pack("<H", rows)),
            write(0x0028, 0x0011, b"US", struct.pack("<H", columns)),
            write(0x0028, 0x0030, b"DS", _ds("0.4\\0.4")),
            write(0x0028, 0x0100, b"US", struct.pack("<H", 16)),
            write(0x0028, 0x0103, b"US", struct.pack("<H", 0)),
            write(0x0028, 0x1052, b"DS", _ds("-1024")),
            write(0x0028, 0x1053, b"DS", _ds("1")),
            write(0x7FE0, 0x0010, b"OW", pixels.tobytes()),
        ]
    )

    meta_payload = _explicit(0x0002, 0x0010, b"UI", _ds(transfer_syntax))
    meta = _explicit(0x0002, 0x0000, b"UL", struct.pack("<I", len(meta_payload))) + meta_payload
    return b"\x00" * 128 + b"DICM" + meta + body


def dicom_zip(count: int = 12, *, transfer_syntax: str = _EXPLICIT_VR_LE) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        # A real export also carries files that are not images.
        archive.writestr("DICOMDIR", b"not an image")
        archive.writestr("readme.txt", b"viewer instructions")
        for index in range(count):
            archive.writestr(
                f"series/IM{index:04d}.dcm",
                dicom_slice(index, transfer_syntax=transfer_syntax),
            )
    return buffer.getvalue()


def nifti(
    *,
    depth: int = 16,
    height: int = 24,
    width: int = 32,
    spacing: tuple[float, float, float] = (0.3, 0.3, 0.6),
    datatype: int = 4,
    big_endian: bool = False,
) -> bytes:
    order = ">" if big_endian else "<"
    header = bytearray(348)
    struct.pack_into(f"{order}i", header, 0, 348)
    struct.pack_into(f"{order}8h", header, 40, 3, width, height, depth, 1, 1, 1, 1)
    struct.pack_into(f"{order}h", header, 70, datatype)
    struct.pack_into(f"{order}h", header, 72, 16)
    struct.pack_into(f"{order}8f", header, 76, 1.0, *spacing, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into(f"{order}f", header, 108, 352.0)
    struct.pack_into(f"{order}f", header, 112, 1.0)
    struct.pack_into(f"{order}f", header, 116, 0.0)
    header[344:348] = b"n+1\x00"

    voxels = (np.random.default_rng(3).random((depth, height, width)) * 1200).astype(
        np.dtype(f"{order}i2")
    )
    return bytes(header) + b"\x00\x00\x00\x00" + voxels.tobytes()


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------
def test_nifti_round_trip_preserves_geometry() -> None:
    geometry = codec.parse(nifti())

    assert geometry.source_format is VolumeFormat.NIFTI
    assert geometry.shape == (16, 24, 32)
    assert geometry.spacing == pytest.approx((0.3, 0.3, 0.6), abs=1e-4)
    assert geometry.physical_size == pytest.approx((9.6, 7.2, 9.6), abs=1e-3)


def test_nifti_big_endian_is_detected_from_the_header() -> None:
    """``sizeof_hdr`` being byte-swapped is how the format states its order."""
    geometry = codec.parse(nifti(big_endian=True))
    assert geometry.shape == (16, 24, 32)


def test_nifti_rejects_a_two_dimensional_image() -> None:
    payload = bytearray(nifti())
    struct.pack_into("<h", payload, 40, 2)
    with pytest.raises(UnsupportedMediaTypeError, match="двумерный"):
        codec.parse(bytes(payload))


def test_nifti_rejects_an_unsupported_datatype() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="Тип данных"):
        codec.parse(nifti(datatype=1792))


def test_nifti_rejects_truncated_voxels() -> None:
    payload = nifti()
    with pytest.raises(UnsupportedMediaTypeError, match="обрываются"):
        codec.parse(payload[: len(payload) - 400])


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------
def test_dicom_series_in_a_zip_assembles_into_a_volume() -> None:
    geometry = codec.parse(dicom_zip(count=12))

    assert geometry.source_format is VolumeFormat.DICOM
    assert geometry.shape == (12, 32, 32)
    assert geometry.source_slice_count == 12
    # Pixel spacing is written rows-then-columns; the geometry reports x, y, z.
    assert geometry.spacing == pytest.approx((0.4, 0.4, 0.5), abs=1e-4)


def test_dicom_implicit_vr_is_decoded_from_the_transfer_syntax() -> None:
    """Implicit VR carries no VR on the wire; the tag table is the authority."""
    geometry = codec.parse(dicom_zip(count=8, transfer_syntax=_IMPLICIT_VR_LE))
    assert geometry.shape == (8, 32, 32)


def test_dicom_slice_order_follows_geometry_not_filename() -> None:
    """Position along the slice normal is the physical truth.

    The archive is written with instance numbers that run *opposite* to the
    positions, which is what a re-exported series looks like. Sorting by
    filename or instance number would invert the volume.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(8):
            payload = dicom_slice(index, fill=index * 90)
            # Reversed name so alphabetical order disagrees with geometry.
            archive.writestr(f"series/IM{7 - index:04d}.dcm", payload)

    geometry = codec.parse(buffer.getvalue())
    means = [float(geometry.voxels[z].mean()) for z in range(geometry.shape[0])]
    assert means == sorted(means), "slices should ascend along the acquisition axis"


def test_dicom_picks_the_largest_series_in_the_archive() -> None:
    """A scanner export ships the reconstruction beside a scout view."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(10):
            archive.writestr(f"recon/{index}.dcm", dicom_slice(index, series="1.1.1"))
        for index in range(4):
            archive.writestr(f"scout/{index}.dcm", dicom_slice(index, series="2.2.2"))

    geometry = codec.parse(buffer.getvalue())
    assert geometry.shape[0] == 10


def test_dicom_rejects_a_compressed_transfer_syntax_by_name() -> None:
    """A black volume would be worse than a refusal that says why."""
    with pytest.raises(UnsupportedMediaTypeError, match="сжатием"):
        codec.parse(dicom_zip(count=6, transfer_syntax=_JPEG_BASELINE))


def test_a_handful_of_slices_is_rejected_as_a_radiograph() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="рентгенограмма"):
        codec.parse(dicom_zip(count=3))


def test_archive_without_readable_instances_is_rejected() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", b"x" * 2048)
    with pytest.raises(UnsupportedMediaTypeError, match="читаемых"):
        codec.parse(buffer.getvalue())


# ---------------------------------------------------------------------------
# Canonical container
# ---------------------------------------------------------------------------
def test_canonical_round_trip_is_lossless_for_voxels_and_geometry() -> None:
    geometry = codec.parse(nifti())
    payload = codec.encode_canonical(geometry)

    header = codec.decode_header(payload)
    assert (header.depth, header.height, header.width) == geometry.shape
    assert header.spacing == pytest.approx(geometry.spacing, abs=1e-5)
    assert header.hu_slope == pytest.approx(geometry.hu_slope, rel=1e-5)

    voxels = codec.decode_voxels(payload, header)
    assert np.array_equal(voxels, geometry.voxels)


def test_canonical_header_is_fixed_size() -> None:
    """The viewer reads voxels at a constant offset, so this cannot drift."""
    geometry = codec.parse(nifti())
    payload = codec.encode_canonical(geometry)
    depth, height, width = geometry.shape
    assert len(payload) == 64 + depth * height * width


def test_decode_header_rejects_foreign_bytes() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="каноническим"):
        codec.decode_header(b"not a volume" * 16)


def test_hounsfield_mapping_survives_quantisation() -> None:
    """An 8-bit sample still has to report a plausible Hounsfield value.

    The stored range is fitted to the volume's own percentiles, so the mapping
    is what makes a density readout meaningful after quantisation.
    """
    geometry = codec.parse(nifti())
    lowest = geometry.hu_intercept
    highest = 255 * geometry.hu_slope + geometry.hu_intercept
    assert lowest < highest
    # The phantom spans 0-1200 in modality units; the fitted range must cover
    # the bulk of it without collapsing.
    assert highest - lowest > 100


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------
def test_large_volumes_are_decimated_and_spacing_is_scaled_to_match() -> None:
    """Halving the grid must double the millimetres per voxel.

    If it did not, every measurement taken in the viewer would be wrong by the
    decimation factor — which is the one way this pipeline could produce a
    confidently incorrect number.
    """
    geometry = codec.parse(nifti(depth=64, height=64, width=64), max_dimension=32)

    assert max(geometry.shape) <= 32
    # Field of view is a physical fact and must not change.
    assert geometry.physical_size == pytest.approx((19.2, 19.2, 38.4), abs=0.6)


def test_decimation_is_skipped_when_the_volume_already_fits() -> None:
    geometry = codec.parse(nifti(depth=16, height=24, width=32), max_dimension=64)
    assert geometry.shape == (16, 24, 32)


def test_preview_planes_are_corrected_to_square_pixels() -> None:
    """An anisotropic scan must not produce a squashed preview."""
    geometry = codec.parse(nifti(depth=16, height=32, width=32, spacing=(0.5, 0.5, 1.0)))
    axial, coronal, sagittal = codec.middle_slices(geometry)

    # Axial is 16 × 16 mm and square; the others are 16 mm wide by 16 mm tall
    # (16 slices at 1 mm), so all three come out square despite the coronal and
    # sagittal planes having half as many rows as columns in voxels.
    for image in (axial, coronal, sagittal):
        assert abs(image.width - image.height) <= 1


def test_too_small_a_file_is_rejected_before_parsing() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="слишком мал"):
        codec.parse(b"tiny")
