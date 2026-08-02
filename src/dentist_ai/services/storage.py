"""Content-addressed storage for patient files.

Both stores work the same way. The upload is decoded, re-encoded into one
canonical format, and written under a path derived from the SHA-256 of those
canonical bytes. The client's filename never reaches the filesystem, the
storage root is outside any mounted static directory, and re-encoding
destroys anything that was riding along in the uploaded file — EXIF on a
radiograph, an embedded script in a polyglot.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile

from dentist_ai.core.config import StorageSettings
from dentist_ai.core.errors import PayloadTooLargeError, UnsupportedMediaTypeError
from dentist_ai.core.logging import get_logger
from dentist_ai.db.models import MeshFormat, VolumeFormat
from dentist_ai.services import mesh, volume

log = get_logger(__name__)

_READ_CHUNK_BYTES = 512 * 1024
_ACCEPTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})
#: ~180 MP. Above this a "valid" PNG is almost certainly a decompression bomb.
_MAX_PIXELS = 180_000_000
#: Length of a hex-encoded SHA-256 digest.
_HASH_LENGTH = 64
#: Preview planes written alongside every stored volume, in the order
#: :func:`volume.middle_slices` returns them.
_PREVIEW_PLANES = ("axial", "coronal", "sagittal")


@dataclass(frozen=True, slots=True)
class StoredImage:
    content_hash: str
    content_type: str
    byte_size: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class StoredMesh:
    content_hash: str
    byte_size: int
    triangle_count: int
    source_format: MeshFormat
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


class ImageStorage:
    """Stores normalised radiographs plus derived thumbnails."""

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings
        self._root = settings.resolved_root
        self._root.mkdir(parents=True, exist_ok=True)
        # `0o700`: readable only by the service account, even if the
        # deployment misconfigures a reverse proxy to serve the data volume.
        self._root.chmod(0o700)

    # -- public API -------------------------------------------------------
    async def save_upload(self, upload: UploadFile) -> StoredImage:
        """Validate, normalise and store an uploaded file."""
        raw = await self._read_bounded(upload)
        return await self.store_bytes(raw)

    async def store_bytes(self, raw: bytes) -> StoredImage:
        """Store already-in-memory image bytes.

        Decoding and re-encoding are CPU-bound and would block the event loop,
        so they run in a worker thread.
        """
        return await asyncio.to_thread(self._normalise_and_write, raw)

    def master_path(self, content_hash: str) -> Path:
        return self._path_for(content_hash, suffix="master.jpg")

    def thumbnail_path(self, content_hash: str) -> Path:
        return self._path_for(content_hash, suffix="thumb.jpg")

    def delete(self, content_hash: str) -> None:
        """Remove both derivatives. Safe to call for a hash that is already gone."""
        for path in (self.master_path(content_hash), self.thumbnail_path(content_hash)):
            path.unlink(missing_ok=True)
        # Prune the fan-out directory if this was the last study in it.
        with contextlib.suppress(OSError):
            self.master_path(content_hash).parent.rmdir()

    # -- internals --------------------------------------------------------
    async def _read_bounded(self, upload: UploadFile) -> bytes:
        return await read_bounded(upload, self._settings.max_upload_bytes)

    def _normalise_and_write(self, raw: bytes) -> StoredImage:
        """Decode, orient, downscale, strip metadata, write. Runs off-loop."""
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
        try:
            image = self._decode(raw)
            with image:
                # Honour EXIF rotation *before* discarding EXIF, otherwise
                # phone-captured scans come out sideways.
                image = ImageOps.exif_transpose(image) or image
                image = image.convert("L").convert("RGB")
                image.thumbnail(
                    (self._settings.max_image_dimension, self._settings.max_image_dimension),
                    Image.Resampling.LANCZOS,
                )
                width, height = image.size

                master_bytes = _encode_jpeg(image, quality=92)
                content_hash = hashlib.sha256(master_bytes).hexdigest()

                master = self.master_path(content_hash)
                if not master.is_file():
                    master.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_write(master, master_bytes)

                    thumb = image.copy()
                    thumb.thumbnail(
                        (
                            self._settings.thumbnail_dimension,
                            self._settings.thumbnail_dimension,
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    _atomic_write(self.thumbnail_path(content_hash), _encode_jpeg(thumb, 78))
                    thumb.close()

                return StoredImage(
                    content_hash=content_hash,
                    content_type="image/jpeg",
                    byte_size=len(master_bytes),
                    width=width,
                    height=height,
                )
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit

    @staticmethod
    def _decode(raw: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(raw))
            image.verify()  # structural check on a throwaway handle
            image = Image.open(io.BytesIO(raw))  # verify() leaves it unusable
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise UnsupportedMediaTypeError(
                "Не удалось прочитать изображение. Поддерживаются JPEG, PNG, WebP, BMP, TIFF."
            ) from exc

        if image.format not in _ACCEPTED_FORMATS:
            image.close()
            raise UnsupportedMediaTypeError(
                f"Формат {image.format or 'неизвестный'} не поддерживается."
            )
        return image

    def _path_for(self, content_hash: str, *, suffix: str) -> Path:
        return fanned_path(self._root, content_hash, suffix)


class MeshStorage:
    """Stores 3D scans, normalised to binary STL."""

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings
        self._root = settings.resolved_root / "meshes"
        self._root.mkdir(parents=True, exist_ok=True)
        self._root.chmod(0o700)

    async def save_upload(self, upload: UploadFile) -> StoredMesh:
        raw = await read_bounded(upload, self._settings.max_mesh_bytes)
        return await self.store_bytes(raw)

    async def store_bytes(self, raw: bytes) -> StoredMesh:
        """Decode and store already-in-memory mesh bytes.

        Parsing a few million triangles is CPU-bound, so it runs off-loop.
        """
        return await asyncio.to_thread(self._normalise_and_write, raw)

    def path(self, content_hash: str) -> Path:
        return fanned_path(self._root, content_hash, "mesh.stl")

    def delete(self, content_hash: str) -> None:
        path = self.path(content_hash)
        path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            path.parent.rmdir()

    def _normalise_and_write(self, raw: bytes) -> StoredMesh:
        geometry = mesh.parse(raw)
        payload = mesh.encode_binary_stl(geometry.triangles)
        content_hash = hashlib.sha256(payload).hexdigest()

        target = self.path(content_hash)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, payload)

        low, high = geometry.bounds
        return StoredMesh(
            content_hash=content_hash,
            byte_size=len(payload),
            triangle_count=geometry.triangle_count,
            source_format=geometry.source_format,
            bounds_min=low,
            bounds_max=high,
        )


@dataclass(frozen=True, slots=True)
class StoredVolume:
    content_hash: str
    byte_size: int
    width: int
    height: int
    depth: int
    spacing: tuple[float, float, float]
    hu_slope: float
    hu_intercept: float
    window_center: float
    window_width: float
    source_format: VolumeFormat
    source_slice_count: int


class VolumeStorage:
    """Stores CBCT studies, normalised to the canonical ``DVOL`` container.

    Alongside the voxels it writes three JPEG previews — mid axial, coronal
    and sagittal — so a list of studies can show what is in each one without
    every row pulling a 16 MB volume.
    """

    def __init__(self, settings: StorageSettings) -> None:
        self._settings = settings
        self._root = settings.resolved_root / "volumes"
        self._root.mkdir(parents=True, exist_ok=True)
        self._root.chmod(0o700)

    async def save_upload(self, upload: UploadFile) -> StoredVolume:
        raw = await read_bounded(upload, self._settings.max_volume_bytes)
        return await self.store_bytes(raw)

    async def store_bytes(self, raw: bytes) -> StoredVolume:
        """Decode, decimate and store a CBCT study.

        Decoding a few hundred DICOM slices and block-averaging 200 MB of
        voxels is seconds of pure CPU, so it runs off-loop.
        """
        return await asyncio.to_thread(self._normalise_and_write, raw)

    def path(self, content_hash: str) -> Path:
        return fanned_path(self._root, content_hash, "volume.dvol")

    def preview_path(self, content_hash: str, plane: str) -> Path:
        return fanned_path(self._root, content_hash, f"{plane}.jpg")

    def delete(self, content_hash: str) -> None:
        paths = [self.path(content_hash)]
        paths.extend(self.preview_path(content_hash, plane) for plane in _PREVIEW_PLANES)
        for path in paths:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            paths[0].parent.rmdir()

    def _normalise_and_write(self, raw: bytes) -> StoredVolume:
        geometry = volume.parse(raw, max_dimension=self._settings.max_volume_dimension)
        payload = volume.encode_canonical(geometry)
        content_hash = hashlib.sha256(payload).hexdigest()

        target = self.path(content_hash)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, payload)
            for plane, image in zip(_PREVIEW_PLANES, volume.middle_slices(geometry), strict=True):
                with image:
                    preview = image.copy()
                preview.thumbnail(
                    (self._settings.thumbnail_dimension, self._settings.thumbnail_dimension),
                    Image.Resampling.LANCZOS,
                )
                _atomic_write(
                    self.preview_path(content_hash, plane),
                    _encode_jpeg(preview.convert("RGB"), 82),
                )
                preview.close()

        depth, height, width = geometry.shape
        return StoredVolume(
            content_hash=content_hash,
            byte_size=len(payload),
            width=width,
            height=height,
            depth=depth,
            spacing=geometry.spacing,
            hu_slope=geometry.hu_slope,
            hu_intercept=geometry.hu_intercept,
            window_center=geometry.window_center,
            window_width=geometry.window_width,
            source_format=geometry.source_format,
            source_slice_count=geometry.source_slice_count,
        )


async def read_bounded(upload: UploadFile, limit: int) -> bytes:
    """Read at most ``limit`` bytes, then reject.

    Content-Length is attacker-controlled, so the ceiling is enforced on bytes
    actually read rather than on the advertised header.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(f"Максимальный размер файла — {limit // (1024 * 1024)} МБ.")
        chunks.append(chunk)

    if total == 0:
        raise UnsupportedMediaTypeError("Файл пуст.")
    return b"".join(chunks)


def fanned_path(root: Path, content_hash: str, suffix: str) -> Path:
    if len(content_hash) != _HASH_LENGTH or not content_hash.isalnum():
        # The value always comes from our own hashing; this is the guard that
        # keeps a malformed one from escaping the storage root anyway.
        raise ValueError("Invalid content hash")
    # Two levels of fan-out keep directory entry counts sane at millions of files.
    return root / content_hash[:2] / content_hash[2:4] / f"{content_hash}.{suffix}"


def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write via a temp file + rename so readers never observe a partial file."""
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_bytes(payload)
    temp.chmod(0o600)
    temp.replace(path)
