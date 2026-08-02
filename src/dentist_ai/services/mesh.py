"""Reading 3D scans and normalising them to one canonical format.

Dental scanners export STL, PLY or OBJ, in ASCII or binary, with or without
colour, normals and texture coordinates. All of that is decoded here and
re-emitted as binary STL: triangles and nothing else.

Normalising on ingest buys the same things it buys for radiographs. The
browser needs one parser instead of four. A file that claims to be a mesh but
is not gets rejected at the door rather than shipped to a viewer. And the
stored bytes are a pure function of the geometry, so the content hash is a
real duplicate check rather than a checksum of whichever exporter wrote it.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from dentist_ai.core.errors import UnsupportedMediaTypeError
from dentist_ai.db.models import MeshFormat

TriangleArray = npt.NDArray[np.float32]

#: Above this the file is not a clinical scan; a full-arch intraoral scan is
#: 100k-500k triangles even before decimation.
MAX_TRIANGLES: Final[int] = 4_000_000

_BINARY_STL_HEADER: Final[int] = 80
_BINARY_STL_RECORD: Final[int] = 50
_STL_HEADER_TEXT: Final[bytes] = b"dentist-ai canonical mesh"
#: Some writers pad the end of a binary STL; a few bytes of slack are not a
#: reason to reject the file.
_STL_TRAILING_SLACK: Final[int] = 4
#: Shorter than one triangle in any text format.
_MIN_MESH_BYTES: Final[int] = 24

_TRIANGLE_VERTICES: Final[int] = 3
#: Header line shapes: "format <encoding> <version>", "element <name> <count>",
#: "property <type> <name>", "property list <count> <value> <name>".
_PLY_FORMAT_TOKENS: Final[int] = 2
_PLY_ELEMENT_TOKENS: Final[int] = 3
_PLY_PROPERTY_TOKENS: Final[int] = 3
_PLY_LIST_TOKENS: Final[int] = 5
#: "v x y z"
_OBJ_VERTEX_TOKENS: Final[int] = 4

_VERTEX_RE: Final[re.Pattern[bytes]] = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)

_PLY_TYPES: Final[dict[str, str]] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True, slots=True)
class MeshGeometry:
    """A decoded mesh, ready to be written out."""

    triangles: TriangleArray
    source_format: MeshFormat

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        flat = self.triangles.reshape(-1, 3)
        low = flat.min(axis=0)
        high = flat.max(axis=0)
        return (
            (float(low[0]), float(low[1]), float(low[2])),
            (float(high[0]), float(high[1]), float(high[2])),
        )


def parse(raw: bytes) -> MeshGeometry:
    """Decode a mesh file, rejecting anything that is not one."""
    if len(raw) < _MIN_MESH_BYTES:
        raise UnsupportedMediaTypeError("Файл слишком мал для 3D-модели.")

    mesh_format = _detect_format(raw)
    if mesh_format is MeshFormat.PLY:
        triangles = _parse_ply(raw)
    elif mesh_format is MeshFormat.OBJ:
        triangles = _parse_obj(raw)
    elif _is_binary_stl(raw):
        triangles = _parse_binary_stl(raw)
    else:
        triangles = _parse_ascii_stl(raw)

    if triangles.shape[0] == 0:
        raise UnsupportedMediaTypeError("В файле нет треугольников.")
    if triangles.shape[0] > MAX_TRIANGLES:
        raise UnsupportedMediaTypeError(
            f"Слишком детальная модель: {triangles.shape[0]:,} треугольников "
            f"при пределе {MAX_TRIANGLES:,}. Упростите сетку в программе сканера."
        )
    if not np.isfinite(triangles).all():
        raise UnsupportedMediaTypeError("В координатах модели есть NaN или бесконечность.")

    return MeshGeometry(triangles=triangles, source_format=mesh_format)


def encode_binary_stl(triangles: TriangleArray) -> bytes:
    """Write triangles as binary STL with recomputed facet normals."""
    count = triangles.shape[0]
    record = np.zeros(
        count,
        dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]),
    )
    record["normal"] = _face_normals(triangles)
    record["vertices"] = triangles

    header = _STL_HEADER_TEXT.ljust(_BINARY_STL_HEADER, b"\x00")
    return header + struct.pack("<I", count) + record.tobytes()


def _face_normals(triangles: TriangleArray) -> TriangleArray:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # A degenerate facet has no normal; STL allows the zero vector for it.
    np.divide(normals, lengths, out=normals, where=lengths > 0)
    return normals.astype(np.float32)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
def _detect_format(raw: bytes) -> MeshFormat:
    head = raw[:1024]
    if head.lstrip()[:3].lower() == b"ply":
        return MeshFormat.PLY
    if _is_binary_stl(raw) or b"facet normal" in head.lower():
        return MeshFormat.STL
    if _looks_like_obj(head):
        return MeshFormat.OBJ
    raise UnsupportedMediaTypeError(
        "Не удалось распознать 3D-модель. Поддерживаются STL, PLY и OBJ."
    )


def _is_binary_stl(raw: bytes) -> bool:
    """Binary STL has no magic number, only a self-consistent length.

    Its 80-byte header frequently starts with the word "solid", which is also
    how an ASCII STL starts, so the length check has to come first.
    """
    if len(raw) < _BINARY_STL_HEADER + 4:
        return False
    count = struct.unpack_from("<I", raw, _BINARY_STL_HEADER)[0]
    if count == 0 or count > MAX_TRIANGLES:
        return False
    expected = _BINARY_STL_HEADER + 4 + count * _BINARY_STL_RECORD
    return bool(len(raw) >= expected and len(raw) - expected < _STL_TRAILING_SLACK)


def _looks_like_obj(head: bytes) -> bool:
    return any(
        line.startswith((b"v ", b"vn ", b"f ", b"mtllib ", b"o ", b"g "))
        for line in head.splitlines()
    )


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------
def _parse_binary_stl(raw: bytes) -> TriangleArray:
    count = struct.unpack_from("<I", raw, _BINARY_STL_HEADER)[0]
    dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=_BINARY_STL_HEADER + 4)
    return np.ascontiguousarray(records["vertices"], dtype=np.float32)


def _parse_ascii_stl(raw: bytes) -> TriangleArray:
    matches = _VERTEX_RE.findall(raw)
    if not matches:
        raise UnsupportedMediaTypeError("В STL-файле нет вершин.")
    if len(matches) % 3 != 0:
        raise UnsupportedMediaTypeError("В STL-файле незакрытый треугольник.")
    values = _floats_from_tokens(matches)
    return values.reshape(-1, 3, 3)


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _PlyProperty:
    name: str
    value_type: str
    count_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None


@dataclass(frozen=True, slots=True)
class _PlyElement:
    name: str
    count: int
    properties: tuple[_PlyProperty, ...]


def _parse_ply(raw: bytes) -> TriangleArray:
    terminator = raw.find(b"end_header")
    if terminator == -1:
        raise UnsupportedMediaTypeError("В PLY-файле нет заголовка.")
    body_start = raw.find(b"\n", terminator) + 1
    header = raw[:terminator].decode("ascii", errors="replace")

    byte_order, elements = _parse_ply_header(header)
    vertices, faces = _read_ply_body(raw[body_start:], byte_order, elements)
    if faces.shape[0] == 0:
        raise UnsupportedMediaTypeError("В PLY-файле нет граней.")
    if int(faces.max()) >= vertices.shape[0]:
        raise UnsupportedMediaTypeError("В PLY-файле грань ссылается на несуществующую вершину.")
    return vertices[faces].astype(np.float32)


def _parse_ply_header(header: str) -> tuple[str, tuple[_PlyElement, ...]]:
    byte_order = ""
    elements: list[_PlyElement] = []
    properties: list[_PlyProperty] = []
    name = ""
    count = 0

    def flush() -> None:
        if name:
            elements.append(_PlyElement(name, count, tuple(properties)))

    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0].lower()
        if keyword == "format" and len(parts) >= _PLY_FORMAT_TOKENS:
            byte_order = {
                "ascii": "ascii",
                "binary_little_endian": "<",
                "binary_big_endian": ">",
            }.get(parts[1].lower(), "")
        elif keyword == "element" and len(parts) >= _PLY_ELEMENT_TOKENS:
            flush()
            name, count, properties = parts[1].lower(), int(parts[2]), []
        elif keyword == "property" and len(parts) >= _PLY_PROPERTY_TOKENS:
            if parts[1].lower() == "list" and len(parts) >= _PLY_LIST_TOKENS:
                properties.append(
                    _PlyProperty(parts[4].lower(), parts[3].lower(), parts[2].lower())
                )
            else:
                properties.append(_PlyProperty(parts[2].lower(), parts[1].lower()))
    flush()

    if not byte_order:
        raise UnsupportedMediaTypeError("Неизвестный формат PLY.")
    return byte_order, tuple(elements)


def _read_ply_body(
    body: bytes, byte_order: str, elements: tuple[_PlyElement, ...]
) -> tuple[TriangleArray, npt.NDArray[np.int64]]:
    if byte_order == "ascii":
        return _read_ply_ascii(body, elements)
    return _read_ply_binary(body, byte_order, elements)


def _read_ply_ascii(
    body: bytes, elements: tuple[_PlyElement, ...]
) -> tuple[TriangleArray, npt.NDArray[np.int64]]:
    lines = [line for line in body.splitlines() if line.strip()]
    cursor = 0
    vertices = np.zeros((0, 3), dtype=np.float32)
    faces: list[tuple[int, int, int]] = []

    for element in elements:
        chunk = lines[cursor : cursor + element.count]
        cursor += element.count
        if element.name == "vertex":
            columns = _ply_coordinate_columns(element)
            rows = [line.split() for line in chunk]
            vertices = np.array(
                [[row[index] for index in columns] for row in rows], dtype="S"
            ).astype(np.float32)
        elif element.name == "face":
            for line in chunk:
                tokens = line.split()
                indices = [int(token) for token in tokens[1 : 1 + int(tokens[0])]]
                faces.extend(_fan(indices))

    return vertices, _as_faces(faces)


def _read_ply_binary(
    body: bytes, byte_order: str, elements: tuple[_PlyElement, ...]
) -> tuple[TriangleArray, npt.NDArray[np.int64]]:
    offset = 0
    vertices = np.zeros((0, 3), dtype=np.float32)
    faces = _no_faces()

    for element in elements:
        if any(prop.is_list for prop in element.properties):
            indices, offset = _read_ply_binary_faces(body, offset, byte_order, element)
            if element.name == "face":
                faces = indices
            continue

        dtype = np.dtype(
            [(prop.name, byte_order + _ply_dtype(prop.value_type)) for prop in element.properties]
        )
        chunk = np.frombuffer(body, dtype=dtype, count=element.count, offset=offset)
        offset += dtype.itemsize * element.count
        if element.name == "vertex":
            vertices = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1, dtype=np.float32)

    return vertices, faces


def _read_ply_binary_faces(
    body: bytes, offset: int, byte_order: str, element: _PlyElement
) -> tuple[npt.NDArray[np.int64], int]:
    prop = next(item for item in element.properties if item.is_list)
    count_type = np.dtype(byte_order + _ply_dtype(prop.count_type or "uchar"))
    value_type = np.dtype(byte_order + _ply_dtype(prop.value_type))

    # Fast path: an all-triangle mesh is a fixed-size record, which is what
    # every scanner and every mesh tool actually writes. Reading it as one
    # array avoids a Python loop over a million faces.
    fixed = count_type.itemsize + _TRIANGLE_VERTICES * value_type.itemsize
    if len(body) - offset == fixed * element.count:
        dtype = np.dtype([("count", count_type), ("indices", value_type, _TRIANGLE_VERTICES)])
        chunk = np.frombuffer(body, dtype=dtype, count=element.count, offset=offset)
        if np.all(chunk["count"] == _TRIANGLE_VERTICES):
            return chunk["indices"].astype(np.int64), len(body)

    faces: list[tuple[int, int, int]] = []
    for _ in range(element.count):
        size = int(np.frombuffer(body, dtype=count_type, count=1, offset=offset)[0])
        offset += count_type.itemsize
        indices = np.frombuffer(body, dtype=value_type, count=size, offset=offset)
        offset += value_type.itemsize * size
        faces.extend(_fan([int(value) for value in indices]))
    return _as_faces(faces), offset


def _no_faces() -> npt.NDArray[np.int64]:
    return np.zeros((0, _TRIANGLE_VERTICES), dtype=np.int64)


def _as_faces(faces: list[tuple[int, int, int]]) -> npt.NDArray[np.int64]:
    if not faces:
        return _no_faces()
    return np.array(faces, dtype=np.int64).reshape(-1, _TRIANGLE_VERTICES)


def _ply_dtype(name: str) -> str:
    try:
        return _PLY_TYPES[name]
    except KeyError as exc:
        raise UnsupportedMediaTypeError(f"Тип {name!r} в PLY не поддерживается.") from exc


def _ply_coordinate_columns(element: _PlyElement) -> tuple[int, int, int]:
    names = [prop.name for prop in element.properties]
    try:
        return names.index("x"), names.index("y"), names.index("z")
    except ValueError as exc:
        raise UnsupportedMediaTypeError("В PLY-файле нет координат x/y/z.") from exc


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------
def _parse_obj(raw: bytes) -> TriangleArray:
    positions: list[tuple[bytes, bytes, bytes]] = []
    faces: list[tuple[int, int, int]] = []

    for line in raw.splitlines():
        if line.startswith(b"v "):
            parts = line.split()
            if len(parts) >= _OBJ_VERTEX_TOKENS:
                positions.append((parts[1], parts[2], parts[3]))
        elif line.startswith(b"f "):
            # Vertices may be "12", "12/4", "12//7" or "12/4/7"; only the
            # position index matters here. Negative indices count back from
            # the vertices seen so far.
            indices = [
                int(token.split(b"/")[0]) for token in line.split()[1:] if token.split(b"/")[0]
            ]
            resolved = [
                index - 1 if index > 0 else len(positions) + index
                for index in indices
                if index != 0
            ]
            faces.extend(_fan(resolved))

    if not positions:
        raise UnsupportedMediaTypeError("В OBJ-файле нет вершин.")
    if not faces:
        raise UnsupportedMediaTypeError("В OBJ-файле нет граней.")

    vertices = _floats_from_tokens(positions).reshape(-1, 3)
    index_array = np.array(faces, dtype=np.int64)
    if index_array.min() < 0 or index_array.max() >= vertices.shape[0]:
        raise UnsupportedMediaTypeError("В OBJ-файле грань ссылается на несуществующую вершину.")
    return vertices[index_array].astype(np.float32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _fan(indices: list[int]) -> list[tuple[int, int, int]]:
    """Triangulate a convex polygon as a fan around its first vertex."""
    return [
        (indices[0], indices[position], indices[position + 1])
        for position in range(1, len(indices) - 1)
    ]


def _floats_from_tokens(tokens: list[tuple[bytes, bytes, bytes]]) -> TriangleArray:
    try:
        return np.array(tokens, dtype="S").astype(np.float32).reshape(-1)
    except ValueError as exc:
        raise UnsupportedMediaTypeError("В координатах модели нечисловое значение.") from exc
