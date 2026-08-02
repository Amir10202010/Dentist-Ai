"""Mesh decoding: every supported format must yield the same geometry."""

from __future__ import annotations

import struct
from collections.abc import Callable

import pytest

from dentist_ai.core.errors import UnsupportedMediaTypeError
from dentist_ai.db.models import MeshFormat
from dentist_ai.services import mesh
from tests.conftest import (
    CUBE_FACES,
    CUBE_VERTICES,
    ascii_ply_bytes,
    ascii_stl_bytes,
    binary_ply_bytes,
    binary_stl_bytes,
    obj_bytes,
)

FIXTURES: list[tuple[Callable[[], bytes], MeshFormat]] = [
    (binary_stl_bytes, MeshFormat.STL),
    (ascii_stl_bytes, MeshFormat.STL),
    (obj_bytes, MeshFormat.OBJ),
    (ascii_ply_bytes, MeshFormat.PLY),
    (binary_ply_bytes, MeshFormat.PLY),
]


@pytest.mark.parametrize(("factory", "expected_format"), FIXTURES)
def test_every_format_decodes_to_the_same_cube(
    factory: Callable[[], bytes], expected_format: MeshFormat
) -> None:
    geometry = mesh.parse(factory())

    assert geometry.source_format is expected_format
    assert geometry.triangle_count == len(CUBE_FACES)
    assert geometry.bounds == ((0.0, 0.0, 0.0), (10.0, 8.0, 6.0))


@pytest.mark.parametrize(("factory", "_expected"), FIXTURES)
def test_canonical_output_is_identical_across_formats(
    factory: Callable[[], bytes], _expected: MeshFormat
) -> None:
    """Same geometry, same bytes — which is what makes the hash a dedup key."""
    reference = mesh.encode_binary_stl(mesh.parse(binary_stl_bytes()).triangles)
    assert mesh.encode_binary_stl(mesh.parse(factory()).triangles) == reference


def test_canonical_stl_round_trips() -> None:
    payload = mesh.encode_binary_stl(mesh.parse(obj_bytes()).triangles)
    reparsed = mesh.parse(payload)

    assert reparsed.source_format is MeshFormat.STL
    assert reparsed.triangle_count == len(CUBE_FACES)
    assert reparsed.bounds == ((0.0, 0.0, 0.0), (10.0, 8.0, 6.0))


def test_normals_are_recomputed_not_trusted() -> None:
    """The fixtures declare a zero normal; the canonical file must not."""
    payload = mesh.encode_binary_stl(mesh.parse(binary_stl_bytes()).triangles)
    first_normal = struct.unpack_from("<3f", payload, 84)
    assert any(abs(component) > 0.5 for component in first_normal)


def test_a_jpeg_is_not_a_mesh() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        mesh.parse(b"\xff\xd8\xff\xe0" + b"\x00" * 200)


def test_a_face_pointing_past_the_vertex_list_is_rejected() -> None:
    lines = [f"v {x} {y} {z}" for x, y, z in CUBE_VERTICES]
    lines.append("f 1 2 999")
    with pytest.raises(UnsupportedMediaTypeError):
        mesh.parse("\n".join(lines).encode())


def test_an_ascii_stl_with_a_half_written_triangle_is_rejected() -> None:
    truncated = "\n".join(
        [
            "solid cube",
            "  facet normal 0 0 0",
            "    outer loop",
            "      vertex 0 0 0",
            "      vertex 1 0 0",
            "    endloop",
            "  endfacet",
            "endsolid cube",
        ]
    ).encode()
    with pytest.raises(UnsupportedMediaTypeError):
        mesh.parse(truncated)


def test_a_binary_stl_header_starting_with_solid_is_still_binary() -> None:
    """The one ambiguity in the format: header text is not a format marker."""
    body = binary_stl_bytes()[80:]
    disguised = b"solid ".ljust(80, b"\x00") + body

    geometry = mesh.parse(disguised)
    assert geometry.triangle_count == len(CUBE_FACES)


def test_a_quad_is_triangulated() -> None:
    lines = [f"v {x} {y} {z}" for x, y, z in CUBE_VERTICES[:4]]
    lines.append("f 1 2 3 4")
    geometry = mesh.parse("\n".join(lines).encode())
    assert geometry.triangle_count == 2
