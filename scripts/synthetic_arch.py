#!/usr/bin/env python3
"""Generate a synthetic dental arch mesh for seeding and tests.

An intraoral scan is a surface, so the demo data has to be one too: a
horseshoe ridge with a bump per tooth, triangulated over an arch-shaped grid.
Nothing here is a patient's anatomy — that is the point.
"""

from __future__ import annotations

import math
import struct
from typing import Final

#: Teeth per arch in the demo model: everything but the third molars.
TEETH_PER_ARCH: Final[int] = 14
#: Half-width and depth of the arch, in millimetres.
ARCH_WIDTH_MM: Final[float] = 30.0
ARCH_DEPTH_MM: Final[float] = 26.0
GUM_HEIGHT_MM: Final[float] = 4.0
CROWN_HEIGHT_MM: Final[float] = 7.5
RIDGE_HALF_WIDTH_MM: Final[float] = 5.0

_ALONG_STEPS: Final[int] = 168
_ACROSS_STEPS: Final[int] = 16

Vector = tuple[float, float, float]


def arch_surface(*, upper: bool = True, seed: float = 0.0) -> list[tuple[Vector, Vector, Vector]]:
    """Triangulate a horseshoe ridge with one bump per tooth."""
    grid = [
        [
            _point(along / _ALONG_STEPS, across / _ACROSS_STEPS, upper, seed)
            for across in range(_ACROSS_STEPS + 1)
        ]
        for along in range(_ALONG_STEPS + 1)
    ]

    triangles: list[tuple[Vector, Vector, Vector]] = []
    for along in range(_ALONG_STEPS):
        for across in range(_ACROSS_STEPS):
            a = grid[along][across]
            b = grid[along + 1][across]
            c = grid[along + 1][across + 1]
            d = grid[along][across + 1]
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return triangles


def _point(along: float, across: float, upper: bool, seed: float) -> Vector:
    # The arch runs from one second molar to the other over an elliptical arc.
    angle = math.pi * (along - 0.5) * 1.15
    centre_x = ARCH_WIDTH_MM * math.sin(angle)
    centre_y = ARCH_DEPTH_MM * math.cos(angle)

    # Outward normal of the arch curve, used to give the ridge its width.
    normal_x = math.sin(angle)
    normal_y = math.cos(angle)
    lateral = (across - 0.5) * 2.0 * RIDGE_HALF_WIDTH_MM

    tooth_phase = math.cos(along * TEETH_PER_ARCH * 2.0 * math.pi)
    crown = CROWN_HEIGHT_MM * (0.55 + 0.45 * tooth_phase)
    # Molars sit lower than incisors, and the ridge falls away at its edges.
    posterior = 1.0 - 0.35 * abs(along - 0.5) * 2.0
    shoulder = math.cos((across - 0.5) * math.pi)
    wobble = 0.4 * math.sin(along * 37.0 + seed)

    height = GUM_HEIGHT_MM + crown * posterior * shoulder + wobble
    return (
        centre_x + normal_x * lateral,
        centre_y + normal_y * lateral,
        height if upper else -height,
    )


def binary_stl(triangles: list[tuple[Vector, Vector, Vector]], header: bytes) -> bytes:
    payload = [header.ljust(80, b"\x00"), struct.pack("<I", len(triangles))]
    for triangle in triangles:
        payload.append(struct.pack("<3f", *_normal(triangle)))
        for vertex in triangle:
            payload.append(struct.pack("<3f", *vertex))
        payload.append(struct.pack("<H", 0))
    return b"".join(payload)


def _normal(triangle: tuple[Vector, Vector, Vector]) -> Vector:
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (nx / length, ny / length, nz / length)


def arch_stl_bytes(*, upper: bool = True, seed: float = 0.0) -> bytes:
    label = b"dentist-ai synthetic upper arch" if upper else b"dentist-ai synthetic lower arch"
    return binary_stl(arch_surface(upper=upper, seed=seed), label)


if __name__ == "__main__":
    for upper in (True, False):
        data = arch_stl_bytes(upper=upper)
        name = "upper" if upper else "lower"
        print(f"{name}: {len(data):,} bytes")
