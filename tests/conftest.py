"""Test fixtures.

The suite drives the real HTTP stack — middleware, CSRF, cookies, the
database — through an in-process ASGI transport. Only the detector is
swapped, and that is a first-class backend rather than a mock.
"""

from __future__ import annotations

import io
import struct
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from pydantic import SecretStr

from dentist_ai.core.config import (
    DatabaseSettings,
    MLSettings,
    SecuritySettings,
    Settings,
    StorageSettings,
)
from dentist_ai.db.base import Base
from dentist_ai.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        secret_key=SecretStr("test-secret-key-that-is-definitely-long-enough-1234"),
        log_level="WARNING",
        database=DatabaseSettings(url=f"sqlite+aiosqlite:///{tmp_path / 'test.sqlite3'}"),
        ml=MLSettings(backend="stub", warm_up_on_startup=False),
        storage=StorageSettings(root=tmp_path / "storage"),
        security=SecuritySettings(
            # Generous limits so functional tests do not trip them; the
            # rate-limit tests build their own tight settings.
            login_rate_limit="1000/1m",
            register_rate_limit="1000/1m",
            upload_rate_limit="1000/1m",
            api_rate_limit="10000/1m",
            # Argon2 at production cost would add ~40 ms per login across the
            # suite for no extra coverage.
            argon2_memory_kib=8 * 1024,
            argon2_time_cost=1,
        ),
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    transport = ASGITransport(app=app)

    async with (
        AsyncClient(
            transport=transport,
            base_url="http://testserver",
            # The CSRF origin check requires these on unsafe methods, exactly
            # as a real browser would send them.
            headers={"Origin": "http://testserver", "Host": "testserver"},
        ) as http_client,
        app.router.lifespan_context(app),
    ):
        async with app.state.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        yield http_client


@pytest.fixture
def radiograph_bytes() -> bytes:
    """A small synthetic greyscale image that passes real Pillow decoding."""
    image = Image.new("L", (900, 500), color=32)
    for x in range(0, 900, 60):
        for y in range(120, 380):
            image.putpixel((x, y), 220)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


#: A unit cube as eight corners and twelve triangles. Small enough to inline,
#: real enough that every parser has to agree on the same 12 facets.
CUBE_VERTICES: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (10.0, 8.0, 0.0),
    (0.0, 8.0, 0.0),
    (0.0, 0.0, 6.0),
    (10.0, 0.0, 6.0),
    (10.0, 8.0, 6.0),
    (0.0, 8.0, 6.0),
)
CUBE_FACES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 3),
    (4, 6, 5),
    (4, 7, 6),
    (0, 4, 5),
    (0, 5, 1),
    (1, 5, 6),
    (1, 6, 2),
    (2, 6, 7),
    (2, 7, 3),
    (3, 7, 4),
    (3, 4, 0),
)


def binary_stl_bytes() -> bytes:
    payload = [b"binary stl fixture".ljust(80, b"\x00"), struct.pack("<I", len(CUBE_FACES))]
    for face in CUBE_FACES:
        payload.append(struct.pack("<3f", 0.0, 0.0, 0.0))
        for index in face:
            payload.append(struct.pack("<3f", *CUBE_VERTICES[index]))
        payload.append(struct.pack("<H", 0))
    return b"".join(payload)


def ascii_stl_bytes() -> bytes:
    lines = ["solid cube"]
    for face in CUBE_FACES:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        lines.extend(f"      vertex {x} {y} {z}" for x, y, z in (CUBE_VERTICES[i] for i in face))
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid cube")
    return "\n".join(lines).encode()


def obj_bytes() -> bytes:
    lines = ["# cube fixture"]
    lines.extend(f"v {x} {y} {z}" for x, y, z in CUBE_VERTICES)
    lines.extend(f"f {a + 1}//1 {b + 1}//1 {c + 1}//1" for a, b, c in CUBE_FACES)
    return "\n".join(lines).encode()


def ascii_ply_bytes() -> bytes:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(CUBE_VERTICES)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(CUBE_FACES)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    body = [f"{x} {y} {z}" for x, y, z in CUBE_VERTICES]
    body += [f"3 {a} {b} {c}" for a, b, c in CUBE_FACES]
    return "\n".join(header + body).encode()


def binary_ply_bytes() -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(CUBE_VERTICES)}",
            "property float x",
            "property float y",
            "property float z",
            f"element face {len(CUBE_FACES)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode()
    body = [struct.pack("<3f", *vertex) for vertex in CUBE_VERTICES]
    body += [struct.pack("<B3i", 3, *face) for face in CUBE_FACES]
    return header + b"".join(body)


REGISTRATION = {
    "fullName": "Айгуль Сагиндикова",
    "email": "aigul@clinic.kz",
    "organizationName": "Клиника Смайл",
    "password": "correct-horse-battery",
    "passwordConfirm": "correct-horse-battery",
}


async def bootstrap_csrf(client: AsyncClient) -> str:
    """Fetch an anonymous CSRF token, exactly as the browser bundle does."""
    response = await client.get("/api/v1/auth/csrf")
    assert response.status_code == 200, response.text
    return str(response.json()["csrfToken"])


async def register_and_login(client: AsyncClient) -> str:
    """Register a clinic and return the CSRF token for subsequent writes."""
    token = await bootstrap_csrf(client)
    response = await client.post(
        "/api/v1/auth/register", json=REGISTRATION, headers={"X-CSRF-Token": token}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["csrfToken"])


@pytest.fixture
async def authed_client(client: AsyncClient) -> AsyncClient:
    token = await register_and_login(client)
    client.headers["X-CSRF-Token"] = token
    return client


@pytest.fixture
async def two_clinics(settings: Settings) -> AsyncIterator[tuple[AsyncClient, AsyncClient]]:
    """Two authenticated clients in different organisations, one database."""
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        async with app.state.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        clients: list[AsyncClient] = []
        for index, email in enumerate(("a@clinic.kz", "b@clinic.kz")):
            client = AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
                headers={"Origin": "http://testserver"},
            )
            token = await bootstrap_csrf(client)
            response = await client.post(
                "/api/v1/auth/register",
                json={
                    **REGISTRATION,
                    "email": email,
                    "organizationName": f"Clinic {index}",
                },
                headers={"X-CSRF-Token": token},
            )
            assert response.status_code == 201, response.text
            client.headers["X-CSRF-Token"] = response.json()["csrfToken"]
            clients.append(client)

        yield clients[0], clients[1]

        for client in clients:
            await client.aclose()
