"""Multi-tenant isolation and rate limiting.

Tenant leakage is the failure mode that would end this product, so it gets
direct tests rather than being assumed from the query code.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
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
from tests.conftest import bootstrap_csrf


async def test_patients_are_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    alpha, beta = two_clinics

    created = await alpha.post("/api/v1/patients", json={"fullName": "Иванов Иван"})
    assert created.status_code == 201
    patient_id = created.json()["id"]

    listing = await beta.get("/api/v1/patients")
    assert listing.status_code == 200
    assert listing.json()["items"] == []

    # 404, not 403: confirming existence would itself leak information.
    assert (await beta.get(f"/api/v1/patients/{patient_id}")).status_code == 404
    assert (
        await beta.put(f"/api/v1/patients/{patient_id}", json={"fullName": "Взлом"})
    ).status_code == 404
    assert (await beta.delete(f"/api/v1/patients/{patient_id}")).status_code == 404


async def test_studies_and_images_are_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient], radiograph_bytes: bytes
) -> None:
    alpha, beta = two_clinics

    upload = await alpha.post(
        "/api/v1/studies", files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")}
    )
    assert upload.status_code == 201
    public_id = upload.json()["publicId"]

    assert (await beta.get(f"/api/v1/studies/{public_id}")).status_code == 404
    # The blob is shared on disk by content hash — authorisation, not storage
    # layout, is what keeps it private.
    assert (await beta.get(f"/api/v1/studies/{public_id}/image")).status_code == 404
    assert (await beta.get(f"/api/v1/studies/{public_id}/thumbnail")).status_code == 404
    assert (await beta.get(f"/api/v1/studies/{public_id}/export.csv")).status_code == 404
    assert (await beta.delete(f"/api/v1/studies/{public_id}")).status_code == 404
    assert (await beta.get("/api/v1/studies")).json()["items"] == []


async def test_cannot_attach_study_to_another_clinics_patient(
    two_clinics: tuple[AsyncClient, AsyncClient], radiograph_bytes: bytes
) -> None:
    alpha, beta = two_clinics
    patient_id = (await alpha.post("/api/v1/patients", json={"fullName": "Иванов Иван"})).json()[
        "id"
    ]

    response = await beta.post(
        "/api/v1/studies",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )
    assert response.status_code == 404


async def test_dashboard_counts_only_own_clinic(
    two_clinics: tuple[AsyncClient, AsyncClient], radiograph_bytes: bytes
) -> None:
    alpha, beta = two_clinics
    await alpha.post("/api/v1/patients", json={"fullName": "Иванов Иван"})
    await alpha.post("/api/v1/studies", files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")})

    assert (await alpha.get("/api/v1/dashboard")).json()["totalPatients"] == 1
    beta_dashboard = (await beta.get("/api/v1/dashboard")).json()
    assert beta_dashboard["totalPatients"] == 0
    assert beta_dashboard["totalStudies"] == 0


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
@pytest.fixture
def strict_settings(settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    directory = tmp_path_factory.mktemp("strict")
    return settings.model_copy(
        update={
            "database": DatabaseSettings(url=f"sqlite+aiosqlite:///{directory / 'db.sqlite3'}"),
            "storage": StorageSettings(root=directory / "storage"),
            "ml": MLSettings(backend="stub", warm_up_on_startup=False),
            "security": SecuritySettings(
                login_rate_limit="3/1m",
                argon2_memory_kib=8 * 1024,
                argon2_time_cost=1,
            ),
            "secret_key": SecretStr("test-secret-key-that-is-definitely-long-enough-1234"),
        }
    )


async def test_login_brute_force_is_throttled(strict_settings: Settings) -> None:
    app = create_app(strict_settings)
    async with app.router.lifespan_context(app):
        async with app.state.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Origin": "http://testserver"},
        ) as client:
            token = await bootstrap_csrf(client)
            attempt = {"email": "victim@clinic.kz", "password": "guess"}

            for _ in range(3):
                response = await client.post(
                    "/api/v1/auth/login", json=attempt, headers={"X-CSRF-Token": token}
                )
                assert response.status_code == 401

            blocked = await client.post(
                "/api/v1/auth/login", json=attempt, headers={"X-CSRF-Token": token}
            )
            assert blocked.status_code == 429
            assert blocked.json()["code"] == "rate_limited"
            assert int(blocked.headers["Retry-After"]) > 0
