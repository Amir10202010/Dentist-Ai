"""Patient CRUD, search and the list projection."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


async def _create(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"fullName": "Иванов Иван Иванович", **overrides}
    response = await client.post("/api/v1/patients", json=payload)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_create_and_list(authed_client: AsyncClient) -> None:
    """Exercises the list projection with a non-empty page.

    An empty-list assertion would pass even if the row presenter were broken,
    which is exactly how a crash in it once slipped through.
    """
    await _create(authed_client, phone="+7 701 111 11 11", dateOfBirth="1990-06-15")

    response = await authed_client.get("/api/v1/patients")
    assert response.status_code == 200

    body = response.json()
    assert body["meta"]["total"] == 1
    item = body["items"][0]
    assert item["fullName"] == "Иванов Иван Иванович"
    # Aggregate columns are layered onto the ORM projection.
    assert item["studyCount"] == 0
    assert item["lastStudyAt"] is None
    # Computed field derived from the date of birth.
    assert isinstance(item["age"], int)
    assert item["age"] >= 35


async def test_study_count_reflects_uploads(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient = await _create(authed_client)

    upload = await authed_client.post(
        "/api/v1/studies",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient["id"])},
    )
    assert upload.status_code == 201
    assert upload.json()["patient"]["fullName"] == "Иванов Иван Иванович"

    item = (await authed_client.get("/api/v1/patients")).json()["items"][0]
    assert item["studyCount"] == 1
    assert item["lastStudyAt"] is not None


async def test_phone_is_normalised(authed_client: AsyncClient) -> None:
    patient = await _create(authed_client, phone="+7 (701) 222-33-44")
    assert patient["phone"] == "+77012223344"


async def test_future_date_of_birth_is_rejected(authed_client: AsyncClient) -> None:
    response = await authed_client.post(
        "/api/v1/patients", json={"fullName": "Тест Тестов", "dateOfBirth": "2099-01-01"}
    )
    assert response.status_code == 422
    assert "dateOfBirth" in response.json()["errors"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [("иванов", 1), ("+7701", 1), ("A-42", 1), ("петров", 0), ("нет-такого", 0)],
)
async def test_search(authed_client: AsyncClient, query: str, expected: int) -> None:
    await _create(authed_client, phone="+7 701 111 11 11", medicalRecordNumber="A-42")

    response = await authed_client.get("/api/v1/patients", params={"q": query})
    assert response.status_code == 200
    assert len(response.json()["items"]) == expected


async def test_duplicate_chart_number_is_rejected(authed_client: AsyncClient) -> None:
    await _create(authed_client, medicalRecordNumber="A-100")

    response = await authed_client.post(
        "/api/v1/patients", json={"fullName": "Петров Пётр", "medicalRecordNumber": "A-100"}
    )
    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


async def test_update_replaces_fields(authed_client: AsyncClient) -> None:
    patient = await _create(authed_client, phone="+77011111111")

    response = await authed_client.put(
        f"/api/v1/patients/{patient['id']}",
        json={"fullName": "Иванова Мария", "notes": "Аллергия на лидокаин"},
    )
    assert response.status_code == 200
    assert response.json()["fullName"] == "Иванова Мария"
    # PUT is a full replacement: an omitted field is cleared, not retained.
    assert response.json()["phone"] is None


async def test_archive_hides_then_restore_returns(authed_client: AsyncClient) -> None:
    patient = await _create(authed_client)

    assert (await authed_client.delete(f"/api/v1/patients/{patient['id']}")).status_code == 200
    assert (await authed_client.get("/api/v1/patients")).json()["items"] == []

    archived = await authed_client.get("/api/v1/patients", params={"include_archived": "true"})
    assert len(archived.json()["items"]) == 1

    restore = await authed_client.post(f"/api/v1/patients/{patient['id']}/restore")
    assert restore.status_code == 200
    assert len((await authed_client.get("/api/v1/patients")).json()["items"]) == 1


async def test_pagination_metadata(authed_client: AsyncClient) -> None:
    for index in range(5):
        await _create(authed_client, fullName=f"Пациент Номер {index}")

    page = await authed_client.get("/api/v1/patients", params={"limit": 2, "offset": 0})
    body = page.json()
    assert len(body["items"]) == 2
    assert body["meta"]["total"] == 5
    assert body["meta"]["hasMore"] is True

    last = await authed_client.get("/api/v1/patients", params={"limit": 2, "offset": 4})
    assert last.json()["meta"]["hasMore"] is False


async def test_patients_require_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/patients")).status_code == 401
