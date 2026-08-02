"""3D scan upload, delivery and tenant isolation."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.conftest import ascii_stl_bytes, binary_ply_bytes, binary_stl_bytes, obj_bytes


async def _patient(client: AsyncClient, name: str = "Кимов Тимур") -> int:
    response = await client.post("/api/v1/patients", json={"fullName": name})
    assert response.status_code == 201, response.text
    patient_id: int = response.json()["id"]
    return patient_id


async def _upload(
    client: AsyncClient,
    patient_id: int,
    payload: bytes,
    filename: str = "arch.stl",
    **fields: str,
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/scans",
        files={"file": (filename, payload, "application/octet-stream")},
        data={"patient_id": str(patient_id), **fields},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_upload_records_geometry_not_just_the_file(authed_client: AsyncClient) -> None:
    patient_id = await _patient(authed_client)
    scan = await _upload(authed_client, patient_id, binary_stl_bytes())

    assert scan["triangleCount"] == 12
    assert scan["sourceFormat"] == "stl"
    assert scan["bounds"]["size"] == [10.0, 8.0, 6.0]
    assert scan["kindLabel"]
    assert scan["meshUrl"].endswith("/mesh")


async def test_every_format_lands_as_the_same_stored_mesh(authed_client: AsyncClient) -> None:
    """Same geometry, one blob: the content hash is a real duplicate check."""
    patient_id = await _patient(authed_client)
    uploads = [
        await _upload(authed_client, patient_id, binary_stl_bytes(), "a.stl"),
        await _upload(authed_client, patient_id, ascii_stl_bytes(), "b.stl"),
        await _upload(authed_client, patient_id, obj_bytes(), "c.obj"),
        await _upload(authed_client, patient_id, binary_ply_bytes(), "d.ply"),
    ]

    assert {scan["sourceFormat"] for scan in uploads} == {"stl", "obj", "ply"}
    meshes = set()
    for scan in uploads:
        response = await authed_client.get(scan["meshUrl"])
        assert response.status_code == 200, response.text
        meshes.add(response.content)
    assert len(meshes) == 1


async def test_the_mesh_route_serves_binary_stl(authed_client: AsyncClient) -> None:
    patient_id = await _patient(authed_client)
    scan = await _upload(authed_client, patient_id, obj_bytes(), "arch.obj")

    response = await authed_client.get(scan["meshUrl"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/stl"
    # 80-byte header, a uint32 count, then 50 bytes per facet.
    assert len(response.content) == 84 + 12 * 50


async def test_a_radiograph_is_not_a_mesh(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id = await _patient(authed_client)
    response = await authed_client.post(
        "/api/v1/scans",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )
    assert response.status_code == 415, response.text


async def test_a_scan_cannot_be_attached_to_another_clinics_patient(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    alpha, beta = two_clinics
    patient_id = await _patient(alpha)

    response = await beta.post(
        "/api/v1/scans",
        files={"file": ("arch.stl", binary_stl_bytes(), "application/octet-stream")},
        data={"patient_id": str(patient_id)},
    )
    assert response.status_code == 404, response.text


async def test_another_clinic_cannot_read_a_stored_mesh(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    """The blob is shared by content hash; the authorisation is not."""
    alpha, beta = two_clinics
    patient_id = await _patient(alpha)
    scan = await _upload(alpha, patient_id, binary_stl_bytes())

    assert (await beta.get(scan["meshUrl"])).status_code == 404
    assert (await beta.get(f"/api/v1/scans/{scan['publicId']}")).status_code == 404


async def test_an_anonymous_visitor_cannot_read_a_mesh(
    authed_client: AsyncClient, client: AsyncClient
) -> None:
    patient_id = await _patient(authed_client)
    scan = await _upload(authed_client, patient_id, binary_stl_bytes())
    await authed_client.post("/api/v1/auth/logout")

    assert (await client.get(scan["meshUrl"])).status_code == 401


async def test_metadata_can_be_corrected_after_upload(authed_client: AsyncClient) -> None:
    patient_id = await _patient(authed_client)
    scan = await _upload(authed_client, patient_id, binary_stl_bytes())

    response = await authed_client.patch(
        f"/api/v1/scans/{scan['publicId']}",
        json={
            "kind": "plaster_model",
            "arch": "lower",
            "capturedOn": "2026-05-04",
            "notes": "Скан модели после препарирования",
        },
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["kind"] == "plaster_model"
    assert updated["arch"] == "lower"
    assert updated["capturedOn"] == "2026-05-04"


async def test_deleting_a_scan_reclaims_its_blob(authed_client: AsyncClient) -> None:
    patient_id = await _patient(authed_client)
    scan = await _upload(authed_client, patient_id, binary_stl_bytes())

    assert (await authed_client.delete(f"/api/v1/scans/{scan['publicId']}")).status_code == 200
    assert (await authed_client.get(scan["meshUrl"])).status_code == 404


async def test_scans_are_listed_for_their_patient_only(authed_client: AsyncClient) -> None:
    first = await _patient(authed_client, "Первый Пациент")
    second = await _patient(authed_client, "Второй Пациент")
    await _upload(authed_client, first, binary_stl_bytes())

    listed = (await authed_client.get("/api/v1/scans", params={"patient_id": first})).json()
    assert listed["meta"]["total"] == 1

    other = (await authed_client.get("/api/v1/scans", params={"patient_id": second})).json()
    assert other["meta"]["total"] == 0
