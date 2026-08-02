"""The case library: CRUD, seeding, delimited-column filters, isolation.

The tests that matter most are the two about *copying*. A library entry is
deliberately not a view over the record it was written from, and the only way
that property can be verified is to change or delete the source and look at the
entry afterwards.
"""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient


async def publish(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"title": "Имплантация 46 зуба", **overrides}
    response = await client.post("/api/v1/library", json=payload)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def upload_study(client: AsyncClient, image: bytes, patient_id: int | None = None) -> str:
    data = {} if patient_id is None else {"patient_id": str(patient_id)}
    response = await client.post(
        "/api/v1/studies", files={"file": ("opg.jpg", image, "image/jpeg")}, data=data
    )
    assert response.status_code == 201, response.text
    return str(response.json()["publicId"])


async def test_create_and_list(authed_client: AsyncClient) -> None:
    await publish(
        authed_client,
        summary="Отсроченная имплантация после удаления",
        findingKeys=["caries", "bone_loss"],
        tags=["имплантация", "хирургия"],
    )

    response = await authed_client.get("/api/v1/library")
    assert response.status_code == 200

    body = response.json()
    assert body["meta"]["total"] == 1
    item = body["items"][0]
    assert item["title"] == "Имплантация 46 зуба"
    assert item["tags"] == ["имплантация", "хирургия"]
    assert item["href"].endswith(item["publicId"])
    # Keys are stored; the labels and severities are resolved on the way out.
    assert [finding["key"] for finding in item["findings"]] == ["caries", "bone_loss"]
    assert item["findings"][0]["label"] == "Кариес"
    assert item["createdByName"] == "Айгуль Сагиндикова"


async def test_seeding_from_a_study_copies_its_diagnosis_and_finding_keys(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient = await authed_client.post("/api/v1/patients", json={"fullName": "Петров Пётр"})
    patient_id = patient.json()["id"]
    public_id = await upload_study(authed_client, radiograph_bytes, patient_id)
    study = (await authed_client.get(f"/api/v1/studies/{public_id}")).json()
    expected = {finding["classKey"] for finding in study["findings"]}

    entry = await publish(authed_client, title="Учебный разбор", fromStudyPublicId=public_id)

    assert {finding["key"] for finding in entry["findings"]} == expected
    assert entry["diagnosis"]
    # Provenance, and the patient the case came from, both carried over.
    assert entry["studyPublicId"] == public_id
    assert entry["patientId"] == patient_id


async def test_a_seeded_case_survives_the_study_it_was_written_from(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """The whole point of the copy, asserted the only way it can be."""
    public_id = await upload_study(authed_client, radiograph_bytes)
    entry = await publish(authed_client, title="Разбор", fromStudyPublicId=public_id)

    assert (await authed_client.delete(f"/api/v1/studies/{public_id}")).status_code == 200

    after = await authed_client.get(f"/api/v1/library/{entry['publicId']}")
    assert after.status_code == 200
    assert after.json()["diagnosis"] == entry["diagnosis"]
    assert after.json()["findings"] == entry["findings"]
    assert after.json()["studyPublicId"] == public_id


async def test_seeding_from_another_clinics_study_is_not_found(
    two_clinics: tuple[AsyncClient, AsyncClient], radiograph_bytes: bytes
) -> None:
    alpha, beta = two_clinics
    public_id = await upload_study(alpha, radiograph_bytes)

    response = await beta.post(
        "/api/v1/library", json={"title": "Чужой случай", "fromStudyPublicId": public_id}
    )
    assert response.status_code == 404


async def test_explicit_input_wins_over_the_seed(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    public_id = await upload_study(authed_client, radiograph_bytes)

    entry = await publish(
        authed_client,
        title="Разбор",
        diagnosis="Хронический периодонтит 36",
        findingKeys=["periapical_lesion"],
        fromStudyPublicId=public_id,
    )
    assert entry["diagnosis"] == "Хронический периодонтит 36"
    assert [finding["key"] for finding in entry["findings"]] == ["periapical_lesion"]


async def test_tags_are_folded_and_deduplicated(authed_client: AsyncClient) -> None:
    entry = await publish(
        authed_client, tags=["Имплантация", "имплантация", "  Хирургия  ", "с,запятой"]
    )
    # The comma is the stored delimiter, so it cannot survive inside a tag.
    assert entry["tags"] == ["имплантация", "хирургия", "с запятой"]


async def test_the_tag_filter_matches_whole_tags(authed_client: AsyncClient) -> None:
    await publish(authed_client, title="Случай A", tags=["имплантация"])

    exact = await authed_client.get("/api/v1/library", params={"tag": "имплантация"})
    assert exact.json()["meta"]["total"] == 1

    # A substring is not a tag: `LIKE '%имплант%'` would match here.
    prefix = await authed_client.get("/api/v1/library", params={"tag": "имплант"})
    assert prefix.json()["meta"]["total"] == 0


async def test_the_finding_filter_matches_whole_class_keys(authed_client: AsyncClient) -> None:
    """`caries` and `caries_3d` are both real keys, one per taxonomy."""
    await publish(authed_client, title="КЛКТ-случай", findingKeys=["caries_3d"])

    volumetric = await authed_client.get("/api/v1/library", params={"finding": "caries_3d"})
    assert volumetric.json()["meta"]["total"] == 1

    flat = await authed_client.get("/api/v1/library", params={"finding": "caries"})
    assert flat.json()["meta"]["total"] == 0


async def test_free_text_finds_a_case_by_its_cyrillic_title(authed_client: AsyncClient) -> None:
    await publish(authed_client, title="Синус-лифтинг слева", summary="Костная пластика")

    # Lower-cased needle against a folded column: SQL `lower()` would find
    # nothing here on SQLite.
    for query in ("синус", "СИНУС", "костная"):
        response = await authed_client.get("/api/v1/library", params={"q": query})
        assert response.json()["meta"]["total"] == 1, query


async def test_update_replaces_fields_and_refreshes_the_search_column(
    authed_client: AsyncClient,
) -> None:
    entry = await publish(authed_client, title="Старое название", summary="Заметка", tags=["a"])

    updated = await authed_client.put(
        f"/api/v1/library/{entry['publicId']}", json={"title": "Новое название"}
    )
    assert updated.status_code == 200
    # PUT is a full replacement: an omitted field is cleared, not retained.
    assert updated.json()["summary"] == ""
    assert updated.json()["tags"] == []

    found = await authed_client.get("/api/v1/library", params={"q": "новое"})
    assert found.json()["meta"]["total"] == 1
    stale = await authed_client.get("/api/v1/library", params={"q": "старое"})
    assert stale.json()["meta"]["total"] == 0


async def test_tags_endpoint_counts_distinct_tags(authed_client: AsyncClient) -> None:
    await publish(authed_client, title="Один", tags=["имплантация", "хирургия"])
    await publish(authed_client, title="Два", tags=["имплантация"])

    response = await authed_client.get("/api/v1/library/tags")
    assert response.status_code == 200
    # Most used first, so the chip row leads with the useful filter.
    assert response.json() == [
        {"tag": "имплантация", "count": 2},
        {"tag": "хирургия", "count": 1},
    ]


async def test_delete_removes_the_entry(authed_client: AsyncClient) -> None:
    entry = await publish(authed_client)

    assert (await authed_client.delete(f"/api/v1/library/{entry['publicId']}")).status_code == 200
    assert (await authed_client.get(f"/api/v1/library/{entry['publicId']}")).status_code == 404
    assert (await authed_client.get("/api/v1/library")).json()["items"] == []


async def test_a_blank_title_is_rejected(authed_client: AsyncClient) -> None:
    response = await authed_client.post("/api/v1/library", json={"title": "x"})
    assert response.status_code == 422
    assert "title" in response.json()["errors"]


async def test_cases_are_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    alpha, beta = two_clinics
    entry = await publish(alpha, title="Клинический разбор", tags=["имплантация"])
    public_id = entry["publicId"]

    assert (await beta.get("/api/v1/library")).json()["items"] == []
    assert (await beta.get("/api/v1/library", params={"q": "разбор"})).json()["items"] == []
    assert (await beta.get("/api/v1/library/tags")).json() == []

    # 404 rather than 403 everywhere: a 403 would confirm the entry exists.
    assert (await beta.get(f"/api/v1/library/{public_id}")).status_code == 404
    assert (
        await beta.put(f"/api/v1/library/{public_id}", json={"title": "Взлом"})
    ).status_code == 404
    assert (await beta.delete(f"/api/v1/library/{public_id}")).status_code == 404

    # And the original is untouched by any of it.
    assert (await alpha.get(f"/api/v1/library/{public_id}")).json()["title"] == "Клинический разбор"


async def test_library_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/library")).status_code == 401
    assert (await client.get("/api/v1/library/tags")).status_code == 401
