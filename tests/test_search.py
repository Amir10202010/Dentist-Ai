"""Workspace search: grouping, filters, facets and tenant isolation.

Assertions are on the class keys the records actually came back with rather
than on keys written into the test. The 2D backend is content-seeded and the
CBCT pipeline is a rules classifier; pinning either one's output would make
this file fail on every legitimate tuning change while catching none of the
regressions it exists for.
"""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient
from scripts.synthetic_cbct import build_preset


async def create_patient(client: AsyncClient, name: str = "Иванов Иван Иванович") -> int:
    response = await client.post("/api/v1/patients", json={"fullName": name})
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


async def upload_study(
    client: AsyncClient, image: bytes, patient_id: int | None = None
) -> dict[str, Any]:
    data = {} if patient_id is None else {"patient_id": str(patient_id)}
    response = await client.post(
        "/api/v1/studies", files={"file": ("opg.jpg", image, "image/jpeg")}, data=data
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def upload_volume(client: AsyncClient, patient_id: int) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/volumes",
        files={
            "file": ("cbct.nii", build_preset("periapical", seed=7), "application/octet-stream")
        },
        data={"patient_id": str(patient_id), "field_of_view": "both_jaws"},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def publish_case(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"title": "Клинический разбор", **overrides}
    response = await client.post("/api/v1/library", json=payload)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def group(body: dict[str, Any], kind: str) -> dict[str, Any]:
    matches = [item for item in body["groups"] if item["kind"] == kind]
    assert matches, f"no {kind} group in {[item['kind'] for item in body['groups']]}"
    return dict(matches[0])


async def search(client: AsyncClient, **params: Any) -> dict[str, Any]:
    response = await client.get("/api/v1/search", params=params)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_free_text_matches_a_patient_by_a_cyrillic_name(authed_client: AsyncClient) -> None:
    await create_patient(authed_client)

    # Both cases, against a column folded in Python on write. SQL `lower()`
    # would answer this differently on SQLite than on Postgres.
    for query in ("иванов", "ИВАНОВ"):
        body = await search(authed_client, q=query)
        assert group(body, "patient")["total"] == 1, query
        assert group(body, "patient")["items"][0]["title"] == "Иванов Иван Иванович"


async def test_a_radiograph_is_found_through_its_patient_and_by_its_id(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id = await create_patient(authed_client)
    study = await upload_study(authed_client, radiograph_bytes, patient_id)

    by_patient = await search(authed_client, q="иванов", kind="study")
    assert [item["id"] for item in group(by_patient, "study")["items"]] == [study["publicId"]]

    by_id = await search(authed_client, q=study["publicId"].lower(), kind="study")
    assert group(by_id, "study")["total"] == 1


async def test_every_row_carries_a_title_a_subtitle_a_href_and_a_timestamp(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """One list component renders all four kinds, so all four owe it the same
    fields."""
    patient_id = await create_patient(authed_client)
    await upload_study(authed_client, radiograph_bytes, patient_id)
    await publish_case(authed_client, tags=["имплантация"])

    body = await search(authed_client)
    rows = [row for item in body["groups"] for row in item["items"]]
    assert {row["kind"] for row in rows} == {"patient", "study", "case"}
    for row in rows:
        assert row["title"], row
        assert row["subtitle"], row
        assert row["href"].startswith("/app/"), row
        assert row["at"].endswith("Z"), row
        assert row["id"]


async def test_the_kind_filter_returns_only_that_group(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id = await create_patient(authed_client)
    await upload_study(authed_client, radiograph_bytes, patient_id)

    body = await search(authed_client, kind="patient")
    assert [item["kind"] for item in body["groups"]] == ["patient"]
    assert body["total"] == 1


async def test_search_by_finding_class_matches_both_2d_and_3d_findings(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id = await create_patient(authed_client)
    study = await upload_study(authed_client, radiograph_bytes, patient_id)
    volume = await upload_volume(authed_client, patient_id)

    flat_key = study["findings"][0]["classKey"]
    volumetric_key = volume["findings"][0]["classKey"]
    # A case cites keys from either taxonomy, so one filter reaches all three.
    await publish_case(authed_client, title="Разбор", findingKeys=[flat_key])

    by_flat = await search(authed_client, finding=flat_key)
    assert group(by_flat, "study")["total"] == 1
    assert group(by_flat, "case")["total"] == 1

    by_volumetric = await search(authed_client, finding=volumetric_key)
    assert group(by_volumetric, "volume")["total"] == 1

    absent = await search(authed_client, finding="no_such_class")
    assert absent["total"] == 0


async def test_the_finding_filter_reaches_a_patient_through_their_imaging(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """A chart carries no findings; someone filtering by one wants the people
    who have it."""
    patient_id = await create_patient(authed_client)
    other = await create_patient(authed_client, "Петров Пётр")
    study = await upload_study(authed_client, radiograph_bytes, patient_id)

    body = await search(authed_client, finding=study["findings"][0]["classKey"], kind="patient")
    assert [item["id"] for item in group(body, "patient")["items"]] == [str(patient_id)]
    assert str(other) not in [item["id"] for item in group(body, "patient")["items"]]


async def test_facets_ignore_the_filter_they_drive(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """Chips counted over the filtered set would all read zero but the one
    already selected, which is the state in which a filter UI says nothing."""
    patient_id = await create_patient(authed_client)
    study = await upload_study(authed_client, radiograph_bytes, patient_id)

    unfiltered = (await search(authed_client))["facets"]
    assert unfiltered["findings"]
    assert unfiltered["severities"]
    # Every chip is nameable and points at a class the result set contains.
    keys = {finding["classKey"] for finding in study["findings"]}
    assert {chip["key"] for chip in unfiltered["findings"]} == keys
    assert all(chip["label"] and chip["count"] > 0 for chip in unfiltered["findings"])

    picked = unfiltered["findings"][0]["key"]
    filtered = (await search(authed_client, finding=picked))["facets"]
    assert filtered["findings"] == unfiltered["findings"]
    assert filtered["severities"] == unfiltered["severities"]


async def test_the_severity_facet_counts_records_and_not_findings(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """One study with two findings of the same severity is one result."""
    patient_id = await create_patient(authed_client)
    study = await upload_study(authed_client, radiograph_bytes, patient_id)

    facets = (await search(authed_client))["facets"]
    assert all(chip["count"] == 1 for chip in facets["severities"])
    assert len(study["findings"]) > len(facets["severities"])


async def test_the_confidence_floor_excludes_written_cases(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """A library entry is written, not detected; it has no confidence to clear."""
    await upload_study(authed_client, radiograph_bytes)
    await publish_case(authed_client)

    assert group(await search(authed_client), "case")["total"] == 1
    assert group(await search(authed_client, min_confidence=0.5), "case")["total"] == 0


async def test_the_confidence_floor_narrows_the_imaging_result(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await upload_study(authed_client, radiograph_bytes)
    top = max(finding["confidence"] for finding in study["findings"])

    kept = await search(authed_client, min_confidence=top, kind="study")
    assert group(kept, "study")["total"] == 1

    # Nothing scores above 1.0, so the floor must empty the group.
    dropped = await search(authed_client, min_confidence=1.0, kind="study")
    assert group(dropped, "study")["total"] == 0


async def test_the_date_window_includes_its_last_day(authed_client: AsyncClient) -> None:
    await create_patient(authed_client)
    created = (await search(authed_client))["groups"][0]["items"][0]["at"]
    today = created[:10]

    inside = await search(authed_client, date_from=today, date_to=today, kind="patient")
    assert group(inside, "patient")["total"] == 1

    before = await search(authed_client, date_to="2000-01-01", kind="patient")
    assert group(before, "patient")["total"] == 0


async def test_pagination_reports_the_unpaginated_total(authed_client: AsyncClient) -> None:
    for index in range(3):
        await create_patient(authed_client, f"Пациент Номер {index}")

    body = await search(authed_client, kind="patient", limit=1)
    patients = group(body, "patient")
    assert len(patients["items"]) == 1
    # The total rides along as a window function rather than a second count.
    assert patients["total"] == 3


async def test_archived_patients_stay_out_of_the_results(authed_client: AsyncClient) -> None:
    patient_id = await create_patient(authed_client)
    assert (await authed_client.delete(f"/api/v1/patients/{patient_id}")).status_code == 200

    assert group(await search(authed_client, q="иванов"), "patient")["total"] == 0


async def test_results_are_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient], radiograph_bytes: bytes
) -> None:
    alpha, beta = two_clinics
    patient_id = await create_patient(alpha)
    study = await upload_study(alpha, radiograph_bytes, patient_id)
    await publish_case(alpha, title="Иванов: разбор", findingKeys=["caries"])

    for params in (
        {"q": "иванов"},
        {"q": study["publicId"]},
        {"finding": study["findings"][0]["classKey"]},
        {},
    ):
        body = await search(beta, **params)
        assert body["total"] == 0, params
        assert all(item["items"] == [] for item in body["groups"]), params
        # The chips must not leak either: a count is still a fact about
        # another clinic's records.
        assert body["facets"]["findings"] == []
        assert body["facets"]["severities"] == []


async def test_search_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/search")).status_code == 401
