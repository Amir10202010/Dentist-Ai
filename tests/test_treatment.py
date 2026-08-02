"""Protocol table, plan proposals and the study report."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from dentist_ai.clinical import protocols
from dentist_ai.ml.taxonomy import FINDING_CLASSES, Category


def test_every_protocol_key_is_a_real_finding_class() -> None:
    known = {item.key for item in FINDING_CLASSES}
    assert set(protocols.PROTOCOLS) <= known


def test_every_protocol_procedure_exists() -> None:
    for codes in protocols.PROTOCOLS.values():
        for code in codes:
            assert protocols.by_code(code) is not None, code


def test_every_pathology_maps_to_at_least_one_procedure() -> None:
    """A pathology the plan cannot act on is a gap in the table, not a design."""
    for item in FINDING_CLASSES:
        if item.category is Category.PATHOLOGY:
            assert protocols.procedures_for(item.key), item.key


def test_recording_past_work_implies_no_treatment() -> None:
    for key in ("crown", "filling", "implant", "root_canal_treatment"):
        assert protocols.procedures_for(key) == ()


def test_priorities_rank_urgent_first() -> None:
    ranks = [priority.rank for priority in protocols.Priority]
    assert ranks == sorted(ranks)
    assert protocols.Priority.URGENT.rank < protocols.Priority.OPTIONAL.rank


@pytest.mark.parametrize("locale", ["ru", "en", "kk"])
def test_every_procedure_is_translated(locale: str) -> None:
    for procedure in protocols.PROCEDURES:
        assert procedure.label(locale).strip()


# ---------------------------------------------------------------------------
# End-to-end through the API
# ---------------------------------------------------------------------------
async def _patient_with_study(
    client: AsyncClient, radiograph_bytes: bytes
) -> tuple[int, dict[str, Any]]:
    created = await client.post("/api/v1/patients", json={"fullName": "Ахметова Дана"})
    assert created.status_code == 201, created.text
    patient_id: int = created.json()["id"]

    uploaded = await client.post(
        "/api/v1/studies",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )
    assert uploaded.status_code == 201, uploaded.text
    study: dict[str, Any] = uploaded.json()
    return patient_id, study


async def test_findings_arrive_already_charted(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    _, study = await _patient_with_study(authed_client, radiograph_bytes)

    charted = [f for f in study["findings"] if f["toothNumber"] is not None]
    assert charted, "no finding was assigned a tooth"
    for finding in charted:
        assert 11 <= finding["toothNumber"] <= 48
        assert finding["toothName"]
        assert finding["toothConfirmed"] is False


async def test_a_clinician_can_correct_the_tooth_number(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    _, study = await _patient_with_study(authed_client, radiograph_bytes)
    finding = next(f for f in study["findings"] if f["toothNumber"] is not None)

    response = await authed_client.put(
        f"/api/v1/studies/{study['publicId']}/findings/{finding['id']}/tooth",
        json={"toothNumber": 47},
    )
    assert response.status_code == 200, response.text
    assert response.json()["toothNumber"] == 47
    assert response.json()["toothConfirmed"] is True


async def test_an_impossible_tooth_number_is_rejected(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    _, study = await _patient_with_study(authed_client, radiograph_bytes)
    finding = study["findings"][0]

    response = await authed_client.put(
        f"/api/v1/studies/{study['publicId']}/findings/{finding['id']}/tooth",
        json={"toothNumber": 59},
    )
    assert response.status_code == 422, response.text


async def test_the_report_groups_findings_by_tooth(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    _, study = await _patient_with_study(authed_client, radiograph_bytes)

    response = await authed_client.get(f"/api/v1/studies/{study['publicId']}/report")
    assert response.status_code == 200, response.text
    report = response.json()

    assert len(report["chart"]) == 32
    assert report["summary"]
    assert report["disclaimer"]
    assert report["affectedTeeth"] == len(report["teeth"])
    for group in report["teeth"]:
        assert group["toothName"]
        assert all(f["toothNumber"] == group["toothNumber"] for f in group["findings"])


async def test_rejected_findings_leave_the_report(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    _, study = await _patient_with_study(authed_client, radiograph_bytes)
    finding = study["findings"][0]

    before = (await authed_client.get(f"/api/v1/studies/{study['publicId']}/report")).json()
    await authed_client.patch(
        f"/api/v1/studies/{study['publicId']}/findings/{finding['id']}",
        json={"review": "rejected"},
    )
    after = (await authed_client.get(f"/api/v1/studies/{study['publicId']}/report")).json()

    assert after["findingCount"] == before["findingCount"] - 1


async def test_a_plan_can_be_drafted_from_a_study(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id, study = await _patient_with_study(authed_client, radiograph_bytes)

    response = await authed_client.post(
        "/api/v1/treatment/plans/propose", json={"studyPublicId": study["publicId"]}
    )
    assert response.status_code == 200, response.text
    plan = response.json()

    assert plan["patientId"] == patient_id
    assert plan["status"] == "draft"
    assert plan["items"], "the protocol table produced nothing"
    for item in plan["items"]:
        assert item["status"] == "proposed"
        assert item["sourceStudyPublicId"] == study["publicId"]
        assert item["procedureLabel"]


async def test_drafting_twice_does_not_duplicate_steps(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    _, study = await _patient_with_study(authed_client, radiograph_bytes)

    first = await authed_client.post(
        "/api/v1/treatment/plans/propose", json={"studyPublicId": study["publicId"]}
    )
    second = await authed_client.post(
        "/api/v1/treatment/plans/propose", json={"studyPublicId": study["publicId"]}
    )
    assert len(second.json()["items"]) == len(first.json()["items"])


async def test_a_study_with_no_patient_cannot_start_a_plan(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    uploaded = await authed_client.post(
        "/api/v1/studies", files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")}
    )
    assert uploaded.status_code == 201

    response = await authed_client.post(
        "/api/v1/treatment/plans/propose",
        json={"studyPublicId": uploaded.json()["publicId"]},
    )
    assert response.status_code == 422, response.text


async def test_a_step_carries_its_estimate_from_the_protocol(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id, _ = await _patient_with_study(authed_client, radiograph_bytes)
    created = await authed_client.post("/api/v1/treatment/plans", json={"patientId": patient_id})
    plan_id = created.json()["publicId"]

    response = await authed_client.post(
        f"/api/v1/treatment/plans/{plan_id}/items",
        json={"procedureCode": "root_canal", "toothNumber": 36},
    )
    assert response.status_code == 201, response.text
    item = response.json()

    procedure = protocols.by_code("root_canal")
    assert procedure is not None
    assert item["estimatedVisits"] == procedure.visits
    assert item["estimatedMinutes"] == procedure.minutes
    assert item["priority"] == procedure.priority.value
    assert item["toothName"]


async def test_completing_a_step_stamps_it_and_leaves_the_open_count(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id, _ = await _patient_with_study(authed_client, radiograph_bytes)
    created = await authed_client.post("/api/v1/treatment/plans", json={"patientId": patient_id})
    plan_id = created.json()["publicId"]
    item = (
        await authed_client.post(
            f"/api/v1/treatment/plans/{plan_id}/items",
            json={"procedureCode": "caries_restoration", "toothNumber": 16},
        )
    ).json()

    response = await authed_client.patch(
        f"/api/v1/treatment/plans/{plan_id}/items/{item['id']}",
        json={
            "status": "done",
            "toothNumber": 16,
            "scheduledFor": None,
            "estimatedVisits": 1,
            "estimatedMinutes": 60,
            "notes": None,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["completedAt"] is not None

    plan = (await authed_client.get(f"/api/v1/treatment/plans/{plan_id}")).json()
    assert plan["doneCount"] == 1
    assert plan["openCount"] == 0


async def test_an_unknown_procedure_is_rejected(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id, _ = await _patient_with_study(authed_client, radiograph_bytes)
    created = await authed_client.post("/api/v1/treatment/plans", json={"patientId": patient_id})

    response = await authed_client.post(
        f"/api/v1/treatment/plans/{created.json()['publicId']}/items",
        json={"procedureCode": "teleportation", "toothNumber": None},
    )
    assert response.status_code == 422, response.text


async def test_plans_are_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    alpha, beta = two_clinics
    patient = await alpha.post("/api/v1/patients", json={"fullName": "Иванов Иван"})
    plan = await alpha.post("/api/v1/treatment/plans", json={"patientId": patient.json()["id"]})
    plan_id = plan.json()["publicId"]

    assert (await beta.get(f"/api/v1/treatment/plans/{plan_id}")).status_code == 404


async def test_the_patient_overview_merges_every_record_into_one_timeline(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    from tests.conftest import binary_stl_bytes

    patient_id, study = await _patient_with_study(authed_client, radiograph_bytes)
    await authed_client.post(
        "/api/v1/scans",
        files={"file": ("arch.stl", binary_stl_bytes(), "application/octet-stream")},
        data={"patient_id": str(patient_id)},
    )
    await authed_client.post(
        "/api/v1/treatment/plans/propose", json={"studyPublicId": study["publicId"]}
    )

    response = await authed_client.get(f"/api/v1/patients/{patient_id}/overview")
    assert response.status_code == 200, response.text
    overview = response.json()

    kinds = {entry["kind"] for entry in overview["timeline"]}
    assert {"patient_created", "study", "scan", "plan_item"} <= kinds
    assert overview["patient"]["scanCount"] == 1
    assert overview["patient"]["studyCount"] == 1
    assert overview["patient"]["openPlanItems"] > 0

    stamps = [entry["at"] for entry in overview["timeline"]]
    assert stamps == sorted(stamps, reverse=True)
