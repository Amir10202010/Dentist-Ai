"""Treatment planning: the pure planner, and the service that persists it.

The planner is tested directly against hand-built finding lists rather than
through the API, because that is what it is for — it takes no session and its
whole value is being readable and checkable without one.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from scripts.synthetic_cbct import build_preset

from dentist_ai.clinical.treatment_planner import (
    FindingSource,
    PlannedFinding,
    propose,
)
from dentist_ai.db.models import PlanComplexity, TreatmentApproach


def finding(
    class_key: str,
    *,
    source: FindingSource = FindingSource.CBCT,
    tooth: int | None = None,
    confidence: float = 0.8,
    confirmed: bool = False,
) -> PlannedFinding:
    return PlannedFinding(
        class_key=class_key,
        source=source,
        confidence=confidence,
        tooth_number=tooth,
        confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------
def test_anatomy_alone_produces_no_plan() -> None:
    """The canal and an existing implant are records, not tasks."""
    assert propose([finding("mandibular_canal"), finding("implant", tooth=46)]) is None


def test_a_sound_root_filling_does_not_propose_retreatment() -> None:
    """Retreatment is indicated by the lesion beside it, never by the filling.

    Proposing it from the filling alone would re-treat every sound endodontic
    case in the chart.
    """
    assert propose([finding("root_canal_filling", tooth=36)]) is None


def test_a_lesion_produces_the_endodontic_pathway() -> None:
    plan = propose([finding("apical_lesion", tooth=46)])
    assert plan is not None
    standard = next(
        item for item in plan.options if item.approach is TreatmentApproach.STANDARD
    )
    assert "root_canal" in standard.procedure_codes


def test_options_are_ordered_by_scope_and_never_duplicated() -> None:
    plan = propose(
        [
            finding("apical_lesion", tooth=46),
            finding("caries", source=FindingSource.RADIOGRAPH, tooth=26),
            finding("missing_tooth", tooth=36),
            finding("attrition", source=FindingSource.RADIOGRAPH, tooth=16),
        ]
    )
    assert plan is not None

    visits = [item.visits for item in plan.options]
    assert visits == sorted(visits), "a wider scope cannot mean fewer appointments"

    signatures = {
        tuple((step.code, step.tooth_number) for step in item.steps)
        for item in plan.options
    }
    assert len(signatures) == len(plan.options), "identical options must be collapsed"


def test_sequencing_puts_disease_control_before_restoration() -> None:
    """A restoration placed before the infection is treated is the wrong order."""
    plan = propose(
        [
            finding("apical_lesion", tooth=46),
            finding("caries", source=FindingSource.RADIOGRAPH, tooth=26),
        ]
    )
    assert plan is not None
    codes = [step.code for step in plan.options[-1].steps]
    assert codes.index("root_canal") < codes.index("caries_restoration")


def test_calendar_time_exceeds_chair_time_when_a_site_must_heal() -> None:
    """An implant case is a couple of hours of work over several months."""
    plan = propose([finding("missing_tooth", tooth=36)])
    assert plan is not None
    option = plan.options[-1]
    chair_time_weeks = option.minutes / 60 / 40
    assert option.weeks > chair_time_weeks * 2


def test_risk_comes_from_the_combination_not_the_finding() -> None:
    """Neither the canal nor an extraction warns on its own; together they do."""
    canal_only = propose([finding("mandibular_canal"), finding("apical_lesion", tooth=46)])
    assert canal_only is not None
    assert not any("нерв" in item for item in canal_only.risks)

    together = propose(
        [
            finding("mandibular_canal"),
            finding("impacted_third_molar", tooth=48),
        ]
    )
    assert together is not None
    assert any("нерв" in item for item in together.risks)


def test_a_suspicious_mass_defers_elective_treatment() -> None:
    plan = propose([finding("suspicious_mass"), finding("caries", tooth=26)])
    assert plan is not None
    assert any("специалист" in item for item in plan.risks)


def test_a_low_quality_scan_adds_a_caveat_without_removing_work() -> None:
    findings = [finding("apical_lesion", tooth=46)]
    clean = propose(findings, quality_score=0.95)
    poor = propose(findings, quality_score=0.3)

    assert clean is not None
    assert poor is not None
    assert len(poor.risks) > len(clean.risks)
    assert len(poor.options[0].steps) == len(clean.options[0].steps)


def test_complexity_rises_with_scope() -> None:
    simple = propose([finding("caries", source=FindingSource.RADIOGRAPH, tooth=26)])
    broad = propose(
        [
            finding("apical_lesion", tooth=46),
            finding("missing_tooth", tooth=36),
            finding("periodontal_disease"),
            finding("caries", source=FindingSource.RADIOGRAPH, tooth=26),
        ]
    )
    assert simple is not None
    assert broad is not None
    assert simple.complexity is PlanComplexity.SIMPLE
    assert broad.complexity in (PlanComplexity.COMPLEX, PlanComplexity.ADVANCED)


def test_findings_that_imply_nothing_are_named_rather_than_dropped() -> None:
    """A short plan beside a long finding list otherwise reads as a failure."""
    plan = propose([finding("apical_lesion", tooth=46), finding("mandibular_canal")])
    assert plan is not None
    assert any("канал" in item.lower() for item in plan.unaddressed)


def test_follow_up_interval_tracks_the_worst_finding() -> None:
    urgent = propose([finding("cyst")])
    routine = propose([finding("attrition", source=FindingSource.RADIOGRAPH, tooth=16)])
    assert urgent is not None
    assert routine is not None
    assert "3 мес" in urgent.follow_up
    assert "12 мес" in routine.follow_up


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
async def setup_case(client: AsyncClient, preset: str = "periapical") -> tuple[int, dict[str, Any]]:
    patient = await client.post("/api/v1/patients", json={"fullName": "Иванов Иван Петрович"})
    patient_id = int(patient.json()["id"])
    volume = await client.post(
        "/api/v1/volumes",
        files={"file": (f"{preset}.nii", build_preset(preset, seed=8), "application/octet-stream")},
        data={"patient_id": str(patient_id), "field_of_view": "both_jaws"},
    )
    assert volume.status_code == 201, volume.text
    return patient_id, dict(volume.json())


async def test_generated_plan_is_a_draft_with_options_and_no_items(
    authed_client: AsyncClient,
) -> None:
    """Generating must never put work into the schedule on its own."""
    patient_id, volume = await setup_case(authed_client)

    response = await authed_client.post(
        "/api/v1/planning/generate",
        json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
    )
    assert response.status_code == 201, response.text
    plan = response.json()

    assert plan["origin"] == "generated"
    assert plan["status"] == "draft"
    assert plan["items"] == []
    assert len(plan["options"]) >= 2
    assert plan["rationale"]
    assert plan["followUp"]


async def test_accepting_an_option_creates_its_steps_with_tooth_numbers(
    authed_client: AsyncClient,
) -> None:
    """Two teeth needing the same procedure are two line items, not one."""
    patient_id, volume = await setup_case(authed_client)
    plan = (
        await authed_client.post(
            "/api/v1/planning/generate",
            json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
        )
    ).json()

    response = await authed_client.post(
        f"/api/v1/planning/{plan['publicId']}/accept", json={"approach": "standard"}
    )
    assert response.status_code == 200, response.text
    accepted = response.json()

    assert accepted["status"] == "active"
    assert accepted["items"], "accepting must create the option's steps"
    assert [item for item in accepted["options"] if item["isSelected"]]

    root_canals = [
        item for item in accepted["items"] if item["procedureCode"] == "root_canal"
    ]
    if len(root_canals) > 1:
        teeth = {item["toothNumber"] for item in root_canals}
        assert len(teeth) == len(root_canals), "each tooth is its own appointment"


async def test_re_accepting_is_refused_rather_than_replacing_the_schedule(
    authed_client: AsyncClient,
) -> None:
    patient_id, volume = await setup_case(authed_client)
    plan = (
        await authed_client.post(
            "/api/v1/planning/generate",
            json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
        )
    ).json()
    await authed_client.post(
        f"/api/v1/planning/{plan['publicId']}/accept", json={"approach": "standard"}
    )

    again = await authed_client.post(
        f"/api/v1/planning/{plan['publicId']}/accept", json={"approach": "conservative"}
    )
    assert again.status_code == 409
    assert again.json()["code"] == "conflict"


async def test_a_scan_with_no_actionable_findings_declines_to_invent_a_plan(
    authed_client: AsyncClient,
) -> None:
    patient_id, volume = await setup_case(authed_client, preset="healthy")

    response = await authed_client.post(
        "/api/v1/planning/generate",
        json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_failed"


async def test_rejected_findings_do_not_drive_treatment(authed_client: AsyncClient) -> None:
    """A rejected finding is a clinician overruling the model, not a task."""
    patient_id, volume = await setup_case(authed_client)
    for item in volume["findings"]:
        await authed_client.patch(
            f"/api/v1/volumes/{volume['publicId']}/findings/{item['id']}",
            json={"review": "rejected"},
        )

    response = await authed_client.post(
        "/api/v1/planning/generate",
        json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
    )
    assert response.status_code == 422


async def test_another_clinic_cannot_plan_for_a_patient_it_cannot_see(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    first, second = two_clinics
    patient_id, volume = await setup_case(first)

    response = await second.post(
        "/api/v1/planning/generate",
        json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
    )
    assert response.status_code == 404


async def test_another_clinic_cannot_accept_an_option(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    first, second = two_clinics
    patient_id, volume = await setup_case(first)
    plan = (
        await first.post(
            "/api/v1/planning/generate",
            json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
        )
    ).json()

    response = await second.post(
        f"/api/v1/planning/{plan['publicId']}/accept", json={"approach": "standard"}
    )
    assert response.status_code == 404


@pytest.mark.parametrize("approach", ["conservative", "standard", "comprehensive"])
async def test_every_offered_approach_can_be_accepted(
    authed_client: AsyncClient, approach: str
) -> None:
    patient_id, volume = await setup_case(authed_client)
    plan = (
        await authed_client.post(
            "/api/v1/planning/generate",
            json={"patientId": patient_id, "volumePublicId": volume["publicId"]},
        )
    ).json()

    offered = {item["approach"] for item in plan["options"]}
    response = await authed_client.post(
        f"/api/v1/planning/{plan['publicId']}/accept", json={"approach": approach}
    )
    if approach in offered:
        assert response.status_code == 200, response.text
    else:
        # An approach that collapsed into a narrower one is not on this plan.
        assert response.status_code == 404
