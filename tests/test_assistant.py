"""The case assistant: routing, grounding, and what it refuses to do.

The routing tests are the important half. An assistant that answers the wrong
question confidently is worse than one that admits it did not understand, so
every mis-route found in review has a case here.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from scripts.synthetic_cbct import build_preset

from dentist_ai.services.assistant import Intent, _address, route


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Кратко опиши этот снимок", Intent.SUMMARISE),
        ("Что на снимке?", Intent.SUMMARISE),
        ("Почему AI это нашёл?", Intent.WHY_DETECTED),
        ("На основании чего сделан вывод?", Intent.WHY_DETECTED),
        ("Объясни периапикальное поражение", Intent.EXPLAIN_FINDING),
        ("Что такое киста?", Intent.EXPLAIN_FINDING),
        ("Какие возможны варианты лечения?", Intent.TREATMENTS),
        ("Что проверить на приёме?", Intent.NEXT_CHECKS),
        ("Покажи похожие случаи", Intent.SIMILAR_CASES),
        ("Объясни простыми словами для пациента", Intent.PATIENT_FRIENDLY),
        ("Какие есть измерения?", Intent.MEASUREMENTS),
        ("Насколько надёжен этот снимок?", Intent.QUALITY),
        ("Какое качество исследования?", Intent.QUALITY),
    ],
)
def test_questions_route_to_their_intent(question: str, expected: Intent) -> None:
    assert route(question) is expected


def test_a_patient_facing_request_beats_the_generic_explain() -> None:
    """Both open with "объясни"; the one naming the patient is more specific."""
    assert route("Объясни простыми словами для пациента") is Intent.PATIENT_FRIENDLY
    assert route("Объясни находку") is Intent.EXPLAIN_FINDING


def test_matching_is_on_word_boundaries_not_substrings() -> None:
    """ "имплант" contains "план".

    A plain substring test routes a cost question to the treatment handler,
    which is the kind of wrong answer that is invisible in review and obvious
    to a user.
    """
    assert route("Сколько стоит имплант?") is Intent.CAPABILITIES
    assert route("Какой план лечения?") is Intent.TREATMENTS


def test_an_unrecognised_question_admits_it() -> None:
    assert route("Какая сегодня погода в Алматы?") is Intent.CAPABILITIES


@pytest.mark.parametrize(
    ("full_name", "expected"),
    [
        ("Иванов Иван Петрович", "Иван Петрович"),
        ("Петрова Мария", "Мария"),
        ("Асель", "Асель"),
        ("", "Пациент"),
    ],
)
def test_patients_are_addressed_by_given_name_not_surname(full_name: str, expected: str) -> None:
    """Records are surname-first, so the obvious first token is the surname."""
    assert _address(full_name) == expected


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
async def setup_case(client: AsyncClient, preset: str = "periapical") -> dict[str, Any]:
    patient = await client.post("/api/v1/patients", json={"fullName": "Иванов Иван Петрович"})
    patient_id = int(patient.json()["id"])
    volume = await client.post(
        "/api/v1/volumes",
        files={"file": (f"{preset}.nii", build_preset(preset, seed=8), "application/octet-stream")},
        data={"patient_id": str(patient_id), "field_of_view": "both_jaws"},
    )
    assert volume.status_code == 201, volume.text
    return dict(volume.json())


async def ask(client: AsyncClient, question: str, **kwargs: Any) -> dict[str, Any]:
    response = await client.post("/api/v1/assistant/ask", json={"question": question, **kwargs})
    assert response.status_code == 200, response.text
    return dict(response.json())


async def test_a_summary_reports_the_counts_that_are_actually_stored(
    authed_client: AsyncClient,
) -> None:
    volume = await setup_case(authed_client)
    result = await ask(authed_client, "Кратко опиши этот снимок", volumePublicId=volume["publicId"])

    body = result["answer"]["body"]
    assert str(volume["findingCount"]) in body
    assert result["answer"]["citations"], "an answer must name what it rests on"


async def test_why_answers_with_the_stored_rationale_and_the_stage(
    authed_client: AsyncClient,
) -> None:
    """ "Why" has to be answerable, or the finding is not reviewable."""
    volume = await setup_case(authed_client)
    result = await ask(authed_client, "Почему AI это нашёл?", volumePublicId=volume["publicId"])

    body = result["answer"]["body"]
    assert volume["findings"][0]["producedBy"] in body
    assert "%" in body, "the confidence belongs in the explanation"


async def test_an_unrecognised_question_lists_what_it_can_do(
    authed_client: AsyncClient,
) -> None:
    volume = await setup_case(authed_client)
    result = await ask(authed_client, "Сколько стоит имплант?", volumePublicId=volume["publicId"])

    assert result["answer"]["intent"] == "capabilities"
    assert result["answer"]["suggestions"]


async def test_the_patient_register_keeps_the_caveat(authed_client: AsyncClient) -> None:
    """Simplifying the wording must not simplify away the disclaimer."""
    volume = await setup_case(authed_client)
    result = await ask(
        authed_client, "Объясни простыми словами для пациента", volumePublicId=volume["publicId"]
    )

    body = result["answer"]["body"]
    assert "диагноз" in body.lower()


async def test_the_patient_register_does_not_repeat_itself(
    authed_client: AsyncClient,
) -> None:
    """The canal is found in several segments; the patient hears it once."""
    volume = await setup_case(authed_client)
    result = await ask(
        authed_client, "Объясни простыми словами для пациента", volumePublicId=volume["publicId"]
    )

    lines = [line for line in result["answer"]["body"].split("\n") if line.startswith("•")]
    assert len(lines) == len(set(lines))


async def test_a_rejected_finding_stops_being_explained_back(
    authed_client: AsyncClient,
) -> None:
    volume = await setup_case(authed_client)
    target = volume["findings"][0]
    await authed_client.patch(
        f"/api/v1/volumes/{volume['publicId']}/findings/{target['id']}",
        json={"review": "rejected"},
    )

    result = await ask(authed_client, "Кратко опиши этот снимок", volumePublicId=volume["publicId"])
    assert str(volume["findingCount"]) not in result["answer"]["body"].split("\n")[0]


async def test_a_conversation_keeps_its_turns(authed_client: AsyncClient) -> None:
    volume = await setup_case(authed_client)
    first = await ask(authed_client, "Кратко опиши этот снимок", volumePublicId=volume["publicId"])
    thread_id = first["threadPublicId"]

    await ask(authed_client, "Почему AI это нашёл?", threadPublicId=thread_id)

    history = await authed_client.get(f"/api/v1/assistant/threads/{thread_id}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [item["role"] for item in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert history.json()["title"] == "Кратко опиши этот снимок"


async def test_an_answer_without_a_scan_says_so_rather_than_guessing(
    authed_client: AsyncClient,
) -> None:
    patient = await authed_client.post("/api/v1/patients", json={"fullName": "Без снимков"})
    result = await ask(authed_client, "Кратко опиши этот снимок", patientId=patient.json()["id"])
    assert "не привязано" in result["answer"]["body"]


async def test_an_empty_question_is_refused(authed_client: AsyncClient) -> None:
    volume = await setup_case(authed_client)
    response = await authed_client.post(
        "/api/v1/assistant/ask",
        json={"question": "   ", "volumePublicId": volume["publicId"]},
    )
    assert response.status_code == 422


async def test_threads_are_private_to_the_user_who_started_them(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    first, second = two_clinics
    volume = await setup_case(first)
    started = await ask(first, "Кратко опиши этот снимок", volumePublicId=volume["publicId"])

    response = await second.get(f"/api/v1/assistant/threads/{started['threadPublicId']}")
    assert response.status_code == 404


async def test_another_clinic_cannot_open_a_thread_on_a_foreign_scan(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    first, second = two_clinics
    volume = await setup_case(first)

    response = await second.post(
        "/api/v1/assistant/ask",
        json={"question": "Кратко опиши этот снимок", "volumePublicId": volume["publicId"]},
    )
    assert response.status_code == 404
