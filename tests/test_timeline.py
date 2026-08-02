"""The merged patient history, and the notes and visits it owns."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dentist_ai.api.v1 import timeline
from dentist_ai.clinical import protocols
from dentist_ai.core.ids import generate_public_id
from dentist_ai.db.models import Patient, PlanItemStatus, TreatmentPlan, TreatmentPlanItem

API_PREFIX = "/api/v1"
#: A real entry from the protocol table, so the timeline renders the procedure
#: label a clinician would actually see rather than a made-up code.
PROCEDURE = protocols.PROCEDURES[0]


def app_of(client: AsyncClient) -> FastAPI:
    """The application an in-process client is driving."""
    transport = client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    return app


def mount(app: FastAPI) -> None:
    """Register the timeline router on an app that may already have it.

    ``api/v1/router.py`` is where this belongs and is owned elsewhere; doing it
    here keeps the suite runnable before that lands, and is a no-op after.
    """
    prefix = f"{API_PREFIX}{timeline.router.prefix}"
    if any(getattr(route, "path", "").startswith(prefix) for route in app.routes):
        return
    app.include_router(timeline.router, prefix=API_PREFIX)


def at(entry: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(entry["at"]))


async def create_patient(client: AsyncClient, name: str = "Иванов Иван") -> int:
    response = await client.post(f"{API_PREFIX}/patients", json={"fullName": name})
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


async def read_timeline(client: AsyncClient, patient_id: int) -> dict[str, Any]:
    response = await client.get(f"{API_PREFIX}/timeline/{patient_id}")
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


async def add_note(
    client: AsyncClient, patient_id: int, body: str, kind: str = "clinical"
) -> dict[str, Any]:
    response = await client.post(
        f"{API_PREFIX}/timeline/{patient_id}/notes", json={"kind": kind, "body": body}
    )
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


async def add_plan_step(app: FastAPI, patient_id: int, *, tooth_number: int | None = None) -> int:
    """Write a plan and one open step straight to the database.

    The treatment API would do the same thing; going around it keeps this file
    testing how the timeline *reads* plan steps rather than how another feature
    writes them.
    """
    factory: async_sessionmaker[AsyncSession] = app.state.session_factory
    async with factory() as session:
        organization_id = await session.scalar(
            select(Patient.organization_id).where(Patient.id == patient_id)
        )
        assert organization_id is not None
        plan = TreatmentPlan(
            public_id=generate_public_id(),
            organization_id=organization_id,
            patient_id=patient_id,
            title="План лечения",
        )
        session.add(plan)
        await session.flush()

        step = TreatmentPlanItem(
            plan_id=plan.id,
            procedure_code=PROCEDURE.code,
            tooth_number=tooth_number,
            priority=PROCEDURE.priority.value,
            estimated_visits=PROCEDURE.visits,
            estimated_minutes=PROCEDURE.minutes,
            status=PlanItemStatus.ACCEPTED,
        )
        session.add(step)
        await session.commit()
        return step.id


async def book(
    client: AsyncClient,
    patient_id: int,
    *,
    title: str,
    starts_at: datetime,
    duration_minutes: int = 60,
) -> dict[str, Any]:
    response = await client.post(
        f"{API_PREFIX}/timeline/{patient_id}/appointments",
        json={
            "title": title,
            "startsAt": starts_at.isoformat(),
            "durationMinutes": duration_minutes,
        },
    )
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture
def clinic(authed_client: AsyncClient) -> AsyncClient:
    mount(app_of(authed_client))
    return authed_client


# --------------------------------------------------------------------------
# The merge
# --------------------------------------------------------------------------
async def test_entries_from_different_tables_arrive_newest_first(
    clinic: AsyncClient, radiograph_bytes: bytes
) -> None:
    """Ordering has to hold across sources, not within one.

    Four tables contribute here — patients, studies, patient_notes and
    appointments — and the two visits are dated a year either side of now, so
    the assertion cannot be satisfied by whatever order the queries ran in.
    """
    patient_id = await create_patient(clinic)
    upload = await clinic.post(
        f"{API_PREFIX}/studies",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )
    assert upload.status_code == 201

    past = datetime.now(UTC) - timedelta(days=365)
    future = datetime.now(UTC) + timedelta(days=30)
    await book(clinic, patient_id, title="Первичный осмотр", starts_at=past)
    await book(clinic, patient_id, title="Контроль", starts_at=future)
    await add_note(clinic, patient_id, "Жалобы на боль справа.")

    entries = (await read_timeline(clinic, patient_id))["entries"]
    stamps = [at(entry) for entry in entries]
    assert stamps == sorted(stamps, reverse=True)

    assert entries[0]["kind"] == "appointment"
    assert entries[0]["title"] == "Контроль"
    assert entries[-1]["kind"] == "appointment"
    assert entries[-1]["title"] == "Первичный осмотр"
    # Everything written "now" sits between the two visits.
    assert {entry["kind"] for entry in entries[1:-1]} == {"patient_created", "study", "note"}


async def test_a_study_entry_carries_its_severity_and_a_plottable_metric(
    clinic: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id = await create_patient(clinic)
    upload = await clinic.post(
        f"{API_PREFIX}/studies",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )
    assert upload.status_code == 201
    attention = upload.json()["attentionCount"]

    entries = (await read_timeline(clinic, patient_id))["entries"]
    study = next(entry for entry in entries if entry["kind"] == "study")
    assert study["href"] == f"/app/studies/{upload.json()['publicId']}"
    assert study["metric"]["value"] == attention
    assert study["metric"]["label"] == "Требуют внимания"


async def test_the_summary_bounds_the_past_and_reports_the_next_visit(
    clinic: AsyncClient,
) -> None:
    """A booked visit is not activity: it must not become "last activity"."""
    patient_id = await create_patient(clinic)
    future = datetime.now(UTC) + timedelta(days=14)
    await book(clinic, patient_id, title="Контроль", starts_at=future)
    await add_note(clinic, patient_id, "План обсуждён.")

    summary = (await read_timeline(clinic, patient_id))["summary"]
    assert summary["total"] == 3
    assert at({"at": summary["lastAt"]}) < future
    assert at({"at": summary["nextAppointmentAt"]}) == future
    assert {row["kind"]: row["count"] for row in summary["counts"]} == {
        "patient_created": 1,
        "note": 1,
        "appointment": 1,
    }
    assert summary["counts"][0]["label"] == "Приёмы"


async def test_open_plan_steps_are_counted_in_the_summary(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    procedures = (await clinic.get(f"{API_PREFIX}/treatment/procedures")).json()
    plan = await clinic.post(
        f"{API_PREFIX}/treatment/plans", json={"patientId": patient_id, "title": "План"}
    )
    assert plan.status_code == 201
    added = await clinic.post(
        f"{API_PREFIX}/treatment/plans/{plan.json()['publicId']}/items",
        json={"procedureCode": procedures[0]["code"], "toothNumber": 36},
    )
    assert added.status_code == 201

    body = await read_timeline(clinic, patient_id)
    assert body["summary"]["openPlanItems"] == 1
    step = next(entry for entry in body["entries"] if entry["kind"] == "plan_item")
    assert step["title"].endswith(" · 36")
    assert step["metric"]["label"] == "Визитов"


async def test_the_summary_counts_more_than_the_page_shows(clinic: AsyncClient) -> None:
    """A truncated list must not make the header understate the history."""
    patient_id = await create_patient(clinic)
    for index in range(4):
        await add_note(clinic, patient_id, f"Заметка {index}")

    response = await clinic.get(f"{API_PREFIX}/timeline/{patient_id}", params={"limit": 2})
    body = response.json()
    assert len(body["entries"]) == 2
    assert body["summary"]["total"] == 5


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
async def test_a_note_appears_on_the_timeline_and_leaves_when_deleted(
    clinic: AsyncClient,
) -> None:
    patient_id = await create_patient(clinic)
    note = await add_note(clinic, patient_id, "Аллергия на лидокаин.", kind="follow_up")
    assert note["kindLabel"] == "Напоминание"
    assert note["authorName"] == "Айгуль Сагиндикова"

    entries = (await read_timeline(clinic, patient_id))["entries"]
    assert [entry["subtitle"] for entry in entries if entry["kind"] == "note"] == [
        "Аллергия на лидокаин."
    ]

    removed = await clinic.delete(f"{API_PREFIX}/timeline/{patient_id}/notes/{note['id']}")
    assert removed.status_code == 200
    assert (await read_timeline(clinic, patient_id))["summary"]["total"] == 1


async def test_an_empty_note_is_refused(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    response = await clinic.post(
        f"{API_PREFIX}/timeline/{patient_id}/notes", json={"body": "   "}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Appointments
# --------------------------------------------------------------------------
async def test_a_start_time_without_an_offset_is_refused(clinic: AsyncClient) -> None:
    """`UtcDateTime` reads a naive bind as UTC, so "14:00" from a clinic six
    hours ahead would silently book the evening."""
    patient_id = await create_patient(clinic)
    response = await clinic.post(
        f"{API_PREFIX}/timeline/{patient_id}/appointments",
        json={"title": "Осмотр", "startsAt": "2026-09-01T14:00:00"},
    )
    assert response.status_code == 422
    assert "startsAt" in response.json()["errors"]


async def test_a_visit_can_be_rescheduled_and_confirmed(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    original = datetime.now(UTC) + timedelta(days=7)
    appointment = await book(clinic, patient_id, title="Осмотр", starts_at=original)
    assert appointment["statusLabel"] == "Запланирован"
    assert appointment["isUpcoming"] is True

    moved = datetime.now(UTC) + timedelta(days=9)
    response = await clinic.patch(
        f"{API_PREFIX}/timeline/{patient_id}/appointments/{appointment['id']}",
        json={"startsAt": moved.isoformat(), "status": "confirmed"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "confirmed"
    assert at({"at": response.json()["startsAt"]}) == moved
    # An omitted field keeps what it had.
    assert response.json()["title"] == "Осмотр"
    assert response.json()["durationMinutes"] == 60


async def test_a_visit_cannot_realise_another_patients_plan_step(
    clinic: AsyncClient,
) -> None:
    theirs = await create_patient(clinic, "Петров Пётр")
    mine = await create_patient(clinic)
    procedures = (await clinic.get(f"{API_PREFIX}/treatment/procedures")).json()
    plan = await clinic.post(
        f"{API_PREFIX}/treatment/plans", json={"patientId": theirs, "title": "Чужой план"}
    )
    item = await clinic.post(
        f"{API_PREFIX}/treatment/plans/{plan.json()['publicId']}/items",
        json={"procedureCode": procedures[0]["code"]},
    )

    response = await clinic.post(
        f"{API_PREFIX}/timeline/{mine}/appointments",
        json={
            "title": "Осмотр",
            "startsAt": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "planItemId": item.json()["id"],
        },
    )
    assert response.status_code == 404


async def test_an_absurd_duration_is_refused(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    response = await clinic.post(
        f"{API_PREFIX}/timeline/{patient_id}/appointments",
        json={
            "title": "Марафон",
            "startsAt": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "durationMinutes": 5000,
        },
    )
    assert response.status_code == 422


async def test_a_deleted_visit_leaves_the_timeline(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    appointment = await book(
        clinic, patient_id, title="Осмотр", starts_at=datetime.now(UTC) + timedelta(days=3)
    )

    listed = await clinic.get(f"{API_PREFIX}/timeline/{patient_id}/appointments")
    assert [row["title"] for row in listed.json()] == ["Осмотр"]

    removed = await clinic.delete(
        f"{API_PREFIX}/timeline/{patient_id}/appointments/{appointment['id']}"
    )
    assert removed.status_code == 200
    assert (await read_timeline(clinic, patient_id))["summary"]["total"] == 1


# --------------------------------------------------------------------------
# Tenancy and authentication
# --------------------------------------------------------------------------
async def test_a_timeline_is_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    alpha, beta = two_clinics
    mount(app_of(alpha))

    patient_id = await create_patient(alpha)
    note = await add_note(alpha, patient_id, "Только для нас.")
    appointment = await book(
        alpha, patient_id, title="Осмотр", starts_at=datetime.now(UTC) + timedelta(days=2)
    )

    # 404, not 403: confirming existence would itself leak information.
    assert (await beta.get(f"{API_PREFIX}/timeline/{patient_id}")).status_code == 404
    assert (
        await beta.post(
            f"{API_PREFIX}/timeline/{patient_id}/notes", json={"body": "Взлом"}
        )
    ).status_code == 404
    assert (
        await beta.delete(f"{API_PREFIX}/timeline/{patient_id}/notes/{note['id']}")
    ).status_code == 404
    assert (
        await beta.patch(
            f"{API_PREFIX}/timeline/{patient_id}/appointments/{appointment['id']}",
            json={"status": "cancelled"},
        )
    ).status_code == 404
    assert (
        await beta.get(f"{API_PREFIX}/timeline/{patient_id}/appointments")
    ).status_code == 404


async def test_the_timeline_requires_authentication(client: AsyncClient) -> None:
    mount(app_of(client))
    assert (await client.get(f"{API_PREFIX}/timeline/1")).status_code == 401
