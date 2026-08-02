"""Comment threads, review assignments and who gets told about them."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dentist_ai.api.v1 import collaboration
from dentist_ai.db.models import Notification, Organization, User, UserRole
from tests.conftest import bootstrap_csrf

API_PREFIX = "/api/v1"
COMMENTS = f"{API_PREFIX}/collaboration/comments"
ASSIGNMENTS = f"{API_PREFIX}/collaboration/assignments"
COLLEAGUE_EMAIL = "kollega@clinic.kz"
COLLEAGUE_PASSWORD = "second-correct-horse-battery"


def app_of(client: AsyncClient) -> FastAPI:
    """The application an in-process client is driving."""
    transport = client._transport
    assert isinstance(transport, ASGITransport)
    app = transport.app
    assert isinstance(app, FastAPI)
    return app


def mount(app: FastAPI) -> None:
    """Register the collaboration router on an app that may already have it.

    ``api/v1/router.py`` is where this belongs and is owned elsewhere; doing it
    here keeps the suite runnable before that lands, and is a no-op after.
    """
    prefix = f"{API_PREFIX}{collaboration.router.prefix}"
    if any(getattr(route, "path", "").startswith(prefix) for route in app.routes):
        return
    app.include_router(collaboration.router, prefix=API_PREFIX)


async def add_member(app: FastAPI, *, email: str, full_name: str) -> int:
    """Put a second user into the clinic that already exists.

    Registration always creates a new organisation and there is no invite
    endpoint yet, so this one row is written directly. Everything the tests
    then assert still goes through the real HTTP stack.
    """
    factory: async_sessionmaker[AsyncSession] = app.state.session_factory
    async with factory() as session:
        organization_id = await session.scalar(select(Organization.id).order_by(Organization.id))
        assert organization_id is not None
        member = User(
            organization_id=organization_id,
            email=email,
            full_name=full_name,
            password_hash=app.state.passwords.hash(COLLEAGUE_PASSWORD),
            role=UserRole.DENTIST,
        )
        session.add(member)
        await session.commit()
        return member.id


async def notifications_for(app: FastAPI, user_id: int) -> list[Notification]:
    """Read the notification centre directly: it has no endpoint of its own."""
    factory: async_sessionmaker[AsyncSession] = app.state.session_factory
    async with factory() as session:
        rows = await session.scalars(
            select(Notification)
            .where(Notification.user_id == user_id)
            .order_by(Notification.id)
        )
        return list(rows.all())


async def user_id_of(client: AsyncClient) -> int:
    response = await client.get(f"{API_PREFIX}/auth/me")
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


async def create_patient(client: AsyncClient, name: str = "Иванов Иван") -> str:
    response = await client.post(f"{API_PREFIX}/patients", json={"fullName": name})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def post_comment(
    client: AsyncClient,
    *,
    resource_id: str,
    body: str,
    resource_type: str = "patient",
    parent_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "resourceType": resource_type,
        "resourceId": resource_id,
        "body": body,
    }
    if parent_id is not None:
        payload["parentId"] = parent_id
    response = await client.post(COMMENTS, json=payload)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


async def read_thread(
    client: AsyncClient, *, resource_id: str, resource_type: str = "patient"
) -> dict[str, Any]:
    response = await client.get(
        COMMENTS, params={"resourceType": resource_type, "resourceId": resource_id}
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


@pytest.fixture
def clinic(authed_client: AsyncClient) -> AsyncClient:
    mount(app_of(authed_client))
    return authed_client


@pytest.fixture
async def colleague(clinic: AsyncClient) -> AsyncIterator[AsyncClient]:
    """A second signed-in member of the same clinic."""
    app = app_of(clinic)
    await add_member(app, email=COLLEAGUE_EMAIL, full_name="Пётр Коллегов")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        headers={"Origin": "http://testserver", "Host": "testserver"},
    ) as client:
        token = await bootstrap_csrf(client)
        response = await client.post(
            f"{API_PREFIX}/auth/login",
            json={"email": COLLEAGUE_EMAIL, "password": COLLEAGUE_PASSWORD},
            headers={"X-CSRF-Token": token},
        )
        assert response.status_code == 200, response.text
        client.headers["X-CSRF-Token"] = response.json()["csrfToken"]
        yield client


# --------------------------------------------------------------------------
# Threads
# --------------------------------------------------------------------------
async def test_a_thread_returns_replies_nested_under_their_root(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    root = await post_comment(clinic, resource_id=patient_id, body="Проверить 36 зуб.")
    await post_comment(clinic, resource_id=patient_id, body="Согласен.", parent_id=root["id"])
    await post_comment(clinic, resource_id=patient_id, body="Отдельный вопрос.")

    thread = await read_thread(clinic, resource_id=patient_id)
    assert [comment["body"] for comment in thread["comments"]] == [
        "Проверить 36 зуб.",
        "Отдельный вопрос.",
    ]
    assert [reply["body"] for reply in thread["comments"][0]["replies"]] == ["Согласен."]
    # Roots and replies are both counted, so a busy thread reads as busy.
    assert thread["totalCount"] == 3
    assert thread["unresolvedCount"] == 2


async def test_answering_a_reply_joins_the_root_instead_of_nesting_deeper(
    clinic: AsyncClient,
) -> None:
    patient_id = await create_patient(clinic)
    root = await post_comment(clinic, resource_id=patient_id, body="Корень.")
    reply = await post_comment(
        clinic, resource_id=patient_id, body="Ответ.", parent_id=root["id"]
    )

    nested = await post_comment(
        clinic, resource_id=patient_id, body="Ответ на ответ.", parent_id=reply["id"]
    )
    assert nested["parentId"] == root["id"]

    thread = await read_thread(clinic, resource_id=patient_id)
    assert len(thread["comments"]) == 1
    assert len(thread["comments"][0]["replies"]) == 2


async def test_an_unknown_resource_type_is_refused(clinic: AsyncClient) -> None:
    """A free-string column plus no check would let a client file comments
    against something no screen can ever render."""
    response = await clinic.post(
        COMMENTS,
        json={"resourceType": "admin", "resourceId": "42", "body": "Тест"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_failed"
    assert "resourceType" in response.json()["errors"]


async def test_a_reply_to_another_resource_is_refused(clinic: AsyncClient) -> None:
    first = await create_patient(clinic)
    second = await create_patient(clinic, "Петров Пётр")
    root = await post_comment(clinic, resource_id=first, body="Корень.")

    response = await clinic.post(
        COMMENTS,
        json={
            "resourceType": "patient",
            "resourceId": second,
            "body": "Не туда.",
            "parentId": root["id"],
        },
    )
    assert response.status_code == 422


async def test_resolving_records_who_closed_the_question(clinic: AsyncClient) -> None:
    patient_id = await create_patient(clinic)
    comment = await post_comment(clinic, resource_id=patient_id, body="Вопрос.")
    me = await user_id_of(clinic)

    resolved = await clinic.post(f"{COMMENTS}/{comment['id']}/resolve")
    assert resolved.status_code == 200
    assert resolved.json()["isResolved"] is True
    assert resolved.json()["resolvedById"] == me

    thread = await read_thread(clinic, resource_id=patient_id)
    assert thread["unresolvedCount"] == 0


async def test_a_colleague_cannot_delete_someone_elses_comment(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    patient_id = await create_patient(clinic)
    theirs = await post_comment(colleague, resource_id=patient_id, body="Моя реплика.")
    mine = await post_comment(clinic, resource_id=patient_id, body="Реплика владельца.")

    assert (await colleague.delete(f"{COMMENTS}/{mine['id']}")).status_code == 403
    assert (await colleague.delete(f"{COMMENTS}/{theirs['id']}")).status_code == 200
    # The owner may clear anything in their own clinic.
    assert (await clinic.delete(f"{COMMENTS}/{mine['id']}")).status_code == 200
    assert (await read_thread(clinic, resource_id=patient_id))["comments"] == []


# --------------------------------------------------------------------------
# Notifications
# --------------------------------------------------------------------------
async def test_a_reply_notifies_the_parents_author_and_never_its_own(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    app = app_of(clinic)
    owner_id = await user_id_of(clinic)
    colleague_id = await user_id_of(colleague)

    patient_id = await create_patient(clinic)
    root = await post_comment(clinic, resource_id=patient_id, body="Что скажете о 36?")
    await post_comment(
        colleague, resource_id=patient_id, body="Похоже на периодонтит.", parent_id=root["id"]
    )

    delivered = await notifications_for(app, owner_id)
    assert [item.kind.value for item in delivered] == ["comment_added"]
    assert delivered[0].title == "Ответ в обсуждении"
    assert delivered[0].body is not None
    assert "Пётр Коллегов" in delivered[0].body

    # The person who just typed it does not need telling.
    assert await notifications_for(app, colleague_id) == []


async def test_commenting_on_someone_elses_upload_notifies_its_author(
    clinic: AsyncClient, colleague: AsyncClient, radiograph_bytes: bytes
) -> None:
    """The uploader is a participant even before they have said anything."""
    app = app_of(clinic)
    owner_id = await user_id_of(clinic)

    upload = await clinic.post(
        f"{API_PREFIX}/studies", files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")}
    )
    assert upload.status_code == 201
    public_id = upload.json()["publicId"]

    await post_comment(
        colleague, resource_type="study", resource_id=public_id, body="Пересмотрите 46."
    )

    delivered = await notifications_for(app, owner_id)
    assert [item.kind.value for item in delivered] == ["comment_added"]
    assert delivered[0].href == f"/app/studies/{public_id}"


async def test_a_comment_with_nobody_else_involved_notifies_nobody(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    app = app_of(clinic)
    patient_id = await create_patient(clinic)
    await post_comment(clinic, resource_id=patient_id, body="Заметка для себя.")

    assert await notifications_for(app, await user_id_of(clinic)) == []
    assert await notifications_for(app, await user_id_of(colleague)) == []


# --------------------------------------------------------------------------
# Review assignments
# --------------------------------------------------------------------------
async def test_assigning_a_case_notifies_the_named_colleague(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    app = app_of(clinic)
    colleague_id = await user_id_of(colleague)
    patient_id = await create_patient(clinic)

    response = await clinic.post(
        ASSIGNMENTS,
        json={
            "resourceType": "patient",
            "resourceId": patient_id,
            "assigneeId": colleague_id,
            "dueOn": "2026-09-01",
            "note": "Нужно второе мнение.",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["statusLabel"] == "Ожидает"
    assert body["assigneeName"] == "Пётр Коллегов"
    assert body["isOpen"] is True

    delivered = await notifications_for(app, colleague_id)
    assert [item.kind.value for item in delivered] == ["review_assigned"]
    assert await notifications_for(app, await user_id_of(clinic)) == []


async def test_an_open_assignment_shows_up_in_the_assignees_own_list(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    colleague_id = await user_id_of(colleague)
    patient_id = await create_patient(clinic)
    await clinic.post(
        ASSIGNMENTS,
        json={
            "resourceType": "patient",
            "resourceId": patient_id,
            "assigneeId": colleague_id,
        },
    )

    mine = await colleague.get(f"{ASSIGNMENTS}/mine")
    assert mine.status_code == 200
    assert len(mine.json()) == 1
    # The assigner is not the one who owes an answer.
    assert (await clinic.get(f"{ASSIGNMENTS}/mine")).json() == []


async def test_only_the_assignee_may_accept_and_complete(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    colleague_id = await user_id_of(colleague)
    patient_id = await create_patient(clinic)
    assignment_id = (
        await clinic.post(
            ASSIGNMENTS,
            json={
                "resourceType": "patient",
                "resourceId": patient_id,
                "assigneeId": colleague_id,
            },
        )
    ).json()["id"]

    refused = await clinic.patch(f"{ASSIGNMENTS}/{assignment_id}", json={"status": "accepted"})
    assert refused.status_code == 403

    accepted = await colleague.patch(
        f"{ASSIGNMENTS}/{assignment_id}", json={"status": "accepted"}
    )
    assert accepted.status_code == 200
    assert accepted.json()["completedAt"] is None

    completed = await colleague.patch(
        f"{ASSIGNMENTS}/{assignment_id}", json={"status": "completed"}
    )
    assert completed.json()["completedAt"] is not None
    assert completed.json()["isOpen"] is False
    assert (await colleague.get(f"{ASSIGNMENTS}/mine")).json() == []


async def test_the_assigner_may_withdraw_but_not_reopen(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    colleague_id = await user_id_of(colleague)
    patient_id = await create_patient(clinic)
    assignment_id = (
        await clinic.post(
            ASSIGNMENTS,
            json={
                "resourceType": "patient",
                "resourceId": patient_id,
                "assigneeId": colleague_id,
            },
        )
    ).json()["id"]

    withdrawn = await clinic.patch(f"{ASSIGNMENTS}/{assignment_id}", json={"status": "declined"})
    assert withdrawn.status_code == 200
    assert withdrawn.json()["status"] == "declined"

    reopened = await colleague.patch(f"{ASSIGNMENTS}/{assignment_id}", json={"status": "pending"})
    assert reopened.status_code == 422


async def test_assigning_the_same_open_case_twice_is_refused(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    """Otherwise a double click puts a second request in the colleague's list
    and a second row in their notification centre."""
    colleague_id = await user_id_of(colleague)
    patient_id = await create_patient(clinic)
    payload = {
        "resourceType": "patient",
        "resourceId": patient_id,
        "assigneeId": colleague_id,
    }

    assert (await clinic.post(ASSIGNMENTS, json=payload)).status_code == 201
    duplicate = await clinic.post(ASSIGNMENTS, json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "conflict"


async def test_assignments_list_for_one_resource(
    clinic: AsyncClient, colleague: AsyncClient
) -> None:
    colleague_id = await user_id_of(colleague)
    patient_id = await create_patient(clinic)
    other_id = await create_patient(clinic, "Петров Пётр")
    await clinic.post(
        ASSIGNMENTS,
        json={
            "resourceType": "patient",
            "resourceId": patient_id,
            "assigneeId": colleague_id,
        },
    )

    listed = await clinic.get(
        ASSIGNMENTS, params={"resourceType": "patient", "resourceId": patient_id}
    )
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["assignedByName"] == "Айгуль Сагиндикова"

    empty = await clinic.get(
        ASSIGNMENTS, params={"resourceType": "patient", "resourceId": other_id}
    )
    assert empty.json() == []


# --------------------------------------------------------------------------
# Tenancy and authentication
# --------------------------------------------------------------------------
async def test_collaboration_is_invisible_across_clinics(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    alpha, beta = two_clinics
    mount(app_of(alpha))

    patient_id = await create_patient(alpha)
    comment = await post_comment(alpha, resource_id=patient_id, body="Только для нас.")
    assignment_id = (
        await alpha.post(
            ASSIGNMENTS,
            json={
                "resourceType": "patient",
                "resourceId": patient_id,
                "assigneeId": await user_id_of(alpha),
            },
        )
    ).json()["id"]

    assert (await read_thread(beta, resource_id=patient_id))["comments"] == []
    assert (
        await beta.get(ASSIGNMENTS, params={"resourceType": "patient", "resourceId": patient_id})
    ).json() == []

    # 404, not 403: confirming existence would itself leak information.
    assert (await beta.post(f"{COMMENTS}/{comment['id']}/resolve")).status_code == 404
    assert (await beta.delete(f"{COMMENTS}/{comment['id']}")).status_code == 404
    assert (
        await beta.patch(f"{ASSIGNMENTS}/{assignment_id}", json={"status": "accepted"})
    ).status_code == 404
    # A colleague in another clinic is not a colleague.
    assert (
        await beta.post(
            ASSIGNMENTS,
            json={
                "resourceType": "patient",
                "resourceId": patient_id,
                "assigneeId": await user_id_of(alpha),
            },
        )
    ).status_code == 404


async def test_collaboration_requires_authentication(client: AsyncClient) -> None:
    mount(app_of(client))
    response = await client.get(
        COMMENTS, params={"resourceType": "patient", "resourceId": "1"}
    )
    assert response.status_code == 401
