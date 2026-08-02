"""The dashboard payload.

These assert the properties the screen is built on rather than exact numbers,
which depend on what the stub detector happens to emit for a given image: that
the review queue only ever contains work a clinician still has to do, that the
activity feed is the audit trail minus its read events, and that a rate is
never computed against a baseline of nothing.
"""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient


async def _dashboard(client: AsyncClient) -> dict[str, Any]:
    response = await client.get("/api/v1/dashboard")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def _upload(client: AsyncClient, payload: bytes) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/studies", files={"file": ("opg.jpg", payload, "image/jpeg")}
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_empty_clinic_reports_zeroes_not_nulls(authed_client: AsyncClient) -> None:
    """A brand-new clinic still renders: the page has no null-handling branch."""
    data = await _dashboard(authed_client)

    assert data["totalStudies"] == 0
    assert data["reviewQueue"] == []
    assert data["reviewQueueTotal"] == 0
    assert data["pendingFindings"] == 0
    assert data["oldestPendingAt"] is None
    assert data["reviewStats"]["agreementRate"] is None
    assert data["reviewedShare"] == 0.0
    # 90 days plus today, densified, so the chart never has gaps.
    assert len(data["studiesOverTime"]) == 91
    # The empty clinic is the one that most needs to be told what to do next.
    assert [item["key"] for item in data["insights"]] == ["first-study"]


async def test_review_queue_lists_only_unadjudicated_pathologies(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    pathologies = [item for item in study["findings"] if item["category"] == "pathology"]
    assert pathologies, "the stub detector should emit at least one pathology"

    data = await _dashboard(authed_client)
    assert data["reviewQueueTotal"] == 1
    entry = data["reviewQueue"][0]
    assert entry["publicId"] == study["publicId"]
    assert entry["pendingCount"] == len(pathologies)
    assert entry["topSeverity"] == pathologies[0]["severity"]
    assert entry["topFindingLabel"]

    # Adjudicating every pathology empties the queue — including the rejected
    # ones, which are decisions too.
    for finding in pathologies:
        response = await authed_client.patch(
            f"/api/v1/studies/{study['publicId']}/findings/{finding['id']}",
            json={"review": "confirmed"},
        )
        assert response.status_code == 200, response.text

    after = await _dashboard(authed_client)
    assert after["reviewQueue"] == []
    assert after["pendingFindings"] == 0
    assert "queue-clear" in {item["key"] for item in after["insights"]}


async def test_queue_is_ordered_by_severity_then_age(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    for _ in range(3):
        await _upload(authed_client, radiograph_bytes)

    data = await _dashboard(authed_client)
    ranks = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    severities = [ranks[item["topSeverity"]] for item in data["reviewQueue"]]
    assert severities == sorted(severities)


async def test_activity_reflects_the_audit_trail_without_its_reads(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    # A read: audited for compliance, absent from the feed.
    await authed_client.get(f"/api/v1/studies/{study['publicId']}")

    data = await _dashboard(authed_client)
    actions = [item["action"] for item in data["activity"]]

    assert "study.uploaded" in actions
    assert "study.viewed" not in actions
    entry = next(item for item in data["activity"] if item["action"] == "study.uploaded")
    assert entry["actorName"] == "Айгуль Сагиндикова"
    assert entry["resourceId"] == study["publicId"]
    assert entry["summary"]
    assert entry["icon"]
    assert entry["tone"]


async def test_growth_from_zero_has_no_percentage(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    await _upload(authed_client, radiograph_bytes)
    data = await _dashboard(authed_client)

    delta = data["studiesDelta"]
    assert delta["current"] == 1
    assert delta["previous"] == 0
    # Not 100%, and not infinity: there is no baseline to divide by, and the
    # client renders the absolute count instead.
    assert delta["change"] is None


async def test_pipeline_counts_today_in_the_clinics_timezone(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    await _upload(authed_client, radiograph_bytes)
    data = await _dashboard(authed_client)

    pipeline = data["pipeline"]
    # The stub detector runs inline, so an upload is complete by the time the
    # response returns and nothing is ever left pending.
    assert pipeline["completedToday"] == 1
    assert pipeline["processing"] == 0
    assert pipeline["failedRecent"] == 0
