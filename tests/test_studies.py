"""Upload, inference, findings and authorisation on radiographs."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pytest
from httpx import AsyncClient
from PIL import Image

from dentist_ai.core.config import Settings


async def _upload(client: AsyncClient, payload: bytes, filename: str = "opg.jpg") -> dict[str, Any]:
    response = await client.post(
        "/api/v1/studies",
        files={"file": (filename, payload, "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_upload_returns_structured_findings(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await _upload(authed_client, radiograph_bytes)

    assert study["status"] == "completed"
    assert study["findingCount"] > 0
    assert study["modelVersion"] == "stub-1.0.0"

    finding = study["findings"][0]
    # The whole point of the rewrite: confidences and coordinates survive to
    # the client instead of being burned into a JPEG and discarded.
    assert 0.0 <= finding["confidence"] <= 1.0
    assert finding["label"]
    assert finding["category"] in {
        "pathology",
        "restoration",
        "orthodontic",
        "anatomy",
        "condition",
    }
    box = finding["box"]
    assert 0.0 <= box["x"] <= 1.0
    assert 0.0 < box["width"] <= 1.0
    assert box["x"] + box["width"] <= 1.0001


async def test_findings_are_sorted_by_triage_priority(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    ranks = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    severities = [ranks[f["severity"]] for f in study["findings"]]
    assert severities == sorted(severities)


async def test_upload_is_deterministic_for_identical_images(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    first = await _upload(authed_client, radiograph_bytes)
    second = await _upload(authed_client, radiograph_bytes)
    assert first["findingCount"] == second["findingCount"]
    assert first["publicId"] != second["publicId"]


async def test_identical_uploads_share_one_stored_blob(
    authed_client: AsyncClient, radiograph_bytes: bytes, settings: Settings
) -> None:
    """Content-addressed storage deduplicates automatically."""
    await _upload(authed_client, radiograph_bytes)
    await _upload(authed_client, radiograph_bytes)

    masters = list(settings.storage.resolved_root.rglob("*.master.jpg"))
    assert len(masters) == 1


async def test_timestamps_serialise_with_an_explicit_offset(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """Instants on the wire must be unambiguous.

    SQLite returns naive datetimes for ``DateTime(timezone=True)`` columns, so
    without :class:`~dentist_ai.db.base.UtcDateTime` these serialise with no
    ``Z`` and no offset. Any client doing ``new Date(iso)`` then reads them as
    *local* time and every relative timestamp shifts by the viewer's offset.
    """
    await _upload(authed_client, radiograph_bytes)

    response = await authed_client.get("/api/v1/studies")
    assert response.status_code == 200, response.text

    created_at = str(response.json()["items"][0]["createdAt"])
    assert datetime.fromisoformat(created_at).tzinfo is not None, created_at


async def test_uploaded_file_never_lands_in_static(
    authed_client: AsyncClient, radiograph_bytes: bytes, settings: Settings
) -> None:
    """PHI must not be reachable through the public static mount."""
    from dentist_ai.web.templating import STATIC_DIR

    await _upload(authed_client, radiograph_bytes)
    assert not list(STATIC_DIR.rglob("*.master.jpg"))
    assert settings.storage.resolved_root.exists()


async def test_traversal_filename_cannot_escape_storage(
    authed_client: AsyncClient, radiograph_bytes: bytes, settings: Settings
) -> None:
    """The client filename must never reach the filesystem."""
    await _upload(authed_client, radiograph_bytes, filename="../../../../etc/passwd.jpg")

    root = settings.storage.resolved_root
    for path in root.rglob("*"):
        assert root in path.resolve().parents or path.resolve() == root


async def test_filename_is_sanitised_but_keeps_cyrillic(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await _upload(authed_client, radiograph_bytes, filename="../Иванов_ОПТГ.jpg")
    assert "Иванов" in str(study["originalFilename"])
    assert ".." not in str(study["originalFilename"])
    assert "/" not in str(study["originalFilename"])


async def test_non_image_upload_is_rejected(authed_client: AsyncClient) -> None:
    response = await authed_client.post(
        "/api/v1/studies",
        files={"file": ("payload.jpg", b"#!/bin/sh\nrm -rf /\n", "image/jpeg")},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"


async def test_oversized_upload_is_rejected(authed_client: AsyncClient) -> None:
    # Random-ish noise so it does not compress below the limit.
    image = Image.effect_noise((3000, 3000), 120).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    oversized = buffer.getvalue() * 12

    response = await authed_client.post(
        "/api/v1/studies", files={"file": ("huge.png", oversized, "image/png")}
    )
    assert response.status_code == 413


async def test_exif_metadata_is_stripped(authed_client: AsyncClient) -> None:
    """Radiographs can carry patient identifiers in EXIF; storage must drop them."""
    from PIL import Image as PILImage

    exif = PILImage.Exif()
    exif[0x010E] = "PATIENT: Ivanov Ivan, MRN 12345"  # ImageDescription
    buffer = io.BytesIO()
    PILImage.new("L", (800, 600), 40).save(buffer, format="JPEG", exif=exif)

    study = await _upload(authed_client, buffer.getvalue())
    image = await authed_client.get(str(study["imageUrl"]))
    assert image.status_code == 200

    with PILImage.open(io.BytesIO(image.content)) as stored:
        assert not dict(stored.getexif())


async def test_upload_requires_authentication(client: AsyncClient, radiograph_bytes: bytes) -> None:
    token = (await client.get("/api/v1/auth/csrf")).json()["csrfToken"]
    response = await client.post(
        "/api/v1/studies",
        files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 401


async def test_image_requires_authentication(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    url = str(study["imageUrl"])

    assert (await authed_client.get(url)).status_code == 200

    authed_client.cookies.clear()
    assert (await authed_client.get(url)).status_code == 401


async def test_review_flow(authed_client: AsyncClient, radiograph_bytes: bytes) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    finding_id = study["findings"][0]["id"]

    response = await authed_client.patch(
        f"/api/v1/studies/{study['publicId']}/findings/{finding_id}",
        json={"review": "confirmed"},
    )
    assert response.status_code == 200
    assert response.json()["review"] == "confirmed"
    assert response.json()["reviewedAt"] is not None


async def test_csv_export(authed_client: AsyncClient, radiograph_bytes: bytes) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    response = await authed_client.get(f"/api/v1/studies/{study['publicId']}/export.csv")

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    # UTF-8 BOM so Excel renders Cyrillic labels instead of mojibake.
    assert response.content.startswith(b"\xef\xbb\xbf")


async def test_delete_removes_blob_when_unreferenced(
    authed_client: AsyncClient, radiograph_bytes: bytes, settings: Settings
) -> None:
    study = await _upload(authed_client, radiograph_bytes)
    assert list(settings.storage.resolved_root.rglob("*.master.jpg"))

    response = await authed_client.delete(f"/api/v1/studies/{study['publicId']}")
    assert response.status_code == 200
    assert not list(settings.storage.resolved_root.rglob("*.master.jpg"))


async def test_delete_keeps_blob_still_referenced(
    authed_client: AsyncClient, radiograph_bytes: bytes, settings: Settings
) -> None:
    first = await _upload(authed_client, radiograph_bytes)
    await _upload(authed_client, radiograph_bytes)

    await authed_client.delete(f"/api/v1/studies/{first['publicId']}")
    # The second study still points at the same content hash.
    assert list(settings.storage.resolved_root.rglob("*.master.jpg"))


@pytest.mark.parametrize("path", ["/api/v1/studies/NOPE", "/api/v1/studies/NOPE/image"])
async def test_unknown_study_is_404(authed_client: AsyncClient, path: str) -> None:
    assert (await authed_client.get(path)).status_code == 404
