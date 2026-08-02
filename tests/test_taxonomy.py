"""The finding taxonomy and its localisation.

The taxonomy is the contract between the model's integer outputs and what a
clinician reads. A silent change here mislabels findings without failing
anything else, so it gets direct tests.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from dentist_ai.ml.taxonomy import (
    CATEGORY_LABELS,
    FINDING_CLASSES,
    SEVERITY_LABELS,
    SUPPORTED_LOCALES,
    UNKNOWN_CLASS,
    Category,
    Severity,
    by_id,
    by_key,
)

EXPECTED_CLASS_COUNT = 31


def test_class_ids_are_contiguous_and_complete() -> None:
    """Indices must match the trained model's output positions exactly."""
    ids = [item.class_id for item in FINDING_CLASSES]
    assert ids == list(range(EXPECTED_CLASS_COUNT))


def test_keys_are_unique() -> None:
    keys = [item.key for item in FINDING_CLASSES]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_every_class_is_translated(locale: str) -> None:
    for item in FINDING_CLASSES:
        label = item.labels.get(locale)
        assert label, f"{item.key} has no {locale} label"
        assert label.strip() == label


@pytest.mark.parametrize("locale", SUPPORTED_LOCALES)
def test_categories_and_severities_are_translated(locale: str) -> None:
    for category in Category:
        assert CATEGORY_LABELS[category].get(locale)
    for severity in Severity:
        assert SEVERITY_LABELS[severity].get(locale)


def test_unknown_class_id_degrades_gracefully() -> None:
    """Weights newer than this table must not crash the app."""
    resolved = by_id(999)
    assert resolved is UNKNOWN_CLASS
    assert resolved.label("ru")
    assert not resolved.needs_attention


def test_unknown_key_degrades_gracefully() -> None:
    assert by_key("not-a-real-key") is UNKNOWN_CLASS


def test_only_pathologies_need_attention() -> None:
    """The triage badge counts disease, not existing dental work."""
    for item in FINDING_CLASSES:
        assert item.needs_attention == (item.category is Category.PATHOLOGY)

    assert by_key("caries").needs_attention
    assert by_key("cyst").needs_attention
    # A crown is a detection, not a problem.
    assert not by_key("crown").needs_attention
    assert not by_key("filling").needs_attention
    assert not by_key("mandibular_canal").needs_attention


def test_severity_rank_orders_critical_first() -> None:
    ranks = [severity.rank for severity in Severity]
    assert ranks == sorted(ranks)
    assert Severity.CRITICAL.rank < Severity.INFO.rank


def test_unfamiliar_locale_falls_back_to_russian() -> None:
    assert by_key("caries").label("fr") == "Кариес"


# --------------------------------------------------------------------------
# End-to-end localisation
# --------------------------------------------------------------------------
async def test_labels_follow_the_user_locale(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    """Switching interface language must change finding labels, not just chrome.

    The old header had a language switcher that did nothing; this asserts the
    replacement actually resolves to translated strings end to end.
    """
    upload = await authed_client.post(
        "/api/v1/studies", files={"file": ("opg.jpg", radiograph_bytes, "image/jpeg")}
    )
    assert upload.status_code == 201
    public_id = upload.json()["publicId"]

    russian = (await authed_client.get(f"/api/v1/studies/{public_id}")).json()
    assert russian["findings"][0]["severityLabel"] in {
        item["ru"] for item in SEVERITY_LABELS.values()
    }

    switched = await authed_client.put(
        "/api/v1/settings/profile", json={"fullName": "Aigul S", "locale": "en"}
    )
    assert switched.status_code == 200

    english = (await authed_client.get(f"/api/v1/studies/{public_id}")).json()
    assert english["findings"][0]["severityLabel"] in {
        item["en"] for item in SEVERITY_LABELS.values()
    }
    # Class labels move too, not only severity.
    assert english["findings"][0]["label"] != russian["findings"][0]["label"] or (
        russian["findings"][0]["classKey"] in {"implant", "crown"}
    )


async def test_unsupported_locale_is_rejected(authed_client: AsyncClient) -> None:
    response = await authed_client.put(
        "/api/v1/settings/profile", json={"fullName": "Aigul S", "locale": "fr"}
    )
    assert response.status_code == 422


async def test_taxonomy_endpoint_lists_every_class(authed_client: AsyncClient) -> None:
    response = await authed_client.get("/api/v1/taxonomy")
    assert response.status_code == 200

    entries = response.json()
    assert len(entries) == EXPECTED_CLASS_COUNT
    assert {entry["classId"] for entry in entries} == set(range(EXPECTED_CLASS_COUNT))
    assert sum(1 for entry in entries if entry["needsAttention"]) == sum(
        1 for item in FINDING_CLASSES if item.needs_attention
    )
