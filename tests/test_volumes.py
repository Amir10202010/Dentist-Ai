"""CBCT analysis and the HTTP surface around it.

The pipeline tests assert on *properties* rather than on exact finding lists.
A heuristic classifier's output moves whenever a threshold is tuned, and a test
that pins the list would fail on every legitimate improvement while catching
none of the regressions that matter. What has to hold is that a phantom with no
pathology produces no pathology, that one with a lesion produces a lesion, and
that the measurements are calibrated — those are the properties a clinician
relies on.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from scripts.synthetic_cbct import Pathology, Phantom, build, build_preset, to_nifti

from dentist_ai.ml.cbct import build_registry
from dentist_ai.ml.cbct_taxonomy import VolumeCategory, by_key
from dentist_ai.ml.pipeline import RunRecord, StageKind, StageStatus, VolumeInput
from dentist_ai.services import volume as codec


def analyse(preset: str, *, seed: int = 5, field_of_view: str = "both_jaws") -> RunRecord:
    geometry = codec.parse(build_preset(preset, seed=seed))
    return (
        build_registry()
        .get()
        .run(
            VolumeInput(
                voxels=geometry.voxels,
                spacing=geometry.spacing,
                hu_slope=geometry.hu_slope,
                hu_intercept=geometry.hu_intercept,
                field_of_view=field_of_view,
            )
        )
    )


def pathologies(run: RunRecord) -> list[str]:
    return [
        item.class_key
        for item in run.detections
        if by_key(item.class_key).category is VolumeCategory.PATHOLOGY
    ]


# ---------------------------------------------------------------------------
# Pipeline mechanics
# ---------------------------------------------------------------------------
def test_every_stage_runs_and_is_recorded() -> None:
    run = analyse("periapical")

    assert [item.kind for item in run.stages] == [
        StageKind.QUALITY_CONTROL,
        StageKind.SEGMENTATION,
        StageKind.DETECTION,
        StageKind.CLASSIFICATION,
        StageKind.REPORT,
        StageKind.TREATMENT,
    ]
    assert all(item.status is StageStatus.OK for item in run.stages)
    # Every stage has to say what it did: the run log is the audit surface.
    assert all(item.summary for item in run.stages)
    assert run.succeeded


def test_a_failing_stage_does_not_discard_the_analysis() -> None:
    """Isolation is the point of the orchestrator, so it is tested directly."""
    from dentist_ai.ml.pipeline import Pipeline, PipelineState

    class Exploding:
        name = "exploding"
        kind = StageKind.TREATMENT
        version = "0.0.1"

        def applies_to(self, volume: VolumeInput) -> bool:
            return True

        def run(self, volume: VolumeInput, state: PipelineState) -> str:
            msg = "deliberate"
            raise RuntimeError(msg)

    geometry = codec.parse(build_preset("healthy", seed=2))
    reading = build_registry().get().stages[:-1]
    pipeline = Pipeline("test", "0", (*reading, Exploding()))
    run = pipeline.run(
        VolumeInput(
            voxels=geometry.voxels,
            spacing=geometry.spacing,
            hu_slope=geometry.hu_slope,
            hu_intercept=geometry.hu_intercept,
            field_of_view="both_jaws",
        )
    )

    assert [item.name for item in run.failed_stages] == ["exploding"]
    # A failed treatment stage costs a panel, not the reading.
    assert run.succeeded
    assert run.quality is not None


def test_the_radiology_pipeline_omits_the_treatment_stage() -> None:
    """A radiology service reports findings and does not propose treatment."""
    registry = build_registry()
    kinds = {stage.kind for stage in registry.get("radiology-review").describe()}
    assert StageKind.TREATMENT not in kinds
    assert StageKind.TREATMENT in {stage.kind for stage in registry.get().describe()}


def test_segmentation_locates_the_occlusal_plane_near_the_middle() -> None:
    """The phantom's arches meet at the centre, so the landmark must land there.

    Every regional label depends on this one number, which is why it is asserted
    rather than merely exercised.
    """
    run = analyse("periapical")
    assert run.landmarks["occlusal_z"] == pytest.approx(0.5, abs=0.08)
    assert run.landmarks["midline_x"] == pytest.approx(0.5, abs=0.06)


# ---------------------------------------------------------------------------
# Sensitivity and specificity, as properties
# ---------------------------------------------------------------------------
def test_a_healthy_phantom_yields_no_pathology() -> None:
    """The property that matters most: no disease where there is none."""
    run = analyse("healthy")
    assert pathologies(run) == []


def test_anatomy_is_still_reported_on_a_healthy_phantom() -> None:
    """Silence would mean the detector had simply stopped looking."""
    run = analyse("healthy")
    assert "mandibular_canal" in [item.class_key for item in run.detections]


def test_periapical_lesions_are_found_where_the_phantom_places_them() -> None:
    run = analyse("periapical")
    assert "apical_lesion" in pathologies(run)


def test_a_fitted_implant_is_reported_as_a_restoration() -> None:
    run = analyse("restored")
    assert "implant" in [item.class_key for item in run.detections]


def test_edentulous_spans_are_reported_as_missing_teeth() -> None:
    run = analyse("implant-site")
    assert "missing_tooth" in [item.class_key for item in run.detections]


def test_an_unerupted_third_molar_is_reported() -> None:
    run = analyse("cyst")
    assert "impacted_third_molar" in [item.class_key for item in run.detections]


def test_a_narrow_field_of_view_suppresses_joint_findings() -> None:
    """A 5 × 5 cm volume does not contain the condyles.

    Reporting on them anyway would be reporting on soft tissue, which is the
    failure mode the field-of-view gate exists to prevent.
    """
    run = analyse("healthy", field_of_view="implant_site")
    assert "tmj_abnormality" not in [item.class_key for item in run.detections]


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
def test_a_noisy_scan_scores_below_a_clean_one() -> None:
    clean = analyse("healthy")
    noisy = analyse("poor-quality")

    assert clean.quality is not None
    assert noisy.quality is not None
    assert noisy.quality.score < clean.quality.score


def test_confidence_is_discounted_on_a_poor_scan() -> None:
    """A finding on an unreadable scan must not carry a clean scan's confidence."""
    run = analyse("poor-quality")
    if not run.detections:
        pytest.skip("no findings on this phantom to compare")
    assert run.quality is not None
    assert all(item.confidence <= 0.95 for item in run.detections)


def test_metal_is_reported_as_a_quality_finding_not_a_disease() -> None:
    geometry = codec.parse(
        to_nifti(
            build(Phantom(seed=4, pathology=Pathology(implants=(46, 36), root_fillings=(16, 26)))),
            (0.5, 0.5, 0.5),
        )
    )
    run = (
        build_registry()
        .get()
        .run(
            VolumeInput(
                voxels=geometry.voxels,
                spacing=geometry.spacing,
                hu_slope=geometry.hu_slope,
                hu_intercept=geometry.hu_intercept,
                field_of_view="both_jaws",
            )
        )
    )
    for item in run.detections:
        if item.class_key == "metal_artifact":
            assert by_key(item.class_key).category is VolumeCategory.QUALITY


# ---------------------------------------------------------------------------
# Reporting invariants
# ---------------------------------------------------------------------------
def test_findings_are_ordered_for_triage() -> None:
    run = analyse("cyst")
    ranks = [by_key(item.class_key).severity.rank for item in run.detections]
    assert ranks == sorted(ranks), "most severe first"


def test_no_class_is_reported_more_than_four_times() -> None:
    """Six identical 'infections' is one rule misfiring, not six diseases."""
    for preset in ("healthy", "periapical", "cyst", "poor-quality", "restored"):
        run = analyse(preset)
        counts: dict[str, int] = {}
        for item in run.detections:
            counts[item.class_key] = counts.get(item.class_key, 0) + 1
        assert max(counts.values(), default=0) <= 4, preset


def test_boxes_stay_inside_the_unit_cube() -> None:
    """The database CHECK constraint would reject anything else."""
    run = analyse("cyst")
    for item in run.detections:
        box = item.box
        assert box.x >= 0.0
        assert box.y >= 0.0
        assert box.z >= 0.0
        # The constraint allows a hair over 1 for floating-point slop.
        assert box.x + box.width <= 1.0001
        assert box.y + box.height <= 1.0001
        assert box.z + box.depth <= 1.0001


def test_referral_only_classes_are_capped_low() -> None:
    """A mass or a fracture line can only ever be a referral.

    The cap is what stops the interface presenting one as settled, so it is
    asserted against the taxonomy rather than trusted.
    """
    for preset in ("healthy", "periapical", "cyst", "periodontal", "poor-quality", "restored"):
        for item in analyse(preset).detections:
            if by_key(item.class_key).requires_confirmation:
                assert item.confidence <= 0.6, f"{preset}: {item.class_key}"


def test_treatment_stage_derives_procedures_from_the_findings() -> None:
    run = analyse("periapical")
    if not pathologies(run):
        pytest.skip("no pathology to derive a pathway from")
    # The stage writes its codes into the run summary; the planner consumes them.
    treatment = next(item for item in run.stages if item.kind is StageKind.TREATMENT)
    assert treatment.status is StageStatus.OK
    assert "процедур" in treatment.summary


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------
async def create_patient(client: AsyncClient) -> int:
    response = await client.post("/api/v1/patients", json={"fullName": "Тестовый Пациент"})
    assert response.status_code == 201, response.text
    return int(response.json()["id"])


async def upload_volume(
    client: AsyncClient, patient_id: int, preset: str = "periapical"
) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/volumes",
        files={"file": (f"{preset}.nii", build_preset(preset, seed=7), "application/octet-stream")},
        data={"patient_id": str(patient_id), "field_of_view": "both_jaws"},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


async def test_upload_analyses_and_returns_geometry(authed_client: AsyncClient) -> None:
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)

    assert volume["status"] == "completed"
    assert volume["geometry"]["width"] > 0
    # The viewer sizes its textures from these before the voxels arrive.
    assert volume["geometry"]["physicalSize"][0] > 0
    assert volume["quality"]["score"] > 0
    assert volume["analysisMs"] is not None
    assert volume["pipelineVersion"]


async def test_voxels_are_served_with_a_strong_validator(authed_client: AsyncClient) -> None:
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)

    first = await authed_client.get(volume["voxelsUrl"])
    assert first.status_code == 200
    etag = first.headers["etag"]
    geometry = volume["geometry"]
    expected = 64 + geometry["width"] * geometry["height"] * geometry["depth"]
    assert len(first.content) == expected
    assert first.content.startswith(b"DVOL")

    # A revisit must not re-read 16 MB off disk.
    again = await authed_client.get(volume["voxelsUrl"], headers={"If-None-Match": etag})
    assert again.status_code == 304


async def test_previews_are_served_per_plane(authed_client: AsyncClient) -> None:
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)

    for plane in ("axial", "coronal", "sagittal"):
        response = await authed_client.get(f"/api/v1/volumes/{volume['publicId']}/preview/{plane}")
        assert response.status_code == 200, plane
        assert response.headers["content-type"] == "image/jpeg"

    missing = await authed_client.get(f"/api/v1/volumes/{volume['publicId']}/preview/oblique")
    assert missing.status_code == 404


async def test_findings_carry_their_explanation_and_next_steps(
    authed_client: AsyncClient,
) -> None:
    """A finding a clinician cannot interrogate is not reviewable."""
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)

    assert volume["findings"], "the phantom should produce findings"
    for finding in volume["findings"]:
        assert finding["rationale"]
        assert finding["nextSteps"]
        assert finding["regionLabel"]
        assert 0.0 < finding["confidence"] <= 1.0
        assert isinstance(finding["requiresConfirmation"], bool)


async def test_a_measurement_is_calibrated_against_the_geometry(
    authed_client: AsyncClient,
) -> None:
    """The one number in the product that could be confidently wrong.

    Two points 0.2 apart along z on a volume 90 mm deep must measure 18 mm. If
    the spacing were applied wrongly the value would be plausible but false,
    which is why it is checked against arithmetic rather than a fixture.
    """
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)
    depth_mm = volume["geometry"]["physicalSize"][2]

    response = await authed_client.post(
        f"/api/v1/volumes/{volume['publicId']}/measurements",
        json={
            "kind": "distance",
            "plane": "sagittal",
            "points": [[0.5, 0.5, 0.4], [0.5, 0.5, 0.6]],
            "label": "Высота кости",
        },
    )
    assert response.status_code == 201, response.text
    measurement = response.json()
    assert measurement["value"] == pytest.approx(depth_mm * 0.2, rel=0.01)
    assert measurement["unit"] == "мм"


async def test_an_angle_measurement_needs_three_points(authed_client: AsyncClient) -> None:
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)

    bad = await authed_client.post(
        f"/api/v1/volumes/{volume['publicId']}/measurements",
        json={"kind": "angle", "plane": "axial", "points": [[0.4, 0.5, 0.5], [0.5, 0.5, 0.5]]},
    )
    assert bad.status_code == 422

    good = await authed_client.post(
        f"/api/v1/volumes/{volume['publicId']}/measurements",
        json={
            "kind": "angle",
            "plane": "axial",
            "points": [[0.4, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.6, 0.5]],
        },
    )
    assert good.status_code == 201, good.text
    assert good.json()["value"] == pytest.approx(90.0, abs=1.0)
    assert good.json()["unit"] == "°"


async def test_reviewing_a_finding_records_the_adjudication(
    authed_client: AsyncClient,
) -> None:
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)
    finding_id = volume["findings"][0]["id"]

    response = await authed_client.patch(
        f"/api/v1/volumes/{volume['publicId']}/findings/{finding_id}",
        json={"review": "confirmed"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["review"] == "confirmed"
    assert response.json()["reviewedAt"] is not None


async def test_reanalysis_preserves_a_clinician_s_review(authed_client: AsyncClient) -> None:
    """A pipeline upgrade may change the model's opinion, never the reviewer's."""
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)
    finding = volume["findings"][0]

    await authed_client.patch(
        f"/api/v1/volumes/{volume['publicId']}/findings/{finding['id']}",
        json={"review": "confirmed"},
    )
    again = await authed_client.post(
        f"/api/v1/volumes/{volume['publicId']}/analyse", json={"pipeline": None}
    )
    assert again.status_code == 200, again.text

    carried = [item for item in again.json()["findings"] if item["classKey"] == finding["classKey"]]
    assert carried, "the same class should be found again"
    assert any(item["review"] == "confirmed" for item in carried)


async def test_annotations_round_trip(authed_client: AsyncClient) -> None:
    patient_id = await create_patient(authed_client)
    volume = await upload_volume(authed_client, patient_id)

    created = await authed_client.post(
        f"/api/v1/volumes/{volume['publicId']}/annotations",
        json={
            "kind": "marker",
            "plane": "axial",
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "title": "Проверить на приёме",
        },
    )
    assert created.status_code == 201, created.text

    listed = await authed_client.get(f"/api/v1/volumes/{volume['publicId']}/annotations")
    assert [item["title"] for item in listed.json()] == ["Проверить на приёме"]

    removed = await authed_client.delete(
        f"/api/v1/volumes/{volume['publicId']}/annotations/{created.json()['id']}"
    )
    assert removed.status_code == 200

    emptied = await authed_client.get(f"/api/v1/volumes/{volume['publicId']}/annotations")
    assert emptied.json() == []


async def test_pipelines_are_inspectable(authed_client: AsyncClient) -> None:
    """ "Which model produced this" is only answerable if the set is listable."""
    response = await authed_client.get("/api/v1/volumes/pipelines")
    assert response.status_code == 200
    names = {item["name"] for item in response.json()}
    assert {"dental-cbct", "radiology-review"} <= names
    assert all(item["stages"] for item in response.json())


async def test_a_radiograph_uploaded_as_a_volume_is_refused(
    authed_client: AsyncClient, radiograph_bytes: bytes
) -> None:
    patient_id = await create_patient(authed_client)
    response = await authed_client.post(
        "/api/v1/volumes",
        files={"file": ("scan.jpg", radiograph_bytes, "image/jpeg")},
        data={"patient_id": str(patient_id)},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------
async def test_another_clinic_cannot_reach_a_volume_or_its_voxels(
    two_clinics: tuple[AsyncClient, AsyncClient],
) -> None:
    """404 rather than 403: a 403 would confirm the record exists."""
    first, second = two_clinics

    patient_id = await create_patient(first)
    volume = await upload_volume(first, patient_id)
    public_id = volume["publicId"]

    for path in (
        f"/api/v1/volumes/{public_id}",
        f"/api/v1/volumes/{public_id}/voxels",
        f"/api/v1/volumes/{public_id}/preview/axial",
        f"/api/v1/volumes/{public_id}/annotations",
    ):
        response = await second.get(path)
        assert response.status_code == 404, path

    listed = await second.get("/api/v1/volumes")
    assert listed.json()["meta"]["total"] == 0
