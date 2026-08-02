"""FDI numbering from a box on a panoramic radiograph."""

from __future__ import annotations

import pytest

from dentist_ai.clinical import charting
from dentist_ai.ml.taxonomy import by_key

BOX = 0.04


def tooth_at(class_key: str, centre_x: float, centre_y: float) -> int | None:
    return charting.estimate_tooth(
        by_key(class_key),
        x=centre_x - BOX / 2,
        y=centre_y - BOX / 2,
        width=BOX,
        height=BOX,
    )


def test_the_chart_covers_every_permanent_tooth_exactly_once() -> None:
    assert len(charting.PERMANENT_TEETH) == 32
    assert len(set(charting.PERMANENT_TEETH)) == 32
    assert all(charting.is_valid(tooth) for tooth in charting.PERMANENT_TEETH)


def test_the_upper_row_reads_left_to_right_as_the_viewer_sees_it() -> None:
    """On a radiograph the patient's right is the viewer's left."""
    assert charting.UPPER_ROW[0] == 18
    assert charting.UPPER_ROW[7] == 11
    assert charting.UPPER_ROW[8] == 21
    assert charting.LOWER_ROW[0] == 48
    assert charting.LOWER_ROW[-1] == 38


@pytest.mark.parametrize(
    ("centre_x", "centre_y", "expected_quadrant"),
    [
        (0.30, 0.20, 1),  # viewer's left, above the occlusal plane
        (0.70, 0.20, 2),
        (0.70, 0.80, 3),
        (0.30, 0.80, 4),
    ],
)
def test_quadrants_follow_the_fdi_clock(
    centre_x: float, centre_y: float, expected_quadrant: int
) -> None:
    tooth = tooth_at("caries", centre_x, centre_y)
    assert tooth is not None
    assert charting.quadrant_of(tooth) == expected_quadrant


def test_a_finding_at_the_midline_is_a_central_incisor() -> None:
    assert tooth_at("caries", 0.49, 0.25) == 11
    assert tooth_at("caries", 0.51, 0.25) == 21


def test_a_finding_at_the_edge_of_the_arch_is_a_third_molar() -> None:
    assert tooth_at("caries", 0.10, 0.25) == 18
    assert tooth_at("caries", 0.92, 0.75) == 38


def test_teeth_are_not_split_evenly_across_the_arch() -> None:
    """Molars are wider than incisors; an even split would shift them distally."""
    positions = [tooth_at("caries", x / 100, 0.25) for x in range(50, 90, 2)]
    numbered = [tooth for tooth in positions if tooth is not None]
    # Monotonic from the midline outwards, and it reaches the third molar.
    assert numbered == sorted(numbered)
    assert numbered[0] == 21
    assert numbered[-1] == 28


def test_the_occlusal_plane_sags_towards_the_midline() -> None:
    """Anterior teeth sit lower on the image than the rami do."""
    assert charting.occlusal_plane_at(0.5) > charting.occlusal_plane_at(0.05)


def test_a_finding_just_below_the_sagging_plane_is_still_a_lower_tooth() -> None:
    # y = 0.50 is above the plane at the midline (0.52) but below it at the ramus.
    assert charting.is_upper(tooth_at("caries", 0.5, 0.50) or 0)
    assert not charting.is_upper(tooth_at("caries", 0.12, 0.50) or 0)


@pytest.mark.parametrize(
    "class_key", ["mandibular_canal", "maxillary_sinus", "bone_loss", "wire", "permanent_teeth"]
)
def test_regional_findings_get_no_number(class_key: str) -> None:
    """A number would imply a precision the detection does not have."""
    assert tooth_at(class_key, 0.3, 0.3) is None


def test_a_box_outside_the_dentition_clamps_to_the_last_tooth() -> None:
    """A detection over the ramus is wrong, but it must not produce tooth 39."""
    tooth = tooth_at("caries", 0.01, 0.25)
    assert tooth == 18


def test_tooth_names_are_localised_and_fall_back_to_russian() -> None:
    assert charting.tooth_name(36, "ru") == "нижний левый первый моляр"
    assert charting.tooth_name(36, "en") == "lower left first molar"
    assert charting.tooth_name(36, "de") == charting.tooth_name(36, "ru")


@pytest.mark.parametrize("value", [10, 19, 39, 49, 0, 50])
def test_invalid_fdi_numbers_are_rejected(value: int) -> None:
    assert not charting.is_valid(value)
