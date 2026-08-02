"""FDI tooth numbering for detections on a panoramic radiograph.

A detection carries a box, not a tooth. Turning one into the other is
geometry plus two assumptions about how a panoramic image is laid out, and
both of them fail on a badly positioned patient. The number this module
returns is therefore an estimate that the viewer shows as editable — the
clinician's correction is what gets stored as fact.
"""

from __future__ import annotations

from typing import Final

from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, FindingClass, Locale

#: Quadrants run clockwise from the patient's upper right, which on a
#: radiograph is the viewer's upper left.
QUADRANTS: Final[tuple[int, ...]] = (1, 2, 3, 4)
POSITIONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7, 8)

#: (upper, patient's right) -> quadrant.
_QUADRANT_BY_POSITION: Final[dict[tuple[bool, bool], int]] = {
    (True, True): 1,
    (True, False): 2,
    (False, False): 3,
    (False, True): 4,
}

#: Every permanent tooth, in the order an odontogram draws them: upper row
#: left-to-right as the viewer sees it, then the lower row.
UPPER_ROW: Final[tuple[int, ...]] = tuple(18 - offset for offset in range(8)) + tuple(
    21 + offset for offset in range(8)
)
LOWER_ROW: Final[tuple[int, ...]] = tuple(48 - offset for offset in range(8)) + tuple(
    31 + offset for offset in range(8)
)
PERMANENT_TEETH: Final[tuple[int, ...]] = UPPER_ROW + LOWER_ROW

#: Horizontal span of the dentition on a correctly positioned image. Outside
#: it there is ramus and condyle, never a tooth.
_ARCH_LEFT: Final[float] = 0.12
_ARCH_RIGHT: Final[float] = 0.88
_MIDLINE: Final[float] = 0.5

#: The occlusal plane is not a straight line: it sags towards the midline.
#: Modelled as a parabola between these two heights so that an upper molar
#: near the ramus is not read as a lower one.
_OCCLUSAL_AT_RAMUS: Final[float] = 0.46
_OCCLUSAL_SAG: Final[float] = 0.06

#: Cumulative mesiodistal width of a permanent half-arch, from the midline
#: outwards, as a fraction of the whole half-arch. Teeth are not equally
#: wide, so an even split would push every posterior finding one tooth distal.
_TOOTH_BOUNDS: Final[tuple[float, ...]] = (
    0.134,  # 1 central incisor
    0.236,  # 2 lateral incisor
    0.354,  # 3 canine
    0.465,  # 4 first premolar
    0.567,  # 5 second premolar
    0.724,  # 6 first molar
    0.866,  # 7 second molar
    1.000,  # 8 third molar
)

#: Half-width of the dental arch as a fraction of a CBCT's left-right extent,
#: and its anterior-posterior depth on the same scale. A dental field of view
#: is framed on the jaws, so the arch occupies a predictable share of it.
_ARCH_HALF_WIDTH: Final[float] = 0.34
_ARCH_DEPTH: Final[float] = 0.30
#: Arc length from midline to third molar, in the same normalised units.
_ARCH_HALF_ARC: Final[float] = 0.42

_POSITION_NAMES: Final[dict[int, dict[Locale, str]]] = {
    1: {"ru": "центральный резец", "en": "central incisor", "kk": "орталық күрек тіс"},
    2: {"ru": "боковой резец", "en": "lateral incisor", "kk": "бүйір күрек тіс"},
    3: {"ru": "клык", "en": "canine", "kk": "азу тіс"},
    4: {"ru": "первый премоляр", "en": "first premolar", "kk": "бірінші премоляр"},
    5: {"ru": "второй премоляр", "en": "second premolar", "kk": "екінші премоляр"},
    6: {"ru": "первый моляр", "en": "first molar", "kk": "бірінші моляр"},
    7: {"ru": "второй моляр", "en": "second molar", "kk": "екінші моляр"},
    8: {"ru": "зуб мудрости", "en": "third molar", "kk": "ақыл тіс"},
}

_QUADRANT_NAMES: Final[dict[int, dict[Locale, str]]] = {
    1: {"ru": "верхний правый", "en": "upper right", "kk": "жоғарғы оң"},
    2: {"ru": "верхний левый", "en": "upper left", "kk": "жоғарғы сол"},
    3: {"ru": "нижний левый", "en": "lower left", "kk": "төменгі сол"},
    4: {"ru": "нижний правый", "en": "lower right", "kk": "төменгі оң"},
}


def is_valid(tooth_number: int) -> bool:
    quadrant, position = divmod(tooth_number, 10)
    return quadrant in QUADRANTS and position in POSITIONS


def quadrant_of(tooth_number: int) -> int:
    return tooth_number // 10


def is_upper(tooth_number: int) -> bool:
    return quadrant_of(tooth_number) in (1, 2)


def occlusal_plane_at(x: float) -> float:
    """Height of the occlusal plane at horizontal position ``x``."""
    offset = 2.0 * x - 1.0
    return _OCCLUSAL_AT_RAMUS + _OCCLUSAL_SAG * (1.0 - offset * offset)


def estimate_tooth(
    taxonomy: FindingClass,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> int | None:
    """Estimate the FDI number for a detection, or ``None`` if it has none.

    Regional findings — the mandibular canal, an archwire, bone loss across a
    sextant — are not numbered, because a number would imply a precision the
    detection does not have.
    """
    if not taxonomy.is_tooth_level:
        return None

    centre_x = x + width / 2.0
    centre_y = y + height / 2.0

    upper = centre_y < occlusal_plane_at(centre_x)
    patient_right = centre_x < _MIDLINE
    quadrant = _QUADRANT_BY_POSITION[(upper, patient_right)]

    if patient_right:
        span = _MIDLINE - _ARCH_LEFT
        distance = _MIDLINE - centre_x
    else:
        span = _ARCH_RIGHT - _MIDLINE
        distance = centre_x - _MIDLINE

    fraction = min(max(distance / span, 0.0), 1.0)
    position = next(
        index for index, bound in enumerate(_TOOTH_BOUNDS, start=1) if fraction <= bound
    )
    return quadrant * 10 + position


def estimate_tooth_3d(
    *,
    x: float,
    y: float,
    z: float,
    occlusal_z: float,
    midline_x: float,
    arch_center_y: float,
) -> int | None:
    """Estimate the FDI number for a point in a CBCT volume.

    The same idea as :func:`estimate_tooth` with one axis fewer of guesswork.
    On a panoramic image the occlusal plane has to be *modelled*, because the
    projection flattens it into a curve whose sag depends on how the patient
    was positioned. A reconstruction has a real occlusal plane, which the
    segmentation stage measured, so upper-versus-lower is read off rather than
    assumed.

    What remains an estimate is the position along the arch. The arch is a
    parabola in the axial plane, so distance from the midline is measured
    along it rather than across the chord: the straight-line distance from the
    midline to a second molar is much shorter than the arc, and using it would
    push every posterior finding two teeth mesial.

    Returns ``None`` outside the dentition, where the number would be
    invented — the ramus, the condyle, the sinus.
    """
    upper = z >= occlusal_z
    patient_right = x < midline_x
    quadrant = _QUADRANT_BY_POSITION[(upper, patient_right)]

    lateral = abs(x - midline_x)
    if lateral > _ARCH_HALF_WIDTH:
        return None

    # Arc length along a parabola, approximated by the chord plus the
    # posterior displacement. Good to a few per cent over a dental arch, which
    # is finer than the tooth widths it is compared against.
    posterior = max(y - arch_center_y + _ARCH_DEPTH / 2, 0.0)
    arc = (lateral**2 + posterior**2) ** 0.5

    fraction = min(arc / _ARCH_HALF_ARC, 1.0)
    position = next(
        index for index, bound in enumerate(_TOOTH_BOUNDS, start=1) if fraction <= bound
    )
    return quadrant * 10 + position


def tooth_name(tooth_number: int, locale: Locale = DEFAULT_LOCALE) -> str:
    """Descriptive name, e.g. ``"нижний левый первый моляр"``."""
    quadrant, position = divmod(tooth_number, 10)
    quadrant_name = _localised(_QUADRANT_NAMES.get(quadrant), locale)
    position_name = _localised(_POSITION_NAMES.get(position), locale)
    if not quadrant_name or not position_name:
        return str(tooth_number)
    return f"{quadrant_name} {position_name}"


def _localised(labels: dict[Locale, str] | None, locale: Locale) -> str:
    if labels is None:
        return ""
    return labels.get(locale) or labels[DEFAULT_LOCALE]
