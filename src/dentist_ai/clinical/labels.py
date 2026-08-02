"""Localised wording for clinical workflow states.

Kept out of ``db.models`` so the persistence layer stays free of interface
strings, and out of the frontend so a translation lives in one place.
"""

from __future__ import annotations

from typing import Final

from dentist_ai.db.models import (
    PlanItemStatus,
    PlanStatus,
    ScanArch,
    ScanKind,
    VolumeFieldOfView,
)
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale

SCAN_KIND_LABELS: Final[dict[ScanKind, dict[Locale, str]]] = {
    ScanKind.INTRAORAL: {
        "ru": "Интраоральный скан",
        "en": "Intraoral scan",
        "kk": "Интраоралды скан",
    },
    ScanKind.PLASTER_MODEL: {
        "ru": "Скан гипсовой модели",
        "en": "Plaster model scan",
        "kk": "Гипс моделінің сканы",
    },
    ScanKind.CBCT_SURFACE: {
        "ru": "Поверхность из КЛКТ",
        "en": "CBCT surface",
        "kk": "КЛКТ беті",
    },
    ScanKind.RESTORATION_DESIGN: {
        "ru": "CAD-конструкция",
        "en": "Restoration design",
        "kk": "CAD-конструкция",
    },
}

SCAN_ARCH_LABELS: Final[dict[ScanArch, dict[Locale, str]]] = {
    ScanArch.UPPER: {"ru": "Верхняя челюсть", "en": "Upper arch", "kk": "Жоғарғы жақ"},
    ScanArch.LOWER: {"ru": "Нижняя челюсть", "en": "Lower arch", "kk": "Төменгі жақ"},
    ScanArch.BOTH: {"ru": "Обе челюсти", "en": "Both arches", "kk": "Екі жақ"},
}

PLAN_STATUS_LABELS: Final[dict[PlanStatus, dict[Locale, str]]] = {
    PlanStatus.DRAFT: {"ru": "Черновик", "en": "Draft", "kk": "Жоба"},
    PlanStatus.ACTIVE: {"ru": "В работе", "en": "Active", "kk": "Жүргізілуде"},
    PlanStatus.COMPLETED: {"ru": "Завершён", "en": "Completed", "kk": "Аяқталды"},
    PlanStatus.CANCELLED: {"ru": "Отменён", "en": "Cancelled", "kk": "Тоқтатылды"},
}

PLAN_ITEM_STATUS_LABELS: Final[dict[PlanItemStatus, dict[Locale, str]]] = {
    PlanItemStatus.PROPOSED: {"ru": "Предложено", "en": "Proposed", "kk": "Ұсынылды"},
    PlanItemStatus.ACCEPTED: {"ru": "Согласовано", "en": "Accepted", "kk": "Келісілді"},
    PlanItemStatus.SCHEDULED: {"ru": "Записан", "en": "Scheduled", "kk": "Жазылды"},
    PlanItemStatus.IN_PROGRESS: {"ru": "В процессе", "en": "In progress", "kk": "Орындалуда"},
    PlanItemStatus.DONE: {"ru": "Выполнено", "en": "Done", "kk": "Орындалды"},
    PlanItemStatus.DECLINED: {"ru": "Отказ пациента", "en": "Declined", "kk": "Бас тартты"},
}

#: Shown wherever the product states a conclusion. The wording is fixed rather
#: than composed, because it is a legal statement, not copy.
DISCLAIMER: Final[dict[Locale, str]] = {
    "ru": (
        "Заключение сформировано автоматически из находок модели и таблицы "
        "протоколов. Это не диагноз: решение принимает врач."
    ),
    "en": (
        "Assembled automatically from model findings and the protocol table. "
        "This is not a diagnosis; the clinician decides."
    ),
    "kk": (
        "Қорытынды модель тапқан белгілер мен хаттамалар кестесі бойынша "
        "автоматты түрде құрылған. Бұл диагноз емес: шешімді дәрігер қабылдайды."
    ),
}


#: What the scanner was aimed at. Shown beside every CBCT finding, because a
#: finding's credibility depends on whether the anatomy it describes was in
#: the field of view at all.
VOLUME_FIELD_OF_VIEW_LABELS: Final[dict[VolumeFieldOfView, dict[Locale, str]]] = {
    VolumeFieldOfView.FULL_HEAD: {
        "ru": "Вся голова",
        "en": "Full head",
        "kk": "Толық бас",
    },
    VolumeFieldOfView.BOTH_JAWS: {
        "ru": "Обе челюсти",
        "en": "Both jaws",
        "kk": "Екі жақ",
    },
    VolumeFieldOfView.MAXILLA: {
        "ru": "Верхняя челюсть",
        "en": "Maxilla",
        "kk": "Жоғарғы жақ",
    },
    VolumeFieldOfView.MANDIBLE: {
        "ru": "Нижняя челюсть",
        "en": "Mandible",
        "kk": "Төменгі жақ",
    },
    VolumeFieldOfView.TMJ: {"ru": "ВНЧС", "en": "TMJ", "kk": "ТЖБ"},
    VolumeFieldOfView.SINUS: {
        "ru": "Пазухи",
        "en": "Sinuses",
        "kk": "Қуыстар",
    },
    VolumeFieldOfView.IMPLANT_SITE: {
        "ru": "Область имплантации",
        "en": "Implant site",
        "kk": "Имплантация аймағы",
    },
}

#: Bands the acquisition-quality score is reported in. A bare 0.72 means
#: nothing to a clinician; "хорошее" does.
_QUALITY_BANDS: Final[tuple[tuple[float, dict[Locale, str]], ...]] = (
    (0.85, {"ru": "Отличное", "en": "Excellent", "kk": "Өте жақсы"}),
    (0.70, {"ru": "Хорошее", "en": "Good", "kk": "Жақсы"}),
    (0.50, {"ru": "Удовлетворительное", "en": "Adequate", "kk": "Қанағаттанарлық"}),
    (
        0.0,
        {
            "ru": "Низкое — интерпретировать с осторожностью",
            "en": "Poor — interpret with caution",
            "kk": "Төмен — сақтықпен бағалаңыз",
        },
    ),
)

#: Added to every CBCT report and to the viewer's finding list. Stronger than
#: the 2D disclaimer, because a volumetric reading looks more authoritative
#: than it is and because two of the classes can only ever be a referral.
VOLUME_DISCLAIMER: Final[dict[Locale, str]] = {
    "ru": (
        "Автоматический анализ КЛКТ носит вспомогательный характер. Находки — "
        "это гипотезы, требующие подтверждения врачом; объёмные образования и "
        "переломы устанавливаются только по результатам клинического "
        "обследования и, при необходимости, биопсии."
    ),
    "en": (
        "Automated CBCT analysis is assistive only. Findings are hypotheses for "
        "a clinician to confirm; masses and fractures are established by "
        "clinical examination and, where indicated, biopsy — never by imaging "
        "alone."
    ),
    "kk": ("КЛКТ автоматты талдауы қосалқы сипатта. Табылымдар дәрігердің растауын талап етеді."),
}


def scan_kind_label(kind: ScanKind, locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(SCAN_KIND_LABELS[kind], locale)


def field_of_view_label(field_of_view: VolumeFieldOfView, locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(VOLUME_FIELD_OF_VIEW_LABELS[field_of_view], locale)


def quality_label(score: float, locale: Locale = DEFAULT_LOCALE) -> str:
    """Name the band an acquisition-quality score falls in."""
    for threshold, labels in _QUALITY_BANDS:
        if score >= threshold:
            return _pick(labels, locale)
    return _pick(_QUALITY_BANDS[-1][1], locale)


def volume_disclaimer(locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(VOLUME_DISCLAIMER, locale)


def scan_arch_label(arch: ScanArch, locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(SCAN_ARCH_LABELS[arch], locale)


def plan_status_label(status: PlanStatus, locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(PLAN_STATUS_LABELS[status], locale)


def plan_item_status_label(status: PlanItemStatus, locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(PLAN_ITEM_STATUS_LABELS[status], locale)


def disclaimer(locale: Locale = DEFAULT_LOCALE) -> str:
    return _pick(DISCLAIMER, locale)


def _pick(labels: dict[Locale, str], locale: Locale) -> str:
    return labels.get(locale) or labels[DEFAULT_LOCALE]
