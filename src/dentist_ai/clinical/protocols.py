"""What a finding implies for treatment.

An explicit lookup table, not model output: the detector says "periapical
lesion, 0.87"; this table says a periapical lesion is normally handled with
root canal treatment and that it should not wait. The clinician accepts,
edits or discards each proposal, so the table only has to be a sane starting
point — it is never a prescription.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale


class ProcedureCategory(enum.StrEnum):
    THERAPY = "therapy"
    ENDODONTICS = "endodontics"
    SURGERY = "surgery"
    PERIODONTICS = "periodontics"
    ORTHODONTICS = "orthodontics"
    PROSTHETICS = "prosthetics"
    DIAGNOSTICS = "diagnostics"


class Priority(enum.StrEnum):
    URGENT = "urgent"
    HIGH = "high"
    ROUTINE = "routine"
    OPTIONAL = "optional"

    @property
    def rank(self) -> int:
        return _PRIORITY_RANK[self]


_PRIORITY_RANK: Final[dict[Priority, int]] = {
    Priority.URGENT: 0,
    Priority.HIGH: 1,
    Priority.ROUTINE: 2,
    Priority.OPTIONAL: 3,
}


@dataclass(frozen=True, slots=True)
class Procedure:
    code: str
    category: ProcedureCategory
    priority: Priority
    #: Typical number of appointments and minutes per appointment. Editable
    #: per item, because every clinic's numbers differ.
    visits: int
    minutes: int
    labels: dict[Locale, str]

    def label(self, locale: Locale = DEFAULT_LOCALE) -> str:
        return self.labels.get(locale) or self.labels[DEFAULT_LOCALE]


def _proc(
    code: str,
    category: ProcedureCategory,
    priority: Priority,
    visits: int,
    minutes: int,
    ru: str,
    en: str,
    kk: str,
) -> Procedure:
    return Procedure(code, category, priority, visits, minutes, {"ru": ru, "en": en, "kk": kk})


PROCEDURES: Final[tuple[Procedure, ...]] = (
    _proc(
        "caries_restoration",
        ProcedureCategory.THERAPY,
        Priority.HIGH,
        1,
        60,
        "Лечение кариеса с реставрацией",
        "Caries removal and restoration",
        "Тіс жегісін емдеу және реставрация",
    ),
    _proc(
        "root_canal",
        ProcedureCategory.ENDODONTICS,
        Priority.URGENT,
        2,
        90,
        "Эндодонтическое лечение (каналы)",
        "Root canal treatment",
        "Тамыр өзегін емдеу",
    ),
    _proc(
        "endo_retreatment",
        ProcedureCategory.ENDODONTICS,
        Priority.HIGH,
        2,
        90,
        "Повторное эндодонтическое лечение",
        "Endodontic retreatment",
        "Қайталама эндодонтиялық емдеу",
    ),
    _proc(
        "extraction",
        ProcedureCategory.SURGERY,
        Priority.HIGH,
        1,
        45,
        "Удаление зуба",
        "Extraction",
        "Тісті жұлу",
    ),
    _proc(
        "root_removal",
        ProcedureCategory.SURGERY,
        Priority.HIGH,
        1,
        45,
        "Удаление корня",
        "Root removal",
        "Тамырды жұлу",
    ),
    _proc(
        "cyst_removal",
        ProcedureCategory.SURGERY,
        Priority.URGENT,
        1,
        90,
        "Цистэктомия",
        "Cyst enucleation",
        "Цистэктомия",
    ),
    _proc(
        "surgical_consult",
        ProcedureCategory.SURGERY,
        Priority.HIGH,
        1,
        30,
        "Консультация хирурга",
        "Surgical consultation",
        "Хирург кеңесі",
    ),
    _proc(
        "periodontal_therapy",
        ProcedureCategory.PERIODONTICS,
        Priority.HIGH,
        2,
        60,
        "Пародонтологическое лечение",
        "Periodontal therapy",
        "Пародонтологиялық емдеу",
    ),
    _proc(
        "bone_graft",
        ProcedureCategory.SURGERY,
        Priority.ROUTINE,
        1,
        90,
        "Костная пластика",
        "Bone grafting",
        "Сүйек пластикасы",
    ),
    _proc(
        "implant_placement",
        ProcedureCategory.SURGERY,
        Priority.ROUTINE,
        1,
        90,
        "Имплантация",
        "Implant placement",
        "Имплантация",
    ),
    _proc(
        "prosthetic_plan",
        ProcedureCategory.PROSTHETICS,
        Priority.ROUTINE,
        1,
        45,
        "Консультация ортопеда о замещении дефекта",
        "Prosthodontic consultation",
        "Ортопед кеңесі",
    ),
    _proc(
        "crown_restoration",
        ProcedureCategory.PROSTHETICS,
        Priority.ROUTINE,
        2,
        60,
        "Ортопедическая коронка",
        "Crown restoration",
        "Ортопедиялық коронка",
    ),
    _proc(
        "orthodontic_consult",
        ProcedureCategory.ORTHODONTICS,
        Priority.ROUTINE,
        1,
        45,
        "Консультация ортодонта",
        "Orthodontic consultation",
        "Ортодонт кеңесі",
    ),
    _proc(
        "occlusion_review",
        ProcedureCategory.DIAGNOSTICS,
        Priority.ROUTINE,
        1,
        30,
        "Оценка окклюзии",
        "Occlusion assessment",
        "Окклюзияны бағалау",
    ),
    _proc(
        "night_guard",
        ProcedureCategory.PROSTHETICS,
        Priority.OPTIONAL,
        2,
        30,
        "Защитная каппа",
        "Night guard",
        "Қорғаныш каппа",
    ),
    _proc(
        "radiographic_followup",
        ProcedureCategory.DIAGNOSTICS,
        Priority.ROUTINE,
        1,
        20,
        "Контрольный снимок",
        "Follow-up radiograph",
        "Бақылау түсірілімі",
    ),
    _proc(
        "cbct_referral",
        ProcedureCategory.DIAGNOSTICS,
        Priority.HIGH,
        1,
        30,
        "Направление на КЛКТ",
        "CBCT referral",
        "КЛКТ-ға жолдама",
    ),
)

_BY_CODE: Final[dict[str, Procedure]] = {item.code: item for item in PROCEDURES}

#: Finding class key -> procedure codes, in the order a treatment plan should
#: list them. Classes absent from this table imply nothing on their own:
#: a crown or an implant is a record of past work, not a task.
PROTOCOLS: Final[dict[str, tuple[str, ...]]] = {
    "caries": ("caries_restoration",),
    "periapical_lesion": ("root_canal", "radiographic_followup"),
    "cyst": ("cbct_referral", "cyst_removal"),
    "tooth_fracture": ("surgical_consult",),
    "bone_loss": ("periodontal_therapy", "radiographic_followup"),
    "bone_defect": ("cbct_referral", "bone_graft"),
    "root_resorption": ("endo_retreatment", "radiographic_followup"),
    "retained_root": ("root_removal",),
    "root_piece": ("root_removal",),
    "missing_teeth": ("prosthetic_plan", "implant_placement"),
    "impacted_tooth": ("cbct_referral", "surgical_consult"),
    "malaligned": ("orthodontic_consult",),
    "attrition": ("occlusion_review", "night_guard"),
    "supra_eruption": ("occlusion_review",),
    "root_canal_treatment": (),
    "crown": (),
    "filling": (),
    "implant": (),
    "abutment": (),
    "post_and_core": (),
    "gingival_former": (),
}


def by_code(code: str) -> Procedure | None:
    return _BY_CODE.get(code)


def procedures_for(class_key: str) -> tuple[Procedure, ...]:
    codes = PROTOCOLS.get(class_key, ())
    return tuple(_BY_CODE[code] for code in codes)


CATEGORY_LABELS: Final[dict[ProcedureCategory, dict[Locale, str]]] = {
    ProcedureCategory.THERAPY: {"ru": "Терапия", "en": "Restorative", "kk": "Терапия"},
    ProcedureCategory.ENDODONTICS: {
        "ru": "Эндодонтия",
        "en": "Endodontics",
        "kk": "Эндодонтия",
    },
    ProcedureCategory.SURGERY: {"ru": "Хирургия", "en": "Surgery", "kk": "Хирургия"},
    ProcedureCategory.PERIODONTICS: {
        "ru": "Пародонтология",
        "en": "Periodontics",
        "kk": "Пародонтология",
    },
    ProcedureCategory.ORTHODONTICS: {
        "ru": "Ортодонтия",
        "en": "Orthodontics",
        "kk": "Ортодонтия",
    },
    ProcedureCategory.PROSTHETICS: {
        "ru": "Ортопедия",
        "en": "Prosthodontics",
        "kk": "Ортопедия",
    },
    ProcedureCategory.DIAGNOSTICS: {
        "ru": "Диагностика",
        "en": "Diagnostics",
        "kk": "Диагностика",
    },
}

PRIORITY_LABELS: Final[dict[Priority, dict[Locale, str]]] = {
    Priority.URGENT: {"ru": "Срочно", "en": "Urgent", "kk": "Шұғыл"},
    Priority.HIGH: {"ru": "В ближайшее время", "en": "Soon", "kk": "Жақын арада"},
    Priority.ROUTINE: {"ru": "Планово", "en": "Routine", "kk": "Жоспарлы"},
    Priority.OPTIONAL: {"ru": "По желанию", "en": "Optional", "kk": "Қалауы бойынша"},
}


def category_label(category: ProcedureCategory, locale: Locale = DEFAULT_LOCALE) -> str:
    labels = CATEGORY_LABELS[category]
    return labels.get(locale) or labels[DEFAULT_LOCALE]


def priority_label(priority: Priority, locale: Locale = DEFAULT_LOCALE) -> str:
    labels = PRIORITY_LABELS[priority]
    return labels.get(locale) or labels[DEFAULT_LOCALE]
