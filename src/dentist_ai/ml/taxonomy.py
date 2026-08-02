"""The 31-class finding taxonomy: model class index -> meaning.

Each class carries a stable key (persisted with every finding, so re-ordering
model outputs cannot relabel stored history), a category, a severity and
labels in ru / en / kk.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

Locale = str
SUPPORTED_LOCALES: Final[tuple[str, ...]] = ("ru", "en", "kk")
DEFAULT_LOCALE: Final[str] = "ru"


class Category(enum.StrEnum):
    #: Disease requiring clinical attention.
    PATHOLOGY = "pathology"
    #: Existing dental work.
    RESTORATION = "restoration"
    #: Appliances and hardware.
    ORTHODONTIC = "orthodontic"
    #: Normal structures, useful for orientation.
    ANATOMY = "anatomy"
    #: Non-pathological states worth noting.
    CONDITION = "condition"


class Severity(enum.StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: Final[dict[Severity, int]] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


@dataclass(frozen=True, slots=True)
class FindingClass:
    class_id: int
    key: str
    category: Category
    severity: Severity
    labels: dict[Locale, str]

    def label(self, locale: Locale = DEFAULT_LOCALE) -> str:
        return self.labels.get(locale) or self.labels[DEFAULT_LOCALE]

    @property
    def needs_attention(self) -> bool:
        """Whether this class counts toward the triage badge."""
        return self.category is Category.PATHOLOGY

    @property
    def is_tooth_level(self) -> bool:
        """Whether the finding belongs to one tooth and can be numbered."""
        return self.key in TOOTH_LEVEL_KEYS


def _cls(
    class_id: int,
    key: str,
    category: Category,
    severity: Severity,
    ru: str,
    en: str,
    kk: str,
) -> FindingClass:
    return FindingClass(class_id, key, category, severity, {"ru": ru, "en": en, "kk": kk})


#: Order matches the trained model's class indices exactly (see
#: ``training/dataset.yaml``). Do not reorder — append only.
FINDING_CLASSES: Final[tuple[FindingClass, ...]] = (
    _cls(0, "caries", Category.PATHOLOGY, Severity.HIGH, "Кариес", "Caries", "Тіс жегісі"),
    _cls(1, "crown", Category.RESTORATION, Severity.INFO, "Коронка", "Crown", "Коронка"),
    _cls(2, "filling", Category.RESTORATION, Severity.INFO, "Пломба", "Filling", "Пломба"),
    _cls(3, "implant", Category.RESTORATION, Severity.INFO, "Имплант", "Implant", "Имплант"),
    _cls(
        4,
        "malaligned",
        Category.CONDITION,
        Severity.MEDIUM,
        "Неправильное положение зубов",
        "Malaligned teeth",
        "Тістердің қисаюы",
    ),
    _cls(
        5,
        "mandibular_canal",
        Category.ANATOMY,
        Severity.INFO,
        "Нижнечелюстной канал",
        "Mandibular canal",
        "Төменгі жақ өзегі",
    ),
    _cls(
        6,
        "missing_teeth",
        Category.CONDITION,
        Severity.MEDIUM,
        "Отсутствующие зубы",
        "Missing teeth",
        "Жоқ тістер",
    ),
    _cls(
        7,
        "periapical_lesion",
        Category.PATHOLOGY,
        Severity.CRITICAL,
        "Периапикальное поражение",
        "Periapical lesion",
        "Периапикалды зақымдану",
    ),
    _cls(
        8,
        "retained_root",
        Category.PATHOLOGY,
        Severity.MEDIUM,
        "Удержанный корень",
        "Retained root",
        "Қалған тамыр",
    ),
    _cls(
        9,
        "root_canal_treatment",
        Category.RESTORATION,
        Severity.INFO,
        "Лечение корневого канала",
        "Root canal treatment",
        "Тамыр өзегін емдеу",
    ),
    _cls(
        10,
        "root_piece",
        Category.PATHOLOGY,
        Severity.MEDIUM,
        "Осколок корня",
        "Root fragment",
        "Тамыр сынығы",
    ),
    _cls(
        11,
        "impacted_tooth",
        Category.CONDITION,
        Severity.MEDIUM,
        "Ретинированный зуб",
        "Impacted tooth",
        "Ретинирленген тіс",
    ),
    _cls(
        12,
        "maxillary_sinus",
        Category.ANATOMY,
        Severity.INFO,
        "Верхнечелюстная пазуха",
        "Maxillary sinus",
        "Жоғарғы жақ қуысы",
    ),
    _cls(
        13,
        "bone_loss",
        Category.PATHOLOGY,
        Severity.HIGH,
        "Потеря костной ткани",
        "Bone loss",
        "Сүйек тінінің жоғалуы",
    ),
    _cls(
        14,
        "tooth_fracture",
        Category.PATHOLOGY,
        Severity.CRITICAL,
        "Перелом зуба",
        "Tooth fracture",
        "Тіс сынуы",
    ),
    _cls(
        15,
        "permanent_teeth",
        Category.ANATOMY,
        Severity.INFO,
        "Постоянные зубы",
        "Permanent teeth",
        "Тұрақты тістер",
    ),
    _cls(
        16,
        "supra_eruption",
        Category.CONDITION,
        Severity.LOW,
        "Супраэрупция",
        "Supra-eruption",
        "Супраэрупция",
    ),
    _cls(
        17,
        "tad",
        Category.ORTHODONTIC,
        Severity.INFO,
        "Мини-имплант (TAD)",
        "Mini-implant (TAD)",
        "Мини-имплант (TAD)",
    ),
    _cls(18, "abutment", Category.RESTORATION, Severity.INFO, "Абатмент", "Abutment", "Абатмент"),
    _cls(
        19,
        "attrition",
        Category.CONDITION,
        Severity.LOW,
        "Стираемость (атриция)",
        "Attrition",
        "Тіс тозуы",
    ),
    _cls(
        20,
        "bone_defect",
        Category.PATHOLOGY,
        Severity.HIGH,
        "Костный дефект",
        "Bone defect",
        "Сүйек ақауы",
    ),
    _cls(
        21,
        "gingival_former",
        Category.RESTORATION,
        Severity.INFO,
        "Формирователь десны",
        "Gingival former",
        "Қызыл иек қалыптастырғыш",
    ),
    _cls(
        22,
        "metal_band",
        Category.ORTHODONTIC,
        Severity.INFO,
        "Металлическая лента",
        "Metal band",
        "Металл таспа",
    ),
    _cls(
        23,
        "orthodontic_brackets",
        Category.ORTHODONTIC,
        Severity.INFO,
        "Ортодонтические брекеты",
        "Orthodontic brackets",
        "Ортодонтиялық брекеттер",
    ),
    _cls(
        24,
        "permanent_retainer",
        Category.ORTHODONTIC,
        Severity.INFO,
        "Несъёмный ретейнер",
        "Permanent retainer",
        "Тұрақты ретейнер",
    ),
    _cls(
        25,
        "post_and_core",
        Category.RESTORATION,
        Severity.INFO,
        "Пост и кор",
        "Post and core",
        "Пост және кор",
    ),
    _cls(
        26,
        "fixation_plate",
        Category.RESTORATION,
        Severity.INFO,
        "Пластина (фиксационная)",
        "Fixation plate",
        "Бекіту пластинасы",
    ),
    _cls(27, "wire", Category.ORTHODONTIC, Severity.INFO, "Проволока", "Wire", "Сым"),
    _cls(28, "cyst", Category.PATHOLOGY, Severity.CRITICAL, "Киста", "Cyst", "Киста"),
    _cls(
        29,
        "root_resorption",
        Category.PATHOLOGY,
        Severity.HIGH,
        "Резорбция корня",
        "Root resorption",
        "Тамыр резорбциясы",
    ),
    _cls(
        30,
        "primary_teeth",
        Category.ANATOMY,
        Severity.INFO,
        "Молочные зубы",
        "Primary teeth",
        "Сүт тістері",
    ),
)

#: Classes that describe one tooth and can therefore be given an FDI number.
#: The rest are regional — jaw anatomy, appliances spanning several teeth, or
#: statements about the dentition as a whole.
TOOTH_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "abutment",
        "attrition",
        "caries",
        "crown",
        "filling",
        "gingival_former",
        "impacted_tooth",
        "implant",
        "missing_teeth",
        "periapical_lesion",
        "post_and_core",
        "retained_root",
        "root_canal_treatment",
        "root_piece",
        "root_resorption",
        "supra_eruption",
        "tooth_fracture",
    }
)

_BY_ID: Final[dict[int, FindingClass]] = {item.class_id: item for item in FINDING_CLASSES}
_BY_KEY: Final[dict[str, FindingClass]] = {item.key: item for item in FINDING_CLASSES}

UNKNOWN_CLASS: Final[FindingClass] = _cls(
    -1,
    "unknown",
    Category.CONDITION,
    Severity.INFO,
    "Неизвестно",
    "Unknown",
    "Белгісіз",
)


def by_id(class_id: int) -> FindingClass:
    """Resolve a model class index, tolerating weights newer than this table."""
    return _BY_ID.get(class_id, UNKNOWN_CLASS)


def by_key(key: str) -> FindingClass:
    return _BY_KEY.get(key, UNKNOWN_CLASS)


CATEGORY_LABELS: Final[dict[Category, dict[Locale, str]]] = {
    Category.PATHOLOGY: {"ru": "Патологии", "en": "Pathologies", "kk": "Патологиялар"},
    Category.RESTORATION: {"ru": "Реставрации", "en": "Restorations", "kk": "Реставрациялар"},
    Category.ORTHODONTIC: {"ru": "Ортодонтия", "en": "Orthodontics", "kk": "Ортодонтия"},
    Category.ANATOMY: {"ru": "Анатомия", "en": "Anatomy", "kk": "Анатомия"},
    Category.CONDITION: {"ru": "Состояния", "en": "Conditions", "kk": "Жағдайлар"},
}

SEVERITY_LABELS: Final[dict[Severity, dict[Locale, str]]] = {
    Severity.CRITICAL: {"ru": "Критично", "en": "Critical", "kk": "Критикалық"},
    Severity.HIGH: {"ru": "Высокая", "en": "High", "kk": "Жоғары"},
    Severity.MEDIUM: {"ru": "Средняя", "en": "Medium", "kk": "Орташа"},
    Severity.LOW: {"ru": "Низкая", "en": "Low", "kk": "Төмен"},
    Severity.INFO: {"ru": "Информация", "en": "Info", "kk": "Ақпарат"},
}
