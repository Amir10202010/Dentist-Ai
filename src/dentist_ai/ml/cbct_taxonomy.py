"""The CBCT finding taxonomy: what volumetric analysis is allowed to report.

Separate from :mod:`dentist_ai.ml.taxonomy` because it describes a different
examination. A panoramic radiograph is a flattened projection — it can show
that a lesion exists; a CBCT reconstruction places it in three dimensions and
can measure its relationship to the mandibular canal. The two produce
different claims, so they get different vocabularies rather than one union
that would let a 2D detector emit a finding it cannot support.

Each class carries what a clinician needs in order to act on a detection, and
what a reviewer needs in order to disagree with it:

* **``severity``** drives triage order everywhere in the UI.
* **``rationale``** states which image feature the finding rests on. It is a
  fixed sentence per class, not generated prose: "why did the model say this"
  should answer the same way every time it is asked.
* **``next_steps``** is the clinical follow-through, mirroring the role of
  ``clinical/protocols.py`` for 2D findings.
* **``requires_confirmation``** marks classes that must never be presented as
  a conclusion. Anything neoplastic is research/demonstration output only, and
  the UI is required to say so beside the finding.

Keys are stable and persisted with every stored finding, exactly as in the 2D
taxonomy, so re-ordering this table cannot relabel history.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale, Severity


class VolumeCategory(enum.StrEnum):
    """What kind of statement a finding is."""

    #: Disease. Drives the attention badge.
    PATHOLOGY = "pathology"
    #: Normal structures a plan must avoid — canals, sinuses, foramina.
    ANATOMY = "anatomy"
    #: Existing dental work visible in the reconstruction.
    RESTORATION = "restoration"
    #: Shape, position and symmetry of the skeleton and dentition.
    STRUCTURAL = "structural"
    #: Statements about the acquisition rather than the patient.
    QUALITY = "quality"


class Region(enum.StrEnum):
    """Anatomical region a finding sits in.

    Coarser than a tooth number on purpose: many volumetric findings — a cyst,
    an asymmetry, a condylar change — belong to a region rather than to one
    tooth, and forcing them onto an FDI number would invent precision.
    """

    MAXILLA_LEFT = "maxilla_left"
    MAXILLA_RIGHT = "maxilla_right"
    MANDIBLE_LEFT = "mandible_left"
    MANDIBLE_RIGHT = "mandible_right"
    ANTERIOR_MAXILLA = "anterior_maxilla"
    ANTERIOR_MANDIBLE = "anterior_mandible"
    MAXILLARY_SINUS_LEFT = "maxillary_sinus_left"
    MAXILLARY_SINUS_RIGHT = "maxillary_sinus_right"
    TMJ_LEFT = "tmj_left"
    TMJ_RIGHT = "tmj_right"
    FULL_VOLUME = "full_volume"


REGION_LABELS: Final[dict[Region, dict[Locale, str]]] = {
    Region.MAXILLA_LEFT: {
        "ru": "Верхняя челюсть слева",
        "en": "Left maxilla",
        "kk": "Сол жақ жоғарғы жақсүйек",
    },
    Region.MAXILLA_RIGHT: {
        "ru": "Верхняя челюсть справа",
        "en": "Right maxilla",
        "kk": "Оң жақ жоғарғы жақсүйек",
    },
    Region.MANDIBLE_LEFT: {
        "ru": "Нижняя челюсть слева",
        "en": "Left mandible",
        "kk": "Сол жақ төменгі жақсүйек",
    },
    Region.MANDIBLE_RIGHT: {
        "ru": "Нижняя челюсть справа",
        "en": "Right mandible",
        "kk": "Оң жақ төменгі жақсүйек",
    },
    Region.ANTERIOR_MAXILLA: {
        "ru": "Фронтальный отдел верхней челюсти",
        "en": "Anterior maxilla",
        "kk": "Жоғарғы жақтың алдыңғы бөлігі",
    },
    Region.ANTERIOR_MANDIBLE: {
        "ru": "Фронтальный отдел нижней челюсти",
        "en": "Anterior mandible",
        "kk": "Төменгі жақтың алдыңғы бөлігі",
    },
    Region.MAXILLARY_SINUS_LEFT: {
        "ru": "Левая верхнечелюстная пазуха",
        "en": "Left maxillary sinus",
        "kk": "Сол жақ гайморит қуысы",
    },
    Region.MAXILLARY_SINUS_RIGHT: {
        "ru": "Правая верхнечелюстная пазуха",
        "en": "Right maxillary sinus",
        "kk": "Оң жақ гайморит қуысы",
    },
    Region.TMJ_LEFT: {
        "ru": "Левый ВНЧС",
        "en": "Left TMJ",
        "kk": "Сол жақ ТЖБ",
    },
    Region.TMJ_RIGHT: {
        "ru": "Правый ВНЧС",
        "en": "Right TMJ",
        "kk": "Оң жақ ТЖБ",
    },
    Region.FULL_VOLUME: {
        "ru": "Весь объём",
        "en": "Whole volume",
        "kk": "Толық көлем",
    },
}


def region_label(region: Region, locale: Locale = DEFAULT_LOCALE) -> str:
    labels = REGION_LABELS[region]
    return labels.get(locale) or labels[DEFAULT_LOCALE]


@dataclass(frozen=True, slots=True)
class VolumeFindingClass:
    key: str
    category: VolumeCategory
    severity: Severity
    labels: dict[Locale, str]
    #: The image feature the detection rests on. Answers "why did AI find this".
    rationale: dict[Locale, str]
    #: What to do next, clinically.
    next_steps: dict[Locale, str]
    #: Procedure codes from ``clinical/protocols.py`` this finding implies.
    procedures: tuple[str, ...] = ()
    #: Whether the finding belongs to a single tooth and can carry an FDI number.
    tooth_level: bool = False
    #: Findings that must be presented as requiring specialist confirmation and
    #: never as a diagnosis. Enforced in the presenter, not left to the caller.
    requires_confirmation: bool = False
    #: Whether a millimetre measurement is the natural way to express the
    #: finding — bone height, proximity to a canal, condylar dimensions.
    measurable: bool = False

    def label(self, locale: Locale = DEFAULT_LOCALE) -> str:
        return self.labels.get(locale) or self.labels[DEFAULT_LOCALE]

    def why(self, locale: Locale = DEFAULT_LOCALE) -> str:
        return self.rationale.get(locale) or self.rationale[DEFAULT_LOCALE]

    def what_next(self, locale: Locale = DEFAULT_LOCALE) -> str:
        return self.next_steps.get(locale) or self.next_steps[DEFAULT_LOCALE]

    @property
    def needs_attention(self) -> bool:
        return self.category is VolumeCategory.PATHOLOGY


def _cls(
    key: str,
    category: VolumeCategory,
    severity: Severity,
    labels: tuple[str, str, str],
    rationale: tuple[str, str, str],
    next_steps: tuple[str, str, str],
    *,
    procedures: tuple[str, ...] = (),
    tooth_level: bool = False,
    requires_confirmation: bool = False,
    measurable: bool = False,
) -> VolumeFindingClass:
    def localise(values: tuple[str, str, str]) -> dict[Locale, str]:
        return {"ru": values[0], "en": values[1], "kk": values[2]}

    return VolumeFindingClass(
        key=key,
        category=category,
        severity=severity,
        labels=localise(labels),
        rationale=localise(rationale),
        next_steps=localise(next_steps),
        procedures=procedures,
        tooth_level=tooth_level,
        requires_confirmation=requires_confirmation,
        measurable=measurable,
    )


#: Append only. The key is what is persisted; the order here is presentation
#: order in the taxonomy reference screen and nothing else.
VOLUME_FINDING_CLASSES: Final[tuple[VolumeFindingClass, ...]] = (
    _cls(
        "impacted_third_molar",
        VolumeCategory.STRUCTURAL,
        Severity.MEDIUM,
        (
            "Ретинированный зуб мудрости",
            "Impacted wisdom tooth",
            "Ретинирленген ақыл тісі",
        ),
        (
            "Коронка восьмого зуба полностью окружена костью и не достигла "
            "окклюзионной плоскости; ось наклонена к соседнему моляру.",
            "The third molar crown is fully enclosed in bone and has not reached "
            "the occlusal plane; its axis is angled toward the adjacent molar.",
            "Ақыл тісінің сауыты толығымен сүйекпен қоршалған және окклюзиялық "
            "жазықтыққа жетпеген.",
        ),
        (
            "Оцените отношение корней к нижнечелюстному каналу по срезам и "
            "направьте на консультацию хирурга.",
            "Assess the root-to-canal relationship on the slices and refer for "
            "a surgical consultation.",
            "Тамырлардың төменгі жақ өзегіне қатынасын бағалап, хирургқа жіберіңіз.",
        ),
        procedures=("surgical_consult",),
        tooth_level=True,
        measurable=True,
    ),
    _cls(
        "bone_loss_3d",
        VolumeCategory.PATHOLOGY,
        Severity.HIGH,
        ("Убыль костной ткани", "Alveolar bone loss", "Сүйек тінінің жоғалуы"),
        (
            "Уровень альвеолярного гребня снижен относительно эмалево-цементной "
            "границы более чем на 3 мм по нескольким срезам.",
            "The alveolar crest sits more than 3 mm apical to the cemento-enamel "
            "junction across several slices.",
            "Альвеолалық жоталар деңгейі эмаль-цемент шекарасынан 3 мм-ден астам төмен.",
        ),
        (
            "Измерьте убыль в миллиметрах на щёчной и нёбной поверхностях и "
            "назначьте пародонтологическое лечение.",
            "Measure the loss buccally and palatally in millimetres and start periodontal therapy.",
            "Жоғалуды миллиметрмен өлшеп, пародонтологиялық емдеуді бастаңыз.",
        ),
        procedures=("periodontal_therapy", "radiographic_followup"),
        tooth_level=True,
        measurable=True,
    ),
    _cls(
        "periodontal_disease",
        VolumeCategory.PATHOLOGY,
        Severity.HIGH,
        ("Пародонтит", "Periodontal disease", "Пародонтит"),
        (
            "Горизонтальная и угловая резорбция кости прослеживается в нескольких "
            "сегментах; межзубные перегородки потеряли кортикальную пластинку.",
            "Horizontal and angular resorption is present across several segments "
            "and the interdental septa have lost their cortical plate.",
            "Бірнеше сегментте сүйектің резорбциясы, аралық перделер кортикалды "
            "пластинкасын жоғалтқан.",
        ),
        (
            "Зондирование карманов, оценка индекса гигиены, пародонтологический "
            "протокол с контролем через 3 месяца.",
            "Probe pocket depths, record a hygiene index, and start a periodontal "
            "protocol with review at three months.",
            "Қалталарды зондтау, гигиена индексін бағалау, 3 айдан кейін бақылау.",
        ),
        procedures=("periodontal_therapy",),
        measurable=True,
    ),
    _cls(
        "cyst",
        VolumeCategory.PATHOLOGY,
        Severity.CRITICAL,
        ("Киста челюсти", "Jaw cyst", "Жақ кистасы"),
        (
            "Округлое разрежение с чётким кортикальным ободком, вытесняющее "
            "соседние структуры без их разрушения.",
            "A rounded radiolucency with a well-defined corticated rim that "
            "displaces adjacent structures without destroying them.",
            "Айқын кортикалды жиегі бар домалақ сирену, көрші құрылымдарды ығыстырады.",
        ),
        (
            "Измерьте размеры в трёх плоскостях, оцените отношение к каналу и "
            "пазухе, направьте к челюстно-лицевому хирургу.",
            "Measure it in three planes, assess its relationship to the canal and "
            "sinus, and refer to a maxillofacial surgeon.",
            "Үш жазықтықта өлшеп, өзек пен қуысқа қатынасын бағалап, хирургқа жіберіңіз.",
        ),
        procedures=("cbct_referral", "cyst_removal"),
        measurable=True,
    ),
    _cls(
        "suspicious_mass",
        VolumeCategory.PATHOLOGY,
        Severity.CRITICAL,
        (
            "Объёмное образование (требует верификации)",
            "Space-occupying lesion (requires verification)",
            "Көлемді түзіліс (тексеруді талап етеді)",
        ),
        (
            "Разрежение с нечёткими границами и признаками разрушения кортикальной "
            "пластинки. Разграничить доброкачественный и злокачественный процесс "
            "по изображению невозможно.",
            "A lesion with indistinct margins and cortical destruction. Imaging "
            "alone cannot separate a benign from a malignant process.",
            "Шекарасы анық емес, кортикалды пластинканың бұзылу белгілері бар түзіліс.",
        ),
        (
            "Немедленное направление к челюстно-лицевому хирургу или онкологу. "
            "Диагноз устанавливается только по результатам биопсии.",
            "Immediate referral to a maxillofacial surgeon or oncologist. The "
            "diagnosis is established by biopsy, never by imaging.",
            "Дереу хирургқа немесе онкологқа жіберу. Диагноз тек биопсия арқылы қойылады.",
        ),
        procedures=("surgical_consult",),
        requires_confirmation=True,
        measurable=True,
    ),
    _cls(
        "odontogenic_infection",
        VolumeCategory.PATHOLOGY,
        Severity.CRITICAL,
        ("Одонтогенная инфекция", "Odontogenic infection", "Одонтогенді инфекция"),
        (
            "Разрежение вокруг корня в сочетании с реакцией окружающей кости и "
            "утолщением слизистой пазухи над поражённым зубом.",
            "Periradicular radiolucency together with a reaction in the surrounding "
            "bone and mucosal thickening in the sinus above the affected tooth.",
            "Тамыр айналасындағы сирену және қоршаған сүйектің реакциясы.",
        ),
        (
            "Срочная санация очага: эндодонтическое лечение или удаление, оценка "
            "показаний к антибактериальной терапии.",
            "Urgent source control: endodontic treatment or extraction, plus an "
            "assessment of whether antibiotics are indicated.",
            "Ошақты шұғыл санациялау: эндодонтиялық емдеу немесе жұлу.",
        ),
        procedures=("root_canal", "surgical_consult"),
        tooth_level=True,
    ),
    _cls(
        "abscess",
        VolumeCategory.PATHOLOGY,
        Severity.CRITICAL,
        ("Абсцесс", "Abscess", "Абсцесс"),
        (
            "Полость с плотностью жидкости и прерыванием кортикальной пластинки, "
            "с признаками распространения в мягкие ткани.",
            "A fluid-density collection with cortical interruption and signs of "
            "spread into the soft tissues.",
            "Сұйықтық тығыздығындағы қуыс және кортикалды пластинканың үзілуі.",
        ),
        (
            "Неотложное дренирование и устранение причинного зуба. При признаках "
            "распространения — направление в стационар.",
            "Emergency drainage and management of the causative tooth. Refer to "
            "hospital if there are signs of spread.",
            "Шұғыл дренаждау және себепші тісті емдеу.",
        ),
        procedures=("surgical_consult", "extraction"),
        tooth_level=True,
    ),
    _cls(
        "root_fracture",
        VolumeCategory.PATHOLOGY,
        Severity.CRITICAL,
        ("Перелом корня", "Root fracture", "Тамыр сынуы"),
        (
            "Линия просветления, проходящая через корень и прослеживаемая более "
            "чем на одном срезе в двух плоскостях.",
            "A lucent line crossing the root and traceable on more than one slice in two planes.",
            "Тамыр арқылы өтетін және екі жазықтықта көрінетін жарық сызық.",
        ),
        (
            "Оцените уровень перелома: вертикальный перелом корня обычно является "
            "показанием к удалению.",
            "Determine the fracture level: a vertical root fracture is usually an "
            "indication for extraction.",
            "Сыну деңгейін бағалаңыз: тік сыну әдетте жұлуға көрсетілім.",
        ),
        procedures=("surgical_consult", "extraction"),
        tooth_level=True,
    ),
    _cls(
        "missing_tooth",
        VolumeCategory.STRUCTURAL,
        Severity.MEDIUM,
        ("Отсутствующий зуб", "Missing tooth", "Жоқ тіс"),
        (
            "В позиции зубного ряда нет коронки и корня, альвеолярный гребень ремоделирован.",
            "No crown or root at the arch position, with a remodelled alveolar ridge.",
            "Тіс қатарында сауыт пен тамыр жоқ, альвеолалық жота қайта қалыптасқан.",
        ),
        (
            "Измерьте высоту и ширину гребня для планирования имплантации или "
            "иного замещения дефекта.",
            "Measure ridge height and width to plan an implant or another way of "
            "restoring the space.",
            "Имплантацияны жоспарлау үшін жотаның биіктігі мен енін өлшеңіз.",
        ),
        procedures=("prosthetic_plan", "implant_placement"),
        tooth_level=True,
        measurable=True,
    ),
    _cls(
        "implant",
        VolumeCategory.RESTORATION,
        Severity.INFO,
        ("Имплант", "Dental implant", "Имплант"),
        (
            "Рентгеноконтрастное тело правильной резьбовой формы, интегрированное "
            "в альвеолярную кость.",
            "A radio-opaque threaded body integrated into the alveolar bone.",
            "Альвеолалық сүйекке біріктірілген рентгенконтрастты бұрандалы дене.",
        ),
        (
            "Проверьте краевой уровень кости вокруг шейки импланта и отсутствие "
            "периимплантного разрежения.",
            "Check the marginal bone level around the implant neck and the absence "
            "of peri-implant radiolucency.",
            "Имплант мойны айналасындағы сүйек деңгейін тексеріңіз.",
        ),
        tooth_level=True,
        measurable=True,
    ),
    _cls(
        "sinus_proximity",
        VolumeCategory.ANATOMY,
        Severity.MEDIUM,
        (
            "Близость к верхнечелюстной пазухе",
            "Maxillary sinus proximity",
            "Гайморит қуысына жақындық",
        ),
        (
            "Расстояние от верхушки корня или планируемого ложа импланта до дна "
            "пазухи меньше 2 мм.",
            "Less than 2 mm separates the root apex or planned implant site from the sinus floor.",
            "Тамыр ұшынан қуыс түбіне дейінгі қашықтық 2 мм-ден аз.",
        ),
        (
            "Измерьте остаточную высоту кости; при недостатке рассмотрите синус-лифтинг.",
            "Measure the residual bone height; consider a sinus lift where it is insufficient.",
            "Қалдық сүйек биіктігін өлшеңіз; жеткіліксіз болса синус-лифтингті қарастырыңыз.",
        ),
        procedures=("bone_graft", "surgical_consult"),
        measurable=True,
    ),
    _cls(
        "mandibular_canal",
        VolumeCategory.ANATOMY,
        Severity.INFO,
        ("Нижнечелюстной канал", "Mandibular canal", "Төменгі жақ өзегі"),
        (
            "Трубчатая структура с кортикальными стенками, прослеживаемая от "
            "нижнечелюстного отверстия до подбородочного.",
            "A corticated tubular structure traced from the mandibular foramen to "
            "the mental foramen.",
            "Кортикалды қабырғалы түтікше құрылым.",
        ),
        (
            "Отметьте ход канала перед любой операцией в боковом отделе нижней "
            "челюсти и выдержите безопасный зазор не менее 2 мм.",
            "Mark the canal before any posterior mandibular surgery and keep a "
            "safety margin of at least 2 mm.",
            "Кез келген операция алдында өзектің жүрісін белгілеңіз.",
        ),
        measurable=True,
    ),
    _cls(
        "root_canal_filling",
        VolumeCategory.RESTORATION,
        Severity.INFO,
        (
            "Пломбировка корневого канала",
            "Root canal filling",
            "Тамыр өзегінің пломбасы",
        ),
        (
            "Рентгеноконтрастный материал в просвете канала на всём или части его протяжения.",
            "Radio-opaque material within the canal lumen along all or part of its length.",
            "Өзек аймағындағы рентгенконтрастты материал.",
        ),
        (
            "Оцените плотность и уровень обтурации; при недопломбировке и "
            "периапикальных изменениях показано перелечивание.",
            "Assess density and level of obturation; retreatment is indicated when "
            "it is short and there are periapical changes.",
            "Обтурацияның тығыздығы мен деңгейін бағалаңыз.",
        ),
        # Deliberately empty, matching `root_canal_treatment` in the 2D
        # protocol table: a filled canal is a record of past work, not a task.
        # Retreatment is indicated by the *apical lesion* beside it, and that
        # class carries the procedure — putting it here too would propose
        # re-treating every sound endodontic case in the chart.
        procedures=(),
        tooth_level=True,
        measurable=True,
    ),
    _cls(
        "apical_lesion",
        VolumeCategory.PATHOLOGY,
        Severity.HIGH,
        ("Периапикальное поражение", "Apical lesion", "Периапикалды зақымдану"),
        (
            "Разрежение кости у верхушки корня, окружающее апикальное отверстие и "
            "видимое минимум в двух плоскостях.",
            "Bone lucency at the root apex surrounding the apical foramen and "
            "visible in at least two planes.",
            "Тамыр ұшындағы сүйектің сирену аймағы.",
        ),
        (
            "Эндодонтическое лечение или перелечивание причинного зуба, "
            "контрольный снимок через 6 месяцев.",
            "Endodontic treatment or retreatment of the causative tooth, with a "
            "follow-up image at six months.",
            "Себепші тісті эндодонтиялық емдеу, 6 айдан кейін бақылау.",
        ),
        procedures=("root_canal", "radiographic_followup"),
        tooth_level=True,
        measurable=True,
    ),
    _cls(
        "caries_3d",
        VolumeCategory.PATHOLOGY,
        Severity.HIGH,
        ("Кариозное поражение", "Carious lesion", "Кариес зақымдануы"),
        (
            "Зона пониженной плотности в твёрдых тканях коронки, распространяющаяся "
            "от поверхности вглубь к дентину.",
            "A low-density zone in the crown's hard tissues extending from the "
            "surface inward toward dentine.",
            "Сауыттың қатты тіндеріндегі тығыздығы төмен аймақ.",
        ),
        (
            "Подтвердите клинически и внутриротовым снимком: КЛКТ склонна "
            "переоценивать глубину из-за артефактов.",
            "Confirm clinically and with an intraoral radiograph: CBCT tends to "
            "overstate depth because of artefacts.",
            "Клиникалық және ауызішілік суретпен растаңыз.",
        ),
        procedures=("caries_restoration",),
        tooth_level=True,
    ),
    _cls(
        "orthodontic_anomaly",
        VolumeCategory.STRUCTURAL,
        Severity.MEDIUM,
        (
            "Ортодонтическая аномалия",
            "Orthodontic abnormality",
            "Ортодонтиялық ауытқу",
        ),
        (
            "Отклонение осей зубов и формы дуги от нормы: скученность, ротации или "
            "несоответствие ширины челюстей.",
            "Tooth axes and arch form deviate from normal: crowding, rotations or "
            "a transverse discrepancy between the jaws.",
            "Тіс осьтері мен доға формасының нормадан ауытқуы.",
        ),
        (
            "Направьте на ортодонтическую консультацию с расчётом по ТРГ и моделям.",
            "Refer for an orthodontic consultation with cephalometric and model analysis.",
            "Ортодонт кеңесіне жіберіңіз.",
        ),
        procedures=("orthodontic_consult",),
    ),
    _cls(
        "jaw_asymmetry",
        VolumeCategory.STRUCTURAL,
        Severity.MEDIUM,
        ("Асимметрия челюстей", "Jaw asymmetry", "Жақ асимметриясы"),
        (
            "Разница линейных размеров ветви и тела нижней челюсти между сторонами "
            "превышает 3 мм при сопоставимой ориентации.",
            "Ramus and body dimensions differ between sides by more than 3 mm at "
            "comparable orientation.",
            "Екі жақтың өлшемдері арасындағы айырмашылық 3 мм-ден асады.",
        ),
        (
            "Сопоставьте с клинической картиной и фотопротоколом; при значимой "
            "асимметрии — консультация ортогнатического хирурга.",
            "Correlate with the clinical picture and photographs; refer to an "
            "orthognathic surgeon when the asymmetry is significant.",
            "Клиникалық көрініспен салыстырыңыз.",
        ),
        procedures=("orthodontic_consult",),
        measurable=True,
    ),
    _cls(
        "tmj_abnormality",
        VolumeCategory.PATHOLOGY,
        Severity.HIGH,
        ("Изменения ВНЧС", "TMJ abnormality", "ТЖБ өзгерістері"),
        (
            "Уплощение, эрозия или остеофит суставной головки, сужение суставной "
            "щели по сравнению с противоположной стороной.",
            "Flattening, erosion or osteophyte formation of the condylar head, with "
            "joint space narrowing relative to the contralateral side.",
            "Буын басының жалпақтануы, эрозиясы немесе остеофиті.",
        ),
        (
            "Клиническая оценка функции сустава; при болевом синдроме — "
            "гнатологическое обследование и защитная каппа.",
            "Assess joint function clinically; where there is pain, arrange a "
            "gnathological work-up and a night guard.",
            "Буын функциясын клиникалық бағалау.",
        ),
        procedures=("occlusion_review", "night_guard"),
        measurable=True,
    ),
    _cls(
        "metal_artifact",
        VolumeCategory.QUALITY,
        Severity.LOW,
        ("Артефакты от металла", "Metal artefact", "Металл артефактілері"),
        (
            "Полосы повышенной и пониженной плотности, расходящиеся от "
            "металлических реставраций и снижающие достоверность оценки рядом.",
            "Bright and dark streaks radiating from metal restorations, reducing "
            "confidence in everything they cross.",
            "Металл реставрациялардан таралатын жолақтар.",
        ),
        (
            "Оценивайте зоны рядом с артефактами с осторожностью; при "
            "необходимости используйте внутриротовой снимок.",
            "Read the affected zones with caution; use an intraoral radiograph "
            "where a decision depends on them.",
            "Артефакт аймақтарын мұқият бағалаңыз.",
        ),
    ),
    _cls(
        "motion_artifact",
        VolumeCategory.QUALITY,
        Severity.MEDIUM,
        ("Артефакт движения", "Motion artefact", "Қозғалыс артефактісі"),
        (
            "Двоение контуров и размытие кортикальных границ, характерные для "
            "смещения пациента во время сканирования.",
            "Doubled outlines and blurred cortical borders, characteristic of "
            "patient movement during acquisition.",
            "Контурлардың екіленуі және шекаралардың бұлыңғырлануы.",
        ),
        (
            "Достоверность измерений снижена. При планировании имплантации "
            "рассмотрите повторное сканирование.",
            "Measurement reliability is reduced. Consider a repeat scan before implant planning.",
            "Өлшеу сенімділігі төмен. Қайта сканерлеуді қарастырыңыз.",
        ),
    ),
)

_BY_KEY: Final[dict[str, VolumeFindingClass]] = {item.key: item for item in VOLUME_FINDING_CLASSES}

UNKNOWN_VOLUME_CLASS: Final[VolumeFindingClass] = _cls(
    "unknown",
    VolumeCategory.QUALITY,
    Severity.INFO,
    ("Неизвестная находка", "Unknown finding", "Белгісіз табылым"),
    (
        "Класс находки отсутствует в текущей версии таксономии.",
        "The finding class is not present in the current taxonomy.",
        "Табылым класы ағымдағы таксономияда жоқ.",
    ),
    (
        "Проверьте версию модели.",
        "Check the model version.",
        "Модель нұсқасын тексеріңіз.",
    ),
)


def by_key(key: str) -> VolumeFindingClass:
    """Resolve a stored key, tolerating findings written by a newer model."""
    return _BY_KEY.get(key, UNKNOWN_VOLUME_CLASS)


VOLUME_CATEGORY_LABELS: Final[dict[VolumeCategory, dict[Locale, str]]] = {
    VolumeCategory.PATHOLOGY: {
        "ru": "Патологии",
        "en": "Pathology",
        "kk": "Патологиялар",
    },
    VolumeCategory.ANATOMY: {
        "ru": "Анатомические ориентиры",
        "en": "Anatomical landmarks",
        "kk": "Анатомиялық бағдарлар",
    },
    VolumeCategory.RESTORATION: {
        "ru": "Реставрации",
        "en": "Restorations",
        "kk": "Реставрациялар",
    },
    VolumeCategory.STRUCTURAL: {
        "ru": "Строение и положение",
        "en": "Structure and position",
        "kk": "Құрылым және орналасу",
    },
    VolumeCategory.QUALITY: {
        "ru": "Качество исследования",
        "en": "Study quality",
        "kk": "Зерттеу сапасы",
    },
}


def volume_category_label(category: VolumeCategory, locale: Locale = DEFAULT_LOCALE) -> str:
    labels = VOLUME_CATEGORY_LABELS[category]
    return labels.get(locale) or labels[DEFAULT_LOCALE]
