"""Turning findings into a plan a clinician can accept, edit or discard.

This is the third explicit table in ``clinical/``, alongside ``charting`` and
``protocols``, and it sits here for the same reason they do: a plan is the part
of the product that proposes doing something to a person, so it has to be the
part a clinician can audit line by line. Nothing in this module is model
output. The detector says "periapical lesion, 0.87"; ``protocols`` says a
periapical lesion is normally handled with root canal treatment; this module
decides what a *set* of such findings adds up to — in what order, over how
long, at what risk, and what the alternatives are.

Three ideas carry the design.

**A plan is a choice, not an answer.** The same findings support several
defensible courses of action, and which one is right depends on what the
patient wants, can afford and will tolerate. So the output is three
:class:`TreatmentOption` objects — conservative, standard, comprehensive —
rather than one recommendation. They are not "good, better, best": the
conservative option is the correct answer for a patient who wants the pain
gone and nothing else, and the interface presents them side by side.

**Sequencing is clinical, not cosmetic.** Urgent work comes first because
delay causes harm, but the ordering inside that is a real constraint chain:
disease is controlled before anything is restored, extraction sites heal
before implants go into them, and a diagnostic referral precedes the surgery
it informs. That chain is in :data:`_SEQUENCE_RANK`.

**Risk is derived from the combination, not the finding.** An extraction is
routine; an extraction in a mandible whose canal runs close to the roots is a
nerve-injury conversation. Neither finding alone produces that warning — the
pair does, which is what :data:`_RISK_RULES` encodes.

Every duration here is a planning estimate in the units clinics actually work
in — appointments, chair minutes, and calendar weeks including healing — and
every one is editable per item once the plan exists, because no two practices
run the same schedule.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from dentist_ai.clinical import protocols
from dentist_ai.clinical.protocols import Priority, Procedure, ProcedureCategory
from dentist_ai.core.text import plural_ru
from dentist_ai.db.models import PlanComplexity, TreatmentApproach
from dentist_ai.ml import cbct_taxonomy
from dentist_ai.ml import taxonomy as flat_taxonomy
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale, Severity


class FindingSource(enum.StrEnum):
    """Which examination a finding came from.

    Kept on every input because it changes what the plan may assert. A CBCT
    can measure a lesion's relationship to the canal; a panoramic radiograph
    cannot, and a plan built only from one should not imply it did.
    """

    RADIOGRAPH = "radiograph"
    CBCT = "cbct"


@dataclass(frozen=True, slots=True)
class PlannedFinding:
    """The subset of a finding the planner reasons over.

    Deliberately not an ORM row: the planner is pure, so it can be exercised
    against a hand-written list in a test without a database, and so nothing
    here can accidentally reach the patient's name.
    """

    class_key: str
    source: FindingSource
    confidence: float
    tooth_number: int | None = None
    region: str | None = None
    #: Longest extent in millimetres, where the finding class carries one.
    extent_mm: float | None = None
    #: Whether a clinician has confirmed it. Rejected findings never reach
    #: this module; confirmed ones outrank unreviewed ones in ordering.
    confirmed: bool = False

    @property
    def severity(self) -> Severity:
        if self.source is FindingSource.CBCT:
            return cbct_taxonomy.by_key(self.class_key).severity
        return flat_taxonomy.by_key(self.class_key).severity

    def label(self, locale: Locale = DEFAULT_LOCALE) -> str:
        if self.source is FindingSource.CBCT:
            return cbct_taxonomy.by_key(self.class_key).label(locale)
        return flat_taxonomy.by_key(self.class_key).label(locale)

    def procedure_codes(self) -> tuple[str, ...]:
        if self.source is FindingSource.CBCT:
            return cbct_taxonomy.by_key(self.class_key).procedures
        return tuple(item.code for item in protocols.procedures_for(self.class_key))


@dataclass(frozen=True, slots=True)
class PlannedStep:
    """One procedure in a proposed plan, on a tooth or on the whole mouth."""

    procedure: Procedure
    tooth_number: int | None
    #: Which finding put this step in the plan. Every step traces to one.
    reason: str
    source_class_key: str
    priority: Priority
    #: Position in the constraint chain, not merely the display order.
    sequence: int

    @property
    def code(self) -> str:
        return self.procedure.code


@dataclass(frozen=True, slots=True)
class TreatmentOption:
    """One defensible way to treat the same case."""

    approach: TreatmentApproach
    title: str
    summary: str
    steps: tuple[PlannedStep, ...]
    priority: Priority
    complexity: PlanComplexity
    visits: int
    minutes: int
    weeks: int
    benefits: str
    risks: str

    @property
    def procedure_codes(self) -> tuple[str, ...]:
        """Just the codes, for display."""
        return tuple(step.code for step in self.steps)

    def encoded_steps(self) -> str:
        """The steps as ``code:tooth`` pairs, for storage.

        The tooth number has to survive: two teeth each needing a root canal
        are two appointments and two line items on an estimate, and a plan that
        collapses them into one is wrong about both the cost and the schedule.
        """
        return ",".join(f"{step.code}:{step.tooth_number or ''}" for step in self.steps)


@dataclass(frozen=True, slots=True)
class ProposedPlan:
    """Everything the planner concluded, ready to be persisted or discarded."""

    title: str
    options: tuple[TreatmentOption, ...]
    #: The option the planner would start from. Never auto-accepted.
    recommended: TreatmentApproach
    priority: Priority
    complexity: PlanComplexity
    estimated_weeks: int
    risks: tuple[str, ...]
    follow_up: str
    rationale: str
    #: Findings that produced no step, with why — an empty plan beside a long
    #: finding list otherwise reads as the planner having failed.
    unaddressed: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------
#: Where each procedure category sits in the constraint chain. Lower runs
#: first, and the order is causal rather than aesthetic: a diagnostic referral
#: informs the surgery that follows it, acute infection is controlled before
#: anything is restored, and a site is not rebuilt until it has healed.
_SEQUENCE_RANK: Final[dict[ProcedureCategory, int]] = {
    ProcedureCategory.DIAGNOSTICS: 0,
    ProcedureCategory.ENDODONTICS: 1,
    ProcedureCategory.SURGERY: 2,
    ProcedureCategory.PERIODONTICS: 3,
    ProcedureCategory.THERAPY: 4,
    ProcedureCategory.ORTHODONTICS: 5,
    ProcedureCategory.PROSTHETICS: 6,
}

#: Procedures whose site has to heal before the next stage can begin, and for
#: how long. This is the difference between chair time and calendar time: an
#: implant case is two hours of work spread across four months, and quoting the
#: two hours is how a patient ends up feeling misled.
_HEALING_WEEKS: Final[dict[str, int]] = {
    "extraction": 8,
    "root_removal": 8,
    "cyst_removal": 12,
    "bone_graft": 20,
    "implant_placement": 14,
}

#: Weeks between ordinary appointments when nothing has to heal.
_APPOINTMENT_INTERVAL_WEEKS: Final[int] = 2

#: Which approaches admit which priorities. The conservative option is not a
#: worse plan — it is the right plan for a patient who wants the problem
#: treated and nothing else, so it carries everything urgent and stops there.
_APPROACH_PRIORITIES: Final[dict[TreatmentApproach, frozenset[Priority]]] = {
    TreatmentApproach.CONSERVATIVE: frozenset({Priority.URGENT, Priority.HIGH}),
    TreatmentApproach.STANDARD: frozenset({Priority.URGENT, Priority.HIGH, Priority.ROUTINE}),
    TreatmentApproach.COMPREHENSIVE: frozenset(
        {Priority.URGENT, Priority.HIGH, Priority.ROUTINE, Priority.OPTIONAL}
    ),
}

_APPROACH_LABELS: Final[dict[TreatmentApproach, dict[Locale, str]]] = {
    TreatmentApproach.CONSERVATIVE: {
        "ru": "Минимально необходимое",
        "en": "Minimum necessary",
        "kk": "Ең қажетті",
    },
    TreatmentApproach.STANDARD: {
        "ru": "Стандартный объём",
        "en": "Standard scope",
        "kk": "Стандартты көлем",
    },
    TreatmentApproach.COMPREHENSIVE: {
        "ru": "Комплексная реабилитация",
        "en": "Comprehensive rehabilitation",
        "kk": "Кешенді оңалту",
    },
}

_APPROACH_SUMMARIES: Final[dict[TreatmentApproach, dict[Locale, str]]] = {
    TreatmentApproach.CONSERVATIVE: {
        "ru": (
            "Устранение источников боли и инфекции. Остальные находки остаются "
            "под наблюдением с контрольным снимком."
        ),
        "en": (
            "Resolve the sources of pain and infection. Everything else stays "
            "under observation with a follow-up image."
        ),
        "kk": "Ауырсыну мен инфекция көздерін жою; қалғаны бақылауда қалады.",
    },
    TreatmentApproach.STANDARD: {
        "ru": (
            "Санация полости рта: неотложное лечение плюс плановое "
            "восстановление зубов, где это показано."
        ),
        "en": (
            "Full oral rehabilitation of what is diseased: urgent care plus the "
            "restorative work that is indicated."
        ),
        "kk": "Ауыз қуысын санациялау: шұғыл емдеу және жоспарлы қалпына келтіру.",
    },
    TreatmentApproach.COMPREHENSIVE: {
        "ru": (
            "Полная реабилитация, включая замещение дефектов и коррекцию "
            "прикуса. Наибольший объём и наибольший срок."
        ),
        "en": (
            "Full rehabilitation including replacing missing teeth and "
            "correcting the bite. The largest scope and the longest timeline."
        ),
        "kk": "Толық оңалту, оның ішінде ақауларды алмастыру және прикусты түзету.",
    },
}

_APPROACH_BENEFITS: Final[dict[TreatmentApproach, dict[Locale, str]]] = {
    TreatmentApproach.CONSERVATIVE: {
        "ru": "Наименьший объём вмешательства, наименьшая стоимость, быстрый результат.",
        "en": "Least intervention, lowest cost, fastest relief.",
        "kk": "Ең аз араласу, ең төмен құн, жылдам нәтиже.",
    },
    TreatmentApproach.STANDARD: {
        "ru": "Устраняет активные процессы и восстанавливает функцию жевания.",
        "en": "Resolves active disease and restores chewing function.",
        "kk": "Белсенді процестерді жояды және шайнау қызметін қалпына келтіреді.",
    },
    TreatmentApproach.COMPREHENSIVE: {
        "ru": "Восстанавливает функцию и эстетику полностью, снижает риск повторных проблем.",
        "en": "Restores function and appearance in full, and lowers the risk of recurrence.",
        "kk": "Қызмет пен эстетиканы толық қалпына келтіреді.",
    },
}

_APPROACH_RISKS: Final[dict[TreatmentApproach, dict[Locale, str]]] = {
    TreatmentApproach.CONSERVATIVE: {
        "ru": (
            "Находки, оставленные под наблюдением, могут прогрессировать и "
            "потребовать большего объёма лечения позже."
        ),
        "en": (
            "Findings left under observation may progress and need more extensive treatment later."
        ),
        "kk": "Бақылауда қалған табылымдар үдеп, кейін көбірек емдеуді талап етуі мүмкін.",
    },
    TreatmentApproach.STANDARD: {
        "ru": "Требует нескольких визитов и дисциплины пациента в соблюдении плана.",
        "en": "Needs several appointments and patient adherence to the schedule.",
        "kk": "Бірнеше рет келуді және жоспарды сақтауды талап етеді.",
    },
    TreatmentApproach.COMPREHENSIVE: {
        "ru": (
            "Наибольшая длительность и стоимость; часть этапов зависит от "
            "заживления и может сместиться по срокам."
        ),
        "en": (
            "The longest and most expensive course; several stages depend on "
            "healing and their dates can move."
        ),
        "kk": "Ең ұзақ және қымбат; кейбір кезеңдер жазылуға байланысты.",
    },
}


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _RiskRule:
    """A risk that exists only when a finding and a procedure meet.

    Written as a pair rather than as a property of either one, because that is
    what makes the warning worth showing: a clinician already knows extraction
    carries risk in general, and a warning that fires on every extraction is a
    warning that gets ignored.
    """

    finding_keys: frozenset[str]
    procedure_codes: frozenset[str]
    text: dict[Locale, str]


_RISK_RULES: Final[tuple[_RiskRule, ...]] = (
    _RiskRule(
        frozenset({"mandibular_canal", "impacted_third_molar"}),
        frozenset({"extraction", "root_removal", "surgical_consult", "implant_placement"}),
        {
            "ru": (
                "Нижнечелюстной канал проходит близко к зоне вмешательства: риск "
                "повреждения нижнего альвеолярного нерва. Измерьте расстояние по "
                "срезам до операции."
            ),
            "en": (
                "The mandibular canal runs close to the surgical site: risk of "
                "inferior alveolar nerve injury. Measure the clearance on the "
                "slices before operating."
            ),
            "kk": "Төменгі жақ өзегі араласу аймағына жақын: нерв зақымдану қаупі.",
        },
    ),
    _RiskRule(
        frozenset({"sinus_proximity", "maxillary_sinus"}),
        frozenset({"implant_placement", "extraction", "bone_graft"}),
        {
            "ru": (
                "Дно верхнечелюстной пазухи находится близко: риск перфорации. "
                "Оцените остаточную высоту кости и рассмотрите синус-лифтинг."
            ),
            "en": (
                "The maxillary sinus floor is close: risk of perforation. Check "
                "the residual bone height and consider a sinus lift."
            ),
            "kk": "Гайморит қуысының түбі жақын: перфорация қаупі.",
        },
    ),
    _RiskRule(
        frozenset({"bone_loss_3d", "bone_loss", "periodontal_disease"}),
        frozenset({"implant_placement", "crown_restoration", "prosthetic_plan"}),
        {
            "ru": (
                "На фоне убыли костной ткани ортопедическая конструкция без "
                "предварительной стабилизации пародонта имеет сниженный прогноз."
            ),
            "en": (
                "With existing bone loss, a prosthetic restoration placed before "
                "the periodontium is stabilised has a reduced prognosis."
            ),
            "kk": "Сүйек жоғалуы аясында ортопедиялық конструкцияның болжамы төмен.",
        },
    ),
    _RiskRule(
        frozenset({"root_fracture", "tooth_fracture"}),
        frozenset({"root_canal", "endo_retreatment", "crown_restoration"}),
        {
            "ru": (
                "При подозрении на перелом корня эндодонтическое лечение может "
                "оказаться бесперспективным — подтвердите целостность корня до начала."
            ),
            "en": (
                "Where a root fracture is suspected, endodontic treatment may be "
                "futile — confirm root integrity before starting."
            ),
            "kk": "Тамыр сынуы күдігі кезінде эндодонтиялық емдеу нәтижесіз болуы мүмкін.",
        },
    ),
    _RiskRule(
        frozenset({"suspicious_mass"}),
        frozenset(),
        {
            "ru": (
                "В объёме выявлено образование, требующее верификации. Плановое "
                "стоматологическое лечение следует отложить до заключения "
                "профильного специалиста."
            ),
            "en": (
                "A lesion requiring verification was reported. Elective dental "
                "treatment should wait for a specialist opinion."
            ),
            "kk": "Тексеруді талап ететін түзіліс анықталды: жоспарлы емдеуді кейінге қалдырыңыз.",
        },
    ),
    _RiskRule(
        frozenset({"motion_artifact"}),
        frozenset({"implant_placement", "bone_graft", "extraction"}),
        {
            "ru": (
                "Снимок содержит артефакты движения: измерения по нему менее "
                "надёжны. Для хирургического планирования рассмотрите повторное "
                "сканирование."
            ),
            "en": (
                "The scan carries motion artefact, so measurements taken from it "
                "are less reliable. Consider repeating it before surgical planning."
            ),
            "kk": "Суретте қозғалыс артефактісі бар: өлшемдер сенімділігі төмен.",
        },
    ),
)

#: Follow-up interval by the most severe finding in the plan, in months.
_FOLLOW_UP_MONTHS: Final[dict[Severity, int]] = {
    Severity.CRITICAL: 3,
    Severity.HIGH: 6,
    Severity.MEDIUM: 12,
    Severity.LOW: 12,
    Severity.INFO: 12,
}

#: Thresholds that separate the complexity bands. Chosen against what a
#: practice actually schedules: up to three appointments is a short course,
#: and anything crossing three specialities is a case that needs coordinating.
_COMPLEX_VISITS: Final[int] = 6
_MODERATE_VISITS: Final[int] = 3
_ADVANCED_SPECIALITIES: Final[int] = 3

#: Cap on steps in one proposal. A plan longer than this is not a plan, it is
#: a list, and it should be split across courses of treatment.
_MAX_STEPS: Final[int] = 24


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def propose(
    findings: Sequence[PlannedFinding],
    *,
    locale: Locale = DEFAULT_LOCALE,
    quality_score: float | None = None,
) -> ProposedPlan | None:
    """Build a plan from a set of findings, or ``None`` if none implies work.

    ``quality_score`` is the acquisition-quality figure from the CBCT pipeline
    where one exists. It never removes a step — it adds a caveat, because a
    plan built on a scan the clinician was told to distrust should say so.
    """
    steps = _steps_for(findings)
    if not steps:
        return None

    options = _distinct_options(steps, locale)
    if not options:
        return None

    present = {finding.class_key for finding in findings}
    risks = _risks_for(present, steps, locale, quality_score)
    worst = min((finding.severity for finding in findings), key=lambda item: item.rank)
    standard = next(
        (item for item in options if item.approach is TreatmentApproach.STANDARD),
        options[0],
    )

    return ProposedPlan(
        title=_title(worst, locale),
        options=options,
        recommended=standard.approach,
        priority=standard.priority,
        complexity=standard.complexity,
        estimated_weeks=standard.weeks,
        risks=risks,
        follow_up=_follow_up(worst, locale),
        rationale=_rationale(findings, steps, locale),
        unaddressed=_unaddressed(findings, locale),
    )


def _distinct_options(steps: Sequence[PlannedStep], locale: Locale) -> tuple[TreatmentOption, ...]:
    """Build the three approaches, dropping any that duplicate a narrower one.

    When a case has nothing elective in it, the comprehensive option contains
    exactly the standard one's steps. Presenting both would ask the clinician
    to choose between two identical plans, which reads as a bug and erodes
    trust in the ones that genuinely do differ.
    """
    options: list[TreatmentOption] = []
    seen: set[tuple[tuple[str, int | None], ...]] = set()

    for approach in TreatmentApproach:
        option = _build_option(approach, steps, locale)
        if option is None:
            continue
        signature = tuple((step.code, step.tooth_number) for step in option.steps)
        if signature in seen:
            continue
        seen.add(signature)
        options.append(option)
    return tuple(options)


def _steps_for(findings: Sequence[PlannedFinding]) -> tuple[PlannedStep, ...]:
    """Expand findings into steps, one per (procedure, tooth) pair.

    Deduplicated on that pair rather than on the procedure alone: two carious
    teeth are two restorations, but one tooth with two findings that both
    imply a crown is still one crown.
    """
    seen: set[tuple[str, int | None]] = set()
    steps: list[PlannedStep] = []

    # Confirmed findings first, then by severity, so that when the step cap
    # bites it drops the least certain and least urgent work.
    ordered = sorted(
        findings,
        key=lambda item: (not item.confirmed, item.severity.rank, -item.confidence),
    )

    for finding in ordered:
        for code in finding.procedure_codes():
            procedure = protocols.by_code(code)
            if procedure is None:
                continue
            key = (code, finding.tooth_number)
            if key in seen:
                continue
            seen.add(key)
            steps.append(
                PlannedStep(
                    procedure=procedure,
                    tooth_number=finding.tooth_number,
                    reason=finding.label(),
                    source_class_key=finding.class_key,
                    priority=procedure.priority,
                    sequence=_SEQUENCE_RANK.get(procedure.category, len(_SEQUENCE_RANK)),
                )
            )

    steps.sort(key=lambda item: (item.priority.rank, item.sequence, item.code))
    return tuple(steps[:_MAX_STEPS])


def _build_option(
    approach: TreatmentApproach,
    steps: Sequence[PlannedStep],
    locale: Locale,
) -> TreatmentOption | None:
    admitted = _APPROACH_PRIORITIES[approach]
    selected = tuple(step for step in steps if step.priority in admitted)
    if not selected:
        return None

    visits = sum(step.procedure.visits for step in selected)
    minutes = sum(step.procedure.visits * step.procedure.minutes for step in selected)
    weeks = _calendar_weeks(selected)
    priority = min((step.priority for step in selected), key=lambda item: item.rank)

    return TreatmentOption(
        approach=approach,
        title=_localised(_APPROACH_LABELS[approach], locale),
        summary=_localised(_APPROACH_SUMMARIES[approach], locale),
        steps=selected,
        priority=priority,
        complexity=_complexity(selected, visits),
        visits=visits,
        minutes=minutes,
        weeks=weeks,
        benefits=_localised(_APPROACH_BENEFITS[approach], locale),
        risks=_localised(_APPROACH_RISKS[approach], locale),
    )


def _calendar_weeks(steps: Sequence[PlannedStep]) -> int:
    """Elapsed weeks, not chair time.

    Appointments are spaced, and the healing steps dominate: an implant case
    is a couple of hours of work over several months. Healing runs once per
    procedure kind rather than once per tooth, because sites that heal in
    parallel do heal in parallel.
    """
    appointments = sum(step.procedure.visits for step in steps)
    spacing = max(appointments - 1, 0) * _APPOINTMENT_INTERVAL_WEEKS
    healing = sum(_HEALING_WEEKS.get(code, 0) for code in {step.code for step in steps})
    return max(1, spacing + healing)


def _complexity(steps: Sequence[PlannedStep], visits: int) -> PlanComplexity:
    specialities = {step.procedure.category for step in steps}
    if len(specialities) >= _ADVANCED_SPECIALITIES and visits >= _COMPLEX_VISITS:
        return PlanComplexity.ADVANCED
    if visits >= _COMPLEX_VISITS:
        return PlanComplexity.COMPLEX
    if visits > _MODERATE_VISITS:
        return PlanComplexity.MODERATE
    return PlanComplexity.SIMPLE


def _risks_for(
    present: set[str],
    steps: Sequence[PlannedStep],
    locale: Locale,
    quality_score: float | None,
) -> tuple[str, ...]:
    planned = {step.code for step in steps}
    risks = [
        _localised(rule.text, locale)
        for rule in _RISK_RULES
        if (present & rule.finding_keys)
        and (not rule.procedure_codes or (planned & rule.procedure_codes))
    ]

    if quality_score is not None and quality_score < _LOW_QUALITY:
        risks.append(
            _localised(
                {
                    "ru": (
                        "Качество исходного снимка оценено как низкое — план "
                        "следует подтвердить дополнительным обследованием."
                    ),
                    "en": (
                        "The source scan scored low on quality — confirm this "
                        "plan against a further examination."
                    ),
                    "kk": "Бастапқы суреттің сапасы төмен — жоспарды растаңыз.",
                },
                locale,
            )
        )
    return tuple(risks)


#: Below this the acquisition-quality score is worth putting in front of the
#: clinician as a caveat on the plan itself, not only on the findings.
_LOW_QUALITY: Final[float] = 0.55


def _follow_up(worst: Severity, locale: Locale) -> str:
    months = _FOLLOW_UP_MONTHS[worst]
    table = {
        "ru": f"Контрольный осмотр и снимок через {months} мес. после завершения лечения.",
        "en": f"Review with imaging {months} months after treatment is complete.",
        "kk": f"Емдеу аяқталғаннан кейін {months} айдан соң бақылау.",
    }
    return _localised(table, locale)


def _title(worst: Severity, locale: Locale) -> str:
    if worst in (Severity.CRITICAL, Severity.HIGH):
        table = {
            "ru": "План лечения по результатам анализа",
            "en": "Treatment plan from the analysis",
            "kk": "Талдау нәтижесі бойынша емдеу жоспары",
        }
    else:
        table = {
            "ru": "Плановое лечение и наблюдение",
            "en": "Elective treatment and review",
            "kk": "Жоспарлы емдеу және бақылау",
        }
    return _localised(table, locale)


def _rationale(
    findings: Sequence[PlannedFinding],
    steps: Sequence[PlannedStep],
    locale: Locale,
) -> str:
    """One sentence naming what drove the plan.

    Composed from counts rather than generated prose: the plan has to be
    explainable, and a sentence assembled from the same numbers the table
    shows can be checked against it.
    """
    attention = sum(
        1 for finding in findings if finding.severity in (Severity.CRITICAL, Severity.HIGH)
    )
    sources = {finding.source for finding in findings}
    basis = {
        "ru": "КЛКТ и рентгенограммы"
        if len(sources) > 1
        else ("КЛКТ" if FindingSource.CBCT in sources else "рентгенограммы"),
        "en": "CBCT and radiographs"
        if len(sources) > 1
        else ("CBCT" if FindingSource.CBCT in sources else "radiographs"),
        "kk": "КЛКТ және рентгенограммалар"
        if len(sources) > 1
        else ("КЛКТ" if FindingSource.CBCT in sources else "рентгенограммалар"),
    }
    stage_word = plural_ru(len(steps), "этап", "этапа", "этапов")
    finding_word = plural_ru(len(findings), "находке", "находкам", "находкам")
    table = {
        "ru": (
            f"План собран по {len(findings)} {finding_word} ({basis['ru']}), "
            f"из них требующих внимания — {attention}. "
            f"Предложено {len(steps)} {stage_word}, сгруппированных по срочности."
        ),
        "en": (
            f"Assembled from {len(findings)} findings ({basis['en']}), "
            f"{attention} of which need attention. "
            f"{len(steps)} steps proposed, grouped by urgency."
        ),
        "kk": (
            f"Жоспар {len(findings)} табылым бойынша құрылды ({basis['kk']}), "
            f"олардың {attention} назар аударуды талап етеді."
        ),
    }
    return _localised(table, locale)


def _unaddressed(findings: Sequence[PlannedFinding], locale: Locale) -> tuple[str, ...]:
    """Findings that imply no work, named so the omission is deliberate.

    A crown, an implant and the mandibular canal are records of past work or
    normal anatomy — they belong in the report and not in the plan. Saying so
    is what stops a short plan beside a long finding list reading as a bug.
    """
    silent = sorted(
        {finding.label(locale) for finding in findings if not finding.procedure_codes()}
    )
    return tuple(silent)


def _localised(table: dict[Locale, str], locale: Locale) -> str:
    return table.get(locale) or table[DEFAULT_LOCALE]


def approach_label(approach: TreatmentApproach, locale: Locale = DEFAULT_LOCALE) -> str:
    return _localised(_APPROACH_LABELS[approach], locale)


def complexity_label(complexity: PlanComplexity, locale: Locale = DEFAULT_LOCALE) -> str:
    table: dict[PlanComplexity, dict[Locale, str]] = {
        PlanComplexity.SIMPLE: {"ru": "Простой", "en": "Simple", "kk": "Қарапайым"},
        PlanComplexity.MODERATE: {"ru": "Средний", "en": "Moderate", "kk": "Орташа"},
        PlanComplexity.COMPLEX: {"ru": "Сложный", "en": "Complex", "kk": "Күрделі"},
        PlanComplexity.ADVANCED: {
            "ru": "Комплексный, междисциплинарный",
            "en": "Advanced, multidisciplinary",
            "kk": "Кешенді, пәнаралық",
        },
    }
    return _localised(table[complexity], locale)


def findings_from(
    radiograph_keys: Iterable[tuple[str, float, int | None, bool]] = (),
    volume_keys: Iterable[tuple[str, float, int | None, str | None, float | None, bool]] = (),
) -> tuple[PlannedFinding, ...]:
    """Adapter from stored rows to planner inputs.

    Takes tuples rather than ORM objects so the service layer does the row
    reading and this module stays free of the database.
    """
    planned: list[PlannedFinding] = []
    for class_key, confidence, tooth, confirmed in radiograph_keys:
        planned.append(
            PlannedFinding(
                class_key=class_key,
                source=FindingSource.RADIOGRAPH,
                confidence=confidence,
                tooth_number=tooth,
                confirmed=confirmed,
            )
        )
    for class_key, confidence, tooth, region, extent, confirmed in volume_keys:
        planned.append(
            PlannedFinding(
                class_key=class_key,
                source=FindingSource.CBCT,
                confidence=confidence,
                tooth_number=tooth,
                region=region,
                extent_mm=extent,
                confirmed=confirmed,
            )
        )
    return tuple(planned)
