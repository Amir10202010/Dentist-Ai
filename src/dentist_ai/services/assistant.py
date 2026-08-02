"""The case assistant: questions about *this* patient, answered from the record.

What this is, stated plainly, because the word "assistant" invites the wrong
assumption: it is a retrieval and composition engine over one case, not a
language model. A question is matched to an intent, the intent's handler reads
the relevant rows, and the answer is assembled from them. It does not
generalise, it does not reason beyond the tables it reads, and it cannot
produce a claim that is not already in the record.

That is a narrower product than a chatbot and a considerably safer one. Every
sentence it emits traces to a row, and every answer carries the citations to
prove it — which is the property that makes it usable on patient data at all.
A model that fluently invents a measurement is worse than no assistant, and on
a clinical record the difference is not academic.

The honest limitation is coverage: a question the router does not recognise
gets a list of what it *can* answer rather than a guess. :data:`_INTENTS` is
the whole vocabulary, and widening it is adding a handler, not retraining
anything.

Where a real language model belongs in this design is as a *rephraser* of
answers this module has already grounded — turning the composed text into
better prose without being allowed to add facts. The seam for that is
:meth:`AssistantService.answer`, which returns structured text plus citations
rather than a finished string.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.clinical import charting, protocols
from dentist_ai.clinical.labels import field_of_view_label, quality_label
from dentist_ai.core.errors import NotFoundError, ValidationError
from dentist_ai.core.ids import generate_public_id
from dentist_ai.core.text import plural_ru
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import (
    AssistantMessage,
    AssistantRole,
    AssistantThread,
    CaseEntry,
    FindingReview,
    Patient,
    Study,
    TreatmentPlan,
    User,
    Volume,
    VolumeFinding,
)
from dentist_ai.ml import cbct_taxonomy
from dentist_ai.ml.taxonomy import Severity
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext


class Intent(enum.StrEnum):
    """The question shapes the assistant recognises."""

    SUMMARISE = "summarise"
    EXPLAIN_FINDING = "explain_finding"
    WHY_DETECTED = "why_detected"
    TREATMENTS = "treatments"
    NEXT_CHECKS = "next_checks"
    SIMILAR_CASES = "similar_cases"
    PATIENT_FRIENDLY = "patient_friendly"
    MEASUREMENTS = "measurements"
    QUALITY = "quality"
    CAPABILITIES = "capabilities"


@dataclass(frozen=True, slots=True)
class Citation:
    """A record the answer rests on."""

    kind: str
    label: str
    href: str | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    intent: Intent
    body: str
    citations: tuple[Citation, ...] = ()
    #: Questions worth asking next, so the clinician is not left guessing at
    #: the vocabulary.
    suggestions: tuple[str, ...] = ()


@dataclass(slots=True)
class CaseContext:
    """Everything the assistant is allowed to see, loaded once per question."""

    patient: Patient
    volume: Volume | None = None
    studies: list[Study] = field(default_factory=list)
    plans: list[TreatmentPlan] = field(default_factory=list)

    @property
    def volume_findings(self) -> list[VolumeFinding]:
        if self.volume is None:
            return []
        # A rejected finding is a clinician saying the model was wrong, so the
        # assistant must not keep explaining it back to them.
        return [
            item for item in self.volume.findings if item.review is not FindingReview.REJECTED
        ]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
#: Keyword patterns per intent, Russian first because that is the interface
#: language. Matched against a normalised question — accents stripped, case
#: folded — rather than parsed, because the useful question space here is a
#: couple of dozen phrasings and a parser would be machinery without a payoff.
_INTENTS: Final[tuple[tuple[Intent, tuple[str, ...]], ...]] = (
    (
        Intent.WHY_DETECTED,
        ("почему", "на основании чего", "как определ", "why", "how did"),
    ),

    (
        Intent.TREATMENTS,
        ("лечен", "что делать", "варианты", "план", "treat", "options"),
    ),
    (
        Intent.NEXT_CHECKS,
        ("проверить", "дальше", "следующ", "дообследов", "next", "check"),
    ),
    (
        Intent.SIMILAR_CASES,
        ("похож", "аналогич", "такие же", "similar", "cases"),
    ),
    (
        Intent.PATIENT_FRIENDLY,
        ("пациент", "простыми словами", "простым языком", "patient", "plain"),
    ),
    # Tested after the patient-facing intent: "объясни" opens both questions,
    # and the one that also says "для пациента" is the more specific request.
    (
        Intent.EXPLAIN_FINDING,
        ("объясни", "что такое", "расскажи", "что значит", "explain", "what is"),
    ),
    (
        Intent.MEASUREMENTS,
        ("измер", "размер", "сколько мм", "расстояние", "measure", "size"),
    ),
    (
        Intent.QUALITY,
        ("качеств", "надёж", "надеж", "достоверн", "quality", "reliable"),
    ),
    (
        Intent.SUMMARISE,
        ("сводк", "кратко", "итог", "обзор", "что на снимке", "summar", "overview"),
    ),
)

_SUGGESTIONS: Final[tuple[str, ...]] = (
    "Кратко опиши этот снимок",
    "Почему AI это нашёл?",
    "Какие возможны варианты лечения?",
    "Что проверить на приёме?",
    "Объясни простыми словами для пациента",
    "Насколько надёжен этот снимок?",
    "Покажи похожие случаи",
)

#: Findings quoted in a summary before it starts counting instead. Past this
#: the answer stops being a summary.
_SUMMARY_LIMIT: Final[int] = 5
#: Similar cases returned. A shortlist is useful; a search result is not an
#: answer to "show me similar cases".
_SIMILAR_LIMIT: Final[int] = 5
#: Confidence below which the assistant says so out loud when explaining a
#: finding, rather than reporting the number and letting it pass as settled.
_LOW_CONFIDENCE: Final[float] = 0.5
#: Quality below which the assistant volunteers that measurements are less
#: dependable, instead of leaving the reader to interpret a bare score.
_QUALITY_CAVEAT: Final[float] = 0.7
#: Name-part counts that tell surname-first records apart.
_NAME_WITH_SURNAME: Final[int] = 2
_NAME_WITH_PATRONYMIC: Final[int] = 3


def route(question: str) -> Intent:
    """Match a question to an intent, or :attr:`Intent.CAPABILITIES`.

    Ordered rather than scored: "объясни простыми словами пациенту" is a
    patient-facing request even though it also opens with "объясни", so the
    more specific intents are tested first and the first match wins.

    Matching is on **token prefixes**, not raw substrings. A plain `in` test
    looks equivalent and routes "сколько стоит имплант" to the treatment
    handler, because "имплант" contains "план" — the kind of wrong answer that
    is hard to see in review and obvious to a user. Prefixes still let one
    pattern cover a whole Russian paradigm: "лечен" matches лечение, лечения
    and лечению.
    """
    normalised = re.sub(r"[^\w\s]", " ", question.lower())
    tokens = normalised.split()
    for intent, patterns in _INTENTS:
        for pattern in patterns:
            if " " in pattern:
                if pattern in normalised:
                    return intent
            elif any(token.startswith(pattern) for token in tokens):
                return intent
    return Intent.CAPABILITIES


class AssistantService:
    def __init__(self, session: AsyncSession, audit: AuditService) -> None:
        self._session = session
        self._audit = audit

    # -- conversation -----------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        actor: User,
        context: RequestContext,
        patient_id: int | None = None,
        volume_public_id: str | None = None,
        thread_public_id: str | None = None,
    ) -> tuple[AssistantThread, Answer]:
        """Answer one question and record both turns on a thread."""
        text = question.strip()
        if not text:
            raise ValidationError("Вопрос не может быть пустым.")

        thread = await self._thread(
            actor=actor,
            patient_id=patient_id,
            volume_public_id=volume_public_id,
            thread_public_id=thread_public_id,
        )
        case = await self._load_case(thread, actor.organization_id)
        answer = await self.answer(text, case, locale=actor.locale)

        now = utcnow()
        self._session.add(
            AssistantMessage(
                thread_id=thread.id,
                role=AssistantRole.USER,
                body=text,
                created_at=now,
            )
        )
        self._session.add(
            AssistantMessage(
                thread_id=thread.id,
                role=AssistantRole.ASSISTANT,
                body=answer.body,
                intent=answer.intent.value,
                citations=json.dumps(
                    [
                        {"kind": item.kind, "label": item.label, "href": item.href}
                        for item in answer.citations
                    ],
                    ensure_ascii=False,
                ),
                created_at=now,
            )
        )
        if thread.title == "Новый разговор":
            thread.title = text[:160]
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.ASSISTANT_QUERIED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="assistant_thread",
            resource_id=thread.public_id,
            context=context,
            detail=answer.intent.value,
        )
        return thread, answer

    async def answer(self, question: str, case: CaseContext, *, locale: str = "ru") -> Answer:
        """Compose an answer without touching the conversation record.

        Separated from :meth:`ask` so the composition can be exercised in a
        test against a hand-built context, and so a future rephrasing layer has
        a grounded answer to work from rather than a free hand.
        """
        intent = route(question)
        handler: Callable[[str, CaseContext, str], Answer] = _HANDLERS[intent]
        return handler(question, case, locale)

    async def history(
        self, thread_public_id: str, *, actor: User
    ) -> tuple[AssistantThread, list[AssistantMessage]]:
        thread = await self._session.scalar(
            select(AssistantThread)
            .options(selectinload(AssistantThread.messages))
            .where(
                AssistantThread.public_id == thread_public_id,
                AssistantThread.organization_id == actor.organization_id,
                AssistantThread.user_id == actor.id,
            )
        )
        if thread is None:
            raise NotFoundError("Разговор не найден.")
        return thread, list(thread.messages)

    async def threads(self, *, actor: User, limit: int = 20) -> list[AssistantThread]:
        rows = await self._session.scalars(
            select(AssistantThread)
            .where(
                AssistantThread.organization_id == actor.organization_id,
                AssistantThread.user_id == actor.id,
            )
            .order_by(AssistantThread.created_at.desc())
            .limit(limit)
        )
        return list(rows)

    # -- internals --------------------------------------------------------
    async def _thread(
        self,
        *,
        actor: User,
        patient_id: int | None,
        volume_public_id: str | None,
        thread_public_id: str | None,
    ) -> AssistantThread:
        if thread_public_id is not None:
            thread, _ = await self.history(thread_public_id, actor=actor)
            return thread

        volume_id: int | None = None
        if volume_public_id is not None:
            volume_id = await self._session.scalar(
                select(Volume.id).where(
                    Volume.public_id == volume_public_id,
                    Volume.organization_id == actor.organization_id,
                )
            )
            if volume_id is None:
                raise NotFoundError("КЛКТ-исследование не найдено.")

        if patient_id is None and volume_id is None:
            raise ValidationError("Укажите пациента или исследование для контекста.")

        thread = AssistantThread(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            user_id=actor.id,
            patient_id=patient_id,
            volume_id=volume_id,
        )
        self._session.add(thread)
        await self._session.flush()
        return thread

    async def _load_case(self, thread: AssistantThread, organization_id: int) -> CaseContext:
        volume: Volume | None = None
        if thread.volume_id is not None:
            volume = await self._session.scalar(
                select(Volume)
                .options(
                    selectinload(Volume.findings),
                    selectinload(Volume.measurements),
                    selectinload(Volume.patient),
                )
                .where(Volume.id == thread.volume_id, Volume.organization_id == organization_id)
            )

        patient_id = thread.patient_id or (volume.patient_id if volume else None)
        if patient_id is None:
            raise NotFoundError("Не удалось определить пациента для этого разговора.")

        patient = await self._session.scalar(
            select(Patient).where(
                Patient.id == patient_id, Patient.organization_id == organization_id
            )
        )
        if patient is None:
            raise NotFoundError("Пациент не найден.")

        studies = list(
            await self._session.scalars(
                select(Study)
                .options(selectinload(Study.findings))
                .where(Study.patient_id == patient.id, Study.organization_id == organization_id)
                .order_by(Study.created_at.desc())
                .limit(5)
            )
        )
        plans = list(
            await self._session.scalars(
                select(TreatmentPlan)
                .options(selectinload(TreatmentPlan.items))
                .where(
                    TreatmentPlan.patient_id == patient.id,
                    TreatmentPlan.organization_id == organization_id,
                )
                .order_by(TreatmentPlan.created_at.desc())
                .limit(3)
            )
        )
        return CaseContext(patient=patient, volume=volume, studies=studies, plans=plans)

    async def similar_cases(self, case: CaseContext, *, organization_id: int) -> list[CaseEntry]:
        """Library entries sharing this case's finding classes.

        Ranked by how many classes overlap, which is a crude similarity and an
        honest one: it is exactly "cases that had some of the same things", and
        the interface says so rather than implying a learned embedding.
        """
        keys = {item.class_key for item in case.volume_findings}
        if not keys:
            return []

        entries = list(
            await self._session.scalars(
                select(CaseEntry).where(CaseEntry.organization_id == organization_id).limit(200)
            )
        )
        scored = [
            (len(keys & set(entry.finding_keys.split(","))), entry)
            for entry in entries
            if entry.finding_keys
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _score, entry in scored[:_SIMILAR_LIMIT]]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _volume_href(case: CaseContext) -> str | None:
    return f"/app/volumes/{case.volume.public_id}" if case.volume else None


def _ranked(case: CaseContext) -> list[VolumeFinding]:
    return sorted(
        case.volume_findings,
        key=lambda item: (cbct_taxonomy.by_key(item.class_key).severity.rank, -item.confidence),
    )


def _no_scan(intent: Intent) -> Answer:
    return Answer(
        intent=intent,
        body=(
            "К этому разговору не привязано КЛКТ-исследование, поэтому "
            "отвечать не на чем. Откройте исследование пациента и задайте "
            "вопрос оттуда."
        ),
        suggestions=_SUGGESTIONS[:3],
    )


def _handle_summarise(question: str, case: CaseContext, locale: str) -> Answer:
    _ = question
    if case.volume is None:
        return _no_scan(Intent.SUMMARISE)

    findings = _ranked(case)
    attention = [
        item for item in findings if cbct_taxonomy.by_key(item.class_key).needs_attention
    ]
    lines = [
        f"Исследование {case.volume.original_filename}, "
        f"поле зрения — {field_of_view_label(case.volume.field_of_view, locale).lower()}. "
        f"Найдено {len(findings)} "
        f"{plural_ru(len(findings), 'находка', 'находки', 'находок')}, "
        f"из них требующих внимания — {len(attention)}."
    ]
    if case.volume.quality_score is not None:
        lines.append(
            f"Качество исследования: {quality_label(case.volume.quality_score, locale).lower()} "
            f"({round(case.volume.quality_score * 100)} из 100)."
        )

    for item in findings[:_SUMMARY_LIMIT]:
        taxonomy = cbct_taxonomy.by_key(item.class_key)
        where = cbct_taxonomy.region_label(_region(item), locale)
        tooth = f", зуб {item.tooth_number}" if item.tooth_number else ""
        extent = f", протяжённость {item.extent_mm} мм" if item.extent_mm else ""
        lines.append(
            f"• {taxonomy.label(locale)} — {where}{tooth}{extent}; "
            f"уверенность {round(item.confidence * 100)}%."
        )
    if len(findings) > _SUMMARY_LIMIT:
        lines.append(f"…и ещё {len(findings) - _SUMMARY_LIMIT} — полный список в панели находок.")

    if not findings:
        lines.append("Патологических изменений автоматический анализ не выявил.")

    return Answer(
        intent=Intent.SUMMARISE,
        body="\n".join(lines),
        citations=(
            Citation("volume", case.volume.original_filename, _volume_href(case)),
        ),
        suggestions=("Почему AI это нашёл?", "Какие возможны варианты лечения?"),
    )


def _handle_explain(question: str, case: CaseContext, locale: str) -> Answer:
    if case.volume is None:
        return _no_scan(Intent.EXPLAIN_FINDING)

    target = _target_finding(question, case, locale)
    if target is None:
        return Answer(
            intent=Intent.EXPLAIN_FINDING,
            body=(
                "Не понял, о какой находке речь. Назовите её словами из списка "
                "находок — например «объясни периапикальное поражение»."
            ),
            suggestions=tuple(
                f"Объясни {cbct_taxonomy.by_key(item.class_key).label(locale).lower()}"
                for item in _ranked(case)[:3]
            ),
        )

    taxonomy = cbct_taxonomy.by_key(target.class_key)
    lines = [
        f"{taxonomy.label(locale)} — "
        f"{cbct_taxonomy.region_label(_region(target), locale).lower()}"
        + (f", зуб {target.tooth_number}" if target.tooth_number else "")
        + ".",
        f"Что это: {taxonomy.why(locale)}",
        f"Что делать: {taxonomy.what_next(locale)}",
    ]
    if target.extent_mm:
        lines.append(f"Измеренная протяжённость: {target.extent_mm} мм.")
    if target.mean_density is not None:
        lines.append(f"Средняя плотность в области: {round(target.mean_density)} HU.")
    if taxonomy.requires_confirmation:
        lines.append(
            "Важно: эта находка не является диагнозом и требует подтверждения "
            "профильным специалистом."
        )
    if target.confidence < _LOW_CONFIDENCE:
        lines.append(
            f"Уверенность модели низкая ({round(target.confidence * 100)}%) — "
            "отнеситесь к находке как к поводу посмотреть внимательнее, не более."
        )

    return Answer(
        intent=Intent.EXPLAIN_FINDING,
        body="\n".join(lines),
        citations=(Citation("finding", taxonomy.label(locale), _volume_href(case)),),
        suggestions=("Почему AI это нашёл?", "Какие возможны варианты лечения?"),
    )


def _handle_why(question: str, case: CaseContext, locale: str) -> Answer:
    if case.volume is None:
        return _no_scan(Intent.WHY_DETECTED)

    target = _target_finding(question, case, locale) or next(iter(_ranked(case)), None)
    if target is None:
        return Answer(
            intent=Intent.WHY_DETECTED,
            body="На этом исследовании находок нет, поэтому объяснять нечего.",
            suggestions=_SUGGESTIONS[:2],
        )

    taxonomy = cbct_taxonomy.by_key(target.class_key)
    lines = [
        f"«{taxonomy.label(locale)}» отмечено потому, что: "
        f"{target.rationale or taxonomy.why(locale)}",
        f"Признак нашёл этап «{target.produced_by}» конвейера "
        f"{case.volume.pipeline_version or 'анализа'}.",
        f"Итоговая уверенность — {round(target.confidence * 100)}%; она уже уменьшена "
        "с учётом качества исследования.",
    ]
    if case.volume.quality_score is not None:
        lines.append(
            "Качество снимка оценено как "
            f"{quality_label(case.volume.quality_score, locale).lower()}."
        )
    lines.append(
        "Это правило, а не «чёрный ящик»: те же измерения на другом снимке дадут "
        "тот же ответ, и с правилом можно не согласиться."
    )

    return Answer(
        intent=Intent.WHY_DETECTED,
        body="\n".join(lines),
        citations=(
            Citation("finding", taxonomy.label(locale), _volume_href(case)),
            Citation("pipeline", case.volume.pipeline_version or "—", None),
        ),
        suggestions=("Что проверить на приёме?", "Какие возможны варианты лечения?"),
    )


def _handle_treatments(question: str, case: CaseContext, locale: str) -> Answer:
    _ = question
    findings = case.volume_findings
    codes: list[str] = []
    for item in findings:
        codes.extend(cbct_taxonomy.by_key(item.class_key).procedures)
    for study in case.studies:
        for finding in study.findings:
            if finding.review is FindingReview.REJECTED:
                continue
            codes.extend(item.code for item in protocols.procedures_for(finding.class_key))

    ordered = list(dict.fromkeys(codes))
    if not ordered:
        return Answer(
            intent=Intent.TREATMENTS,
            body=(
                "Находки не подразумевают лечения: это анатомические ориентиры "
                "или уже выполненные работы. Плановое наблюдение по обычному графику."
            ),
            suggestions=("Что проверить на приёме?",),
        )

    lines = ["По находкам применимы следующие протоколы:"]
    for code in ordered:
        procedure = protocols.by_code(code)
        if procedure is None:
            continue
        visits = plural_ru(procedure.visits, "визит", "визита", "визитов")
        lines.append(
            f"• {procedure.label(locale)} — "
            f"{protocols.priority_label(procedure.priority, locale).lower()}, "
            f"{procedure.visits} {visits} по {procedure.minutes} мин."
        )
    lines.append(
        "Это таблица протоколов, а не назначение: сформируйте план лечения, "
        "чтобы увидеть варианты объёма, сроки и риски."
    )

    if case.plans:
        plan = case.plans[0]
        lines.append(f"У пациента уже есть план «{plan.title}» ({plan.status.value}).")

    cited = [
        procedure
        for code in ordered[:_SUMMARY_LIMIT]
        if (procedure := protocols.by_code(code)) is not None
    ]
    return Answer(
        intent=Intent.TREATMENTS,
        body="\n".join(lines),
        citations=tuple(
            Citation("procedure", procedure.label(locale), None) for procedure in cited
        ),
        suggestions=("Что проверить на приёме?", "Объясни простыми словами для пациента"),
    )


def _handle_next_checks(question: str, case: CaseContext, locale: str) -> Answer:
    _ = question
    findings = _ranked(case)
    if not findings:
        return Answer(
            intent=Intent.NEXT_CHECKS,
            body="Дополнительных проверок по этому исследованию не требуется.",
            suggestions=_SUGGESTIONS[:2],
        )

    lines = ["На приёме имеет смысл проверить:"]
    seen: set[str] = set()
    for item in findings:
        taxonomy = cbct_taxonomy.by_key(item.class_key)
        step = taxonomy.what_next(locale)
        if step in seen:
            continue
        seen.add(step)
        tooth = f" (зуб {item.tooth_number})" if item.tooth_number else ""
        lines.append(f"• {taxonomy.label(locale)}{tooth}: {step}")

    return Answer(
        intent=Intent.NEXT_CHECKS,
        body="\n".join(lines),
        citations=(Citation("volume", "Находки исследования", _volume_href(case)),),
        suggestions=("Какие возможны варианты лечения?", "Насколько надёжен этот снимок?"),
    )


def _handle_patient_friendly(question: str, case: CaseContext, locale: str) -> Answer:
    """Rewrite the findings for the person in the chair.

    A separate register, not a separate set of facts. The clinical wording is
    replaced with everyday language and the caveats are kept — dropping them
    would be the one thing a patient-facing explanation must not do.
    """
    _ = question
    findings = _ranked(case)
    name = _address(case.patient.full_name)

    if not findings:
        return Answer(
            intent=Intent.PATIENT_FRIENDLY,
            body=(
                f"{name}, на объёмном снимке мы не нашли изменений, которые "
                "требовали бы лечения прямо сейчас. Продолжаем наблюдать в "
                "обычном режиме."
            ),
            suggestions=("Что проверить на приёме?",),
        )

    lines = [
        f"{name}, вот что мы увидели на объёмном снимке, простыми словами:",
    ]
    # Deduplicated on the sentence, not the finding: the canal is detected in
    # several segments, and telling a patient about their mandibular nerve four
    # times reads as a malfunction.
    said: set[str] = set()
    for item in findings:
        taxonomy = cbct_taxonomy.by_key(item.class_key)
        plain = _PLAIN_LANGUAGE.get(item.class_key) or taxonomy.label(locale).lower()
        tooth = f" в области зуба {item.tooth_number}" if item.tooth_number else ""
        sentence = f"• {plain}{tooth}."
        if sentence in said:
            continue
        said.add(sentence)
        lines.append(sentence)
        if len(said) >= _SUMMARY_LIMIT:
            break

    lines.append(
        "Это предварительная находка компьютерного анализа. Окончательное "
        "решение врач принимает после осмотра — снимок сам по себе диагноз не ставит."
    )
    return Answer(
        intent=Intent.PATIENT_FRIENDLY,
        body="\n".join(lines),
        citations=(Citation("volume", "Заключение по исследованию", _volume_href(case)),),
        suggestions=("Какие возможны варианты лечения?",),
    )


def _handle_measurements(question: str, case: CaseContext, locale: str) -> Answer:
    _ = question, locale
    if case.volume is None:
        return _no_scan(Intent.MEASUREMENTS)

    measurements = list(case.volume.measurements)
    measured = [item for item in case.volume_findings if item.extent_mm]

    if not measurements and not measured:
        return Answer(
            intent=Intent.MEASUREMENTS,
            body=(
                "Измерений на этом исследовании ещё нет. Выберите инструмент "
                "«Расстояние» или «Угол» в просмотрщике и отметьте точки на срезе — "
                "значение считается по реальному шагу вокселя."
            ),
            suggestions=("Кратко опиши этот снимок",),
        )

    lines: list[str] = []
    if measurements:
        lines.append("Измерения, сделанные врачом:")
        lines.extend(
            f"• {item.label or item.kind.value}: {item.value} {item.unit}"
            f" ({item.created_by.full_name if item.created_by else '—'})"
            for item in measurements
        )
    if measured:
        lines.append("Размеры, измеренные автоматически:")
        lines.extend(
            f"• {cbct_taxonomy.by_key(item.class_key).label()}: {item.extent_mm} мм"
            for item in measured
        )

    geometry = case.volume
    lines.append(
        f"Шаг вокселя — {geometry.spacing_x:.2f} × {geometry.spacing_y:.2f} × "
        f"{geometry.spacing_z:.2f} мм; все значения пересчитаны по нему."
    )
    return Answer(
        intent=Intent.MEASUREMENTS,
        body="\n".join(lines),
        citations=(Citation("volume", "Измерения", _volume_href(case)),),
        suggestions=("Насколько надёжен этот снимок?",),
    )


def _handle_quality(question: str, case: CaseContext, locale: str) -> Answer:
    _ = question
    if case.volume is None:
        return _no_scan(Intent.QUALITY)

    score = case.volume.quality_score
    if score is None:
        return Answer(
            intent=Intent.QUALITY,
            body="Оценка качества для этого исследования не сохранена.",
            suggestions=("Кратко опиши этот снимок",),
        )

    lines = [
        f"Качество исследования — {quality_label(score, locale).lower()} "
        f"({round(score * 100)} из 100).",
        "Оценка складывается из уровня шума, признаков движения пациента, "
        "металлических артефактов и полноты охвата.",
    ]
    if score < _QUALITY_CAVEAT:
        lines.append(
            "Уверенность всех находок на этом снимке уже снижена пропорционально "
            "оценке. Для планирования имплантации имеет смысл рассмотреть "
            "повторное сканирование."
        )
    else:
        lines.append("Снимок пригоден для измерений и планирования.")

    lines.append(
        f"Сетка {case.volume.width}×{case.volume.height}×{case.volume.depth} вокселей, "
        f"срезов в источнике — {case.volume.source_slice_count}."
    )
    return Answer(
        intent=Intent.QUALITY,
        body="\n".join(lines),
        citations=(Citation("volume", "Контроль качества", _volume_href(case)),),
        suggestions=("Кратко опиши этот снимок", "Почему AI это нашёл?"),
    )


def _handle_similar(question: str, case: CaseContext, locale: str) -> Answer:
    """Similar cases need a database round trip, so the router defers it.

    Handled by the service rather than here: the handlers are pure functions of
    a loaded context by design, and reaching for a session inside one would
    make every handler need one.
    """
    _ = question, locale
    keys = sorted({item.class_key for item in case.volume_findings})
    if not keys:
        return Answer(
            intent=Intent.SIMILAR_CASES,
            body="Находок нет, поэтому сравнивать не с чем.",
            suggestions=_SUGGESTIONS[:2],
        )
    labels = ", ".join(cbct_taxonomy.by_key(key).label().lower() for key in keys[:4])
    return Answer(
        intent=Intent.SIMILAR_CASES,
        body=(
            f"Ищу в библиотеке случаев записи со схожими находками: {labels}. "
            "Совпадение считается по совпадающим классам находок — это грубая "
            "мера сходства, а не «похожая анатомия»."
        ),
        citations=(),
        suggestions=("Какие возможны варианты лечения?",),
    )


def _handle_capabilities(question: str, case: CaseContext, locale: str) -> Answer:
    _ = question, case, locale
    return Answer(
        intent=Intent.CAPABILITIES,
        body=(
            "Я отвечаю только по данным этого случая — по находкам, измерениям "
            "и оценке качества конкретного исследования. Я ничего не додумываю "
            "и не заменяю врача.\n\nСпросите, например:\n"
            + "\n".join(f"• {item}" for item in _SUGGESTIONS)
        ),
        suggestions=_SUGGESTIONS[:4],
    )


_HANDLERS: Final[dict[Intent, Callable[[str, CaseContext, str], Answer]]] = {
    Intent.SUMMARISE: _handle_summarise,
    Intent.EXPLAIN_FINDING: _handle_explain,
    Intent.WHY_DETECTED: _handle_why,
    Intent.TREATMENTS: _handle_treatments,
    Intent.NEXT_CHECKS: _handle_next_checks,
    Intent.SIMILAR_CASES: _handle_similar,
    Intent.PATIENT_FRIENDLY: _handle_patient_friendly,
    Intent.MEASUREMENTS: _handle_measurements,
    Intent.QUALITY: _handle_quality,
    Intent.CAPABILITIES: _handle_capabilities,
}

#: Everyday wording per finding class, for the patient-facing register. Written
#: out rather than derived, because "periapical lesion" does not simplify by
#: rule — it simplifies by someone deciding what a patient should hear.
_PLAIN_LANGUAGE: Final[dict[str, str]] = {
    "apical_lesion": "воспаление у верхушки корня зуба — обычно следствие инфекции внутри канала",
    "cyst": "полость в кости, заполненная жидкостью; её нужно показать хирургу",
    "odontogenic_infection": "воспалительный процесс, связанный с зубом",
    "abscess": "гнойное воспаление, которое требует срочного лечения",
    "periodontal_disease": "убыль кости вокруг зубов — то, что обычно называют пародонтитом",
    "bone_loss_3d": "снижение уровня кости вокруг зуба",
    "impacted_third_molar": "зуб мудрости, который не прорезался и остался в кости",
    "missing_tooth": "отсутствующий зуб; место можно восстановить",
    "implant": "установленный ранее имплант",
    "root_canal_filling": "ранее запломбированный корневой канал",
    "sinus_proximity": "близкое расположение гайморовой пазухи — важно для имплантации",
    "mandibular_canal": "нижнечелюстной нерв — ориентир, который хирург обязан учитывать",
    "caries_3d": "участок разрушения твёрдых тканей зуба",
    "root_fracture": "подозрение на трещину корня — требует подтверждения",
    "tmj_abnormality": "изменения в височно-нижнечелюстном суставе",
    "jaw_asymmetry": "небольшая разница между правой и левой половинами челюсти",
    "orthodontic_anomaly": "неровное положение зубов",
    "suspicious_mass": "образование, которое обязательно должен посмотреть профильный специалист",
    "metal_artifact": "помехи от металлических коронок или пломб на снимке",
    "motion_artifact": "смазывание снимка из-за движения во время съёмки",
}


def _address(full_name: str) -> str:
    """How to address the patient in a patient-facing explanation.

    Records here are written surname-first — "Иванов Иван Петрович" — so the
    obvious first token is the surname, and greeting someone by it reads as a
    summons rather than a conversation. Given name plus patronymic is the
    polite Russian form; a two-part or single-part name degrades to what is
    there.
    """
    parts = [part for part in full_name.split() if part]
    if len(parts) >= _NAME_WITH_PATRONYMIC:
        return f"{parts[1]} {parts[2]}"
    if len(parts) == _NAME_WITH_SURNAME:
        return parts[1]
    return parts[0] if parts else "Пациент"


def _region(finding: VolumeFinding) -> cbct_taxonomy.Region:
    try:
        return cbct_taxonomy.Region(finding.region)
    except ValueError:
        return cbct_taxonomy.Region.FULL_VOLUME


def _target_finding(
    question: str, case: CaseContext, locale: str
) -> VolumeFinding | None:
    """Which finding a question is about, by matching its label or tooth number.

    Falls back to the most severe finding only for questions that make sense
    without a target; :func:`_handle_explain` deliberately does not, because
    explaining an arbitrary finding to someone who asked about another one is
    worse than admitting the question was not understood.
    """
    normalised = question.lower()

    tooth_match = re.search(r"\b([1-4][1-8])\b", normalised)
    if tooth_match:
        wanted = int(tooth_match.group(1))
        if charting.is_valid(wanted):
            for item in _ranked(case):
                if item.tooth_number == wanted:
                    return item

    for item in _ranked(case):
        label = cbct_taxonomy.by_key(item.class_key).label(locale).lower()
        # Match on the first significant word so "объясни кисту" finds "Киста".
        head = label.split()[0][:6]
        if head and head in normalised:
            return item
    return None


def severity_of(finding: VolumeFinding) -> Severity:
    return cbct_taxonomy.by_key(finding.class_key).severity
