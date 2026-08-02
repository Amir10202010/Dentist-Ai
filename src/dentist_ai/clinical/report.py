"""Assembling a study's findings into a report a clinician can read.

Nothing here is generated text in the language-model sense. The summary is a
sentence built from counts, the tooth groups are the findings sorted, and the
recommendations are lookups in ``clinical.protocols``. If the report says
something, a row in the database says it too.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from dentist_ai.clinical import charting, protocols
from dentist_ai.clinical.labels import disclaimer
from dentist_ai.core.text import plural_ru
from dentist_ai.db.models import FindingReview
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Category, Locale, Severity
from dentist_ai.schemas.clinical import (
    FindingResponse,
    Recommendation,
    StudyReportResponse,
    ToothCell,
    ToothGroup,
)


def build_report(
    *,
    study_public_id: str,
    patient_name: str | None,
    findings: list[FindingResponse],
    generated_at: datetime,
    locale: Locale = DEFAULT_LOCALE,
) -> StudyReportResponse:
    kept = [item for item in findings if item.review is not FindingReview.REJECTED]

    by_tooth: dict[int, list[FindingResponse]] = defaultdict(list)
    regional: list[FindingResponse] = []
    for finding in kept:
        if finding.tooth_number is None:
            regional.append(finding)
        else:
            by_tooth[finding.tooth_number].append(finding)

    teeth = [
        ToothGroup(
            tooth_number=tooth,
            tooth_name=charting.tooth_name(tooth, locale),
            findings=sorted(items, key=lambda item: (item.severity.rank, -item.confidence)),
        )
        for tooth, items in sorted(by_tooth.items())
    ]

    attention = sum(1 for item in kept if item.category is Category.PATHOLOGY)
    reviewed = sum(1 for item in findings if item.review is not FindingReview.UNREVIEWED)

    return StudyReportResponse(
        study_public_id=study_public_id,
        generated_at=generated_at,
        patient_name=patient_name,
        summary=_summary(len(kept), attention, len(teeth), locale),
        finding_count=len(kept),
        attention_count=attention,
        reviewed_count=reviewed,
        affected_teeth=len(teeth),
        chart=_chart(by_tooth),
        teeth=teeth,
        regional=sorted(regional, key=lambda item: (item.severity.rank, -item.confidence)),
        recommendations=_recommendations(kept, locale),
        disclaimer=disclaimer(locale),
    )


def _chart(by_tooth: dict[int, list[FindingResponse]]) -> list[ToothCell]:
    cells: list[ToothCell] = []
    for tooth in charting.PERMANENT_TEETH:
        items = by_tooth.get(tooth, [])
        pathologies = [item for item in items if item.category is Category.PATHOLOGY]
        cells.append(
            ToothCell(
                tooth_number=tooth,
                severity=_worst(pathologies),
                finding_count=len(items),
                has_restoration=any(item.category is Category.RESTORATION for item in items),
                is_missing=any(item.class_key == "missing_teeth" for item in items),
            )
        )
    return cells


def _worst(findings: list[FindingResponse]) -> Severity | None:
    return min(
        (item.severity for item in findings),
        key=lambda severity: severity.rank,
        default=None,
    )


def _recommendations(findings: list[FindingResponse], locale: Locale) -> list[Recommendation]:
    seen: set[tuple[str, int | None]] = set()
    result: list[Recommendation] = []

    for finding in findings:
        for procedure in protocols.procedures_for(finding.class_key):
            key = (procedure.code, finding.tooth_number)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                Recommendation(
                    procedure_code=procedure.code,
                    label=procedure.label(locale),
                    category=procedure.category.value,
                    category_label=protocols.category_label(procedure.category, locale),
                    priority=procedure.priority.value,
                    priority_label=protocols.priority_label(procedure.priority, locale),
                    tooth_number=finding.tooth_number,
                    reason=_reason(finding, locale),
                    source_finding_id=finding.id,
                )
            )

    result.sort(
        key=lambda item: (
            protocols.Priority(item.priority).rank,
            item.tooth_number or 99,
            item.label,
        )
    )
    return result


def _reason(finding: FindingResponse, locale: Locale) -> str:
    confidence = f"{finding.confidence:.0%}"
    if finding.tooth_number is None:
        return f"{finding.label} · {confidence}"
    tooth_word = {"ru": "зуб", "en": "tooth", "kk": "тіс"}.get(locale, "зуб")
    return f"{finding.label} · {confidence} · {tooth_word} {finding.tooth_number}"


def _summary(findings: int, attention: int, teeth: int, locale: Locale) -> str:
    if locale == "en":
        return (
            f"{findings} finding{'s' if findings != 1 else ''}, "
            f"{attention} needing attention, {teeth} tooth/teeth affected."
        )
    if locale == "kk":
        return f"{findings} белгі, {attention} назар аударуды қажет етеді, {teeth} тіс қамтылды."

    findings_word = plural_ru(findings, "находка", "находки", "находок")
    attention_word = plural_ru(attention, "требует", "требуют", "требуют")
    teeth_word = plural_ru(teeth, "зуб", "зуба", "зубов")
    return (
        f"{findings} {findings_word}, из них {attention} {attention_word} внимания. "
        f"Затронуто {teeth} {teeth_word}."
    )
