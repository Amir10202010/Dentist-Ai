"""Faceted search over patients, radiographs, CBCT volumes and the library.

Four kinds, one row shape, a fixed number of round trips. Each kind costs
exactly one query: the unpaginated total rides along as ``count(*) OVER ()``
rather than as a second ``SELECT count(*)``, and the severity badge comes from
a grouped sub-select joined in, not from loading every finding of every hit.
Facets add at most four more. Nothing here scales with the size of a page.

**Free text never goes through SQL ``lower()``.** It is matched against the
pre-folded ``search_text`` columns, for the reason written on
``Patient.search_text``: SQLite folds ASCII only, so "Иванов" matches "иванов"
on Postgres and not in development. Imaging has no folded column of its own and
is therefore searched *through its patient* and by public id. Matching the
original filename would mean ``lower()`` over a column that can hold Cyrillic —
a wider net on one backend and a narrower one on the other, which is precisely
the split the folded columns exist to close. A radiograph with no patient
attached is reachable by id, by date and by finding, but not by name; that is
the cost, and it is paid by rows that have no name to match.

**Facets ignore the two filters they drive.** Counting the fully filtered set
would leave every chip but the selected one reading zero, which is the one
state in which a filter UI tells the reader nothing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Final

from sqlalchemy import ColumnElement, SQLColumnExpression, case, distinct, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from dentist_ai.clinical.labels import field_of_view_label
from dentist_ai.db.models import CaseEntry, Finding, Patient, Study, Volume, VolumeFinding
from dentist_ai.ml.cbct_taxonomy import VOLUME_FINDING_CLASSES
from dentist_ai.ml.taxonomy import (
    DEFAULT_LOCALE,
    FINDING_CLASSES,
    SEVERITY_LABELS,
    Locale,
    Severity,
)
from dentist_ai.schemas.search import (
    FacetCount,
    SearchFacets,
    SearchGroup,
    SearchHit,
    SearchKind,
    SearchResults,
)
from dentist_ai.services.library import describe_finding, split_tokens, token_predicate

#: Severity as an orderable integer, so "worst finding on this record" is a
#: ``min()`` the database can compute instead of a list the service has to load.
_RANK_BY_KEY: Final[dict[str, int]] = {item.key: item.severity.rank for item in FINDING_CLASSES}
_VOLUME_RANK_BY_KEY: Final[dict[str, int]] = {
    item.key: item.severity.rank for item in VOLUME_FINDING_CLASSES
}
_SEVERITY_BY_RANK: Final[dict[int, Severity]] = {item.rank: item for item in Severity}
#: One past the last real rank. A class this build does not know must never win
#: the badge over one it does, and `min()` would hand it the badge at rank 0.
_UNRANKED: Final[int] = len(Severity)

#: Both key spaces in one table: a severity chip filters imaging of either kind,
#: and a key absent from a table simply never matches there.
_KEYS_BY_SEVERITY: Final[dict[Severity, tuple[str, ...]]] = {
    severity: tuple(
        key
        for ranks in (_RANK_BY_KEY, _VOLUME_RANK_BY_KEY)
        for key, rank in ranks.items()
        if rank == severity.rank
    )
    for severity in Severity
}

_NO_PATIENT: Final[dict[Locale, str]] = {
    "ru": "Без пациента",
    "en": "Unassigned",
    "kk": "Пациентсіз",
}
_PATIENT_CARD: Final[dict[Locale, str]] = {
    "ru": "Карта пациента",
    "en": "Patient record",
    "kk": "Пациент картасы",
}
_CASE_SUBTITLE: Final[dict[Locale, str]] = {
    "ru": "Разбор случая",
    "en": "Case write-up",
    "kk": "Жағдай талдауы",
}


def _pick(labels: dict[Locale, str], locale: Locale) -> str:
    return labels.get(locale) or labels[DEFAULT_LOCALE]


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Everything the query string can say, resolved to types the ORM speaks."""

    query: str | None = None
    kind: SearchKind | None = None
    severity: Severity | None = None
    finding: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    min_confidence: float | None = None
    doctor_id: int | None = None
    patient_id: int | None = None
    limit: int = 10
    offset: int = 0

    @property
    def needle(self) -> str | None:
        """The free text, folded and wrapped, or ``None`` when it is blank."""
        cleaned = (self.query or "").strip().lower()
        return f"%{cleaned}%" if cleaned else None

    @property
    def filters_findings(self) -> bool:
        """Whether anything here can only be answered by looking at findings."""
        return (
            self.finding is not None or self.severity is not None or self.min_confidence is not None
        )


class SearchService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self,
        filters: SearchFilters,
        *,
        organization_id: int,
        locale: Locale = DEFAULT_LOCALE,
    ) -> SearchResults:
        groups: list[SearchGroup] = []
        for kind, run in (
            (SearchKind.PATIENT, self._patients),
            (SearchKind.STUDY, self._studies),
            (SearchKind.VOLUME, self._volumes),
            (SearchKind.CASE, self._cases),
        ):
            if filters.kind is None or filters.kind is kind:
                groups.append(await run(filters, organization_id, locale))

        return SearchResults(
            query=(filters.query or "").strip(),
            total=sum(group.total for group in groups),
            groups=groups,
            facets=await self._facets(filters, organization_id, locale),
        )

    # -- per-kind queries -------------------------------------------------
    async def _patients(
        self, filters: SearchFilters, organization_id: int, locale: Locale
    ) -> SearchGroup:
        statement = select(Patient, func.count().over().label("total")).where(
            Patient.organization_id == organization_id,
            # Archived charts are hidden here exactly as they are in the
            # patient list; a search that resurrects them would make "archive"
            # mean nothing.
            Patient.archived_at.is_(None),
        )
        if filters.needle is not None:
            statement = statement.where(Patient.search_text.like(filters.needle))
        if filters.patient_id is not None:
            statement = statement.where(Patient.id == filters.patient_id)
        for condition in _window(Patient.created_at, filters):
            statement = statement.where(condition)

        through_imaging = self._through_imaging(filters, organization_id)
        if through_imaging is not None:
            statement = statement.where(through_imaging)

        rows = (
            await self._session.execute(
                statement.order_by(Patient.created_at.desc(), Patient.id.desc())
                .limit(filters.limit)
                .offset(filters.offset)
            )
        ).all()

        return SearchGroup(
            kind=SearchKind.PATIENT,
            total=int(rows[0][1]) if rows else 0,
            items=[
                SearchHit(
                    kind=SearchKind.PATIENT,
                    id=str(patient.id),
                    title=patient.full_name,
                    subtitle=_patient_subtitle(patient, locale),
                    href=f"/app/patients/{patient.id}",
                    at=patient.created_at,
                )
                for patient, _ in rows
            ],
        )

    async def _studies(
        self, filters: SearchFilters, organization_id: int, locale: Locale
    ) -> SearchGroup:
        worst = (
            select(
                Finding.study_id.label("record_id"),
                func.min(case(_RANK_BY_KEY, value=Finding.class_key, else_=_UNRANKED)).label(
                    "rank"
                ),
            )
            .join(Study, Finding.study_id == Study.id)
            .where(Study.organization_id == organization_id)
            .group_by(Finding.study_id)
            .subquery()
        )

        statement = (
            select(Study, Patient.full_name, worst.c.rank, func.count().over().label("total"))
            .outerjoin(Patient, Study.patient_id == Patient.id)
            .outerjoin(worst, worst.c.record_id == Study.id)
            .where(*self._study_conditions(filters, organization_id))
        )

        rows = (
            await self._session.execute(
                statement.order_by(Study.created_at.desc(), Study.id.desc())
                .limit(filters.limit)
                .offset(filters.offset)
            )
        ).all()

        return SearchGroup(
            kind=SearchKind.STUDY,
            total=int(rows[0][3]) if rows else 0,
            items=[
                SearchHit(
                    kind=SearchKind.STUDY,
                    id=study.public_id,
                    title=study.original_filename,
                    subtitle=patient_name or _pick(_NO_PATIENT, locale),
                    href=f"/app/studies/{study.public_id}",
                    at=study.created_at,
                    severity=_severity_of(rank),
                )
                for study, patient_name, rank, _ in rows
            ],
        )

    async def _volumes(
        self, filters: SearchFilters, organization_id: int, locale: Locale
    ) -> SearchGroup:
        worst = (
            select(
                VolumeFinding.volume_id.label("record_id"),
                func.min(
                    case(_VOLUME_RANK_BY_KEY, value=VolumeFinding.class_key, else_=_UNRANKED)
                ).label("rank"),
            )
            .join(Volume, VolumeFinding.volume_id == Volume.id)
            .where(Volume.organization_id == organization_id)
            .group_by(VolumeFinding.volume_id)
            .subquery()
        )

        statement = (
            select(Volume, Patient.full_name, worst.c.rank, func.count().over().label("total"))
            .outerjoin(Patient, Volume.patient_id == Patient.id)
            .outerjoin(worst, worst.c.record_id == Volume.id)
            .where(*self._volume_conditions(filters, organization_id))
        )

        rows = (
            await self._session.execute(
                statement.order_by(Volume.created_at.desc(), Volume.id.desc())
                .limit(filters.limit)
                .offset(filters.offset)
            )
        ).all()

        return SearchGroup(
            kind=SearchKind.VOLUME,
            total=int(rows[0][3]) if rows else 0,
            items=[
                SearchHit(
                    kind=SearchKind.VOLUME,
                    id=volume.public_id,
                    title=volume.original_filename,
                    subtitle=" · ".join(
                        (
                            patient_name or _pick(_NO_PATIENT, locale),
                            field_of_view_label(volume.field_of_view, locale),
                        )
                    ),
                    href=f"/app/volumes/{volume.public_id}",
                    at=volume.created_at,
                    severity=_severity_of(rank),
                )
                for volume, patient_name, rank, _ in rows
            ],
        )

    async def _cases(
        self, filters: SearchFilters, organization_id: int, locale: Locale
    ) -> SearchGroup:
        if filters.min_confidence is not None:
            # A library entry is written, not detected: it carries no
            # confidence and so cannot clear a confidence bar. Returning it
            # anyway would make the slider look broken.
            return SearchGroup(kind=SearchKind.CASE, total=0, items=[])

        statement = select(CaseEntry, func.count().over().label("total")).where(
            CaseEntry.organization_id == organization_id
        )
        if filters.needle is not None:
            statement = statement.where(CaseEntry.search_text.like(filters.needle))
        if filters.patient_id is not None:
            statement = statement.where(CaseEntry.patient_id == filters.patient_id)
        if filters.doctor_id is not None:
            statement = statement.where(CaseEntry.created_by_id == filters.doctor_id)
        if filters.finding is not None:
            statement = statement.where(token_predicate(CaseEntry.finding_keys, filters.finding))
        if filters.severity is not None:
            statement = statement.where(
                or_(
                    *(
                        token_predicate(CaseEntry.finding_keys, key)
                        for key in _KEYS_BY_SEVERITY[filters.severity]
                    )
                )
            )
        for condition in _window(CaseEntry.created_at, filters):
            statement = statement.where(condition)

        rows = (
            await self._session.execute(
                statement.order_by(CaseEntry.created_at.desc(), CaseEntry.id.desc())
                .limit(filters.limit)
                .offset(filters.offset)
            )
        ).all()

        return SearchGroup(
            kind=SearchKind.CASE,
            total=int(rows[0][1]) if rows else 0,
            items=[
                SearchHit(
                    kind=SearchKind.CASE,
                    id=entry.public_id,
                    title=entry.title,
                    subtitle=_case_subtitle(entry, locale),
                    href=f"/app/library/{entry.public_id}",
                    at=entry.created_at,
                    severity=_worst_of_keys(entry.finding_keys),
                )
                for entry, _ in rows
            ],
        )

    # -- facets -----------------------------------------------------------
    async def _facets(
        self, filters: SearchFilters, organization_id: int, locale: Locale
    ) -> SearchFacets:
        keys: Counter[str] = Counter()
        ranks: Counter[int] = Counter()

        # A patient- or case-only search has no imaging behind it, so the chips
        # would describe a set the reader is not looking at.
        if filters.kind in (None, SearchKind.STUDY):
            await self._tally_studies(filters, organization_id, keys, ranks)
        if filters.kind in (None, SearchKind.VOLUME):
            await self._tally_volumes(filters, organization_id, keys, ranks)

        described = [(describe_finding(key, locale), count) for key, count in keys.most_common()]
        return SearchFacets(
            findings=[
                FacetCount(
                    key=finding.key,
                    label=finding.label,
                    count=count,
                    severity=finding.severity,
                )
                for finding, count in described
            ],
            severities=[
                FacetCount(
                    key=severity.value,
                    label=_pick(SEVERITY_LABELS[severity], locale),
                    count=ranks[severity.rank],
                    severity=severity,
                )
                for severity in Severity
                if ranks[severity.rank]
            ],
        )

    async def _tally_studies(
        self,
        filters: SearchFilters,
        organization_id: int,
        keys: Counter[str],
        ranks: Counter[int],
    ) -> None:
        """Records per class, then records per severity, over the 2D findings.

        The severity tally cannot be folded out of the class tally. A study
        carrying two ``high`` findings is one result and not two, and only the
        database can say which records a severity covers without the service
        reading every finding row it has.
        """
        conditions = self._study_conditions(filters, organization_id, faceting=True)
        by_class = await self._session.execute(
            select(Finding.class_key, func.count(distinct(Finding.study_id)))
            .join(Study, Finding.study_id == Study.id)
            .outerjoin(Patient, Study.patient_id == Patient.id)
            .where(*conditions)
            .group_by(Finding.class_key)
        )
        keys.update(dict(by_class.tuples().all()))

        rank = case(_RANK_BY_KEY, value=Finding.class_key, else_=_UNRANKED)
        by_rank = await self._session.execute(
            select(rank, func.count(distinct(Finding.study_id)))
            .join(Study, Finding.study_id == Study.id)
            .outerjoin(Patient, Study.patient_id == Patient.id)
            .where(*conditions)
            .group_by(rank)
        )
        ranks.update(dict(by_rank.tuples().all()))

    async def _tally_volumes(
        self,
        filters: SearchFilters,
        organization_id: int,
        keys: Counter[str],
        ranks: Counter[int],
    ) -> None:
        conditions = self._volume_conditions(filters, organization_id, faceting=True)
        by_class = await self._session.execute(
            select(VolumeFinding.class_key, func.count(distinct(VolumeFinding.volume_id)))
            .join(Volume, VolumeFinding.volume_id == Volume.id)
            .outerjoin(Patient, Volume.patient_id == Patient.id)
            .where(*conditions)
            .group_by(VolumeFinding.class_key)
        )
        keys.update(dict(by_class.tuples().all()))

        rank = case(_VOLUME_RANK_BY_KEY, value=VolumeFinding.class_key, else_=_UNRANKED)
        by_rank = await self._session.execute(
            select(rank, func.count(distinct(VolumeFinding.volume_id)))
            .join(Volume, VolumeFinding.volume_id == Volume.id)
            .outerjoin(Patient, Volume.patient_id == Patient.id)
            .where(*conditions)
            .group_by(rank)
        )
        ranks.update(dict(by_rank.tuples().all()))

    # -- shared predicates -------------------------------------------------
    def _study_conditions(
        self, filters: SearchFilters, organization_id: int, *, faceting: bool = False
    ) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = [Study.organization_id == organization_id]
        if filters.needle is not None:
            conditions.append(
                or_(
                    Patient.search_text.like(filters.needle),
                    # An id is quoted back to the clinic in upper case, so it is
                    # matched as typed rather than folded into the pattern.
                    Study.public_id == (filters.query or "").strip().upper(),
                )
            )
        if filters.patient_id is not None:
            conditions.append(Study.patient_id == filters.patient_id)
        if filters.doctor_id is not None:
            conditions.append(Study.uploaded_by_id == filters.doctor_id)
        conditions.extend(_window(Study.created_at, filters))
        if not faceting:
            finding = _finding_exists(Finding, Finding.study_id, Study.id, filters)
            if finding is not None:
                conditions.append(finding)
        return conditions

    def _volume_conditions(
        self, filters: SearchFilters, organization_id: int, *, faceting: bool = False
    ) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = [Volume.organization_id == organization_id]
        if filters.needle is not None:
            conditions.append(
                or_(
                    Patient.search_text.like(filters.needle),
                    Volume.public_id == (filters.query or "").strip().upper(),
                )
            )
        if filters.patient_id is not None:
            conditions.append(Volume.patient_id == filters.patient_id)
        if filters.doctor_id is not None:
            conditions.append(Volume.uploaded_by_id == filters.doctor_id)
        conditions.extend(_window(Volume.created_at, filters))
        if not faceting:
            finding = _finding_exists(VolumeFinding, VolumeFinding.volume_id, Volume.id, filters)
            if finding is not None:
                conditions.append(finding)
        return conditions

    def _through_imaging(
        self, filters: SearchFilters, organization_id: int
    ) -> ColumnElement[bool] | None:
        """Reach a patient by what their imaging contains.

        A chart carries no findings and no uploader of its own. Dropping
        patients from the result whenever a finding or doctor filter is set is
        the other option and it is the wrong one: someone filtering to
        "periapical lesion" is looking for the people who have one.
        """
        if not filters.filters_findings and filters.doctor_id is None:
            return None

        studies = select(Study.id).where(
            Study.patient_id == Patient.id, Study.organization_id == organization_id
        )
        volumes = select(Volume.id).where(
            Volume.patient_id == Patient.id, Volume.organization_id == organization_id
        )
        if filters.doctor_id is not None:
            studies = studies.where(Study.uploaded_by_id == filters.doctor_id)
            volumes = volumes.where(Volume.uploaded_by_id == filters.doctor_id)

        study_finding = _finding_exists(Finding, Finding.study_id, Study.id, filters)
        if study_finding is not None:
            studies = studies.where(study_finding)
        volume_finding = _finding_exists(VolumeFinding, VolumeFinding.volume_id, Volume.id, filters)
        if volume_finding is not None:
            volumes = volumes.where(volume_finding)

        return or_(studies.exists(), volumes.exists())


def _finding_exists(
    model: type[Finding] | type[VolumeFinding],
    parent_column: SQLColumnExpression[int],
    record_id: SQLColumnExpression[int],
    filters: SearchFilters,
) -> ColumnElement[bool] | None:
    """``EXISTS`` rather than a join: a record matches once or not at all.

    Joining would multiply a study by its findings and turn both the window
    total and the page into nonsense.
    """
    if not filters.filters_findings:
        return None

    conditions: list[ColumnElement[bool]] = [parent_column == record_id]
    if filters.finding is not None:
        conditions.append(model.class_key == filters.finding)
    if filters.severity is not None:
        conditions.append(model.class_key.in_(_KEYS_BY_SEVERITY[filters.severity]))
    if filters.min_confidence is not None:
        conditions.append(model.confidence >= filters.min_confidence)
    return select(model.id).where(*conditions).exists()


def _window(
    column: SQLColumnExpression[datetime], filters: SearchFilters
) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = []
    if filters.date_from is not None:
        conditions.append(column >= datetime.combine(filters.date_from, time.min, tzinfo=UTC))
    if filters.date_to is not None:
        # Inclusive on the wire: "до 5 июня" has to contain the 5th, so the
        # bound is the start of the next day rather than its own midnight.
        conditions.append(
            column < datetime.combine(filters.date_to + timedelta(days=1), time.min, tzinfo=UTC)
        )
    return conditions


def _severity_of(rank: int | None) -> Severity | None:
    return None if rank is None else _SEVERITY_BY_RANK.get(rank)


def _worst_of_keys(raw: str) -> Severity | None:
    """The badge for a library entry, read off its frozen copy of the keys."""
    ranks = [describe_finding(key).severity.rank for key in split_tokens(raw)]
    return _SEVERITY_BY_RANK.get(min(ranks)) if ranks else None


def _patient_subtitle(patient: Patient, locale: Locale) -> str:
    parts = [part for part in (patient.medical_record_number, patient.phone) if part]
    return " · ".join(parts) if parts else _pick(_PATIENT_CARD, locale)


def _case_subtitle(entry: CaseEntry, locale: Locale) -> str:
    tags = split_tokens(entry.tags)
    if tags:
        return " · ".join(tags)
    return entry.summary.strip() or _pick(_CASE_SUBTITLE, locale)
