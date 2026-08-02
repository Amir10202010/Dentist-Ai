"""The case library: teaching cases written from work that is finished.

The part worth reading is seeding. ``create`` may name a study or a volume, and
when it does the diagnosis text and the finding class keys are *copied* onto
the new row. Nothing on the read path dereferences ``study_public_id`` or
``volume_public_id`` — they are provenance, and the entry is complete without
them.

That is what :class:`~dentist_ai.db.models.CaseEntry` asks for, and the reason
is editorial rather than technical. A library entry is a statement about a
finished case. One that re-read the live record would rewrite yesterday's
teaching material the next time somebody corrected a tooth number, and a
teaching case that changes after it was taught is worse than no case at all.

Tags and finding keys are stored as one comma-delimited string each, matching
the columns the model declares. Matching is therefore whole-token rather than a
bare ``LIKE`` — see :func:`token_predicate`, which is also what the search
service uses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from sqlalchemy import ColumnElement, Select, SQLColumnExpression, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.core.errors import NotFoundError, PermissionDeniedError
from dentist_ai.core.ids import generate_public_id
from dentist_ai.db.models import CaseEntry, Study, User, Volume
from dentist_ai.ml.cbct_taxonomy import by_key as volume_by_key
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, UNKNOWN_CLASS, Locale, Severity, by_key
from dentist_ai.schemas.library import CaseCreateRequest, CaseWriteRequest
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext

#: ``case_entries.finding_keys`` is ``String(512)`` and ``tags`` is
#: ``String(255)``. Postgres rejects an overflow outright while SQLite stores it
#: silently, so the cap is applied on write to keep the two backends behaving
#: alike. The wire schema already bounds a *client* payload; this bounds the
#: seeded one, which no validator ever sees.
_MAX_FINDING_KEYS_CHARS: Final[int] = 512
_MAX_TAGS_CHARS: Final[int] = 255

_SEPARATOR: Final[str] = ","


@dataclass(frozen=True, slots=True)
class FindingLabel:
    key: str
    label: str
    severity: Severity


@dataclass(frozen=True, slots=True)
class TagCount:
    tag: str
    count: int


def describe_finding(key: str, locale: Locale = DEFAULT_LOCALE) -> FindingLabel:
    """Resolve a stored class key against both taxonomies.

    A case may cite a class from either examination, and three keys — ``cyst``,
    ``implant``, ``mandibular_canal`` — exist in both tables with the same
    meaning, so the 2D table answers first and the CBCT table answers for the
    rest. A key from neither resolves to the taxonomies' own unknown class
    rather than raising: a build older than the model that wrote the row must
    still render the case.
    """
    two_d = by_key(key)
    if two_d is not UNKNOWN_CLASS:
        return FindingLabel(key=key, label=two_d.label(locale), severity=two_d.severity)
    volumetric = volume_by_key(key)
    return FindingLabel(key=key, label=volumetric.label(locale), severity=volumetric.severity)


def token_predicate(column: SQLColumnExpression[str], token: str) -> ColumnElement[bool]:
    """Whole-token match against a comma-delimited column.

    A bare ``LIKE '%caries%'`` would also match ``caries_3d``, which is a real
    pair in this taxonomy rather than a hypothetical one. Wrapping both the
    column and the needle in the delimiter makes the comparison exact, using
    only ``||`` — the one concatenation operator Postgres and SQLite agree on.
    """
    delimited = literal(_SEPARATOR).concat(column).concat(_SEPARATOR)
    return delimited.like(f"%{_SEPARATOR}{token}{_SEPARATOR}%")


def split_tokens(raw: str) -> list[str]:
    return [item for item in (part.strip() for part in raw.split(_SEPARATOR)) if item]


def _join_capped(values: list[str], limit: int) -> str:
    """Join what fits, drop the rest. Order is priority order."""
    kept: list[str] = []
    length = 0
    for value in values:
        cost = len(value) + (len(_SEPARATOR) if kept else 0)
        if length + cost > limit:
            break
        kept.append(value)
        length += cost
    return _SEPARATOR.join(kept)


@dataclass(frozen=True, slots=True)
class _Seed:
    """What an existing record contributes to a new entry."""

    diagnosis: str
    finding_keys: list[str]
    patient_id: int | None
    study_public_id: str | None = None
    volume_public_id: str | None = None


class LibraryService:
    def __init__(self, session: AsyncSession, audit: AuditService) -> None:
        self._session = session
        self._audit = audit

    # -- reads ------------------------------------------------------------
    async def list_entries(
        self,
        *,
        organization_id: int,
        query: str | None = None,
        tag: str | None = None,
        finding: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[CaseEntry], int]:
        """Paginated listing. Returns the page plus the unpaginated total."""
        base = self._scoped(organization_id)
        if query:
            # `search_text` is folded in Python by `refresh_search_text`, for
            # the reason written on `Patient.search_text`: SQL `lower()` folds
            # ASCII only on SQLite, so a Cyrillic title would be searchable in
            # production and not in development.
            base = base.where(CaseEntry.search_text.like(f"%{query.strip().lower()}%"))
        if tag:
            base = base.where(token_predicate(CaseEntry.tags, tag.strip().lower()))
        if finding:
            base = base.where(token_predicate(CaseEntry.finding_keys, finding.strip().lower()))

        total = await self._session.scalar(select(func.count()).select_from(base.subquery()))
        rows = await self._session.scalars(
            base.options(selectinload(CaseEntry.created_by))
            .order_by(CaseEntry.created_at.desc(), CaseEntry.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows.unique().all()), int(total or 0)

    async def get(self, public_id: str, *, organization_id: int) -> CaseEntry:
        entry = await self._session.scalar(
            self._scoped(organization_id)
            .options(selectinload(CaseEntry.created_by))
            .where(CaseEntry.public_id == public_id)
        )
        if entry is None:
            # 404 rather than 403 for another clinic's id: the response must
            # not confirm that the entry exists.
            raise NotFoundError("Случай не найден.")
        return entry

    async def tag_counts(self, *, organization_id: int) -> list[TagCount]:
        """Distinct tags with counts, for the filter chips.

        Counted in Python over one round trip. Splitting a delimited column is
        not portable SQL, and the honest fix is a join table rather than a
        backend-specific ``string_to_array``; until the library outgrows a
        single page of tags, reading one narrow column is cheaper than the
        schema change.
        """
        rows = await self._session.scalars(
            select(CaseEntry.tags).where(
                CaseEntry.organization_id == organization_id, CaseEntry.tags != ""
            )
        )
        counter: Counter[str] = Counter()
        for raw in rows.all():
            counter.update(split_tokens(raw))
        # Frequency first, then alphabetically, so the chip row is stable
        # between two tags used the same number of times.
        return [
            TagCount(tag=tag, count=count)
            for tag, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]

    # -- mutations --------------------------------------------------------
    async def create(
        self,
        payload: CaseCreateRequest,
        *,
        actor: User,
        context: RequestContext,
    ) -> CaseEntry:
        seed = await self._seed(payload, organization_id=actor.organization_id, locale=actor.locale)

        entry = CaseEntry(
            public_id=generate_public_id(),
            organization_id=actor.organization_id,
            created_by_id=actor.id,
            title=payload.title,
            summary=payload.summary,
            # Explicit input wins over the copy: seeding is a starting point
            # for the author, not a value they have to overwrite twice.
            diagnosis=payload.diagnosis or seed.diagnosis,
            treatment=payload.treatment,
            outcome=payload.outcome,
            finding_keys=_join_capped(
                payload.finding_keys or seed.finding_keys, _MAX_FINDING_KEYS_CHARS
            ),
            tags=_join_capped(payload.tags, _MAX_TAGS_CHARS),
            patient_id=payload.patient_id if payload.patient_id is not None else seed.patient_id,
            study_public_id=seed.study_public_id,
            volume_public_id=seed.volume_public_id,
        )
        entry.refresh_search_text()
        self._session.add(entry)
        await self._session.flush()

        await self._audit.record(
            action=AuditAction.CASE_PUBLISHED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="case",
            resource_id=entry.public_id,
            context=context,
        )
        # Re-read with `created_by` loaded: the instance just flushed carries an
        # unloaded proxy, and touching it during serialisation would attempt
        # lazy IO outside the async context.
        return await self.get(entry.public_id, organization_id=actor.organization_id)

    async def update(
        self,
        public_id: str,
        payload: CaseWriteRequest,
        *,
        actor: User,
        context: RequestContext,
    ) -> CaseEntry:
        entry = await self.get(public_id, organization_id=actor.organization_id)
        entry.title = payload.title
        entry.summary = payload.summary
        entry.diagnosis = payload.diagnosis
        entry.treatment = payload.treatment
        entry.outcome = payload.outcome
        entry.finding_keys = _join_capped(payload.finding_keys, _MAX_FINDING_KEYS_CHARS)
        entry.tags = _join_capped(payload.tags, _MAX_TAGS_CHARS)
        entry.patient_id = payload.patient_id
        entry.refresh_search_text()

        await self._audit.record(
            action=AuditAction.CASE_UPDATED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="case",
            resource_id=entry.public_id,
            context=context,
        )
        return entry

    async def delete(self, public_id: str, *, actor: User, context: RequestContext) -> None:
        # The same set that may delete a study. An assistant may write up a
        # case; destroying the clinic's teaching material is a different act.
        if not actor.role.can_delete_patients:
            raise PermissionDeniedError("Недостаточно прав для удаления случая.")

        entry = await self.get(public_id, organization_id=actor.organization_id)
        await self._session.delete(entry)
        await self._audit.record(
            action=AuditAction.CASE_DELETED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type="case",
            resource_id=public_id,
            context=context,
        )

    # -- seeding ----------------------------------------------------------
    async def _seed(
        self,
        payload: CaseCreateRequest,
        *,
        organization_id: int,
        locale: Locale,
    ) -> _Seed:
        if payload.from_study_public_id:
            return await self._seed_from_study(
                payload.from_study_public_id, organization_id=organization_id, locale=locale
            )
        if payload.from_volume_public_id:
            return await self._seed_from_volume(
                payload.from_volume_public_id, organization_id=organization_id, locale=locale
            )
        return _Seed(diagnosis="", finding_keys=[], patient_id=None)

    async def _seed_from_study(
        self, public_id: str, *, organization_id: int, locale: Locale
    ) -> _Seed:
        study = await self._session.scalar(
            select(Study)
            .options(selectinload(Study.findings))
            .where(Study.public_id == public_id, Study.organization_id == organization_id)
        )
        if study is None:
            raise NotFoundError("Снимок не найден.")

        keys = _ranked_keys((item.class_key, item.confidence) for item in study.findings)
        return _Seed(
            diagnosis=_compose_diagnosis(keys, study.notes, locale),
            finding_keys=keys,
            patient_id=study.patient_id,
            study_public_id=study.public_id,
        )

    async def _seed_from_volume(
        self, public_id: str, *, organization_id: int, locale: Locale
    ) -> _Seed:
        volume = await self._session.scalar(
            select(Volume)
            .options(selectinload(Volume.findings))
            .where(Volume.public_id == public_id, Volume.organization_id == organization_id)
        )
        if volume is None:
            raise NotFoundError("Исследование не найдено.")

        keys = _ranked_keys((item.class_key, item.confidence) for item in volume.findings)
        return _Seed(
            diagnosis=_compose_diagnosis(keys, volume.notes, locale),
            finding_keys=keys,
            patient_id=volume.patient_id,
            volume_public_id=volume.public_id,
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _scoped(organization_id: int) -> Select[tuple[CaseEntry]]:
        return select(CaseEntry).where(CaseEntry.organization_id == organization_id)


def _ranked_keys(findings: Iterable[tuple[str, float]]) -> list[str]:
    """Distinct class keys, worst first.

    Severity before confidence, and the cap in :func:`_join_capped` truncates
    from the tail, so a case seeded from a study with more classes than the
    column holds keeps the ones a clinician would have listed first.
    """
    best: dict[str, float] = {}
    for key, confidence in findings:
        best[key] = max(best.get(key, 0.0), confidence)
    return sorted(
        best,
        key=lambda key: (describe_finding(key).severity.rank, -best[key], key),
    )


def _compose_diagnosis(keys: list[str], notes: str | None, locale: Locale) -> str:
    """The copied diagnosis: what was found, then whatever was written on it.

    Assembled from the taxonomy rather than from generated prose, on the same
    reasoning as the study report — nothing in the library should read as a
    sentence no clinician wrote.
    """
    listed = ", ".join(describe_finding(key, locale).label for key in keys)
    parts = [part for part in (listed, (notes or "").strip()) if part]
    return "\n".join(parts)
