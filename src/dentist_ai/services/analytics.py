"""Dashboard metrics, computed from the tenant's own rows.

Everything here runs in a small fixed number of queries regardless of how much
data the clinic has. :func:`_insights` reads the aggregates and applies a fixed
set of rules to them; none of it is model output.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import ColumnElement, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from dentist_ai.db.models import (
    AuditEvent,
    Finding,
    FindingReview,
    Patient,
    Study,
    StudyStatus,
    User,
)
from dentist_ai.ml.taxonomy import (
    CATEGORY_LABELS,
    DEFAULT_LOCALE,
    FINDING_CLASSES,
    SEVERITY_LABELS,
    Category,
    Severity,
    by_key,
)
from dentist_ai.schemas.clinical import (
    ActivityItem,
    DashboardResponse,
    Insight,
    LabelledCount,
    MetricDelta,
    PipelineStatus,
    ReviewQueueItem,
    ReviewStats,
    TimeSeriesPoint,
    Tone,
)
from dentist_ai.services.audit import AuditAction

#: Wide enough that the client can offer 7 / 30 / 90-day views of the trend by
#: slicing what it already has, instead of a round-trip per range change.
_TREND_DAYS: Final[int] = 90
_TOP_FINDINGS: Final[int] = 6
_QUEUE_LIMIT: Final[int] = 5
_ACTIVITY_LIMIT: Final[int] = 8
_MAX_INSIGHTS: Final[int] = 4

#: Findings that require a clinical decision, and how urgent each one is.
#: Built from the taxonomy so adding a class cannot silently drop it out of the
#: review queue.
_PATHOLOGY_RANK: Final[dict[str, int]] = {
    item.key: item.severity.rank for item in FINDING_CLASSES if item.needs_attention
}
_PATHOLOGY_KEYS: Final[tuple[str, ...]] = tuple(_PATHOLOGY_RANK)

#: Application routes an insight can point at. Named here rather than inlined
#: so the compile-time reader can see every screen this module links to.
_STUDIES_PAGE: Final[str] = "/app/studies"
_PATIENTS_PAGE: Final[str] = "/app/patients"

#: Adjudicated findings below which the agreement rate is noise, not signal.
_AGREEMENT_MIN_SAMPLE: Final[int] = 20
#: Above this, the model and the clinic are broadly agreeing — worth saying
#: warmly rather than neutrally.
_GOOD_AGREEMENT_PERCENT: Final[int] = 75
#: Days a finding may wait before the reminder escalates from a nudge.
_STALE_REVIEW_DAYS: Final[int] = 3
#: Week-over-week movement small enough to be the ordinary jitter of a clinic
#: with a handful of chairs. Reporting it would train the reader to ignore the
#: line entirely. Pathology counts move less than volume, so they get the
#: lower bar.
_MIN_REPORTABLE_CHANGE: Final[float] = 0.15
_MIN_REPORTABLE_VOLUME_CHANGE: Final[float] = 0.20


@dataclass(frozen=True, slots=True)
class _ActivityKind:
    """How one audit action reads in the feed."""

    summary: str
    icon: str
    tone: Tone


# Reads are not here. `study.viewed` and `study.image_accessed`
# belong in the audit trail — a compliance record has to answer "who looked at
# this" — but a feed where opening a radiograph outnumbers every real change
# 20:1 tells the clinic nothing. The trail stays complete; the feed is curated.
_ACTIVITY_KINDS: Final[dict[str, _ActivityKind]] = {
    AuditAction.STUDY_UPLOADED.value: _ActivityKind("Загружен снимок", "upload", Tone.INFO),
    AuditAction.FINDING_REVIEWED.value: _ActivityKind(
        "Находка проверена", "check-circle", Tone.POSITIVE
    ),
    AuditAction.STUDY_UPDATED.value: _ActivityKind("Снимок обновлён", "pencil", Tone.INFO),
    AuditAction.STUDY_DELETED.value: _ActivityKind("Снимок удалён", "trash", Tone.WARNING),
    AuditAction.STUDY_EXPORTED.value: _ActivityKind("Снимок выгружен", "download", Tone.INFO),
    AuditAction.PATIENT_CREATED.value: _ActivityKind("Добавлен пациент", "user", Tone.INFO),
    AuditAction.PATIENT_UPDATED.value: _ActivityKind("Карта обновлена", "pencil", Tone.INFO),
    AuditAction.PATIENT_ARCHIVED.value: _ActivityKind(
        "Карта архивирована", "archive", Tone.WARNING
    ),
    AuditAction.USER_REGISTERED.value: _ActivityKind("Новый сотрудник", "user", Tone.POSITIVE),
    AuditAction.LOGIN_SUCCEEDED.value: _ActivityKind("Вход в систему", "lock", Tone.INFO),
    # Surfaced, not buried: a run of these is the earliest signal a clinic gets
    # that someone is guessing at an account.
    AuditAction.LOGIN_FAILED.value: _ActivityKind("Неудачный вход", "shield", Tone.WARNING),
    AuditAction.PASSWORD_CHANGED.value: _ActivityKind("Пароль изменён", "lock", Tone.INFO),
    AuditAction.PROFILE_UPDATED.value: _ActivityKind("Профиль обновлён", "user", Tone.INFO),
}

_TONE_ORDER: Final[dict[Tone, int]] = {
    Tone.CRITICAL: 0,
    Tone.WARNING: 1,
    Tone.POSITIVE: 2,
    Tone.INFO: 3,
}


#: The two irregular groups in Russian numeric agreement. 11–14 take the
#: many-form despite ending in 1–4, which is why the check cannot be on the
#: last digit alone.
_TEEN_REMAINDERS: Final[frozenset[int]] = frozenset(range(11, 15))
_FEW_REMAINDERS: Final[frozenset[int]] = frozenset({2, 3, 4})
_ONE_REMAINDER: Final[frozenset[int]] = frozenset({1})


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Russian numeric agreement.

    "1 находка", "3 находки", "12 находок" — three forms, chosen by the last
    digits. Formatting a clinical readout as "12 находка" reads as a machine
    translation and undermines everything else on the page.
    """
    if count % 100 in _TEEN_REMAINDERS:
        return many
    if count % 10 in _ONE_REMAINDER:
        return one
    if count % 10 in _FEW_REMAINDERS:
        return few
    return many


def _local_day_start(now: datetime, timezone: str) -> datetime:
    """Midnight *for the clinic*, as an aware UTC-comparable instant.

    "Сегодня" is read by a human against their own wall clock. In Almaty
    (UTC+5) a UTC day boundary would report the first five hours of every
    morning as belonging to yesterday.
    """
    zone: tzinfo
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone = UTC
    local = now.astimezone(zone)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class _Counts:
    patients: int
    patients_week: int
    patients_prev_week: int
    studies: int
    studies_week: int
    studies_prev_week: int
    pending: int
    processing: int
    failed_recent: int
    completed_today: int
    avg_inference_ms: int | None


@dataclass(frozen=True, slots=True)
class _FindingAggregates:
    attention_week: int
    attention_prev_week: int
    queue_studies: int
    pending_findings: int
    oldest_pending_at: datetime | None
    total: int
    confirmed: int
    rejected: int
    average_confidence: float | None


class AnalyticsService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def dashboard(
        self,
        *,
        organization_id: int,
        locale: str = DEFAULT_LOCALE,
        timezone: str = "UTC",
    ) -> DashboardResponse:
        now = datetime.now(UTC)
        week_ago = now - timedelta(days=7)
        fortnight_ago = now - timedelta(days=14)
        trend_start = (now - timedelta(days=_TREND_DAYS)).date()

        # Seven round-trips, and seven regardless of how large the clinic is.
        # They are sequential rather than gathered because they share one
        # AsyncSession, which is not safe to use concurrently.
        counts = await self._counts(
            organization_id,
            week_ago=week_ago,
            fortnight_ago=fortnight_ago,
            day_start=_local_day_start(now, timezone),
        )
        series = await self._studies_over_time(organization_id, trend_start)
        top, categories, attention = await self._finding_breakdown(organization_id, locale)
        findings = await self._finding_aggregates(
            organization_id, week_ago=week_ago, fortnight_ago=fortnight_ago
        )
        queue = await self._review_queue(organization_id, locale)
        activity = await self._activity(organization_id)

        adjudicated = findings.confirmed + findings.rejected
        review_stats = ReviewStats(
            confirmed=findings.confirmed,
            rejected=findings.rejected,
            unreviewed=max(0, findings.total - adjudicated),
            agreement_rate=(
                round(findings.confirmed / adjudicated, 4) if adjudicated > 0 else None
            ),
            average_confidence=findings.average_confidence,
        )
        pipeline = PipelineStatus(
            pending=counts.pending,
            processing=counts.processing,
            completed_today=counts.completed_today,
            failed_recent=counts.failed_recent,
        )
        studies_delta = MetricDelta(current=counts.studies_week, previous=counts.studies_prev_week)
        patients_delta = MetricDelta(
            current=counts.patients_week, previous=counts.patients_prev_week
        )
        attention_delta = MetricDelta(
            current=findings.attention_week, previous=findings.attention_prev_week
        )

        return DashboardResponse(
            generated_at=now,
            total_patients=counts.patients,
            new_patients_this_week=counts.patients_week,
            total_studies=counts.studies,
            studies_this_week=counts.studies_week,
            findings_needing_attention=attention,
            average_inference_ms=counts.avg_inference_ms,
            reviewed_share=(round(adjudicated / findings.total, 4) if findings.total > 0 else 0.0),
            studies_over_time=series,
            top_findings=top,
            category_breakdown=categories,
            studies_delta=studies_delta,
            patients_delta=patients_delta,
            attention_delta=attention_delta,
            review_queue=queue,
            review_queue_total=findings.queue_studies,
            pending_findings=findings.pending_findings,
            oldest_pending_at=findings.oldest_pending_at,
            activity=activity,
            insights=_insights(
                _InsightContext(
                    now=now,
                    counts=counts,
                    findings=findings,
                    top=top,
                    pipeline=pipeline,
                    review_stats=review_stats,
                    studies_delta=studies_delta,
                    attention_delta=attention_delta,
                )
            ),
            pipeline=pipeline,
            review_stats=review_stats,
        )

    # ----------------------------------------------------------------------
    # Queries
    # ----------------------------------------------------------------------
    async def _counts(
        self,
        organization_id: int,
        *,
        week_ago: datetime,
        fortnight_ago: datetime,
        day_start: datetime,
    ) -> _Counts:
        # A single row of scalar sub-selects: eleven aggregates in one
        # round-trip instead of eleven statements. Each closure re-applies the
        # tenant predicate, so a new metric cannot be added without it.
        def patients(*extra: ColumnElement[bool]) -> ColumnElement[int]:
            return (
                select(func.count())
                .select_from(Patient)
                .where(
                    Patient.organization_id == organization_id,
                    Patient.archived_at.is_(None),
                    *extra,
                )
                .scalar_subquery()
            )

        def studies(*extra: ColumnElement[bool]) -> ColumnElement[int]:
            return (
                select(func.count())
                .select_from(Study)
                .where(Study.organization_id == organization_id, *extra)
                .scalar_subquery()
            )

        avg_inference = (
            select(func.avg(Study.inference_ms))
            .where(
                Study.organization_id == organization_id,
                Study.status == StudyStatus.COMPLETED,
                Study.inference_ms.is_not(None),
            )
            .scalar_subquery()
        )

        row = (
            await self._session.execute(
                select(
                    patients(),
                    patients(Patient.created_at >= week_ago),
                    patients(Patient.created_at >= fortnight_ago, Patient.created_at < week_ago),
                    studies(),
                    studies(Study.created_at >= week_ago),
                    studies(Study.created_at >= fortnight_ago, Study.created_at < week_ago),
                    studies(Study.status == StudyStatus.PENDING),
                    studies(Study.status == StudyStatus.PROCESSING),
                    studies(Study.status == StudyStatus.FAILED, Study.created_at >= week_ago),
                    studies(Study.status == StudyStatus.COMPLETED, Study.created_at >= day_start),
                    avg_inference,
                )
            )
        ).one()

        return _Counts(
            patients=int(row[0] or 0),
            patients_week=int(row[1] or 0),
            patients_prev_week=int(row[2] or 0),
            studies=int(row[3] or 0),
            studies_week=int(row[4] or 0),
            studies_prev_week=int(row[5] or 0),
            pending=int(row[6] or 0),
            processing=int(row[7] or 0),
            failed_recent=int(row[8] or 0),
            completed_today=int(row[9] or 0),
            avg_inference_ms=int(row[10]) if row[10] is not None else None,
        )

    async def _studies_over_time(self, organization_id: int, start: date) -> list[TimeSeriesPoint]:
        rows = await self._session.execute(
            select(
                func.date(Study.created_at).label("day"),
                func.count(Study.id).label("count"),
            )
            .where(
                Study.organization_id == organization_id,
                func.date(Study.created_at) >= start,
            )
            .group_by(func.date(Study.created_at))
        )
        # Postgres returns `date`, SQLite returns `str` for `func.date`.
        counts: dict[date, int] = {}
        for day, count in rows.all():
            parsed = day if isinstance(day, date) else date.fromisoformat(str(day))
            counts[parsed] = int(count)

        # Densify: a chart with gaps for zero-activity days misleads the reader.
        return [
            TimeSeriesPoint(date=day, value=float(counts.get(day, 0)))
            for day in (start + timedelta(days=offset) for offset in range(_TREND_DAYS + 1))
        ]

    async def _finding_breakdown(
        self, organization_id: int, locale: str
    ) -> tuple[list[LabelledCount], list[LabelledCount], int]:
        rows = await self._session.execute(
            select(Finding.class_key, func.count(Finding.id))
            .join(Study, Finding.study_id == Study.id)
            .where(
                Study.organization_id == organization_id,
                Finding.review != FindingReview.REJECTED,
            )
            .group_by(Finding.class_key)
        )

        per_class = [(by_key(key), int(count)) for key, count in rows.all()]

        top = [
            LabelledCount(
                key=taxonomy.key,
                label=taxonomy.label(locale),
                count=count,
                severity=taxonomy.severity,
            )
            for taxonomy, count in sorted(
                per_class,
                key=lambda pair: (pair[0].severity.rank, -pair[1]),
            )[:_TOP_FINDINGS]
        ]

        category_totals: Counter[Category] = Counter()
        attention = 0
        for taxonomy, count in per_class:
            category_totals[taxonomy.category] += count
            if taxonomy.needs_attention:
                attention += count

        categories = [
            LabelledCount(
                key=category.value,
                label=CATEGORY_LABELS[category].get(locale, CATEGORY_LABELS[category]["ru"]),
                count=category_totals.get(category, 0),
            )
            for category in Category
        ]
        return top, categories, attention

    async def _finding_aggregates(
        self, organization_id: int, *, week_ago: datetime, fortnight_ago: datetime
    ) -> _FindingAggregates:
        """Everything derived from findings joined to their study, in one row."""

        def findings(*extra: ColumnElement[bool]) -> tuple[ColumnElement[bool], ...]:
            return (Study.organization_id == organization_id, *extra)

        def count(*extra: ColumnElement[bool]) -> ColumnElement[int]:
            return (
                select(func.count(Finding.id))
                .select_from(Finding)
                .join(Study, Finding.study_id == Study.id)
                .where(*findings(*extra))
                .scalar_subquery()
            )

        unreviewed_pathology = (
            Finding.review == FindingReview.UNREVIEWED,
            Finding.class_key.in_(_PATHOLOGY_KEYS),
        )

        queue_studies = (
            select(func.count(func.distinct(Finding.study_id)))
            .select_from(Finding)
            .join(Study, Finding.study_id == Study.id)
            .where(*findings(*unreviewed_pathology))
            .scalar_subquery()
        )
        oldest_pending = (
            select(func.min(Study.created_at))
            .select_from(Finding)
            .join(Study, Finding.study_id == Study.id)
            .where(*findings(*unreviewed_pathology))
            .scalar_subquery()
        )
        # Rejected findings are the clinician saying the detection was not
        # there; averaging their confidence in would measure the model against
        # pixels nobody believes.
        average_confidence = (
            select(func.avg(Finding.confidence))
            .select_from(Finding)
            .join(Study, Finding.study_id == Study.id)
            .where(*findings(Finding.review != FindingReview.REJECTED))
            .scalar_subquery()
        )

        attention = (
            Finding.class_key.in_(_PATHOLOGY_KEYS),
            Finding.review != FindingReview.REJECTED,
        )

        row = (
            await self._session.execute(
                select(
                    count(*attention, Study.created_at >= week_ago),
                    count(
                        *attention,
                        Study.created_at >= fortnight_ago,
                        Study.created_at < week_ago,
                    ),
                    queue_studies,
                    count(*unreviewed_pathology),
                    oldest_pending,
                    count(),
                    count(Finding.review == FindingReview.CONFIRMED),
                    count(Finding.review == FindingReview.REJECTED),
                    average_confidence,
                )
            )
        ).one()

        oldest = row[4]
        return _FindingAggregates(
            attention_week=int(row[0] or 0),
            attention_prev_week=int(row[1] or 0),
            queue_studies=int(row[2] or 0),
            pending_findings=int(row[3] or 0),
            # SQLite hands back a naive datetime; the rest of the app compares
            # against aware ones.
            oldest_pending_at=(
                oldest.replace(tzinfo=UTC)
                if isinstance(oldest, datetime) and oldest.tzinfo is None
                else oldest
            ),
            total=int(row[5] or 0),
            confirmed=int(row[6] or 0),
            rejected=int(row[7] or 0),
            average_confidence=round(float(row[8]), 4) if row[8] is not None else None,
        )

    async def _review_queue(self, organization_id: int, locale: str) -> list[ReviewQueueItem]:
        """Studies waiting on a clinician, most urgent first.

        Ordered by severity and *then* by age: a caries lesion found this
        morning outranks an impacted tooth from last week, but among equals the
        one that has waited longest goes first.
        """
        rank = case(_PATHOLOGY_RANK, value=Finding.class_key, else_=len(Severity))
        rows = (
            await self._session.execute(
                select(
                    Study.id,
                    Study.public_id,
                    Study.original_filename,
                    Study.created_at,
                    Patient.full_name,
                    func.count(Finding.id),
                )
                .select_from(Finding)
                .join(Study, Finding.study_id == Study.id)
                .outerjoin(Patient, Patient.id == Study.patient_id)
                .where(
                    Study.organization_id == organization_id,
                    Finding.review == FindingReview.UNREVIEWED,
                    Finding.class_key.in_(_PATHOLOGY_KEYS),
                )
                .group_by(
                    Study.id,
                    Study.public_id,
                    Study.original_filename,
                    Study.created_at,
                    Patient.full_name,
                )
                .order_by(func.min(rank).asc(), Study.created_at.asc())
                .limit(_QUEUE_LIMIT)
            )
        ).all()

        if not rows:
            return []

        # Which class leads each study needs the rows themselves, not an
        # aggregate. Bounded by `_QUEUE_LIMIT` studies, so it stays one small
        # query however large the clinic is.
        study_ids = [row[0] for row in rows]
        detail = (
            await self._session.execute(
                select(Finding.study_id, Finding.class_key, Finding.confidence).where(
                    Finding.study_id.in_(study_ids),
                    Finding.review == FindingReview.UNREVIEWED,
                    Finding.class_key.in_(_PATHOLOGY_KEYS),
                )
            )
        ).all()

        # Sort key, not a value: severity first, then confidence descending
        # (hence the negation), then the class key to break ties reproducibly.
        leaders: dict[int, tuple[int, float, str]] = {}
        for study_id, class_key, confidence in detail:
            candidate = (_PATHOLOGY_RANK[class_key], -float(confidence), class_key)
            if study_id not in leaders or candidate < leaders[study_id]:
                leaders[study_id] = candidate

        items: list[ReviewQueueItem] = []
        for study_id, public_id, filename, created_at, patient_name, pending in rows:
            leader = leaders.get(study_id)
            if leader is None:
                continue
            _, negated_confidence, class_key = leader
            taxonomy = by_key(class_key)
            items.append(
                ReviewQueueItem(
                    public_id=public_id,
                    patient_name=patient_name,
                    original_filename=filename,
                    created_at=created_at,
                    pending_count=int(pending),
                    top_severity=taxonomy.severity,
                    top_severity_label=SEVERITY_LABELS[taxonomy.severity].get(
                        locale, SEVERITY_LABELS[taxonomy.severity][DEFAULT_LOCALE]
                    ),
                    top_finding_label=taxonomy.label(locale),
                    top_confidence=round(-negated_confidence, 4),
                )
            )
        return items

    async def _activity(self, organization_id: int) -> list[ActivityItem]:
        rows = (
            await self._session.execute(
                select(AuditEvent, User.full_name)
                # `actor_id` is not a foreign key — audit rows
                # outlive the users they name — so the join is explicit and
                # outer.
                .outerjoin(User, User.id == AuditEvent.actor_id)
                .where(
                    AuditEvent.organization_id == organization_id,
                    AuditEvent.action.in_(tuple(_ACTIVITY_KINDS)),
                )
                .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
                .limit(_ACTIVITY_LIMIT)
            )
        ).all()

        items: list[ActivityItem] = []
        for event, actor_name in rows:
            kind = _ACTIVITY_KINDS.get(event.action)
            if kind is None:
                continue
            items.append(
                ActivityItem(
                    id=event.id,
                    action=event.action,
                    actor_name=actor_name,
                    summary=kind.summary,
                    icon=kind.icon,
                    tone=kind.tone,
                    resource_type=event.resource_type,
                    resource_id=event.resource_id,
                    created_at=event.created_at,
                )
            )
        return items


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _InsightContext:
    """Everything the rules below are allowed to look at."""

    now: datetime
    counts: _Counts
    findings: _FindingAggregates
    top: list[LabelledCount]
    pipeline: PipelineStatus
    review_stats: ReviewStats
    studies_delta: MetricDelta
    attention_delta: MetricDelta


def _failed_studies(ctx: _InsightContext) -> Insight | None:
    count = ctx.pipeline.failed_recent
    if count == 0:
        return None
    return Insight(
        key="failed-studies",
        tone=Tone.CRITICAL,
        icon="alert-circle",
        title=f"{count} {_plural(count, 'снимок', 'снимка', 'снимков')} не обработано",
        body=(
            "Анализ не прошёл за последнюю неделю. Обычная причина — повреждённый "
            "файл или неподдерживаемый формат: снимок можно загрузить заново."
        ),
        metric=str(count),
        action_label="Проверить снимки",
        action_href=_STUDIES_PAGE,
    )


def _pending_review(ctx: _InsightContext) -> Insight | None:
    """The queue, stated as a total — which the queue list itself cannot show.

    The widget shows the five most urgent studies. This says how much is behind
    them and how long the oldest has been waiting, which is the part that
    decides whether the clinic is keeping up.
    """
    count = ctx.findings.pending_findings
    if count == 0:
        # Nothing to review is only worth celebrating once there is something
        # to have reviewed.
        if ctx.findings.total == 0:
            return None
        return Insight(
            key="queue-clear",
            tone=Tone.POSITIVE,
            icon="check-circle",
            title="Очередь проверки пуста",
            body=(
                "Каждая находка категории «Патологии» получила решение врача — "
                "статистика клиники построена на подтверждённых данных."
            ),
            metric="0",
        )

    waiting_days = (
        max(0, (ctx.now - ctx.findings.oldest_pending_at).days)
        if ctx.findings.oldest_pending_at is not None
        else 0
    )
    studies = ctx.findings.queue_studies
    waited = (
        f"Самый ранний ждёт {waiting_days} {_plural(waiting_days, 'день', 'дня', 'дней')}."
        if waiting_days > 0
        else "Все — за сегодня."
    )
    return Insight(
        key="pending-review",
        tone=Tone.WARNING if waiting_days >= _STALE_REVIEW_DAYS else Tone.INFO,
        icon="stethoscope",
        title=(
            f"{count} {_plural(count, 'находка ждёт', 'находки ждут', 'находок ждут')} "
            "решения врача"
        ),
        body=f"В {studies} {_plural(studies, 'снимке', 'снимках', 'снимках')}. {waited}",
        metric=str(count),
        action_label="Открыть снимки",
        action_href=_STUDIES_PAGE,
    )


def _in_flight(ctx: _InsightContext) -> Insight | None:
    active = ctx.pipeline.processing + ctx.pipeline.pending
    if active == 0:
        return None
    return Insight(
        key="in-flight",
        tone=Tone.INFO,
        icon="activity",
        title=f"{active} {_plural(active, 'снимок', 'снимка', 'снимков')} в обработке",
        body="Результат появится на странице снимка, как только анализ завершится.",
        metric=str(active),
        action_label="К снимкам",
        action_href=_STUDIES_PAGE,
    )


def _attention_trend(ctx: _InsightContext) -> Insight | None:
    delta = ctx.attention_delta
    change = delta.change
    # A baseline of nothing makes any change infinite, and a change under a
    # sixth is inside the week-to-week noise of a small clinic.
    if delta.previous == 0 or change is None or abs(change) < _MIN_REPORTABLE_CHANGE:
        return None
    grew = change > 0
    percent = round(abs(change) * 100)
    return Insight(
        key="attention-trend",
        tone=Tone.WARNING if grew else Tone.POSITIVE,
        icon="trending-up" if grew else "trending-down",
        title=f"Патологий на {percent}% {'больше' if grew else 'меньше'}, чем неделей ранее",
        body=(
            f"{delta.current} против {delta.previous} за предыдущие семь дней. "
            "Учитываются находки категории «Патологии» на снимках за период."
        ),
        metric=f"{'+' if grew else '−'}{percent}%",
    )


def _agreement(ctx: _InsightContext) -> Insight | None:
    rate = ctx.review_stats.agreement_rate
    adjudicated = ctx.review_stats.confirmed + ctx.review_stats.rejected
    if rate is None or adjudicated < _AGREEMENT_MIN_SAMPLE:
        return None
    percent = round(rate * 100)
    return Insight(
        key="agreement",
        tone=Tone.POSITIVE if percent >= _GOOD_AGREEMENT_PERCENT else Tone.INFO,
        icon="target",
        title=f"Врачи подтверждают {percent}% находок",
        body=(
            f"По {adjudicated} решениям. Отклонённые находки не попадают в статистику "
            "и становятся обучающими примерами для следующей версии модели."
        ),
        metric=f"{percent}%",
    )


def _most_common(ctx: _InsightContext) -> Insight | None:
    leader = max(ctx.top, key=lambda item: item.count, default=None)
    if leader is None or leader.count == 0:
        return None
    return Insight(
        key="most-common",
        tone=Tone.INFO,
        icon="layers",
        title=f"Чаще всего: {leader.label}",
        body=(
            f"{leader.count} {_plural(leader.count, 'находка', 'находки', 'находок')} "
            "по клинике за всё время."
        ),
        metric=str(leader.count),
    )


def _volume_trend(ctx: _InsightContext) -> Insight | None:
    delta = ctx.studies_delta
    change = delta.change
    if delta.previous == 0 or change is None or abs(change) < _MIN_REPORTABLE_VOLUME_CHANGE:
        return None
    grew = change > 0
    percent = round(abs(change) * 100)
    return Insight(
        key="volume-trend",
        tone=Tone.POSITIVE if grew else Tone.INFO,
        icon="trending-up" if grew else "trending-down",
        title=f"Загрузок на {percent}% {'больше' if grew else 'меньше'}",
        body=(f"{delta.current} за последние семь дней против {delta.previous} за предыдущие."),
        metric=f"{'+' if grew else '−'}{percent}%",
    )


def _getting_started(ctx: _InsightContext) -> Insight | None:
    """The empty clinic, which is otherwise the least useful dashboard of all."""
    if ctx.counts.studies == 0:
        return Insight(
            key="first-study",
            tone=Tone.INFO,
            icon="upload",
            title="Загрузите первый снимок",
            body=(
                "Панорамная рентгенограмма в JPEG или PNG. Анализ занимает несколько "
                "секунд, после чего находки появятся здесь."
            ),
            action_label="Загрузить",
            action_href=_STUDIES_PAGE,
        )
    if ctx.counts.patients == 0:
        return Insight(
            key="no-patients",
            tone=Tone.INFO,
            icon="users",
            title="Снимки не привязаны к пациентам",
            body=(
                "Заведите карты пациентов, чтобы видеть историю обследований и "
                "находить снимки по номеру карты."
            ),
            action_label="Добавить пациента",
            action_href=_PATIENTS_PAGE,
        )
    return None


#: Evaluated in order. Ties inside a tone are broken by this order, so the same
#: clinic state always renders the same page.
_INSIGHT_RULES: Final[tuple[Callable[[_InsightContext], Insight | None], ...]] = (
    _failed_studies,
    _pending_review,
    _in_flight,
    _attention_trend,
    _agreement,
    _most_common,
    _volume_trend,
    _getting_started,
)


def _insights(ctx: _InsightContext) -> list[Insight]:
    """The handful of things worth saying about the aggregates above.

    Each rule states a fact and, where one exists, names the screen that acts
    on it. Nothing here predicts, diagnoses or recommends treatment: the
    product is decision support, and an "insight" that overstepped that would
    be the one part of the page a clinician could not trust.
    """
    candidates = [insight for rule in _INSIGHT_RULES if (insight := rule(ctx)) is not None]
    candidates.sort(key=lambda item: _TONE_ORDER[item.tone])
    return candidates[:_MAX_INSIGHTS]
