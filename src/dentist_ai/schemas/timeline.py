"""Patient-timeline payloads, and the wording the timeline screen reads back.

The label tables sit here rather than in ``clinical/labels.py`` because they
are wording for these responses specifically — ``statusLabel`` is a field the
client renders, not a mapping it maintains — which keeps a translation in one
place on the server, exactly as ``severityLabel`` does for findings.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime, timedelta
from typing import Final

from pydantic import Field, computed_field, field_validator

from dentist_ai.db.models import AppointmentStatus, NoteKind
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale, Severity
from dentist_ai.schemas.common import ApiModel


class TimelineEventKind(enum.StrEnum):
    """Every source the merged history draws on.

    A superset of ``schemas.clinical.TimelineKind``, which describes the
    radiograph-only strip on the patient overview. The two are kept apart
    rather than unified so that widening this one cannot change the shape of a
    response the overview screen already renders.
    """

    PATIENT_CREATED = "patient_created"
    STUDY = "study"
    VOLUME = "volume"
    SCAN = "scan"
    PLAN_ITEM = "plan_item"
    NOTE = "note"
    APPOINTMENT = "appointment"
    CASE = "case"


TIMELINE_KIND_LABELS: Final[dict[TimelineEventKind, dict[Locale, str]]] = {
    TimelineEventKind.PATIENT_CREATED: {
        "ru": "Карта",
        "en": "Record",
        "kk": "Карта",
    },
    TimelineEventKind.STUDY: {"ru": "Снимки", "en": "Radiographs", "kk": "Түсірілімдер"},
    TimelineEventKind.VOLUME: {"ru": "КЛКТ", "en": "CBCT", "kk": "КЛКТ"},
    TimelineEventKind.SCAN: {"ru": "3D-сканы", "en": "3D scans", "kk": "3D сканерлер"},
    TimelineEventKind.PLAN_ITEM: {
        "ru": "Этапы лечения",
        "en": "Plan steps",
        "kk": "Емдеу кезеңдері",
    },
    TimelineEventKind.NOTE: {"ru": "Заметки", "en": "Notes", "kk": "Жазбалар"},
    TimelineEventKind.APPOINTMENT: {"ru": "Приёмы", "en": "Appointments", "kk": "Қабылдаулар"},
    TimelineEventKind.CASE: {
        "ru": "Клинические случаи",
        "en": "Case library",
        "kk": "Клиникалық жағдайлар",
    },
}

NOTE_KIND_LABELS: Final[dict[NoteKind, dict[Locale, str]]] = {
    NoteKind.CLINICAL: {
        "ru": "Клиническая заметка",
        "en": "Clinical note",
        "kk": "Клиникалық жазба",
    },
    NoteKind.ADMINISTRATIVE: {
        "ru": "Административная заметка",
        "en": "Administrative note",
        "kk": "Әкімшілік жазба",
    },
    NoteKind.FOLLOW_UP: {"ru": "Напоминание", "en": "Follow-up", "kk": "Еске салу"},
}

APPOINTMENT_STATUS_LABELS: Final[dict[AppointmentStatus, dict[Locale, str]]] = {
    AppointmentStatus.SCHEDULED: {"ru": "Запланирован", "en": "Scheduled", "kk": "Жоспарланды"},
    AppointmentStatus.CONFIRMED: {"ru": "Подтверждён", "en": "Confirmed", "kk": "Расталды"},
    AppointmentStatus.COMPLETED: {"ru": "Состоялся", "en": "Completed", "kk": "Өтті"},
    AppointmentStatus.CANCELLED: {"ru": "Отменён", "en": "Cancelled", "kk": "Болдырылмады"},
    AppointmentStatus.NO_SHOW: {"ru": "Не пришёл", "en": "No-show", "kk": "Келмеді"},
}

#: A full surgical day. Longer than this is two visits that were entered as
#: one, and letting it through would put a bar across a week of the calendar.
MAX_APPOINTMENT_MINUTES: Final[int] = 480
#: Shorter than a check of a healing socket, i.e. not a booking.
MIN_APPOINTMENT_MINUTES: Final[int] = 5


def _label(table: dict[Locale, str], locale: Locale) -> str:
    return table.get(locale) or table[DEFAULT_LOCALE]


def timeline_kind_label(kind: TimelineEventKind, locale: Locale = DEFAULT_LOCALE) -> str:
    return _label(TIMELINE_KIND_LABELS[kind], locale)


def note_kind_label(kind: NoteKind, locale: Locale = DEFAULT_LOCALE) -> str:
    return _label(NOTE_KIND_LABELS[kind], locale)


def appointment_status_label(status: AppointmentStatus, locale: Locale = DEFAULT_LOCALE) -> str:
    return _label(APPOINTMENT_STATUS_LABELS[status], locale)


def _require_aware(value: datetime) -> datetime:
    """Refuse a datetime with no offset, then normalise it to UTC.

    ``UtcDateTime`` treats a naive bind as UTC, which is right for values this
    codebase produces and wrong for one a browser sends: a clinic on
    ``Asia/Almaty`` posting "14:00" would be booked at 20:00 local, and nothing
    downstream could tell. Refusing the ambiguity costs a round-trip; accepting
    it costs a missed appointment.
    """
    if value.tzinfo is None:
        raise ValueError("Укажите время с часовым поясом, например 2026-08-02T14:00:00+05:00")
    return value.astimezone(UTC)


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
class TimelineMetricResponse(ApiModel):
    """One number the entry carries, so the screen can plot it over time.

    Pre-labelled and pre-scaled by the server: the client charts the series it
    is given rather than deciding what a study's "value" is.
    """

    label: str
    value: float
    unit: str | None = None


class TimelineEventResponse(ApiModel):
    kind: TimelineEventKind
    at: datetime
    title: str
    subtitle: str | None = None
    href: str | None = None
    icon: str
    severity: Severity | None = None
    metric: TimelineMetricResponse | None = None


class TimelineKindCount(ApiModel):
    kind: TimelineEventKind
    label: str
    count: int


class TimelineSummaryResponse(ApiModel):
    """The header strip: how much history there is and what is still open."""

    total: int
    counts: list[TimelineKindCount] = Field(default_factory=list)
    #: Bounds of what has *already* happened. A booked visit is not history,
    #: so it does not move ``last_at`` — it appears as ``next_appointment_at``.
    first_at: datetime | None = None
    last_at: datetime | None = None
    next_appointment_at: datetime | None = None
    open_plan_items: int = 0


class PatientTimelineResponse(ApiModel):
    patient_id: int
    patient_name: str
    summary: TimelineSummaryResponse
    entries: list[TimelineEventResponse] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
class PatientNoteResponse(ApiModel):
    id: int
    kind: NoteKind
    kind_label: str
    body: str
    author_id: int | None
    author_name: str | None
    created_at: datetime


class NoteCreateRequest(ApiModel):
    kind: NoteKind = NoteKind.CLINICAL
    body: str = Field(min_length=1, max_length=8000)

    @field_validator("body")
    @classmethod
    def _trim(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Заметка не может быть пустой")
        return trimmed


# --------------------------------------------------------------------------
# Appointments
# --------------------------------------------------------------------------
class AppointmentResponse(ApiModel):
    id: int
    title: str
    starts_at: datetime
    duration_minutes: int
    status: AppointmentStatus
    status_label: str
    notes: str | None
    plan_item_id: int | None
    created_at: datetime
    created_by_name: str | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ends_at(self) -> datetime:
        return self.starts_at + timedelta(minutes=self.duration_minutes)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_upcoming(self) -> bool:
        return self.starts_at > datetime.now(UTC)


class AppointmentCreateRequest(ApiModel):
    title: str = Field(min_length=1, max_length=160)
    starts_at: datetime
    duration_minutes: int = Field(
        default=60, ge=MIN_APPOINTMENT_MINUTES, le=MAX_APPOINTMENT_MINUTES
    )
    #: The plan step this visit realises, when it realises one.
    plan_item_id: int | None = None
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("starts_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _require_aware(value)


class AppointmentUpdateRequest(ApiModel):
    """PATCH semantics: an omitted field keeps the value it had.

    ``notes`` is therefore cleared with an empty string rather than with
    ``null``, which here means "leave it alone".
    """

    title: str | None = Field(default=None, min_length=1, max_length=160)
    starts_at: datetime | None = None
    duration_minutes: int | None = Field(
        default=None, ge=MIN_APPOINTMENT_MINUTES, le=MAX_APPOINTMENT_MINUTES
    )
    status: AppointmentStatus | None = None
    notes: str | None = Field(default=None, max_length=4000)

    @field_validator("starts_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_aware(value)
