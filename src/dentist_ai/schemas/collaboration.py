"""Comment-thread and review-assignment payloads.

The wording table lives beside the models rather than in ``clinical/labels.py``
because it describes this feature's workflow only, and because the label is
part of the response: the client renders ``statusLabel`` instead of mapping the
enum itself, so a translation exists in exactly one place.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Final

from pydantic import Field, computed_field, field_validator

from dentist_ai.db.models import AssignmentStatus
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale
from dentist_ai.schemas.common import ApiModel

#: A comment is a remark on a case, not a document. Past this length the thread
#: stops being scannable and the content belongs in the patient's notes.
MAX_COMMENT_LENGTH: Final[int] = 4000

ASSIGNMENT_STATUS_LABELS: Final[dict[AssignmentStatus, dict[Locale, str]]] = {
    AssignmentStatus.PENDING: {"ru": "Ожидает", "en": "Pending", "kk": "Күтілуде"},
    AssignmentStatus.ACCEPTED: {"ru": "Принято", "en": "Accepted", "kk": "Қабылданды"},
    AssignmentStatus.COMPLETED: {"ru": "Разобрано", "en": "Completed", "kk": "Қаралды"},
    AssignmentStatus.DECLINED: {"ru": "Отклонено", "en": "Declined", "kk": "Бас тартылды"},
}


def assignment_status_label(status: AssignmentStatus, locale: Locale = DEFAULT_LOCALE) -> str:
    table = ASSIGNMENT_STATUS_LABELS[status]
    return table.get(locale) or table[DEFAULT_LOCALE]


# --------------------------------------------------------------------------
# Comments
# --------------------------------------------------------------------------
class CommentReplyResponse(ApiModel):
    id: int
    parent_id: int | None
    author_id: int | None
    author_name: str | None
    author_initials: str | None
    body: str
    created_at: datetime
    resolved_at: datetime | None
    resolved_by_id: int | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None


class CommentResponse(CommentReplyResponse):
    """A top-level comment together with its replies.

    A reply is a different type, without a ``replies`` field of its own, so the
    contract cannot describe a nesting depth the thread renderer does not have.
    """

    replies: list[CommentReplyResponse] = Field(default_factory=list)


class CommentThreadResponse(ApiModel):
    resource_type: str
    resource_id: str
    comments: list[CommentResponse] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_count(self) -> int:
        return sum(1 + len(comment.replies) for comment in self.comments)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unresolved_count(self) -> int:
        """Open top-level remarks. A reply is part of its root, not a question
        of its own, so resolving the root closes the whole exchange."""
        return sum(1 for comment in self.comments if comment.resolved_at is None)


class CommentCreateRequest(ApiModel):
    resource_type: str = Field(
        max_length=24, description="One of: study, volume, patient, plan, case."
    )
    resource_id: str = Field(min_length=1, max_length=32)
    body: str = Field(min_length=1, max_length=MAX_COMMENT_LENGTH)
    #: Set to answer an existing comment. Answering a reply joins its root
    #: rather than opening a third level.
    parent_id: int | None = None

    @field_validator("body")
    @classmethod
    def _trim(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Комментарий не может быть пустым")
        return trimmed


# --------------------------------------------------------------------------
# Review assignments
# --------------------------------------------------------------------------
class AssignmentResponse(ApiModel):
    id: int
    resource_type: str
    resource_id: str
    status: AssignmentStatus
    status_label: str
    due_on: date | None
    note: str | None
    assignee_id: int
    assignee_name: str | None
    assigned_by_id: int | None
    assigned_by_name: str | None
    href: str | None
    created_at: datetime
    completed_at: datetime | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_overdue(self) -> bool:
        """Past its due date while still expecting an answer.

        A closed assignment is never overdue: the work is done, and colouring
        it red would put permanent noise on the dashboard widget.
        """
        if self.due_on is None or not self.status.is_open:
            return False
        return self.due_on < datetime.now(UTC).date()


class AssignmentCreateRequest(ApiModel):
    resource_type: str = Field(
        max_length=24, description="One of: study, volume, patient, plan, case."
    )
    resource_id: str = Field(min_length=1, max_length=32)
    assignee_id: int
    due_on: date | None = None
    note: str | None = Field(default=None, max_length=2000)


class AssignmentStatusRequest(ApiModel):
    """``pending`` is not accepted: an assignment moves forward, never back."""

    status: AssignmentStatus
