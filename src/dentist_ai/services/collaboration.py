"""Discussion and review hand-off over any resource in the workspace.

Both tables key on ``resource_type``/``resource_id`` rather than a nullable
foreign key per kind, so a new screen gains a thread without a migration. What
that costs is the database's ability to check that a thread points at something
real, which is why :data:`RESOURCE_TYPES` exists and every entry point is
measured against it: an unvalidated string would let a client file comments
against a resource nothing can render, invisible on every screen yet counted by
every total.

Notifications fan out to *participants*, never to the organisation. A remark on
a busy case would otherwise land in ten people's centres, which is the quickest
way to teach a practice that the badge means nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Final

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.core.errors import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import (
    AssignmentStatus,
    CaseEntry,
    Comment,
    NotificationKind,
    NotificationTone,
    ReviewAssignment,
    Study,
    TreatmentPlan,
    User,
    Volume,
)
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext
from dentist_ai.services.notifications import NotificationService

#: The resource kinds a thread or an assignment may attach to. Everything else
#: is refused at the door — see the module docstring.
RESOURCE_TYPES: Final[frozenset[str]] = frozenset({"study", "volume", "patient", "plan", "case"})

#: Screens that can render a resource today. A plan and a library case have no
#: page of their own yet, so their notifications carry no link rather than one
#: that 404s.
_RESOURCE_PAGES: Final[dict[str, str]] = {
    "study": "/app/studies",
    "volume": "/app/volumes",
    "patient": "/app/patients",
}

_OPEN_ASSIGNMENT_STATUSES: Final[tuple[AssignmentStatus, ...]] = (
    AssignmentStatus.PENDING,
    AssignmentStatus.ACCEPTED,
)

#: How much of a remark rides along in the notification body. Enough to
#: recognise which conversation it is, short enough to stay one line.
_EXCERPT_LENGTH: Final[int] = 140

_RESOURCE_TYPE_HINT: Final[str] = ", ".join(sorted(RESOURCE_TYPES))


def resource_href(resource_type: str, resource_id: str) -> str | None:
    """Where a resource can be read, or ``None`` when nothing renders it yet."""
    prefix = _RESOURCE_PAGES.get(resource_type)
    return None if prefix is None else f"{prefix}/{resource_id}"


@dataclass(frozen=True, slots=True)
class CommentThread:
    """One top-level remark and the answers to it, in the order written."""

    root: Comment
    replies: list[Comment]


class CollaborationService:
    def __init__(
        self,
        session: AsyncSession,
        audit: AuditService,
        notifications: NotificationService,
    ) -> None:
        self._session = session
        self._audit = audit
        self._notifications = notifications

    # -- comments ---------------------------------------------------------
    async def list_thread(
        self, *, organization_id: int, resource_type: str, resource_id: str
    ) -> list[CommentThread]:
        """The whole discussion for one resource, in one round-trip.

        Roots and replies come back together and are grouped in Python; asking
        the database for the answers to each root would make the cost of a
        thread scale with how lively it is.
        """
        _assert_known_resource(resource_type)
        rows = await self._session.scalars(
            select(Comment)
            .options(selectinload(Comment.author))
            .where(
                Comment.organization_id == organization_id,
                Comment.resource_type == resource_type,
                Comment.resource_id == resource_id,
            )
            .order_by(Comment.created_at, Comment.id)
        )
        comments = list(rows.all())

        replies: defaultdict[int, list[Comment]] = defaultdict(list)
        for comment in comments:
            if comment.parent_id is not None:
                replies[comment.parent_id].append(comment)

        return [
            CommentThread(root=comment, replies=replies[comment.id])
            for comment in comments
            if comment.parent_id is None
        ]

    async def add_comment(
        self,
        *,
        actor: User,
        context: RequestContext,
        resource_type: str,
        resource_id: str,
        body: str,
        parent_id: int | None = None,
    ) -> Comment:
        _assert_known_resource(resource_type)

        root_id: int | None = None
        if parent_id is not None:
            parent = await self._get_comment(parent_id, actor.organization_id)
            if parent.resource_type != resource_type or parent.resource_id != resource_id:
                raise ValidationError("Ответ должен относиться к тому же ресурсу.")
            # A thread that nests without limit needs a renderer that indents
            # without limit. An answer to an answer joins the same exchange.
            root_id = parent.parent_id or parent.id

        comment = Comment(
            organization_id=actor.organization_id,
            author_id=actor.id,
            resource_type=resource_type,
            resource_id=resource_id,
            parent_id=root_id,
            body=body,
        )
        self._session.add(comment)
        await self._session.flush()

        await self._notify_participants(comment, actor=actor)
        await self._audit.record(
            action=AuditAction.COMMENT_ADDED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type=resource_type,
            resource_id=resource_id,
            context=context,
        )
        return await self._get_comment(comment.id, actor.organization_id)

    async def resolve_comment(self, comment_id: int, *, actor: User) -> Comment:
        """Mark a remark answered.

        Idempotent, and deliberately keeps the *first* resolver: who closed a
        question is the fact worth having, and a second click should not
        rewrite it.
        """
        comment = await self._get_comment(comment_id, actor.organization_id)
        if comment.resolved_at is None:
            comment.resolved_at = utcnow()
            comment.resolved_by_id = actor.id
        return comment

    async def delete_comment(self, comment_id: int, *, actor: User) -> None:
        comment = await self._get_comment(comment_id, actor.organization_id)
        # The same shape as `UserRole.can_delete_patients`: a coarse role
        # property, not a permission table. An owner can clear anything in
        # their clinic; everyone else owns only their own words.
        if comment.author_id != actor.id and not actor.role.can_manage_members:
            raise PermissionDeniedError(
                "Удалить комментарий может только автор или владелец клиники."
            )
        # Replies cascade with the root: an answer without its question is a
        # remark nobody can interpret.
        await self._session.delete(comment)

    # -- assignments ------------------------------------------------------
    async def list_assignments(
        self, *, organization_id: int, resource_type: str, resource_id: str
    ) -> list[ReviewAssignment]:
        _assert_known_resource(resource_type)
        rows = await self._session.scalars(
            self._assignment_query().where(
                ReviewAssignment.organization_id == organization_id,
                ReviewAssignment.resource_type == resource_type,
                ReviewAssignment.resource_id == resource_id,
            )
        )
        return list(rows.all())

    async def list_open_for_user(
        self, *, user_id: int, organization_id: int, limit: int = 20
    ) -> list[ReviewAssignment]:
        """What one colleague still owes an answer on, newest first."""
        rows = await self._session.scalars(
            self._assignment_query()
            .where(
                ReviewAssignment.organization_id == organization_id,
                ReviewAssignment.assignee_id == user_id,
                ReviewAssignment.status.in_(_OPEN_ASSIGNMENT_STATUSES),
            )
            .limit(limit)
        )
        return list(rows.all())

    async def create_assignment(
        self,
        *,
        actor: User,
        context: RequestContext,
        resource_type: str,
        resource_id: str,
        assignee_id: int,
        due_on: date | None = None,
        note: str | None = None,
    ) -> ReviewAssignment:
        _assert_known_resource(resource_type)

        assignee = await self._session.scalar(
            select(User).where(
                User.id == assignee_id,
                User.organization_id == actor.organization_id,
                User.is_active.is_(True),
            )
        )
        if assignee is None:
            # 404 rather than 403 for someone else's clinic, exactly as for a
            # patient: the response must not confirm the account exists.
            raise NotFoundError("Коллега не найден.")

        duplicate = await self._session.scalar(
            select(ReviewAssignment.id).where(
                ReviewAssignment.organization_id == actor.organization_id,
                ReviewAssignment.resource_type == resource_type,
                ReviewAssignment.resource_id == resource_id,
                ReviewAssignment.assignee_id == assignee_id,
                ReviewAssignment.status.in_(_OPEN_ASSIGNMENT_STATUSES),
            )
        )
        if duplicate is not None:
            raise ConflictError("Этот случай уже назначен этому коллеге.")

        assignment = ReviewAssignment(
            organization_id=actor.organization_id,
            resource_type=resource_type,
            resource_id=resource_id,
            assignee_id=assignee.id,
            assigned_by_id=actor.id,
            status=AssignmentStatus.PENDING,
            due_on=due_on,
            note=note,
        )
        self._session.add(assignment)
        await self._session.flush()

        if assignee.id != actor.id:
            # Assigning a case to yourself is a legitimate to-do; telling
            # yourself about it is not.
            await self._notifications.push(
                organization_id=actor.organization_id,
                user_id=assignee.id,
                kind=NotificationKind.REVIEW_ASSIGNED,
                tone=NotificationTone.INFO,
                title="Вас попросили посмотреть случай",
                body=_assignment_body(actor, due_on=due_on, note=note),
                href=resource_href(resource_type, resource_id),
            )

        await self._audit.record(
            action=AuditAction.REVIEW_ASSIGNED,
            organization_id=actor.organization_id,
            actor_id=actor.id,
            resource_type=resource_type,
            resource_id=resource_id,
            context=context,
            detail=str(assignee.id),
        )
        return await self._get_assignment(assignment.id, actor.organization_id)

    async def set_assignment_status(
        self, assignment_id: int, status: AssignmentStatus, *, actor: User
    ) -> ReviewAssignment:
        """Accept, complete or decline. The assigner may only withdraw.

        ``AssignmentStatus`` has no ``cancelled`` value, and does not need one:
        a withdrawn request and a refused one leave the case in the same
        place — closed, with nobody waiting on it.
        """
        if status is AssignmentStatus.PENDING:
            raise ValidationError("Назначение нельзя вернуть в исходное состояние.")

        assignment = await self._get_assignment(assignment_id, actor.organization_id)
        withdrawing = assignment.assigned_by_id == actor.id and status is AssignmentStatus.DECLINED
        if assignment.assignee_id != actor.id and not withdrawing:
            raise PermissionDeniedError("Изменить статус может только исполнитель.")

        assignment.status = status
        assignment.completed_at = utcnow() if status is AssignmentStatus.COMPLETED else None
        return assignment

    # -- notification fan-out ---------------------------------------------
    async def _notify_participants(self, comment: Comment, *, actor: User) -> None:
        """Tell everyone already involved with the resource, never the author.

        "Involved" is whoever has written in the thread, whoever owns the
        resource, and whoever is currently assigned to review it. The author is
        removed from the union at the end rather than skipped per source, so
        someone who qualifies twice cannot slip through one of the branches.
        """
        recipients = await self._participants(comment)
        recipients.discard(actor.id)
        if not recipients:
            return

        href = resource_href(comment.resource_type, comment.resource_id)
        title = "Ответ в обсуждении" if comment.parent_id else "Новый комментарий"
        body = f"{actor.full_name}: {comment.body[:_EXCERPT_LENGTH]}"
        for user_id in sorted(recipients):
            await self._notifications.push(
                organization_id=comment.organization_id,
                user_id=user_id,
                kind=NotificationKind.COMMENT_ADDED,
                tone=NotificationTone.INFO,
                title=title,
                body=body,
                href=href,
            )

    async def _participants(self, comment: Comment) -> set[int]:
        authors = await self._session.scalars(
            select(Comment.author_id).where(
                Comment.organization_id == comment.organization_id,
                Comment.resource_type == comment.resource_type,
                Comment.resource_id == comment.resource_id,
                Comment.author_id.is_not(None),
            )
        )
        assignees = await self._session.scalars(
            select(ReviewAssignment.assignee_id).where(
                ReviewAssignment.organization_id == comment.organization_id,
                ReviewAssignment.resource_type == comment.resource_type,
                ReviewAssignment.resource_id == comment.resource_id,
                ReviewAssignment.status.in_(_OPEN_ASSIGNMENT_STATUSES),
            )
        )
        people = {author_id for author_id in authors.all() if author_id is not None}
        people.update(assignees.all())

        owner = await self._resource_owner(comment)
        if owner is not None:
            people.add(owner)
        return people

    async def _resource_owner(self, comment: Comment) -> int | None:
        """Who a resource belongs to, where the kind has an owner at all.

        A patient chart belongs to the clinic rather than to whoever typed it
        in, so a remark on one reaches the thread and nobody else.
        """
        organization_id = comment.organization_id
        resource_id = comment.resource_id
        if comment.resource_type == "study":
            return await self._session.scalar(
                select(Study.uploaded_by_id).where(
                    Study.public_id == resource_id,
                    Study.organization_id == organization_id,
                )
            )
        if comment.resource_type == "volume":
            return await self._session.scalar(
                select(Volume.uploaded_by_id).where(
                    Volume.public_id == resource_id,
                    Volume.organization_id == organization_id,
                )
            )
        if comment.resource_type == "plan":
            return await self._session.scalar(
                select(TreatmentPlan.created_by_id).where(
                    TreatmentPlan.public_id == resource_id,
                    TreatmentPlan.organization_id == organization_id,
                )
            )
        if comment.resource_type == "case":
            return await self._session.scalar(
                select(CaseEntry.created_by_id).where(
                    CaseEntry.public_id == resource_id,
                    CaseEntry.organization_id == organization_id,
                )
            )
        return None

    # -- helpers ----------------------------------------------------------
    async def _get_comment(self, comment_id: int, organization_id: int) -> Comment:
        comment = await self._session.scalar(
            select(Comment)
            .options(selectinload(Comment.author))
            # Without this a comment already in the identity map keeps the
            # unloaded `author` it was inserted with, and reading the name
            # during serialisation would attempt IO outside the async context.
            .execution_options(populate_existing=True)
            .where(Comment.id == comment_id, Comment.organization_id == organization_id)
        )
        if comment is None:
            raise NotFoundError("Комментарий не найден.")
        return comment

    async def _get_assignment(self, assignment_id: int, organization_id: int) -> ReviewAssignment:
        assignment = await self._session.scalar(
            self._assignment_query()
            .execution_options(populate_existing=True)
            .where(
                ReviewAssignment.id == assignment_id,
                ReviewAssignment.organization_id == organization_id,
            )
        )
        if assignment is None:
            raise NotFoundError("Назначение не найдено.")
        return assignment

    @staticmethod
    def _assignment_query() -> Select[tuple[ReviewAssignment]]:
        return (
            select(ReviewAssignment)
            .options(
                selectinload(ReviewAssignment.assignee),
                selectinload(ReviewAssignment.assigned_by),
            )
            .order_by(ReviewAssignment.created_at.desc(), ReviewAssignment.id.desc())
        )


def _assert_known_resource(resource_type: str) -> None:
    if resource_type not in RESOURCE_TYPES:
        raise ValidationError(
            "Неизвестный тип ресурса.",
            field_errors={"resourceType": f"Допустимые значения: {_RESOURCE_TYPE_HINT}."},
        )


def _assignment_body(actor: User, *, due_on: date | None, note: str | None) -> str:
    parts = [f"{actor.full_name} просит вас посмотреть материал."]
    if due_on is not None:
        parts.append(f"Срок: {due_on:%d.%m.%Y}.")
    if note:
        parts.append(note[:_EXCERPT_LENGTH])
    return " ".join(parts)
