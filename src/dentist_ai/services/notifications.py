"""The notification centre.

Deliberately small. A notification here is a row addressed to one user, with a
tone, a title and somewhere to go — not a delivery mechanism. There is no
email, no push, no websocket, and adding one later is a consumer of this table
rather than a change to it.

Two design points worth stating.

**Addressed to a user, not to an organisation.** A clinic-wide feed would mean
either every member sees every upload — which is noise in a practice of ten —
or a read-state table per member anyway. Fanning out at write time costs a row
per recipient and makes the unread badge a single indexed count.

**Deduplicated within a short window.** Re-running an analysis three times
while tuning a scan should not leave three identical "analysis complete"
entries. The same kind, for the same user, pointing at the same resource,
inside the window, updates the existing row instead of adding one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import Executable

from dentist_ai.db.base import utcnow
from dentist_ai.db.models import Notification, NotificationKind, NotificationTone, User
from dentist_ai.services.audit import AuditService

#: Window inside which an identical notification is folded into the existing
#: one rather than added beside it.
_DEDUPE_WINDOW: Final[timedelta] = timedelta(minutes=10)
#: Read notifications older than this are pruned when the centre is opened, so
#: the table does not grow without bound on a busy clinic.
_RETENTION: Final[timedelta] = timedelta(days=45)


@dataclass(frozen=True, slots=True)
class NotificationCounts:
    total: int
    unread: int


class NotificationService:
    def __init__(self, session: AsyncSession, audit: AuditService | None = None) -> None:
        self._session = session
        self._audit = audit

    async def push(
        self,
        *,
        organization_id: int,
        user_id: int,
        kind: NotificationKind,
        tone: NotificationTone,
        title: str,
        body: str | None = None,
        href: str | None = None,
    ) -> Notification:
        """Deliver one notification, folding it into a recent identical one."""
        now = utcnow()
        existing = await self._session.scalar(
            select(Notification)
            .where(
                Notification.user_id == user_id,
                Notification.kind == kind,
                Notification.href == href,
                Notification.created_at >= now - _DEDUPE_WINDOW,
            )
            .order_by(Notification.created_at.desc())
            .limit(1)
        )
        if existing is not None:
            existing.title = title[:160]
            existing.body = body
            existing.tone = tone
            existing.created_at = now
            # Folding in a new event makes the entry unread again: the user has
            # not seen *this* update, only the one it replaced.
            existing.read_at = None
            return existing

        notification = Notification(
            organization_id=organization_id,
            user_id=user_id,
            kind=kind,
            tone=tone,
            title=title[:160],
            body=body,
            href=href,
            created_at=now,
        )
        self._session.add(notification)
        await self._session.flush()
        return notification

    async def broadcast(
        self,
        *,
        organization_id: int,
        kind: NotificationKind,
        tone: NotificationTone,
        title: str,
        body: str | None = None,
        href: str | None = None,
        exclude_user_id: int | None = None,
    ) -> int:
        """Deliver to every active member of an organisation.

        Used for events that are the clinic's business rather than one
        person's — a critical finding on a shared case, a case assigned for
        review by someone who has since left.
        """
        members = list(
            await self._session.scalars(
                select(User.id).where(
                    User.organization_id == organization_id,
                    User.is_active.is_(True),
                )
            )
        )
        delivered = 0
        for user_id in members:
            if user_id == exclude_user_id:
                continue
            await self.push(
                organization_id=organization_id,
                user_id=user_id,
                kind=kind,
                tone=tone,
                title=title,
                body=body,
                href=href,
            )
            delivered += 1
        return delivered

    async def list_for_user(
        self,
        user_id: int,
        *,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Notification], int]:
        base = select(Notification).where(Notification.user_id == user_id)
        if unread_only:
            base = base.where(Notification.read_at.is_(None))

        total = await self._session.scalar(select(func.count()).select_from(base.subquery()))
        rows = await self._session.scalars(
            base.order_by(Notification.created_at.desc(), Notification.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows.all()), int(total or 0)

    async def counts(self, user_id: int) -> NotificationCounts:
        total = await self._session.scalar(
            select(func.count()).select_from(Notification).where(Notification.user_id == user_id)
        )
        unread = await self._session.scalar(
            select(func.count())
            .select_from(Notification)
            .where(Notification.user_id == user_id, Notification.read_at.is_(None))
        )
        return NotificationCounts(total=int(total or 0), unread=int(unread or 0))

    async def mark_read(self, user_id: int, notification_id: int) -> bool:
        return bool(
            await self._affected_rows(
                update(Notification)
                .where(
                    Notification.id == notification_id,
                    Notification.user_id == user_id,
                    Notification.read_at.is_(None),
                )
                .values(read_at=utcnow())
            )
        )

    async def mark_all_read(self, user_id: int) -> int:
        return await self._affected_rows(
            update(Notification)
            .where(Notification.user_id == user_id, Notification.read_at.is_(None))
            .values(read_at=utcnow())
        )

    async def prune(self, user_id: int) -> int:
        """Drop read notifications past the retention window."""
        return await self._affected_rows(
            delete(Notification).where(
                Notification.user_id == user_id,
                Notification.read_at.is_not(None),
                Notification.created_at < utcnow() - _RETENTION,
            )
        )

    async def _affected_rows(self, statement: Executable) -> int:
        """Row count for a bulk UPDATE or DELETE.

        ``Session.execute`` is typed as returning a plain ``Result``, which has
        no ``rowcount``; for a DML statement the runtime object is always a
        ``CursorResult``, which does. Narrowing once here keeps the cast out of
        three call sites.
        """
        result = await self._session.execute(statement)
        assert isinstance(result, CursorResult)
        return int(result.rowcount or 0)
