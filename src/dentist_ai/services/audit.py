"""Audit trail: who looked at or changed a patient's data, and when."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from dentist_ai.db.base import utcnow
from dentist_ai.db.models import AuditEvent


class AuditAction(enum.StrEnum):
    LOGIN_SUCCEEDED = "login.succeeded"
    LOGIN_FAILED = "login.failed"
    LOGOUT = "logout"
    USER_REGISTERED = "user.registered"
    PASSWORD_CHANGED = "user.password_changed"  # noqa: S105 - an event name, not a secret
    PROFILE_UPDATED = "user.profile_updated"

    PATIENT_CREATED = "patient.created"
    PATIENT_VIEWED = "patient.viewed"
    PATIENT_UPDATED = "patient.updated"
    PATIENT_ARCHIVED = "patient.archived"

    STUDY_UPLOADED = "study.uploaded"
    STUDY_VIEWED = "study.viewed"
    STUDY_IMAGE_ACCESSED = "study.image_accessed"
    STUDY_UPDATED = "study.updated"
    STUDY_DELETED = "study.deleted"
    STUDY_EXPORTED = "study.exported"
    FINDING_REVIEWED = "finding.reviewed"
    FINDING_RECHARTED = "finding.recharted"

    SCAN_UPLOADED = "scan.uploaded"
    SCAN_VIEWED = "scan.viewed"
    SCAN_UPDATED = "scan.updated"
    SCAN_DELETED = "scan.deleted"

    VOLUME_UPLOADED = "volume.uploaded"
    VOLUME_VIEWED = "volume.viewed"
    VOLUME_VOXELS_ACCESSED = "volume.voxels_accessed"
    VOLUME_ANALYSED = "volume.analysed"
    VOLUME_UPDATED = "volume.updated"
    VOLUME_DELETED = "volume.deleted"
    VOLUME_EXPORTED = "volume.exported"
    MEASUREMENT_ADDED = "volume.measurement_added"
    MEASUREMENT_REMOVED = "volume.measurement_removed"
    ANNOTATION_ADDED = "volume.annotation_added"
    ANNOTATION_REMOVED = "volume.annotation_removed"

    REPORT_GENERATED = "report.generated"
    CASE_PUBLISHED = "case.published"
    CASE_UPDATED = "case.updated"
    CASE_DELETED = "case.deleted"
    COMMENT_ADDED = "collaboration.comment_added"
    REVIEW_ASSIGNED = "collaboration.review_assigned"
    ASSISTANT_QUERIED = "assistant.queried"

    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    PLAN_ITEM_ADDED = "plan.item_added"
    PLAN_ITEM_UPDATED = "plan.item_updated"
    PLAN_ITEM_REMOVED = "plan.item_removed"


@dataclass(frozen=True, slots=True)
class RequestContext:
    """The bits of the HTTP request an audit row cares about."""

    ip_address: str | None
    user_agent: str | None


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        *,
        action: AuditAction,
        organization_id: int,
        actor_id: int | None,
        resource_type: str,
        resource_id: str | int | None = None,
        context: RequestContext | None = None,
        detail: str | None = None,
    ) -> None:
        self._session.add(
            AuditEvent(
                organization_id=organization_id,
                actor_id=actor_id,
                action=action.value,
                resource_type=resource_type,
                resource_id=None if resource_id is None else str(resource_id),
                ip_address=context.ip_address if context else None,
                # Truncated: the column is indexed-adjacent and a 4 KB
                # user-agent from a crawler is not worth the storage.
                user_agent=(context.user_agent[:255] if context and context.user_agent else None),
                detail=detail,
                created_at=utcnow(),
            )
        )
