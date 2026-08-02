"""Comment threads and review assignments."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from dentist_ai.api.deps import (
    ApiRateLimit,
    AuditDep,
    CurrentUser,
    DbSession,
    NotificationDep,
    RequestCtx,
)
from dentist_ai.db.models import Comment, ReviewAssignment
from dentist_ai.ml.taxonomy import DEFAULT_LOCALE, Locale
from dentist_ai.schemas.collaboration import (
    AssignmentCreateRequest,
    AssignmentResponse,
    AssignmentStatusRequest,
    CommentCreateRequest,
    CommentReplyResponse,
    CommentResponse,
    CommentThreadResponse,
    assignment_status_label,
)
from dentist_ai.schemas.common import OkResponse
from dentist_ai.services.collaboration import (
    CollaborationService,
    CommentThread,
    resource_href,
)


# The composition root for this feature. It sits beside the routes rather than
# in `api/deps.py` only because the feature is self-contained; nothing else
# constructs a `CollaborationService`.
def get_collaboration_service(
    session: DbSession, audit: AuditDep, notifications: NotificationDep
) -> CollaborationService:
    return CollaborationService(session, audit, notifications)


type CollaborationDep = Annotated[CollaborationService, Depends(get_collaboration_service)]

#: The two halves of a resource reference, spelled the way the client already
#: holds them. Aliased rather than snake_cased so a screen can pass its own
#: descriptor straight through.
ResourceTypeQuery = Annotated[
    str,
    Query(alias="resourceType", max_length=24, description="study, volume, patient, plan or case"),
]
ResourceIdQuery = Annotated[str, Query(alias="resourceId", min_length=1, max_length=32)]

router = APIRouter(prefix="/collaboration", tags=["collaboration"], dependencies=[ApiRateLimit])

#: A dashboard widget, not a work queue: past this the clinician needs the
#: full list rather than a taller card.
MAX_OPEN_ASSIGNMENTS = 20


@router.get("/comments", response_model=CommentThreadResponse, summary="Thread for a resource")
async def list_comments(
    user: CurrentUser,
    collaboration: CollaborationDep,
    resource_type: ResourceTypeQuery,
    resource_id: ResourceIdQuery,
) -> CommentThreadResponse:
    threads = await collaboration.list_thread(
        organization_id=user.organization_id,
        resource_type=resource_type,
        resource_id=resource_id,
    )
    return CommentThreadResponse(
        resource_type=resource_type,
        resource_id=resource_id,
        comments=[_present_thread(thread) for thread in threads],
    )


@router.post(
    "/comments",
    response_model=CommentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a comment or a reply",
)
async def add_comment(
    payload: CommentCreateRequest,
    user: CurrentUser,
    collaboration: CollaborationDep,
    context: RequestCtx,
) -> CommentResponse:
    comment = await collaboration.add_comment(
        actor=user,
        context=context,
        resource_type=payload.resource_type,
        resource_id=payload.resource_id,
        body=payload.body,
        parent_id=payload.parent_id,
    )
    return _present_thread(CommentThread(root=comment, replies=[]))


@router.post(
    "/comments/{comment_id}/resolve",
    response_model=CommentResponse,
    summary="Mark a comment answered",
)
async def resolve_comment(
    comment_id: int,
    user: CurrentUser,
    collaboration: CollaborationDep,
) -> CommentResponse:
    comment = await collaboration.resolve_comment(comment_id, actor=user)
    return _present_thread(CommentThread(root=comment, replies=[]))


@router.delete("/comments/{comment_id}", response_model=OkResponse, summary="Delete a comment")
async def delete_comment(
    comment_id: int,
    user: CurrentUser,
    collaboration: CollaborationDep,
) -> OkResponse:
    await collaboration.delete_comment(comment_id, actor=user)
    return OkResponse()


@router.get(
    "/assignments",
    response_model=list[AssignmentResponse],
    summary="Review assignments on a resource",
)
async def list_assignments(
    user: CurrentUser,
    collaboration: CollaborationDep,
    resource_type: ResourceTypeQuery,
    resource_id: ResourceIdQuery,
) -> list[AssignmentResponse]:
    rows = await collaboration.list_assignments(
        organization_id=user.organization_id,
        resource_type=resource_type,
        resource_id=resource_id,
    )
    return [_present_assignment(row, user.locale) for row in rows]


@router.get(
    "/assignments/mine",
    response_model=list[AssignmentResponse],
    summary="What the current user still owes an answer on",
)
async def list_my_assignments(
    user: CurrentUser,
    collaboration: CollaborationDep,
) -> list[AssignmentResponse]:
    rows = await collaboration.list_open_for_user(
        user_id=user.id,
        organization_id=user.organization_id,
        limit=MAX_OPEN_ASSIGNMENTS,
    )
    return [_present_assignment(row, user.locale) for row in rows]


@router.post(
    "/assignments",
    response_model=AssignmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ask a colleague to review a case",
)
async def create_assignment(
    payload: AssignmentCreateRequest,
    user: CurrentUser,
    collaboration: CollaborationDep,
    context: RequestCtx,
) -> AssignmentResponse:
    assignment = await collaboration.create_assignment(
        actor=user,
        context=context,
        resource_type=payload.resource_type,
        resource_id=payload.resource_id,
        assignee_id=payload.assignee_id,
        due_on=payload.due_on,
        note=payload.note,
    )
    return _present_assignment(assignment, user.locale)


@router.patch(
    "/assignments/{assignment_id}",
    response_model=AssignmentResponse,
    summary="Accept, complete or decline an assignment",
)
async def set_assignment_status(
    assignment_id: int,
    payload: AssignmentStatusRequest,
    user: CurrentUser,
    collaboration: CollaborationDep,
) -> AssignmentResponse:
    assignment = await collaboration.set_assignment_status(
        assignment_id, payload.status, actor=user
    )
    return _present_assignment(assignment, user.locale)


def _present_reply(comment: Comment) -> CommentReplyResponse:
    return CommentReplyResponse(
        id=comment.id,
        parent_id=comment.parent_id,
        author_id=comment.author_id,
        author_name=comment.author.full_name if comment.author else None,
        author_initials=comment.author.initials if comment.author else None,
        body=comment.body,
        created_at=comment.created_at,
        resolved_at=comment.resolved_at,
        resolved_by_id=comment.resolved_by_id,
    )


def _present_thread(thread: CommentThread) -> CommentResponse:
    root = thread.root
    return CommentResponse(
        id=root.id,
        parent_id=root.parent_id,
        author_id=root.author_id,
        author_name=root.author.full_name if root.author else None,
        author_initials=root.author.initials if root.author else None,
        body=root.body,
        created_at=root.created_at,
        resolved_at=root.resolved_at,
        resolved_by_id=root.resolved_by_id,
        replies=[_present_reply(reply) for reply in thread.replies],
    )


def _present_assignment(
    assignment: ReviewAssignment, locale: Locale = DEFAULT_LOCALE
) -> AssignmentResponse:
    return AssignmentResponse(
        id=assignment.id,
        resource_type=assignment.resource_type,
        resource_id=assignment.resource_id,
        status=assignment.status,
        status_label=assignment_status_label(assignment.status, locale),
        due_on=assignment.due_on,
        note=assignment.note,
        assignee_id=assignment.assignee_id,
        assignee_name=assignment.assignee.full_name if assignment.assignee else None,
        assigned_by_id=assignment.assigned_by_id,
        assigned_by_name=(assignment.assigned_by.full_name if assignment.assigned_by else None),
        href=resource_href(assignment.resource_type, assignment.resource_id),
        created_at=assignment.created_at,
        completed_at=assignment.completed_at,
    )
