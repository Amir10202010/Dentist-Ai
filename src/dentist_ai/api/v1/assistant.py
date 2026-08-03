"""The case assistant."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Query

from dentist_ai.api.deps import ApiRateLimit, AssistantDep, CurrentUser, RequestCtx
from dentist_ai.db.models import AssistantMessage, AssistantThread
from dentist_ai.schemas.assistant import (
    AnswerResponse,
    AskRequest,
    AskResponse,
    AssistantMessageResponse,
    AssistantThreadResponse,
    AssistantThreadSummary,
    CitationResponse,
)

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/ask", response_model=AskResponse, dependencies=[ApiRateLimit])
async def ask(
    payload: AskRequest,
    user: CurrentUser,
    assistant: AssistantDep,
    context: RequestCtx,
) -> AskResponse:
    thread, answer = await assistant.ask(
        payload.question,
        actor=user,
        context=context,
        patient_id=payload.patient_id,
        volume_public_id=payload.volume_public_id,
        thread_public_id=payload.thread_public_id,
    )
    return AskResponse(
        thread_public_id=thread.public_id,
        answer=AnswerResponse(
            intent=answer.intent.value,
            body=answer.body,
            citations=[
                CitationResponse(kind=item.kind, label=item.label, href=item.href)
                for item in answer.citations
            ],
            suggestions=list(answer.suggestions),
        ),
    )


@router.get(
    "/threads",
    response_model=list[AssistantThreadSummary],
    dependencies=[ApiRateLimit],
)
async def list_threads(
    user: CurrentUser,
    assistant: AssistantDep,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[AssistantThreadSummary]:
    threads = await assistant.threads(actor=user, limit=limit)
    return [
        AssistantThreadSummary(
            public_id=item.public_id,
            title=item.title,
            patient_id=item.patient_id,
            created_at=item.created_at,
        )
        for item in threads
    ]


@router.get(
    "/threads/{public_id}",
    response_model=AssistantThreadResponse,
    dependencies=[ApiRateLimit],
)
async def get_thread(
    public_id: str,
    user: CurrentUser,
    assistant: AssistantDep,
) -> AssistantThreadResponse:
    thread, messages = await assistant.history(public_id, actor=user)
    return _present(thread, messages)


def _present(thread: AssistantThread, messages: list[AssistantMessage]) -> AssistantThreadResponse:
    return AssistantThreadResponse(
        public_id=thread.public_id,
        title=thread.title,
        patient_id=thread.patient_id,
        volume_public_id=thread.study_public_id,
        created_at=thread.created_at,
        messages=[
            AssistantMessageResponse(
                id=item.id,
                role=item.role.value,
                body=item.body,
                intent=item.intent,
                citations=_citations(item.citations),
                created_at=item.created_at,
            )
            for item in messages
        ],
    )


def _citations(raw: str) -> list[CitationResponse]:
    """Decode stored citations, tolerating a row written by another build."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [
        CitationResponse(
            kind=str(item.get("kind", "")),
            label=str(item.get("label", "")),
            href=item.get("href"),
        )
        for item in parsed
        if isinstance(item, dict)
    ]
