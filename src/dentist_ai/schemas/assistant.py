"""Assistant payloads.

``citations`` is not decoration. The assistant is only usable on patient data
because every answer names the rows it was built from, so the wire model
carries them beside the text and the client is expected to render them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from dentist_ai.schemas.common import ApiModel


class CitationResponse(ApiModel):
    kind: str
    label: str
    href: str | None = None


class AnswerResponse(ApiModel):
    #: Which question shape the router matched. Surfaced so the UI can style
    #: an unrecognised question differently from an answered one.
    intent: str
    body: str
    citations: list[CitationResponse]
    suggestions: list[str]


class AssistantMessageResponse(ApiModel):
    id: int
    role: str
    body: str
    intent: str | None
    citations: list[CitationResponse]
    created_at: datetime


class AssistantThreadResponse(ApiModel):
    public_id: str
    title: str
    patient_id: int | None
    volume_public_id: str | None
    created_at: datetime
    messages: list[AssistantMessageResponse]


class AssistantThreadSummary(ApiModel):
    public_id: str
    title: str
    patient_id: int | None
    created_at: datetime


class AskRequest(ApiModel):
    question: str = Field(min_length=1, max_length=1000)
    #: Continue an existing thread, or start one against a case.
    thread_public_id: str | None = Field(default=None, max_length=26)
    patient_id: int | None = None
    volume_public_id: str | None = Field(default=None, max_length=26)


class AskResponse(ApiModel):
    thread_public_id: str
    answer: AnswerResponse
