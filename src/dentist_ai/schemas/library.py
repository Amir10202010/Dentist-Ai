"""Case-library payloads.

Tags and finding keys are lists on the wire and one delimited string in the
column. The denormalisation is deliberate — see :class:`CaseEntry` — and the
translation between the two forms is confined to this module and
:mod:`dentist_ai.services.library`, so nothing else has to know that a tag may
not contain a comma.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Annotated

from pydantic import Field, field_validator

from dentist_ai.ml.taxonomy import Severity
from dentist_ai.schemas.common import ApiModel

#: Item caps chosen so a payload at its limit still fits ``case_entries.tags``
#: (255 chars) and ``finding_keys`` (512). Rejecting the overflow at the edge is
#: better than truncating it in the service, where the caller never learns that
#: half of what they sent was dropped.
MAX_TAGS = 16
MAX_TAG_LENGTH = 40
MAX_FINDING_KEYS = 32
MAX_FINDING_KEY_LENGTH = 64
#: Long enough for a case write-up, short enough that one row cannot become a
#: file store.
MAX_PROSE_LENGTH = 4000

Tag = Annotated[str, Field(min_length=1, max_length=MAX_TAG_LENGTH)]
FindingKey = Annotated[str, Field(min_length=1, max_length=MAX_FINDING_KEY_LENGTH)]


def _normalise(values: Iterable[str]) -> list[str]:
    """Fold, collapse whitespace, drop commas, de-duplicate, keep the order.

    The comma is load-bearing: it is the delimiter of the stored column, so a
    tag containing one would come back as two tags that match nothing. Folding
    matters for the same reason the filter does an exact match — "Имплантация"
    and "имплантация" arriving together would split one chip in two.
    """
    seen: dict[str, None] = {}
    for value in values:
        cleaned = " ".join(value.replace(",", " ").split()).lower()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


class CaseWriteRequest(ApiModel):
    title: str = Field(min_length=2, max_length=200)
    summary: str = Field(default="", max_length=MAX_PROSE_LENGTH)
    diagnosis: str = Field(default="", max_length=MAX_PROSE_LENGTH)
    treatment: str = Field(default="", max_length=MAX_PROSE_LENGTH)
    outcome: str = Field(default="", max_length=MAX_PROSE_LENGTH)
    finding_keys: list[FindingKey] = Field(default_factory=list, max_length=MAX_FINDING_KEYS)
    tags: list[Tag] = Field(default_factory=list, max_length=MAX_TAGS)
    #: Whose case this was. Cleared rather than cascaded if the chart is
    #: deleted, so the teaching material survives the patient record.
    patient_id: int | None = None

    @field_validator("tags", "finding_keys")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        return _normalise(value)


class CaseCreateRequest(CaseWriteRequest):
    """Optionally seeded from a record the clinic already has.

    Naming a study or a volume copies its diagnosis text and finding classes
    onto the new entry, which is what makes publishing a teaching case one
    click. Anything sent explicitly wins over what would have been copied.
    """

    from_study_public_id: str | None = Field(default=None, max_length=26)
    from_volume_public_id: str | None = Field(default=None, max_length=26)


class CaseUpdateRequest(CaseWriteRequest):
    """PUT semantics: an omitted field is cleared, not retained.

    No seeding here. Re-reading the source record on every edit is exactly the
    behaviour :class:`CaseEntry` exists to prevent.
    """


class CaseFinding(ApiModel):
    key: str
    label: str
    severity: Severity


class CaseListItemResponse(ApiModel):
    public_id: str
    title: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    findings: list[CaseFinding] = Field(default_factory=list)
    created_at: datetime
    created_by_name: str | None
    href: str


class CaseResponse(CaseListItemResponse):
    diagnosis: str
    treatment: str
    outcome: str
    patient_id: int | None
    #: Provenance, not a join. Nothing on the read path dereferences these: an
    #: entry outlives the study it was written from, and is meant to.
    study_public_id: str | None
    volume_public_id: str | None
    updated_at: datetime


class CaseTagResponse(ApiModel):
    tag: str
    count: int
