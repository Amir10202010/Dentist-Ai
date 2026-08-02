"""Search payloads.

One row shape for four record kinds. A per-kind response model would be the
obvious alternative and it moves the wrong decision into the browser: what a
radiograph's title is, and which screen a hit links to, are answers this server
already holds beside the taxonomy and the route table. The client gets one list
component and no knowledge of what it is listing.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from dentist_ai.ml.taxonomy import Severity
from dentist_ai.schemas.common import ApiModel


class SearchKind(enum.StrEnum):
    PATIENT = "patient"
    STUDY = "study"
    VOLUME = "volume"
    CASE = "case"


class SearchHit(ApiModel):
    """One result row, rendered without knowing which table it came from."""

    kind: SearchKind
    #: Public id for imaging and cases, the chart id for a patient — a string
    #: either way, because the client routes on it rather than counts with it.
    id: str
    title: str
    subtitle: str
    href: str
    at: datetime
    #: Worst severity among the record's findings, where it has any.
    severity: Severity | None = None


class SearchGroup(ApiModel):
    kind: SearchKind
    #: Rows matching the filter, ignoring pagination.
    total: int
    items: list[SearchHit] = Field(default_factory=list)


class FacetCount(ApiModel):
    """One filter chip: what it is called and how many records it would keep."""

    key: str
    label: str
    count: int
    severity: Severity | None = None


class SearchFacets(ApiModel):
    findings: list[FacetCount] = Field(default_factory=list)
    severities: list[FacetCount] = Field(default_factory=list)


class SearchResults(ApiModel):
    query: str
    #: Sum of the per-kind totals, so the header can be written before the
    #: groups are walked.
    total: int
    groups: list[SearchGroup] = Field(default_factory=list)
    facets: SearchFacets
