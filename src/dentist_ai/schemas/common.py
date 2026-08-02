"""Shared response envelopes."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiModel(BaseModel):
    """Base for every wire model: camelCase out, both accepted in."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        alias_generator=lambda name: "".join(
            part if index == 0 else part.capitalize() for index, part in enumerate(name.split("_"))
        ),
    )


class PageMeta(ApiModel):
    total: int = Field(description="Total rows matching the filter, ignoring pagination.")
    limit: int
    offset: int
    has_more: bool


class Page[T](ApiModel):
    items: list[T]
    meta: PageMeta


class OkResponse(ApiModel):
    ok: bool = True
