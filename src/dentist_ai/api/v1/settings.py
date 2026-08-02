"""Account settings: profile, locale, password."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter
from pydantic import Field, field_validator, model_validator

from dentist_ai.api.deps import ApiRateLimit, AuditDep, AuthDep, CurrentUser, DbSession, RequestCtx
from dentist_ai.api.v1.auth import present_user
from dentist_ai.ml.taxonomy import SUPPORTED_LOCALES
from dentist_ai.schemas.auth import (
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    UserResponse,
)
from dentist_ai.schemas.common import ApiModel, OkResponse
from dentist_ai.services.audit import AuditAction

router = APIRouter(prefix="/settings", tags=["settings"], dependencies=[ApiRateLimit])


class ProfileUpdateRequest(ApiModel):
    full_name: Annotated[str, Field(min_length=2, max_length=160)]
    locale: str = "ru"

    @field_validator("full_name")
    @classmethod
    def _collapse(cls, value: str) -> str:
        collapsed = " ".join(value.split())
        if not collapsed:
            raise ValueError("Укажите имя")
        return collapsed

    @field_validator("locale")
    @classmethod
    def _supported(cls, value: str) -> str:
        if value not in SUPPORTED_LOCALES:
            raise ValueError(f"Поддерживаемые языки: {', '.join(SUPPORTED_LOCALES)}")
        return value


class PasswordChangeRequest(ApiModel):
    current_password: Annotated[str, Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)]
    new_password: Annotated[
        str, Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    ]
    new_password_confirm: str

    @model_validator(mode="after")
    def _match(self) -> PasswordChangeRequest:
        if self.new_password != self.new_password_confirm:
            raise ValueError("Пароли не совпадают")
        if self.new_password == self.current_password:
            raise ValueError("Новый пароль должен отличаться от текущего")
        return self


@router.put("/profile", response_model=UserResponse, summary="Update profile")
async def update_profile(
    payload: ProfileUpdateRequest,
    user: CurrentUser,
    audit: AuditDep,
    context: RequestCtx,
    session: DbSession,
) -> UserResponse:
    user.full_name = payload.full_name
    user.locale = payload.locale
    await session.flush()
    await audit.record(
        action=AuditAction.PROFILE_UPDATED,
        organization_id=user.organization_id,
        actor_id=user.id,
        resource_type="user",
        resource_id=user.id,
        context=context,
    )
    return present_user(user)


@router.put("/password", response_model=OkResponse, summary="Change password")
async def change_password(
    payload: PasswordChangeRequest,
    user: CurrentUser,
    auth: AuthDep,
    context: RequestCtx,
) -> OkResponse:
    await auth.change_password(user, payload.current_password, payload.new_password, context)
    return OkResponse()
