"""Authentication payloads."""

from __future__ import annotations

from datetime import datetime

from pydantic import EmailStr, Field, field_validator, model_validator

from dentist_ai.db.models import UserRole
from dentist_ai.schemas.common import ApiModel

#: Long enough to resist offline cracking, short enough not to trip password
#: managers. NIST 800-63B: length over composition rules.
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 128

#: A password using fewer than this many distinct characters is a pattern
#: like 'aaaaaaaaaa' or '1212121212' regardless of its length.
MIN_DISTINCT_CHARACTERS = 4

_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "password123",
        "12345678",
        "123456789",
        "1234567890",
        "qwertyuiop",
        "qwerty123",
        "iloveyou",
        "admin123",
        "welcome1",
        "letmein1",
        "dentist123",
        "dentistai",
        "changeme1",
    }
)


class RegisterRequest(ApiModel):
    full_name: str = Field(min_length=2, max_length=160)
    email: EmailStr
    organization_name: str = Field(min_length=2, max_length=200)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    password_confirm: str

    @field_validator("full_name", "organization_name")
    @classmethod
    def _collapse_whitespace(cls, value: str) -> str:
        collapsed = " ".join(value.split())
        if not collapsed:
            raise ValueError("Поле не может быть пустым")
        return collapsed

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def _reject_weak(cls, value: str) -> str:
        if value.lower() in _COMMON_PASSWORDS:
            raise ValueError("Этот пароль слишком распространён")
        if len(set(value)) < MIN_DISTINCT_CHARACTERS:
            raise ValueError("Пароль слишком однообразный")
        return value

    @model_validator(mode="after")
    def _passwords_match(self) -> RegisterRequest:
        if self.password != self.password_confirm:
            raise ValueError("Пароли не совпадают")
        return self


class LoginRequest(ApiModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class OrganizationResponse(ApiModel):
    id: int
    name: str
    slug: str


class UserResponse(ApiModel):
    id: int
    email: str
    full_name: str
    initials: str
    role: UserRole
    locale: str
    last_login_at: datetime | None
    organization: OrganizationResponse


class SessionResponse(ApiModel):
    user: UserResponse
    csrf_token: str
