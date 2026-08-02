"""Registration, login and profile management."""

from __future__ import annotations

import re
import secrets
import unicodedata

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from dentist_ai.core.errors import (
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    PermissionDeniedError,
)
from dentist_ai.core.logging import get_logger
from dentist_ai.core.security import PasswordService
from dentist_ai.db.base import utcnow
from dentist_ai.db.models import Organization, User, UserRole
from dentist_ai.schemas.auth import LoginRequest, RegisterRequest
from dentist_ai.services.audit import AuditAction, AuditService, RequestContext

log = get_logger(__name__)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class AuthService:
    def __init__(
        self,
        session: AsyncSession,
        passwords: PasswordService,
        audit: AuditService,
    ) -> None:
        self._session = session
        self._passwords = passwords
        self._audit = audit

    async def register(self, payload: RegisterRequest, context: RequestContext) -> User:
        """Create an organisation and its first (owner) user."""
        existing = await self._session.scalar(
            select(User.id).where(func.lower(User.email) == payload.email)
        )
        if existing is not None:
            raise EmailAlreadyRegisteredError

        organization = Organization(
            name=payload.organization_name,
            slug=await self._unique_slug(payload.organization_name),
        )
        user = User(
            organization=organization,
            email=payload.email,
            full_name=payload.full_name,
            password_hash=self._passwords.hash(payload.password),
            role=UserRole.OWNER,
        )
        self._session.add(organization)
        self._session.add(user)

        try:
            await self._session.flush()
        except IntegrityError as exc:
            await self._session.rollback()
            # Two simultaneous registrations for the same address: the unique
            # index is the real arbiter, the SELECT above is only a fast path.
            # Any *other* constraint violation is a bug, and swallowing it as
            # "email taken" would make it invisible.
            if not _violates(exc, "uq_users_email"):
                raise
            raise EmailAlreadyRegisteredError from exc

        await self._audit.record(
            action=AuditAction.USER_REGISTERED,
            organization_id=organization.id,
            actor_id=user.id,
            resource_type="user",
            resource_id=user.id,
            context=context,
        )
        log.info("user_registered", user_id=user.id, organization_id=organization.id)
        return user

    async def authenticate(self, payload: LoginRequest, context: RequestContext) -> User:
        user = await self._session.scalar(
            select(User)
            .options(selectinload(User.organization))
            .where(func.lower(User.email) == payload.email)
        )

        # Always run verification, even when no user matched, so response time
        # does not distinguish "unknown email" from "wrong password".
        matched = self._passwords.verify(payload.password, user.password_hash if user else None)

        if user is None or not matched:
            if user is not None:
                await self._audit.record(
                    action=AuditAction.LOGIN_FAILED,
                    organization_id=user.organization_id,
                    actor_id=user.id,
                    resource_type="user",
                    resource_id=user.id,
                    context=context,
                )
            log.info("login_failed", email_hint=_mask_email(payload.email))
            raise InvalidCredentialsError

        if not user.is_active:
            raise PermissionDeniedError("Аккаунт отключён. Обратитесь к администратору клиники.")

        # Transparent upgrade when Argon2 cost parameters are raised.
        if self._passwords.needs_rehash(user.password_hash):
            user.password_hash = self._passwords.hash(payload.password)
            log.info("password_rehashed", user_id=user.id)

        user.last_login_at = utcnow()
        await self._audit.record(
            action=AuditAction.LOGIN_SUCCEEDED,
            organization_id=user.organization_id,
            actor_id=user.id,
            resource_type="user",
            resource_id=user.id,
            context=context,
        )
        log.info("login_succeeded", user_id=user.id)
        return user

    async def get_active_user(self, user_id: int) -> User | None:
        user = await self._session.scalar(
            select(User).options(selectinload(User.organization)).where(User.id == user_id)
        )
        return user if user is not None and user.is_active else None

    async def change_password(
        self, user: User, current_password: str, new_password: str, context: RequestContext
    ) -> None:
        if not self._passwords.verify(current_password, user.password_hash):
            raise InvalidCredentialsError("Текущий пароль указан неверно.")
        user.password_hash = self._passwords.hash(new_password)
        await self._audit.record(
            action=AuditAction.PASSWORD_CHANGED,
            organization_id=user.organization_id,
            actor_id=user.id,
            resource_type="user",
            resource_id=user.id,
            context=context,
        )

    async def _unique_slug(self, name: str) -> str:
        base = _slugify(name) or "clinic"
        candidate = base
        for _ in range(5):
            taken = await self._session.scalar(
                select(Organization.id).where(Organization.slug == candidate)
            )
            if taken is None:
                return candidate
            candidate = f"{base}-{secrets.token_hex(3)}"
        return f"{base}-{secrets.token_hex(6)}"


def _violates(error: IntegrityError, constraint_name: str) -> bool:
    """Whether an IntegrityError names a specific constraint.

    Postgres reports the constraint by name; SQLite reports ``table.column``.
    Checking both keeps the same service code correct on either backend.
    """
    message = str(error.orig).lower()
    return constraint_name.lower() in message or "users.email" in message


def _slugify(value: str) -> str:
    """ASCII slug, transliterating Cyrillic so clinic names produce usable URLs."""
    # Lower-case *first*: the translation table only has lower-case keys, so
    # translating first would let "Клиника" through untranslated, and the
    # ASCII encode below would then silently eat every capital letter.
    transliterated = value.lower().translate(_CYRILLIC_MAP)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _SLUG_STRIP.sub("-", ascii_only).strip("-")[:60]


def _mask_email(email: str) -> str:
    """``a***@example.com`` — enough to correlate logs, not enough to leak."""
    local, _, domain = email.partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


_CYRILLIC_MAP = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
        "ә": "a",
        "ғ": "g",
        "қ": "q",
        "ң": "n",
        "ө": "o",
        "ұ": "u",
        "ү": "u",
        "һ": "h",
        "і": "i",
    }
)
