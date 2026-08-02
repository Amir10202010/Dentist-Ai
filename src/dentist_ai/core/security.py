"""Password hashing, signed sessions and CSRF tokens.

Argon2id for passwords, with transparent re-hashing when the cost parameters
are raised. Sessions are stateless signed cookies carrying an opaque session id
plus the user id; rotating the id on login defeats session fixation, and the
``issued_at`` stamp expires cookies independently of the browser. CSRF uses
signed double-submit, with the token bound to the session id so it cannot be
replayed in another session.
"""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from dentist_ai.core.config import Settings

_SESSION_SALT = "dentist-ai.session.v1"
_CSRF_SALT = "dentist-ai.csrf.v1"


@dataclass(frozen=True, slots=True)
class SessionData:
    """Everything the server trusts from a session cookie."""

    user_id: int
    session_id: str
    issued_at: datetime

    def as_dict(self) -> dict[str, str | int]:
        return {
            "uid": self.user_id,
            "sid": self.session_id,
            "iat": int(self.issued_at.timestamp()),
        }


class PasswordService:
    """Hash and verify passwords, upgrading cost parameters in place."""

    def __init__(self, settings: Settings) -> None:
        security = settings.security
        self._hasher = PasswordHasher(
            memory_cost=security.argon2_memory_kib,
            time_cost=security.argon2_time_cost,
            parallelism=security.argon2_parallelism,
        )
        # A genuine hash under the configured cost parameters. Verifying
        # against it for unknown accounts burns the same CPU as a real check,
        # which is the entire point — a hard-coded literal would fail to parse
        # and return early, restoring the timing oracle we are closing.
        self._dummy_hash = self._hasher.hash(secrets.token_urlsafe(32))

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password: str, password_hash: str | None) -> bool:
        """Return whether ``password`` matches, in constant-ish time.

        Passing ``None`` (no such user) still performs a full verification
        against a dummy hash so that response time does not reveal whether the
        account exists.
        """
        candidate = password_hash or self._dummy_hash
        try:
            self._hasher.verify(candidate, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
        return password_hash is not None

    def needs_rehash(self, password_hash: str) -> bool:
        try:
            return self._hasher.check_needs_rehash(password_hash)
        except InvalidHashError:
            return True


class SessionService:
    """Serialise and verify the session cookie."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._serializer = URLSafeTimedSerializer(
            settings.secret_key.get_secret_value(), salt=_SESSION_SALT
        )
        self._csrf_serializer = URLSafeTimedSerializer(
            settings.secret_key.get_secret_value(), salt=_CSRF_SALT
        )

    # -- session ----------------------------------------------------------
    def issue(self, user_id: int) -> tuple[str, SessionData]:
        """Mint a brand-new session. Always called on login, never reused."""
        data = SessionData(
            user_id=user_id,
            session_id=secrets.token_urlsafe(24),
            issued_at=datetime.now(UTC),
        )
        return self._serializer.dumps(data.as_dict()), data

    def load(self, raw_cookie: str | None) -> SessionData | None:
        if not raw_cookie:
            return None
        try:
            payload = self._serializer.loads(
                raw_cookie, max_age=self._settings.security.session_max_age_seconds
            )
        except (BadSignature, SignatureExpired):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return SessionData(
                user_id=int(payload["uid"]),
                session_id=str(payload["sid"]),
                issued_at=datetime.fromtimestamp(int(payload["iat"]), tz=UTC),
            )
        except (KeyError, TypeError, ValueError):
            return None

    # -- csrf -------------------------------------------------------------
    def issue_csrf(self, session_id: str) -> str:
        return self._csrf_serializer.dumps({"sid": session_id, "n": secrets.token_urlsafe(16)})

    def verify_csrf(self, token: str | None, session_id: str) -> bool:
        if not token:
            return False
        try:
            payload = self._csrf_serializer.loads(
                token, max_age=self._settings.security.session_max_age_seconds
            )
        except (BadSignature, SignatureExpired):
            return False
        return isinstance(payload, dict) and hmac.compare_digest(
            str(payload.get("sid", "")), session_id
        )


def anonymous_session_id(request_host: str) -> str:
    """Stable-enough id for CSRF on pre-login forms.

    Anonymous visitors have no session yet, but the login form still needs a
    token. Binding it to the host keeps the double-submit check meaningful
    without introducing server-side state.
    """
    return f"anon:{request_host}"
