"""Public identifiers for records that appear in URLs."""

from __future__ import annotations

import secrets
from typing import Final

#: Crockford-style alphabet: no I/L/O/U, so an id read aloud over the phone to
#: support cannot be mistranscribed.
_ALPHABET: Final[str] = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LENGTH: Final[int] = 16


def generate_public_id() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(_LENGTH))
