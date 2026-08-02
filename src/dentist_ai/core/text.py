"""Small text helpers shared across services."""

from __future__ import annotations

from typing import Final

#: 11-14 take the "many" form despite ending in 1-4: 11 находок, not 11 находка.
_TEENS: Final[range] = range(11, 15)
_FEW: Final[range] = range(2, 5)


def plural_ru(count: int, one: str, few: str, many: str) -> str:
    """Pick the Russian plural form: 1 находка / 2 находки / 5 находок."""
    if abs(count) % 100 in _TEENS:
        return many
    remainder = abs(count) % 10
    if remainder == 1:
        return one
    if remainder in _FEW:
        return few
    return many


_UNSAFE_NAME_CHARS: Final[frozenset[str]] = frozenset('/\\<>:"|?*\x00')


def safe_display_name(filename: str | None, *, fallback: str) -> str:
    """Sanitise a client filename for display only.

    It never reaches the filesystem — storage paths derive from the content
    hash — but it is echoed into HTML and CSV, so path separators and control
    characters are stripped here rather than trusted downstream. Letters in
    any script survive: a clinic naming files "Иванов_ОПТГ.jpg" should see
    that name back.
    """
    if not filename:
        return fallback
    cleaned = "".join(
        char
        for char in filename
        if char not in _UNSAFE_NAME_CHARS and (char.isprintable() or char == " ")
    ).strip()
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned[:255] or fallback
