"""Fixed-window rate limiting.

The default backend counts in-process, which is only correct for a single
container. ``RateLimiter`` is a protocol, so swapping in Redis for a
multi-replica deployment touches one line in the composition root.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from dentist_ai.core.config import RateLimitRule

__all__ = ["InMemoryRateLimiter", "RateLimitDecision", "RateLimiter"]


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int


class RateLimiter(Protocol):
    async def check(self, key: str, rule: RateLimitRule) -> RateLimitDecision: ...

    async def reset(self, key: str) -> None: ...


@dataclass(slots=True)
class _Window:
    started_at: float
    count: int


class InMemoryRateLimiter:
    """Fixed-window counter guarded by a lock.

    Stale windows are swept opportunistically so a burst of unique keys (an
    attacker rotating IPs) cannot grow the dict without bound.
    """

    _SWEEP_EVERY_SECONDS = 60.0

    def __init__(self) -> None:
        self._windows: dict[str, _Window] = defaultdict(lambda: _Window(0.0, 0))
        self._lock = asyncio.Lock()
        self._last_sweep = time.monotonic()

    async def check(self, key: str, rule: RateLimitRule) -> RateLimitDecision:
        now = time.monotonic()
        async with self._lock:
            self._maybe_sweep(now, rule.window_seconds)
            window = self._windows[key]

            if now - window.started_at >= rule.window_seconds:
                window.started_at = now
                window.count = 0

            elapsed = now - window.started_at
            retry_after = max(1, int(rule.window_seconds - elapsed))

            if window.count >= rule.limit:
                return RateLimitDecision(False, 0, retry_after)

            window.count += 1
            return RateLimitDecision(True, rule.limit - window.count, retry_after)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._windows.pop(key, None)

    def _maybe_sweep(self, now: float, window_seconds: int) -> None:
        if now - self._last_sweep < self._SWEEP_EVERY_SECONDS:
            return
        self._last_sweep = now
        cutoff = now - window_seconds
        expired = [key for key, win in self._windows.items() if win.started_at < cutoff]
        for key in expired:
            del self._windows[key]
