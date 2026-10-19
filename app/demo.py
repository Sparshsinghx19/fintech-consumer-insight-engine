"""
Public demo mode: write protection and request rate limiting.

PUBLIC_DEMO_MODE turns a deployment into a read-only showcase of the
indexed corpus. Anonymous visitors can search and generate answers,
insights, and reports, but cannot change the shared documents or index.

Rate limiting uses in-memory sliding windows - one per browser session and
one per server process - to cap how many model calls the demo makes. They
hold only timestamps (no user content) and reset when the process restarts.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Iterable, Optional

from app.config import Settings


class DemoModeRestriction(PermissionError):
    """A write action was attempted while PUBLIC_DEMO_MODE is enabled."""


def require_write_access(settings: Settings, action: str) -> None:
    """Raise DemoModeRestriction if `action` would modify shared data in
    public demo mode."""
    if settings.public_demo_mode:
        raise DemoModeRestriction(
            f"{action} is disabled in this public demo. The document collection is read-only."
        )


class SlidingWindowLimiter:
    """Allows at most `max_requests` units per `window_seconds`. Thread-safe."""

    def __init__(self, max_requests: int, window_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0] >= self.window_seconds:
            self._events.popleft()

    def remaining(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return max(self.max_requests - len(self._events), 0)

    def retry_after(self, cost: int = 1) -> float:
        """Seconds until `cost` units would be available (0 if now)."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            excess = len(self._events) + cost - self.max_requests
            if excess <= 0:
                return 0.0
            if cost > self.max_requests:
                return float("inf")
            return max(self._events[excess - 1] + self.window_seconds - now, 0.0)

    def acquire(self, cost: int = 1) -> bool:
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._events) + cost > self.max_requests:
                return False
            self._events.extend([now] * cost)
            return True


def try_consume(limiters: Iterable[SlidingWindowLimiter], cost: int) -> tuple[bool, float]:
    """Take `cost` units from every limiter, or from none if any lacks
    capacity. Returns (allowed, seconds_until_retry)."""
    limiters = list(limiters)
    wait = max((limiter.retry_after(cost) for limiter in limiters), default=0.0)
    if wait > 0:
        return False, wait
    for limiter in limiters:
        if not limiter.acquire(cost):
            # Another session took the last units between the check and now.
            # Units already taken from earlier limiters stay counted, which
            # errs on the strict side.
            return False, max(limiter.retry_after(cost), 1.0)
    return True, 0.0


def format_wait(seconds: Optional[float]) -> str:
    if seconds is None or seconds == float("inf"):
        return "later"
    minutes = int(seconds // 60) + (1 if seconds % 60 else 0)
    return f"in about {minutes} minute{'s' if minutes != 1 else ''}" if minutes > 1 else "in about a minute"
