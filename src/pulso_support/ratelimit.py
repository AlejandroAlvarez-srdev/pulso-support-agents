"""Per-thread token-bucket rate limiting for the HTTP API.

Every user message costs a model call or several, so a runaway client (or a
retry loop) on one thread should not be able to burn the budget for everyone.
State is in process memory: with several API replicas each enforces its own
limit, so the effective limit is per replica. A shared store (Redis) would be
the next step for a horizontally scaled deployment.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(
        self,
        per_minute: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if per_minute <= 0 or burst <= 0:
            raise ValueError("per_minute and burst must be positive")
        self._rate = per_minute / 60.0
        self._burst = float(burst)
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float | None:
        """Take one token for `key`. Returns None if allowed, else seconds to wait."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._evict_idle(now)
                bucket = self._buckets[key] = _Bucket(self._burst, now)
            bucket.tokens = min(self._burst, bucket.tokens + (now - bucket.updated) * self._rate)
            bucket.updated = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return None
            return (1 - bucket.tokens) / self._rate

    def _evict_idle(self, now: float) -> None:
        # A bucket that would be full again carries no state worth keeping.
        full_after = self._burst / self._rate
        for key in [k for k, b in self._buckets.items() if now - b.updated >= full_after]:
            del self._buckets[key]
