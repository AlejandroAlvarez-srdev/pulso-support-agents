"""Retry and timeout policy for calls from tools to the Pulso backend.

Only transient failures are retried: timeouts and ``BackendUnavailableError``.
Business errors (``PulsoBackendError``) propagate on the first attempt because
retrying "order not found" cannot succeed.

A timed-out attempt keeps running in its worker thread; the next attempt does not
cancel it. That is why only idempotent operations get more than one attempt
(reads, and refunds once they carry an idempotency key).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import TypeVar

from pulso_support.backend import BackendUnavailableError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    timeout_s: float = 2.0
    backoff_s: float = 0.1
    backoff_multiplier: float = 2.0

    def delay(self, attempt: int) -> float:
        return self.backoff_s * self.backoff_multiplier ** (attempt - 1)


class BackendCallFailedError(Exception):
    def __init__(self, operation: str, attempts: int, last_error: str) -> None:
        super().__init__(f"{operation} failed after {attempts} attempt(s): {last_error}")
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error


class ResilientExecutor:
    def __init__(self, sleep: Callable[[float], None] = time.sleep, max_workers: int = 8) -> None:
        self._pool = ThreadPoolExecutor(max_workers, thread_name_prefix="pulso-backend")
        self._sleep = sleep

    def call(self, operation: str, fn: Callable[[], T], policy: RetryPolicy) -> T:
        last_error = ""
        for attempt in range(1, policy.max_attempts + 1):
            future = self._pool.submit(fn)
            try:
                return future.result(timeout=policy.timeout_s)
            except FutureTimeout:
                last_error = f"timed out after {policy.timeout_s:g}s"
            except BackendUnavailableError as exc:
                last_error = str(exc)
            if attempt < policy.max_attempts:
                self._sleep(policy.delay(attempt))
        raise BackendCallFailedError(operation, policy.max_attempts, last_error)
