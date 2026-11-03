from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pulso_support.api import create_app
from pulso_support.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_burst_then_refill_at_the_sustained_rate():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=6, burst=2, clock=clock)

    assert limiter.acquire("t") is None
    assert limiter.acquire("t") is None
    assert limiter.acquire("t") == pytest.approx(10.0)

    clock.now = 10.0
    assert limiter.acquire("t") is None
    assert limiter.acquire("t") == pytest.approx(10.0)


def test_buckets_are_per_thread():
    limiter = RateLimiter(per_minute=1, burst=1, clock=FakeClock())
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is not None
    assert limiter.acquire("b") is None


def test_idle_buckets_are_evicted_when_the_table_is_full():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=60, burst=1, clock=clock, max_keys=2)
    limiter.acquire("a")
    limiter.acquire("b")

    clock.now = 5.0
    limiter.acquire("c")

    assert set(limiter._buckets) == {"c"}


def test_api_returns_429_with_retry_after(make_service):
    clock = FakeClock()
    client = TestClient(create_app(make_service(), RateLimiter(2, burst=2, clock=clock)))

    def send(thread: str, path: str = "messages"):
        return client.post(f"/threads/{thread}/{path}", json={"message": "talk to a human"})

    assert send("rl-1").status_code == 200
    assert send("rl-1", "messages/stream").status_code == 200
    limited = send("rl-1")
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "30"
    assert send("rl-2").status_code == 200

    clock.now = 30.0
    assert send("rl-1").status_code == 200


def test_resume_is_not_rate_limited(make_service):
    client = TestClient(create_app(make_service(), RateLimiter(1, burst=1, clock=FakeClock())))
    client.post(
        "/threads/rl-3/messages",
        json={"message": "refund order 0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d04"},
    )
    assert client.post("/threads/rl-3/resume", json={"approved": True}).status_code == 200
