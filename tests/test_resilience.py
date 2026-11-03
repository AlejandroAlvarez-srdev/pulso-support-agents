from __future__ import annotations

import pytest

from pulso_support import fixtures as fx
from pulso_support.backend import BackendUnavailableError, FakePulsoBackend, NotFoundError
from pulso_support.config import Settings
from pulso_support.resilience import BackendCallFailedError, ResilientExecutor, RetryPolicy
from pulso_support.tools import REASSIGN_ORDER, build_tools, default_policies

FAST = RetryPolicy(max_attempts=3, timeout_s=0.05, backoff_s=0.01)


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def executor(sleeps) -> ResilientExecutor:
    return ResilientExecutor(sleep=sleeps.append)


def test_transient_failures_are_retried_with_exponential_backoff(backend, executor, sleeps):
    backend.inject_fault("get_order", "unavailable", times=2)

    order = executor.call("get_order", lambda: backend.get_order(fx.ORDER_DELIVERED), FAST)

    assert order.id == fx.ORDER_DELIVERED
    assert sleeps == [0.01, 0.02]


def test_slow_calls_time_out_and_count_as_attempts(backend, executor):
    backend.inject_fault("get_order", "slow", times=3, delay_s=0.2)

    with pytest.raises(BackendCallFailedError, match=r"after 3 attempt.*timed out after 0\.05s"):
        executor.call("get_order", lambda: backend.get_order(fx.ORDER_DELIVERED), FAST)


def test_business_errors_are_not_retried(backend, executor, sleeps):
    with pytest.raises(NotFoundError):
        executor.call("get_order", lambda: backend.get_order("missing"), FAST)
    assert sleeps == []


def test_lost_response_still_applies_the_write(backend):
    backend.inject_fault("reassign_order", "lost_response")

    with pytest.raises(BackendUnavailableError):
        backend.reassign_order(fx.ORDER_ASSIGNED_STUCK, "offline")
    assert backend.get_assignment(fx.ORDER_ASSIGNED_STUCK).pulser_id == fx.PULSER_DIEGO


def test_non_idempotent_writes_get_a_single_attempt(backend, executor):
    tools = build_tools(backend, executor, default_policies(FAST))
    backend.inject_fault("reassign_order", "lost_response")

    result = tools[REASSIGN_ORDER].invoke({"order_id": fx.ORDER_ASSIGNED_STUCK, "reason": "x"})

    assert result["retryable"] is True
    assert "after 1 attempt" in result["error"]
    assert [e.type for e in backend.events] == ["pulser.assigned"]


def test_exhausted_retries_are_surfaced_to_the_agent(make_service, backend):
    backend.inject_fault("get_order", "unavailable", times=10)
    backend.inject_fault("get_assignment", "unavailable", times=10)
    service = make_service(
        settings=Settings(tool_timeout_s=0.5),
        executor=ResilientExecutor(sleep=lambda _: None),
    )

    result = service.send("flaky-1", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    assert result.status == "completed"
    assert "temporarily unavailable" in result.reply


def test_agent_recovers_when_a_retry_succeeds(make_service, backend):
    backend.inject_fault("get_order", "unavailable", times=2)
    service = make_service(executor=ResilientExecutor(sleep=lambda _: None))

    result = service.send("flaky-2", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    assert "IN_PROGRESS" in result.reply


def test_fake_backend_fault_queue_is_per_method():
    backend = FakePulsoBackend.with_fixtures()
    backend.inject_fault("get_assignment", "unavailable")
    assert backend.get_order(fx.ORDER_DELIVERED).id == fx.ORDER_DELIVERED
    with pytest.raises(BackendUnavailableError):
        backend.get_assignment(fx.ORDER_DELIVERED)
