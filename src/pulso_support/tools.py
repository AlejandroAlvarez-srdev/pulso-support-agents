"""LangChain tools the agents can call, bound to a Pulso backend instance.

Tools never raise: business errors and exhausted retries come back as
``{"error": ...}`` so the agent can explain the problem to the user instead of
crashing the graph. ``retryable: true`` marks infrastructure failures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any

from langchain_core.tools import BaseTool, InjectedToolArg, tool

from pulso_support.backend import FakePulsoBackend, PulsoBackendError
from pulso_support.resilience import BackendCallFailedError, ResilientExecutor, RetryPolicy

GET_ORDER = "get_order"
GET_ASSIGNMENT = "get_assignment"
ISSUE_REFUND = "issue_refund"
REASSIGN_ORDER = "reassign_order"

# Arguments the graph fills in itself; never accepted from the model.
INJECTED_ARGS = frozenset({"approved_by", "idempotency_key"})


def default_policies(base: RetryPolicy) -> dict[str, RetryPolicy]:
    """Per-tool policies. Non-idempotent writes get exactly one attempt: retrying
    after a lost response would reassign twice."""
    return {
        GET_ORDER: base,
        GET_ASSIGNMENT: base,
        # Safe to retry: the graph always sends an idempotency key.
        ISSUE_REFUND: base,
        REASSIGN_ORDER: replace(base, max_attempts=1),
    }


def build_tools(
    backend: FakePulsoBackend,
    executor: ResilientExecutor | None = None,
    policies: dict[str, RetryPolicy] | None = None,
) -> dict[str, BaseTool]:
    executor = executor or ResilientExecutor()
    policies = policies or default_policies(RetryPolicy())

    def guarded(name: str, fn: Callable[[], Any]) -> tuple[Any, dict | None]:
        try:
            return executor.call(name, fn, policies[name]), None
        except PulsoBackendError as exc:
            return None, {"error": str(exc)}
        except BackendCallFailedError as exc:
            return None, {"error": str(exc), "retryable": True}

    @tool(GET_ORDER)
    def get_order(order_id: str) -> dict:
        """Look up a Pulso order: status, amount, items, pickup/dropoff and how much
        has already been refunded. `order_id` is the order UUID."""
        order, error = guarded(GET_ORDER, lambda: backend.get_order(order_id))
        if error:
            return error
        return {**order.to_json(), "refundedCents": backend.refunded_cents(order_id)}

    @tool(GET_ASSIGNMENT)
    def get_assignment(order_id: str) -> dict:
        """Get the pulser currently assigned to an order, with their ETA in minutes."""
        assignment, error = guarded(GET_ASSIGNMENT, lambda: backend.get_assignment(order_id))
        if error:
            return error
        if assignment is None:
            return {"orderId": order_id, "assignment": None}
        pulser = backend.get_pulser(assignment.pulser_id)
        return {
            **assignment.to_json(),
            "pulser": {
                "name": pulser.name,
                "vehicle": pulser.vehicle,
                "rating": pulser.rating,
                "online": pulser.online,
            },
        }

    @tool(ISSUE_REFUND)
    def issue_refund(
        order_id: str,
        amount_cents: int,
        reason: str,
        approved_by: Annotated[str | None, InjectedToolArg] = None,
        idempotency_key: Annotated[str | None, InjectedToolArg] = None,
    ) -> dict:
        """Refund part or all of a paid order to the merchant. `amount_cents` is in
        the order currency's minor units (MXN centavos). Large refunds are paused
        for human approval before they execute."""
        refund, error = guarded(
            ISSUE_REFUND,
            lambda: backend.issue_refund(
                order_id,
                amount_cents,
                reason,
                approved_by=approved_by,
                idempotency_key=idempotency_key,
            ),
        )
        return error or refund.to_json()

    @tool(REASSIGN_ORDER)
    def reassign_order(order_id: str, reason: str) -> dict:
        """Move an order that has not been picked up yet to a different available
        pulser. Use when the current pulser is offline, unresponsive or cannot make it."""
        assignment, error = guarded(
            REASSIGN_ORDER, lambda: backend.reassign_order(order_id, reason)
        )
        if error:
            return error
        pulser = backend.get_pulser(assignment.pulser_id)
        return {**assignment.to_json(), "pulser": {"name": pulser.name, "vehicle": pulser.vehicle}}

    return {t.name: t for t in (get_order, get_assignment, issue_refund, reassign_order)}
