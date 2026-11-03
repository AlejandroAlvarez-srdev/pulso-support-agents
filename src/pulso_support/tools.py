"""LangChain tools the agents can call, bound to a Pulso backend instance.

Tools never raise on business errors: they return ``{"error": ...}`` so the
agent can explain the problem to the user instead of crashing the graph.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg, tool

from pulso_support.backend import FakePulsoBackend, PulsoBackendError

GET_ORDER = "get_order"
GET_ASSIGNMENT = "get_assignment"
ISSUE_REFUND = "issue_refund"
REASSIGN_ORDER = "reassign_order"


def build_tools(backend: FakePulsoBackend) -> dict[str, BaseTool]:
    @tool(GET_ORDER)
    def get_order(order_id: str) -> dict:
        """Look up a Pulso order: status, amount, items, pickup/dropoff and how much
        has already been refunded. `order_id` is the order UUID."""
        try:
            order = backend.get_order(order_id)
        except PulsoBackendError as exc:
            return {"error": str(exc)}
        return {**order.to_json(), "refundedCents": backend.refunded_cents(order_id)}

    @tool(GET_ASSIGNMENT)
    def get_assignment(order_id: str) -> dict:
        """Get the pulser currently assigned to an order, with their ETA in minutes."""
        try:
            assignment = backend.get_assignment(order_id)
        except PulsoBackendError as exc:
            return {"error": str(exc)}
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
    ) -> dict:
        """Refund part or all of a paid order to the merchant. `amount_cents` is in
        the order currency's minor units (MXN centavos). Large refunds are paused
        for human approval before they execute."""
        try:
            refund = backend.issue_refund(order_id, amount_cents, reason, approved_by=approved_by)
        except PulsoBackendError as exc:
            return {"error": str(exc)}
        return refund.to_json()

    @tool(REASSIGN_ORDER)
    def reassign_order(order_id: str, reason: str) -> dict:
        """Move an order that has not been picked up yet to a different available
        pulser. Use when the current pulser is offline, unresponsive or cannot make it."""
        try:
            assignment = backend.reassign_order(order_id, reason)
        except PulsoBackendError as exc:
            return {"error": str(exc)}
        pulser = backend.get_pulser(assignment.pulser_id)
        return {**assignment.to_json(), "pulser": {"name": pulser.name, "vehicle": pulser.vehicle}}

    return {t.name: t for t in (get_order, get_assignment, issue_refund, reassign_order)}
