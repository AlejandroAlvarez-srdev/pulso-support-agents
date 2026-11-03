"""In-process fake of the Pulso backend.

The agents never touch this directly: they go through the LangChain tools in
``pulso_support.tools``. Keeping the backend in-process makes the whole system
run offline and lets tests assert on side effects (refunds, events, tickets).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Literal

from pulso_support import fixtures
from pulso_support.domain import (
    REASSIGNABLE_STATUSES,
    REFUNDABLE_STATUSES,
    Assignment,
    Event,
    Order,
    OrderStatus,
    Pulser,
    Refund,
    Vehicle,
)
from pulso_support.ledger import InMemoryRefundLedger, RefundLedger

# Average urban speed per vehicle, used for ETA estimates.
_SPEED_KMH = {Vehicle.BIKE: 14.0, Vehicle.MOTO: 24.0, Vehicle.CAR: 18.0}
_PICKUP_BUFFER_MIN = 5


class PulsoBackendError(Exception):
    """A business rule rejected the request; the message is safe to show a user."""


class NotFoundError(PulsoBackendError):
    pass


class BackendUnavailableError(Exception):
    """Transient infrastructure failure (network, 5xx). Safe to retry; not a business error."""


FaultKind = Literal["unavailable", "slow", "lost_response"]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class FakePulsoBackend:
    def __init__(
        self,
        orders: Iterable[Order],
        pulsers: Iterable[Pulser],
        assignments: Iterable[Assignment] = (),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        ledger: RefundLedger | None = None,
    ) -> None:
        self._orders = {o.id: o for o in orders}
        self._pulsers = {p.id: p for p in pulsers}
        self._assignments = {a.order_id: a for a in assignments}
        self._ledger = ledger or InMemoryRefundLedger()
        self._tickets: dict[str, dict] = {}
        self._clock = clock
        self._lock = threading.Lock()
        self._faults: dict[str, deque[tuple[FaultKind, float]]] = {}
        self.events: list[Event] = []

    @classmethod
    def with_fixtures(
        cls,
        clock: Callable[[], datetime] | None = None,
        ledger: RefundLedger | None = None,
    ) -> FakePulsoBackend:
        kwargs = {"clock": clock} if clock else {}
        return cls(
            fixtures.seed_orders(),
            fixtures.seed_pulsers(),
            fixtures.seed_assignments(),
            ledger=ledger,
            **kwargs,
        )

    # --- fault injection ---------------------------------------------------

    def inject_fault(
        self, method: str, kind: FaultKind, times: int = 1, delay_s: float = 0.0
    ) -> None:
        """Make the next `times` calls to `method` misbehave like a flaky network.

        - unavailable: fail before doing anything.
        - slow: sleep `delay_s`, then succeed (the caller may already have timed out).
        - lost_response: perform the operation, then fail as if the reply was lost.
        """
        queue = self._faults.setdefault(method, deque())
        queue.extend([(kind, delay_s)] * times)

    def _next_fault(self, method: str) -> tuple[FaultKind, float] | None:
        queue = self._faults.get(method)
        return queue.popleft() if queue else None

    def _before(self, method: str) -> FaultKind | None:
        fault = self._next_fault(method)
        if fault is None:
            return None
        kind, delay = fault
        if kind == "unavailable":
            raise BackendUnavailableError(f"{method}: Pulso backend unavailable (503)")
        if kind == "slow":
            time.sleep(delay)
        return kind

    @staticmethod
    def _after(method: str, fault: FaultKind | None) -> None:
        if fault == "lost_response":
            raise BackendUnavailableError(f"{method}: connection reset before the response")

    # --- reads -----------------------------------------------------------

    def get_order(self, order_id: str) -> Order:
        fault = self._before("get_order")
        order = self._get_order(order_id)
        self._after("get_order", fault)
        return order

    def _get_order(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError:
            raise NotFoundError(f"Order {order_id} does not exist.") from None

    def get_pulser(self, pulser_id: str) -> Pulser:
        try:
            return self._pulsers[pulser_id]
        except KeyError:
            raise NotFoundError(f"Pulser {pulser_id} does not exist.") from None

    def get_assignment(self, order_id: str) -> Assignment | None:
        fault = self._before("get_assignment")
        self._get_order(order_id)
        assignment = self._assignments.get(order_id)
        self._after("get_assignment", fault)
        return assignment

    def refunded_cents(self, order_id: str) -> int:
        return sum(r.amount_cents for r in self._ledger.for_order(order_id))

    def refunds_for(self, order_id: str) -> list[Refund]:
        return self._ledger.for_order(order_id)

    @property
    def tickets(self) -> dict[str, dict]:
        return dict(self._tickets)

    # --- writes ----------------------------------------------------------

    def issue_refund(
        self,
        order_id: str,
        amount_cents: int,
        reason: str,
        approved_by: str | None = None,
        idempotency_key: str | None = None,
    ) -> Refund:
        """Refund an order. Repeating a call with the same `idempotency_key` returns
        the original refund instead of paying again."""
        fault = self._before("issue_refund")
        with self._lock, self._ledger.transaction():
            existing = self._ledger.by_key(idempotency_key) if idempotency_key else None
            if existing is not None:
                if (existing.order_id, existing.amount_cents) != (order_id, amount_cents):
                    raise PulsoBackendError(
                        "Idempotency key was already used for a different refund."
                    )
                refund = existing
            else:
                refund = self._new_refund(
                    order_id, amount_cents, reason, approved_by, idempotency_key
                )
        self._after("issue_refund", fault)
        return refund

    def _new_refund(
        self,
        order_id: str,
        amount_cents: int,
        reason: str,
        approved_by: str | None,
        idempotency_key: str | None,
    ) -> Refund:
        order = self._get_order(order_id)
        if order.status not in REFUNDABLE_STATUSES:
            raise PulsoBackendError(
                f"Order {order_id} is {order.status}; nothing has been charged yet."
            )
        if amount_cents <= 0:
            raise PulsoBackendError("Refund amount must be positive.")
        remaining = order.amount_cents - self.refunded_cents(order_id)
        if amount_cents > remaining:
            raise PulsoBackendError(
                f"Refund of {amount_cents} cents exceeds the refundable balance "
                f"of {remaining} cents."
            )
        refund = Refund(
            id=self._ledger.next_id(),
            order_id=order_id,
            amount_cents=amount_cents,
            currency=order.currency,
            reason=reason,
            approved_by=approved_by,
            idempotency_key=idempotency_key,
            issued_at=self._clock(),
        )
        self._ledger.add(refund)
        self._emit("refund.issued", refund.to_json())
        return refund

    def reassign_order(self, order_id: str, reason: str) -> Assignment:
        """Move an order to the best available pulser other than the current one."""
        fault = self._before("reassign_order")
        with self._lock:
            order = self._get_order(order_id)
            if order.status not in REASSIGNABLE_STATUSES:
                raise PulsoBackendError(
                    f"Order {order_id} is {order.status} and can no longer be reassigned."
                )
            current = self._assignments.get(order_id)
            candidates = [
                p
                for p in self._pulsers.values()
                if p.has_capacity and (current is None or p.id != current.pulser_id)
            ]
            if not candidates:
                self._emit("assignment.failed", {"orderId": order_id, "reason": "no_capacity"})
                raise PulsoBackendError("No pulser with free capacity is online right now.")

            best, score, eta = max(
                (self._score(order, p) for p in candidates), key=lambda t: (t[1], t[0].rating)
            )
            if current is not None:
                previous = self._pulsers[current.pulser_id]
                self._pulsers[previous.id] = previous.model_copy(
                    update={"active_tasks": max(previous.active_tasks - 1, 0)}
                )
            self._pulsers[best.id] = best.model_copy(update={"active_tasks": best.active_tasks + 1})
            assignment = Assignment(
                order_id=order_id,
                pulser_id=best.id,
                score=round(score, 3),
                eta_minutes=eta,
                assigned_at=self._clock(),
            )
            self._assignments[order_id] = assignment
            self._orders[order_id] = order.model_copy(update={"status": OrderStatus.ASSIGNED})
            self._emit("pulser.assigned", {**assignment.to_json(), "reason": reason})
        self._after("reassign_order", fault)
        return assignment

    def open_ticket(self, thread_id: str, summary: str) -> str:
        with self._lock:
            ticket_id = f"T-{len(self._tickets) + 1:04d}"
            self._tickets[ticket_id] = {
                "threadId": thread_id,
                "summary": summary,
                "openedAt": self._clock().isoformat(),
            }
            return ticket_id

    # --- internals -------------------------------------------------------

    @staticmethod
    def _score(order: Order, pulser: Pulser) -> tuple[Pulser, float, int]:
        """Blend proximity, rating and spare capacity into a 0-1 score."""
        km = haversine_km(
            pulser.location.lat, pulser.location.lng, order.pickup.lat, order.pickup.lng
        )
        proximity = 1 - min(km, 10.0) / 10.0
        quality = pulser.rating / 5
        spare = (pulser.max_tasks - pulser.active_tasks) / pulser.max_tasks
        score = 0.45 * proximity + 0.4 * quality + 0.15 * spare
        eta = math.ceil(km / _SPEED_KMH[pulser.vehicle] * 60) + _PICKUP_BUFFER_MIN
        return pulser, score, eta

    def _emit(self, event_type: str, payload: dict) -> None:
        self.events.append(Event(type=event_type, payload=payload, occurred_at=self._clock()))
