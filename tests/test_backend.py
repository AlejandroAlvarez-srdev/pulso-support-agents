from __future__ import annotations

import pytest

from pulso_support import fixtures as fx
from pulso_support.backend import FakePulsoBackend, NotFoundError, PulsoBackendError
from pulso_support.domain import OrderStatus


def test_order_serialises_with_shared_contract_field_names(backend: FakePulsoBackend):
    payload = backend.get_order(fx.ORDER_DELIVERED).to_json()
    assert {"merchantId", "amountCents", "createdAt", "currency"} <= payload.keys()
    assert payload["currency"] == "MXN"


def test_unknown_order_raises_not_found(backend: FakePulsoBackend):
    with pytest.raises(NotFoundError):
        backend.get_order("does-not-exist")


def test_partial_refunds_cannot_exceed_order_amount(backend: FakePulsoBackend):
    backend.issue_refund(fx.ORDER_DELIVERED, 20_000, "missing item")
    with pytest.raises(PulsoBackendError, match="refundable balance of 14900"):
        backend.issue_refund(fx.ORDER_DELIVERED, 20_000, "again")
    assert backend.refunded_cents(fx.ORDER_DELIVERED) == 20_000
    assert [e.type for e in backend.events] == ["refund.issued"]


def test_refund_rejected_before_payment_is_captured(backend: FakePulsoBackend):
    with pytest.raises(PulsoBackendError, match="nothing has been charged"):
        backend.issue_refund(fx.ORDER_PAYMENT_PENDING, 100, "test")


def test_reassign_picks_best_online_pulser_with_capacity(backend: FakePulsoBackend):
    assignment = backend.reassign_order(fx.ORDER_ASSIGNED_STUCK, "pulser offline")

    # Luis (current, offline) and Mariana (full) are excluded; Diego is at the pickup.
    assert assignment.pulser_id == fx.PULSER_DIEGO
    assert backend.get_assignment(fx.ORDER_ASSIGNED_STUCK) == assignment
    assert backend.get_pulser(fx.PULSER_DIEGO).active_tasks == 1
    assert backend.get_pulser(fx.PULSER_LUIS).active_tasks == 0
    assert backend.events[-1].type == "pulser.assigned"


def test_in_progress_order_cannot_be_reassigned(backend: FakePulsoBackend):
    with pytest.raises(PulsoBackendError, match="can no longer be reassigned"):
        backend.reassign_order(fx.ORDER_IN_PROGRESS, "slow")
    assert backend.get_order(fx.ORDER_IN_PROGRESS).status is OrderStatus.IN_PROGRESS


def test_reassign_without_capacity_emits_assignment_failed():
    backend = FakePulsoBackend(
        fx.seed_orders(),
        [p for p in fx.seed_pulsers() if p.id == fx.PULSER_LUIS],
        fx.seed_assignments(),
    )
    with pytest.raises(PulsoBackendError, match="No pulser"):
        backend.reassign_order(fx.ORDER_ASSIGNED_STUCK, "offline")
    assert backend.events[-1].type == "assignment.failed"
