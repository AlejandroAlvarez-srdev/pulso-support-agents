from __future__ import annotations

import pytest

from pulso_support import fixtures as fx
from pulso_support.config import Settings
from pulso_support.graph import RefundDecision
from pulso_support.service import ApprovalPendingError, NoPendingApprovalError

LARGE = f"Full refund please for order {fx.ORDER_DELIVERED_LARGE}, items arrived broken"


def test_large_refund_pauses_before_any_money_moves(make_service, backend):
    result = make_service().send("hitl-1", LARGE)

    assert result.status == "awaiting_approval"
    assert result.reply is None
    assert result.pending_approval == {
        "type": "refund_approval",
        "tool_call_id": result.pending_approval["tool_call_id"],
        "order_id": fx.ORDER_DELIVERED_LARGE,
        "amount_cents": 189_000,
        "reason": LARGE,
        "threshold_cents": 50_000,
    }
    assert backend.refunds_for(fx.ORDER_DELIVERED_LARGE) == []
    assert [e.type for e in backend.events] == []


def test_approved_refund_executes_once_and_records_reviewer(make_service, backend):
    service = make_service()
    service.send("hitl-2", LARGE)

    result = service.resume("hitl-2", RefundDecision(approved=True, reviewer="lucia@pulso.mx"))

    assert result.status == "completed"
    assert "has been issued" in result.reply
    [refund] = backend.refunds_for(fx.ORDER_DELIVERED_LARGE)
    assert refund.approved_by == "lucia@pulso.mx"
    assert [e.type for e in backend.events] == ["refund.issued"]


def test_denied_refund_is_not_executed(make_service, backend):
    service = make_service()
    service.send("hitl-3", LARGE)

    result = service.resume("hitl-3", RefundDecision(approved=False, note="photos unclear"))

    assert result.status == "completed"
    assert "not approved" in result.reply
    assert backend.refunds_for(fx.ORDER_DELIVERED_LARGE) == []


def test_threshold_is_configurable(make_service, backend):
    service = make_service(settings=Settings(refund_approval_threshold_cents=200_000))
    result = service.send("hitl-4", LARGE)

    assert result.status == "completed"
    assert backend.refunded_cents(fx.ORDER_DELIVERED_LARGE) == 189_000


def test_paused_thread_rejects_new_messages_and_bad_resumes(make_service):
    service = make_service()
    with pytest.raises(NoPendingApprovalError):
        service.resume("hitl-5", RefundDecision(approved=True))

    service.send("hitl-5", LARGE)
    with pytest.raises(ApprovalPendingError):
        service.send("hitl-5", "hello?")
