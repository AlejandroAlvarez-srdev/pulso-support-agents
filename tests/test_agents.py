from __future__ import annotations

from pulso_support import fixtures as fx


def test_order_status_agent_calls_both_lookup_tools(make_service):
    result = make_service().send("t1", f"Where is my order {fx.ORDER_IN_PROGRESS}?")

    assert [c["name"] for c in result.tool_calls] == ["get_order", "get_assignment"]
    assert all(c["args"] == {"order_id": fx.ORDER_IN_PROGRESS} for c in result.tool_calls)
    assert "IN_PROGRESS" in result.reply
    assert "Ana Torres" in result.reply


def test_small_refund_executes_without_approval(make_service, backend):
    result = make_service().send("t2", f"Please refund $50 on order {fx.ORDER_DELIVERED}")

    assert result.status == "completed"
    assert [c["name"] for c in result.tool_calls] == ["get_order", "issue_refund"]
    assert result.tool_calls[1]["args"]["amount_cents"] == 5_000
    [refund] = backend.refunds_for(fx.ORDER_DELIVERED)
    assert refund.amount_cents == 5_000
    assert refund.approved_by is None


def test_refund_business_error_is_explained_to_the_user(make_service, backend):
    result = make_service().send("t3", f"refund order {fx.ORDER_PAYMENT_PENDING}")

    assert "couldn't issue the refund" in result.reply
    assert backend.refunds_for(fx.ORDER_PAYMENT_PENDING) == []


def test_pulser_ops_agent_reassigns_the_order(make_service, backend):
    result = make_service().send(
        "t4", f"Our pulser is offline, please reassign {fx.ORDER_ASSIGNED_STUCK}"
    )

    assert [c["name"] for c in result.tool_calls] == ["reassign_order"]
    assert backend.get_assignment(fx.ORDER_ASSIGNED_STUCK).pulser_id == fx.PULSER_DIEGO
    assert "Diego Herrera" in result.reply


def test_agent_asks_for_order_id_instead_of_guessing(make_service):
    result = make_service().send("t5", "where is my order?")

    assert result.tool_calls == []
    assert "order ID" in result.reply


def test_escalation_opens_a_ticket(make_service, backend):
    result = make_service().send("t6", "I want to speak to a human now")

    assert result.route == "escalation"
    assert result.ticket_id == "T-0001"
    assert backend.tickets["T-0001"]["threadId"] == "t6"
    assert "T-0001" in result.reply
