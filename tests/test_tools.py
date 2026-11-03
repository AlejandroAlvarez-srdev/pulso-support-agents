from __future__ import annotations

from pulso_support import fixtures as fx
from pulso_support.backend import FakePulsoBackend
from pulso_support.tools import ISSUE_REFUND, build_tools


def test_tool_schemas_hide_injected_reviewer_argument(backend: FakePulsoBackend):
    schema = build_tools(backend)[ISSUE_REFUND].tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"order_id", "amount_cents", "reason"}


def test_get_assignment_includes_pulser_summary(backend: FakePulsoBackend):
    result = build_tools(backend)["get_assignment"].invoke({"order_id": fx.ORDER_IN_PROGRESS})
    assert result["pulser"]["name"] == "Ana Torres"
    assert result["etaMinutes"] == 18


def test_business_errors_are_returned_not_raised(backend: FakePulsoBackend):
    tools = build_tools(backend)
    assert "error" in tools["get_order"].invoke({"order_id": "nope"})
    refund = tools[ISSUE_REFUND].invoke(
        {"order_id": fx.ORDER_PAYMENT_PENDING, "amount_cents": 100, "reason": "x"}
    )
    assert "nothing has been charged" in refund["error"]


def test_injected_reviewer_is_recorded_on_refund(backend: FakePulsoBackend):
    tools = build_tools(backend)
    tools[ISSUE_REFUND].invoke(
        {
            "order_id": fx.ORDER_DELIVERED,
            "amount_cents": 1_000,
            "reason": "late",
            "approved_by": "ops@pulso.mx",
        }
    )
    assert backend.refunds_for(fx.ORDER_DELIVERED)[0].approved_by == "ops@pulso.mx"
    assert tools["get_order"].invoke({"order_id": fx.ORDER_DELIVERED})["refundedCents"] == 1_000
