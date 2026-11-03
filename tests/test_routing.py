from __future__ import annotations

import pytest

from pulso_support import fixtures as fx
from pulso_support.graph import parse_route
from pulso_support.llm import classify_by_keywords


@pytest.mark.parametrize(
    ("message", "route"),
    [
        (f"Where is order {fx.ORDER_IN_PROGRESS}?", "order_status"),
        (f"What's the ETA for {fx.ORDER_IN_PROGRESS}", "order_status"),
        (f"I need a refund for {fx.ORDER_DELIVERED}, the tacos were cold", "refunds"),
        (f"Customer was charged twice on {fx.ORDER_DELIVERED}", "refunds"),
        (f"My pulser is offline, please reassign {fx.ORDER_ASSIGNED_STUCK}", "pulser_ops"),
        (f"The pulser never showed up for {fx.ORDER_ASSIGNED_STUCK}", "pulser_ops"),
        ("I want to talk to a human", "escalation"),
        (f"Something is wrong with order {fx.ORDER_DELIVERED}", "escalation"),
        ("Can you write me a poem about tacos?", "out_of_scope"),
    ],
)
def test_router_dispatches_to_the_right_agent(make_service, message, route):
    result = make_service().send("t-route", message)
    assert result.route == route


@pytest.mark.parametrize(
    ("raw", "route"),
    [
        ("refunds", "refunds"),
        ("  Pulser_Ops.\n", "pulser_ops"),
        ("`order_status`", "order_status"),
        ("The route is: refunds", "refunds"),
        ("I am not sure", "escalation"),
    ],
)
def test_router_output_parsing_is_tolerant_and_fails_closed(raw, route):
    assert parse_route(raw) == route


def test_specific_intents_win_over_generic_order_questions():
    assert classify_by_keywords("where is my refund for this order") == "refunds"
