from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from pulso_support import fixtures as fx
from pulso_support.graph import RefundDecision


def test_follow_up_turn_reuses_order_from_thread_memory(make_service, backend):
    service = make_service()
    service.send("mem-1", f"Where is order {fx.ORDER_DELIVERED}?")

    result = service.send("mem-1", "The tacos were cold, can I get a $40 refund?")

    assert result.route == "refunds"
    assert result.tool_calls[0]["args"] == {"order_id": fx.ORDER_DELIVERED}
    assert backend.refunded_cents(fx.ORDER_DELIVERED) == 4_000
    assert [h["role"] for h in service.history("mem-1")] == ["user", "assistant"] * 2


def test_threads_are_isolated(make_service):
    service = make_service()
    service.send("mem-a", f"Where is order {fx.ORDER_DELIVERED}?")

    result = service.send("mem-b", "can I get a refund?")

    assert result.tool_calls == []
    assert "order ID" in result.reply


def test_sqlite_checkpoint_survives_a_process_restart(make_service, tmp_path):
    db = tmp_path / "threads.sqlite"

    first = make_service(checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    first.send("mem-sql", f"Refund order {fx.ORDER_DELIVERED_LARGE}")
    assert first.pending_approval("mem-sql") is not None

    # A brand-new graph and connection, as after a deploy: the paused run is still there.
    second = make_service(checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    result = second.resume("mem-sql", RefundDecision(approved=True, reviewer="oncall"))

    assert result.status == "completed"
    assert "has been issued" in result.reply
    assert len(second.history("mem-sql")) == 2
