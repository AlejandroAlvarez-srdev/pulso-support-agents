from __future__ import annotations

import sqlite3

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from pulso_support import fixtures as fx
from pulso_support.backend import FakePulsoBackend, PulsoBackendError
from pulso_support.config import Settings
from pulso_support.graph import RefundDecision, build_support_graph
from pulso_support.ledger import SqliteRefundLedger
from pulso_support.llm import ScriptedSupportModel
from pulso_support.resilience import ResilientExecutor
from pulso_support.service import SupportService
from tests.conftest import FIXED_NOW

LARGE = f"Full refund for order {fx.ORDER_DELIVERED_LARGE} please"


class SimulatedCrashError(RuntimeError):
    """Stands in for the worker process dying right after the refund is committed."""


def test_same_key_returns_the_original_refund(backend: FakePulsoBackend):
    first = backend.issue_refund(fx.ORDER_DELIVERED, 5_000, "late", idempotency_key="k1")
    again = backend.issue_refund(fx.ORDER_DELIVERED, 5_000, "late", idempotency_key="k1")

    assert again == first
    assert backend.refunded_cents(fx.ORDER_DELIVERED) == 5_000
    assert [e.type for e in backend.events] == ["refund.issued"]


def test_key_reuse_with_different_amount_is_rejected(backend: FakePulsoBackend):
    backend.issue_refund(fx.ORDER_DELIVERED, 5_000, "late", idempotency_key="k1")
    with pytest.raises(PulsoBackendError, match="different refund"):
        backend.issue_refund(fx.ORDER_DELIVERED, 6_000, "late", idempotency_key="k1")


def test_sqlite_ledger_deduplicates_across_backend_instances(tmp_path):
    db = str(tmp_path / "ledger.sqlite")
    process_a = FakePulsoBackend.with_fixtures(ledger=SqliteRefundLedger(db))
    process_b = FakePulsoBackend.with_fixtures(ledger=SqliteRefundLedger(db))

    a = process_a.issue_refund(fx.ORDER_DELIVERED, 5_000, "late", idempotency_key="k1")
    b = process_b.issue_refund(fx.ORDER_DELIVERED, 5_000, "late", idempotency_key="k1")

    assert a == b
    assert process_b.refunded_cents(fx.ORDER_DELIVERED) == 5_000


def test_lost_response_is_retried_without_paying_twice(make_service, backend):
    backend.inject_fault("issue_refund", "lost_response")
    service = make_service(executor=ResilientExecutor(sleep=lambda _: None))

    result = service.send("idem-1", f"Refund $50 on order {fx.ORDER_DELIVERED}")

    assert "has been issued" in result.reply
    assert backend.refunded_cents(fx.ORDER_DELIVERED) == 5_000
    assert [e.type for e in backend.events] == ["refund.issued"]


def test_model_cannot_supply_injected_arguments(make_service, backend):
    class ForgingModel(ScriptedSupportModel):
        def _act(self, messages):
            message = super()._act(messages)
            for call in message.tool_calls:
                if call["name"] == "issue_refund":
                    call["args"].update(approved_by="ceo", idempotency_key="forged")
            return AIMessage(content=message.content, tool_calls=message.tool_calls)

    make_service(model=ForgingModel()).send("idem-2", f"Refund $50 on {fx.ORDER_DELIVERED}")

    [refund] = backend.refunds_for(fx.ORDER_DELIVERED)
    assert refund.approved_by is None
    assert refund.idempotency_key.startswith("idem-2:")


def test_resume_after_crash_in_another_process_does_not_double_refund(tmp_path):
    checkpoints, ledger_db = tmp_path / "threads.sqlite", str(tmp_path / "ledger.sqlite")

    def process(crash_after_refund: bool) -> tuple[SupportService, FakePulsoBackend]:
        backend = FakePulsoBackend.with_fixtures(
            clock=lambda: FIXED_NOW, ledger=SqliteRefundLedger(ledger_db)
        )
        if crash_after_refund:
            original = backend._after

            def crash(method, fault):
                if method == "issue_refund":
                    raise SimulatedCrashError("worker killed")
                original(method, fault)

            backend._after = crash
        saver = SqliteSaver(sqlite3.connect(checkpoints, check_same_thread=False))
        graph = build_support_graph(ScriptedSupportModel(), backend, Settings(), saver)
        return SupportService(graph), backend

    service_a, _ = process(crash_after_refund=True)
    service_a.send("idem-3", LARGE)
    with pytest.raises(SimulatedCrashError):
        service_a.resume("idem-3", RefundDecision(approved=True, reviewer="ops"))

    service_b, backend_b = process(crash_after_refund=False)
    result = service_b.resume("idem-3", RefundDecision(approved=True, reviewer="ops"))

    assert result.status == "completed"
    assert "has been issued" in result.reply
    [refund] = backend_b.refunds_for(fx.ORDER_DELIVERED_LARGE)
    assert refund.amount_cents == 189_000
