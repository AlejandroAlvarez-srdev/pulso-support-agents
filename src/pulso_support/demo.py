"""Scripted walkthrough of one support thread, runnable offline.

python -m pulso_support.demo
"""

from __future__ import annotations

from pulso_support import fixtures as fx
from pulso_support.config import Settings
from pulso_support.graph import RefundDecision
from pulso_support.service import SupportService, TurnResult

SCRIPT = [
    f"Hi, where is order {fx.ORDER_IN_PROGRESS}?",
    f"Also, my pulser is offline, please reassign {fx.ORDER_ASSIGNED_STUCK}",
    f"Order {fx.ORDER_DELIVERED_LARGE} arrived broken, I want a full refund",
]


def _print_turn(result: TurnResult) -> None:
    tools = ", ".join(c["name"] for c in result.tool_calls) or "-"
    print(f"  [route={result.route} tools={tools}]")
    if result.pending_approval:
        p = result.pending_approval
        print(f"  ⏸  refund of {p['amount_cents']} cents on {p['order_id']} needs approval")
    if result.reply:
        print(f"agent> {result.reply}")


def main() -> None:
    settings = Settings.from_env()
    service = SupportService.from_settings(settings)
    thread = "demo-merchant-farmacia"
    print(f"provider={settings.llm_provider} thread={thread}\n")
    for text in SCRIPT:
        print(f"user > {text}")
        result = service.send(thread, text)
        _print_turn(result)
        if result.status == "awaiting_approval":
            print("reviewer> approve (as ops-lead@pulso.mx)")
            _print_turn(
                service.resume(thread, RefundDecision(approved=True, reviewer="ops-lead@pulso.mx"))
            )
        print()


if __name__ == "__main__":
    main()
