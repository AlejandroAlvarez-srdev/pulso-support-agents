"""Eval harness: replay scripted conversations and score routing and tool use.

    python -m pulso_support.evals --cases evals/cases.yaml --min-pass-rate 0.9

Each case can check the route, the ordered tool calls, whether the run paused for
approval, reply text that must or must not appear, text that must never reach the
stored transcript (PII), and how much was actually refunded per order. Cases can
inject backend faults to test retries. Results are reported overall and per
category.

Runs against whatever model ``PULSO_LLM_PROVIDER`` selects, so the same cases
gate the scripted model in CI and measure a real LLM when a key is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from pulso_support.backend import FakePulsoBackend
from pulso_support.config import Settings
from pulso_support.graph import RefundDecision, build_support_graph
from pulso_support.llm import create_chat_model
from pulso_support.resilience import ResilientExecutor
from pulso_support.service import SupportService, TurnResult

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "cases.yaml"


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    checks: dict[str, bool]
    route: str | None
    tools: list[str]
    reply: str | None
    error: str | None = None


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    def by_category(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for r in self.results:
            entry = stats.setdefault(r.category, {"passed": 0, "total": 0})
            entry["total"] += 1
            entry["passed"] += r.passed
        for entry in stats.values():
            entry["pass_rate"] = round(entry["passed"] / entry["total"], 4)
        return dict(sorted(stats.items()))

    def to_json(self) -> dict[str, Any]:
        return {
            "total": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "pass_rate": round(self.pass_rate, 4),
            "categories": self.by_category(),
            "cases": [asdict(r) for r in self.results],
        }


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())
    cases = data["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case ids in eval file")
    return cases


def run_case(case: dict[str, Any], settings: Settings) -> CaseResult:
    backend = FakePulsoBackend.with_fixtures()
    for fault in case.get("faults", []):
        backend.inject_fault(fault["method"], fault["kind"], times=fault.get("times", 1))
    graph = build_support_graph(
        create_chat_model(settings),
        backend,
        settings,
        checkpointer=InMemorySaver(),
        executor=ResilientExecutor(sleep=lambda _: None),
    )
    service = SupportService(graph, settings)
    thread_id = f"eval-{case['id']}"
    category = case.get("category", "uncategorized")
    expect = case["expect"]
    try:
        result: TurnResult | None = None
        for text in case["turns"]:
            result = service.send(thread_id, text)
        assert result is not None
        interrupted = result.status == "awaiting_approval"
        if interrupted and "resume" in case:
            result = service.resume(thread_id, RefundDecision.model_validate(case["resume"]))
    except Exception as exc:  # an eval run must report, not crash, on model misbehaviour
        return CaseResult(case["id"], category, False, {}, None, [], None, error=repr(exc))

    tools = [c["name"] for c in result.tool_calls]
    reply = (result.reply or "").lower()
    checks = {
        "route": result.route == expect["route"],
        "tools": tools == expect.get("tools", tools),
        "interrupt": interrupted == expect.get("interrupt", interrupted),
    }
    if "reply_contains" in expect:
        checks["reply"] = expect["reply_contains"].lower() in reply
    if "reply_not_contains" in expect:
        checks["reply_clean"] = expect["reply_not_contains"].lower() not in reply
    if "transcript_not_contains" in expect:
        state = graph.get_state(service._config(thread_id)).values
        stored = " ".join(m.text for m in state["messages"]) + (state.get("summary") or "")
        checks["transcript_clean"] = expect["transcript_not_contains"] not in stored
    if "refunded" in expect:
        checks["refunded"] = all(
            backend.refunded_cents(order_id) == cents
            for order_id, cents in expect["refunded"].items()
        )
    return CaseResult(
        case["id"], category, all(checks.values()), checks, result.route, tools, result.reply
    )


def run_evals(cases: list[dict[str, Any]], settings: Settings) -> EvalReport:
    return EvalReport([run_case(case, settings) for case in cases])


def format_report(report: EvalReport, settings: Settings) -> str:
    width = max(len(r.id) for r in report.results)
    lines = [f"Pulso support evals - provider={settings.llm_provider}", ""]
    for r in report.results:
        failed = [name for name, ok in r.checks.items() if not ok]
        detail = r.error or (f"failed: {', '.join(failed)}" if failed else "")
        lines.append(f"{'PASS' if r.passed else 'FAIL'}  {r.id:<{width}}  {detail}".rstrip())
    lines += ["", "By category:"]
    for category, stats in report.by_category().items():
        lines.append(
            f"  {category:<14} {stats['passed']:>3}/{stats['total']:<3} {stats['pass_rate']:.0%}"
        )
    passed = sum(r.passed for r in report.results)
    lines += ["", f"{passed}/{len(report.results)} passed ({report.pass_rate:.1%})"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--json", type=Path, help="write a machine-readable report here")
    parser.add_argument("--category", action="append", help="only run these categories")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.get("category") in args.category]
    report = run_evals(cases, settings)
    print(format_report(report, settings))
    if args.json:
        args.json.write_text(json.dumps(report.to_json(), indent=2))
    return 0 if report.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
