"""Eval harness: replay scripted conversations and score routing and tool use.

    python -m pulso_support.evals --cases evals/cases.yaml --min-pass-rate 0.9

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
from pulso_support.service import SupportService, TurnResult

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "evals" / "cases.yaml"


@dataclass
class CaseResult:
    id: str
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

    def to_json(self) -> dict[str, Any]:
        return {
            "total": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "pass_rate": round(self.pass_rate, 4),
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
    graph = build_support_graph(
        create_chat_model(settings),
        FakePulsoBackend.with_fixtures(),
        settings,
        checkpointer=InMemorySaver(),
    )
    service = SupportService(graph, settings)
    thread_id = f"eval-{case['id']}"
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
        return CaseResult(case["id"], False, {}, None, [], None, error=repr(exc))

    tools = [c["name"] for c in result.tool_calls]
    checks = {
        "route": result.route == expect["route"],
        "tools": tools == expect.get("tools", tools),
        "interrupt": interrupted == expect.get("interrupt", interrupted),
    }
    if "reply_contains" in expect:
        checks["reply"] = expect["reply_contains"].lower() in (result.reply or "").lower()
    return CaseResult(case["id"], all(checks.values()), checks, result.route, tools, result.reply)


def run_evals(cases: list[dict[str, Any]], settings: Settings) -> EvalReport:
    return EvalReport([run_case(case, settings) for case in cases])


def format_report(report: EvalReport, settings: Settings) -> str:
    width = max(len(r.id) for r in report.results)
    lines = [f"Pulso support evals - provider={settings.llm_provider}", ""]
    for r in report.results:
        failed = [name for name, ok in r.checks.items() if not ok]
        detail = r.error or (f"failed: {', '.join(failed)}" if failed else "")
        lines.append(f"{'PASS' if r.passed else 'FAIL'}  {r.id:<{width}}  {detail}".rstrip())
    passed = sum(r.passed for r in report.results)
    lines += ["", f"{passed}/{len(report.results)} passed ({report.pass_rate:.1%})"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--json", type=Path, help="write a machine-readable report here")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    report = run_evals(load_cases(args.cases), settings)
    print(format_report(report, settings))
    if args.json:
        args.json.write_text(json.dumps(report.to_json(), indent=2))
    return 0 if report.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
