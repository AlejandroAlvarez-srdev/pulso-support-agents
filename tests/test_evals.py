from __future__ import annotations

import json
import re

from pulso_support.config import Settings
from pulso_support.evals import DEFAULT_CASES, load_cases, main, run_case, run_evals


def test_bundled_cases_all_pass_with_scripted_model():
    report = run_evals(load_cases(DEFAULT_CASES), Settings())
    failing = [r for r in report.results if not r.passed]
    assert failing == []
    assert report.pass_rate == 1.0


def test_bundled_cases_cover_every_category():
    cases = load_cases(DEFAULT_CASES)
    assert len(cases) >= 30
    assert {c["category"] for c in cases} >= {
        "order_status",
        "refunds",
        "pulser_ops",
        "memory",
        "escalation",
        "scope",
        "privacy",
        "adversarial",
        "resilience",
    }


def test_side_effect_and_privacy_checks_catch_violations():
    case = {
        "id": "leaky",
        "category": "privacy",
        "turns": ["I'm ana@example.com, refund 50 MXN on 0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d01"],
        "expect": {
            "route": "refunds",
            "transcript_not_contains": "[email redacted]",
            "refunded": {"0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d01": 0},
        },
    }
    result = run_case(case, Settings())
    assert result.checks["transcript_clean"] is False
    assert result.checks["refunded"] is False


def test_faults_are_injected_per_case():
    case = {
        "id": "down",
        "turns": ["Where is order 0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d02?"],
        "faults": [{"method": "get_order", "kind": "unavailable", "times": 5}],
        "expect": {"route": "order_status", "reply_contains": "temporarily unavailable"},
    }
    assert run_case(case, Settings()).passed


def test_harness_flags_wrong_expectations():
    case = {
        "id": "wrong-route",
        "turns": ["I want to talk to a human"],
        "expect": {"route": "refunds", "tools": ["issue_refund"]},
    }
    result = run_case(case, Settings())
    assert not result.passed
    assert result.checks == {"route": False, "tools": False, "interrupt": True}


def test_cli_exit_code_respects_min_pass_rate(tmp_path, capsys):
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        "cases:\n"
        "  - id: ok\n    category: a\n    turns: ['talk to a human']\n"
        "    expect: {route: escalation}\n"
        "  - id: bad\n    category: b\n    turns: ['talk to a human']\n"
        "    expect: {route: refunds}\n"
    )
    report_path = tmp_path / "report.json"

    assert main(["--cases", str(cases), "--min-pass-rate", "0.5", "--json", str(report_path)]) == 0
    assert main(["--cases", str(cases), "--min-pass-rate", "0.9"]) == 1
    assert main(["--cases", str(cases), "--min-pass-rate", "1.0", "--category", "a"]) == 0
    out = capsys.readouterr().out
    assert "1/2 passed (50.0%)" in out
    assert re.search(r"^  b +0/1 +0%$", out, re.M)
    report = json.loads(report_path.read_text())
    assert report["pass_rate"] == 0.5
    assert report["categories"]["a"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
