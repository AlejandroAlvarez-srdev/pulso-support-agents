from __future__ import annotations

from pulso_support.config import Settings
from pulso_support.evals import DEFAULT_CASES, load_cases, main, run_case, run_evals


def test_bundled_cases_all_pass_with_scripted_model():
    report = run_evals(load_cases(DEFAULT_CASES), Settings())
    failing = [r for r in report.results if not r.passed]
    assert failing == []
    assert report.pass_rate == 1.0


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
        "  - id: ok\n    turns: ['talk to a human']\n    expect: {route: escalation}\n"
        "  - id: bad\n    turns: ['talk to a human']\n    expect: {route: refunds}\n"
    )
    report_path = tmp_path / "report.json"

    assert main(["--cases", str(cases), "--min-pass-rate", "0.5", "--json", str(report_path)]) == 0
    assert main(["--cases", str(cases), "--min-pass-rate", "0.9"]) == 1
    assert "1/2 passed (50.0%)" in capsys.readouterr().out
    assert '"pass_rate": 0.5' in report_path.read_text()
