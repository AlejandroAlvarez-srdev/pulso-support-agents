"""Score the deterministic guardrails against a labelled corpus.

    python -m pulso_support.guardrail_evals --strict

Injection screen: precision, recall and false-positive rate over messages
labelled ``attack`` or ``benign``. PII redaction: how many PII strings leak and
how many strings that must survive (order ids, amounts, dates) get mangled.

Attacks and PII marked ``known_miss`` are scored like everything else, so the
reported recall is the real one. The gates apply to the rest: a known miss that
starts passing is reported, and a new miss fails the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pulso_support.guardrails import detect_injection, redact_pii

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "evals" / "guardrails.yaml"


@dataclass
class InjectionScore:
    tp: list[str] = field(default_factory=list)
    fn: list[str] = field(default_factory=list)
    fp: list[str] = field(default_factory=list)
    tn: list[str] = field(default_factory=list)
    known_misses: set[str] = field(default_factory=set)

    @property
    def recall(self) -> float:
        attacks = len(self.tp) + len(self.fn)
        return len(self.tp) / attacks if attacks else 1.0

    @property
    def precision(self) -> float:
        flagged = len(self.tp) + len(self.fp)
        return len(self.tp) / flagged if flagged else 1.0

    @property
    def false_positive_rate(self) -> float:
        benign = len(self.fp) + len(self.tn)
        return len(self.fp) / benign if benign else 0.0

    @property
    def unexpected_misses(self) -> list[str]:
        return [i for i in self.fn if i not in self.known_misses]

    @property
    def fixed_known_misses(self) -> list[str]:
        return [i for i in self.tp if i in self.known_misses]


@dataclass
class PiiScore:
    leaked: dict[str, list[str]] = field(default_factory=dict)
    mangled: dict[str, list[str]] = field(default_factory=dict)
    leak_total: int = 0
    keep_total: int = 0
    known_misses: set[str] = field(default_factory=set)

    @property
    def recall(self) -> float:
        leaks = sum(len(v) for v in self.leaked.values())
        return 1 - leaks / self.leak_total if self.leak_total else 1.0

    @property
    def false_redaction_rate(self) -> float:
        mangled = sum(len(v) for v in self.mangled.values())
        return mangled / self.keep_total if self.keep_total else 0.0

    @property
    def unexpected_leaks(self) -> list[str]:
        return [i for i in self.leaked if i not in self.known_misses]


def load_corpus(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text())
    for section in ("injection", "pii"):
        ids = [c["id"] for c in data[section]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate ids in the {section} section")
    for case in data["injection"]:
        if case["label"] not in ("attack", "benign"):
            raise ValueError(f"{case['id']}: label must be attack or benign")
    return data


def score_injection(cases: list[dict[str, Any]]) -> InjectionScore:
    score = InjectionScore(known_misses={c["id"] for c in cases if c.get("known_miss")})
    for case in cases:
        flagged = detect_injection(case["text"]) is not None
        attack = case["label"] == "attack"
        bucket = {
            (True, True): score.tp,
            (True, False): score.fn,
            (False, True): score.fp,
            (False, False): score.tn,
        }[(attack, flagged)]
        bucket.append(case["id"])
    return score


def score_pii(cases: list[dict[str, Any]]) -> PiiScore:
    score = PiiScore(known_misses={c["id"] for c in cases if c.get("known_miss")})
    for case in cases:
        redacted = redact_pii(case["text"])
        leaks = [s for s in case.get("leaks", []) if s in redacted]
        mangled = [s for s in case.get("keep", []) if s not in redacted]
        score.leak_total += len(case.get("leaks", []))
        score.keep_total += len(case.get("keep", []))
        if leaks:
            score.leaked[case["id"]] = leaks
        if mangled:
            score.mangled[case["id"]] = mangled
    return score


def report_json(inj: InjectionScore, pii: PiiScore) -> dict[str, Any]:
    return {
        "injection": {
            "attacks": len(inj.tp) + len(inj.fn),
            "benign": len(inj.fp) + len(inj.tn),
            "recall": round(inj.recall, 4),
            "precision": round(inj.precision, 4),
            "false_positive_rate": round(inj.false_positive_rate, 4),
            "missed": inj.fn,
            "false_positives": inj.fp,
            "unexpected_misses": inj.unexpected_misses,
            "fixed_known_misses": inj.fixed_known_misses,
        },
        "pii": {
            "pii_strings": pii.leak_total,
            "keep_strings": pii.keep_total,
            "recall": round(pii.recall, 4),
            "false_redaction_rate": round(pii.false_redaction_rate, 4),
            "leaked": pii.leaked,
            "mangled": pii.mangled,
            "unexpected_leaks": pii.unexpected_leaks,
        },
    }


def format_report(inj: InjectionScore, pii: PiiScore) -> str:
    lines = [
        "Guardrail evals",
        "",
        f"Injection screen: {len(inj.tp) + len(inj.fn)} attacks, "
        f"{len(inj.fp) + len(inj.tn)} benign",
        f"  recall {inj.recall:.1%}  precision {inj.precision:.1%}  "
        f"false-positive rate {inj.false_positive_rate:.1%}",
    ]
    lines += [f"  MISS  {i}{'  (known)' if i in inj.known_misses else ''}" for i in inj.fn]
    lines += [f"  FP    {i}" for i in inj.fp]
    lines += [f"  FIXED {i}  (was a known miss)" for i in inj.fixed_known_misses]
    lines += [
        "",
        f"PII redaction: {pii.leak_total} PII strings, {pii.keep_total} strings to keep",
        f"  recall {pii.recall:.1%}  false-redaction rate {pii.false_redaction_rate:.1%}",
    ]
    for case_id, leaks in pii.leaked.items():
        known = "  (known)" if case_id in pii.known_misses else ""
        lines.append(f"  LEAK  {case_id}: {', '.join(leaks)}{known}")
    for case_id, mangled in pii.mangled.items():
        lines.append(f"  MANGLED {case_id}: {', '.join(mangled)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--json", type=Path, help="write a machine-readable report here")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on any miss, false positive, leak or mangled string not marked known_miss",
    )
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    inj = score_injection(corpus["injection"])
    pii = score_pii(corpus["pii"])
    print(format_report(inj, pii))
    if args.json:
        args.json.write_text(json.dumps(report_json(inj, pii), indent=2, ensure_ascii=False))
    if not args.strict:
        return 0
    failed = inj.unexpected_misses or inj.fp or pii.unexpected_leaks or pii.mangled
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
