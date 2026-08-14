"""Evaluate the packaged model on the fixed behavioral test cases."""

from __future__ import annotations

import json
from pathlib import Path

from detector import SimpleScamDetector


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> int:
    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    cases = json.loads((PROJECT_DIR / "data/behavioral_test_cases.json").read_text(encoding="utf-8"))
    detector = SimpleScamDetector(PROJECT_DIR / config["model_path"])
    rows = []
    for case in cases:
        result = detector.predict(case["text"])
        predicted = "scam" if result.is_scam else "not_scam"
        rows.append(
            {
                **case,
                "predicted": predicted,
                "score": result.scam_score,
                "threshold": result.threshold,
                "passed": predicted == case["expected"],
            }
        )
    passed = sum(row["passed"] for row in rows)
    report = {
        "test_cases": len(rows),
        "passed": passed,
        "accuracy": passed / len(rows),
        "results": rows,
        "note": "Project-authored behavioral checks; not an independent real-world benchmark.",
    }
    output = PROJECT_DIR / "reports/behavioral_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Behavioral accuracy: {passed}/{len(rows)} = {report['accuracy']:.1%}")
    for row in rows:
        marker = "PASS" if row["passed"] else "FAIL"
        print(f"{marker:4} {row['id']:14} expected={row['expected']:8} predicted={row['predicted']:8} score={row['score']:.4f}")
    print(f"Report: {output}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
