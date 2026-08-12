import hashlib
import json
from pathlib import Path

from arrestshield.verification import verify_json_value, verify_sha256


def test_sha_verifier_passes_and_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"trusted")
    expected = hashlib.sha256(b"trusted").hexdigest()
    passed = verify_sha256(tmp_path, "artifact.bin", expected, "artifact")
    assert passed.status == "passed"
    artifact.write_bytes(b"tampered")
    failed = verify_sha256(tmp_path, "artifact.bin", expected, "artifact")
    assert failed.status == "failed"
    assert failed.actual != failed.expected


def test_json_boundary_verifier_reports_missing_and_wrong_values(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"boundary": {"llm": False}}), encoding="utf-8")
    passed = verify_json_value(
        tmp_path, "policy.json", ("boundary", "llm"), False, "boundary"
    )
    assert passed.status == "passed"
    wrong = verify_json_value(
        tmp_path, "policy.json", ("boundary", "llm"), True, "boundary"
    )
    assert wrong.status == "failed"
    missing = verify_json_value(
        tmp_path, "policy.json", ("missing",), False, "boundary"
    )
    assert missing.status == "failed"
    assert "missing JSON path" in missing.detail
