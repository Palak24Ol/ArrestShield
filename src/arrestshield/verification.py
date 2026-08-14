"""Authoritative checksum and boundary verification for local ML artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import sha256_file


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    status: str
    path: str
    expected: Any = None
    actual: Any = None
    detail: str | None = None


def read_json(path: Path) -> dict[str, Any]:
    # utf-8-sig accepts both plain UTF-8 and the BOM emitted by some Windows tools.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_sha256(
    root: Path,
    relative_path: str,
    expected: str,
    name: str,
) -> VerificationCheck:
    path = root / relative_path
    if not path.is_file():
        return VerificationCheck(name, "failed", relative_path, expected, None, "file missing")
    actual = sha256_file(path)
    return VerificationCheck(
        name,
        "passed" if actual == expected else "failed",
        relative_path,
        expected,
        actual,
    )


def verify_json_value(
    root: Path,
    relative_path: str,
    key_path: tuple[str, ...],
    expected: Any,
    name: str,
) -> VerificationCheck:
    path = root / relative_path
    if not path.is_file():
        return VerificationCheck(name, "failed", relative_path, expected, None, "file missing")
    value: Any = read_json(path)
    try:
        for key in key_path:
            value = value[key]
    except (KeyError, TypeError):
        return VerificationCheck(
            name,
            "failed",
            relative_path,
            expected,
            None,
            f"missing JSON path {'.'.join(key_path)}",
        )
    return VerificationCheck(
        name,
        "passed" if value == expected else "failed",
        relative_path,
        expected,
        value,
    )


def verify_local_artifacts(
    root: Path,
    require_multitask: bool = True,
    *,
    require_transformer: bool | None = None,
) -> dict[str, Any]:
    # Keep the old keyword as a compatibility alias for callers created before the
    # laptop deployment switched from a transformer export to compact XGBoost heads.
    if require_transformer is not None:
        require_multitask = require_transformer
    root = root.resolve()
    checks: list[VerificationCheck] = []

    processed_report_path = root / "docs/datasets/PROCESSED_DATA_VALIDATION.json"
    if processed_report_path.is_file():
        processed = read_json(processed_report_path)
        for relative_path, metadata in processed.get("artifacts", {}).items():
            checks.append(
                verify_sha256(
                    root,
                    relative_path,
                    str(metadata["sha256"]),
                    f"data:{relative_path}",
                )
            )
    else:
        checks.append(
            VerificationCheck(
                "processed-data-manifest",
                "failed",
                "docs/datasets/PROCESSED_DATA_VALIDATION.json",
                detail="manifest missing",
            )
        )

    baseline_metadata_path = root / "artifacts/models/baseline_v1/metadata.json"
    if baseline_metadata_path.is_file():
        baseline = read_json(baseline_metadata_path)
        checks.append(
            verify_sha256(
                root,
                "artifacts/models/baseline_v1/model.joblib",
                str(baseline["model_sha256"]),
                "baseline-v1-model",
            )
        )
        checks.append(
            VerificationCheck(
                "baseline-llm-boundary",
                "passed" if "No LLM" in str(baseline.get("decision_boundary")) else "failed",
                "artifacts/models/baseline_v1/metadata.json",
                "No LLM in decision boundary",
                baseline.get("decision_boundary"),
            )
        )
    else:
        checks.append(
            VerificationCheck(
                "baseline-v1-model",
                "failed",
                "artifacts/models/baseline_v1/metadata.json",
                detail="metadata missing",
            )
        )

    ladder_metadata_path = root / "reports/model_ladder_v1/run_metadata.json"
    if ladder_metadata_path.is_file():
        ladder = read_json(ladder_metadata_path)
        checks.append(
            verify_sha256(
                root,
                "artifacts/models/model_ladder_v1/selected_detector.joblib",
                str(ladder["selected_detector_sha256"]),
                "selected-classical-detector",
            )
        )

    candidate_metadata_path = root / "reports/mixed_source_candidate_v2/run_metadata.json"
    if candidate_metadata_path.is_file():
        candidate = read_json(candidate_metadata_path)
        checks.extend(
            [
                verify_sha256(
                    root,
                    "artifacts/models/mixed_source_candidate_v2/selected_detector.joblib",
                    str(candidate["artifact_sha256"]),
                    "calibrated-mixed-source-detector",
                ),
                verify_json_value(
                    root,
                    "reports/mixed_source_candidate_v2/run_metadata.json",
                    ("promotion_status",),
                    "research_only_not_promoted",
                    "candidate-promotion-status",
                ),
                verify_json_value(
                    root,
                    "reports/mixed_source_candidate_v2/metrics.json",
                    ("llm_used_for_detection",),
                    False,
                    "candidate-llm-boundary",
                ),
            ]
        )

    simple_metadata_path = root / "reports/simple_content_detector_v1/run_metadata.json"
    if simple_metadata_path.is_file():
        simple = read_json(simple_metadata_path)
        checks.extend(
            [
                verify_sha256(
                    root,
                    "artifacts/models/simple_content_detector_v1/selected_detector.joblib",
                    str(simple["artifact_sha256"]),
                    "simple-content-detector",
                ),
                verify_sha256(
                    root,
                    "major_project/models/selected_detector.joblib",
                    str(simple["artifact_sha256"]),
                    "packaged-simple-content-detector",
                ),
                verify_json_value(
                    root,
                    "major_project/reports/behavioral_evaluation.json",
                    ("accuracy",),
                    0.9,
                    "major-project-behavioral-accuracy",
                ),
            ]
        )

    risk_metadata_path = root / "reports/risk_fusion_v1/run_metadata.json"
    if risk_metadata_path.is_file():
        risk = read_json(risk_metadata_path)
        checks.append(
            verify_sha256(
                root,
                "artifacts/models/risk_fusion_v1/risk_fusion.joblib",
                str(risk["artifact_sha256"]),
                "risk-fusion-model",
            )
        )
        checks.append(
            verify_json_value(
                root,
                "reports/risk_fusion_v1/run_metadata.json",
                ("llm_used_for_detection",),
                False,
                "risk-fusion-llm-boundary",
            )
        )
        checks.append(
            verify_json_value(
                root,
                "reports/risk_fusion_v1/run_metadata.json",
                ("promotion_status",),
                "research_only_not_promoted",
                "risk-fusion-promotion-status",
            )
        )

    whisper_manifest_path = root / "reports/asr_smoke_v1/MODEL_MANIFEST.json"
    if whisper_manifest_path.is_file():
        whisper = read_json(whisper_manifest_path)
        local = str(whisper["local_path"])
        for item in whisper["files"]:
            checks.append(
                verify_sha256(
                    root,
                    f"{local}/{item['path']}",
                    str(item["sha256"]),
                    f"whisper:{item['path']}",
                )
            )

    classical_report = root / "reports/classical_multitask_v1/run_metadata.json"
    classical_manifest = root / "artifacts/models/classical_multitask_v1/manifest.json"
    transformer_report = root / "reports/multitask_transformer_v1/run_metadata.json"
    transformer_manifest = root / "artifacts/models/multitask_transformer_v1/manifest.json"
    if classical_report.is_file() and classical_manifest.is_file():
        metadata = read_json(classical_report)
        checks.extend(
            [
                verify_sha256(
                    root,
                    "artifacts/models/classical_multitask_v1/manifest.json",
                    str(metadata["manifest_sha256"]),
                    "classical-multitask-manifest",
                ),
                verify_sha256(
                    root,
                    "artifacts/models/classical_multitask_v1/classical_multitask_heads.joblib",
                    str(metadata["heads_sha256"]),
                    "classical-multitask-heads",
                ),
                verify_json_value(
                    root,
                    "artifacts/models/classical_multitask_v1/manifest.json",
                    ("llm_used_for_detection",),
                    False,
                    "classical-multitask-llm-boundary",
                ),
                verify_json_value(
                    root,
                    "artifacts/models/classical_multitask_v1/manifest.json",
                    ("used_as_api_decision_source",),
                    False,
                    "classical-multitask-auxiliary-only",
                ),
            ]
        )
    elif transformer_report.is_file() and transformer_manifest.is_file():
        metadata = read_json(transformer_report)
        checks.extend(
            [
                verify_sha256(
                    root,
                    "artifacts/models/multitask_transformer_v1/manifest.json",
                    str(metadata["manifest_sha256"]),
                    "multitask-transformer-manifest",
                ),
                verify_sha256(
                    root,
                    "artifacts/models/multitask_transformer_v1/multitask_heads.pt",
                    str(metadata["heads_sha256"]),
                    "multitask-transformer-heads",
                ),
                verify_json_value(
                    root,
                    "artifacts/models/multitask_transformer_v1/manifest.json",
                    ("llm_used_for_detection",),
                    False,
                    "multitask-transformer-llm-boundary",
                ),
            ]
        )
    else:
        checks.append(
            VerificationCheck(
                "multitask-artifact",
                "failed" if require_multitask else "skipped",
                "artifacts/models/classical_multitask_v1",
                detail="neither completed classical nor transformer multi-task artifact/report is available",
            )
        )

    checks.extend(
        [
            verify_json_value(
                root,
                "data/human_test/COLLECTION_STATUS.json",
                ("human_gold_available",),
                False,
                "human-gold-status-truthful",
            ),
            verify_json_value(
                root,
                "data/audio_validation/COLLECTION_STATUS.json",
                ("metrics_available",),
                False,
                "audio-validation-status-truthful",
            ),
            verify_json_value(
                root,
                "configs/deployment/api.json",
                ("boundaries", "llm_used_for_detection"),
                False,
                "api-llm-boundary",
            ),
            verify_json_value(
                root,
                "configs/deployment/api.json",
                ("policy", "enable_honeypot_handoff"),
                False,
                "honeypot-handoff-disabled",
            ),
            verify_json_value(
                root,
                "configs/deployment/api.json",
                ("models", "base_detector_path"),
                "artifacts/models/mixed_source_candidate_v2/selected_detector.joblib",
                "api-loads-calibrated-candidate",
            ),
            verify_json_value(
                root,
                "configs/deployment/api.json",
                ("policy", "allow_research_fusion"),
                False,
                "research-fusion-disabled",
            ),
            verify_json_value(
                root,
                "reports/external_text_v1/metrics.json",
                ("threshold_was_frozen",),
                True,
                "external-threshold-frozen",
            ),
        ]
    )

    serialized = [asdict(check) for check in checks]
    failed = [check for check in serialized if check["status"] == "failed"]
    skipped = [check for check in serialized if check["status"] == "skipped"]
    return {
        "schema_version": "1.0.0",
        "root": str(root),
        "require_multitask": require_multitask,
        "status": "passed" if not failed else "failed",
        "counts": {
            "checks": len(serialized),
            "passed": sum(check["status"] == "passed" for check in serialized),
            "failed": len(failed),
            "skipped": len(skipped),
        },
        "checks": serialized,
    }
