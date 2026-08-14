from __future__ import annotations

import ast
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from detector import SimpleScamDetector  # noqa: E402


def detector() -> SimpleScamDetector:
    return SimpleScamDetector(PROJECT_DIR / "models/selected_detector.joblib")


def test_scam_and_legitimate_demo_predictions() -> None:
    model = detector()
    scam = model.predict(
        "Main CBI se bol raha hoon, kisi ko mat batana aur safe account me paise bhejo"
    )
    legitimate = model.predict("Your courier will arrive tomorrow and no payment is required")
    assert scam.label == "SCAM"
    assert legitimate.label == "NOT_SCAM"
    assert scam.llm_used_for_detection is False
    assert legitimate.llm_used_for_detection is False


def test_detector_module_cannot_import_honeypot() -> None:
    tree = ast.parse((PROJECT_DIR / "detector.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert all("honeypot" not in name and "groq" not in name for name in imports)
