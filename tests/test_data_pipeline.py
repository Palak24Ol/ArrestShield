import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_canonical_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_canonical_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DataPipelineTests(unittest.TestCase):
    def test_normalize_unicode_and_space(self):
        self.assertEqual(MODULE.normalize_text("  A\u200b  B  "), "A B")

    def test_role_parser(self):
        turns = MODULE.split_role_dialogue("caller: Hello receiver: Hi caller: Urgent", "caller_receiver")
        self.assertEqual(turns, [("caller", "Hello"), ("recipient", "Hi"), ("caller", "Urgent")])

    def test_template_fingerprint_masks_numbers(self):
        a = MODULE.build_turns([("caller", "Pay 100 now")], "english", 1, "other_scam", True, False)
        b = MODULE.build_turns([("caller", "Pay 500 now")], "english", 1, "other_scam", True, False)
        self.assertNotEqual(MODULE.fingerprint_text(a, False), MODULE.fingerprint_text(b, False))
        self.assertEqual(MODULE.fingerprint_text(a, True), MODULE.fingerprint_text(b, True))

    def test_split_is_deterministic(self):
        value = "a" * 64
        self.assertEqual(MODULE.assign_split(value), MODULE.assign_split(value))

    def test_simhash_near_text(self):
        a = MODULE.build_turns([("caller", "Officer says account is blocked pay 100 immediately")], "english", 1, "other_scam", True, False)
        b = MODULE.build_turns([("caller", "Officer says account is blocked pay 500 immediately")], "english", 1, "other_scam", True, False)
        self.assertLessEqual((MODULE.simhash64(a) ^ MODULE.simhash64(b)).bit_count(), 6)

    def test_weak_labels_are_not_gold(self):
        labels = MODULE.positive_weak_labels("CBI officer: pay immediately", "digital_arrest", 0)
        self.assertEqual(labels["provenance"], "weak_rule")
        self.assertEqual(labels["tactics"]["authority_impersonation"], 1)
        self.assertEqual(labels["tactics"]["urgency"], 1)


if __name__ == "__main__":
    unittest.main()
