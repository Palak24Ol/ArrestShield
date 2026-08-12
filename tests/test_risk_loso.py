from arrestshield.protocol import summarize_seed_values


def test_strict_audit_config_requires_every_source_not_only_mean() -> None:
    source_fprs = [0.01, 0.02, 0.03, 0.08]
    assert sum(source_fprs) / len(source_fprs) <= 0.05
    assert not all(value <= 0.05 for value in source_fprs)
    summary = summarize_seed_values(source_fprs)
    assert summary["maximum"] == 0.08
