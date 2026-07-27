from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.demo_cases_artifact import (
    DEMO_BENIGN_CASE_COUNT,
    DEMO_CASE_COUNT,
    DEMO_EXPECTED_REFERENCE_LABELS,
    DEMO_EXPECTED_SAMPLE_INDICES,
    DEMO_FORBIDDEN_CASE_KEYS,
    DEMO_MALIGNANT_CASE_COUNT,
    DEMO_OFFICIAL_HOLDOUT_ACCURACY,
    DEMO_SELECTION_RULE,
    load_demo_cases_artifact,
)


def test_committed_demo_cases_artifact_loads_successfully() -> None:
    payload = load_demo_cases_artifact()

    assert payload["selection_rule"] == DEMO_SELECTION_RULE
    assert payload["official_holdout_accuracy"] == DEMO_OFFICIAL_HOLDOUT_ACCURACY
    assert payload["case_count"] == DEMO_CASE_COUNT
    assert payload["malignant_case_count"] == DEMO_MALIGNANT_CASE_COUNT
    assert payload["benign_case_count"] == DEMO_BENIGN_CASE_COUNT
    assert payload["sample_indices"] == DEMO_EXPECTED_SAMPLE_INDICES
    assert payload["feature_names"] == WDBC_FEATURE_NAMES

    for position, case in enumerate(payload["cases"], start=1):
        assert case["case_id"] == f"demo-{position:02d}"
        assert case["sample_index"] == DEMO_EXPECTED_SAMPLE_INDICES[position - 1]
        assert case["reference_label"] == DEMO_EXPECTED_REFERENCE_LABELS[case["sample_index"]]
        assert not (set(case) & DEMO_FORBIDDEN_CASE_KEYS)
        assert list(case["features"]) == WDBC_FEATURE_NAMES
        assert len(case["features"]) == 30
