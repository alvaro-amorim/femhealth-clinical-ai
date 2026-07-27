import json
from copy import deepcopy

import pytest

from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.demo_cases_artifact import (
    DEMO_ARTIFACT_VERSION,
    DEMO_BENIGN_CASE_COUNT,
    DEMO_CASE_COUNT,
    DEMO_CASES_PATH,
    DEMO_EXPECTED_REFERENCE_LABELS,
    DEMO_EXPECTED_SAMPLE_INDICES,
    DEMO_HOLDOUT_SAMPLE_COUNT,
    DEMO_MALIGNANT_CASE_COUNT,
    DEMO_OFFICIAL_HOLDOUT_ACCURACY,
    DEMO_SELECTION_RULE,
    DEMO_SOURCE_DATASET,
    DEMO_SOURCE_SPLIT,
    DEMO_TRAINING_SAMPLE_COUNT,
    load_demo_cases_artifact,
    validate_demo_cases_payload,
)


def test_default_demo_cases_path() -> None:
    assert DEMO_CASES_PATH.as_posix() == "artifacts/demo/holdout_demo_cases.json"


def test_load_demo_cases_artifact_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_demo_cases_artifact(tmp_path / "missing.json")


def test_load_demo_cases_artifact_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "holdout_demo_cases.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_demo_cases_artifact(path)


def test_load_demo_cases_artifact_accepts_valid_payload(tmp_path) -> None:
    payload = _valid_payload()
    path = tmp_path / "holdout_demo_cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_demo_cases_artifact(path)

    assert loaded == payload
    assert loaded["case_count"] == DEMO_CASE_COUNT
    assert loaded["malignant_case_count"] == DEMO_MALIGNANT_CASE_COUNT
    assert loaded["benign_case_count"] == DEMO_BENIGN_CASE_COUNT
    assert [case["sample_index"] for case in loaded["cases"]] == DEMO_EXPECTED_SAMPLE_INDICES
    assert [case["case_id"] for case in loaded["cases"]] == [
        f"demo-{index:02d}" for index in range(1, 9)
    ]


@pytest.mark.parametrize(
    "key",
    [
        "artifact_version",
        "source_dataset",
        "source_split",
        "selection_rule",
        "used_for_training",
        "used_for_model_selection",
        "created_after_final_evaluation",
        "training_sample_count",
        "holdout_sample_count",
        "official_holdout_accuracy",
        "case_count",
        "malignant_case_count",
        "benign_case_count",
        "sample_indices",
        "feature_names",
        "cases",
    ],
)
def test_validate_demo_cases_payload_rejects_missing_top_level_key(key) -> None:
    payload = _valid_payload()
    del payload[key]

    with pytest.raises(ValueError, match="missing required keys"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_extra_top_level_key() -> None:
    payload = _valid_payload()
    payload["extra"] = "unexpected"

    with pytest.raises(ValueError, match="unexpected keys"):
        validate_demo_cases_payload(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("artifact_version", "2.0.0"),
        ("source_dataset", "other"),
        ("source_split", "other"),
        ("selection_rule", "manual"),
        ("training_sample_count", 454),
        ("holdout_sample_count", 113),
        ("official_holdout_accuracy", 0.5),
    ],
)
def test_validate_demo_cases_payload_rejects_invalid_metadata(key, value) -> None:
    payload = _valid_payload()
    payload[key] = value

    with pytest.raises(ValueError, match="metadata"):
        validate_demo_cases_payload(payload)


@pytest.mark.parametrize(
    "key",
    ["used_for_training", "used_for_model_selection"],
)
def test_validate_demo_cases_payload_rejects_selection_or_training_flags_true(key) -> None:
    payload = _valid_payload()
    payload[key] = True

    with pytest.raises(ValueError, match="metadata"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_created_before_final_evaluation() -> None:
    payload = _valid_payload()
    payload["created_after_final_evaluation"] = False

    with pytest.raises(ValueError, match="metadata"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_wrong_case_count() -> None:
    payload = _valid_payload()
    payload["cases"] = payload["cases"][:-1]

    with pytest.raises(ValueError, match="case count"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_indices_out_of_order() -> None:
    payload = _valid_payload()
    payload["sample_indices"] = list(reversed(DEMO_EXPECTED_SAMPLE_INDICES))

    with pytest.raises(ValueError, match="metadata"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_missing_feature() -> None:
    payload = _valid_payload()
    del payload["cases"][0]["features"][WDBC_FEATURE_NAMES[0]]

    with pytest.raises(ValueError, match="feature names"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_extra_feature() -> None:
    payload = _valid_payload()
    payload["cases"][0]["features"]["extra"] = 1.0

    with pytest.raises(ValueError, match="feature names"):
        validate_demo_cases_payload(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.0"])
def test_validate_demo_cases_payload_rejects_invalid_feature_values(value) -> None:
    payload = _valid_payload()
    payload["cases"][0]["features"][WDBC_FEATURE_NAMES[0]] = value

    with pytest.raises(ValueError, match="finite numbers"):
        validate_demo_cases_payload(payload)


def test_validate_demo_cases_payload_rejects_label_class_divergence() -> None:
    payload = _valid_payload()
    payload["cases"][0]["reference_class"] = "benign"

    with pytest.raises(ValueError, match="reference class"):
        validate_demo_cases_payload(payload)


@pytest.mark.parametrize("key", ["predicted_label", "probability_malignant"])
def test_validate_demo_cases_payload_rejects_prediction_fields_in_cases(key) -> None:
    payload = _valid_payload()
    payload["cases"][0][key] = 0

    with pytest.raises(ValueError, match="prediction fields"):
        validate_demo_cases_payload(payload)


def _valid_payload() -> dict:
    cases = []
    for position, sample_index in enumerate(DEMO_EXPECTED_SAMPLE_INDICES, start=1):
        reference_label = DEMO_EXPECTED_REFERENCE_LABELS[sample_index]
        cases.append(
            {
                "case_id": f"demo-{position:02d}",
                "sample_index": sample_index,
                "reference_label": reference_label,
                "reference_class": "malignant" if reference_label == 0 else "benign",
                "features": {
                    feature_name: float(position + feature_position / 100)
                    for feature_position, feature_name in enumerate(WDBC_FEATURE_NAMES)
                },
            }
        )

    return deepcopy(
        {
            "artifact_version": DEMO_ARTIFACT_VERSION,
            "source_dataset": DEMO_SOURCE_DATASET,
            "source_split": DEMO_SOURCE_SPLIT,
            "selection_rule": DEMO_SELECTION_RULE,
            "used_for_training": False,
            "used_for_model_selection": False,
            "created_after_final_evaluation": True,
            "training_sample_count": DEMO_TRAINING_SAMPLE_COUNT,
            "holdout_sample_count": DEMO_HOLDOUT_SAMPLE_COUNT,
            "official_holdout_accuracy": DEMO_OFFICIAL_HOLDOUT_ACCURACY,
            "case_count": DEMO_CASE_COUNT,
            "malignant_case_count": DEMO_MALIGNANT_CASE_COUNT,
            "benign_case_count": DEMO_BENIGN_CASE_COUNT,
            "sample_indices": DEMO_EXPECTED_SAMPLE_INDICES,
            "feature_names": WDBC_FEATURE_NAMES,
            "cases": cases,
        }
    )
