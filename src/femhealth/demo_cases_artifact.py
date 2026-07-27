"""Validated loader for the holdout demonstration cases artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from femhealth.data import WDBC_FEATURE_NAMES

DEMO_CASES_PATH = Path("artifacts/demo/holdout_demo_cases.json")
DEMO_ARTIFACT_VERSION = "1.0.0"
DEMO_SOURCE_DATASET = "WDBC"
DEMO_SOURCE_SPLIT = "final_holdout"
DEMO_SELECTION_RULE = "first_8_rows_of_frozen_final_holdout_order"
DEMO_TRAINING_SAMPLE_COUNT = 455
DEMO_HOLDOUT_SAMPLE_COUNT = 114
DEMO_OFFICIAL_HOLDOUT_ACCURACY = 0.9736842105263158
DEMO_EXPECTED_SAMPLE_INDICES = [256, 428, 501, 363, 564, 464, 358, 343]
DEMO_EXPECTED_REFERENCE_LABELS = {
    256: 0,
    428: 1,
    501: 0,
    363: 1,
    564: 0,
    464: 1,
    358: 1,
    343: 0,
}
DEMO_LABEL_TO_CLASS = {
    0: "malignant",
    1: "benign",
}
DEMO_CASE_COUNT = 8
DEMO_MALIGNANT_CASE_COUNT = 4
DEMO_BENIGN_CASE_COUNT = 4
DEMO_FORBIDDEN_CASE_KEYS = {
    "predicted_label",
    "predicted_class",
    "probability_malignant",
    "probability_benign",
    "correct",
    "error_type",
}

_TOP_LEVEL_KEYS = {
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
}
_CASE_KEYS = {
    "case_id",
    "sample_index",
    "reference_label",
    "reference_class",
    "features",
}


def load_demo_cases_artifact(
    path: Path = DEMO_CASES_PATH,
) -> dict:
    """Load and validate the frozen demonstration cases artifact."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Demo cases artifact not found: {artifact_path}")

    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Demo cases artifact is not valid JSON") from exc

    validate_demo_cases_payload(payload)
    return payload


def validate_demo_cases_payload(payload: Any) -> None:
    """Validate the demonstration cases artifact contract."""
    if not isinstance(payload, dict):
        raise ValueError("Demo cases artifact must be an object")

    _validate_exact_keys(payload, _TOP_LEVEL_KEYS, "Demo cases artifact")
    _validate_metadata(payload)
    _validate_cases(payload["cases"])


def _validate_metadata(payload: dict) -> None:
    expected_values = {
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
    }

    for key, expected_value in expected_values.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"Unexpected demo cases metadata: {key}")


def _validate_cases(cases: Any) -> None:
    if not isinstance(cases, list):
        raise ValueError("Demo cases must be a list")

    if len(cases) != DEMO_CASE_COUNT:
        raise ValueError("Unexpected demo case count")

    reference_labels = []
    for position, case in enumerate(cases, start=1):
        _validate_case(case, position)
        reference_labels.append(case["reference_label"])

    if reference_labels.count(0) != DEMO_MALIGNANT_CASE_COUNT:
        raise ValueError("Unexpected malignant demo case count")

    if reference_labels.count(1) != DEMO_BENIGN_CASE_COUNT:
        raise ValueError("Unexpected benign demo case count")


def _validate_case(case: Any, position: int) -> None:
    if not isinstance(case, dict):
        raise ValueError("Demo case must be an object")

    forbidden_keys = set(case) & DEMO_FORBIDDEN_CASE_KEYS
    if forbidden_keys:
        raise ValueError("Demo case contains prediction fields")

    _validate_exact_keys(case, _CASE_KEYS, "Demo case")

    expected_index = DEMO_EXPECTED_SAMPLE_INDICES[position - 1]
    expected_label = DEMO_EXPECTED_REFERENCE_LABELS[expected_index]
    expected_class = DEMO_LABEL_TO_CLASS[expected_label]

    if case["case_id"] != f"demo-{position:02d}":
        raise ValueError("Unexpected demo case id")

    if case["sample_index"] != expected_index:
        raise ValueError("Unexpected demo case sample index")

    if case["reference_label"] != expected_label:
        raise ValueError("Unexpected demo case reference label")

    if case["reference_class"] != expected_class:
        raise ValueError("Unexpected demo case reference class")

    _validate_features(case["features"])


def _validate_features(features: Any) -> None:
    if not isinstance(features, dict):
        raise ValueError("Demo case features must be an object")

    if list(features) != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected demo case feature names")

    for value in features.values():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("Demo case feature values must be finite numbers")

        if not math.isfinite(float(value)):
            raise ValueError("Demo case feature values must be finite numbers")


def _validate_exact_keys(payload: dict, expected_keys: set[str], context: str) -> None:
    missing_keys = sorted(expected_keys - set(payload))
    extra_keys = sorted(set(payload) - expected_keys)

    if missing_keys:
        raise ValueError(f"{context} is missing required keys")

    if extra_keys:
        raise ValueError(f"{context} contains unexpected keys")
