"""Single-use command to materialize selected holdout demonstration cases."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from femhealth.data import WDBC_FEATURE_NAMES, load_wdbc_data
from femhealth.data_split import split_development_test
from femhealth.demo_cases_artifact import (
    DEMO_ARTIFACT_VERSION,
    DEMO_BENIGN_CASE_COUNT,
    DEMO_CASE_COUNT,
    DEMO_CASES_PATH,
    DEMO_EXPECTED_REFERENCE_LABELS,
    DEMO_EXPECTED_SAMPLE_INDICES,
    DEMO_HOLDOUT_SAMPLE_COUNT,
    DEMO_LABEL_TO_CLASS,
    DEMO_MALIGNANT_CASE_COUNT,
    DEMO_OFFICIAL_HOLDOUT_ACCURACY,
    DEMO_SELECTION_RULE,
    DEMO_SOURCE_DATASET,
    DEMO_SOURCE_SPLIT,
    DEMO_TRAINING_SAMPLE_COUNT,
    validate_demo_cases_payload,
)

FINAL_HOLDOUT_PREDICTIONS_PATH = Path("reports/results/final_holdout_predictions.csv")
FINAL_HOLDOUT_SUMMARY_PATH = Path("reports/results/final_holdout_summary.json")
PREDICTION_REQUIRED_COLUMNS = {
    "sample_index",
    "true_label",
    "probability_malignant",
    "predicted_label",
    "correct",
    "error_type",
}


def run_demo_cases_once(
    predictions_path: Path = FINAL_HOLDOUT_PREDICTIONS_PATH,
    final_summary_path: Path = FINAL_HOLDOUT_SUMMARY_PATH,
    output_path: Path = DEMO_CASES_PATH,
    data_loader: Callable[[], tuple[pd.DataFrame, pd.Series]] = load_wdbc_data,
    splitter: Callable[
        [pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    ] = split_development_test,
) -> dict:
    """Build and persist the demonstration cases artifact once."""
    output_path = Path(output_path)
    _ensure_output_path_is_new(output_path)

    predictions = _load_final_holdout_predictions(Path(predictions_path))
    selected_predictions = select_demo_prediction_rows(predictions)
    _validate_selected_prediction_rows(selected_predictions)

    X, y = data_loader()
    X_development, X_holdout, y_development, y_holdout = splitter(X, y)
    _validate_demo_indices_against_split(
        X_development,
        X_holdout,
        y_development,
        y_holdout,
        selected_predictions["sample_index"].tolist(),
    )

    official_accuracy = _load_official_holdout_accuracy(Path(final_summary_path))
    payload = build_demo_cases_payload(
        X_holdout,
        y_holdout,
        selected_predictions["sample_index"].tolist(),
        official_accuracy,
    )
    validate_demo_cases_payload(payload)
    _write_json_atomically(payload, output_path)
    return payload


def select_demo_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the first eight rows in the frozen holdout order."""
    _validate_prediction_columns(predictions)
    return predictions.head(DEMO_CASE_COUNT).copy()


def build_demo_cases_payload(
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    sample_indices: list[int],
    official_accuracy: float,
) -> dict:
    """Build the JSON-ready demonstration cases payload without prediction fields."""
    if sample_indices != DEMO_EXPECTED_SAMPLE_INDICES:
        raise ValueError("Unexpected demo sample indices")

    cases = []
    for position, sample_index in enumerate(sample_indices, start=1):
        reference_label = int(y_holdout.loc[sample_index])
        features = {
            feature_name: _finite_float(X_holdout.loc[sample_index, feature_name])
            for feature_name in WDBC_FEATURE_NAMES
        }
        cases.append(
            {
                "case_id": f"demo-{position:02d}",
                "sample_index": int(sample_index),
                "reference_label": reference_label,
                "reference_class": DEMO_LABEL_TO_CLASS[reference_label],
                "features": features,
            }
        )

    return {
        "artifact_version": DEMO_ARTIFACT_VERSION,
        "source_dataset": DEMO_SOURCE_DATASET,
        "source_split": DEMO_SOURCE_SPLIT,
        "selection_rule": DEMO_SELECTION_RULE,
        "used_for_training": False,
        "used_for_model_selection": False,
        "created_after_final_evaluation": True,
        "training_sample_count": DEMO_TRAINING_SAMPLE_COUNT,
        "holdout_sample_count": DEMO_HOLDOUT_SAMPLE_COUNT,
        "official_holdout_accuracy": official_accuracy,
        "case_count": DEMO_CASE_COUNT,
        "malignant_case_count": DEMO_MALIGNANT_CASE_COUNT,
        "benign_case_count": DEMO_BENIGN_CASE_COUNT,
        "sample_indices": list(sample_indices),
        "feature_names": list(WDBC_FEATURE_NAMES),
        "cases": cases,
    }


def _load_final_holdout_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Final holdout predictions not found: {path}")

    predictions = pd.read_csv(path)
    _validate_prediction_columns(predictions)
    return predictions


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("Final holdout predictions must be a DataFrame")

    missing_columns = sorted(PREDICTION_REQUIRED_COLUMNS - set(predictions.columns))
    if missing_columns:
        raise ValueError("Final holdout predictions are missing required columns")


def _validate_selected_prediction_rows(selected_predictions: pd.DataFrame) -> None:
    if len(selected_predictions) != DEMO_CASE_COUNT:
        raise ValueError("Unexpected selected demo case count")

    sample_indices = [int(value) for value in selected_predictions["sample_index"].tolist()]
    labels = [int(value) for value in selected_predictions["true_label"].tolist()]

    if sample_indices != DEMO_EXPECTED_SAMPLE_INDICES:
        raise ValueError("Unexpected selected demo sample indices")

    expected_labels = [DEMO_EXPECTED_REFERENCE_LABELS[index] for index in sample_indices]
    if labels != expected_labels:
        raise ValueError("Unexpected selected demo labels")

    if labels.count(0) != DEMO_MALIGNANT_CASE_COUNT or labels.count(1) != DEMO_BENIGN_CASE_COUNT:
        raise ValueError("Unexpected selected demo class distribution")


def _validate_demo_indices_against_split(
    X_development: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_development: pd.Series,
    y_holdout: pd.Series,
    sample_indices: list[int],
) -> None:
    development_indices = set(X_development.index)
    holdout_indices = set(X_holdout.index)

    if set(sample_indices) - holdout_indices:
        raise ValueError("Demo indices must belong to the final holdout")

    if set(sample_indices) & development_indices:
        raise ValueError("Demo indices must not belong to development data")

    if not X_development.index.equals(y_development.index):
        raise ValueError("Development split index mismatch")

    if not X_holdout.index.equals(y_holdout.index):
        raise ValueError("Holdout split index mismatch")

    if list(X_holdout.columns) != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected holdout feature order")

    for sample_index in sample_indices:
        observed_label = int(y_holdout.loc[sample_index])
        if observed_label != DEMO_EXPECTED_REFERENCE_LABELS[sample_index]:
            raise ValueError("Unexpected holdout reference label")


def _load_official_holdout_accuracy(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Final holdout summary not found: {path}")

    summary = json.loads(path.read_text(encoding="utf-8"))
    accuracy = summary.get("accuracy")
    if accuracy != DEMO_OFFICIAL_HOLDOUT_ACCURACY:
        raise ValueError("Unexpected official holdout accuracy")

    return float(accuracy)


def _write_json_atomically(payload: dict, output_path: Path) -> None:
    _ensure_output_path_is_new(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _ensure_output_path_is_new(output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"Demo cases artifact already exists: {output_path}")


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Demo feature values must be finite numbers")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("Demo feature values must be finite numbers")

    return numeric_value


def _sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> None:
    """Generate the demonstration artifact and print a compact summary."""
    payload = run_demo_cases_once()
    print(f"Artefato salvo em: {DEMO_CASES_PATH}")
    print(f"SHA-256: {_sha256_file(DEMO_CASES_PATH)}")
    print(f"Indices: {payload['sample_indices']}")
    print(
        "Classes: "
        f"malignant={payload['malignant_case_count']}, "
        f"benign={payload['benign_case_count']}"
    )


if __name__ == "__main__":
    main()
