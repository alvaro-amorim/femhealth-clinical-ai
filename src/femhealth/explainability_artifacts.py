"""Validated loader for persisted global explainability artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from femhealth.data import WDBC_FEATURE_NAMES

EXPLAINABILITY_DIRECTORY = Path("reports/explainability")
DETAILS_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_details.csv"
SUMMARY_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_summary.csv"
FOLD_SCORES_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_fold_scores.csv"
METADATA_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_metadata.json"
PLOT_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_top15.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

DETAIL_COLUMNS = [
    "fold",
    "feature_name",
    "feature_position",
    "repeat",
    "importance",
    "baseline_roc_auc",
    "validation_sample_count",
    "validation_malignant_count",
    "validation_benign_count",
]
SUMMARY_COLUMNS = [
    "rank",
    "feature_name",
    "feature_position",
    "mean_importance",
    "std_importance",
    "median_importance",
    "min_importance",
    "max_importance",
    "positive_fraction",
    "fold_count",
    "observation_count",
]
FOLD_SCORE_COLUMNS = [
    "fold",
    "train_sample_count",
    "validation_sample_count",
    "train_malignant_count",
    "train_benign_count",
    "validation_malignant_count",
    "validation_benign_count",
    "baseline_roc_auc",
]

EXPECTED_METADATA = {
    "method": "cross_validated_permutation_importance",
    "scorer": "roc_auc_malignant",
    "selected_variant": "svm_sigmoid",
    "selected_model": "svm",
    "selected_calibration": "sigmoid",
    "selected_threshold": 0.51,
    "malignant_label": 0,
    "benign_label": 1,
    "development_sample_count": 455,
    "cv_splits": 5,
    "shuffle": True,
    "random_state": 42,
    "permutation_repeats": 10,
    "feature_count": 30,
    "detail_row_count": 1500,
    "holdout_used": False,
    "final_model_artifact_modified": False,
}


def load_explainability_artifacts(
    details_path: Path = DETAILS_PATH,
    summary_path: Path = SUMMARY_PATH,
    fold_scores_path: Path = FOLD_SCORES_PATH,
    metadata_path: Path = METADATA_PATH,
    plot_path: Path = PLOT_PATH,
) -> tuple[dict, bytes]:
    """Load and validate persisted explainability artifacts for read-only serving."""
    paths = [details_path, summary_path, fold_scores_path, metadata_path, plot_path]
    _require_existing_files(paths)

    details = pd.read_csv(details_path)
    summary = pd.read_csv(summary_path)
    fold_scores = pd.read_csv(fold_scores_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plot_bytes = plot_path.read_bytes()

    _validate_details(details)
    _validate_summary(summary)
    _validate_fold_scores(fold_scores)
    _validate_metadata(metadata)
    _validate_png(plot_bytes)

    payload = _build_payload(metadata, summary, fold_scores)
    return payload, plot_bytes


def _require_existing_files(paths: list[Path]) -> None:
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing explainability artifact: {missing_paths[0]}")


def _validate_details(details: pd.DataFrame) -> None:
    _require_columns(details, DETAIL_COLUMNS, "details")
    _require_row_count(details, 1500, "details")
    _require_no_nulls(details, "details")
    _require_finite_numeric(details, _numeric_columns(DETAIL_COLUMNS), "details")

    if sorted(details["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("Unexpected detail folds")

    if sorted(details["repeat"].unique().tolist()) != list(range(1, 11)):
        raise ValueError("Unexpected detail repeats")

    if sorted(details["feature_name"].unique().tolist()) != sorted(WDBC_FEATURE_NAMES):
        raise ValueError("Unexpected detail features")

    if not details.groupby("feature_name").size().eq(50).all():
        raise ValueError("Unexpected detail observations by feature")

    if not details["importance"].lt(0).any():
        raise ValueError("Expected at least one negative importance")


def _validate_summary(summary: pd.DataFrame) -> None:
    _require_columns(summary, SUMMARY_COLUMNS, "summary")
    _require_row_count(summary, 30, "summary")
    _require_no_nulls(summary, "summary")
    _require_finite_numeric(summary, _numeric_columns(SUMMARY_COLUMNS), "summary")

    if summary["rank"].tolist() != list(range(1, 31)):
        raise ValueError("Unexpected summary ranks")

    if not summary["mean_importance"].is_monotonic_decreasing:
        raise ValueError("Unexpected summary importance order")

    if summary["feature_name"].nunique() != 30:
        raise ValueError("Duplicated summary features")

    if set(summary["feature_name"]) != set(WDBC_FEATURE_NAMES):
        raise ValueError("Unexpected summary features")

    canonical_positions = {
        feature_name: position for position, feature_name in enumerate(WDBC_FEATURE_NAMES, start=1)
    }
    for row in summary.itertuples(index=False):
        if row.feature_position != canonical_positions[row.feature_name]:
            raise ValueError("Unexpected feature position")

    if not summary["fold_count"].eq(5).all():
        raise ValueError("Unexpected fold count")

    if not summary["observation_count"].eq(50).all():
        raise ValueError("Unexpected observation count")

    if not summary["positive_fraction"].between(0, 1).all():
        raise ValueError("Unexpected positive fraction")


def _validate_fold_scores(fold_scores: pd.DataFrame) -> None:
    _require_columns(fold_scores, FOLD_SCORE_COLUMNS, "fold scores")
    _require_row_count(fold_scores, 5, "fold scores")
    _require_no_nulls(fold_scores, "fold scores")
    _require_finite_numeric(fold_scores, _numeric_columns(FOLD_SCORE_COLUMNS), "fold scores")

    if fold_scores["fold"].tolist() != [1, 2, 3, 4, 5]:
        raise ValueError("Unexpected fold score folds")

    if not fold_scores["baseline_roc_auc"].between(0, 1).all():
        raise ValueError("Unexpected fold ROC AUC")

    sample_sums = fold_scores["train_sample_count"] + fold_scores["validation_sample_count"]
    if not sample_sums.eq(455).all():
        raise ValueError("Unexpected fold sample counts")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for key, expected_value in EXPECTED_METADATA.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"Unexpected metadata value: {key}")

    if metadata.get("feature_names") != WDBC_FEATURE_NAMES:
        raise ValueError("Unexpected metadata feature names")

    limitations = metadata.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item for item in limitations)
    ):
        raise ValueError("Unexpected metadata limitations")


def _validate_png(plot_bytes: bytes) -> None:
    if not plot_bytes:
        raise ValueError("Empty explainability plot")

    if not plot_bytes.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid explainability plot signature")


def _build_payload(
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    fold_scores: pd.DataFrame,
) -> dict:
    return _to_python(
        {
            "method": metadata["method"],
            "scorer": metadata["scorer"],
            "selected_variant": metadata["selected_variant"],
            "selected_model": metadata["selected_model"],
            "selected_calibration": metadata["selected_calibration"],
            "selected_threshold": metadata["selected_threshold"],
            "development_sample_count": metadata["development_sample_count"],
            "cv_splits": metadata["cv_splits"],
            "permutation_repeats": metadata["permutation_repeats"],
            "feature_count": metadata["feature_count"],
            "detail_row_count": metadata["detail_row_count"],
            "holdout_used": metadata["holdout_used"],
            "mean_fold_roc_auc": fold_scores["baseline_roc_auc"].mean(),
            "std_fold_roc_auc": fold_scores["baseline_roc_auc"].std(),
            "features": summary.to_dict(orient="records"),
            "fold_scores": fold_scores.to_dict(orient="records"),
            "limitations": metadata["limitations"],
        }
    )


def _require_columns(frame: pd.DataFrame, expected_columns: list[str], label: str) -> None:
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected {label} columns")


def _require_row_count(frame: pd.DataFrame, expected_count: int, label: str) -> None:
    if len(frame) != expected_count:
        raise ValueError(f"Unexpected {label} row count")


def _require_no_nulls(frame: pd.DataFrame, label: str) -> None:
    if frame.isna().any().any():
        raise ValueError(f"Unexpected null values in {label}")


def _require_finite_numeric(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Unexpected non-numeric values in {label}")
        if not frame[column].map(math.isfinite).all():
            raise ValueError(f"Unexpected non-finite values in {label}")


def _numeric_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column != "feature_name"]


def _to_python(value):
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_to_python(item) for item in value]

    if hasattr(value, "item"):
        return value.item()

    return value
