import json
from pathlib import Path

import pandas as pd
import pytest

from femhealth.data import WDBC_FEATURE_NAMES

EXPLAINABILITY_DIRECTORY = Path("reports/explainability")
DETAILS_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_details.csv"
SUMMARY_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_summary.csv"
FOLD_SCORES_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_fold_scores.csv"
METADATA_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_metadata.json"
PLOT_PATH = EXPLAINABILITY_DIRECTORY / "permutation_importance_top15.png"

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
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def explainability_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    details = pd.read_csv(DETAILS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    fold_scores = pd.read_csv(FOLD_SCORES_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return details, summary, fold_scores, metadata


def test_committed_explainability_files_exist() -> None:
    for path in [DETAILS_PATH, SUMMARY_PATH, FOLD_SCORES_PATH, METADATA_PATH, PLOT_PATH]:
        assert path.exists()


def test_committed_explainability_csv_shapes_and_columns(explainability_artifacts) -> None:
    details, summary, fold_scores, _ = explainability_artifacts

    assert len(details) == 1500
    assert len(summary) == 30
    assert len(fold_scores) == 5
    assert details.columns.to_list() == DETAIL_COLUMNS
    assert summary.columns.to_list() == SUMMARY_COLUMNS
    assert fold_scores.columns.to_list() == FOLD_SCORE_COLUMNS


def test_committed_explainability_summary_integrity(explainability_artifacts) -> None:
    _, summary, _, _ = explainability_artifacts
    canonical_positions = {
        feature_name: position for position, feature_name in enumerate(WDBC_FEATURE_NAMES, start=1)
    }

    assert summary["rank"].tolist() == list(range(1, 31))
    assert summary["mean_importance"].is_monotonic_decreasing
    assert summary["fold_count"].eq(5).all()
    assert summary["observation_count"].eq(50).all()

    for row in summary.itertuples(index=False):
        assert row.feature_position == canonical_positions[row.feature_name]


def test_committed_explainability_details_integrity(explainability_artifacts) -> None:
    details, _, fold_scores, _ = explainability_artifacts

    assert sorted(details["fold"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert fold_scores["fold"].tolist() == [1, 2, 3, 4, 5]
    assert details["importance"].lt(0).any()


def test_committed_explainability_metadata_integrity(explainability_artifacts) -> None:
    _, _, _, metadata = explainability_artifacts

    assert metadata["method"] == "cross_validated_permutation_importance"
    assert metadata["scorer"] == "roc_auc_malignant"
    assert metadata["selected_variant"] == "svm_sigmoid"
    assert metadata["selected_threshold"] == 0.51
    assert metadata["development_sample_count"] == 455
    assert metadata["cv_splits"] == 5
    assert metadata["permutation_repeats"] == 10
    assert metadata["feature_count"] == 30
    assert metadata["detail_row_count"] == 1500
    assert metadata["holdout_used"] is False
    assert metadata["final_model_artifact_modified"] is False
    assert metadata["feature_names"] == WDBC_FEATURE_NAMES


def test_committed_explainability_png_is_valid_png() -> None:
    assert PLOT_PATH.stat().st_size > 0
    assert PLOT_PATH.read_bytes()[:8] == PNG_SIGNATURE
