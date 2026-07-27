import inspect
import json
import math
from pathlib import Path

import pandas as pd
import pytest

import femhealth.explainability_artifacts as explainability_artifacts_module
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.explainability_artifacts import (
    DETAILS_PATH,
    FOLD_SCORES_PATH,
    METADATA_PATH,
    PLOT_PATH,
    PNG_SIGNATURE,
    SUMMARY_PATH,
    load_explainability_artifacts,
)


def test_default_paths_are_expected() -> None:
    assert DETAILS_PATH == Path("reports/explainability/permutation_importance_details.csv")
    assert SUMMARY_PATH == Path("reports/explainability/permutation_importance_summary.csv")
    assert FOLD_SCORES_PATH == Path(
        "reports/explainability/permutation_importance_fold_scores.csv"
    )
    assert METADATA_PATH == Path("reports/explainability/permutation_importance_metadata.json")
    assert PLOT_PATH == Path("reports/explainability/permutation_importance_top15.png")


def test_missing_files_are_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing explainability artifact"):
        load_explainability_artifacts(
            details_path=tmp_path / "details.csv",
            summary_path=tmp_path / "summary.csv",
            fold_scores_path=tmp_path / "folds.csv",
            metadata_path=tmp_path / "metadata.json",
            plot_path=tmp_path / "plot.png",
        )


def test_valid_artifacts_return_payload_and_png(tmp_path) -> None:
    paths = _write_valid_artifacts(tmp_path)

    payload, plot_bytes = load_explainability_artifacts(**paths)

    assert payload["feature_count"] == 30
    assert len(payload["features"]) == 30
    assert len(payload["fold_scores"]) == 5
    assert payload["mean_fold_roc_auc"] == pytest.approx(0.93)
    expected_std = pd.Series([0.91, 0.92, 0.93, 0.94, 0.95]).std()
    assert payload["std_fold_roc_auc"] == pytest.approx(expected_std)
    assert plot_bytes == PNG_SIGNATURE + b"synthetic-png"
    json.dumps(payload)


@pytest.mark.parametrize(
    ("frame_name", "expected_error"),
    [
        ("details_short", "Unexpected details row count"),
        ("summary_short", "Unexpected summary row count"),
        ("summary_ranks", "Unexpected summary ranks"),
        ("summary_order", "Unexpected summary importance order"),
        ("summary_position", "Unexpected feature position"),
        ("summary_observations", "Unexpected observation count"),
        ("metadata_variant", "Unexpected metadata value: selected_variant"),
        ("metadata_holdout", "Unexpected metadata value: holdout_used"),
        ("details_non_finite", "Unexpected non-finite values in details"),
    ],
)
def test_invalid_artifacts_are_rejected(tmp_path, frame_name, expected_error) -> None:
    paths = _write_valid_artifacts(tmp_path)

    if frame_name == "details_short":
        details = pd.read_csv(paths["details_path"]).iloc[:-1]
        details.to_csv(paths["details_path"], index=False)
    elif frame_name == "summary_short":
        summary = pd.read_csv(paths["summary_path"]).iloc[:-1]
        summary.to_csv(paths["summary_path"], index=False)
    elif frame_name == "summary_ranks":
        summary = pd.read_csv(paths["summary_path"])
        summary.loc[0, "rank"] = 2
        summary.to_csv(paths["summary_path"], index=False)
    elif frame_name == "summary_order":
        summary = pd.read_csv(paths["summary_path"])
        summary.loc[0, "mean_importance"] = -1.0
        summary.to_csv(paths["summary_path"], index=False)
    elif frame_name == "summary_position":
        summary = pd.read_csv(paths["summary_path"])
        summary.loc[0, "feature_position"] = 2
        summary.to_csv(paths["summary_path"], index=False)
    elif frame_name == "summary_observations":
        summary = pd.read_csv(paths["summary_path"])
        summary.loc[0, "observation_count"] = 49
        summary.to_csv(paths["summary_path"], index=False)
    elif frame_name == "metadata_variant":
        metadata = _metadata()
        metadata["selected_variant"] = "other"
        paths["metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")
    elif frame_name == "metadata_holdout":
        metadata = _metadata()
        metadata["holdout_used"] = True
        paths["metadata_path"].write_text(json.dumps(metadata), encoding="utf-8")
    elif frame_name == "details_non_finite":
        details = pd.read_csv(paths["details_path"])
        details.loc[0, "importance"] = math.inf
        details.to_csv(paths["details_path"], index=False)

    with pytest.raises(ValueError, match=expected_error):
        load_explainability_artifacts(**paths)


def test_empty_png_is_rejected(tmp_path) -> None:
    paths = _write_valid_artifacts(tmp_path)
    paths["plot_path"].write_bytes(b"")

    with pytest.raises(ValueError, match="Empty explainability plot"):
        load_explainability_artifacts(**paths)


def test_invalid_png_signature_is_rejected(tmp_path) -> None:
    paths = _write_valid_artifacts(tmp_path)
    paths["plot_path"].write_bytes(b"not-png")

    with pytest.raises(ValueError, match="Invalid explainability plot signature"):
        load_explainability_artifacts(**paths)


def test_loader_module_does_not_load_dataset_or_model() -> None:
    source = inspect.getsource(explainability_artifacts_module)
    forbidden_terms = [
        "load_wdbc_data",
        "load_model_artifact",
        "joblib",
        ".fit(",
        "predict_proba",
        "permutation_importance(",
        "evaluate_final_holdout",
    ]

    for term in forbidden_terms:
        assert term not in source


def _write_valid_artifacts(tmp_path: Path) -> dict[str, Path]:
    details_path = tmp_path / "details.csv"
    summary_path = tmp_path / "summary.csv"
    fold_scores_path = tmp_path / "fold_scores.csv"
    metadata_path = tmp_path / "metadata.json"
    plot_path = tmp_path / "plot.png"

    _details().to_csv(details_path, index=False)
    _summary().to_csv(summary_path, index=False)
    _fold_scores().to_csv(fold_scores_path, index=False)
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    plot_path.write_bytes(PNG_SIGNATURE + b"synthetic-png")

    return {
        "details_path": details_path,
        "summary_path": summary_path,
        "fold_scores_path": fold_scores_path,
        "metadata_path": metadata_path,
        "plot_path": plot_path,
    }


def _details() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for feature_position, feature_name in enumerate(WDBC_FEATURE_NAMES, start=1):
            for repeat in range(1, 11):
                rows.append(
                    {
                        "fold": fold,
                        "feature_name": feature_name,
                        "feature_position": feature_position,
                        "repeat": repeat,
                        "importance": _importance(feature_name, repeat),
                        "baseline_roc_auc": 0.90 + fold / 100,
                        "validation_sample_count": 91,
                        "validation_malignant_count": 34,
                        "validation_benign_count": 57,
                    }
                )

    return pd.DataFrame(rows)


def _summary() -> pd.DataFrame:
    rows = []
    for rank, feature_name in enumerate(WDBC_FEATURE_NAMES, start=1):
        rows.append(
            {
                "rank": rank,
                "feature_name": feature_name,
                "feature_position": WDBC_FEATURE_NAMES.index(feature_name) + 1,
                "mean_importance": 1.0 / rank,
                "std_importance": 0.01,
                "median_importance": 1.0 / rank,
                "min_importance": -0.001,
                "max_importance": 0.02,
                "positive_fraction": 0.8,
                "fold_count": 5,
                "observation_count": 50,
            }
        )

    return pd.DataFrame(rows)


def _importance(feature_name: str, repeat: int) -> float:
    if feature_name == WDBC_FEATURE_NAMES[-1] and repeat == 1:
        return -0.001

    return 0.001


def _fold_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [1, 2, 3, 4, 5],
            "train_sample_count": [364] * 5,
            "validation_sample_count": [91] * 5,
            "train_malignant_count": [136] * 5,
            "train_benign_count": [228] * 5,
            "validation_malignant_count": [34] * 5,
            "validation_benign_count": [57] * 5,
            "baseline_roc_auc": [0.91, 0.92, 0.93, 0.94, 0.95],
        }
    )


def _metadata() -> dict:
    return {
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
        "feature_names": WDBC_FEATURE_NAMES,
        "limitations": ["Limitação sintética para teste."],
    }
