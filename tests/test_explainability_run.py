import importlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import femhealth.explainability_run as explainability_run
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL


def _fake_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details = pd.DataFrame(
        {
            "fold": [1, 1],
            "feature_name": WDBC_FEATURE_NAMES[:2],
            "feature_position": [1, 2],
            "repeat": [1, 1],
            "importance": [0.2, -0.1],
            "baseline_roc_auc": [0.9, 0.9],
            "validation_sample_count": [91, 91],
            "validation_malignant_count": [34, 34],
            "validation_benign_count": [57, 57],
        }
    )
    summary = pd.DataFrame(
        {
            "rank": range(1, 31),
            "feature_name": WDBC_FEATURE_NAMES,
            "feature_position": range(1, 31),
            "mean_importance": [0.30 - index * 0.01 for index in range(30)],
            "std_importance": [0.01] * 30,
            "median_importance": [0.01] * 30,
            "min_importance": [-0.01] * 30,
            "max_importance": [0.03] * 30,
            "positive_fraction": [0.5] * 30,
            "fold_count": [5] * 30,
            "observation_count": [10] * 30,
        }
    )
    fold_scores = pd.DataFrame(
        {
            "fold": range(1, 6),
            "train_sample_count": [364] * 5,
            "validation_sample_count": [91] * 5,
            "train_malignant_count": [136] * 5,
            "train_benign_count": [228] * 5,
            "validation_malignant_count": [34] * 5,
            "validation_benign_count": [57] * 5,
            "baseline_roc_auc": [0.90, 0.91, 0.92, 0.93, 0.94],
        }
    )
    return details, summary, fold_scores


@pytest.fixture()
def split_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_development = pd.DataFrame(0.0, index=range(455), columns=WDBC_FEATURE_NAMES)
    y_development = pd.Series(
        [MALIGNANT_LABEL] * 170 + [BENIGN_LABEL] * 285,
        index=X_development.index,
        name="diagnosis",
    )
    X_holdout = pd.DataFrame(1.0, index=range(1000, 1114), columns=WDBC_FEATURE_NAMES)
    y_holdout = pd.Series(
        [MALIGNANT_LABEL] * 42 + [BENIGN_LABEL] * 72,
        index=X_holdout.index,
        name="diagnosis",
    )
    return X_development, y_development, X_holdout, y_holdout


def test_default_paths_and_names_are_expected() -> None:
    assert explainability_run.EXPLAINABILITY_DIRECTORY == Path("reports/explainability")
    assert explainability_run.DETAILS_FILENAME == "permutation_importance_details.csv"
    assert explainability_run.SUMMARY_FILENAME == "permutation_importance_summary.csv"
    assert explainability_run.FOLD_SCORES_FILENAME == "permutation_importance_fold_scores.csv"
    assert explainability_run.METADATA_FILENAME == "permutation_importance_metadata.json"
    assert explainability_run.PLOT_FILENAME == "permutation_importance_top15.png"


def test_run_explainability_once_persists_outputs_and_uses_only_development(
    monkeypatch,
    tmp_path,
    split_data,
) -> None:
    X_development, y_development, X_holdout, y_holdout = split_data
    calls = {"load": 0, "split": 0, "compute": 0}
    captured = {}
    X = pd.concat([X_development, X_holdout])
    y = pd.concat([y_development, y_holdout])
    details, summary, fold_scores = _fake_frames()

    def fake_load_wdbc_data():
        calls["load"] += 1
        return X, y

    def fake_split_development_test(loaded_X, loaded_y):
        calls["split"] += 1
        assert loaded_X is X
        assert loaded_y is y
        return X_development, X_holdout, y_development, y_holdout

    def fake_compute(received_X, received_y):
        calls["compute"] += 1
        captured["X"] = received_X
        captured["y"] = received_y
        return details, summary, fold_scores

    monkeypatch.setattr(explainability_run, "load_wdbc_data", fake_load_wdbc_data)
    monkeypatch.setattr(explainability_run, "split_development_test", fake_split_development_test)
    monkeypatch.setattr(
        explainability_run,
        "compute_cross_validated_permutation_importance",
        fake_compute,
    )

    returned_details, returned_summary, returned_fold_scores, metadata, paths = (
        explainability_run.run_explainability_once(tmp_path)
    )

    assert calls == {"load": 1, "split": 1, "compute": 1}
    assert captured["X"] is X_development
    assert captured["y"] is y_development
    assert set(captured["X"].index).isdisjoint(set(X_holdout.index))
    assert returned_details.equals(details)
    assert returned_summary.equals(summary)
    assert returned_fold_scores.equals(fold_scores)
    assert metadata["holdout_used"] is False
    assert metadata["detail_row_count"] == len(details)
    assert all(path.exists() for path in paths.values())
    assert paths["details"].read_text(encoding="utf-8")
    assert paths["summary"].read_text(encoding="utf-8")
    assert paths["fold_scores"].read_text(encoding="utf-8")
    assert json.loads(paths["metadata"].read_text(encoding="utf-8"))["holdout_used"] is False
    assert paths["plot"].stat().st_size > 0


def test_run_explainability_once_refuses_overwrite(tmp_path) -> None:
    existing_path = tmp_path / explainability_run.DETAILS_FILENAME
    existing_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Explainability outputs already exist"):
        explainability_run.run_explainability_once(tmp_path)


def test_run_explainability_once_removes_temporary_files_on_failure(
    monkeypatch,
    tmp_path,
    split_data,
) -> None:
    X_development, y_development, X_holdout, y_holdout = split_data
    details, summary, fold_scores = _fake_frames()

    monkeypatch.setattr(
        explainability_run,
        "load_wdbc_data",
        lambda: (X_development, y_development),
    )
    monkeypatch.setattr(
        explainability_run,
        "split_development_test",
        lambda loaded_X, loaded_y: (X_development, X_holdout, y_development, y_holdout),
    )
    monkeypatch.setattr(
        explainability_run,
        "compute_cross_validated_permutation_importance",
        lambda received_X, received_y: (details, summary, fold_scores),
    )

    def failing_plot(summary_frame, path):
        path.write_bytes(b"partial")
        raise RuntimeError("plot failed")

    monkeypatch.setattr(explainability_run, "_save_top15_plot", failing_plot)

    with pytest.raises(RuntimeError, match="plot failed"):
        explainability_run.run_explainability_once(tmp_path)

    assert not list(tmp_path.glob("*.tmp*"))
    assert not (tmp_path / explainability_run.DETAILS_FILENAME).exists()
    assert not (tmp_path / explainability_run.PLOT_FILENAME).exists()


def test_import_does_not_execute_analysis(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    importlib.reload(explainability_run)

    assert not Path("reports/explainability").exists()


def test_explainability_run_has_no_forbidden_operations() -> None:
    source = inspect.getsource(explainability_run)

    assert "evaluate_final_holdout" not in source
    assert "joblib.dump" not in source
    assert "build_and_persist_final_artifact" not in source
    assert "select_operating_points" not in source
    assert "build_threshold_table" not in source
    assert "MODEL_PATH" not in source
    assert "METADATA_PATH" not in source
