import importlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import femhealth.final_run as final_run
import femhealth.final_selection as final_selection
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.final_selection import SELECTED_THRESHOLD, SELECTED_VARIANT


def _fake_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selected_variant": SELECTED_VARIANT,
                "threshold": SELECTED_THRESHOLD,
                "test_sample_count": 2,
                "accuracy": 0.5,
                "true_malignant": 1,
                "false_negative_malignant": 0,
                "false_positive_malignant": 1,
                "true_benign": 0,
            }
        ]
    )


def _fake_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_label": [0, 1],
            "probability_malignant": [0.80, 0.60],
            "predicted_label": [0, 0],
            "correct": [True, False],
            "error_type": ["correct", "false_positive_malignant"],
        },
        index=pd.Index([10, 20], name=None),
    )


def test_default_result_paths_are_expected() -> None:
    assert final_run.FINAL_RESULTS_DIRECTORY == Path("reports/results")
    assert final_run.FINAL_SUMMARY_PATH == Path("reports/results/final_holdout_summary.json")
    assert final_run.FINAL_PREDICTIONS_PATH == Path(
        "reports/results/final_holdout_predictions.csv"
    )


def test_run_final_holdout_once_writes_summary_and_predictions(monkeypatch, tmp_path) -> None:
    calls = {
        "load": 0,
        "split": 0,
        "evaluate": 0,
    }
    summary = _fake_summary()
    predictions = _fake_predictions()
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([0, 1], name="diagnosis")
    X_development = X.iloc[[0]]
    X_test = X.iloc[[1]]
    y_development = y.iloc[[0]]
    y_test = y.iloc[[1]]

    def fake_load_wdbc_data():
        calls["load"] += 1
        return X, y

    def fake_split_development_test(loaded_X, loaded_y):
        calls["split"] += 1
        assert loaded_X is X
        assert loaded_y is y
        return X_development, X_test, y_development, y_test

    def fake_evaluate_final_holdout(split_X_dev, split_y_dev, split_X_test, split_y_test):
        calls["evaluate"] += 1
        assert split_X_dev is X_development
        assert split_y_dev is y_development
        assert split_X_test is X_test
        assert split_y_test is y_test
        return summary, predictions, object()

    monkeypatch.setattr(final_run, "load_wdbc_data", fake_load_wdbc_data)
    monkeypatch.setattr(final_run, "split_development_test", fake_split_development_test)
    monkeypatch.setattr(final_run, "evaluate_final_holdout", fake_evaluate_final_holdout)

    returned_summary, returned_predictions = final_run.run_final_holdout_once(tmp_path)

    summary_path = tmp_path / final_run.FINAL_SUMMARY_PATH.name
    predictions_path = tmp_path / final_run.FINAL_PREDICTIONS_PATH.name
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    written_predictions = pd.read_csv(predictions_path)

    assert calls == {"load": 1, "split": 1, "evaluate": 1}
    assert summary_path.exists()
    assert predictions_path.exists()
    assert written_summary == summary.iloc[0].to_dict()
    assert isinstance(written_summary, dict)
    assert "sample_index" in written_predictions.columns
    assert not set(WDBC_FEATURE_NAMES).intersection(written_predictions.columns)
    assert returned_summary.equals(summary)
    assert returned_predictions.equals(predictions)
    assert not list(tmp_path.rglob("*.joblib"))
    assert not list(tmp_path.rglob("*.pkl"))


def test_run_final_holdout_once_refuses_to_overwrite(monkeypatch, tmp_path) -> None:
    summary_path = tmp_path / final_run.FINAL_SUMMARY_PATH.name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    def fail_if_called():
        raise AssertionError("load_wdbc_data must not be called when results exist")

    monkeypatch.setattr(final_run, "load_wdbc_data", fail_if_called)

    with pytest.raises(FileExistsError, match="Final holdout results already exist"):
        final_run.run_final_holdout_once(tmp_path)


def test_import_does_not_execute_evaluation(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    importlib.reload(final_run)

    assert not Path("reports/results/final_holdout_summary.json").exists()
    assert not Path("reports/results/final_holdout_predictions.csv").exists()


def test_module_does_not_change_frozen_selection() -> None:
    module_source = inspect.getsource(final_run)

    assert "SELECTED_VARIANT =" not in module_source
    assert "SELECTED_THRESHOLD =" not in module_source
    assert final_selection.SELECTED_VARIANT == SELECTED_VARIANT
    assert final_selection.SELECTED_THRESHOLD == SELECTED_THRESHOLD
