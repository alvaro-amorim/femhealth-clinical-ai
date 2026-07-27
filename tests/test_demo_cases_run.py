import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

import femhealth.demo_cases_run as demo_cases_run
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.demo_cases_artifact import (
    DEMO_BENIGN_CASE_COUNT,
    DEMO_CASE_COUNT,
    DEMO_EXPECTED_REFERENCE_LABELS,
    DEMO_EXPECTED_SAMPLE_INDICES,
    DEMO_FORBIDDEN_CASE_KEYS,
    DEMO_MALIGNANT_CASE_COUNT,
    DEMO_OFFICIAL_HOLDOUT_ACCURACY,
)


def test_select_demo_prediction_rows_uses_head_without_filtering() -> None:
    predictions = _synthetic_predictions()
    predictions.loc[:7, "correct"] = False
    predictions.loc[:7, "error_type"] = "kept_without_filtering"

    selected = demo_cases_run.select_demo_prediction_rows(predictions)

    assert selected["sample_index"].tolist() == DEMO_EXPECTED_SAMPLE_INDICES
    assert selected["correct"].tolist() == [False] * DEMO_CASE_COUNT
    assert selected["error_type"].tolist() == ["kept_without_filtering"] * DEMO_CASE_COUNT


def test_selected_prediction_rows_validate_indices_labels_distribution_and_order() -> None:
    selected = demo_cases_run.select_demo_prediction_rows(_synthetic_predictions())

    demo_cases_run._validate_selected_prediction_rows(selected)

    assert selected["sample_index"].tolist() == DEMO_EXPECTED_SAMPLE_INDICES
    assert selected["true_label"].tolist() == [
        DEMO_EXPECTED_REFERENCE_LABELS[index] for index in DEMO_EXPECTED_SAMPLE_INDICES
    ]
    assert selected["true_label"].tolist().count(0) == DEMO_MALIGNANT_CASE_COUNT
    assert selected["true_label"].tolist().count(1) == DEMO_BENIGN_CASE_COUNT


def test_selected_prediction_rows_reject_indices_out_of_order() -> None:
    predictions = _synthetic_predictions(
        sample_indices=list(reversed(DEMO_EXPECTED_SAMPLE_INDICES))
    )
    selected = demo_cases_run.select_demo_prediction_rows(predictions)

    with pytest.raises(ValueError, match="sample indices"):
        demo_cases_run._validate_selected_prediction_rows(selected)


def test_run_demo_cases_once_writes_payload_from_holdout_without_predictions(tmp_path) -> None:
    predictions_path = _write_predictions(tmp_path)
    summary_path = _write_summary(tmp_path)
    output_path = tmp_path / "holdout_demo_cases.json"
    X_development, X_holdout, y_development, y_holdout = _synthetic_split()

    payload = demo_cases_run.run_demo_cases_once(
        predictions_path=predictions_path,
        final_summary_path=summary_path,
        output_path=output_path,
        data_loader=lambda: (pd.DataFrame(), pd.Series(dtype=int)),
        splitter=lambda X, y: (X_development, X_holdout, y_development, y_holdout),
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == payload
    assert output_path.exists()
    assert not output_path.with_name(f".{output_path.name}.tmp").exists()
    assert payload["sample_indices"] == DEMO_EXPECTED_SAMPLE_INDICES
    assert payload["malignant_case_count"] == DEMO_MALIGNANT_CASE_COUNT
    assert payload["benign_case_count"] == DEMO_BENIGN_CASE_COUNT
    assert len(payload["cases"]) == DEMO_CASE_COUNT

    for case in payload["cases"]:
        assert not (set(case) & DEMO_FORBIDDEN_CASE_KEYS)
        assert list(case["features"]) == WDBC_FEATURE_NAMES
        assert len(case["features"]) == 30
        assert case["reference_label"] == DEMO_EXPECTED_REFERENCE_LABELS[case["sample_index"]]


def test_run_demo_cases_once_rejects_demo_indices_outside_holdout(tmp_path) -> None:
    predictions_path = _write_predictions(tmp_path)
    summary_path = _write_summary(tmp_path)
    output_path = tmp_path / "holdout_demo_cases.json"
    X_development, X_holdout, y_development, y_holdout = _synthetic_split()
    X_holdout = X_holdout.drop(index=DEMO_EXPECTED_SAMPLE_INDICES[0])
    y_holdout = y_holdout.drop(index=DEMO_EXPECTED_SAMPLE_INDICES[0])

    with pytest.raises(ValueError, match="belong to the final holdout"):
        demo_cases_run.run_demo_cases_once(
            predictions_path=predictions_path,
            final_summary_path=summary_path,
            output_path=output_path,
            data_loader=lambda: (pd.DataFrame(), pd.Series(dtype=int)),
            splitter=lambda X, y: (X_development, X_holdout, y_development, y_holdout),
        )

    assert not output_path.exists()


def test_run_demo_cases_once_rejects_demo_indices_in_development(tmp_path) -> None:
    predictions_path = _write_predictions(tmp_path)
    summary_path = _write_summary(tmp_path)
    output_path = tmp_path / "holdout_demo_cases.json"
    X_development, X_holdout, y_development, y_holdout = _synthetic_split()
    duplicate_index = DEMO_EXPECTED_SAMPLE_INDICES[0]
    X_development.loc[duplicate_index] = X_holdout.loc[duplicate_index]
    y_development.loc[duplicate_index] = DEMO_EXPECTED_REFERENCE_LABELS[duplicate_index]

    with pytest.raises(ValueError, match="must not belong to development"):
        demo_cases_run.run_demo_cases_once(
            predictions_path=predictions_path,
            final_summary_path=summary_path,
            output_path=output_path,
            data_loader=lambda: (pd.DataFrame(), pd.Series(dtype=int)),
            splitter=lambda X, y: (X_development, X_holdout, y_development, y_holdout),
        )


def test_run_demo_cases_once_rejects_wrong_holdout_label(tmp_path) -> None:
    predictions_path = _write_predictions(tmp_path)
    summary_path = _write_summary(tmp_path)
    output_path = tmp_path / "holdout_demo_cases.json"
    X_development, X_holdout, y_development, y_holdout = _synthetic_split()
    y_holdout.loc[DEMO_EXPECTED_SAMPLE_INDICES[0]] = 1

    with pytest.raises(ValueError, match="reference label"):
        demo_cases_run.run_demo_cases_once(
            predictions_path=predictions_path,
            final_summary_path=summary_path,
            output_path=output_path,
            data_loader=lambda: (pd.DataFrame(), pd.Series(dtype=int)),
            splitter=lambda X, y: (X_development, X_holdout, y_development, y_holdout),
        )


def test_run_demo_cases_once_protects_against_overwrite(tmp_path) -> None:
    output_path = tmp_path / "holdout_demo_cases.json"
    output_path.write_text("already exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        demo_cases_run.run_demo_cases_once(output_path=output_path)


def test_write_json_atomically_removes_temporary_file_when_dump_fails(
    monkeypatch,
    tmp_path,
) -> None:
    output_path = tmp_path / "holdout_demo_cases.json"

    def fail_dump(*args, **kwargs):
        raise RuntimeError("cannot serialize")

    monkeypatch.setattr(demo_cases_run.json, "dumps", fail_dump)

    with pytest.raises(RuntimeError, match="cannot serialize"):
        demo_cases_run._write_json_atomically({"payload": True}, output_path)

    assert not output_path.exists()
    assert not output_path.with_name(f".{output_path.name}.tmp").exists()


def test_write_json_atomically_uses_temporary_replace() -> None:
    source = inspect.getsource(demo_cases_run._write_json_atomically)

    assert ".replace(" in source


def test_demo_cases_run_source_has_no_model_inference_training_or_metric_recalculation() -> None:
    source = inspect.getsource(demo_cases_run)
    forbidden_terms = [
        "joblib",
        "load_model_artifact",
        "predict_with_artifact",
        "predict_proba",
        ".fit(",
        "evaluate_final_holdout",
        "accuracy_score",
        "balanced_accuracy_score",
        "precision_score",
        "recall_score",
        "roc_auc_score",
        "log_loss",
    ]

    for term in forbidden_terms:
        assert term not in source


def _synthetic_predictions(
    sample_indices: list[int] | None = None,
) -> pd.DataFrame:
    indices = DEMO_EXPECTED_SAMPLE_INDICES if sample_indices is None else sample_indices
    extra_indices = [999, 998]
    rows = []
    for sample_index in [*indices, *extra_indices]:
        true_label = DEMO_EXPECTED_REFERENCE_LABELS.get(sample_index, 0)
        rows.append(
            {
                "sample_index": sample_index,
                "true_label": true_label,
                "probability_malignant": 0.25,
                "predicted_label": true_label,
                "correct": True,
                "error_type": "correct",
            }
        )

    return pd.DataFrame(rows)


def _write_predictions(tmp_path: Path) -> Path:
    path = tmp_path / "final_holdout_predictions.csv"
    _synthetic_predictions().to_csv(path, index=False)
    return path


def _write_summary(tmp_path: Path) -> Path:
    path = tmp_path / "final_holdout_summary.json"
    path.write_text(
        json.dumps({"accuracy": DEMO_OFFICIAL_HOLDOUT_ACCURACY}),
        encoding="utf-8",
    )
    return path


def _synthetic_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_development = pd.DataFrame(
        [[1.0] * len(WDBC_FEATURE_NAMES), [2.0] * len(WDBC_FEATURE_NAMES)],
        index=[10, 11],
        columns=WDBC_FEATURE_NAMES,
    )
    y_development = pd.Series([0, 1], index=X_development.index, name="diagnosis")

    X_holdout = pd.DataFrame(
        [
            [float(sample_index + feature_position / 100) for feature_position in range(30)]
            for sample_index in DEMO_EXPECTED_SAMPLE_INDICES
        ],
        index=DEMO_EXPECTED_SAMPLE_INDICES,
        columns=WDBC_FEATURE_NAMES,
    )
    y_holdout = pd.Series(
        [DEMO_EXPECTED_REFERENCE_LABELS[index] for index in DEMO_EXPECTED_SAMPLE_INDICES],
        index=DEMO_EXPECTED_SAMPLE_INDICES,
        name="diagnosis",
    )

    return X_development, X_holdout, y_development, y_holdout
