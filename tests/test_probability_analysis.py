import inspect

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import femhealth.probability_analysis as probability_analysis_module
from femhealth.data import load_wdbc_data
from femhealth.data_split import split_development_test
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL
from femhealth.probability_analysis import (
    CALIBRATION_METHOD,
    DEFAULT_THRESHOLDS,
    MIN_RECALL_MALIGNANT,
    PROBABILITY_VARIANTS,
    _predict_malignant_probabilities,
    build_calibration_table,
    build_probability_estimators,
    build_probability_quality_summary,
    build_threshold_table,
    generate_oof_malignant_probabilities,
    run_probability_analysis,
    select_operating_points,
)

EXPECTED_THRESHOLD_COLUMNS = [
    "variant",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "precision_malignant",
    "recall_malignant",
    "f1_malignant",
    "specificity_benign",
    "true_malignant",
    "false_negative_malignant",
    "false_positive_malignant",
    "true_benign",
]


@pytest.fixture(scope="module")
def development_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = load_wdbc_data()
    X_development, _, y_development, _ = split_development_test(X, y)
    return X_development, y_development


@pytest.fixture(scope="module")
def fast_oof_probabilities(
    development_data: tuple[pd.DataFrame, pd.Series],
) -> pd.DataFrame:
    X_development, y_development = development_data
    estimators = {
        "fast_logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
    }
    return generate_oof_malignant_probabilities(X_development, y_development, estimators)


def _synthetic_probabilities() -> tuple[pd.Series, pd.DataFrame]:
    y = pd.Series([MALIGNANT_LABEL, MALIGNANT_LABEL, BENIGN_LABEL, BENIGN_LABEL])
    probabilities = pd.DataFrame(
        {variant: [0.90, 0.40, 0.60, 0.10] for variant in PROBABILITY_VARIANTS},
        index=y.index,
    )
    return y, probabilities


def test_probability_variants_are_exact_and_ordered() -> None:
    assert PROBABILITY_VARIANTS == (
        "logistic_regression_native",
        "logistic_regression_sigmoid",
        "random_forest_native",
        "random_forest_sigmoid",
        "svm_sigmoid",
    )


def test_calibrated_variants_use_sigmoid_stratified_cv_and_no_ensemble() -> None:
    estimators = build_probability_estimators()

    for variant in ("logistic_regression_sigmoid", "random_forest_sigmoid", "svm_sigmoid"):
        estimator = estimators[variant]
        assert isinstance(estimator, CalibratedClassifierCV)
        assert estimator.method == CALIBRATION_METHOD
        assert estimator.cv.n_splits == 5
        assert estimator.cv.shuffle is True
        assert estimator.cv.random_state == 42
        assert estimator.ensemble is False
        assert estimator.n_jobs == -1


def test_native_and_calibrated_variants_do_not_share_estimators() -> None:
    estimators = build_probability_estimators()

    assert estimators["logistic_regression_native"] is not (
        estimators["logistic_regression_sigmoid"].estimator
    )
    assert estimators["random_forest_native"] is not estimators["random_forest_sigmoid"].estimator


def test_oof_probabilities_preserve_index_and_record_count(
    development_data: tuple[pd.DataFrame, pd.Series],
    fast_oof_probabilities: pd.DataFrame,
) -> None:
    _, y_development = development_data

    assert fast_oof_probabilities.index.equals(y_development.index)
    assert fast_oof_probabilities.shape == (455, 1)


def test_oof_probabilities_are_between_zero_and_one(
    fast_oof_probabilities: pd.DataFrame,
) -> None:
    assert fast_oof_probabilities.notna().all().all()
    assert fast_oof_probabilities.apply(lambda column: column.between(0, 1).all()).all()


def test_oof_generation_does_not_modify_development_data(
    development_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_development, y_development = development_data
    original_X = X_development.copy(deep=True)
    original_y = y_development.copy(deep=True)

    estimators = {
        "fast_logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
    }
    generate_oof_malignant_probabilities(X_development, y_development, estimators)

    assert X_development.equals(original_X)
    assert y_development.equals(original_y)


def test_predict_malignant_probabilities_uses_explicit_class_column() -> None:
    class ProbabilityEstimator:
        classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]

        def predict_proba(self, X: pd.DataFrame):
            return pd.DataFrame([[0.80, 0.20], [0.25, 0.75]], index=X.index).to_numpy()

    X = pd.DataFrame({"feature": [1.0, 2.0]})

    scores = _predict_malignant_probabilities(ProbabilityEstimator(), X)

    assert list(scores) == [0.20, 0.75]


def test_probability_quality_summary_has_valid_values() -> None:
    y, probabilities = _synthetic_probabilities()

    summary = build_probability_quality_summary(y, probabilities)

    assert summary.columns.tolist() == [
        "variant",
        "brier_score",
        "log_loss",
        "roc_auc_malignant",
        "average_precision_malignant",
    ]
    assert summary["variant"].tolist() == list(PROBABILITY_VARIANTS)
    assert summary.drop(columns="variant").apply(lambda column: column.between(0, 1).all()).all()


def test_calibration_table_has_probabilities_between_zero_and_one() -> None:
    y, probabilities = _synthetic_probabilities()

    calibration = build_calibration_table(y, probabilities, n_bins=2)

    assert set(calibration.columns) == {
        "variant",
        "bin",
        "mean_predicted_probability",
        "observed_malignant_fraction",
    }
    assert calibration["mean_predicted_probability"].between(0, 1).all()
    assert calibration["observed_malignant_fraction"].between(0, 1).all()


def test_default_thresholds_include_point_fifty() -> None:
    assert DEFAULT_THRESHOLDS[0] == 0.05
    assert DEFAULT_THRESHOLDS[-1] == 0.95
    assert 0.50 in DEFAULT_THRESHOLDS


def test_threshold_rule_predicts_malignant_when_probability_reaches_threshold() -> None:
    y = pd.Series([MALIGNANT_LABEL, BENIGN_LABEL])
    probabilities = pd.DataFrame({"variant": [0.50, 0.49]}, index=y.index)

    thresholds = build_threshold_table(y, probabilities, thresholds=(0.50,))
    row = thresholds.iloc[0]

    assert row["true_malignant"] == 1
    assert row["true_benign"] == 1


def test_threshold_counts_sum_to_total_and_metrics_are_correct() -> None:
    y, probabilities = _synthetic_probabilities()

    thresholds = build_threshold_table(y, probabilities, thresholds=(0.50,))

    assert thresholds.columns.tolist() == EXPECTED_THRESHOLD_COLUMNS
    count_columns = [
        "true_malignant",
        "false_negative_malignant",
        "false_positive_malignant",
        "true_benign",
    ]
    assert (thresholds[count_columns].sum(axis=1) == len(y)).all()
    assert (thresholds["recall_malignant"] == 0.5).all()
    assert (thresholds["specificity_benign"] == 0.5).all()


def test_select_operating_points_respects_recall_minimum_and_order() -> None:
    rows = []
    for variant in PROBABILITY_VARIANTS:
        rows.extend(
            [
                {
                    "variant": variant,
                    "threshold": 0.40,
                    "recall_malignant": MIN_RECALL_MALIGNANT,
                    "specificity_benign": 0.80,
                    "f1_malignant": 0.70,
                    "precision_malignant": 0.70,
                },
                {
                    "variant": variant,
                    "threshold": 0.50,
                    "recall_malignant": MIN_RECALL_MALIGNANT,
                    "specificity_benign": 0.80,
                    "f1_malignant": 0.70,
                    "precision_malignant": 0.70,
                },
            ]
        )
    threshold_table = pd.DataFrame(rows)

    selected = select_operating_points(threshold_table)

    assert selected["variant"].tolist() == list(PROBABILITY_VARIANTS)
    assert (selected["threshold"] == 0.50).all()


def test_select_operating_points_raises_when_recall_minimum_is_not_met() -> None:
    threshold_table = pd.DataFrame(
        {
            "variant": PROBABILITY_VARIANTS,
            "threshold": [0.50] * len(PROBABILITY_VARIANTS),
            "recall_malignant": [0.50] * len(PROBABILITY_VARIANTS),
            "specificity_benign": [1.0] * len(PROBABILITY_VARIANTS),
            "f1_malignant": [0.50] * len(PROBABILITY_VARIANTS),
            "precision_malignant": [0.50] * len(PROBABILITY_VARIANTS),
        }
    )

    with pytest.raises(ValueError, match="No threshold reaches recall minimum"):
        select_operating_points(threshold_table)


def test_run_probability_analysis_returns_expected_keys_with_injected_fast_estimator(
    development_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X_development, y_development = development_data
    placeholder = pd.DataFrame(index=y_development.index)

    monkeypatch.setattr(
        probability_analysis_module,
        "generate_oof_malignant_probabilities",
        lambda X, y: placeholder,
    )
    monkeypatch.setattr(
        probability_analysis_module,
        "build_probability_quality_summary",
        lambda y, probabilities: placeholder,
    )
    monkeypatch.setattr(
        probability_analysis_module,
        "build_calibration_table",
        lambda y, probabilities: placeholder,
    )
    monkeypatch.setattr(
        probability_analysis_module,
        "build_threshold_table",
        lambda y, probabilities: placeholder,
    )
    monkeypatch.setattr(
        probability_analysis_module,
        "select_operating_points",
        lambda thresholds: placeholder,
    )

    results = run_probability_analysis(X_development, y_development)

    assert list(results) == [
        "oof_probabilities",
        "probability_quality",
        "calibration",
        "thresholds",
        "operating_points",
    ]


def test_probability_analysis_module_does_not_use_final_holdout_data() -> None:
    module_source = inspect.getsource(probability_analysis_module)

    assert "split_development_test" not in module_source
    assert "data_split" not in module_source
    assert "X_test" not in module_source
    assert "y_test" not in module_source
