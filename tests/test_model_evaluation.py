import inspect

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import femhealth.model_evaluation as model_evaluation_module
from femhealth.data import load_wdbc_data
from femhealth.data_split import split_development_test
from femhealth.model_evaluation import (
    BENIGN_LABEL,
    CV_SPLITS,
    MALIGNANT_LABEL,
    RANDOM_STATE,
    _get_malignant_scores,
    build_stratified_cv,
    evaluate_baseline_candidates,
)
from femhealth.model_pipelines import build_candidate_pipelines

EXPECTED_COLUMNS = [
    "model",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "precision_malignant_mean",
    "precision_malignant_std",
    "recall_malignant_mean",
    "recall_malignant_std",
    "f1_malignant_mean",
    "f1_malignant_std",
    "specificity_benign_mean",
    "specificity_benign_std",
    "roc_auc_malignant_mean",
    "roc_auc_malignant_std",
    "average_precision_malignant_mean",
    "average_precision_malignant_std",
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
def baseline_results(development_data: tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    X_development, y_development = development_data
    return evaluate_baseline_candidates(X_development, y_development)


def test_build_stratified_cv_configuration() -> None:
    cv = build_stratified_cv()

    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == CV_SPLITS
    assert cv.shuffle is True
    assert cv.random_state == RANDOM_STATE


def test_evaluate_baseline_candidates_returns_exact_candidates(
    baseline_results: pd.DataFrame,
) -> None:
    assert set(baseline_results["model"]) == set(build_candidate_pipelines())


def test_evaluate_baseline_candidates_has_expected_columns(
    baseline_results: pd.DataFrame,
) -> None:
    assert list(baseline_results.columns) == EXPECTED_COLUMNS


def test_metric_means_are_between_zero_and_one(baseline_results: pd.DataFrame) -> None:
    mean_columns = [column for column in baseline_results.columns if column.endswith("_mean")]

    for column in mean_columns:
        assert baseline_results[column].between(0, 1).all()


def test_metric_standard_deviations_are_non_negative(baseline_results: pd.DataFrame) -> None:
    std_columns = [column for column in baseline_results.columns if column.endswith("_std")]

    for column in std_columns:
        assert (baseline_results[column] >= 0).all()


def test_confusion_counts_sum_to_development_size(baseline_results: pd.DataFrame) -> None:
    count_columns = [
        "true_malignant",
        "false_negative_malignant",
        "false_positive_malignant",
        "true_benign",
    ]

    assert (baseline_results[count_columns].sum(axis=1) == 455).all()


def test_malignant_confusion_counts_sum_to_malignant_total(
    baseline_results: pd.DataFrame,
) -> None:
    assert (
        baseline_results["true_malignant"] + baseline_results["false_negative_malignant"]
        == 170
    ).all()


def test_benign_confusion_counts_sum_to_benign_total(baseline_results: pd.DataFrame) -> None:
    assert (
        baseline_results["false_positive_malignant"] + baseline_results["true_benign"] == 285
    ).all()


def test_evaluate_baseline_candidates_does_not_modify_development_data(
    development_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X_development, y_development = development_data
    original_X = X_development.copy(deep=True)
    original_y = y_development.copy(deep=True)

    evaluate_baseline_candidates(X_development, y_development)

    assert X_development.equals(original_X)
    assert y_development.equals(original_y)


def test_candidate_order_is_preserved(baseline_results: pd.DataFrame) -> None:
    assert baseline_results["model"].tolist() == list(build_candidate_pipelines())


def test_model_evaluation_does_not_use_final_holdout_data() -> None:
    module_source = inspect.getsource(model_evaluation_module)

    assert "split_development_test" not in module_source
    assert "data_split" not in module_source
    assert "X_test" not in module_source
    assert "y_test" not in module_source


def test_malignant_continuous_score_orientation() -> None:
    class DecisionOnlyEstimator:
        classes_ = [MALIGNANT_LABEL, BENIGN_LABEL]

        def decision_function(self, X: pd.DataFrame):
            return pd.Series([2.0, -1.0], index=X.index)

    X = pd.DataFrame({"feature": [10.0, 20.0]})

    scores = _get_malignant_scores(DecisionOnlyEstimator(), X)

    assert list(scores) == [-2.0, 1.0]


def test_malignant_continuous_score_uses_explicit_predict_proba_class_column() -> None:
    class ProbabilityEstimator:
        classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]

        def predict_proba(self, X: pd.DataFrame):
            return pd.DataFrame(
                [
                    [0.80, 0.20],
                    [0.25, 0.75],
                ],
                index=X.index,
            ).to_numpy()

    X = pd.DataFrame({"feature": [10.0, 20.0]})

    scores = _get_malignant_scores(ProbabilityEstimator(), X)

    assert list(scores) == [0.20, 0.75]
