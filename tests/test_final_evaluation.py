import inspect

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

import femhealth.final_evaluation as final_evaluation_module
import femhealth.final_selection as final_selection_module
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.final_evaluation import (
    DEVELOPMENT_CLASS_DISTRIBUTION,
    DEVELOPMENT_SAMPLE_COUNT,
    FINAL_TEST_CLASS_DISTRIBUTION,
    FINAL_TEST_SAMPLE_COUNT,
    evaluate_final_holdout,
)
from femhealth.final_selection import SELECTED_THRESHOLD, SELECTED_VARIANT
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL

SUMMARY_COLUMNS = [
    "selected_variant",
    "threshold",
    "test_sample_count",
    "accuracy",
    "balanced_accuracy",
    "precision_malignant",
    "recall_malignant",
    "f1_malignant",
    "specificity_benign",
    "roc_auc_malignant",
    "average_precision_malignant",
    "brier_score",
    "log_loss",
    "true_malignant",
    "false_negative_malignant",
    "false_positive_malignant",
    "true_benign",
]


class RecordingProbabilityEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, malignant_probabilities=None):
        self.malignant_probabilities = malignant_probabilities

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_index_ = X.index.copy()
        self.fit_target_index_ = y.index.copy()
        self.classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]
        return self

    def predict_proba(self, X: pd.DataFrame):
        self.predict_index_ = X.index.copy()
        probabilities = list(self.malignant_probabilities)[: len(X)]
        return pd.DataFrame(
            {
                BENIGN_LABEL: [1 - probability for probability in probabilities],
                MALIGNANT_LABEL: probabilities,
            },
            index=X.index,
        )[[BENIGN_LABEL, MALIGNANT_LABEL]].to_numpy()


@pytest.fixture()
def synthetic_final_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_development = pd.DataFrame(
        0.0,
        index=range(DEVELOPMENT_SAMPLE_COUNT),
        columns=WDBC_FEATURE_NAMES,
    )
    y_development = pd.Series(
        [MALIGNANT_LABEL] * DEVELOPMENT_CLASS_DISTRIBUTION[MALIGNANT_LABEL]
        + [BENIGN_LABEL] * DEVELOPMENT_CLASS_DISTRIBUTION[BENIGN_LABEL],
        index=X_development.index,
        name="diagnosis",
    )
    X_test = pd.DataFrame(
        1.0,
        index=range(1000, 1000 + FINAL_TEST_SAMPLE_COUNT),
        columns=WDBC_FEATURE_NAMES,
    )
    y_test = pd.Series(
        [MALIGNANT_LABEL] * FINAL_TEST_CLASS_DISTRIBUTION[MALIGNANT_LABEL]
        + [BENIGN_LABEL] * FINAL_TEST_CLASS_DISTRIBUTION[BENIGN_LABEL],
        index=X_test.index,
        name="diagnosis",
    )
    return X_development, y_development, X_test, y_test


@pytest.fixture()
def malignant_probabilities() -> tuple[float, ...]:
    return (
        tuple([0.90] * 40)
        + tuple([0.20] * 2)
        + tuple([0.80] * 3)
        + tuple([0.10] * 69)
    )


def test_final_evaluation_validates_frozen_sizes_and_distributions(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    estimator = RecordingProbabilityEstimator(malignant_probabilities)

    summary, _, _ = evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
        estimator=estimator,
    )

    assert summary.loc[0, "test_sample_count"] == FINAL_TEST_SAMPLE_COUNT
    assert y_development.value_counts().sort_index().to_dict() == DEVELOPMENT_CLASS_DISTRIBUTION
    assert y_test.value_counts().sort_index().to_dict() == FINAL_TEST_CLASS_DISTRIBUTION


def test_final_evaluation_rejects_overlapping_indices(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    X_test = X_test.copy()
    y_test = y_test.copy()
    X_test.index = range(FINAL_TEST_SAMPLE_COUNT)
    y_test.index = X_test.index

    with pytest.raises(ValueError, match="indices must be disjoint"):
        evaluate_final_holdout(
            X_development,
            y_development,
            X_test,
            y_test,
            estimator=RecordingProbabilityEstimator(malignant_probabilities),
        )


def test_final_evaluation_rejects_unexpected_distribution(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    y_test = y_test.copy()
    y_test.iloc[0] = BENIGN_LABEL

    with pytest.raises(ValueError, match="Unexpected final test class distribution"):
        evaluate_final_holdout(
            X_development,
            y_development,
            X_test,
            y_test,
            estimator=RecordingProbabilityEstimator(malignant_probabilities),
        )


def test_final_evaluation_rejects_unexpected_final_test_sample_count(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    X_test = X_test.iloc[:-1].copy()
    y_test = y_test.iloc[:-1].copy()

    with pytest.raises(ValueError, match="Unexpected final test sample count"):
        evaluate_final_holdout(
            X_development,
            y_development,
            X_test,
            y_test,
            estimator=RecordingProbabilityEstimator(malignant_probabilities),
        )


def test_final_evaluation_rejects_unexpected_development_distribution(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    y_development = y_development.copy()
    y_development.iloc[0] = BENIGN_LABEL

    with pytest.raises(ValueError, match="Unexpected development class distribution"):
        evaluate_final_holdout(
            X_development,
            y_development,
            X_test,
            y_test,
            estimator=RecordingProbabilityEstimator(malignant_probabilities),
        )


def test_final_evaluation_trains_only_on_development_and_predicts_only_on_test(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data

    _, _, fitted_estimator = evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
        estimator=RecordingProbabilityEstimator(malignant_probabilities),
    )

    assert fitted_estimator.fit_index_.equals(X_development.index)
    assert fitted_estimator.fit_target_index_.equals(y_development.index)
    assert fitted_estimator.predict_index_.equals(X_test.index)


def test_final_evaluation_uses_explicit_malignant_probability_column_and_threshold_rule(
    synthetic_final_data,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    threshold_probabilities = (
        (SELECTED_THRESHOLD,)
        + tuple([0.90] * 39)
        + tuple([0.20] * 2)
        + tuple([0.80] * 3)
        + tuple([0.10] * 69)
    )

    _, predictions, _ = evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
        estimator=RecordingProbabilityEstimator(threshold_probabilities),
    )

    first_test_index = X_test.index[0]
    assert predictions.loc[first_test_index, "probability_malignant"] == SELECTED_THRESHOLD
    assert predictions.loc[first_test_index, "predicted_label"] == MALIGNANT_LABEL


def test_final_evaluation_summary_predictions_and_counts_are_correct(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data

    summary, predictions, _ = evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
        estimator=RecordingProbabilityEstimator(malignant_probabilities),
    )

    assert list(summary.columns) == SUMMARY_COLUMNS
    assert summary.loc[0, "selected_variant"] == SELECTED_VARIANT
    assert summary.loc[0, "threshold"] == SELECTED_THRESHOLD
    assert summary.loc[0, "true_malignant"] == 40
    assert summary.loc[0, "false_negative_malignant"] == 2
    assert summary.loc[0, "false_positive_malignant"] == 3
    assert summary.loc[0, "true_benign"] == 69
    assert summary.loc[0, "accuracy"] == pytest.approx(109 / 114)
    assert summary.loc[0, "balanced_accuracy"] == pytest.approx(((40 / 42) + (69 / 72)) / 2)
    assert summary.loc[0, "precision_malignant"] == pytest.approx(40 / 43)
    assert summary.loc[0, "recall_malignant"] == pytest.approx(40 / 42)
    assert summary.loc[0, "f1_malignant"] == pytest.approx(80 / 85)
    assert summary.loc[0, "specificity_benign"] == pytest.approx(69 / 72)
    assert (
        summary[
            [
                "true_malignant",
                "false_negative_malignant",
                "false_positive_malignant",
                "true_benign",
            ]
        ].sum(axis=1).iloc[0]
        == FINAL_TEST_SAMPLE_COUNT
    )
    assert summary.loc[0, "true_malignant"] + summary.loc[0, "false_negative_malignant"] == 42
    assert summary.loc[0, "false_positive_malignant"] + summary.loc[0, "true_benign"] == 72
    assert predictions.index.equals(X_test.index)
    assert list(predictions.columns) == [
        "true_label",
        "probability_malignant",
        "predicted_label",
        "correct",
        "error_type",
    ]


def test_final_evaluation_error_type_is_calculated_correctly(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data

    _, predictions, _ = evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
        estimator=RecordingProbabilityEstimator(malignant_probabilities),
    )

    assert set(predictions["error_type"]) == {
        "correct",
        "false_negative_malignant",
        "false_positive_malignant",
    }
    assert (predictions["error_type"] == "false_negative_malignant").sum() == 2
    assert (predictions["error_type"] == "false_positive_malignant").sum() == 3


def test_final_evaluation_does_not_modify_inputs(
    synthetic_final_data,
    malignant_probabilities,
) -> None:
    X_development, y_development, X_test, y_test = synthetic_final_data
    original_X_development = X_development.copy(deep=True)
    original_y_development = y_development.copy(deep=True)
    original_X_test = X_test.copy(deep=True)
    original_y_test = y_test.copy(deep=True)

    evaluate_final_holdout(
        X_development,
        y_development,
        X_test,
        y_test,
        estimator=RecordingProbabilityEstimator(malignant_probabilities),
    )

    assert X_development.equals(original_X_development)
    assert y_development.equals(original_y_development)
    assert X_test.equals(original_X_test)
    assert y_test.equals(original_y_test)


def test_final_modules_do_not_load_dataset_split_search_threshold_or_persist() -> None:
    module_source = inspect.getsource(final_evaluation_module) + inspect.getsource(
        final_selection_module
    )

    assert "load_wdbc_data" not in module_source
    assert "split_development_test" not in module_source
    assert "GridSearchCV" not in module_source
    assert "joblib" not in module_source
    assert ".dump" not in module_source
