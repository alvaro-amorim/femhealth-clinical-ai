import inspect

import pandas as pd
import pytest

import femhealth.inference as inference
from femhealth.data import WDBC_FEATURE_NAMES
from femhealth.final_selection import SELECTED_THRESHOLD
from femhealth.model_evaluation import BENIGN_LABEL, MALIGNANT_LABEL


class OrderedProbabilityEstimator:
    classes_ = [BENIGN_LABEL, MALIGNANT_LABEL]

    def __init__(self, malignant_probabilities):
        self.malignant_probabilities = malignant_probabilities

    def fit(self, X, y):
        raise AssertionError("Inference must not fit the estimator")

    def predict_proba(self, X):
        probabilities = self.malignant_probabilities[: len(X)]
        return pd.DataFrame(
            {
                BENIGN_LABEL: [1 - probability for probability in probabilities],
                MALIGNANT_LABEL: probabilities,
            },
            index=X.index,
        )[[BENIGN_LABEL, MALIGNANT_LABEL]].to_numpy()


@pytest.fixture()
def inference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        1.0,
        index=[101, 202, 303],
        columns=WDBC_FEATURE_NAMES,
    )


@pytest.fixture()
def metadata() -> dict[str, object]:
    return {
        "threshold": SELECTED_THRESHOLD,
        "feature_names": WDBC_FEATURE_NAMES,
    }


def test_predict_with_artifact_preserves_index_and_returns_expected_columns(
    inference_frame,
    metadata,
) -> None:
    estimator = OrderedProbabilityEstimator([0.20, SELECTED_THRESHOLD, 0.90])

    predictions = inference.predict_with_artifact(inference_frame, estimator, metadata)

    assert predictions.index.equals(inference_frame.index)
    assert list(predictions.columns) == [
        "probability_malignant",
        "probability_benign",
        "predicted_label",
        "predicted_class",
    ]


def test_predict_with_artifact_uses_explicit_malignant_column_and_threshold(
    inference_frame,
    metadata,
) -> None:
    estimator = OrderedProbabilityEstimator([0.20, SELECTED_THRESHOLD, 0.90])

    predictions = inference.predict_with_artifact(inference_frame, estimator, metadata)

    assert predictions.loc[101, "probability_malignant"] == 0.20
    assert predictions.loc[202, "probability_malignant"] == SELECTED_THRESHOLD
    assert predictions.loc[202, "predicted_label"] == MALIGNANT_LABEL
    assert predictions.loc[303, "predicted_label"] == MALIGNANT_LABEL
    assert predictions.loc[101, "predicted_label"] == BENIGN_LABEL


def test_predict_with_artifact_calculates_benign_probability_and_class_names(
    inference_frame,
    metadata,
) -> None:
    estimator = OrderedProbabilityEstimator([0.20, SELECTED_THRESHOLD, 0.90])

    predictions = inference.predict_with_artifact(inference_frame, estimator, metadata)

    assert predictions.loc[101, "probability_benign"] == pytest.approx(0.80)
    assert predictions.loc[202, "probability_benign"] == pytest.approx(1 - SELECTED_THRESHOLD)
    assert predictions.loc[101, "predicted_class"] == "benign"
    assert predictions.loc[202, "predicted_class"] == "malignant"


def test_predict_with_artifact_rejects_missing_column(inference_frame, metadata) -> None:
    with pytest.raises(ValueError, match="Unexpected feature count"):
        inference.predict_with_artifact(
            inference_frame.drop(columns=[WDBC_FEATURE_NAMES[0]]),
            OrderedProbabilityEstimator([0.20, 0.30, 0.40]),
            metadata,
        )


def test_predict_with_artifact_rejects_extra_column(inference_frame, metadata) -> None:
    X = inference_frame.copy()
    X["extra"] = 1.0

    with pytest.raises(ValueError, match="Unexpected feature count"):
        inference.predict_with_artifact(
            X,
            OrderedProbabilityEstimator([0.20, 0.30, 0.40]),
            metadata,
        )


def test_predict_with_artifact_rejects_incorrect_order(inference_frame, metadata) -> None:
    X = inference_frame[list(reversed(WDBC_FEATURE_NAMES))]

    with pytest.raises(ValueError, match="Unexpected feature columns"):
        inference.predict_with_artifact(
            X,
            OrderedProbabilityEstimator([0.20, 0.30, 0.40]),
            metadata,
        )


def test_predict_with_artifact_rejects_nulls(inference_frame, metadata) -> None:
    X = inference_frame.copy()
    X.iloc[0, 0] = None

    with pytest.raises(ValueError, match="Null values found"):
        inference.predict_with_artifact(
            X,
            OrderedProbabilityEstimator([0.20, 0.30, 0.40]),
            metadata,
        )


def test_predict_with_artifact_rejects_non_numeric_values(inference_frame, metadata) -> None:
    X = inference_frame.copy()
    X[WDBC_FEATURE_NAMES[0]] = "invalid"

    with pytest.raises(ValueError, match="Features must be numeric"):
        inference.predict_with_artifact(
            X,
            OrderedProbabilityEstimator([0.20, 0.30, 0.40]),
            metadata,
        )


def test_predict_with_artifact_does_not_modify_input(inference_frame, metadata) -> None:
    original = inference_frame.copy(deep=True)

    inference.predict_with_artifact(
        inference_frame,
        OrderedProbabilityEstimator([0.20, 0.30, 0.40]),
        metadata,
    )

    assert inference_frame.equals(original)


def test_inference_module_has_no_training_holdout_evaluation_or_gridsearch() -> None:
    module_source = inspect.getsource(inference)

    assert ".fit(" not in module_source
    assert "evaluate_final_holdout" not in module_source
    assert "GridSearchCV" not in module_source
